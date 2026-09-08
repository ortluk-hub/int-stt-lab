#!/usr/bin/env python3
"""
JSONL instrumentation for the integer CTC training loop.

One compact JSON line per checkpoint (eval), covering:
  progress (step/time/throughput/LR), loss components, fixed-slice WER/CER,
  reference/hypothesis pairs, token diagnostics, integer health (weight churn,
  saturation, clamp counts, scale ranges), gradient health, and provenance
  metadata (manifest/tokenizer/config hashes, seed, quantization format).

Used by src/train/train_ctc.py; record schema documented in docs/assumptions.md
until promoted to its own spec.
"""

import hashlib
import json
import time
from pathlib import Path

import torch

from src.model import int_layers
from src.model.int_layers import TiLinear, TiReLU, UpdateWeight


def sha256_file(path: str | Path, _cache: dict = {}) -> str:
    p = str(path)
    if p not in _cache:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _cache[p] = h.hexdigest()
    return _cache[p]


def sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


QUANT_FORMAT = {
    "weights": "int8 symmetric, per-tensor weight_exp (int64), xavier init quantized",
    "activations": "int8 + tracked exponent; act_calc rescales int32 accum to 7-bit",
    "features": "int8 per-sample max-abs quantization of normalized log-mel",
    "grads": "psto_shift quantization to 7 bits (BITWIDTH), int32 accumulation",
    "updates": "int16 subtract in weight_update (NITI-style integer SGD)",
    "ctc_error": "float CTCLoss grad wrt int8 logits, quantized to int8 (documented compromise)",
}


def snapshot_weights(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if p.dtype == torch.int8}


def weight_change(model: torch.nn.Module, prev: dict[str, torch.Tensor] | None) -> dict:
    """Fraction of int8 weight codes changed since the previous checkpoint."""
    if not prev:
        return {"pct_changed": None, "per_layer": {}}
    per_layer, total_ch, total_n = {}, 0, 0
    for n, p in model.named_parameters():
        if p.dtype != torch.int8 or n not in prev:
            continue
        ch = int((p.detach() != prev[n]).sum())
        per_layer[n] = round(100 * ch / p.numel(), 4)
        total_ch += ch
        total_n += p.numel()
    return {"pct_changed": round(100 * total_ch / max(total_n, 1), 4),
            "per_layer": per_layer}


class ActivationProbe:
    """Forward hooks capturing (act_int8, exp) per TiLinear/TiReLU for saturation stats."""

    def __init__(self, model: torch.nn.Module):
        self.stats: dict[str, dict] = {}
        self._handles = []
        for name, mod in model.named_modules():
            if isinstance(mod, (TiLinear, TiReLU)):
                self._handles.append(mod.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(_mod, _inp, out):
            act, exp = out
            sat = float(((act >= 127) | (act <= -128)).float().mean())
            uniq = len(set(act.flatten().tolist()))
            self.stats[name] = {"exp": int(exp), "sat_frac": round(sat, 6),
                               "min_code": int(act.min()), "max_code": int(act.max()),
                               "unique_codes": uniq}
        return hook

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()


def weight_saturation(model: torch.nn.Module) -> dict[str, float]:
    out = {}
    for n, p in model.named_parameters():
        if p.dtype == torch.int8:
            out[n] = round(float(((p >= 127) | (p <= -128)).float().mean()), 6)
    return out


def scale_ranges(model: torch.nn.Module, probe_stats: dict) -> dict:
    acts = {n: s["exp"] for n, s in probe_stats.items()}
    weights = {n: int(p.item()) for n, p in model.named_parameters()
               if p.dtype == torch.int64}
    return {"act_exps": acts, "weight_exps": weights}


def clamp_window() -> dict:
    return {"clamped": int_layers.CLAMP_STATS["clamped"],
            "total": int_layers.CLAMP_STATS["total"],
            "pct": round(100 * int_layers.CLAMP_STATS["clamped"]
                         / max(int_layers.CLAMP_STATS["total"], 1), 6)}


def reset_clamp():
    int_layers.CLAMP_STATS["clamped"] = 0
    int_layers.CLAMP_STATS["total"] = 0


def enable_clamp_tracking(on: bool):
    int_layers.CLAMP_STATS["enabled"] = on


def grad_health(model: torch.nn.Module) -> dict:
    """Per-layer gradient stats from the most recent backward pass."""
    out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, UpdateWeight) and getattr(mod, "grad_int32acc", None) is not None:
            acc = mod.grad_int32acc
            grad = getattr(mod, "grad", None)
            entry = {
                "acc_max_abs": int(acc.abs().max()),
                "acc_bitwidth": int(acc.abs().max().clamp_min(1).log2().ceil()),
            }
            if grad is not None:
                entry.update({
                    "rails_frac": round(float(((grad >= 127) | (grad <= -128)).float().mean()), 6),
                    "zero_frac": round(float((grad == 0).float().mean()), 6),
                })
            out[name] = entry
    return out


def token_diag(frame_argmax: list[list[int]], decoded_ids: list[list[int]],
               blank_id: int, eos_id: int = 3) -> dict:
    """Token-level diagnostics from eval decodes.

    frame_argmax: per-utterance argmax class per frame (pre-collapse)
    decoded_ids: per-utterance collapsed token ids (post greedy CTC)
    """
    total_frames = sum(len(f) for f in frame_argmax)
    blank_frames = sum(1 for f in frame_argmax for t in f if t == blank_id)
    eos_frames = sum(1 for f in frame_argmax for t in f if t == eos_id)
    all_dec = [t for d in decoded_ids for t in d]
    repeats = sum(1 for d in decoded_ids for a, b in zip(d, d[1:]) if a == b)
    max_run = max((max((sum(1 for _ in g) for k, g in __import__("itertools").groupby(d)), default=0)
                  for d in decoded_ids), default=0)
    return {
        "blank_rate": round(blank_frames / max(total_frames, 1), 6),
        "eos_rate": round(eos_frames / max(total_frames, 1), 6),
        "avg_decoded_len": round(sum(len(d) for d in decoded_ids) / max(len(decoded_ids), 1), 3),
        "unique_tokens": len(set(all_dec)),
        "repeat_rate": round(repeats / max(len(all_dec), 1), 6),
        "max_run": max_run,
    }


def build_record(*, step, elapsed_s, window_s, samples, optimizer_updates,
                 loss_components, nonfinite_grad_count, eval_metrics, pairs,
                 tok, model, prev_weights, probe_stats, meta) -> dict:
    return {
        "step": step,
        "elapsed_s": round(elapsed_s, 2),
        "window_s": round(window_s, 2),
        "samples_per_sec": round(samples / max(window_s, 1e-9), 4),
        "optimizer_updates": optimizer_updates,
        "lr": {"scheme": "int8-grad-downshift",
               "grad_downshift": int_layers.GRAD_DOWNSHIFT,
               "effective_scale": 2.0 ** -int_layers.GRAD_DOWNSHIFT},
        "loss": {**loss_components, "nonfinite_grad_count": nonfinite_grad_count},
        "eval": eval_metrics,
        "pairs": pairs,
        "tokens": tok,
        "integer_health": {
            "weights": weight_change(model, prev_weights),
            "weight_saturation": weight_saturation(model),
            "activation_stats": probe_stats,
            "clamp": clamp_window(),
            "scale": scale_ranges(model, probe_stats),
        },
        "grad_health": grad_health(model),
        "meta": meta,
    }


def append_jsonl(path: str | Path, record: dict):
    with open(path, "a") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")