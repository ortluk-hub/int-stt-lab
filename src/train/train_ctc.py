#!/usr/bin/env python3
"""
Integer-only CTC training loop for sentence-level ASR (CPU).

Model state is integer-only throughout training: int8 weights + int64 exponents,
int8 errors propagated by manual backward, integer SGD weight updates in place.

Documented compromise (consistent with TiConv1dFloatBackward precedent):
the CTC *error signal* w.r.t. logits is computed via torch's float CTCLoss
internals, then immediately quantized to int8 via float_to_int8 before entering
the integer backward chain. No float training state is retained across steps.

Usage:
    python -m src.train.train_ctc --steps 3000 --batch 4 --dim 256 --depth 6 \
        --max-train-samples 50000 --max-duration 8

Smoke: scripts/smoke_train_ctc.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.asr_dataset import ASRDataset
from src.eval.wer import greedy_decode_ids, batch_metrics
from src.model.int_ctc_encoder import IntCTCEncoder
from src.model.int_layers import float_to_int8, int8_to_float

DEFAULT_CLIPS = "/home/ortluk/ortluk-hub/common-voice-26-en-recovered/cv-corpus-26.0-2026-06-12/en"


def align_feature_exps(batch: dict) -> tuple[torch.Tensor, int]:
    """Align per-sample feature exponents to the batch minimum (common exponent)."""
    feats = batch["features"]  # (B, T, n_mels) int8
    exps = batch["feature_exps"]  # (B,) int64
    base = int(exps.min())
    if int(exps.max()) > base:
        feats = feats.clone()
        for i in range(feats.size(0)):
            shift = int(exps[i]) - base
            if shift > 0:
                feats[i] = (feats[i].to(torch.int16) >> shift).to(torch.int8)
    return feats, base


def ctc_error_signal(logits_int8: torch.Tensor, logits_exp: int, targets: torch.Tensor,
                     input_lens: torch.Tensor, target_lens: torch.Tensor,
                     blank_id: int) -> tuple[float, torch.Tensor, int]:
    """CTC loss value + grad w.r.t. logits, quantized to int8 for the integer chain.

    The raw int8 head outputs are used directly as logits (the exponent is
    intentionally dropped): softmax temperature is absorbed by the integer
    update rule, since UpdateWeight.weight_update renormalizes gradients to
    BITWIDTH magnitude via grad_calc (NITI idiom).
    """
    logits_f = logits_int8.float().requires_grad_(True)  # (B, T', V)
    log_probs = F.log_softmax(logits_f, dim=-1).transpose(0, 1)  # (T', B, V)
    loss = F.ctc_loss(log_probs, targets, input_lens, target_lens,
                      blank=blank_id, zero_infinity=True)
    grad, = torch.autograd.grad(loss, logits_f)
    err_int8, err_exp = float_to_int8(grad)
    return float(loss.detach()), err_int8, err_exp


def ctc_targets(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Strip BOS/EOS from tokenizer output; padded (B, S) targets + lengths."""
    toks = batch["tokens"]
    lens = batch["token_lens"]
    stripped = [toks[i, 1:int(lens[i]) - 1] for i in range(toks.size(0))]
    max_s = max(s.numel() for s in stripped)
    out = torch.zeros(toks.size(0), max_s, dtype=torch.long)
    for i, s in enumerate(stripped):
        out[i, : s.numel()] = s
    return out, torch.tensor([s.numel() for s in stripped], dtype=torch.long)


@torch.no_grad()
def evaluate(model: IntCTCEncoder, dataset: ASRDataset, n_samples: int, blank_id: int) -> dict:
    idxs = list(range(min(n_samples, len(dataset))))
    hyps, refs = [], []
    for i in idxs:
        item = dataset[i]
        feats = torch.from_numpy(item["features"]).unsqueeze(0)
        logits, exp = model((feats, int(item["feature_exp"])))
        ids = greedy_decode_ids(logits.float(), torch.tensor([logits.size(1)]), blank_id)[0]
        ids = [t for t in ids if t != blank_id]
        hyps.append(dataset.tokenizer.decode(ids))
        refs.append(item["transcript"])
    return batch_metrics(hyps, refs)


def run_training(cfg: argparse.Namespace) -> dict:
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)
    rng = torch.random.get_rng_state()

    train_ds = ASRDataset(cfg.train_manifest, cfg.feature_dir_train, cfg.tokenizer, cfg.stats,
                          clips_dir=cfg.clips_dir, max_duration=cfg.max_duration)
    if cfg.max_train_samples:
        train_ds.entries = train_ds.entries[: cfg.max_train_samples]
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch, shuffle=True, num_workers=cfg.workers,
        pin_memory=False, collate_fn=train_ds.collate, drop_last=True)

    dev_ds = ASRDataset(cfg.dev_manifest, cfg.feature_dir_dev, cfg.tokenizer, cfg.stats,
                        max_duration=cfg.max_duration)

    model = IntCTCEncoder(dim=cfg.dim, depth=cfg.depth, blank_id=cfg.blank_id)
    report = model.integer_state_report()
    assert report["ok"], report

    start_step = 0
    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest = ckpt_dir / "latest.pt"
    if cfg.resume and latest.exists():
        state = torch.load(latest, weights_only=False)
        model.load_state_dict(state["model"])
        torch.random.set_rng_state(state["rng"])
        start_step = state["step"]
        print(f"resumed from {latest} at step {start_step}", flush=True)

    history = {"losses": [], "evals": []}
    best_wer = float("inf")
    data_iter = iter(loader)
    t_start = time.time()

    for step in range(start_step + 1, cfg.steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        feats, fexp = align_feature_exps(batch)
        logits, logits_exp = model((feats, fexp))
        assert logits.dtype == torch.int8

        in_lens = model.out_lengths(batch["feature_lens"])
        targets, target_lens = ctc_targets(batch)
        loss, err, err_exp = ctc_error_signal(logits, logits_exp, targets, in_lens,
                                              target_lens, cfg.blank_id)
        assert err.dtype == torch.int8

        model.backward((err, err_exp))

        if cfg.strict:
            r = model.integer_state_report()
            assert r["ok"], r
            assert all(not p.requires_grad for p in model.parameters())

        history["losses"].append(loss)
        if step % cfg.log_every == 0 or step == 1:
            el = time.time() - t_start
            print(f"step {step}/{cfg.steps} loss {loss:.4f} "
                  f"({el / (step - start_step):.2f} s/step)", flush=True)

        if step % cfg.eval_every == 0 or step == cfg.steps:
            m = evaluate(model, dev_ds, cfg.eval_samples, cfg.blank_id)
            history["evals"].append({"step": step, **m})
            print(f"  eval step {step}: WER {m['wer']:.3f} CER {m['cer']:.3f} "
                  f"(n={m['n']})", flush=True)
            torch.save({"model": model.state_dict(), "step": step, "rng": torch.random.get_rng_state(),
                        "config": vars(cfg), "metrics": m}, latest)
            if m["wer"] < best_wer:
                best_wer = m["wer"]
                torch.save({"model": model.state_dict(), "step": step,
                            "config": vars(cfg), "metrics": m}, ckpt_dir / "best.pt")

    history["wall_s"] = time.time() - t_start
    history["best_wer"] = best_wer
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    return history


def main():
    p = argparse.ArgumentParser(description="Integer-only CTC ASR training (CPU)")
    p.add_argument("--train-manifest", default="data/manifests/train.jsonl")
    p.add_argument("--dev-manifest", default="data/manifests/dev.jsonl")
    p.add_argument("--feature-dir-train", default="data/features/train")
    p.add_argument("--feature-dir-dev", default="data/features/dev")
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--stats", default="data/features/global_stats.json")
    p.add_argument("--clips-dir", default=DEFAULT_CLIPS)
    p.add_argument("--ckpt-dir", default="checkpoints/int-ctc")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--max-duration", type=float, default=8.0)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-samples", type=int, default=64)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blank-id", type=int, default=4)
    p.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action="store_true")
    run_training(p.parse_args())


if __name__ == "__main__":
    main()
