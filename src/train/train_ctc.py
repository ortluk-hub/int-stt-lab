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
from src.train import instrumentation as inst

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
                     blank_id: int, blank_suppress: int = 0) -> tuple[float, torch.Tensor, int]:
    """CTC loss value + grad w.r.t. logits, quantized to int8 for the integer chain.

    The raw int8 head outputs are used directly as logits (the exponent is
    intentionally dropped): softmax temperature is absorbed by the integer
    update rule, since UpdateWeight.weight_update renormalizes gradients to
    BITWIDTH magnitude via grad_calc (NITI idiom).

    blank_suppress subtracts a fixed code offset from the blank column before
    the loss (training-time objective shaping against the all-blank shortcut;
    D-006 option A). The shift is constant, so the returned gradient is
    unchanged by the chain rule; eval decodes unbiased logits.
    """
    logits_f = logits_int8.float()
    if blank_suppress:
        logits_f[..., blank_id] -= blank_suppress
    logits_f.requires_grad_(True)  # (B, T', V)
    log_probs = F.log_softmax(logits_f, dim=-1).transpose(0, 1)  # (T', B, V)
    per_sample = F.ctc_loss(log_probs, targets, input_lens, target_lens,
                            blank=blank_id, zero_infinity=True, reduction="none")
    per_token = per_sample / target_lens
    loss = per_token.mean()  # identical to reduction='mean'
    grad, = torch.autograd.grad(loss, logits_f)
    nonfinite = int(torch.isnan(grad).sum()) + int(torch.isinf(grad).sum())
    err_int8, err_exp = float_to_int8(grad)
    components = {
        "mean": round(float(loss.detach()), 4),
        "min": round(float(per_token.min()), 4),
        "max": round(float(per_token.max()), 4),
        "sample_sum_mean": round(float(per_sample.mean()), 4),
    }
    return float(loss.detach()), components, nonfinite, err_int8, err_exp


def blank_suppress_offset(step: int, peak: int, full_until: int, zero_after: int) -> int:
    """Training-step blank-suppression schedule: peak, then half, then off."""
    if peak <= 0:
        return 0
    if step < full_until:
        return peak
    if step < zero_after:
        return max(peak // 2, 1)
    return 0


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
def evaluate(model: IntCTCEncoder, dataset: ASRDataset, n_samples: int, blank_id: int,
             n_pairs: int = 8) -> dict:
    """Fixed-slice eval: WER/CER + raw materials for token/pair diagnostics."""
    idxs = list(range(min(n_samples, len(dataset))))
    hyps, refs, frame_argmax, decoded_ids = [], [], [], []
    gap_sum, gap_tied, gap_n = 0.0, 0, 0
    for i in idxs:
        item = dataset[i]
        feats = torch.from_numpy(item["features"]).unsqueeze(0)
        logits, exp = model((feats, int(item["feature_exp"])))
        argmax = logits[0].argmax(dim=-1).tolist()
        top2 = logits[0].topk(2, dim=-1).values
        gaps = top2[:, 0] - top2[:, 1]
        gap_sum += float(gaps.float().sum()); gap_tied += int((gaps == 0).sum()); gap_n += gaps.numel()
        ids = [t for t in greedy_decode_ids(logits.float(), torch.tensor([logits.size(1)]),
                                            blank_id)[0] if t != blank_id]
        frame_argmax.append(argmax)
        decoded_ids.append(ids)
        hyps.append(dataset.tokenizer.decode(ids))
        refs.append(item["transcript"])
    m = batch_metrics(hyps, refs)
    m["slice_sha"] = inst.sha256_obj([dataset.entries[i]["audio_path"] for i in idxs])
    m["decode"] = "greedy-argmax-collapse-repeat-drop-blank"
    pairs = [{"ref": refs[i], "hyp": hyps[i]}
             for i in range(min(n_pairs, len(idxs)))]
    head_sep = {"mean_gap_codes": round(gap_sum / max(gap_n, 1), 4),
                "tied_frac": round(gap_tied / max(gap_n, 1), 6)}
    return {"metrics": m, "pairs": pairs, "frame_argmax": frame_argmax,
            "decoded_ids": decoded_ids, "head_sep": head_sep}


def derive_rescale_targets(metrics_jsonl: str, depth: int) -> list[int]:
    """Per-block exponent targets derived from a healthy reference run's record.

    Uses the settled post-ReLU activation exponent of each reference block
    (blocks.N.relu), extending with the last observed value for deeper nets.
    """
    import json as _json
    lines = open(metrics_jsonl).readlines()
    act = _json.loads(lines[-1])["integer_health"]["activation_stats"]
    ref = [v["exp"] for k, v in act.items() if k.endswith(".relu")]
    assert ref, f"no blocks.N.relu entries in {metrics_jsonl}"
    return [ref[i] if i < len(ref) else ref[-1] for i in range(depth)]


def run_training(cfg: argparse.Namespace) -> dict:
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(cfg.threads)

    import src.model.int_layers as _il
    _il.GRAD_DOWNSHIFT = cfg.grad_shift
    _il.RAIL_RESCALE_THRESHOLD = cfg.rail_rescale_threshold

    rescale_targets = None
    if cfg.rescale_from:
        rescale_targets = derive_rescale_targets(cfg.rescale_from, cfg.depth)
        print(f"rescale targets (derived from {cfg.rescale_from}): {rescale_targets}", flush=True)

    train_ds = ASRDataset(cfg.train_manifest, cfg.feature_dir_train, cfg.tokenizer, cfg.stats,
                          clips_dir=cfg.clips_dir, max_duration=cfg.max_duration)
    if cfg.max_train_samples:
        train_ds.entries = train_ds.entries[: cfg.max_train_samples]
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch, shuffle=True, num_workers=cfg.workers,
        pin_memory=False, collate_fn=train_ds.collate, drop_last=True)

    dev_ds = ASRDataset(cfg.dev_manifest, cfg.feature_dir_dev, cfg.tokenizer, cfg.stats,
                        max_duration=cfg.max_duration)

    model = IntCTCEncoder(dim=cfg.dim, depth=cfg.depth, blank_id=cfg.blank_id,
                          rescale_targets=rescale_targets, branch_shift=cfg.branch_shift)
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

    meta = {
        "manifest_sha256": inst.sha256_file(cfg.train_manifest),
        "dev_manifest_sha256": inst.sha256_file(cfg.dev_manifest),
        "tokenizer_sha256": inst.sha256_file(cfg.tokenizer),
        "stats_sha256": inst.sha256_file(cfg.stats),
        "config_sha256": inst.sha256_obj(vars(cfg)),
        "seed": cfg.seed,
        "quant": inst.QUANT_FORMAT,
        "ckpt_latest": str(latest.resolve()),
        "ckpt_best": str((ckpt_dir / "best.pt").resolve()),
        "torch": torch.__version__,
    }

    history = {"losses": [], "evals": []}
    best_wer = float("inf")
    prev_weights: dict | None = None
    window_start = time.time()
    window_samples = 0
    probe = inst.ActivationProbe(model)
    metrics_path = ckpt_dir / "metrics.jsonl"
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
        loss, loss_components, nonfinite, err, err_exp = ctc_error_signal(
            logits, logits_exp, targets, in_lens, target_lens, cfg.blank_id,
            blank_suppress=blank_suppress_offset(
                step, cfg.blank_suppress, cfg.blank_suppress_steps, cfg.blank_suppress_end))
        assert err.dtype == torch.int8

        model.backward((err, err_exp))
        window_samples += cfg.batch

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
            inst.reset_clamp()
            inst.enable_clamp_tracking(True)
            ev = evaluate(model, dev_ds, cfg.eval_samples, cfg.blank_id)
            inst.enable_clamp_tracking(False)
            head_collapsed = (cfg.stop_on_head_collapse
                              and ev["head_sep"]["tied_frac"] >= 0.99)
            m = ev["metrics"]
            tok = inst.token_diag(ev["frame_argmax"], ev["decoded_ids"], cfg.blank_id)
            record = inst.build_record(
                step=step, elapsed_s=time.time() - t_start,
                window_s=time.time() - window_start, samples=window_samples,
                optimizer_updates=step, loss_components=loss_components,
                nonfinite_grad_count=nonfinite, eval_metrics=m, pairs=ev["pairs"],
                tok=tok, model=model, prev_weights=prev_weights,
                probe_stats=probe.stats, meta=meta)
            record["integer_health"]["head_separation"] = ev["head_sep"]
            record["blank_suppress"] = {
                "offset": blank_suppress_offset(
                    step, cfg.blank_suppress, cfg.blank_suppress_steps, cfg.blank_suppress_end),
                "peak": cfg.blank_suppress,
                "full_until": cfg.blank_suppress_steps,
                "zero_after": cfg.blank_suppress_end,
            }
            inst.append_jsonl(metrics_path, record)
            window_start = time.time()
            window_samples = 0
            prev_weights = inst.snapshot_weights(model)

            history["evals"].append({"step": step, **m, **tok})
            print(f"  eval step {step}: WER {m['wer']:.3f} CER {m['cer']:.3f} "
                  f"blank_rate {tok['blank_rate']:.3f} "
                  f"(n={m['n']}) -> {metrics_path}", flush=True)
            torch.save({"model": model.state_dict(), "step": step, "rng": torch.random.get_rng_state(),
                        "config": vars(cfg), "metrics": m}, latest)
            if m["wer"] < best_wer:
                best_wer = m["wer"]
                torch.save({"model": model.state_dict(), "step": step,
                            "config": vars(cfg), "metrics": m}, ckpt_dir / "best.pt")

            if head_collapsed:
                print(f"  EARLY STOP step {step}: head separation collapsed "
                      f"(tied_frac={ev['head_sep']['tied_frac']})", flush=True)
                history["early_stop"] = "head_collapse"
                break

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
    p.add_argument("--grad-shift", type=int, default=0,
                   help="integer LR: extra right-shift on quantized gradients (0 = run-1 behavior)")
    p.add_argument("--rail-rescale-threshold", type=float, default=0.25,
                   help="weight-rail management (D-004): rescale tensor value-preserving "
                        "(codes>>1, weight_exp+1) when at-rail code fraction >= threshold; 0 disables")
    p.add_argument("--blank-suppress", type=int, default=0,
                   help="anti-collapse objective shaping (D-006 option A): code offset subtracted "
                        "from the head's blank-column logits in the training CTC loss; 0 disables")
    p.add_argument("--blank-suppress-steps", type=int, default=1000,
                   help="full blank-suppress offset until this step")
    p.add_argument("--blank-suppress-end", type=int, default=1500,
                   help="half offset until this step, zero after")
    p.add_argument("--rescale-from", default=None,
                   help="healthy run's metrics.jsonl; derive per-block exponent targets from it")
    p.add_argument("--branch-shift", type=int, default=0,
                   help="scale residual branch by 2^-k before the add (0 = off)")
    p.add_argument("--stop-on-head-collapse", action="store_true",
                   help="early stop when head top1-top2 ties on >=99% of frames")
    p.add_argument("--resume", action="store_true")
    run_training(p.parse_args())


if __name__ == "__main__":
    main()
