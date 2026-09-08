#!/usr/bin/env python3
"""
Depth/width ablation harness for the integer CTC encoder.

Arms (select with --arms, comma-separated):
  A : depth 6, dim 128, recenter-only (branch_shift=0). Causal test: is the
      blank attractor from recentering itself or from halving the branch?
  B : depth 6, dim 128, recenter + branch 2^-1 (the original treated arm).
      Held per user decision — run only on explicit request.
  C : depth 6, dim 256, no management. Tests the width-coupling hypothesis
      with an early numerical stop on head-separation collapse.

Pass criteria (last metrics.jsonl record):
  head_tied_frac < 0.5, act sat < 5%, blank < 0.95, repeat < 0.9,
  uniq >= 10, head exp <= 20.

Usage: ./venv/bin/python scripts/run_depth_ablation.py --arms A,C [--steps 60]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.train.train_ctc import run_training

HEALTHY = ROOT / "checkpoints/int-ctc-instr-test/metrics.jsonl"
CLIPS = "/home/ortluk/ortluk-hub/common-voice-26-en-recovered/cv-corpus-26.0-2026-06-12/en"


def arm_cfg(arm: str, steps: int, ckpt: str) -> argparse.Namespace:
    common = dict(
        train_manifest="data/manifests/train.jsonl",
        dev_manifest="data/manifests/dev.jsonl",
        feature_dir_train="data/features/train",
        feature_dir_dev="data/features/dev",
        tokenizer="data/tokenizer/tokenizer.json",
        stats="data/features/global_stats.json",
        clips_dir=CLIPS, ckpt_dir=ckpt,
        batch=2, steps=steps, max_duration=6.0, max_train_samples=512,
        eval_every=max(steps // 8, 1), eval_samples=16, log_every=10,
        workers=2, threads=4, seed=0, blank_id=4, strict=True,
        resume=False, grad_shift=0, branch_shift=0, rescale_from=None,
        stop_on_head_collapse=False,
    )
    if arm == "A":   # recenter-only, short smoke
        common.update(dim=128, depth=6, rescale_from=str(HEALTHY), branch_shift=0)
    elif arm == "AX":  # A extended: recenter-only depth 6, trajectory test
        common.update(dim=128, depth=6, rescale_from=str(HEALTHY), branch_shift=0)
    elif arm == "D3R":  # depth 3 + recentering, healthy reference trajectory
        common.update(dim=128, depth=3, rescale_from=str(HEALTHY), branch_shift=0)
    elif arm == "B":  # recenter + branch half (held by default)
        common.update(dim=128, depth=6, rescale_from=str(HEALTHY), branch_shift=1)
    elif arm == "C":  # width probe, no management, early stop
        common.update(dim=256, depth=6, stop_on_head_collapse=True)
    else:
        raise ValueError(arm)
    return argparse.Namespace(**common)


def evaluate_criteria(rec: dict) -> list[tuple[str, bool, str]]:
    sat = max(s["sat_frac"] for s in rec["integer_health"]["activation_stats"].values())
    head_exp = rec["integer_health"]["scale"]["act_exps"]["head"]
    sep = rec["integer_health"]["head_separation"]
    tok = rec["tokens"]
    return [
        ("head_tied_frac < 0.5", sep["tied_frac"] < 0.5, f"tied_frac={sep['tied_frac']}"),
        ("act_sat_frac < 0.05", sat < 0.05, f"max_sat={sat}"),
        ("blank_rate < 0.95", tok["blank_rate"] < 0.95, f"blank={tok['blank_rate']}"),
        ("repeat_rate < 0.9", tok["repeat_rate"] < 0.9, f"repeat={tok['repeat_rate']}"),
        ("unique_tokens >= 10", tok["unique_tokens"] >= 10, f"uniq={tok['unique_tokens']}"),
        ("head_exp <= 20", head_exp <= 20, f"head_exp={head_exp}"),
    ]


def run_arm(arm: str, steps: int):
    ckpt = f"checkpoints/ablation-{arm.lower()}"
    shutil.rmtree(ckpt, ignore_errors=True)
    cfg = arm_cfg(arm, steps, ckpt)
    print(f"\n===== ARM {arm}: dim={cfg.dim} depth={cfg.depth} "
          f"rescale={'ON' if cfg.rescale_from else 'off'} "
          f"branch_shift={cfg.branch_shift} "
          f"early_stop={'ON' if cfg.stop_on_head_collapse else 'off'} =====", flush=True)
    run_training(cfg)
    rec = json.loads(open(Path(ckpt) / "metrics.jsonl").readlines()[-1])
    print(f"\n--- ARM {arm} criteria (step {rec['step']}) ---")
    ok = True
    for label, passed, detail in evaluate_criteria(rec):
        print(f"  {'PASS' if passed else 'FAIL'}  {label:22s} {detail}")
        ok = ok and passed
    print(f"  ARM {arm} RESULT: {'PASS' if ok else 'FAIL'}")
    for p in rec["pairs"][:3]:
        print(f"    ref: {p['ref'][:60]!r}\n    hyp: {p['hyp'][:60]!r}")
    return ok, rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,C")
    ap.add_argument("--steps", type=int, default=60)
    a = ap.parse_args()
    results = {}
    for arm in [x.strip() for x in a.arms.split(",") if x.strip()]:
        results[arm] = run_arm(arm, a.steps)[0]
    print("\n===== ABLATION SUMMARY =====")
    for arm, ok in results.items():
        print(f"arm {arm}: {'PASS' if ok else 'FAIL'}")