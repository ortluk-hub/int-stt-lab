#!/usr/bin/env python3
"""
Depth-control ablation: can six blocks stay numerically healthy at dim=128?

Isolates depth as the only moved axis relative to the healthy depth-3 run
(dim 128 held constant). Two arms:
  control   : depth 6, no exponent management (reproduces run-1 conditions)
  treated   : depth 6 + per-block residual scaling (2^-1) + BFP exponent
              recentering with targets derived from the healthy depth-3 record

Pass criteria (evaluated on the last metrics.jsonl record):
  head_tied_frac < 0.5   (all-tied head = immediate numerical fail)
  max act sat_frac < 0.05 (no rail growth)
  blank_rate < 0.95       (non-blank text emerging)
  repeat_rate < 0.9 and unique_tokens >= 10 (non-repetitive)
  head exp <= 20          (exponent growth bounded vs healthy ~15)

Usage: ./venv/bin/python scripts/run_depth_ablation.py [--steps 60]
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


def arm_cfg(depth_rescale: bool, steps: int, ckpt: str) -> argparse.Namespace:
    cfg = argparse.Namespace(
        train_manifest="data/manifests/train.jsonl",
        dev_manifest="data/manifests/dev.jsonl",
        feature_dir_train="data/features/train",
        feature_dir_dev="data/features/dev",
        tokenizer="data/tokenizer/tokenizer.json",
        stats="data/features/global_stats.json",
        clips_dir="/home/ortluk/ortluk-hub/common-voice-26-en-recovered/cv-corpus-26.0-2026-06-12/en",
        ckpt_dir=ckpt,
        dim=128, depth=6, batch=2, steps=steps,
        max_duration=6.0, max_train_samples=512,
        eval_every=max(steps // 3, 1), eval_samples=16, log_every=10,
        workers=2, threads=4, seed=0, blank_id=4, strict=True,
        resume=False, grad_shift=0,
        rescale_from=str(HEALTHY) if depth_rescale else None,
        branch_shift=1 if depth_rescale else 0,
    )
    return cfg


def last_record(ckpt: str) -> dict:
    return json.loads(open(Path(ckpt) / "metrics.jsonl").readlines()[-1])


def evaluate_criteria(rec: dict) -> list[tuple[str, bool, str]]:
    sat = max(s["sat_frac"] for s in rec["integer_health"]["activation_stats"].values())
    head_exp = rec["integer_health"]["scale"]["act_exps"]["head"]
    sep = rec["integer_health"]["head_separation"]
    tok = rec["tokens"]
    checks = [
        ("head_tied_frac < 0.5", sep["tied_frac"] < 0.5, f"tied_frac={sep['tied_frac']}"),
        ("act_sat_frac < 0.05", sat < 0.05, f"max_sat={sat}"),
        ("blank_rate < 0.95", tok["blank_rate"] < 0.95, f"blank={tok['blank_rate']}"),
        ("repeat_rate < 0.9", tok["repeat_rate"] < 0.9, f"repeat={tok['repeat_rate']}"),
        ("unique_tokens >= 10", tok["unique_tokens"] >= 10, f"uniq={tok['unique_tokens']}"),
        ("head_exp <= 20", head_exp <= 20, f"head_exp={head_exp}"),
    ]
    return checks


def run_arm(name: str, treated: bool, steps: int):
    ckpt = f"checkpoints/ablation-d6-{name}"
    shutil.rmtree(ckpt, ignore_errors=True)
    print(f"\n===== ARM: {name} (depth 6, dim 128, exponent mgmt={'ON' if treated else 'OFF'}) =====", flush=True)
    run_training(arm_cfg(treated, steps, ckpt))
    rec = last_record(ckpt)
    print(f"\n--- {name} criteria ---")
    ok = True
    for label, passed, detail in evaluate_criteria(rec):
        print(f"  {'PASS' if passed else 'FAIL'}  {label:22s} {detail}")
        ok = ok and passed
    print(f"  ARM RESULT: {'PASS' if ok else 'FAIL'}")
    pairs = rec["pairs"][:3]
    for p in pairs:
        print(f"    ref: {p['ref'][:60]!r}\n    hyp: {p['hyp'][:60]!r}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    a = ap.parse_args()
    control = run_arm("control", treated=False, steps=a.steps)
    treated = run_arm("treated", treated=True, steps=a.steps)
    print("\n===== ABLATION SUMMARY =====")
    print(f"control (no mgmt):  {'PASS' if control else 'FAIL'}")
    print(f"treated (mgmt on):  {'PASS' if treated else 'FAIL'}")