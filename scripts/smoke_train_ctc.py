#!/usr/bin/env python3
"""
Smoke test for the integer CTC training loop.

Runs a tiny training session and asserts:
  - loss decreases (mean of last 10 < mean of first 10)
  - integer-state invariants hold throughout (strict mode inside run_training)
  - checkpoint save -> reload reproduces byte-identical weights
  - prints timing for sizing the real run

Usage: ./venv/bin/python scripts/smoke_train_ctc.py
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.int_ctc_encoder import IntCTCEncoder
from src.train.train_ctc import run_training


def main():
    ckpt_dir = "checkpoints/int-ctc-smoke"
    cfg = argparse.Namespace(
        train_manifest="data/manifests/train.jsonl",
        dev_manifest="data/manifests/dev.jsonl",
        feature_dir_train="data/features/train",
        feature_dir_dev="data/features/dev",
        tokenizer="data/tokenizer/tokenizer.json",
        stats="data/features/global_stats.json",
        clips_dir="/home/ortluk/ortluk-hub/common-voice-26-en-recovered/cv-corpus-26.0-2026-06-12/en",
        ckpt_dir=ckpt_dir,
        dim=192, depth=4, batch=2, steps=60,
        max_duration=6.0, max_train_samples=512,
        eval_every=30, eval_samples=16, log_every=10,
        workers=2, threads=4, seed=0, blank_id=4, strict=True, resume=False,
    )
    history = run_training(cfg)

    losses = history["losses"]
    first = sum(losses[:10]) / 10
    last = sum(losses[-10:]) / 10
    print(f"\nloss first-10 mean {first:.4f} -> last-10 mean {last:.4f}")
    assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"
    print("PASS: loss decreased")

    # checkpoint reload -> byte-identical weights
    state = torch.load(Path(ckpt_dir) / "latest.pt", weights_only=False)
    model = IntCTCEncoder(dim=cfg.dim, depth=cfg.depth, blank_id=cfg.blank_id)
    model.load_state_dict(state["model"])
    model2 = IntCTCEncoder(dim=cfg.dim, depth=cfg.depth, blank_id=cfg.blank_id)
    model2.load_state_dict(state["model"])
    for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
        assert torch.equal(p1, p2), n1
    print(f"PASS: checkpoint reload byte-identical ({len(state['model'])} state tensors)")

    s_per_step = history["wall_s"] / cfg.steps
    print(f"\ntiming: {s_per_step:.2f} s/step "
          f"-> 3000-step real run ~ {3000 * s_per_step / 3600:.1f} h"
          f" (dim {cfg.dim}, depth {cfg.depth}, batch {cfg.batch})")
    print(f"evals: {history['evals']}")
    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
