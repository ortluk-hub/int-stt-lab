#!/usr/bin/env python3
"""
Diagnose blank collapse in the integer CTC encoder.

Loads a checkpoint, runs train-domain and dev-domain samples, and reports:
  - argmax class distribution over frames (is blank really winning?)
  - top-2 logit gap at frames where blank wins (how close is the escape?)
  - decoded strings (with junk-token filtering variants)
  - per-token CTC loss on the same samples

Usage: ./venv/bin/python scripts/diagnose_blank_collapse.py [--ckpt path] [--n 8]
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.asr_dataset import ASRDataset
from src.model.int_ctc_encoder import IntCTCEncoder
from src.model.int_layers import int8_to_float
from src.train.train_ctc import align_feature_exps, ctc_targets, ctc_error_signal

CLIPS = "/home/ortluk/ortluk-hub/common-voice-26-en-recovered/cv-corpus-26.0-2026-06-12/en"
SPECIAL = {0, 1, 2, 3, 4}  # PAD UNK BOS EOS BLANK


def describe(enc, ds, idxs, label, blank_id=4):
    print(f"\n--- {label} ---")
    for idx in idxs:
        item = ds[idx]
        feats = torch.from_numpy(item["features"]).unsqueeze(0)
        logits, exp = enc((feats, int(item["feature_exp"])))
        argmax = logits[0].argmax(dim=-1)  # (T',)
        counts = torch.bincount(argmax, minlength=1024)
        top = counts.topk(3)
        blank_frac = float(counts[blank_id]) / logits.size(1)
        # blank runner-up gap: at blank frames, margin blank - best other
        blank_mask = argmax == blank_id
        if blank_mask.any():
            lf = int8_to_float(logits[0], exp)
            best_other = lf.clone()
            best_other[:, blank_id] = -1e9
            gap = (lf[:, blank_id] - best_other.max(dim=-1).values)[blank_mask]
            gap_mean = float(gap.mean())
        else:
            gap_mean = float("nan")
        # decode variants
        def dec(keep_special):
            ids = [int(t) for t in argmax]
            collapsed = []
            for t in ids:
                if t != blank_id and (keep_special or t not in SPECIAL):
                    if not collapsed or t != collapsed[-1]:
                        collapsed.append(t)
            return ds.tokenizer.decode(collapsed)

        loss, _, _ = ctc_error_signal(
            logits, exp,
            torch.tensor([item["tokens"][1:int(item["token_len"]) - 1]]),
            torch.tensor([logits.size(1)]),
            torch.tensor([int(item["token_len"]) - 2]), blank_id)
        print(f"[{idx}] T'={logits.size(1)} blank%={blank_frac:.2f} gap={gap_mean:.1f} "
              f"top={[(int(i), int(c)) for i, c in zip(top.indices, top.values)]} "
              f"loss/tok={loss:.1f}")
        print(f"   all-ids : {dec(True)!r}")
        print(f"   filtered: {dec(False)!r}")
        print(f"   ref     : {item['transcript']!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/int-ctc/latest.pt")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()

    state = torch.load(a.ckpt, weights_only=False)
    cfg = state["config"]
    print(f"checkpoint step {state['step']}, dim {cfg['dim']} depth {cfg['depth']}")

    enc = IntCTCEncoder(dim=cfg["dim"], depth=cfg["depth"])
    enc.load_state_dict(state["model"])
    r = enc.integer_state_report()
    print(f"integer state ok: {r['ok']} ({r['num_params']} params)")

    train_ds = ASRDataset("data/manifests/train.jsonl", "data/features/train",
                          "data/tokenizer/tokenizer.json", "data/features/global_stats.json",
                          clips_dir=CLIPS, max_duration=cfg["max_duration"])
    dev_ds = ASRDataset("data/manifests/dev.jsonl", "data/features/dev",
                        "data/tokenizer/tokenizer.json", "data/features/global_stats.json",
                        max_duration=cfg["max_duration"])

    describe(enc, train_ds, list(range(a.n)), "TRAIN domain (in-distribution)")
    describe(enc, dev_ds, list(range(a.n)), "DEV domain")


if __name__ == "__main__":
    main()