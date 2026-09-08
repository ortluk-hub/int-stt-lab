#!/usr/bin/env python3
"""Diagnose weight-rail freeze in an integer CTC checkpoint (D-002 evidence).

Loads a trained IntCTCEncoder checkpoint, runs one forward/backward on a real
dev batch, and reports per TiLinear layer:
  - fraction of int8 weight codes at the rails (>=127 or <=-128)
  - fraction of quantized gradient codes that are nonzero, and their max
  - fraction of codes where the grad points outward while the weight sits at
    a rail (update will be clamped back by int8_clip -> permanent no-op)
  - predicted vs actual weight-code changes from the single update

A frozen layer shows nz_grad% >> 0 but act_ch == 0 and outward% ~ at_rail%.

Usage:
  ./venv/bin/python scripts/probe_weight_rail_freeze.py \
      --ckpt checkpoints/cleanbase-d3-128/latest.pt \
      [--dim 128 --depth 3 --batch 4]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch

from src.data.asr_dataset import ASRDataset
from src.model.int_ctc_encoder import IntCTCEncoder
from src.train.train_ctc import align_feature_exps, ctc_error_signal, ctc_targets


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dev-manifest", default="data/manifests/dev.jsonl")
    p.add_argument("--feature-dir-dev", default="data/features/dev")
    p.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    p.add_argument("--stats", default="data/features/global_stats.json")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--blank-id", type=int, default=4)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max-duration", type=float, default=8.0)
    cfg = p.parse_args()
    torch.set_num_threads(4)

    model = IntCTCEncoder(dim=cfg.dim, depth=cfg.depth, blank_id=cfg.blank_id)
    state = torch.load(cfg.ckpt, weights_only=False)
    model.load_state_dict(state["model"])
    print(f"loaded {cfg.ckpt} (step {state.get('step', '?')})")

    ds = ASRDataset(cfg.dev_manifest, cfg.feature_dir_dev, cfg.tokenizer, cfg.stats,
                    max_duration=cfg.max_duration)
    loader = torch.utils.data.DataLoader(ds, batch_size=cfg.batch, shuffle=False,
                                         num_workers=0, collate_fn=ds.collate,
                                         drop_last=True)
    batch = next(iter(loader))
    feats, fexp = align_feature_exps(batch)
    logits, logits_exp = model((feats, fexp))
    in_lens = model.out_lengths(batch["feature_lens"])
    targets, target_lens = ctc_targets(batch)
    loss, _, _, err, err_exp = ctc_error_signal(logits, logits_exp, targets,
                                                in_lens, target_lens, cfg.blank_id)
    print(f"one batch: loss {loss:.1f}, ctc err codes nonzero "
          f"{int((err != 0).sum())}/{err.numel()} max|code| {int(err.abs().max())} "
          f"err_exp {err_exp}")

    snaps = {n: m.weight.detach().clone() for n, m in model.named_modules()
             if type(m).__name__ == "TiLinear"}
    model.backward((err, err_exp))

    hdr = (f"{'layer':18s} {'at_rail%':>9s} {'nz_grad%':>9s} {'outward%':>9s} "
           f"{'max|g|':>7s} {'pred_ch':>8s} {'act_ch':>8s}")
    print(hdr)
    for n, m in model.named_modules():
        if type(m).__name__ != "TiLinear":
            continue
        w0, g = snaps[n], m.grad
        at_rail = (w0 >= 127) | (w0 <= -127)
        outward = ((w0 >= 127) & (g < 0)) | ((w0 <= -127) & (g > 0))
        pred = torch.clamp(w0.to(torch.int16) - g.to(torch.int16), -127, 127)
        pred_ch = int((pred != w0).sum())
        act_ch = int((m.weight.detach() != w0).sum())
        print(f"{n:18s} {100 * float(at_rail.float().mean()):>9.2f} "
              f"{100 * float((g != 0).float().mean()):>9.2f} "
              f"{100 * float(outward.float().mean()):>9.2f} "
              f"{int(g.abs().max()):>7d} {pred_ch:>8d} {act_ch:>8d}"
              f"{'  <-- FROZEN (clamp reverts updates)' if act_ch == 0 and float((g != 0).float().mean()) > 0.1 else ''}")


if __name__ == "__main__":
    main()