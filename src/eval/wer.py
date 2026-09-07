#!/usr/bin/env python3
"""
WER/CER evaluation helpers: greedy CTC decode + editdistance metrics.

Self-test: python -m src.eval.wer
"""

import editdistance
import torch


def greedy_decode_ids(logits: torch.Tensor, lengths: torch.Tensor, blank_id: int) -> list[list[int]]:
    """Greedy CTC decode: argmax -> collapse repeats -> drop blanks.

    logits: (B, T, V) float or int tensor; lengths: (B,) valid frame counts.
    """
    pred = logits.argmax(dim=-1)  # (B, T)
    out = []
    for b in range(pred.size(0)):
        ids = pred[b, : int(lengths[b])].tolist()
        collapsed = []
        for t in ids:
            if t != blank_id and (not collapsed or t != collapsed[-1]):
                collapsed.append(t)
        out.append(collapsed)
    return out


def decode_texts(id_seqs: list[list[int]], tokenizer) -> list[str]:
    return [tokenizer.decode(ids) for ids in id_seqs]


def compute_wer(hyps: list[str], refs: list[str]) -> tuple[float, int, int]:
    """(word errors, total words) aggregated."""
    err = tot = 0
    for h, r in zip(hyps, refs):
        hw, rw = h.split(), r.split()
        err += editdistance.eval(hw, rw)
        tot += len(rw)
    return err, tot


def compute_cer(hyps: list[str], refs: list[str]) -> tuple[int, int]:
    err = tot = 0
    for h, r in zip(hyps, refs):
        err += editdistance.eval(h, r)
        tot += max(len(r), 1)
    return err, tot


def batch_metrics(hyps: list[str], refs: list[str]) -> dict:
    w_err, w_tot = compute_wer(hyps, refs)
    c_err, c_tot = compute_cer(hyps, refs)
    return {
        "wer": w_err / max(w_tot, 1),
        "cer": c_err / max(c_tot, 1),
        "n": len(refs),
    }


if __name__ == "__main__":
    # Decode logic sanity check without tokenizer dependency
    logits = torch.zeros(1, 8, 10)  # (B, T, V)
    seq = [4, 4, 3, 3, 5, 4, 5, 5]  # blank=4
    for t, c in enumerate(seq):
        logits[0, t, c] = 10.0
    ids = greedy_decode_ids(logits, torch.tensor([8]), blank_id=4)
    assert ids == [[3, 5]], ids
    metrics = batch_metrics(["hello world"], ["hello world"])
    assert metrics["wer"] == 0.0 and metrics["cer"] == 0.0
    metrics = batch_metrics(["hello there"], ["hello world"])
    assert abs(metrics["wer"] - 0.5) < 1e-9
    print("All tests passed!")
