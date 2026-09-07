#!/usr/bin/env python3
"""
Integer-only CTC encoder for sentence-level ASR on CPU.

Architecture: log-mel frame stacking -> TiLinear input projection ->
depth x residual MLP blocks (TiLinear + TiReLU) -> TiLinear CTC head.

Every layer used has a working integer manual backward (NITI idiom):
forward propagates (act_int8, exp); backward propagates (err_int8, exp)
and updates int8 weights in place via UpdateWeight.weight_update.
No float training state exists anywhere in this module.

Self-test: python -m src.model.int_ctc_encoder
"""

import torch
import torch.nn as nn
from typing import Tuple

from .int_layers import TiLinear, TiReLU
from .int_conformer import IntResidual


class IntResidualMLPBlock(nn.Module):
    """out = IntResidual.add(x, ReLU(Linear(x))). Backward copies err to both branches."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = TiLinear(dim, dim)
        self.relu = TiReLU()

    def forward(self, input: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        x, exp = input
        self.residual_in = (x, exp)
        y, y_exp = self.relu(self.linear((x, exp)))
        return IntResidual.add(self.residual_in, (y, y_exp))

    def backward(self, input: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        err, err_exp = input
        # Branch 1: through linear+relu (updates weights, returns err at block input)
        err_chain, err_chain_exp = self.linear.backward(self.relu.backward((err, err_exp)))
        # Branch 2: residual passthrough — same err, unchanged exponent
        err_res, err_res_exp = err, err_exp
        # Fork input receives sum of both error branches (exponent-aligned)
        return IntResidual.add((err_res, err_res_exp), (err_chain, err_chain_exp))


class IntCTCEncoder(nn.Module):
    """
    Integer-only CTC encoder.

    forward(feats_int8, feats_exp) -> (logits_int8 (B, T', vocab), logits_exp)
    backward(err_logits_int8, err_exp) -> integer error propagation + in-place int8 weight updates
    """

    def __init__(
        self,
        n_mels: int = 80,
        stack: int = 3,
        stride: int = 2,
        dim: int = 256,
        depth: int = 6,
        vocab: int = 1024,
        blank_id: int = 4,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.stack = stack
        self.stride = stride
        self.dim = dim
        self.vocab = vocab
        self.blank_id = blank_id

        self.proj = TiLinear(n_mels * stack, dim)
        self.blocks = nn.ModuleList([IntResidualMLPBlock(dim) for _ in range(depth)])
        self.head = TiLinear(dim, vocab)  # act_calc scaling keeps int32 safe in int8 + exp

    def out_lengths(self, feature_lens: torch.Tensor) -> torch.Tensor:
        """Per-sample output lengths after frame stacking (must match unfold semantics)."""
        return ((feature_lens - self.stack) // self.stride + 1).clamp(min=1)

    def _frame_stack(self, feats: torch.Tensor) -> torch.Tensor:
        """(B, T, n_mels) int8 -> (B, T', n_mels*stack) int8 via stacked sliding windows."""
        win = feats.unfold(1, self.stack, self.stride)  # (B, T', n_mels, stack)
        win = win.permute(0, 1, 3, 2).contiguous()  # (B, T', stack, n_mels)
        return win.view(win.size(0), win.size(1), -1)  # (B, T', stack*n_mels)

    def forward(self, input: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        feats, exp = input  # (B, T, n_mels) int8
        x = self._frame_stack(feats)
        x, exp = self.proj((x, exp))
        for block in self.blocks:
            x, exp = block((x, exp))
        logits, logits_exp = self.head((x, exp))
        return logits, logits_exp

    def backward(self, input: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        err, err_exp = self.head.backward(input)
        for block in reversed(self.blocks):
            err, err_exp = block.backward((err, err_exp))
        err, err_exp = self.proj.backward((err, err_exp))
        return err, err_exp  # err at stacked-frame input (data boundary; discarded)

    def integer_state_report(self) -> dict:
        """Verify no float training state: every parameter is int8 (weights) or int64 (exponents)."""
        bad = []
        n_params = 0
        for name, p in self.named_parameters():
            n_params += p.numel()
            if p.requires_grad or p.dtype not in (torch.int8, torch.int64):
                bad.append((name, str(p.dtype), p.requires_grad, p.numel()))
        return {"num_params": n_params, "violations": bad, "ok": len(bad) == 0}


if __name__ == "__main__":
    torch.manual_seed(0)
    enc = IntCTCEncoder(dim=192, depth=4)
    report = enc.integer_state_report()
    assert report["ok"], report
    print(f"integer state OK, {report['num_params']} params (all int8/int64)")

    B, T = 2, 120
    feats = torch.randint(-128, 127, (B, T, 80), dtype=torch.int8)
    lens = torch.tensor([120, 100])
    out_lens = enc.out_lengths(lens)

    logits, logits_exp = enc((feats, -5))
    assert logits.dtype == torch.int8, logits.dtype
    assert logits.shape[:2] == (B, int(out_lens[0])), (logits.shape, out_lens)
    print(f"forward OK: {logits.shape}, exp={logits_exp}, out_lengths={out_lens.tolist()}")

    err = torch.randint(-64, 64, logits.shape, dtype=torch.int8)
    err_out, err_out_exp = enc.backward((err, -10))
    assert err_out.dtype == torch.int8 and err_out.shape == (B, logits.shape[1], 240), err_out.shape
    print(f"backward OK: {err_out.shape}, exp={err_out_exp}")

    report = enc.integer_state_report()
    assert report["ok"], report
    print("post-backward integer state OK")
    print("All tests passed!")
