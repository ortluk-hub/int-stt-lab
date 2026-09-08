#!/usr/bin/env python3
"""
Integer-only layers adapted from NITI framework for CPU simulation.
Uses straight-through estimator (STE) for quantization gradients.
All operations in int8 with int32 accumulation, exponent tracking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# Constants
BITWIDTH = 7  # 7 bits for magnitude, 1 sign bit = int8 range [-127, 127]
QUANT_MAX = 127
QUANT_MIN = -128


# Clamp accounting for instrumentation (enabled during eval/record windows only)
CLAMP_STATS = {"enabled": False, "clamped": 0, "total": 0}

# Extra right-shift applied to quantized gradients in grad_calc.
# Acts as an integer learning rate: effective update scale = 2^-GRAD_DOWNSHIFT.
GRAD_DOWNSHIFT = 0


def int8_clip(input: torch.Tensor, clip_val: int = QUANT_MAX) -> torch.Tensor:
    """Clamp to int8 range."""
    clamped = torch.clamp(input, -clip_val, clip_val)
    if CLAMP_STATS["enabled"]:
        CLAMP_STATS["total"] += input.numel()
        CLAMP_STATS["clamped"] += int((clamped != input).sum())
    return clamped.to(torch.int8)


def round_shift(input: torch.Tensor, shift: int) -> torch.Tensor:
    """Deterministic rounding shift right by `shift` bits."""
    if shift <= 0:
        return input
    round_temp = input // (2 ** shift)
    prob = input - round_temp * (2 ** shift)
    round_decision = prob // (2 ** (shift - 1))
    return int8_clip(round_temp + round_decision)


def sto_shift(input: torch.Tensor, shift: int) -> torch.Tensor:
    """Stochastic rounding shift right by `shift` bits."""
    if shift <= 0:
        return input
    tensor_type = input.dtype
    round_temp = input // (2 ** shift)
    prob = torch.abs(input - round_temp * (2 ** shift))
    rand_num = torch.randint(
        low=0, high=2 ** shift, size=prob.size(), 
        dtype=tensor_type, device=input.device
    )
    round_decision = torch.where(
        prob <= rand_num,
        torch.tensor(0, dtype=tensor_type, device=input.device),
        torch.tensor(1, dtype=tensor_type, device=input.device)
    )
    round_decision = round_decision * torch.sign(input)
    return int8_clip(round_temp + round_decision)


def psto_shift(input: torch.Tensor, shift: int) -> torch.Tensor:
    """Pseudo-stochastic rounding using LSBs as randomness."""
    if shift <= 0:
        return input
    round_temp = input // (2 ** shift)
    prob = torch.abs(input - round_temp * (2 ** shift))
    quantized_prob = prob // (2 ** (shift // 2))
    pseudo_rand_num = prob - quantized_prob * (2 ** (shift // 2))
    
    if shift % 2 == 1:
        pseudo_rand_num = pseudo_rand_num * 2
    
    round_decision = torch.where(
        quantized_prob <= pseudo_rand_num,
        torch.tensor(0, dtype=torch.int32, device=input.device),
        torch.tensor(1, dtype=torch.int32, device=input.device)
    )
    round_decision = round_decision * torch.sign(input)
    return int8_clip(round_temp + round_decision)


def tensor_bitwidth(val: torch.Tensor | int) -> int:
    """Compute bitwidth needed to represent max absolute value."""
    if isinstance(val, int):
        if val == 0:
            return 0
        return int(torch.ceil(torch.log2(torch.tensor(float(val)).clamp_min(1))))
    return int(torch.ceil(torch.log2(val.float().clamp_min(1))))


def range_estimate(input: torch.Tensor) -> int:
    """Estimate bitwidth of tensor range."""
    rng = torch.max(torch.abs(input.float()))
    if rng == 0:
        return 0
    return tensor_bitwidth(rng)


def weight_quant(input: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Quantize float weight to int8 with exponent."""
    input_range = torch.max(torch.abs(input))
    input_bitwidth = torch.ceil(torch.log2(input_range.clamp_min(1)))
    act_exp = int(input_bitwidth.item()) - BITWIDTH
    round_val = torch.round(input / input_range * (2 ** BITWIDTH - 1)).to(torch.int8)
    return round_val, act_exp


def float_to_int8(input: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Convert float tensor to int8 with exponent."""
    input_range = torch.max(torch.abs(input))
    input_bitwidth = torch.ceil(torch.log2(input_range.clamp_min(1)))
    act_exp = int(input_bitwidth.item()) - BITWIDTH
    norm_val = input * (2 ** (BITWIDTH - int(input_bitwidth.item())))
    round_val = torch.round(norm_val)
    clamp_val = torch.clamp(round_val, QUANT_MIN, QUANT_MAX).to(torch.int8)
    return clamp_val, act_exp


def int8_to_float(input_int8: torch.Tensor, exp: int) -> torch.Tensor:
    """Convert int8 tensor back to float."""
    return input_int8.float() * (2 ** exp)


# Rounding methods (configurable)
ACT_ROUND_METHOD = round_shift
ERROR_ROUND_METHOD = round_shift
GRAD_ROUND_METHOD = psto_shift
WEIGHT_INIT_METHOD = weight_quant


def act_calc(int32_acc: torch.Tensor, exp_in: int) -> Tuple[torch.Tensor, int]:
    """Calculate output exponent and quantize int32 accumulation to int8."""
    int32_bitwidth = range_estimate(int32_acc)
    shift = int32_bitwidth - BITWIDTH
    if shift > 0:
        exp_out = exp_in + shift
        temp = ACT_ROUND_METHOD(int32_acc, shift)
    else:
        exp_out = exp_in
        temp = int32_acc.to(torch.int8)
    return temp, exp_out


def err_calc(int32_acc: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Calculate error exponent for backward pass."""
    int32_bitwidth = range_estimate(int32_acc)
    shift = int32_bitwidth - BITWIDTH
    if shift > 0:
        temp = ERROR_ROUND_METHOD(int32_acc, shift)
        exp_out = shift
    else:
        temp = int32_acc.to(torch.int8)
        exp_out = 0
    return temp, exp_out


def grad_calc(int32_acc: torch.Tensor, mu: int) -> Tuple[torch.Tensor, int]:
    """Calculate gradient exponent for weight update."""
    int32_bitwidth = range_estimate(int32_acc)
    shift = int32_bitwidth - mu
    if int32_bitwidth == 0:
        return torch.zeros_like(int32_acc, dtype=torch.int8), 0
    elif shift < 1:
        grad = int32_acc.to(torch.int8)
        shift = 0
    else:
        grad = GRAD_ROUND_METHOD(int32_acc, shift)
    if GRAD_DOWNSHIFT > 0:
        grad = round_shift(grad.to(torch.int32), GRAD_DOWNSHIFT).to(torch.int8)
    return grad, shift


def roundoff4(size: int) -> int:
    """Round up to multiple of 4 for int8mm compatibility."""
    return (size + 3) // 4 * 4


def int8mm(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """
    Integer matrix multiplication: int8 @ int8 -> int32.
    lhs: (M, K), rhs: (K, N) -> output: (M, N) int32
    """
    K = roundoff4(lhs.size(1))
    N = roundoff4(rhs.size(1))  # rhs is (K, N), so N = rhs.size(1)
    
    K_diff = K - lhs.size(1)
    N_diff = N - rhs.size(1)
    
    if K_diff > 0:
        A = F.pad(lhs, (0, K_diff, 0, 0), "constant", 0)
    else:
        A = lhs
    
    if K_diff > 0 or N_diff > 0:
        B = F.pad(rhs, (0, N_diff, 0, K_diff), "constant", 0)
    else:
        B = rhs
    
    # int8 @ int8 -> int32
    # A: (M, K), B: (K, N) -> A @ B = (M, N)
    temp = torch.matmul(A.to(torch.int32), B.to(torch.int32))
    
    if N_diff > 0:
        temp = temp[:, :rhs.size(1)]
    
    return temp.contiguous()


class UpdateWeight(nn.Module):
    """Base class for layers with weight update logic."""
    
    def __init__(self):
        super().__init__()
        self.weight: torch.Tensor
        self.weight_exp: int
        self.grad_int32acc: torch.Tensor
        self.act_in_exp: int
        self.err_exp: int
    
    def weight_update(self):
        """Vanilla SGD weight update in integer domain."""
        p = self.weight
        self.grad, grad_shift = grad_calc(self.grad_int32acc, BITWIDTH)
        self.grad_exp = self.err_exp + grad_shift + self.act_in_exp
        p.data = int8_clip(p.to(torch.int16) - self.grad.to(torch.int16))


class TiLinear(UpdateWeight):
    """Integer-only Linear layer with exponent tracking."""
    
    def __init__(self, in_features: int, out_features: int, last_layer: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.last_layer = last_layer
        
        # Initialize weight
        weight_init = torch.empty(out_features, in_features)
        nn.init.xavier_uniform_(weight_init)
        self.weight, self.weight_exp = WEIGHT_INIT_METHOD(weight_init)
        self.weight = nn.Parameter(self.weight, requires_grad=False)
        self.weight_exp = nn.Parameter(torch.tensor(self.weight_exp), requires_grad=False)
    
    def forward(self, input: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        """
        input: (act_in: (B, *, in_features) int8, exp_in: int)
        returns: (act_out: (B, *, out_features) int8, exp_out: int)
        """
        act_in, exp_in = input
        self.act_in = input  # Save for backward
        self.act_in_exp = exp_in
        
        # int8mm: act_in (B, K) @ weight.T (K, out_features) -> (B, out_features) int32
        # Need to handle batch dimensions
        orig_shape = act_in.shape
        act_in_flat = act_in.view(-1, self.in_features)
        
        # Weight is (out_features, in_features), need (in_features, out_features)
        weight_t = self.weight.t().contiguous()
        temp = int8mm(act_in_flat, weight_t)  # (B*, out_features) int32
        
        # Reshape back
        temp = temp.view(*orig_shape[:-1], self.out_features)
        
        if self.last_layer:
            # No activation scaling for last layer
            act_out = temp.to(torch.int8)
            exp_out = exp_in + self.weight_exp
        else:
            act_out, exp_out = act_calc(temp, exp_in + self.weight_exp)
        
        return act_out, exp_out
    
    def backward(self, input: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        """
        input: (err_in: (B, *, out_features) int8, err_exp: int)
        returns: (err_out: (B, *, in_features) int8, err_out_exp: int)
        """
        err_in, self.err_exp = input
        act, _ = self.act_in
        
        # Flatten
        err_in_flat = err_in.view(-1, self.out_features)
        act_flat = act.view(-1, self.in_features)
        
        # Error backward: err_in @ weight -> (B, in_features) int32
        # weight is (out_features, in_features); err_in_flat is (M, out_features)
        err_out_int32 = int8mm(err_in_flat, self.weight)
        err_out, shift_bits = err_calc(err_out_int32)
        self.err_exp += (shift_bits + int(self.weight_exp))
        
        # Weight gradient: err_in.T @ act -> (out_features, in_features) int32
        self.grad_int32acc = int8mm(err_in_flat.t(), act_flat)
        self.weight_update()
        
        # Reshape error output
        err_out = err_out.view(*err_in.shape[:-1], self.in_features)
        
        return err_out, self.err_exp


class TiReLU(nn.Module):
    """Integer ReLU: max(0, x) in int8 domain."""
    
    def __init__(self):
        super().__init__()
        self.zero_point = torch.tensor(0, dtype=torch.int8)
    
    def forward(self, input: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        act_in, exp_in = input
        self.act_in = input
        act_out = torch.maximum(act_in, self.zero_point.to(act_in.device))
        return act_out, exp_in
    
    def backward(self, input: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        err_in, err_exp = input
        act, _ = self.act_in
        err_out = torch.where(act > self.zero_point.to(act.device), err_in, self.zero_point.to(err_in.device))
        return err_out, err_exp


class TiConv1d(UpdateWeight):
    """Integer-only 1D Convolution (adapted from TiConv2d)."""
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3, 
        stride: int = 1, 
        padding: int = 1,
        groups: int = 1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        
        # Weight: (out_channels, in_channels/groups, kernel_size)
        weight_init = torch.empty(out_channels, in_channels // groups, kernel_size)
        nn.init.xavier_uniform_(weight_init)
        self.weight, self.weight_exp = WEIGHT_INIT_METHOD(weight_init)
        self.weight = nn.Parameter(self.weight, requires_grad=False)
        self.weight_exp = nn.Parameter(torch.tensor(self.weight_exp), requires_grad=False)
    
    def forward(self, input: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        """
        input: (act_in: (B, T, C) int8, exp_in: int)
        returns: (act_out: (B, T_out, out_channels) int8, exp_out: int)
        """
        act_in, exp_in = input
        self.act_in = input
        self.act_in_exp = exp_in
        
        B, T, C = act_in.shape
        # Convert to (B, C, T) for conv1d
        act_in_chw = act_in.permute(0, 2, 1).contiguous()  # (B, C, T)
        weight = self.weight  # (out_channels, in_channels, kernel_size)
        
        # Pad input
        if self.padding > 0:
            act_in_chw = F.pad(act_in_chw, (self.padding, self.padding), "constant", 0)
        
        # Use unfold + matmul for integer convolution
        T_padded = act_in_chw.size(2)
        T_out = (T_padded - self.kernel_size) // self.stride + 1
        
        # Unfold input: (B, C, T_out, K)
        act_unfold = act_in_chw.unfold(2, self.kernel_size, self.stride)  # (B, C, T_out, K)
        act_unfold = act_unfold.permute(0, 2, 1, 3).contiguous()  # (B, T_out, C, K)
        act_unfold = act_unfold.view(B * T_out, C * self.kernel_size)  # (B*T_out, C*K)
        
        # Reshape weight: (out_channels, C*K)
        weight_flat = weight.view(self.out_channels, -1).t()  # (C*K, out_channels)
        
        # int8mm
        temp = int8mm(act_unfold.to(torch.int8), weight_flat.to(torch.int8))  # (B*T_out, out_channels) int32
        temp = temp.view(B, T_out, self.out_channels)  # (B, T_out, out_channels)
        
        act_out, exp_out = act_calc(temp, exp_in + int(self.weight_exp))
        
        return act_out, exp_out
    
    def backward(self, input: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        """Simplified backward - delegate to float for gradient computation."""
        # For integer-only training, we need full integer backward.
        # This is complex; for now, use float backward with STE.
        # TODO: Implement full integer conv1d backward
        raise NotImplementedError("Integer conv1d backward not yet implemented. Use float reference for now.")


class TiConv1dFloatBackward(TiConv1d):
    """TiConv1d with float backward pass (for development)."""
    
    def backward(self, input: Tuple[torch.Tensor, int]) -> Tuple[torch.Tensor, int]:
        err_in, err_exp = input
        act_in, act_exp = self.act_in
        
        # Convert to float for gradient computation
        act_float = int8_to_float(act_in, act_exp)
        weight_float = int8_to_float(self.weight, int(self.weight_exp))
        err_float = int8_to_float(err_in, err_exp)
        
        # Compute gradients in float
        err_out_float = F.conv1d(
            err_float.permute(0, 2, 1),  # (B, T, C) -> (B, C, T)
            weight_float.flip(-1).permute(1, 0, 2),  # (out, in, k) -> (in, out, k)
            stride=self.stride,
            padding=self.padding,
            groups=self.groups
        ).permute(0, 2, 1)  # Back to (B, T, C)
        
        # Weight gradient
        act_float_chw = act_float.permute(0, 2, 1)  # (B, C, T)
        err_float_chw = err_float.permute(0, 2, 1)  # (B, C_out, T)
        grad_weight_float = torch.conv1d(
            act_float_chw,
            err_float_chw.flip(-1),
            stride=self.stride,
            padding=self.padding,
            groups=self.groups
        )
        
        # Quantize gradients
        err_out_int8, err_out_exp = float_to_int8(err_out_float)
        grad_weight_int8, _ = float_to_int8(grad_weight_float)
        
        # Update weight
        self.grad_int32acc = grad_weight_int8.to(torch.int32)
        self.act_in_exp = act_exp
        self.err_exp = err_exp
        self.weight_update()
        
        return err_out_int8, err_out_exp


# Testing
if __name__ == "__main__":
    # Test TiLinear
    print("Testing TiLinear...")
    linear = TiLinear(64, 128)
    x = torch.randint(-128, 127, (2, 10, 64), dtype=torch.int8)
    out, out_exp = linear((x, 0))
    print(f"  Input: {x.shape}, Output: {out.shape}, exp: {out_exp}")
    
    # Test TiReLU
    print("Testing TiReLU...")
    relu = TiReLU()
    x = torch.randint(-128, 127, (2, 10, 64), dtype=torch.int8)
    out, out_exp = relu((x, 0))
    print(f"  Input: {x.shape}, Output: {out.shape}, exp: {out_exp}")
    assert (out >= 0).all(), "ReLU output should be >= 0"
    
    # Test quantization functions
    print("Testing quantization...")
    f = torch.randn(10, 64)
    q, exp = float_to_int8(f)
    f_recon = int8_to_float(q, exp)
    print(f"  Original range: [{f.min():.4f}, {f.max():.4f}]")
    print(f"  Quantized range: [{q.min()}, {q.max()}]")
    print(f"  Exponent: {exp}")
    print(f"  Recon range: [{f_recon.min():.4f}, {f_recon.max():.4f}]")
    print(f"  Max abs error: {(f - f_recon).abs().max():.4f}")
    
    print("\nAll tests passed!")