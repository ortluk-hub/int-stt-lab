# NITI Architecture Research

*Last updated: 2026-09-02 12:34:56*

## Overview

NITI (Neural Network Training using Integer-only arithmetic) is a framework for training deep neural networks using exclusively 8-bit integer arithmetic for parameters, activations, and intermediate values. To maintain sufficient dynamic range during training, NITI employs a **block exponentiation scaling scheme**. 

## Key Architectural Components

### 1. Integer-Only Arithmetic
- All parameters, activations, and gradients are stored and updated as 8-bit integers (int8).
- Matrix multiplications use int8 inputs with int32 accumulation to prevent overflow.
- This follows the common quantized operation pattern: `int8 × int8 → int32` accumulation.

### 2. Block Exponentiation Scaling Scheme
From the research snippets:
- **Per-layer block scaling**: Each layer has its own scaling factor that is dynamically adjusted during training.
- **Block exponentiation**: The scaling scheme uses powers of two (exponentiation) for efficient hardware implementation.
- **Dynamic range adjustment**: The scale is updated per layer to prevent overflow and underflow.
- **Integration with rounding**: The scaling works in tandem with stochastic rounding procedures.
- **Minimal overhead**: The scaling scheme incurs minimal storage and computational overhead.

### 3. Matrix Multiplication Implementation
- Uses NVIDIA CUTLASS framework for both INT4 and INT8 matrix multiplication.
- The operation: `C = A * B` where A and B are int8, accumulated in int32, then rescaled.
- Pseudocode from snippets:
  ```
  a(l)_32, s_a(l) ← INT8MATRIXMULTIPLY(a(l-1), w), s_a(l-1) + s_w
  b ← EFFECTIVEBITWIDTH(a(l)_32)
  a(l), s_a(l) ← SHIFTANDROUND(a(l)_32, s_a(l), max(0, b - 7))
  ```
- This shows: int8 multiplication → int32 accumulation → compute effective bitwidth → shift and round to maintain int8 range.

### 4. Loss Functions and Entropy Calculation
The research mentions "integer entropy loss calculation" as part of the scaling scheme integration.
This suggests that:
- Entropy-based losses (like cross-entropy, which involves log-sum-exp) are computed in the integer domain.
- The scaling scheme is designed to work with these entropy calculations.
- Therefore, log-sum-exp operations (core to softmax and cross-entropy) are present and need to be integer-friendly.

## Integration Points for Piecewise Linear Log-Sum-Exp Approximation

Based on Ada's advice: "integrate the approximation into their int8 matrix operations, leveraging their existing block scaling scheme"

### Where Log-Sum-Exp Occurs in NITI:
1. **Softmax layers**: In attention mechanisms or classification layers.
2. **Loss functions**: Cross-entropy loss involves log-sum-exp.
3. **Entropy regularization**: Used in the scaling scheme adaptation.
4. **Any probability normalization** requiring log-sum-exp.

### How to Leverage Block Scaling Scheme:
1. **Input representation**: Log probabilities in NITI are likely stored as scaled integers.
2. **Dynamic range**: The block scaling ensures values stay in a representable range (e.g., [-10, 0] for log probabilities).
3. **Approximation compatibility**: Our piecewise linear approximation already clamps to [-10, 0], aligning with NITI's range.
4. **Operation mapping**: The approximation uses only multiplication, addition, and comparison - all native to NITI's int8 operations.
5. **Scaling integration**: The approximation can work directly on the scaled integer values, with adjustments for the block scale.

### Proposed Integration Approach:
1. **Replace existing log-sum-exp** in loss/softmax computations with the piecewise linear approximation.
2. **Work with scaled integers**: Accept int8 inputs representing log values (scaled by layer-specific factor).
3. **Internal computation**: Convert to temporary higher precision (e.g., int16 or int32) for the approximation, then convert back.
4. **Preserve scaling**: Ensure the output scaling matches the expected scale for subsequent operations.
5. **Leverage block updates**: The approximation's segment lookup can be designed to work with the block scaling exponents.

## Research Questions for Further Investigation

1. What is the exact bitwidth and scaling format used for log probabilities in NITI?
2. Where specifically are log-sum-exp operations implemented in the NITI codebase?
3. How does the block scaling scheme affect the dynamic range of logits/log-probabilities?
4. What precision is retained after the int8×int8→int32 accumulation in the context of log-sum-exp?
5. Are there existing approximations for log/exp in NITI that we can replace or augment?

## Conclusion

NITI's architecture provides a suitable foundation for integrating the piecewise linear log-sum-exp approximation. The block scaling scheme naturally handles the dynamic range required for log probabilities, and the approximation matches NITI's constraint of using only multiplication, addition, and comparison operations. The next steps are to locate the specific log-sum-exp implementations in NITI and design a drop-in replacement that respects the integer-only and block-scaled nature of the framework.
