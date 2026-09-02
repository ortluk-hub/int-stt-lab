# Phase 0 Research Summary for int-stt-lab

This document summarizes the research conducted on open questions from GitHub Issue #1 section 2 for the int-stt-lab project.

## Key Areas Investigated
1. Prior work on integer-only/fixed-point neural-network training
2. Gradient/accumulator/scaling/saturation/rounding/optimizer-state/update strategies
3. What qualifies as genuinely integer training
4. Architectures minimizing numerically awkward operations for STT
5. CTC and alternative decoding/training implications for integer arithmetic
6. Snapdragon 888/QNN/HTP supported operator and quantization constraints
7. Practical feature-extraction choices (CPU-side vs NPU-compatible preprocessing)
8. Smallest falsifiable proof-of-concept for testing training method cheaply

## Main Findings
- Several integer-only training frameworks exist (NITI, PRIOT, NITRO-D) that demonstrate training with integer arithmetic only, validated on ImageNet-scale models.
- Genuinely integer training requires no floating-point master/shadow weights; all parameters, activations, gradients, and optimizer state must be represented in integer formats throughout training.
- Dynamic range and precision challenges are addressed via per-layer block scaling (NITI), static scale factors with pruning (PRIOT), and local loss blocks (NITRO-D), along with pseudo-stochastic rounding and wider accumulator bits.
- Snapdragon 888 (SM8350) HTP supports 8-bit and 16-bit quantized integer operations only; floating-point operations are not natively supported.
- The HTP backend supports a specific set of operators: Conv2d, DepthConv2d, TransposeConv2D, FullyConnected, Matmul, Batchnorm, LayerNorm (and variations), but does not support dynamic shapes or operators like LSTM, GRU, or custom recurrence.
- Feature extraction for speech (e.g., mel filterbank) is currently performed on CPU in existing NPU-accelerated ASR systems due to difficulty mapping to NPU-supported operators.
- Integer-only zero-shot quantization for ASR shows promise for post-training quantization with minimal WER degradation (<1%) and speedup benefits.
- A minimal proof-of-concept could use a small speech dataset (e.g., subset of Speech Commands) and a tiny model (e.g., 2-layer MLP with 10-20 hidden units) to verify integer-only training by confirming no floating-point tensors are used.

## Files Created/Updated
- sources.md: List of consulted sources (updated with integer-only zero-shot quantization papers)
- evidence_vs_inference.md: Distinction between evidence and inference (expanded with detailed evidence from papers)
- confidence.md: Confidence levels and uncertainties (updated with specific confidence levels based on experimental results)
- constraints.md: Constraints discovered (expanded with hard and soft constraints from framework analysis)
- recommendations.md: Recommendations and open questions for Nemo (enhanced with specific actionable recommendations)
- summary.md: This document (updated)

## Next Steps
Based on the research, Phase 1 should focus on:
1. Selecting and implementing an integer-only training framework (NITI recommended for initial experiments)
2. Creating a minimal STT proof-of-concept using a small speech dataset
3. Verifying integer-only training claims through instrumentation
4. Investigating STT architectures compatible with HTP-supported operations
