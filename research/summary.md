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
- Several integer-only training frameworks exist (NITI, PRIOT, NITRO-D, WAGE, PocketNN) that demonstrate training with integer arithmetic only.
- Genuinely integer training requires no floating-point master/shadow weights; all parameters, activations, gradients, and optimizer state must be represented in integer formats.
- Dynamic range and precision challenges are addressed via per-layer block scaling, pseudo-stochastic rounding, and wider accumulator bits.
- Snapdragon 888 (SM8350) HTP supports 8-bit and 16-bit quantized integer operations, with specific operator support (convolution, depthwise convolution, fully connected, matmul, batch norm, layer norm).
- Feature extraction for speech (e.g., mel filterbank) is often kept on CPU due to difficulty mapping to NPU, while the encoder runs on NPU.
- A minimal proof-of-concept could use a small speech dataset (e.g., subset of Speech Commands) and a tiny model (e.g., small MLP or CNN) to verify integer-only training.

## Files Created
- sources.md: List of consulted sources
- evidence_vs_inference.md: Distinction between evidence and inference
- confidence.md: Confidence levels and uncertainties
- constraints.md: Constraints discovered
- recommendations.md: Recommendations and open questions for Nemo