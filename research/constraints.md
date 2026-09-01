# Constraints Discovered

## Hard Constraints
1. Snapdragon 888 (SM8350) HTP only supports quantized 8-bit and 16-bit integer operations; floating-point operations are not natively supported (though float16 math can be used for float32 networks via conversion).
2. The HTP backend supports a limited set of operations: Conv2d, DepthConv2d, TransposeConv2D, FullyConnected, Matmul, Batchnorm, LayerNorm (and variations). Operations like LSTM, GRU, or custom recurrence may not be directly supported.
3. Genuinely integer training requires that all tensors (weights, activations, gradients, optimizer state) remain in integer format throughout training; any use of floating-point for master weights or accumulator invalidates the claim.
4. Dynamic range for training is limited by integer bitwidth; techniques like block scaling and wider accumulators are needed to prevent overflow/underflow.
5. Stochastic rounding or pseudo-stochastic rounding is essential to prevent bias in gradient updates when using low-bitwidth integers.

## Soft Constraints / Challenges
1. Non-linear activations (e.g., softmax, layer norm) must be approximated with integer-friendly functions; this may affect accuracy.
2. Training deep networks with integer-only arithmetic may require more careful tuning of learning rates and scaling factors.
3. The accumulation of gradients in integer-only training may require wider bitwidths (e.g., 32-bit accumulation for 8-bit weights) to avoid precision loss.
4. For STT, the sequence nature and variable-length inputs may complicate the use of static scaling factors.
5. Deployment proof requires demonstrating actual HTP/NPU execution, not just conversion; silent CPU fallback must be ruled out.