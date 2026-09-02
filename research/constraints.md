1|# Constraints Discovered
2|
3|## Hard Constraints
4|1. Snapdragon 888 (SM8350) HTP only supports quantized 8-bit and 16-bit integer operations; floating-point operations are not natively supported (though float16 math can be used for float32 networks via conversion).
5|2. The HTP backend supports a limited set of operations: Conv2d, DepthConv2d, TransposeConv2D, FullyConnected, Matmul, Batchnorm, LayerNorm (and variations). Operations like LSTM, GRU, or custom recurrence may not be directly supported.
6|3. Genuinely integer training requires that all tensors (weights, activations, gradients, optimizer state) remain in integer format throughout training; any use of floating-point for master weights or accumulator invalidates the claim.
7|4. Dynamic range for training is limited by integer bitwidth; techniques like block scaling and wider accumulators are needed to prevent overflow/underflow.
8|5. Stochastic rounding or pseudo-stochastic rounding is essential to prevent bias in gradient updates when using low-bitwidth integers.
8. 6. Integer-only training frameworks like NITI require per-layer block scaling exponentiation to maintain dynamic range, which adds computational overhead and complexity.
9. 7. PRIOT's pruning-based approach requires additional memory for edge scores, increasing memory footprint compared to standard training.
10. 8. NITRO-D's local loss blocks and NITRO-Scaling layer introduce architectural changes that may not be compatible with existing STT model architectures.
11. 9. Qualcomm QNN HTP backend does not support dynamic shapes, requiring fixed input sizes which complicates variable-length speech sequences.
12. 10. QNN HTP backend lacks support for operators like LSTM, GRU, or custom recurrence, which are common in STT architectures.
9|
10|## Soft Constraints / Challenges
11|1. Non-linear activations (e.g., softmax, layer norm) must be approximated with integer-friendly functions; this may affect accuracy.
12|2. Training deep networks with integer-only arithmetic may require more careful tuning of learning rates and scaling factors.
13|3. The accumulation of gradients in integer-only training may require wider bitwidths (e.g., 32-bit accumulation for 8-bit weights) to avoid precision loss.
14|4. For STT, the sequence nature and variable-length inputs may complicate the use of static scaling factors.
15|5. Deployment proof requires demonstrating actual HTP/NPU execution, not just conversion; silent CPU fallback must be ruled out.
6. 6. The pseudo-stochastic rounding in NITI requires careful implementation to avoid biasing gradient updates.
7. 7. Static scale factors in PRIOT may limit adaptability to changing activation distributions during training.
8. 8. NITRO-D's IntegerSGD optimizer and NITRO Amplification Factor add complexity to the training process.
9. 9. Integer-only training may require wider accumulators (e.g., 32-bit for 8-bit multiplication) to prevent precision loss during gradient accumulation.
10. 10. For STT, the variable sequence lengths and potential need for attention mechanisms may conflict with HTP's operator support and static shape requirements.
11. 11. Feature extraction operations like mel filterbank computation may not be efficiently mapped to HTP-supported operators, likely requiring CPU execution.
12. 12. The energy efficiency gains of integer-only training on HTP must be weighed against potential accuracy degradation and increased algorithmic complexity.