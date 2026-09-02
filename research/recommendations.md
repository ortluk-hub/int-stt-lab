# Recommendations and Open Questions for Nemo

## Recommendations
1. **Select an integer-only training framework**: Based on the survey, NITI or NITRO-D appear to be the most developed frameworks for general neural network training. NITI has been tested on ImageNet with 8-bit integers and 5-bit gradients, showing no accuracy degradation on MNIST/CIFAR10 and comparable results on ImageNet. NITRO-D focuses on CNNs but includes novel scaling layers and local loss blocks.
2. **Start with a minimal STT architecture**: For Phase 1, consider a simple feed-forward network or a small CNN on spectrogram patches (treating audio as images) to verify integer-only training before moving to recurrent or transformer models.
3. **Verify integer-only training claim**: Implement logging or assertions to ensure that no floating-point tensors are used for parameters, gradients, or optimizer state during training. Use framework-specific mechanisms (e.g., NITI's integer-only operations) to confirm.
4. **Investigate CTC loss in integer arithmetic**: Research how to implement CTC loss function using only integer operations (e.g., via log-domain approximations or fixed-point logarithms). This is a critical gap.
5. **Explore NPU-compatible feature extraction**: Investigate whether mel filterbank can be implemented as a small neural network (e.g., using triangular filters as fixed weights) that runs on NPU, or accept CPU-side feature extraction as a temporary constraint.
6. **Use a small, falsifiable proof-of-concept**: For Phase 0/1, use a subset of the Speech Commands dataset (e.g., "yes", "no", "up", "down") and a tiny model (e.g., 2-layer MLP with 10-20 hidden units) to demonstrate loss reduction and decoding without floating-point master weights.

## Open Questions for Nemo
1. Which integer-only training framework (NITI, NITRO-D, PRIOT, or another) is most suitable for extension to STT architectures, considering ease of implementation and community support?
2. How can we handle variable-length sequences and dynamic batching in integer-only training frameworks that assume fixed tensor shapes?
3. What specific modifications are needed to adapt NITI/NITRO-D to work with STT-specific layers (e.g., LSTM, GRU, self-attention) and loss functions (CTC, transducer)?
4. Given the Snapdragon 888 HTP operator support, which STT architectures can be mapped to the supported ops without unsupported operations? For example, can a Conformer encoder be decomposed into supported operations?
5. Is it feasible to train the feature extraction module (e.g., mel filterbank) using integer-only arithmetic and deploy it on NPU, or should it remain CPU-side?
6. What tools or methods can be used to verify that a model is running on HTP and not falling back to CPU (e.g., using QNN profiling tools)?