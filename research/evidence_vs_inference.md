# Evidence vs Inference

## Evidence (Directly from sources)
- NITI framework stores all parameters and accumulates intermediate values as 8-bit integers, using no more than 5 bits for gradients, with per-layer block scaling exponentiation and pseudo-stochastic rounding. (Source: NITI paper)
- PRIOT represents all weights, activations, and gradients as 8-bit integers and performs entire training using only integer arithmetic with static scale factors. (Source: PRIOT paper)
- NITRO-D enables training of integer-only CNNs without requiring a separate quantization scheme, using NITRO-Scaling layer and NITRO-ReLU activation function. (Source: NITRO-D paper)
- Snapdragon 888 (SM8350) HTP supports quantized 8-bit and 16-bit networks, with supported operations including Conv2d, DepthConv2d, TransposeConv2D, FullyConnected, Matmul, Batchnorm, LayerNorm. (Source: Qualcomm QNN HTP backend documentation)
- In Ryzen AI NPU ASR demo, mel feature extraction runs on CPU while the Conformer encoder runs on NPU and LSTM decoder on integrated Radeon GPU. (Source: RyzenAI-SW demo)
- Intel NPU is BF16-native; INT8 operations may be slower than FP32 due to conversion overhead. (Source: Intel NPU ASR blog post)

## Inference (Logical deductions, not direct evidence)
- Genuinely integer training requires that no floating-point master/shadow weights are used at any stage of training, including optimizer state updates. This is inferred from the requirement to distinguish true integer training from QAT, low-precision floating point, and integer-forward/floating-update schemes.
- Architectures for STT that minimize numerically awkward operations likely involve avoiding operations that require non-integer scaling or complex nonlinearities; however, no specific STT architectures were found in the literature, so this is an inference based on general integer-only training principles.
- CTC loss function may pose challenges for integer-only training due to its reliance on logarithms and exponentials, but no direct evidence was found; this is an inference based on the nature of CTC.
- Feature extraction remaining CPU-side is inferred from the observation that mel filterbank operations are not commonly offloaded to NPU in current demonstrations, suggesting they are not yet NPU-friendly or that the overhead outweighs benefits.
- A smallest falsifiable proof-of-concept could be a small MLP trained on a tiny audio dataset (e.g., two words from Speech Commands) using an integer-only training framework like NITI, verified by checking that no floating-point tensors are used in training state. This is inferred from the need for a cheap test and the availability of small datasets and models.