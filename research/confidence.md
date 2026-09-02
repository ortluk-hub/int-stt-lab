# Confidence Levels and Uncertainties

## High Confidence
- The existence of integer-only training frameworks (NITI, PRIOT, NITRO-D) that train neural networks using only integer arithmetic for parameters, activations, and gradients.
- Snapdragon 888 QNN HTP supports 8-bit and 16-bit quantized integer operations with specific operator support as listed in the Qualcomm documentation.
- Feature extraction for speech (e.g., mel filterbank) is currently performed on CPU in existing NPU-accelerated ASR systems (Ryzen AI, Intel NPU examples).

## Medium Confidence
- The claim that these frameworks avoid any floating-point master/shadow weights is based on paper descriptions; however, without examining the actual code, there is some uncertainty.
- The suitability of these frameworks for STT architectures (e.g., RNNs, Transformers, Conformers) is not directly demonstrated in the papers, which focus on CNNs and MLPs.
- The Snapdragon 888 HTP's ability to execute STT models without silent CPU fallback has not been verified; this would require actual deployment testing.

## Low Confidence / Uncertainties
- The exact numerical behavior (saturation, rounding) of optimizer updates in integer-only training for STT-specific loss functions (e.g., CTC, transducer) is unknown.
- Whether the per-layer block scaling approach in NITI generalizes to recurrent layers or attention mechanisms is uncertain.
- The impact of integer-only training on convergence speed and final accuracy for STT tasks compared to floating-point baselines is not established from the sources consulted.
- The feasibility of performing feature extraction entirely on NPU for STT (e.g., using NN layers for filterbank) is not explored in the sources.