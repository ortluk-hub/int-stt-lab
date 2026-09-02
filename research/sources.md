# Sources Consulted

## Papers and Articles
1. NITI: Training Integer Neural Networks Using Integer-Only Arithmetic - https://arxiv.org/pdf/2009.13108
2. PRIOT: Pruning-Based Integer-Only Transfer Learning for Embedded Systems - https://arxiv.org/pdf/2503.16860
3. NITRO-D: Native Integer-only Training of Deep Convolutional Neural Networks - https://arxiv.org/pdf/2407.11698v3
4. Training and Inference with Integers in Deep Neural Networks - https://arxiv.org/abs/1802.04680
5. PocketNN: Integer-only Training and Inference of Neural Networks via Direct Feedback Alignment and Pocket Activations in Pure C++ - https://arxiv.org/html/2201.02863v3
6. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference - https://www.alphaxiv.org/abs/1712.05877
7. What Is int8 Quantization and Why Is It Popular for Deep Neural Networks? - https://de.mathworks.com/company/technical-articles/what-is-int8-quantization-and-why-is-it-popular-for-deep-neural-networks.html
8. An Integer Based Neural Network - SharpNEAT - https://sharpneat.sourceforge.io/research/integer-neuralnet/integer-neuralnet.html
9. Is Integer Arithmetic Enough for Deep Learning Training? - OpenReview - https://openreview.net/forum?id=G7MX_0J6JKX
10. Quantized Neural Networks: The Only Guide You Need - https://mlechner.substack.com/p/quantized-neural-networks-the-only

## Documentation and Guides
11. Qualcomm QNN HTP Backend Documentation - https://docs.qualcomm.com/doc/80-63442-10/topic/htp_backend.html
12. ONNX Runtime QNN Execution Provider - https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
13. Ryzen AI NPU Speech-to-Text Demo - https://github.com/amd/RyzenAI-SW/blob/main/Demos/ASR/Parakeet-TDT/README.md
14. Running ASR for Smart Homes in Intel NPU - https://dev.to/cibernox/running-asr-for-smart-homes-in-the-npu-of-intel-processors-2iec
15. Sherpa-onnx: Speech-to-text, text-to-speech, etc. - https://github.com/k2-fsa/sherpa-onnx

## Miscellaneous
16. Gradient Accumulation resources (various)
17. Hugging Face discussion on integer-only LLM inference - https://discuss.huggingface.co/t/current-state-and-future-of-integer-only-llm-inference-non-floating-point/175216
18. AI Stack Exchange: Why do we need floats for using neural networks? - https://ai.stackexchange.com/questions/7247/why-do-we-need-floats-for-using-neural-networks
19. Matt Log: Quantization in Deep Learning - https://mett29.github.io/posts/quantization
## Concrete Evidence for Integer-Only Training Frameworks

### NITI
- Paper: https://arxiv.org/abs/2009.13108
- GitHub implementation: https://github.com/wangmaolin/niti
- Evidence from code: The repository shows that all weights and activations are stored as int8 tensors, and operations are performed using integer arithmetic via cuBLAS and CUTLASS for matrix multiplications. No floating-point master weights are used in the training loop (see `train.py` and `utils.py`).

### PRIOT
- Paper: https://arxiv.org/pdf/2503.16860
- GitHub implementation: Not publicly available as of the paper date (March 2025). However, the paper describes the algorithm in detail, including that weights, activations, and gradients are quantized to 8-bit integers and training proceeds with integer-only arithmetic using static scale factors.

### NITRO-D
- Paper: https://arxiv.org/pdf/2407.11698v3
- GitHub implementation: Not explicitly linked in the paper, but the authors mention that code is available upon request. The paper details the NITRO-Scaling layer and NITRO-ReLU activation function that enable integer-only training without a separate quantization scheme.

### Qualcomm Snapdragon 888 HTP Operator Set
- Documentation: Qualcomm QNN HTP Backend Documentation - https://docs.qualcomm.com/doc/80-63442-10/topic/htp_backend.html
- Specific operator support (as of the documentation): 
  * Conv2d, DepthConv2d, TransposeConv2D
  * FullyConnected, Matmul
  * Batchnorm, LayerNorm (and variations)
  * Pointwise operations (Add, Mul, etc.)
  * Note: Recurrent operations (LSTM, GRU) are not natively supported; they would need to be implemented via sequences of supported ops or run on CPU.
