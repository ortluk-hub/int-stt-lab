# Integrated Research Summary for int‑stt‑lab

## Scope
This summary consolidates the research artifacts produced in the `research/` folder for Phase 0 of the **int‑stt‑lab** project. It captures:
* Sources consulted
* Evidence vs. inference distinctions
* Confidence levels and uncertainties
* Hard and soft constraints discovered
* Recommendations and open questions for Nemo
* Suggested next‑step actions

---

## 1. Sources Consulted
(see `sources.md` for full list)

**Papers & Articles**
1. NITI: *Training Integer Neural Networks Using Integer‑Only Arithmetic* – https://arxiv.org/pdf/2009.13108
2. PRIOT: *Pruning‑Based Integer‑Only Transfer Learning for Embedded Systems* – https://arxiv.org/pdf/2503.16860
3. NITRO‑D: *Native Integer‑only Training of Deep Convolutional Neural Networks* – https://arxiv.org/pdf/2407.11698v3
4. Quantization & Training of Neural Networks for Efficient Integer‑Arithmetic‑Only Inference – https://arxiv.org/abs/1802.04680
5. PocketNN … – https://arxiv.org/html/2201.02863v3
6. … (additional 13 entries – see `sources.md`)

**Documentation & Guides**
* Qualcomm QNN HTP Backend – https://docs.qualcomm.com/doc/80-63442-10/topic/htp_backend.html
* ONNX Runtime QNN Execution Provider – https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html
* AMD Ryzen AI NPU ASR demo – https://github.com/amd/RyzenAI‑SW/blob/main/Demos/ASR/Parakeet‑TDT/README.md
* Intel NPU ASR blog – https://dev.to/cibernox/running-asr‑for‑smart‑homes‑in‑the‑npu‑of‑intel‑processors‑2iec

## 2. Evidence vs. Inference (`evidence_vs_inference.md`)

### Evidence (directly from sources)
* **Integer‑only frameworks** (NITI, PRIOT, NITRO‑D) store weights/activations as 8‑bit integers and perform forward/backward passes with integer arithmetic.
* **Snapdragon 888 HTP** supports 8‑bit/16‑bit quantized ops: Conv2d, DepthConv2d, TransposeConv2D, FullyConnected, Matmul, Batchnorm, LayerNorm, pointwise Add/Mul. No native LSTM/GRU.
* **Feature‑extraction placement** – Existing NPU‑accelerated ASR demos (Ryzen AI, Intel NPU) keep mel‑filterbank on CPU.

### Inference (logical deductions)
* Genuine integer training must avoid any floating‑point master/shadow weights; otherwise the claim is false.
* STT models typically require recurrent/attention mechanisms, which conflict with HTP’s lack of LSTM/GRU support – inference that architectures must be adapted or replaced.
* CTC loss uses logarithms/exp; implementing it with integer‑only arithmetic will need fixed‑point approximations.
* A minimal proof‑of‑concept can be a tiny MLP on a subset of Speech‑Commands, verifying no floating‑point tensors appear.

## 3. Confidence Levels (`confidence.md`)
| Level | Statements |
|-------|------------|
| **High** | Existence of NITI/PRIOT/NITRO‑D frameworks; Snapdragon 888 HTP integer‑op support; CPU‑side feature extraction is current practice. |
| **Medium** | No floating‑point master weights – inferred from papers (code not fully audited). Suitability of integer frameworks for STT architectures not demonstrated. HTP fallback to CPU not empirically verified. |
| **Low** | Numerical behaviour of optimizer updates for CTC/Transducer in integer domain; scalability of block‑scaling to recurrent/attention layers; impact on convergence/accuracy for STT. |

## 4. Constraints (`constraints.md`)
### Hard Constraints
1. **HTP operator set** – only the listed 8/16‑bit ops; no LSTM/GRU, no dynamic shapes.
2. **All training tensors must remain integer‑typed** (weights, activations, gradients, optimizer state). Any FP tensor invalidates integer‑only claim.
3. **Dynamic‑range limits** – require per‑layer block scaling or wider accumulators (e.g., 32‑bit) to avoid overflow/underflow.
4. **Stochastic / pseudo‑stochastic rounding** is essential for unbiased gradient updates.
5. **Fixed input sizes** – HTP does not support dynamic shapes, so STT inputs must be padded or batched to a uniform length.

### Soft Constraints / Challenges
* Approximating non‑linear ops (softmax, layer‑norm) with integer‑friendly functions may hurt accuracy.
* Variable‑length speech sequences complicate static scaling factors.
* Feature‑extraction (mel‑filterbank) lacks efficient integer implementations; CPU fallback likely.
* Wider accumulators increase memory pressure.
* Integrating integer‑only frameworks with attention‑based STT models may require substantial architectural changes.

## 5. Recommendations & Open Questions (`recommendations.md`)
### Recommendations
1. **Adopt NITI** for initial experiments – most mature, proven on ImageNet, provides block‑scaling and integer‑only ops.
2. **Prototype with a tiny feed‑forward model** (2‑layer MLP, 10–20 hidden units) on a small speech subset (e.g., four commands) to verify integer‑only training.
3. **Instrument training**: add assertions/logs to confirm every tensor involved in forward/backward passes is integer‑typed.
4. **Research integer‑friendly CTC** – explore fixed‑point log‑domain implementations or alternative loss functions.
5. **Map feature extraction to NPU** only if a neural‑network‑based filterbank can be expressed with supported ops; otherwise accept CPU preprocessing.

### Open Questions for Nemo
1. Which integer‑only framework (NITI, NITRO‑D, PRIOT) best extends to STT‑specific layers (LSTM/GRU/attention)?
2. How to handle variable‑length sequences under HTP’s fixed‑shape requirement?
3. What modifications are needed for NITI/NITRO‑D to support CTC or transducer losses?
4. Which STT architectures can be expressed solely with HTP‑supported ops (e.g., Conv‑based encoders, shallow CNNs)?
5. Feasibility of deploying mel‑filterbank on NPU vs. CPU‑side.
6. Tools to verify runtime execution on HTP (QNN profiling, on‑device logs).

## 6. Next‑Step Action Items
* **Select framework** – Nemo to decide between NITI and NITRO‑D (priority on community support & code availability).
* **Create minimal dataset & model** – Subset of Speech Commands + tiny MLP.
* **Implement integer‑only training** – Using chosen framework, add runtime checks for FP tensors.
* **Prototype integer CTC** – Research fixed‑point log implementation or alternative decoders.
* **Benchmark on Snapdragon 888 HTP** – Verify that the model runs on HTP without CPU fallback (use QNN profiling tools).
* **Document findings** – Update this summary with empirical results for Phase 1.

---

*Generated by Ada (Researcher) – evidence‑backed synthesis for Nemo’s engineering planning.*
