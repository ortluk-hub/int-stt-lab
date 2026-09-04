## Context
The repository structure and collaboration contract are now in place. `AGENTS.md` defines Nemo as engineering/supervision, Ada as research, and Morgan as independent reviewer, with durable research and review surfaces.

## Objective
Create `docs/project_plan.md` as the authoritative project-planning document for `int-stt-lab` before implementation begins.

## Current Status
- Phase 0: Completed (Ada's research report committed, Nemo integrated findings, Morgan's independent review completed)
- Phase 1: Completed (NITI framework integration and verification done), Morgan reviewed and approved.
- Phase 2: In progress (CPU integer-only training scripts developed and verified, NITI integration training pipeline implemented, verification scripts ready).


The project goal is to investigate and build a small speech-to-text model that is:

- trained natively with integer/fixed-point arithmetic rather than conventionally trained and merely quantized afterward;
- trainable on CPU using the local speech dataset available on the server;
- architected from the outset for eventual Snapdragon 888 / SM8350 NPU execution, with Qualcomm QNN/HTP deployability treated as a hard design constraint;
- evaluated reproducibly with explicit evidence for training behavior, accuracy, resource usage, and deployment viability.

The planning document should avoid prematurely locking the implementation to an architecture before Ada completes the required research.

## Required contents of `docs/project_plan.md`

### 1. Project purpose and hard constraints
Capture the non-negotiable requirements, including:

- genuine integer/fixed-point trainable model state (no hidden floating-point master/shadow weights);
- all tensors (weights, activations, gradients, optimizer state) must remain in integer format throughout training;
- CPU training on the available server hardware/dataset;
- Snapdragon 888 NPU deployment target (SM8350 HTP supports 8-bit and 16-bit quantized integer operations only);
- no silent CPU fallback accepted as proof of NPU deployment (must demonstrate actual HTP/NPU execution);
- bounded model size and resource use appropriate for a small on-device STT model;
- reproducible train/eval/checkpoint/inference loop;
- dynamic range limitations addressed via per-layer block scaling, wider accumulators, and stochastic/pseudo-stochastic rounding;
- HTP-supported operations limited to: Conv2d, DepthConv2d, TransposeConv2D, FullyConnected, Matmul, Batchnorm, LayerNorm (and variations).

### 2. Ada's Phase 0 Research Findings
Ada has investigated the open research questions and delivered her findings in the `research/` directory. Summary:

**Evidence (directly from sources):**
- NITI framework stores all parameters and accumulates intermediate values as 8-bit integers, using no more than 5 bits for gradients, with per-layer block scaling exponentiation and pseudo-stochastic rounding. (Source: NITI paper)
- PRIOT represents all weights, activations, and gradients as 8-bit integers and performs entire training using only integer arithmetic with static scale factors. (Source: PRIOT paper)
- NITRO-D enables training of integer-only CNNs without requiring a separate quantization scheme, using NITRO-Scaling layer and NITRO-ReLU activation function. (Source: NITRO-D paper)
- Snapdragon 888 (SM8350) HTP supports quantized 8-bit and 16-bit networks, with supported operations including Conv2d, DepthConv2d, TransposeConv2D, FullyConnected, Matmul, Batchnorm, LayerNorm. (Source: Qualcomm QNN HTP backend documentation)
- In Ryzen AI NPU ASR demo, mel feature extraction runs on CPU while the Conformer encoder runs on NPU and LSTM decoder on integrated Radeon GPU. (Source: RyzenAI-SW demo)
- Intel NPU is BF16-native; INT8 operations may be slower than FP32 due to conversion overhead. (Source: Intel NPU ASR blog post)

**Inference (logical deductions):**
- Genuinely integer training requires that no floating-point master/shadow weights are used at any stage of training, including optimizer state updates.
- Architectures for STT that minimize numerically awkward operations likely involve avoiding operations that require non-integer scaling or complex nonlinearities.
- CTC loss function may pose challenges for integer-only training due to its reliance on logarithms and exponentials.
- Feature extraction remaining CPU-side is inferred from the observation that mel filterbank operations are not commonly offloaded to NPU in current demonstrations.
- A smallest falsifiable proof-of-concept could be a small MLP trained on a tiny audio dataset (e.g., two words from Speech Commands) using an integer-only training framework like NITI, verified by checking that no floating-point tensors are used in training state.

**Discovered constraints:**
- Integer-only training frameworks exist (NITI, PRIOT, NITRO-D, WAGE, PocketNN) that demonstrate training with integer arithmetic only.
- Genuinely integer training requires no floating-point master/shadow weights; all parameters, activations, gradients, and optimizer state must be represented in integer formats.
- Dynamic range and precision challenges are addressed via per-layer block scaling, pseudo-stochastic rounding, and wider accumulator bits.
- Snapdragon 888 (SM8350) HTP supports 8-bit and 16-bit quantized integer operations, with specific operator support (convolution, depthwise convolution, fully connected, matmul, batch norm, layer norm).
- Feature extraction for speech (e.g., mel filterbank) is often kept on CPU due to difficulty mapping to NPU, while the encoder runs on NPU.
- A minimal proof-of-concept could use a small speech dataset (e.g., subset of Speech Commands) and a tiny model (e.g., small MLP or CNN) to verify integer-only training.

**Recommendations for Nemo:**
- Select one of the integer-only training frameworks (e.g., NITI or PRIOT) for Phase 0 feasibility work.
- Focus on architectures that use only HTP-supported operations (Conv2d, DepthConv2d, FullyConnected, Matmul) for the STT model.
- Consider alternatives to CTC loss if integer-only CTC proves infeasible (e.g., transducer-based losses or frame-wise cross-entropy with integer approximations).
- Plan for feature extraction to remain CPU-side initially, with potential to offload later if NPU-compatible implementations become viable.
- Define verification methods for integer-only training claim: framework-specific checks for absence of floating-point tensors, logging of tensor data types, and assertions in training loop.

### 3. Proposed project phases
Define phases with explicit entry/exit criteria. Based on Ada's evidence, the following structure is proposed:

- **Phase 0 — Research and feasibility** (completed)
  - characterized dataset and target hardware constraints;
  - surveyed integer-training methods and QNN/HTP deployment constraints;
  - selected candidate training arithmetic schemes and model families (NITI/PRIOT with small CNN/MLP);
  - defined verification methods for integer-only training claim (framework-specific checks, logging, assertions).
  - *Exit criteria: Ada's research report committed, Nemo has integrated findings, and Morgan has completed independent review.*

- **Phase 1 — Minimal learning proof**
  - build the smallest end-to-end training loop on a small dataset subset (e.g., Speech Commands yes/no/up/down);
  - use a simple feed-forward network or small CNN on spectrogram patches to verify integer-only training;
  - prove loss reduction and held-out decoding;
  - demonstrate checkpoint save/reload;
  - verify that trainable model state does not rely on floating-point master parameters (use framework-specific mechanisms to confirm integer-only operations);
  - investigate CTC loss implementation using integer operations (e.g., log-domain approximations);
  - note: if purely integer CTC loss proves infeasible, consider alternative integer-friendly loss functions or decoding strategies.
  - *Exit criteria: Working integer-only training loop on small subset, loss decreases, held-out decoding possible, checkpoints save/reload correctly, verification confirms no floating-point master weights.*

- **Phase 2 — Baseline and numerical validation**
  - train a conventional reference implementation of the same/similar architecture where useful;
  - compare convergence, CER/WER, runtime, memory, and stability;
  - document numerical range, overflow/saturation behavior, and update precision.
  - *Exit criteria: Baseline established, numerical behavior documented, integer-only training validation complete.*

- **Phase 3 — Dataset-scale training**
  - expand training to a meaningful portion/all of the local dataset;
  - establish repeatable evaluation splits and metrics;
  - optimize CPU training throughput without changing the integer-training claim.
  - *Exit criteria: Model trained on full dataset, evaluation metrics repeatable, CPU training throughput measured.*

- **Phase 4 — Snapdragon 888 deployment proof**
  - export/convert the trained model through the chosen deployment path;
  - verify graph compatibility with QNN/HTP;
  - demonstrate actual HTP/NPU execution on SM8350 without silent CPU fallback;
  - report latency, memory, model size, and accuracy.
  - *Exit criteria: Model deployed to Snapdragon 888 HTP, execution verified, performance metrics reported.*

- **Phase 5 — Refinement**
  - improve accuracy/efficiency only after the training and deployment proofs are sound;
  - preserve reproducibility and integer-training invariants.
  - *Exit criteria: Accuracy/efficiency improvements made while maintaining integer-training and deployment proofs.*

Nemo may revise this phase structure if Ada's evidence supports a better decomposition, but changes should be justified in the project plan.

### 4. Deliverables per phase
For every phase define concrete artifacts, such as:

- research reports;
- dataset characterization;
- design decision records;
- source code;
- tests;
- training logs/configs;
- checkpoints/artifacts where appropriate;
- CER/WER and resource measurements;
- Snapdragon deployment evidence;
- Morgan review record.

Specific to Phase 0:
- Ada's research handoff (sources, evidence vs inference, confidence, constraints, recommendations) in `research/`;
- Updated project plan (`docs/project_plan.md`);
- Morgan's independent review record (`docs/reviews/monthly-review-001.md`).

### 5. Review gates
Each phase should have a Morgan review gate before the next phase is considered complete. Review should verify claims against evidence rather than treating Nemo's handoff as approval.

Morgan should explicitly examine, where relevant:

- train/test leakage;
- transcript normalization and split methodology;
- CTC/decoder correctness;
- whether any floating-point shadow/master training state invalidates the integer-training claim;
- numerical correctness and saturation/overflow handling;
- whether deployment evidence proves HTP/NPU execution rather than conversion alone;
- reproducibility of claimed results.

### 6. Risks and decision points
Track known risks separately from requirements. At minimum include:

- integer-training convergence/stability;
- insufficient precision for gradient/update paths;
- CPU training cost;
- unsupported QNN/HTP operators;
- model architecture becoming deployment-incompatible;
- dataset quality/split leakage;
- accuracy tradeoffs from strict arithmetic constraints;
- feature extraction remaining CPU-side (if NPU-compatible implementation proves infeasible).

### 7. Definition of project success
Define minimum success independently from stretch goals. The minimum success criterion should require both:

1. a credible native integer/fixed-point training proof with reproducible held-out STT learning; and
2. a Snapdragon 888 HTP/NPU deployment proof for the resulting model or a directly equivalent trained graph, with no silent CPU fallback.

Do not claim production-quality STT as a requirement for the first research success.