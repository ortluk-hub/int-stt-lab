## Context
The repository structure and collaboration contract are now in place. `AGENTS.md` defines Nemo as engineering/supervision, Ada as research, and Morgan as independent reviewer, with durable research and review surfaces.

## Objective
Create `docs/project_plan.md` as the authoritative project-planning document for `int-stt-lab` before implementation begins.

The project goal is to investigate and build a small speech-to-text model that is:

- trained natively with integer/fixed-point arithmetic rather than conventionally trained and merely quantized afterward;
- trainable on CPU using the local speech dataset available on the server;
- architected from the outset for eventual Snapdragon 888 / SM8350 NPU execution, with Qualcomm QNN/HTP deployability treated as a hard design constraint;
- evaluated reproducibly with explicit evidence for training behavior, accuracy, resource usage, and deployment viability.

The planning document should avoid prematurely locking the implementation to an architecture before Ada completes the required research.

## Required contents of `docs/project_plan.md`

### 1. Project purpose and hard constraints
Capture the non-negotiable requirements, including:
- genuine integer/fixed-point trainable model state;
- no hidden floating-point master/shadow weights used to compute or apply parameter updates;
- distinguish true integer training from QAT, low-precision floating point, and integer-forward/floating-update schemes;
- CPU training on the available server hardware/dataset;
- Snapdragon 888 NPU deployment target;
- no silent CPU fallback accepted as proof of NPU deployment;
- bounded model size and resource use appropriate for a small on-device STT model;
- reproducible train/eval/checkpoint/inference loop.

### 2. Open research questions for Ada
At minimum, Ada should investigate:
- prior work on integer-only / fixed-point neural-network training;
- gradient, accumulator, scaling, saturation, rounding, optimizer-state, and update strategies;
- whether some training state must use wider integer types and what qualifies as genuinely integer training;
- architectures that minimize numerically awkward operations while remaining viable for STT;
- CTC and alternative decoding/training implications for integer arithmetic;
- Snapdragon 888 / QNN / HTP supported operator and quantization constraints relevant to candidate architectures;
- practical feature-extraction choices and whether preprocessing should remain CPU-side or be NPU-compatible;
- smallest falsifiable proof-of-concept that can test the training method cheaply.

Ada's research handoff must follow `AGENTS.md`: sources, evidence vs inference, confidence/uncertainties, discovered constraints, and recommendations/questions for Nemo.

### 3. Proposed project phases
Define phases with explicit entry/exit criteria. A reasonable starting structure is:

- **Phase 0 — Research and feasibility**
  - characterize dataset and target hardware constraints;
  - survey integer-training methods and QNN/HTP deployment constraints;
  - select one or more candidate training arithmetic schemes and model families.

- **Phase 1 — Minimal learning proof**
  - build the smallest end-to-end training loop on a small dataset subset;
  - prove loss reduction and held-out decoding;
  - demonstrate checkpoint save/reload;
  - verify that trainable model state does not rely on floating-point master parameters.

- **Phase 2 — Baseline and numerical validation**
  - train a conventional reference implementation of the same/similar architecture where useful;
  - compare convergence, CER/WER, runtime, memory, and stability;
  - document numerical range, overflow/saturation behavior, and update precision.

- **Phase 3 — Dataset-scale training**
  - expand training to a meaningful portion/all of the local dataset;
  - establish repeatable evaluation splits and metrics;
  - optimize CPU training throughput without changing the integer-training claim.

- **Phase 4 — Snapdragon 888 deployment proof**
  - export/convert the trained model through the chosen deployment path;
  - verify graph compatibility with QNN/HTP;
  - demonstrate actual HTP/NPU execution on SM8350 without silent CPU fallback;
  - report latency, memory, model size, and accuracy.

- **Phase 5 — Refinement**
  - improve accuracy/efficiency only after the training and deployment proofs are sound;
  - preserve reproducibility and integer-training invariants.

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
- accuracy tradeoffs from strict arithmetic constraints.

### 7. Definition of project success
Define minimum success independently from stretch goals. The minimum success criterion should require both:
1. a credible native integer/fixed-point training proof with reproducible held-out STT learning; and
2. a Snapdragon 888 HTP/NPU deployment proof for the resulting model or a directly equivalent trained graph, with no silent CPU fallback.

Do not claim production-quality STT as a requirement for the first research success.