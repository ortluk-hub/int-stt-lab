# int-stt-lab

## Integer-Arithmetic CTC Speech-to-Text for On-Device Deployment

A research project investigating efficient speech-to-text technologies using integer-arithmetic CTC (Connectionist Temporal Classification) models, targeting deployment on Qualcomm Snapdragon 888 via QNN (Qualcomm Neural Network) SDK.

### Project Evolution

This project has evolved through several phases of review and enhancement:

1. **Initial Investigation** - Exploration of integer-arithmetic approaches for CTC-based speech recognition
2. **First Review** - Independent technical review identified evidence gaps requiring empirical validation
3. **Enhanced Documentation** - Evidence anchoring to distinguish verified results from hypotheses/inferences
4. **Second Review** - Two specific gaps remained:
   - Gap 1: Lack of numerical experiments validating CTC integer arithmetic feasibility
   - Gap 2: Missing deployment verification path for Snapdragon 888 hardware
5. **Targeted Gap Resolution** - Empirical work to close the remaining gaps:
   - CTC Integer Arithmetic Numerical Experiment: Quantified log-sum-exp approximation error
   - QNN Deployment Verification Path: Step-by-step hardware validation plan for Snapdragon 888
6. **Final Review** - Independent verification that both gaps are sufficiently closed to authorize Phase 1

### Current Status: **READY FOR PHASE 1**

As confirmed by Morgan's final binary review, the research has sufficiently addressed the identified evidence gaps and is authorized to proceed to Phase 1: prototype implementation and initial validation.

### Key Artifacts

All work is documented in `/home/ortluk/research/`:
- `ctc_numerical_experiment.md` - Empirical validation of CTC integer arithmetic feasibility
- `qnn_deployment_verification_path.md` - Hardware validation pathway for Snapdragon 888
- `morgan_final_binary_review.md` - Final readiness assessment
- Enhanced source documents showing evidence vs. inference distinctions

### Next Steps (Phase 1)

Proceed to prototype implementation including:
1. Integer-arctic CTC model training with validated approximation techniques
2. QNN model conversion and context generation
3. Initial on-device testing on Snapdragon 888 hardware
4. Latency, power, and numerical agreement measurements

### Collaboration

See [AGENTS.md](AGENTS.md) for team procedures, roles (Nemo, Ada, Morgan), and collaboration surfaces.
