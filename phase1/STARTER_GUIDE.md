# Phase 1 Starter Implementation
## Integer-Arithmetic CTC Speech-to-Text

## Environment Status
- Using system Python (consider setting up .venv-qnn for full PyTorch/ONNX)
- Framework cloning: In progress (network-dependent)
- Core principles: Can be verified conceptually

## Immediate Work Plan:

### Phase 1 Week 1: Foundation
1. **Environment Setup**: Configure .venv-qnn or install PyTorch/ONNX
2. **Framework Evaluation**: Test NITI/NITRO-D when accessible
3. **Integer-Only Concept Validation**: Proof-of-concept showing principles
4. **Spectrogram Processing**: Basic audio-to-spectrogram pipeline

### Phase 1 Week 2: Architecture
1. **Minimal STT Architecture**: Simple feed-forward network on spectrograms
2. **Integer-Only Verification**: Logging/assertions to confirm no FP32 tensors
3. **CTC Loss Research**: Study log-domain approximations for integer CTC

### Phase 1 Week 3: Integration
1. **Framework Integration**: Adapt STT to chosen integer-only framework
2. **Verification Setup**: Implement comprehensive integer-only training checks
3. **Small PoC**: Speech Commands subset with tiny model (10-20 hidden units)

### Phase 1 Week 4: QNN Preparation
1. **Model Export**: Prepare trained model for ONNX conversion
2. **QNN Readiness**: Review deployment verification path documentation
3. **Hardware Validation**: Finalize Snapdragon 888 testing approach

## Success Criteria for Phase 1:
- Demonstrate integer-only training principles work for STT
- Validate CTC loss can be approximated in integer domain  
- Prepare model for QNN deployment verification
- Establish clear path to hardware validation on Snapdragon 888

## Available Documentation:
- All research papers in /home/ortluk/research/
- Framework references: NITI (https://github.com/wangmaolin/niti), NITRO-D (https://github.com/albertopirillo/NITRO-D)
- QNN deployment path: /home/ortluk/research/qnn_deployment_verification_path.md
- CTC numerical experiment: /home/ortluk/research/ctc_numerical_experiment.md
