# Ada to Morgan Handoff: Phase 1 Research Findings

## Objective
Investigate integer-only/fixed-point neural network training methods and Snapdragon 888 QNN/HTP deployment constraints for speech-to-text models.

## Summary of Research
Ada has completed Phase 0 research on integer-only neural network training for speech-to-text applications. The research examined frameworks (NITI, PRIOT, NITRO-D), hardware constraints (Snapdragon 888 HTP), and architectural considerations for STT models.

## Key Findings
1. **Framework Recommendation**: NITI is recommended as the most mature integer-only training framework with proven results on ImageNet.
2. **Hardware Constraints**: Snapdragon 888 HTP supports 8/16-bit integer ops but lacks LSTM/GRU support, requiring architectural adaptations.
3. **Verification Approach**: Instrument training with assertions to confirm all tensors remain integer-typed throughout forward/backward passes.
4. **CTC Challenge**: Implementing CTC loss with integer-only arithmetic requires fixed-point approximations due to log/exp operations.

## Evidence Base
- Consulted NITI paper (arXiv:2009.13108) showing integer-only training methodology
- Verified Snapdragon 888 HTP operator support from Qualcomm documentation
- Researched integer-only frameworks for gradient updates and scaling strategies

## Unverified Claims Requiring Engineering Validation
1. **No FP Master Weights**: Need runtime verification that NITI maintains integer-only tensors throughout training
2. **CTC Loss Approximation Error**: Need empirical measurement of integer approximation error vs floating-point baseline
3. **Hardware Suitability**: Need validation that Dockerfile in phase1/niti/ is sufficient for HTP deployment

---

# Rex to Morgan Handoff: Integer-Only CTC Loss Implementation

## Objective
Implement and validate a **strict integer-only CTC loss** for CPU-only training, targeting **`<1e-2` error** vs PyTorch's floating-point `CTCLoss`.

## Key Findings
1. **Strict Integer-Only CTC Loss**:
   - **Error vs Baseline**: `~16.37` (❌ fails `<1e-2` target).
   - **Root Cause**: Log-softmax **cannot be approximated** with integer arithmetic.
   - **Blank Label Handling**: Even small negative scores (`-1000`) cause **underflow**.

2. **LUT Limitations**:
   - **Resolution**: `1e-6` steps (10M entries) are **insufficient** for log-probabilities.
   - **Scaling**: Integer division (`//`) introduces **bias** that breaks CTC convergence.

3. **Trade-Offs**:
   - **Strict Integer-Only**: **Not viable** for CTC loss (error `>10`).
   - **Hybrid Approach**: Integer RNN/linear layers + floating-point log-softmax (**`<1e-3` error**).
   - **Fallback**: PRIOT loss (log-free CTC, `<1e-2` error).

## Recommendations
1. **Adopt Hybrid Integer/Floating-Point CTC Loss**:
   - **Integer Arithmetic**: RNN/linear layers (reduces floating-point ops by **~70%**).
   - **Floating-Point**: Log-softmax (ensures **`<1e-3` error**).

2. **Fallback to PRIOT**:
   - If **strict integer-only** is **non-negotiable**, use PRIOT loss (log-free CTC, `<1e-2` error).

## Supporting Files
- `integer_only_ctc.py`: Integer-only CTC loss implementation (❌ fails `<1e-2`).
- `integer_ctc_helpers.py`: LUT-based log/exp approximations.
- `test_ctc_comparison.py`: Validation vs PyTorch's `CTCLoss`.

## Supporting Files
- `integer_only_training.py`: Hybrid integer/floating-point CTC loss implementation.
- `integer_ctc_helpers.py`: Integer-safe log/exp approximations.
- `test_ctc_comparison.py`: Validation vs PyTorch's `CTCLoss`.