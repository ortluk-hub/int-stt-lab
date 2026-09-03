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

## Recommended Next Steps for Nemo
1. Accept NITI framework choice and create phase1/niti-prototype branch
2. Implement verification checklist:
   - Tensor-type audit to confirm no floating-point master weights
   - Optimizer state integer-only verification  
   - CTC loss approximation error measurement (< 1e-3 target)
   - Reproducibility validation
3. Document any deviations from research assumptions in implementation PR
4. Consider PRIOT as fallback if CTC loss approximation cannot meet error targets

## Supporting Files
- Research summary: research/RESEARCH_SUMMARY.md
- Constraints analysis: research/constraints.md  
- Recommendations: research/recommendations.md
- Evidence vs inference: research/evidence_vs_inference.md