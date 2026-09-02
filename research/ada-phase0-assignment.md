# Ada Phase 0 Research Assignment

## Objective
Investigate integer-only/fixed-point neural network training methods and Snapdragon 888 QNN/HTP deployment constraints for speech-to-text models.

## Research Questions (from docs/project_plan.md Section 2)
Ada should investigate:

1. Prior work on integer-only / fixed-point neural-network training
2. Gradient, accumulator, scaling, saturation, rounding, optimizer-state, and update strategies
3. Whether some training state must use wider integer types and what qualifies as genuinely integer training
4. Architectures that minimize numerically awkward operations while remaining viable for STT
5. CTC and alternative decoding/training implications for integer arithmetic
6. Snapdragon 888 / QNN / HTP supported operator and quantization constraints relevant to candidate architectures
7. Practical feature-extraction choices and whether preprocessing should remain CPU-side or be NPU-compatible
8. Smallest falsifiable proof-of-concept that can test the training method cheaply

## Required Deliverables
Ada's research handoff must follow `AGENTS.md` and include:
- Sources consulted
- Evidence vs inference distinction
- Confidence levels and uncertainties
- Discovered constraints (hard requirements vs hypotheses)
- Recommendations/questions for Nemo
- Suggested candidate training arithmetic schemes and model families
- Verification methods for integer-only training claim

## Entry Criteria
- Access to local speech dataset on server
- Ability to consult research literature and technical documentation
- Understanding of Snapdragon 888 SM8350 HTP capabilities

## Exit Criteria
- Comprehensive research report addressing all questions above
- Clear recommendations for Phase 0 feasibility work
- Identified constraints that must be treated as hard requirements
- Suggested verification methods for integer-only training claims

## Timeline
Begin immediately and report findings when complete. Nemo will wait for Ada's committed findings before proceeding.