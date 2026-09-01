# Morgan's Review of Project Plan (Phase 0)

## Review Date: 2026-09-01
## Reviewer: Morgan (Independent Reviewer)
## Document Reviewed: docs/project_plan.md

### Summary
The project plan adequately captures the hard constraints and outlines a phased approach. However, there are a few points that need clarification or adjustment to ensure the plan is actionable and aligns with the integer-only training claim.

### Findings

1. **Hard Constraints Section**: The updated hard constraints section is good, but it might be beneficial to explicitly mention that the integer-only training claim must be verifiable at each step (e.g., via logging or assertions) to prevent accidental use of floating-point shadow weights.

2. **Phase 1 Details**: The additions to Phase 1 are helpful. However, it might be useful to specify how the integer-only training claim will be verified (e.g., by using the framework's built-in integer-only operations and checking that no floating-point tensors are created for weights, gradients, or optimizer state).

3. **CTC Loss Investigation**: The recommendation to investigate CTC loss in integer arithmetic is noted. It might be worth adding a note that if a purely integer CTC loss proves infeasible, the project may need to consider alternative loss functions or decoding strategies that are integer-friendly.

4. **Feature Extraction**: The plan mentions keeping feature extraction on CPU due to difficulty mapping to NPU. This is a valid constraint, but it should be explicitly called out as a potential deviation from the goal of NPU execution for the entire pipeline. The project should decide whether to accept CPU-side feature extraction as a temporary constraint or to invest in making it NPU-compatible.

5. **Proof-of-concept**: The suggested proof-of-concept using a small subset of Speech Commands and a tiny model is appropriate. However, it should be made explicit that the proof-of-concept must demonstrate both training loss reduction and the ability to decode (even if poorly) to show that the integer-only training is not breaking the model's ability to learn.

6. **Deployment Proof**: The requirement to demonstrate actual HTP/NPU execution without silent CPU fallback is critical. The plan should include a step to use Qualcomm's profiling tools to verify that the model is running on the HTP and not falling back to CPU.

### Recommendations for Nemo
- Add a verification step in each phase to confirm the integer-only training claim (e.g., framework-specific checks).
- Clarify the status of feature extraction (CPU vs NPU) and treat it as a constraint if it remains on CPU.
- Include a deployment verification step using HTP profiling tools in Phase 4.
- Consider adding a note about the potential need to adjust the loss function if integer-only CTC proves too challenging.

### Conclusion
The project plan is in good shape and meets the requirements outlined in GitHub Issue #1. With the above adjustments, it will be even more robust and actionable.
