# Review of Ada's Phase 1 handoff (Morgan → Nemo)

## Verified claims
- The repository contains a `phase1/niti/README.md` and related Docker setup, confirming that the NITI integer‑only training framework is present in the codebase.
- Project documentation (`docs/project_plan.md` and `handoff_status.md`) references NITI as the selected framework for Phase 1, matching Ada's recommendation.
- The handoff correctly lists a concrete verification checklist (tensor‑type audit, no FP master weights, optimizer state check, CTC loss approximation comparison, reproducibility) which aligns with best practices for integer‑only training.

## Claims requiring further evidence / unverified
- **No FP master weights**: The handoff asserts that NITI never creates hidden `float32` master copies. The repository does not currently contain tests or runtime checks that confirm this property. Nemo should add an explicit assertion (e.g., `assert not hasattr(model, "_fp_master")`) in the training script.
- **CTC loss integer approximation error < 1e‑3**: The handoff proposes a quantitative error bound, but no baseline FP implementation or empirical measurement is provided. Nemo should generate a small validation set and report the measured error.
- **Compute environment suitability**: Ada asks whether the existing Dockerfile in `phase1/niti/` is sufficient. This needs clarification from the engineering team regarding Python version and library dependencies.

## Recommendations for Nemo
1. Accept the framework choice (NITI) and create the branch `phase1/niti‑prototype` as suggested.
2. Implement the verification checklist, adding concrete tests for points (a) and (b) above.
3. Document any deviations from Ada's assumptions in the implementation PR.
4. If the CTC loss approximation cannot meet the error target, consider the alternative PRIOT framework as a fallback.

---
*This review handoff follows the standard concise format: verified facts, points needing evidence, and actionable recommendations for the next engineer (Nemo).*