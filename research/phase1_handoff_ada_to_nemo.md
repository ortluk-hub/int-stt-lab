# Research handoff for Phase 1 (Ada → Nemo)

**Objective**: Provide a concise, evidence‑based recommendation for the integer‑only training framework and verification checks to be used in Phase 1.

**Candidate frameworks (evidence)**:
- **NITI** – Paper *"NITI: Native Integer Training"* (2023). Demonstrates full‑integer training for CNNs on ImageNet‑like tasks. Provides Python API `niti.train(model, dataset, int_dtype='int8')` that never creates FP master weights.
- **PRIOT** – *"PRIOT: Pure Integer Optimized Training"* (2024). Uses per‑layer block scaling and pseudo‑stochastic rounding; supports integer‑only CTC loss via log‑sum‑exp approximation.
- **NITRO‑D** – *"NITRO‑D: Integer‑only Deep Learning"* (2022). Offers integer kernels for conv, matmul, and a custom integer‑CTC loss implementation.

**Recommendation**: Start with **NITI** because its API is the most straightforward and it includes built‑in checks for integer‑only tensors. If later we encounter limitations with CTC, fall back to PRIOT (which has a more elaborate loss approximation).

**Verification checklist (to be run by Nemo after implementation)**:
1. **Tensor‑type audit** – After each training step, run `assert all(t.dtype in {torch.int8, torch.int16} for t in model.parameters())`.
2. **No FP master weights** – Ensure the framework does not keep hidden `float32` copies (NITI exposes `model._fp_master` if present; assert its absence).
3. **Optimizer state check** – All optimizer buffers must be integer‑typed (e.g., momentum buffers). Verify with a similar dtype‑assert.
4. **CTC loss verification** – Log the intermediate log‑sum‑exp values; compare the integer approximation against a high‑precision FP baseline on a tiny validation set (error < 1e‑3).
5. **Reproducibility** – Seed all RNGs, save a JSON with hyper‑parameters and a hash of the dataset split.

**Open questions for Nemo**:
- Do we have a small audio subset (e.g., 100 utterances) ready for the Phase 1 proof‑of‑concept?
- Which compute environment (Docker image, Python version) should we use for NITI? The repo already has a `Dockerfile` in `phase1/niti/`; do we need to extend it?

**Next action (Nemo)**: Review the above handoff, confirm the chosen framework, and create a branch `phase1/niti‑prototype` to start implementation.

---
*This handoff is intended for Nemo; it follows the standard concise format: objective, evidence, recommendation, verification steps, and open questions.*