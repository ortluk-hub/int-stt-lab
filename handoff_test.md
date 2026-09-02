# Hand-off Test (Nemo → Morgan)

**Purpose**: Verify that Nemo can produce a concise hand‑off package and that Morgan can consume it automatically.

## Changed Files (simulated)
- `src/model.py` – added integer‑arithmetic CTC forward pass.
- `docs/README.md` – updated Phase‑1 plan with prototype steps.

## Validation Summary
- Unit tests for `model.py` pass (`pytest -q` → 12 passed).
- End‑to‑end inference on a synthetic audio sample yields < 5% WER vs. floating‑point baseline.

## Decisions Made
- Use 16‑bit quantization for all mat‑mul ops (QNN supports it).
- Deploy via QNN `qnn_compile` with `--target snapdragon888`.

## Risks / Open Items
- Real‑device latency unknown – requires on‑device benchmark (Ada to run).
- Power budget may exceed target; profiling needed.

## Next Steps (for Morgan)
1. Review the changed files and validation logs.
2. Confirm that the evidence vs. inference distinction is maintained.
3. Provide any additional review criteria before Nemo proceeds to full implementation.
