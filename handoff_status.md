Who's ball: Nemo

Current Task: Update-safety arm (D-003, option A): cleanbase-d3-128-gs3.
weight_quant now preserves Xavier scale (NITI TiFloatToInt8 semantics); run
launched with grad_shift 3, freeze-signature instrumentation, and
--stop-on-head-collapse. Clean-base config otherwise identical (depth 3,
dim 128, no exponent management, batch 4, 50k samples, 3000 steps, eval/250).

Status:
- Pre-launch smoke passed: all five layers update codes every step (no
  freeze), grad codes max 9-10 (were +-75-116), loss 374->83 over 3 steps,
  eval path OK, integer-state report OK.
- Run: checkpoints/cleanbase-d3-128-gs3, log logs/cleanbase_d3_128_gs3.log.
  Monitor checks every 20 min; completion routine = trajectory summary +
  comparison vs the failed run (checkpoints/cleanbase-d3-128), D-004 draft if
  evidence is clear, artifact commit, cron cleanup.
- D-002 root cause (weight-rail freeze) and probe committed; D-001 queue
  retired as specified.

Verification Artifacts:
- docs/decisions.md D-001..D-003 (committed)
- scripts/probe_weight_rail_freeze.py (committed)
- checkpoints/cleanbase-d3-128/ (failed run, committed)

Review Notes: Morgan review of D-002/D-003 welcome while the arm runs.
Watch items in D-003 (w_rail_frac march, body pct_changed, uniq/blank escape).
No further runs launched without user decision; next candidate levers:
per-layer update scaling (if head underfits), weight re-quantization on rail
(option C, if the march recurs).