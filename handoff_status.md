Who's ball: Nemo

Current Task: Rail-rescale arm (D-005, option C): cleanbase-d3-128-gs3-rq.
Same config as the gs3 arm plus value-preserving weight re-quantization on
rail (at-rail >= 25% -> codes>>1, weight_exp+1, checked every step per layer).

Status:
- Unit + smoke verification passed (rescale triggers correctly, values
  preserved within half the new quantum, no spurious early rescales).
- Instrumentation bug found and fixed: at-rail predicate missed the -127
  negative rail (int8_clip keeps weights >= -127); gs3 metrics undercounted
  rails ~2x - proj's "50% plateau" was ~100% pinned. Recorded in D-005.
- Run launched 2026-09-08 ~10:10, ETA ~11:55; monitor checks every 20 min.
- Completion routine: trajectory summary, three-way comparison (failed run /
  gs3 / gs3-rq), D-006 draft, artifact commit, cron cleanup.

Verification Artifacts:
- docs/decisions.md D-001..D-005 (committed)
- checkpoints/cleanbase-d3-128{,-gs3,-gs3-rq}/ (committed as produced)
- scripts/probe_weight_rail_freeze.py (committed)

Review Notes: Morgan review of D-002..D-005 welcome while the arm runs.
The gs3-rq run directly tests D-004's capacity-loss hypothesis: if uniq
survives past step ~750 with rails managed, the rail march is the blank-
collapse driver; if not, semantic levers next (integer LR schedule, blank
suppression, larger batch). No further runs without user decision.