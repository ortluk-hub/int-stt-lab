Who's ball: Nemo

Current Task: Blank-suppression arm (D-007, D-006 option A):
cleanbase-d3-128-bs. Same config as gs3-rq plus training-time blank-logit
suppression (8-code offset on the head's blank column in the training CTC
loss, steps <1000; halved to 4 until step 1500; 0 after; eval decodes
unbiased). Launched 2026-09-08 ~12:40, ETA ~14:25; monitor every 20 min.

Status:
- Three-run arc complete (failed -> gs3 -> gs3-rq): integer mechanics are
  sound; the binding constraint is the CTC all-blank shortcut. Rail
  management stays in the standard config as free insurance.
- D-007 tests the semantic hypothesis directly. Success: uniq >= ~10 or
  WER < 1 through the step 250-1000 window and surviving ramp-off.
- Completion routine: trajectory + four-way comparison, D-008 draft if
  evidence is clear, artifact commit, cron cleanup.

Verification Artifacts:
- docs/decisions.md D-001..D-007 (committed)
- checkpoints/cleanbase-d3-128{,-gs3,-gs3-rq}/ (committed); -bs as produced
- scripts/probe_weight_rail_freeze.py (committed)

Review Notes: Morgan review of D-002..D-007 welcome while the arm runs.
No further runs without explicit user decision. Next candidates if the
blank-suppression arm fails per its documented modes: longer suppression
hold, permanent low offset, integer LR schedule, or more alignment
capacity/epochs.