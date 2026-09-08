Who's ball: Nemo

Current Task: Blank-suppression arm (D-007/D-008) complete - failed mode (c)
with mechanism: the constant blank tax was elastically compensated (head gap
21 codes at step 250; loss tracked the tax amount; uniq 1 at every eval),
and it prevented the babble phase entirely (uniq 1 at step 250 vs 212
without suppression). Four-arm arc now committed: failed -> gs3 -> gs3-rq ->
bs. No cheap lever tried so far prevents the all-blank shortcut.

Status:
- Integer mechanics are sound (init fix + grad_shift 3 + rail rescale are
  the standard config). The binding question has narrowed to: (i) can a
  compensation-proof shaping term hold diversity (blank-cap arm), or
  (ii) is the babble ceiling a capacity/budget limit (210k-param MLP vs the
  ~74M-param whisper-base accuracy anchor).
- D-008 options pending user: A blank-cap (clamp blank at max(non-blank)+K
  codes, train+eval, K~2-4; ~2h), B capacity/budget (10k steps and/or
  dim 256; ~6h+), C sequential.

Verification Artifacts:
- docs/decisions.md D-001..D-008 (committed)
- checkpoints/cleanbase-d3-128{,-gs3,-gs3-rq,-bs}/ (all committed)
- scripts/probe_weight_rail_freeze.py (committed)

Review Notes: Morgan review of D-002..D-008 welcome. No further runs
launched without explicit user decision.