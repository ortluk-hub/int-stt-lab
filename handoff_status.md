Who's ball: Nemo

Current Task: Blank-cap arm (D-009, D-008 option A): cleanbase-d3-128-bc.
Same config as gs3-rq (init fix + grad_shift 3 + rail-rescale 0.25 +
stop-on-head-collapse, no blank tax) plus a permanent blank cap: blank logit
clamped at max(non-blank)+2 codes per frame, identically in training and
eval, implemented in IntCTCEncoder.forward (part of the model definition).
Launched 2026-09-08 ~16:30, ETA ~18:30; monitor every 20 min.

Status:
- The cap is compensation-proof by construction (D-008 lesson: the bs tax
  was elastically paid; the cap binds exactly when blank tries to dominate
  and over-dominance is invisible at the decision surface).
- Unit + end-to-end smoke passed (cap semantics, capped eval decode,
  integer-state report).
- This arm decides D-008's fork: diversity survives the cap -> the shortcut
  was the binding constraint; collapse under an unpayable cap -> next arm is
  capacity/budget (10k steps and/or dim 256) and shaping arms stop.

Verification Artifacts:
- docs/decisions.md D-001..D-009 (committed)
- checkpoints/cleanbase-d3-128{,-gs3,-gs3-rq,-bs}/ (all committed); -bc as produced
- scripts/probe_weight_rail_freeze.py (committed)

Review Notes: Morgan review of D-002..D-009 welcome while the arm runs.
head_separation reads differently under the cap (gap bounded by K on
blank-won frames) - within-arm comparisons only. No further runs without
explicit user decision.