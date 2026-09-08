Who's ball: Nemo

Current Task: Rail-rescale arm (D-005/D-006) complete. Verdict: the D-004
capacity-loss hypothesis is REFUTED - with rails fully managed (221/354/40/9/0
rescales across layers, rail fractions <= 19%), the run collapsed to all-blank
identically to gs3 (uniq 212->20->8->3->1 by step 1000, WER 1.000 after).

Status:
- Blank collapse is semantic, not mechanical: the head sits at the all-blank
  shortcut optimum (grads ~zero, 0 rescales, exp unchanged) while body layers
  receive consistent-sign gradients (~7.4 codes/step net outward, sustained
  3000 steps - far beyond quantizer-bias magnitude). The rail march was a
  symptom of that pressure all along.
- Body weight VALUE growth (blocks.0 exp -9 -> +345) is forward-invariant
  (act_calc renormalizes; ReLU homogeneous) - parametrization churn, invisible
  to loss. Rail management kept in the standard config as cheap insurance.
- Three-run arc now complete and committed: failed run (mechanism found) ->
  gs3 (mechanics fixed, collapse persists) -> gs3-rq (rails managed, collapse
  identical). The binding constraint is now squarely the CTC shortcut.

Next decision pending user (D-006 options):
- A (recommended): blank-logit suppression arm - integer code offset on the
  head's blank column in the training CTC path only (-8 codes step <1000,
  -4 <1500, 0 after); eval decodes unbiased.
- B: integer LR schedule (gs3 -> gs5 after step ~500).
- C: larger batch (direction unclear).

Verification Artifacts:
- docs/decisions.md D-001..D-006 (committed)
- checkpoints/cleanbase-d3-128{,-gs3,-gs3-rq}/ (all committed)
- scripts/probe_weight_rail_freeze.py (committed)

Review Notes: Morgan review of D-002..D-006 welcome. No further runs launched
without explicit user decision.