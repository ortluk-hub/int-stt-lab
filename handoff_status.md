Who's ball: Nemo

Current Task: Blank-cap arm (D-009/D-010) complete - failed its pre-registered
mode (collapse under an unpayable cap: uniq 1 at every eval, blank ~0.97 at
the cap margin, head frozen from step 500). Shaping arms stop. Five-arm arc
committed: failed -> gs3 -> gs3-rq -> bs -> bc.

Status:
- DECISIVE NEW FINDING (D-010): the int8 error signal discards ~99.9% of the
  gradient at plateau magnitudes. Probe at the step-750 bc checkpoint: float
  CTC grad 88.3% nonzero / max 0.0131 -> int8 err 0.08% nonzero / max code 2
  -> head receives ~zero signal (0 weight codes changed/step). Root cause is
  float_to_int8's clamp_min(1) - the SAME port-artifact family as the D-002
  weight-init bug; NITI's original TiFloatToInt8 has no clamp and would keep
  the error dense (~+-107 codes at exp ~-13). Not intrinsic to integer-only
  training - a one-function fix.
- Answer to the standing question ("is the integer constraint the problem?"):
  one quantizer bug was, and it is repairable; capacity/budget (210k-param
  MLP, 12k samples seen = 0.24 epochs) remains the other suspect.
- D-010 options pending user: A NITI-faithful error scaling (recommended),
  B float control twin (discriminator if A fails), C capacity/budget arm.

Verification Artifacts:
- docs/decisions.md D-001..D-010 (committed)
- checkpoints/cleanbase-d3-128{,-gs3,-gs3-rq,-bs,-bc}/ (all committed)
- scripts/probe_weight_rail_freeze.py (committed)

Review Notes: Morgan review of D-002..D-010 welcome. No further runs without
explicit user decision. For arm A: verify err_exp bookkeeping (grad_exp =
err_exp + grad_shift + act_in_exp) and update alignment in smoke before
launch.