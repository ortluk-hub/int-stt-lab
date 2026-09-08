Who's ball: Nemo

Current Task: Update-safety arm (D-003/D-004) complete. cleanbase-d3-128-gs3
ran the full 3000 steps: mechanics fixed (all layers trainable ~10x longer,
head healthy throughout, no tie-up), but the run still ends blank-collapsed.
The step-250 diversity window (uniq 212, phonetic babble) dies exactly as
the weight-rail march re-pins the body (w_rail 0.1% -> 43% over steps
250-750; body frozen again by step 2500).

Status:
- D-004 drafted in docs/decisions.md: rail-march capacity loss is now the
  primary suspect for BOTH failures; option C (weight re-quantization on
  rail, value-preserving codes>>1 + weight_exp+=1) is the recommended next
  arm and the direct test of that hypothesis.
- Run artifacts committed: checkpoints/cleanbase-d3-128-gs3/
- best.pt caveat: best-by-WER saved a collapsed state; WER is meaningless
  during the babble phase (over-generation > 1.0).

Verification Artifacts:
- docs/decisions.md D-001..D-004 (committed)
- checkpoints/cleanbase-d3-128-gs3/ and cleanbase-d3-128/ (both committed)
- scripts/probe_weight_rail_freeze.py (committed)

Review Notes: Morgan review of D-002..D-004 welcome. Next run pending user
decision: option C (rail re-quantization) recommended; complementary levers
if C alone fails: integer LR schedule, blank-logit suppression, larger batch.
No further runs launched without explicit user confirmation.