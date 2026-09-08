Who's ball: Nemo

Current Task: Clean-base milestone (D-001: depth 3, dim 128, no exponent
management, 50k samples, 3000 steps). Run complete 2026-09-08 02:18; failure
diagnosed to code level (D-002).

Status:
- Outcome: WER 1.000 at all 12 evals, blank 0.988, uniq <=3; loss 599->8
  (all-blank optimum by ~step 100) then climbed to ~72. No exponent rails ever
  appeared (act exps frozen at init ladder 3/9/16/21/27; sat <=0.41%).
- Mechanism (D-002, docs/decisions.md): body weight codes pinned at int8 rails
  with outward-pointing quantized grads; int8_clip reverts every update ->
  proj/blocks.0/blocks.1/blocks.2 permanently frozen (0 code changes); only
  the head trains; head-only cannot escape all-blank.
- Enablers: weight_quant clamp_min(1) discards Xavier scale (all tensors
  renormalized to max|w| ~0.99, mean|code| ~63); update grad codes +-75..116
  vs weight range +-127 (uncontrolled integer LR); rails reached by ~step 250.
- D-001 intervention queue retired as specified (rails trigger never fired).

Verification Artifacts:
- checkpoints/cleanbase-d3-128/ (metrics.jsonl, history.json, best.pt, latest.pt) - committed
- scripts/probe_weight_rail_freeze.py (one-step freeze diagnosis) - committed
- logs/cleanbase_d3_128.log (full run log, on disk)

Review Notes: Ready for Morgan's review of D-002 and the probe evidence.
Next-run options A/B/C drafted in D-002; decision pending user - no new run
launched. Suggested additions for any next run: post-quantization grad-code
instrumentation and --stop-on-head-collapse (this run would have stopped
~step 2000 and saved ~30 min).