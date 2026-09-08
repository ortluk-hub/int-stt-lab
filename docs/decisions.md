# Decision Log — 2026-09-08 (overnight autonomous session)

## D-001: Clean-base training configuration

**Decision:** Launch the clean-base run as **depth 3, dim 128, NO exponent management**
(no recentering, no branch shift, grad_shift 0), scaled to the real training regime:
batch 4, 50k train samples (≤8s), 3000 steps, eval every 250 steps on the fixed 64-utterance
dev slice, full JSONL instrumentation, MUSAN off (clean-base-first milestone).

**Evidence (all runs, fixed 16-utt eval slice, dim 128 unless noted):**

| run | config | outcome |
|---|---|---|
| run 1 (3000 steps) | depth 6, dim 256, no mgmt | WER 1.000 flat; head exp ~47; logits tied at rails; blank collapse terminal |
| healthy ref (40 steps) | depth 3, no mgmt | **uniq 52**, blank 0.63 — the only config that produced diverse text |
| arm C (60 steps) | depth 6, dim 256, no mgmt | head exp 48 at step 60; blank re-collapsing 0.61→0.98 — width confirmed as exponent amplifier |
| control (60 steps) | depth 6, dim 128, no mgmt | blank 0.17 but single-token babble (" is"); head exp 30 climbing |
| arm B (60 steps) | depth 6 + recenter + branch ½ | blank 0.68 stuck, uniq 1 |
| arm A (60 steps) | depth 6 + recenter | escaped blank (0.016) but uniq 3 (" the the the") |
| **arm AX (400 steps)** | depth 6 + recenter | ping-pong: blank 0.26→0.65→0.52→0.82→0.54, uniq 1 after step 100, loss 8→62 |
| **arm D3R (400 steps)** | depth 3 + recenter | ping-pong: blank 0.98→0.04→0.94→0.00→0.85, uniq ≤9 in bursts, uniq 1 at end |

**Reasoning:**
1. Every managed configuration is numerically healthy (head exp ≤17, sat 0%) but
   semantically stuck: max uniq 9 in bursts vs 52 for unmanaged depth 3. Recentering's
   value division is now implicated in diversity loss *independently of depth* (D3R result);
   branch ½ additionally caused blank regression (A vs B). My earlier "recentering
   exonerated" read (from arm A at 60 steps) was premature.
2. The only configuration with evidence of real token diversity is unmanaged depth 3.
   Its known failure mode (exponent growth) has never been observed past step 40 —
   the clean-base run with instrumentation will show whether it rails at scale, and when.
3. The 400-step ablations ran on 512 samples (~1.6 epochs): attractor ping-pong may be
   small-data cycling. The 50k-sample run changes that regime; treat ablation conclusions
   about dynamics as provisional until confirmed at scale.

**Confounds / limits of this evidence:** 512-sample ablation regime; 16-utt eval slice;
recenter targets derived from a dim-128 reference (may not transfer across width);
uniq counts from 16-utt greedy decodes.

**Queued interventions if the clean-base run degrades (in order):**
1. Exponent rails appear mid-run → grad_shift 1 (integer LR downshift, 2× smaller updates)
   on the same config — the designed lever for coarse-update oscillation.
2. If grad-shift insufficient → gentler management: ¾ branch mix (y>>1 + y>>2), recenter
   only at the last block boundary instead of all boundaries.
3. Width ladder (after clean base is healthy at dim 128): healthy dim-256 depth-3 reference
   first to derive width-appropriate targets, then dim-256 depth-3 clean base.

**Baseline anchor (from bench/baseline_ggml-base*.json, committed):** whisper-base fp32
on-device: WER 24.6% clean / 29.8% @20dB / 32.0% @10dB / 45.8% @0dB; RTF 0.89–1.14.
The clean-base milestone target is to close on this curve's clean point with ~10× RTF
headroom via NPU deployment (Phase 4).

**Artifacts:** checkpoints/ablation-{a,b,c,ax,d3r}/metrics.jsonl (all committed),
logs/ablation_AX_D3R.log, checkpoints/int-ctc/{latest,history}.json (run 1).