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

---

# Decision Log — 2026-09-08 (clean-base run completion)

## D-002: Clean-base failure is a weight-rail freeze, not exponent rails

**Decision:** Retire the D-001 intervention queue as specified (its trigger never
fired). The clean-base run failed by a different mechanism, diagnosed to code
level: body-layer weight codes are pinned at the int8 rails with outward-pointing
quantized gradients, so `int8_clip` in `weight_update` reverts every update to
the same code — 4 of 5 weight tensors are permanently frozen. Next-run design
pending user decision (options below); no new run launched.

**Evidence:**

1. Full-run trajectory (checkpoints/cleanbase-d3-128/metrics.jsonl, 12 evals,
   every 250 steps): WER 1.000 at *every* eval, blank 0.992→0.988, uniq 1→3.
   Loss 599→8 by ~step 100 (the trivial all-blank optimum), then climbed with
   oscillation to 71.8 at step 3000 — never better than trivial, ending worse.
   **No exponent rails ever appeared**: act exps frozen at the init ladder
   3/9/16/21/27 for the whole run; activation sat ≤0.41%; clamp ≤0.003%.
2. Freeze timeline (weight codes at rails):
   | when | proj | blocks.0 | blocks.1 | blocks.2 | head |
   |---|---|---|---|---|---|
   | init (seed 0) | 0.21% | 0.20% | 0.19% | 0.27% | 0.20% |
   | step 250 | 46.1% | 50.0% | 36.1% | 29.7% | 4.1% |
   | step 3000 | 100% | 99.2% | 55.3% | 61.1% | 26.0% |
   pct_changed confirms the freeze: proj/blocks.0/blocks.1 exactly 0.0% at
   every eval from step 500 on; blocks.2 decayed 13.6%→0.1%; head 50%→33%.
3. One-step probe at end-of-run weights (scripts/probe_weight_rail_freeze.py,
   real dev batch): body quantized grad codes are large and dense (proj 100%
   nonzero, max|code| 107; blocks.0 99%/116) and point outward on railed
   entries — the int16 subtract overshoots the rail, int8_clip clamps back:
   30720/30720 proj updates revert. Actual code changes: 0 for all four body
   layers, 486 for head (head grads only 0.84% nonzero, max 70).
4. Init enabler (verified at seed 0): `weight_quant` takes codes from
   `round(w/max|w| × 127)` but the exponent from
   `ceil(log2(max|w|.clamp_min(1))) − 7`. Every Xavier tensor has max|w| < 1,
   so the clamp forces weight_exp = −7 for **all** tensors and the effective
   weights are renormalized to max|w| ≈ 0.99 (Xavier bound 0.07–0.15; 6.5–14×
   too large; mean|code| ≈ 63). Two consequences: weights start ~63 codes from
   the rail so ~1–2 consistent ±100-code updates rail them; and the +6/block
   act-exponent ladder is pinned by init (mean|code| ≈ 2⁶), not by dynamics —
   which is why "exponent stability" looked healthy all run.

**Reasoning:**
1. Failure chain: renormalized init → consistent outward grad pressure during
   the first ~100 steps (loss 599→8 while the head learns all-blank) rails the
   body by step ~250 → clamp makes body updates permanent no-ops → only the
   head keeps training → head alone cannot escape the all-blank optimum →
   loss rises/oscillates while head logits tie up (top1–top2 gap 2.67→0.09
   codes, tied 7%→91%).
2. The 40-step healthy reference (uniq 52, D-001) fits: the rail march takes
   ~250 steps, so step-40 diversity predates the freeze.
3. Update magnitude is uncontrolled: quantized grad codes (±75–116 body,
   ±70 head) are the same order as the weight code range (±127) — integer SGD
   at effectively full-range LR. The D-001 lever (grad_shift) targets exactly
   this family but 1 bit (2×) is too little against codes this large; a faithful Xavier init alone only buys ~1.7× headroom (max code
   ≈ 74–77 at the correct exponent), so init fix and update-magnitude control
   are both needed, possibly with weight re-quantization on rail (weight_exp
   management, the weight-space analogue of the act rescale).
4. Exponent management (recenter/branch) remains irrelevant at this scale:
   activation rails never appeared. D-001's rails-triggered grad_shift 1
   queue entry is obsolete; the grad_shift *family* survives with a revised
   purpose (update-magnitude control) and needs a revised magnitude (≥2–3
   bits), which shrinks head updates too — a per-layer or saturation-aware
   update policy is an open design point.

**Next-run options (pending user decision, NOT pre-authorized):**
- A. Update-safety arm (recommended): fix `weight_quant` exponent (drop
  `clamp_min(1)`, quantize by 2^exp so Xavier scale survives) + grad_shift 3,
  same clean-base config otherwise, plus post-quantization grad-code
  instrumentation (nonzero%, max code, at-rail-outward% per layer).
- B. Strict single-variable attribution: two arms — init-fix only, grad_shift
  only — slower but isolates which lever un-freezes the body.
- C. Add weight re-quantization on rail (renormalize tensor + decrement
  weight_exp when >X% of codes rail) — the principled long-term fix, more
  implementation risk.

**Confounds / limits:** probe run at end-of-run weights (the rails are also an
*effect* of training; the step-250 saturation data shows the march was real
and early); one dev batch; head grad sparsity (0.84%) is itself end-state (tied
logits → near-uniform softmax → tiny CTC grads), not a cause.

**Artifacts:** checkpoints/cleanbase-d3-128/{metrics.jsonl,history.json,
best.pt,latest.pt} (committed), logs/cleanbase_d3_128.log,
scripts/probe_weight_rail_freeze.py (committed).