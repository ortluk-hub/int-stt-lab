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

---

## D-003: Launch update-safety arm (option A) — cleanbase-d3-128-gs3

**Decision (user-approved, 2026-09-08):** D-002 option A. Fix the
`weight_quant` init-scale bug, run the clean-base config with grad_shift 3,
record the freeze signature in instrumentation, arm `--stop-on-head-collapse`.
Everything else identical to D-001 (depth 3, dim 128, no exponent management,
batch 4, 50k train samples ≤8s, 3000 steps, eval every 250 on the fixed
64-utterance dev slice).

**Changes:**
- `src/model/int_layers.py` `weight_quant`: NITI `TiFloatToInt8` semantics —
  exponent from the true max-abs (no `clamp_min(1)`), codes = round(w × 2^-exp).
  The `clamp_min(1)` was a port artifact (absent in the original
  `src/niti/ti_torch.py:334` `weight_quant`) that renormalized every sub-unit
  tensor to max|w| ≈ 1 — the D-002 root cause. Verified at seed 0: effective
  max 0.127/0.152/0.072 vs Xavier bounds 0.128/0.153/0.072; weight_exp
  -9/-10 (was forced -7); 0.0% of codes at rails (was 0.2% at init,
  ~50% by step 250).
- `src/train/instrumentation.py` `grad_health`: added per-layer
  `max_abs_code`, `grad_exp`, `w_rail_frac`, `rail_outward_frac`
  (the D-002 freeze signature: weight at rail + grad pointing outward).
- Run flags: `--grad-shift 3` (quantized grad codes ±8-10, were ±75-116),
  `--stop-on-head-collapse` (tied_frac ≥ 0.99 → stop; would have saved the
  failed run ~30 min).

**Pre-launch smoke (3 steps + eval, real dev batch):** all five layers now
update codes every step (proj 91%, blocks 43-48%, head 582 codes), grad codes
max 9-10, loss 374→162→83, eval path OK, integer-state report OK.

**Watch items:**
- `w_rail_frac` / `rail_outward_frac` trajectory: does the rail march recur at
  8× smaller updates? D-002's drift estimate (~0.04 codes/step under gs3)
  puts rails near run end if the outward bias persists. Option C (weight
  re-quantization on rail) is the durable fix if the march recurs.
- Body `pct_changed` > 0 at every eval — no recurrence of the freeze.
- uniq/blank escape vs the failed run (uniq ≤3, blank 0.988 throughout).
- Act-exponent ladders are NOT comparable across runs (init codes ~2× smaller
  → exps ~1 lower per layer); the ablation pass criterion "head exp ≤ 20"
  assumed the old ladder and does not apply to this arm.

**Risk:** grad_shift 3 shrinks head updates too (head grad codes ±9, 0.4%
nonzero at init). If the run underfits (loss plateaus, blank never escapes),
that is evidence for per-layer update scaling (D-002 reasoning 4).

**Artifacts:** checkpoints/cleanbase-d3-128-gs3/ (as produced),
logs/cleanbase_d3_128_gs3.log.

---

## D-004: gs3 arm — mechanics fixed; rail march now implicated in the blank collapse

**Decision:** The D-003 fixes held mechanically (no early freeze, head healthy,
every layer trainable ~10× longer) but the run still ends blank-collapsed
(WER 1.000, uniq 1, blank 0.985). The diversity window at step ~250 produced
the project's first real phonetic text at scale — and dies exactly as the
weight-rail march re-pins the body. Next run pending user decision; primary
candidate is option C (weight re-quantization on rail): it is both the
durable mechanical fix and the direct test of the capacity-loss hypothesis
for the blank collapse.

**Evidence (cleanbase-d3-128-gs3 vs failed run, every 250 steps):**

| step | gs3 loss | WER | uniq | max w_rail | body chg | head gap | | failed loss | uniq | body chg | head gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 250 | 10.2 | **2.42** | **212** | 0.1% | 100% | 3.2 | | 27.4 | 1 | (frozen at 250) | 2.7 |
| 500 | 8.2 | 1.01 | 20 | 7% | 79% | 6.2 | | 31.0 | 1 | 0% | 1.4 |
| 750 | 7.5 | 1.00 | 2 | 43% | 54% | 8.2 | | 48.2 | 1 | 0% | 1.1 |
| 1000 | 8.3 | 1.00 | 1 | 45% | 18% | 6.3 | | 19.0 | 1 | 0% | 2.6 |
| 2000 | 10.4 | 1.00 | 1 | 50% | 1% | 3.0 | | 58.1 | 2 | 0% | 0.19 |
| 3000 | 11.5 | 1.00 | 1 | 50% | 0% | 4.3 | | 71.8 | 3 | 0% | 0.09 |

- Step-250 decodes are genuine phonetic babble, not noise: uniq 212, decoded
  length 53, zero repeats ("tis t inst firstose never t firsted my the
  trothernisedoseo" for "less pressure is needed when playing your
  clarinet"). CER 2.33 — over-generation, but acoustics→text mapping is live.
- Collapse timeline: uniq 212→20→2→1 as w_rail goes 0.1%→7%→43% and body
  change 100%→79%→54% (steps 250→750). By step 2500 proj/blocks.0/blocks.1
  are back to 0.0% change (proj pinned at 50% rail, all outward).
- Head stayed healthy the whole run: gap 3–8 codes, tied ≤ 6.5% (failed run
  ended 0.09 / 91%). Loss held a 7.5–14 band (failed run climbed to 72).
- best.pt is misleading: best-by-WER saved a collapsed state (WER 1.0 "beats"
  the babble's 2.42). WER > 1 during the babble phase is expected
  over-generation; the metric only becomes meaningful once alignment emerges.

**Reasoning:**
1. The mechanical and semantic failures may be one failure: the blank
   collapse (steps 250–750) coincides with the rail march pinning 40–50% of
   body weights. Correlation, not proof — option C is the test: if diversity
   survives past step 750 with rails managed, capacity loss is implicated;
   if it still collapses, the residual is a classic CTC shortcut needing
   semantic countermeasures.
2. Sustained outward pressure comes from the all-blank descent phase: the
   CTC error signal pushes body weights in a consistent direction; at ±8–16
   code updates the march takes ~500–750 steps (vs ~250 at ±100 codes).
3. Option C sketch: when a layer's at-rail fraction crosses a threshold
   (e.g. 25%), rescale value-preserving — codes >>= 1, weight_exp += 1 —
   restoring code headroom at 1 bit of precision per rescale. Risks:
   repeated rescales erode precision; sustained pressure may re-rail
   quickly. Cadence/threshold need the w_rail_frac instrumentation already
   in place.
4. Complementary semantic levers if C alone is insufficient: integer LR
   schedule (gs3 → gs4 after the babble phase), blank-logit suppression
   early, larger batch.

**Artifacts:** checkpoints/cleanbase-d3-128-gs3/{metrics.jsonl,history.json,
best.pt,latest.pt} (committed), logs/cleanbase_d3_128_gs3.log.

---

## D-005: Launch rail-rescale arm (option C) — cleanbase-d3-128-gs3-rq

**Decision (user-approved, 2026-09-08):** Option C on top of the gs3 arm.
Identical config to cleanbase-d3-128-gs3 (scale-preserving init + grad_shift
3, D-001 clean-base config) plus value-preserving weight re-quantization on
rail: when a layer's at-rail code fraction ≥ 25% (checked every training
step, per layer), codes are halved (round-to-nearest) and weight_exp += 1 —
value-preserving within half of the new quantum, restoring code headroom at
1 bit of resolution per rescale. Single-variable attribution vs the gs3 arm.

**Instrumentation fix (found while unit-testing the rescale):** the at-rail
predicate used (>=127)|(<=-128), but int8_clip keeps weights in [-127, 127],
so the negative rail (-127) was never counted. The gs3 arm's recorded
w_rail_frac / rail_outward_frac undercounted by the negative half — proj's
"plateau at 50%" was actually ~100% of codes pinned on both rails. Fixed in
weight_update and grad_health (probe script already used -127). Historical
metrics not rewritten; gs3 comparisons in D-004 should be read with rails at
~2x the recorded fraction.

**Verification:**
- Unit: 30% synthetic rails → one rescale, rails → 0%, exp -9→-8, max value
  error 1 old LSB (half the new quantum), int8 dtype preserved.
- Smoke (5 steps + eval, real dev batch): loss descends, no spurious early
  rescales (init at 0% rails), rail_rescales in the record schema,
  integer-state report OK.

**What this run tests (D-004 hypothesis):** if diversity survives past step
~750 with rails managed, capacity loss is implicated as the blank-collapse
driver; if it still collapses, the residual is a classic CTC shortcut →
semantic levers (integer LR schedule, blank suppression, larger batch).

**Watch items:**
- rail_rescales per layer and cadence — steady accumulation means sustained
  outward pressure; watch weight_exps for unbounded value growth.
- uniq/blank/WER vs the gs3 arm's 212→20→2→1 collapse (steps 250–1000).
- Body pct_changed never reaching 0.
- Rescales shift weight_exp +1 → act exps shift +1; within-run comparisons only.

**Risks:** repeated rescales erode weight resolution 1 bit each; a
pathological pressure regime could rescale every few hundred steps. The
rail_rescales / weight_exps instrumentation makes the cadence visible per eval.

**Artifacts:** checkpoints/cleanbase-d3-128-gs3-rq/ (as produced),
logs/cleanbase_d3_128_gs3_rq.log.

---

## D-006: gs3-rq verdict — capacity hypothesis refuted; blank collapse is semantic

**Decision:** D-004's capacity-loss hypothesis is refuted by direct test: with
rails fully managed (never pinned), the run collapsed identically to gs3
(uniq 212→20→8→3→1 over steps 250–1000, WER 1.000 from step 500 on). Rail
management stays in the standard config — it prevents pinning at zero cost
when pressure is absent (head: 0 rescales) — but it is symptom treatment.
The next arm must target the all-blank collapse semantically; recommended:
blank-logit suppression (option A below). No run launched pending user
decision.

**Evidence (cleanbase-d3-128-gs3-rq, 3000 steps, no early stop):**

| step | loss | WER | uniq | rescales p/b0/b1/b2/h | weight_exp p/b0 |
|---|---|---|---|---|---|
| 250 | 10.2 | 2.42 | 212 | 0/0/0/0/0 | -9/-9 |
| 500 | 8.0 | 1.00 | 20 | 1/0/0/0/0 | -8/-9 |
| 750 | 8.4 | 1.00 | 8 | 10/7/1/0/0 | 1/-2 |
| 1000 | 8.0 | 1.00 | 3 | 26/35/5/1/0 | 17/26 |
| 1500 | 11.9 | 1.00 | 1 | 71/108/14/3/0 | 62/99 |
| 2000 | 9.6 | 1.00 | 1 | 119/189/23/5/0 | 110/180 |
| 3000 | 11.1 | 1.00 | 1 | **221/354/40/9/0** | **212/345** |

- Identical collapse timeline to gs3 despite zero pinning: rail fractions held
  ≤ ~19% throughout (gs3's proj was ~100% pinned by step 2500).
- Rescale cadence quantifies the pressure: blocks.0 averaged ~63 codes of
  outward drift per rescale cycle ≈ **7.4 codes/step net, sustained for
  3000 steps**. Too large for quantizer bias — psto_shift bias is a ±1-code
  effect on small accumulators; magnitude analysis rules out rounding bias
  as the driver. Body layers receive near-consistent-sign gradients through
  the entire all-blank phase.
- Head completely immune: 0 rescales, weight_exp −10 unchanged, final
  gap 3.35 codes / tied 1.7%.
- Structural insight: body weight VALUE growth (proj exp −9→+212, blocks.0
  −9→+345) is **forward-invariant** — act_calc renormalizes activations per
  layer and ReLU is positive-homogeneous, so per-layer weight scale cancels;
  the drift never appears in the loss (band 8–13, same as gs3's 7.5–14).
  The march is parametrization churn along the degenerate manifold, not a
  cause of the semantic state.

**Interpretation:** babble phase (~step 250) → all-blank shortcut discovered
(steps 300–750) → degenerate phase: the head sits at the shortcut optimum
(its grads ~zero), the body keeps receiving consistent-sign gradients, and
nothing pushes the model off the shortcut. Rail mechanics are no longer the
binding constraint.

**Next-run options (pending user decision):**
- **A (recommended): blank-logit suppression arm** — subtract an integer code
  offset from the head's blank-column logits in the training CTC path only
  (e.g. −8 codes for step <1000, −4 for <1500, 0 after; eval decodes with
  unbiased logits). Directly blocks the shortcut during the babble/alignment
  phase; integer-only, cheap, standard CTC anti-collapse practice.
- B: integer LR schedule (gs3 → gs5 after step ~500) — slows everything;
  does not discriminate shortcut vs alignment learning.
- C: larger batch — direction unclear (if the pressure is the true gradient,
  less noise may deepen the shortcut).

**Artifacts:** checkpoints/cleanbase-d3-128-gs3-rq/{metrics.jsonl,history.json,
best.pt,latest.pt} (committed), logs/cleanbase_d3_128_gs3_rq.log.

---

## D-007: Launch blank-suppression arm (option A) — cleanbase-d3-128-bs

**Decision (user-approved, 2026-09-08):** D-006 option A. Identical config to
gs3-rq (scale-preserving init + grad_shift 3 + rail-rescale 0.25 +
stop-on-head-collapse) plus training-time blank-logit suppression: an 8-code
offset subtracted from the head's blank-column logits inside the training
CTC loss (step < 1000), halved to 4 (< 1500), then 0. Eval decodes unbiased
logits. Default off (peak 0); the arm passes `--blank-suppress 8` explicitly.

**Mechanism notes:** the shift is constant, so the returned error signal is
the exact gradient of the shaped objective (chain rule). While active, the
all-blank path costs a factor e^-8 (~3e-4) in relative probability, so CTC
must route probability through real tokens during the babble/alignment
phase; the ramp-off hands the blank decision back to the model. The offset
in effect is recorded in every eval record (`blank_suppress`).

**Verification:** schedule unit-checked (8/4/0 at the right boundaries;
peak 0 disables); suppressed loss differs as expected (init +0.6 —
suppression bites once blank concentrates, not at near-uniform init);
3-step training smoke + unbiased eval path OK; integer-state report OK.

**What this run tests:** D-006's residual hypothesis — the collapse is the
CTC shortcut, not any remaining mechanical constraint. Success: uniq ≥ ~10
or WER < 1 through the step 250–1000 window and surviving ramp-off
(1500+); stretch: WER descending after the babble phase. Failure modes:
(a) collapse returns at ramp-off → longer hold or permanent low offset;
(b) babble persists without alignment (WER > 1, uniq high) → suppression
worked, alignment needs LR schedule/epochs; (c) no change → shortcut is
established before step 250 (unlikely given the babble window).

**Artifacts:** checkpoints/cleanbase-d3-128-bs/ (as produced),
logs/cleanbase_d3_128_bs.log.

---

## D-008: bs arm verdict — constant blank suppression is elastically compensated; mode (c)

**Decision:** D-007's blank-suppression arm failed on pre-registered mode (c)
(no change), with a sharper mechanism: the constant tax was *paid, not
obeyed*. The head pushed the raw blank column up ~20 codes (head gap 21.4 at
step 250 vs 3.2 in gs3) so blank won even the unbiased decode; the suppressed
loss tracked the tax amount exactly (13.7–35.9 at offset 8 → 8–9 at offset 4
→ standard 8–13 band at offset 0) with uniq 1 at *every* eval. The arm also
*prevented the babble phase* — the only diverse-text phenomenon the project
has produced (uniq 1 at step 250 vs 212 without suppression). Constant
suppression is strictly worse than none in this regime. Next decision
pending user (options below); no run launched.

**Evidence (cleanbase-d3-128-bs, 3000 steps, no early stop):**

| step | loss | offset | WER | uniq | blank | head gap | rescales p/b0 |
|---|---|---|---|---|---|---|---|
| 250 | 13.7 | 8 | 1.00 | 1 | 0.988 | 21.4 | 13/3 |
| 750 | 35.9 | 8 | 1.00 | 1 | 0.990 | 11.2 | 74/16 |
| 1000 | 9.0 | 4 | 1.00 | 1 | 0.981 | 11.7 | 105/22 |
| 1500 | 8.7 | 0 | 1.00 | 1 | 0.988 | 8.5 | 160/37 |
| 3000 | 9.7 | 0 | 1.00 | 1 | 0.987 | 4.3 | 305/114 |

- Loss = suppressed objective; its level moves with the offset, not with any
  semantic progress. best_wer 1.0 (no eval ever beat the collapsed baseline).
- Head stayed healthy (gap 4–21, tied ≤ 2%); rescale churn moderate.

**Interpretation:** a constant per-frame logit tax cannot change the
shortcut's status as the easiest descent direction — the gradient compensates
it within ~250 steps. The babble-phase suppression is the notable casualty:
early training under a blank tax skips straight to compensated blank.

**Where the evidence now points:** four consecutive arms (failed, gs3,
gs3-rq, bs) establish that no cheap lever tried so far prevents the all-blank
shortcut, and the one period of diverse text (babble, steps ~250 in gs3/gs3-rq)
never converted into alignment (WER < 1). The remaining explanations diverge:
(i) the shortcut is still preventable with a *compensation-proof* shaping
term, or (ii) this 210k-param MLP encoder at batch 4 / 1.6 epochs simply
cannot begin alignment, and the babble ceiling is a capacity/budget limit —
the whisper-base accuracy anchor is a ~74M-param attention model (~350×
larger).

**Next-run options (pending user decision):**
- **A: blank-cap arm (compensation-proof shaping, recommended first — cheap,
  ~2h).** Clamp the blank logit at `max(non-blank) + K` codes per frame
  (K≈2–4), applied identically in training and eval (the cap is a fixed
  integer op, NPU-trivial, and becomes part of the model definition). When
  blank over-dominates, its gradient is zero at the clamp — the tax cannot be
  paid — and the only descent is raising token logits. If diversity still
  dies under the cap, the shortcut is not the binding constraint and we
  stop spending arms on shaping.
- **B: capacity/budget arm.** Same config at 10k steps (~5 epochs) and/or
  dim 256, no shaping — tests whether alignment is a budget question.
  Costs ~6h+ per arm at current throughput.
- C: A first, then B if A fails (sequential attribution).

**Artifacts:** checkpoints/cleanbase-d3-128-bs/{metrics.jsonl,history.json,
best.pt,latest.pt} (committed), logs/cleanbase_d3_128_bs.log.

---

## D-009: Launch blank-cap arm (option A) — cleanbase-d3-128-bc

**Decision (user-approved, 2026-09-08):** D-008 option A. Identical config to
gs3-rq (scale-preserving init + grad_shift 3 + rail-rescale 0.25 +
stop-on-head-collapse; NO blank-suppress tax) plus a permanent blank cap:
the blank logit is clamped at `max(non-blank) + K` codes per frame, K=2,
applied identically in training and eval (implemented in
`IntCTCEncoder.forward` via `_cap_blank` — part of the model definition, an
NPU-trivial integer op). Single-variable attribution vs gs3-rq.

**Why the cap cannot be compensated (vs the bs tax):** the cap binds exactly
when blank tries to dominate; pushing the raw blank column higher is
invisible at the decision surface, so the head cannot buy the shortcut. The
softmax can never concentrate on blank (bounded within e^K ≈ 7.4× of the
best token per frame), so the all-blank path cannot accumulate probability.
Blank still legitimately wins frames (by up to K codes) — real CTC blank
usage remains representable, at the documented cost of handicapping true
silence frames for this arm. Trade-off accepted to maximize discrimination:
if diversity dies even under an unpayable cap, the shortcut is not the
binding constraint.

**Verification:**
- Unit: cap holds everywhere on synthetic logits (blank ≤ max_other+2,
  saturated frames sit exactly at the cap, non-blank columns untouched, low
  blank passes through, default off).
- Forward smoke through the real model; 4-step training smoke (loss
  descends); capped-eval smoke (at init blank never wins a frame);
  integer-state report OK.

**What this run tests:** D-008's fork (i) — whether a compensation-proof
shaping term holds diversity. Success: uniq ≥ ~10 or WER < 1 through the
step 250–1000 window AND past step 1500 (cap is permanent — no ramp-off
event). Failure mode: collapse under the cap → fork (ii) — capacity/budget
becomes the next arm (10k steps and/or dim 256), and we stop spending arms
on shaping.

**Watch items:**
- uniq/WER/blank_rate from the capped decode; head gap will read ≤ ~2 on
  blank-won frames by construction (head_separation semantics change under
  the cap — compare within-arm only).
- Body rescale churn (gs3-rq levels expected; head 0).
- Raw blank-row drift: with the cap saturated, the head's blank row can
  drift up without loss feedback — cosmetic for this arm (invisible at the
  capped surface), noted for any future fine-tuning from this checkpoint.

**Artifacts:** checkpoints/cleanbase-d3-128-bc/ (as produced),
logs/cleanbase_d3_128_bc.log.

---

## D-010: bc arm verdict — collapse under an unpayable cap; error-signal quantization is the fine-gradient bottleneck

**Decision:** D-009 failed on its pre-registered mode (collapse under the
cap): uniq 1 at every eval, WER 1.000, blank ~0.97 held at the cap margin all
run, head frozen (0% weight changes) from step 500. Per D-008's fork, shaping
arms stop. But the fork verdict is amended by a probe finding that changes
the recommended next arm: the int8 error signal discards ~99.9% of the
gradient at plateau magnitudes — `float_to_int8` carries the same
`clamp_min(1)` port artifact that D-002/D-003 found and fixed in
`weight_quant`. Recommended next arm: NITI-faithful error scaling (option A).
No run launched pending user decision.

**Evidence (cleanbase-d3-128-bc, 3000 steps, no early stop):**

| step | loss | WER | uniq | blank | head gap | rescales p/b0 |
|---|---|---|---|---|---|---|
| 250 | 29.5 | 1.00 | 1 | 0.969 | 1.9 | 30/30 |
| 1000 | 45.6 | 1.00 | 1 | 0.976 | 1.9 | 138/154 |
| 3000 | 46.7 | 1.00 | 1 | 0.966 | 1.9 | 427/490 |

- Cap held mechanically (gap pinned at ~K all run; head rescales 0) but the
  capped shortcut was still reachable: blank wins ~97% of frames at the cap
  margin; the loss parked at the (higher) capped-plateau band 29–51.
- **Error-signal probe** (step-750 checkpoint, real dev batch): float CTC
  gradient 88.3% nonzero, max 0.0131 → int8 error **0.08% nonzero, max code
  2** (err_exp −7) → head quantized grads 0.05% nonzero → **0 head weight
  codes changed per step**. The body churns only because grad_calc
  renormalizes the surviving scraps to full-range codes — magnitude without
  information.
- Root cause: `float_to_int8` (src/model/int_layers.py:113) derives the
  exponent from `ceil(log2(max.clamp_min(1)))`; for max < 1 the clamp forces
  exp −7, so codes = grad×2⁷ ≤ 2 for fine gradients. NITI's original
  `TiFloatToInt8` (src/niti/ti_torch.py:341) has no clamp: bitwidth from the
  true max (−6 for 0.013) → exp −13, codes = grad×2¹³ ≈ ±107 dense. Same
  port-artifact family as D-002's init bug.

**Interpretation:** every arm since D-003 made coarse moves fine (large
gradients quantize well) but was functionally deaf at refinement scale —
precisely the regime where the babble→alignment transition must happen. This
is the first integer-transport bottleneck that is real, *not intrinsic to
integer-only training*, and fixable in one function. The user's question
("is the integer-only constraint making it hard to learn?") now has a
precise answer: one quantizer bug did, and it is not NITI's design.

**Next-run options (pending user decision):**
- **A (recommended): NITI-faithful error scaling.** Fix `float_to_int8`'s
  exponent (true max-abs, no `clamp_min(1)`), everything else at the gs3-rq
  config (no cap, no tax) — single-variable attribution. Expected signature:
  head keeps updating past plateau; uniq survives / WER < 1 appears. Verify
  in smoke that err_exp (≈ −13, not −7) flows correctly through
  grad_exp/act_in_exp bookkeeping and that update alignment (grad_exp vs
  weight_exp) stays sane.
- B: float control twin (same arch/data/budget, float + Adam) — the
  integers-vs-budget discriminator; the follow-up if A fails.
- C: capacity/budget (dim 256 / 10k steps) — fallback if fine gradients
  alone don't unlock alignment.

**Artifacts:** checkpoints/cleanbase-d3-128-bc/{metrics.jsonl,history.json,
best.pt,latest.pt} (committed), logs/cleanbase_d3_128_bc.log.