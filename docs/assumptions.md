# Assumptions Log — 2026-09-07 (autonomous session)

Assumptions I made while you were away, for review on return. Ordered by impact.

## Ablation findings (depth-6, dim-128, 60-step two-arm smoke — both committed, `8835c40`)
- Control (no exponent mgmt): NOT blank-collapsed at step 60 (blank 0.17, babbling one token);
  head exp = 30 and climbing — rails predicted later. Run-1's instant collapse may be
  **width-coupled (dim 256), not depth alone**. At dim 128 the failure is slower.
- Treated (recenter targets [6,9,13,13,13,13] + branch 2^-1): exponent management works
  mechanically (head exp pinned to 15, sat 0%) but the arm sits in the blank attractor
  (blank 0.68, unique tokens 1). Blank escape did not happen by step 60 at depth 6,
  vs step ~40 for healthy depth 3.
- Open attribution question: is the treated arm's blank regression from branch_shift (halved
  residual signal) or from recentering (value division)? Needs recenter-only arm.

## High impact
1. **"Full precision Whisper" baseline = fp32 ggml tiny/base on-device CPU.** Clean-slice
   whisper-base fp32: **WER 24.6%, CER 10.0%, internal RTF 0.89** (64 fixed CV test utterances,
   idle phone, 4 threads). If your production reference is `small` or different flags,
   re-measure.
2. **Raw int8 head outputs are used directly as CTC logits** (exponent dropped). Reason: the
   NITI exponent chain accumulates +13 at the head, making softmax degenerate and diverging
   training. Consistent with the framework (UpdateWeight renormalizes grads via grad_calc),
   but it does decouple logit scale from the exponent bookkeeping.
3. **Blank-collapse attribution moved**: run-1 failure (3000 steps, WER 1.0) is scale/rail
   driven per instrumentation, and is width-coupled — depth-6 at dim 128 escapes blank
   initially. The "integer LR" knob (grad-shift) exists but has not yet been shown necessary.

## Medium impact
4. **Benchmark test slice = first 64 test-manifest utterances with 1s<=dur<=15s**, normalized
   lower/stripped transcripts on both sides. Fixed slice reused for all conditions; slice
   identity hash recorded in every JSONL record.
5. **Noise robustness is a first-class requirement** (per "accuracy falls off sharply in
   noisy environments"). MUSAN extracted to data/musan; SNR-conditioned benchmarks in flight.
6. **Wireless adb is the device's stable transport** (USB dropped under load twice, two cords,
   phone not enumerating at the end). Phone on Wi-Fi at 192.168.12.167:36735.
7. **Ablation pass criteria** (explicit, in run_depth_ablation.py): head tied_frac < 0.5,
   act sat < 5%, blank < 0.95, repeat < 0.9, uniq >= 10, head exp <= 20. These thresholds are
   my choices — review.

## Low impact / housekeeping
8. `docs/RESEARCH_ROADMAP.md` was context bleed from another repo (your statement) — removed
   from git (commit c820d0e) rather than restored.
9. `IntLayerScale`-style float params and float softmax in `IntMultiheadAttention` make the
   Conformer block ineligible for the integer-only claim; current encoder uses TiLinear/TiReLU
   only.
10. Device files live in /data/local/tmp/whisper/ (my pushed whisper-cli, libomp.so, models).
11. whisper.cpp Android build at ~/ortluk-hub/whisper.cpp/build-android (NDK 26.1, -j2).
12. Instrumentation record schema is in src/train/instrumentation.py; one JSONL line per
    checkpoint at checkpoints/<run>/metrics.jsonl.
