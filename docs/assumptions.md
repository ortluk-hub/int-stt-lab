# Assumptions Log — 2026-09-07 (autonomous session)

Assumptions I made while you were away, for review on return. Ordered by impact.

## High impact
1. **"Full precision Whisper" baseline = fp32 ggml tiny/base on-device CPU.** I benchmarked
   ggml-tiny and ggml-base fp32 via whisper.cpp on the S21 Ultra. If your production reference is
   `small` (or a specific whisper.cpp fork/flag set), the baseline numbers need re-measuring.
2. **Raw int8 head outputs are used directly as CTC logits** (exponent dropped). Reason: the
   NITI exponent chain accumulates +13 at the head, making softmax degenerate and diverging
   training. Consistent with the framework (UpdateWeight renormalizes grads via grad_calc),
   but it does decouple logit scale from the exponent bookkeeping.
3. **Blank-collapse intervention threshold at step ~1000.** If dev WER is still exactly 1.000
   there, I pause the run, instrument non-blank decode rate, and add an integer gradient
   downshift (extra right-shift in grad_calc) as an integer LR. Evidence for the threshold:
   loss fell 350->~15 by step 150 and stayed flat-ish; no non-blank decode by 600 steps.

## Medium impact
4. **Benchmark test slice = first 64 test-manifest utterances with 1s<=dur<=15s**, normalized
   lower/stripped transcripts on both sides. Fixed slice reused for all conditions (clean/noise)
   so WER numbers are comparable.
5. **Noise robustness is now a first-class requirement** (per "accuracy falls off sharply in
   noisy environments"). Implies: benchmark measures WER-vs-SNR (20/10/0 dB), and the integer
   training pipeline gets MUSAN waveform augmentation.
6. **Wireless adb is the device's stable transport** (USB dropped under load twice, two cords,
   phone not enumerating at the end). Phone on Wi-Fi at 192.168.12.167; adb wireless debugging.

## Low impact / housekeeping
7. `docs/RESEARCH_ROADMAP.md` was context bleed from another repo (your statement) — removed
   from git (commit c820d0e) rather than restored.
8. `IntLayerScale`-style float params and float softmax in `IntMultiheadAttention` make the
   Conformer block ineligible for the integer-only claim; Step 2 uses TiLinear/TiReLU only.
9. Build caps: whisper.cpp Android build ran with -j2 to protect the training run.
10. Device files live in /data/local/tmp/whisper/ (includes my pushed libomp.so from NDK 26.1).
