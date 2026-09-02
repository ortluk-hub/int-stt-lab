# Phase 1 Implementation Plan
## Integer-Arithmetic CTC Speech-to-Text

**Start Time:** Tue Sep  1 10:11:10 PM MDT 2026

### Framework Selection
Based on research documentation:
- **NITI**: Selected for initial work - tested on ImageNet with 8-bit integers/5-bit gradients
- **NITRO-D**: Available for comparison - focuses on CNNs with novel scaling layers

### Phase 1 Objectives (from recommendations.md):
1. **Select integer-only training framework** → NITI cloned
2. **Start with minimal STT architecture** → Feed-forward network or small CNN on spectrogram patches
3. **Verify integer-only training claim** → Implement logging/assertions
4. **Investigate CTC loss in integer arithmetic** → Critical gap to address
5. **Explore NPU-compatible feature extraction** → mel filterbank investigation
6. **Use small falsifiable PoC** → Speech Commands subset with tiny model

### Immediate Next Steps:
1. Examine NITI structure and understand integer-only training mechanisms
2. Create minimal STT architecture using NITI framework
3. Implement verification for integer-only training
4. Research CTC loss implementation in integer arithmetic
5. Prepare for QNN deployment verification (already documented)

### Artifacts Location:
- Framework clones: /home/ortluk/ortluk-hub/int-stt-lab/phase1/
- Research documentation: /home/ortluk/research/
- Project README: /home/ortluk/ortluk-hub/int-stt-lab/README.md
