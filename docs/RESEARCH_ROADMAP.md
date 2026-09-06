# Phase 3 Research Roadmap: Statistical Rigor, Causality, and Practical Utility

## Preamble
### Statistical Rigor
To ensure scientific validity, all success criteria in this roadmap use **statistically defensible thresholds** for effect size, confidence intervals, and reproducibility. This replaces binary gates (e.g., "above zero") with continuous, rigorous metrics.

#### Key Definitions
- **Effect size**: Cohen’s *d* (standardized mean difference) or Hedges’ *g* (bias-corrected *d*).
  - *d* ≥ 0.2: Small effect.
  - *d* ≥ 0.5: Medium effect.
  - *d* ≥ 0.8: Large effect.
- **Confidence intervals (CI)**: 95% CI for all metrics (e.g., accuracy, WER, consistency).
- **Reproducibility**: 3 independent runs with **consistent results** (CI overlap).
- **Baseline comparisons**: Explicit comparisons against:
  - Random baselines (e.g., random vectors, shuffled candidates).
  - Encoder-only baselines (frozen encoder + linear layer).
  - Prior work (e.g., state-of-the-art token models).

#### Requirements for All Stages
1. **Effect size**: All success criteria must specify a minimum *d* (e.g., *d* ≥ 0.5).
2. **Confidence intervals**: Report 95% CI for all metrics.
3. **Reproducibility**: 3 independent runs with **overlapping CIs**.
4. **Baseline comparisons**: Compare against at least **one random baseline** and **one encoder-only baseline**.

---

## Stage 1: Temporal Reasoning Benchmark
**Claim**: The LCC achieves **above-chance performance** on temporal reasoning tasks (e.g., "before/after" classification).

### Success Criteria (Updated)
- **Effect size**: LCC must outperform **random baseline** by *d* ≥ 0.5.
- **Confidence intervals**: 95% CI for accuracy must **not include chance level** (e.g., 50% for binary tasks).
- **Reproducibility**: 3 independent runs with **overlapping CIs**.
- **Baseline comparisons**: Compare against:
  - Random baseline (e.g., random vectors).
  - Encoder-only baseline (frozen encoder + linear layer).

### Experiments
- Train and evaluate the LCC on temporal reasoning datasets (e.g., "before/after" classification).
- Compare against random and encoder-only baselines.

---

## Stage 2: Consistency Across Domains
**Claim**: The LCC achieves **consistent performance** across diverse domains (e.g., temporal reasoning, spatial reasoning).

### Success Criteria (Updated)
- **Effect size**: LCC must outperform **encoder-only baseline** by *d* ≥ 0.5 in **all domains**.
- **Confidence intervals**: 95% CI for consistency (e.g., standard deviation of accuracy across domains) must be **≤ 0.1**.
- **Reproducibility**: 3 independent runs with **overlapping CIs**.
- **Baseline comparisons**: Compare against:
  - Encoder-only baseline.
  - Prior work (e.g., token models).

### Experiments
- Evaluate the LCC on **multiple domains** (e.g., temporal reasoning, spatial reasoning).
- Compare against encoder-only baseline and prior work.

---

## Stage 3: Parameter Efficiency
**Claim**: The LCC achieves **competitive performance** with **fewer parameters** than token models.

### Success Criteria (Updated)
- **Effect size**: LCC must outperform **token models** by *d* ≥ 0.3 with **≤ 50% of the parameters**.
- **Confidence intervals**: 95% CI for accuracy must **overlap with token models**.
- **Reproducibility**: 3 independent runs with **overlapping CIs**.
- **Baseline comparisons**: Compare against:
  - Token models (e.g., fine-tuned BERT).
  - Encoder-only baseline.

### Experiments
- Train the LCC and token models on the same dataset.
- Compare parameter efficiency and performance.

---

## Stage 4: Parameter Efficiency (Updated with Causality)
**Claim**: The LCC’s **architecture** (not just parameter count) drives performance.

### Success Criteria (Updated)
- **Effect size**: LCC must outperform **linear layer ablation** by *d* ≥ 0.5.
- **Confidence intervals**: 95% CI for accuracy must **not overlap with linear layer ablation**.
- **Reproducibility**: 3 independent runs with **overlapping CIs**.
- **Baseline comparisons**: Compare against:
  - Linear layer ablation (same parameter count as LCC).
  - Encoder-only baseline.

### Experiments
1. **Ablation study**: Replace the LCC with a **linear layer** (same parameter count).
2. **Causality test**: If performance drops, it suggests the LCC’s **architecture** is causal.

---

## Stage 5: Falsification (Updated)
**Claim**: The LCC’s performance is due to **reasoning**, not artefacts (e.g., candidate geometry, pooling).

### Success Criteria (Updated)
- **Effect size**: LCC must outperform all baselines by *d* ≥ 0.5:
  - Random vectors.
  - Shuffled candidates.
  - Pooling ablation (mean/max pool).
- **Confidence intervals**: 95% CI for accuracy must **not overlap with any baseline**.
- **Reproducibility**: 3 independent runs with **overlapping CIs**.
- **Baseline comparisons**: Compare against:
  - Random vectors (same dimensionality as embeddings).
  - Shuffled candidates (order shuffled within each query).
  - Pooling ablation (mean/max pool).
  - Encoder-only baseline.

### Experiments
1. **Random vectors**: Replace all candidates with **random vectors**. If performance drops, it suggests **candidate geometry** is driving results.
2. **Shuffled candidates**: Shuffle the order of candidates **within each query**. If performance drops, it suggests **order sensitivity** (a hallmark of reasoning).
3. **Pooling ablation**: Replace the LCC’s **pooling layer** with a **mean/max pool**. If performance drops, it suggests **pooling artefacts** are driving results.
4. **Encoder leakage test**: Freeze the LCC and **fine-tune the encoder**. If performance improves, it suggests **encoder leakage** is driving results.

### Causality Test
- **Encoder leakage**: Fine-tuning the encoder must **not improve performance by *d* ≥ 0.2**.

---

## Stage 6: Transfer Learning (New)
**Claim**: The LCC generalizes to **new domains** (e.g., math, coding, multilingual tasks).

### Success Criteria
- **Effect size**: LCC must outperform **domain-specific baselines** by *d* ≥ 0.3.
- **Confidence intervals**: 95% CI for accuracy must **not overlap with domain-specific baselines**.
- **Reproducibility**: 3 independent runs with **overlapping CIs**.
- **Baseline comparisons**: Compare against:
  - Domain-specific baselines (e.g., fine-tuned token models).
  - Encoder-only baseline.

### Experiments
1. **Domain transfer**: Train the LCC on **one domain** (e.g., temporal reasoning) and test on **another** (e.g., math word problems).
2. **Baseline comparison**: Compare against **domain-specific baselines** (e.g., fine-tuned token models).
3. **Retention test**: Verify the LCC **retains performance** on the original domain.

---

## Stage 7: Practical Utility (New)
**Claim**: The LCC is **deployable in real-world applications** (e.g., edge devices, low-latency systems).

### Success Criteria
- **Latency**: Inference latency must be **≤ 100ms** on edge devices (e.g., Snapdragon 888).
- **Memory**: Memory usage must be **≤ 1GB** on edge devices.
- **Stability**: Prototype must **function without crashes** for **24 hours**.

### Experiments
1. **Latency benchmark**: Measure inference latency on **edge devices** (e.g., Snapdragon 888).
2. **Memory benchmark**: Measure memory usage on **edge devices**.
3. **Deployment test**: Deploy the LCC in a **real-world prototype** (e.g., voice assistant, chatbot).

---

## Validation
### Self-Review Checklist
1. **Statistical rigor**: All success criteria use **effect size, CI, and reproducibility thresholds**.
2. **Falsifiability**: Stage 5 includes **random vectors, shuffled candidates, and pooling ablation tests**.
3. **Causality**: Stages 4 and 5 include **ablation studies and encoder leakage tests**.
4. **Transfer and utility**: Stages 6 and 7 address **transfer learning and practical deployment**.
5. **Clarity**: The roadmap is **self-contained** and **accessible to newcomers**.