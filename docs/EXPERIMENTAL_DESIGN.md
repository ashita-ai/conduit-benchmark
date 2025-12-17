# Experimental Design Rationale

**Last Updated**: 2025-12-07

Documentation of experimental design decisions and rationale for conduit-benchmark methodology.

---

## Table of Contents

1. [Overview](#overview)
2. [Sample Size Justification](#sample-size-justification)
3. [Algorithm Selection](#algorithm-selection)
4. [Model Selection](#model-selection)
5. [Dataset Selection](#dataset-selection)
6. [Evaluation Strategy](#evaluation-strategy)
7. [Reproducibility](#reproducibility)
8. [Limitations and Threats to Validity](#limitations-and-threats-to-validity)

---

## Overview

Our experimental design aims to answer:

> **Which multi-armed bandit algorithms best balance quality and cost when routing queries to LLMs?**

### Key Design Principles

1. **Comprehensive coverage**: Include algorithms from all major bandit families
2. **Fair comparison**: Same queries, same models, same evaluation for all algorithms
3. **Statistical rigor**: Sufficient sample size for reliable conclusions
4. **Reproducibility**: Fixed seeds, documented configuration

---

## Sample Size Justification

### Primary Sample Size: 1,000 Queries

We use 1,000 queries per dataset for the main benchmark.

### Statistical Power Analysis

**Goal**: Detect medium effect sizes (Cohen's d ≥ 0.5) with 80% power at α = 0.05.

**Required sample size** (two-sample t-test):
$$n = \frac{2(z_{\alpha/2} + z_{\beta})^2}{d^2} = \frac{2(1.96 + 0.84)^2}{0.5^2} \approx 63 \text{ per group}$$

With 1,000 queries, we can detect:
- **Small effects** (d = 0.2): Power > 99%
- **Medium effects** (d = 0.5): Power > 99%
- **Large effects** (d = 0.8): Power > 99%

### Convergence Requirements

For contextual bandits (LinUCB), convergence requires $O(d)$ to $O(10d)$ samples where $d$ = feature dimension.

| Configuration | Feature Dim | Min Samples | Our Sample Size | Sufficient? |
|---------------|-------------|-------------|-----------------|-------------|
| Raw OpenAI embeddings | 1538 | 1500-15000 | 1000 | Marginal |
| PCA (128 components) | 130 | 130-1300 | 1000 | Yes |
| Hybrid warm-start | 130 | ~500 | 1000 | Yes |

**Decision**: Use hybrid algorithms (warm-start with non-contextual) to ensure convergence within 1000 queries.

### Secondary Sample Sizes

| Purpose | Sample Size | Rationale |
|---------|-------------|-----------|
| Quick validation | 100 queries | Fast iteration, 5-10 minutes |
| Development testing | 10-50 queries | Unit tests, debugging |
| Full benchmark | 1000 queries | Statistical power, convergence |
| Extended analysis | 5000+ queries | Asymptotic behavior study |

---

## Algorithm Selection

### Selection Criteria

1. **Coverage**: At least one algorithm from each major family
2. **Practical relevance**: Algorithms used in production systems
3. **Theoretical interest**: Well-understood regret bounds
4. **Diversity**: Range of computational complexity

### Algorithm Taxonomy

```
Multi-Armed Bandits
├── Non-Contextual (ignore query features)
│   ├── Thompson Sampling (Bayesian)
│   ├── UCB1 (Frequentist/Optimistic)
│   └── Epsilon-Greedy (Simple heuristic)
│
├── Contextual (use query embeddings)
│   ├── LinUCB (Linear UCB)
│   ├── Contextual Thompson Sampling (Bayesian)
│   └── Dueling Bandit (Pairwise learning)
│
├── Hybrid (warm-start + contextual)
│   ├── Hybrid Thompson-LinUCB
│   └── Hybrid UCB1-LinUCB
│
└── Baselines
    ├── Random (lower bound)
    ├── Always-Best (quality upper bound)
    └── Always-Cheapest (cost upper bound)
```

### Why These 11 Algorithms?

| Algorithm | Family | Rationale |
|-----------|--------|-----------|
| **Thompson Sampling** | Non-contextual Bayesian | Optimal regret, simple, widely used |
| **UCB1** | Non-contextual Frequentist | Deterministic, proven bounds |
| **Epsilon-Greedy** | Non-contextual Heuristic | Simple baseline, common in practice |
| **LinUCB** | Contextual Frequentist | State-of-art for news recommendation |
| **Contextual Thompson** | Contextual Bayesian | Bayesian alternative to LinUCB |
| **Dueling Bandit** | Contextual Pairwise | Novel approach for LLM comparison |
| **Hybrid Thompson-LinUCB** | Hybrid | Fast start + contextual learning |
| **Hybrid UCB1-LinUCB** | Hybrid | Alternative hybrid strategy |
| **Random** | Baseline | Lower performance bound |
| **Always-Best** | Baseline | Quality upper bound |
| **Always-Cheapest** | Baseline | Cost upper bound |

### Excluded Algorithms (with rationale)

| Algorithm | Reason for Exclusion |
|-----------|---------------------|
| Oracle | 6x cost (runs all models), separate analysis |
| Neural bandits | Computational cost, limited theoretical guarantees |
| Adversarial bandits | Assumes adversarial environment, not applicable |
| Pure exploration | No exploitation, not practical |

---

## Model Selection

### Selection Criteria

1. **Provider diversity**: Multiple cloud providers
2. **Capability range**: From minimal to premium
3. **Cost range**: 100x cost difference between cheapest and most expensive
4. **Availability**: API access, reasonable rate limits

### Selected Models (8)

| Model | Provider | Tier | Rationale |
|-------|----------|------|-----------|
| claude-opus-4.5 | Anthropic | Premium | Highest quality benchmark |
| gpt-5.1 | OpenAI | Premium | Latest OpenAI flagship |
| gemini-3-pro | Google | Premium | Latest Google flagship |
| claude-sonnet-4.5 | Anthropic | Balanced | Best for code, popular |
| gemini-2.5-pro | Google | Balanced | Strong reasoning |
| gpt-5-mini | OpenAI | Efficient | Fast, cost-effective |
| claude-haiku-4.5 | Anthropic | Efficient | Budget Anthropic |
| gemini-2.5-flash | Google | Minimal | Fastest, cheapest |
| gpt-5-nano | OpenAI | Minimal | Smallest footprint |

### Coverage Analysis

| Dimension | Coverage |
|-----------|----------|
| Providers | 3 (Anthropic, OpenAI, Google) |
| Capability tiers | 4 (Premium, Balanced, Efficient, Minimal) |
| Cost range | 168x (opus vs nano) |
| Quality range | ~0.70 - 0.95 expected |

### Excluded Models (with rationale)

| Model | Reason for Exclusion |
|-------|---------------------|
| Open-source (Llama, Mistral) | Self-hosting complexity, variable costs |
| Specialized models | Task-specific, not general-purpose |
| Older versions | Superseded by included models |

---

## Dataset Selection

### Selection Criteria

1. **Task diversity**: Knowledge, reasoning, code generation
2. **Evaluation objectivity**: Clear ground truth
3. **Benchmark recognition**: Widely used, comparable to prior work
4. **Size**: Sufficient for statistical analysis

### Selected Datasets (3)

| Dataset | Domain | Size | Evaluator | Rationale |
|---------|--------|------|-----------|-----------|
| **MMLU** | Knowledge (57 subjects) | 14,042 | Exact match | Standard knowledge benchmark |
| **GSM8K** | Math reasoning | 1,319 | Exact match (`#### N`) | Objective evaluation, no LLM-as-judge |
| **HumanEval** | Code generation | 164 | Code execution | Objective correctness |

### Why These Datasets?

**MMLU (Massive Multitask Language Understanding)**:
- Covers 57 academic subjects
- Multiple choice format enables exact match evaluation
- Industry standard for LLM comparison
- Zero evaluation cost

**GSM8K (Grade School Math)**:
- Multi-step reasoning required
- Uses `#### N` format for objective exact match evaluation
- Tests math reasoning with clear right/wrong answers
- No LLM-as-judge needed (avoids circular dependency)

**HumanEval**:
- Executable test cases provide ground truth
- Standard code generation benchmark
- Binary correctness (pass/fail)
- Tests practical coding ability

### Dataset Subsetting

| Dataset | Full Size | Benchmark Size | Rationale |
|---------|-----------|----------------|-----------|
| MMLU | 14,042 | 1,000 | Statistical sufficiency, cost control |
| GSM8K | 1,319 | 1,000 | Near-complete coverage |
| HumanEval | 164 | 164 | Full coverage |

---

## Evaluation Strategy

### Evaluator Assignment

| Dataset | Evaluator | Rationale |
|---------|-----------|-----------|
| MMLU | Exact Match | Unambiguous A/B/C/D answers |
| GSM8K | Exact Match | Extract `#### N` format, objective evaluation |
| HumanEval | Code Execution | Objective test cases |

### Why Different Evaluators?

**Exact Match** works when:
- Answers are standardized (multiple choice)
- Correctness is binary
- No partial credit needed

**Arbiter (LLM-as-Judge)** needed when:
- Answer format varies
- Reasoning quality matters
- Partial credit is meaningful

**Code Execution** works when:
- Executable test cases exist
- Correctness is deterministic
- No subjective judgment needed

See [EVALUATION_METHODOLOGY.md](./EVALUATION_METHODOLOGY.md) for details.

---

## Reproducibility

### Random Seeds

```python
# Global seed for reproducibility
RANDOM_SEED = 42

# Applied to:
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
```

### Fixed Configuration

| Component | Fixed Value | Location |
|-----------|-------------|----------|
| Thompson α, β | 1.0, 1.0 | conduit.yaml |
| UCB1 c | √2 ≈ 1.414 | conduit.yaml |
| Epsilon initial | 0.1 | conduit.yaml |
| LinUCB α | 1.0 | conduit.yaml |
| Hybrid switch | 2000 queries | conduit.yaml |

### Query Order

- MMLU: HuggingFace dataset order (deterministic)
- GSM8K: HuggingFace dataset order (deterministic)
- HumanEval: Task ID order (deterministic)

### Versioning

| Component | Version |
|-----------|---------|
| Python | 3.11+ |
| LiteLLM | Latest |
| conduit | Git SHA recorded |
| conduit-benchmark | Git SHA recorded |

---

## Limitations and Threats to Validity

### Internal Validity

| Threat | Mitigation |
|--------|------------|
| Model API changes | Record timestamps, version info |
| Rate limiting | Sequential execution, retry logic |
| Evaluation variance (Arbiter) | Fixed judge model, temperature=0 |

### External Validity

| Threat | Mitigation |
|--------|------------|
| Dataset bias | Multiple diverse datasets |
| Model selection bias | Cover multiple providers/tiers |
| Time period effects | Document benchmark date |

### Construct Validity

| Threat | Mitigation |
|--------|------------|
| Quality metric validity | Multiple evaluators, ground truth |
| Cost metric validity | Use official API pricing |
| Convergence definition | Parameter-based detection |

### Statistical Conclusion Validity

| Threat | Mitigation |
|--------|------------|
| Low power | 1000 samples, power analysis |
| Multiple comparisons | Report effect sizes, not just p-values |
| Non-independence | Same queries for all algorithms |

---

## Summary

Our experimental design provides:

1. **11 algorithms** covering all major bandit families
2. **8 models** spanning 3 providers and 4 capability tiers
3. **3 datasets** testing knowledge, reasoning, and code
4. **1000 queries** ensuring statistical power and convergence
5. **Reproducible** configuration with fixed seeds

This enables fair, comprehensive comparison of LLM routing algorithms for practical deployment.
