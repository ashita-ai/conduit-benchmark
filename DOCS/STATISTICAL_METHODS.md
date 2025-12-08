# Statistical Analysis Methods

**Last Updated**: 2025-12-07

Documentation of statistical methods used for algorithm comparison and significance testing in conduit-benchmark.

---

## Table of Contents

1. [Overview](#overview)
2. [Friedman Test](#friedman-test)
3. [Effect Sizes](#effect-sizes)
4. [Bootstrap Confidence Intervals](#bootstrap-confidence-intervals)
5. [Pareto Frontier Analysis](#pareto-frontier-analysis)
6. [When to Use Each Test](#when-to-use-each-test)
7. [Implementation References](#implementation-references)

---

## Overview

Statistical analysis serves three purposes in our benchmark:

1. **Significance Testing**: Are differences between algorithms real or due to chance?
2. **Effect Size Estimation**: How large are the differences?
3. **Uncertainty Quantification**: What are the confidence bounds on our estimates?

### Statistical Framework

| Purpose | Method | Output |
|---------|--------|--------|
| Overall difference | Friedman Test | χ² statistic, p-value |
| Pairwise comparison | Cohen's d | Effect size, interpretation |
| Confidence bounds | Bootstrap CI | 95% interval |
| Trade-off analysis | Pareto Frontier | Optimal algorithm set |

---

## Friedman Test

### Purpose

Test whether there are **significant differences** in performance across multiple algorithms on the same dataset.

### When to Use

- Comparing 3+ algorithms
- Same queries evaluated by all algorithms
- Non-parametric (no normality assumption)

### Mathematical Formulation

Given $k$ algorithms evaluated on $n$ queries, the Friedman statistic is:

$$\chi^2_F = \frac{12n}{k(k+1)} \left[ \sum_{j=1}^{k} R_j^2 - \frac{k(k+1)^2}{4} \right]$$

Where:
- $R_j$ = sum of ranks for algorithm $j$ across all queries
- $k$ = number of algorithms
- $n$ = number of queries

### Interpretation

| p-value | Interpretation |
|---------|----------------|
| p < 0.001 | Highly significant difference |
| p < 0.01 | Very significant difference |
| p < 0.05 | Significant difference |
| p ≥ 0.05 | No significant difference |

### Implementation

```python
from scipy.stats import friedmanchisquare

def friedman_test(algorithm_results: dict[str, list[float]]) -> StatisticalTest:
    """Perform Friedman test for overall algorithm differences.

    Args:
        algorithm_results: Dict mapping algorithm_name -> quality scores

    Returns:
        StatisticalTest with statistic, p_value, significant (p < 0.05)
    """
    if len(algorithm_results) < 3:
        return StatisticalTest(
            test_name="Friedman",
            statistic=0.0,
            p_value=1.0,
            significant=False,
        )

    # Handle unequal sample sizes by truncating to minimum
    min_size = min(len(v) for v in algorithm_results.values())
    samples = [values[:min_size] for values in algorithm_results.values()]

    statistic, p_value = friedmanchisquare(*samples)

    return StatisticalTest(
        test_name="Friedman",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=bool(p_value < 0.05),
    )
```

### Example Output

```json
{
  "friedman": {
    "statistic": 847.32,
    "p_value": 1.2e-175,
    "significant": true
  }
}
```

**Interpretation**: With p < 0.001, we reject the null hypothesis that all algorithms perform equally.

---

## Effect Sizes

### Cohen's d

Measures the **magnitude** of difference between two groups, independent of sample size.

### Formula

$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{\text{pooled}}}$$

Where the pooled standard deviation is:

$$s_{\text{pooled}} = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}$$

### Interpretation Guidelines

| |d| | Interpretation | Practical Meaning |
|-----|----------------|------------------|
| < 0.2 | Negligible | No practical difference |
| 0.2 - 0.5 | Small | Minor difference, may not matter |
| 0.5 - 0.8 | Medium | Noticeable difference |
| > 0.8 | Large | Substantial difference |

### Implementation

```python
import numpy as np

def calculate_cohens_d(group1: list[float], group2: list[float]) -> EffectSizes:
    """Calculate Cohen's d effect size for two groups.

    Args:
        group1: Quality scores for algorithm 1
        group2: Quality scores for algorithm 2

    Returns:
        EffectSizes with cohens_d and interpretation
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(
        ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2)
    )

    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

    # Interpretation
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return EffectSizes(cohens_d=cohens_d, interpretation=interpretation)
```

### Example Output

```json
{
  "pairwise_effect_sizes": {
    "thompson_vs_ucb1": {
      "cohens_d": 0.23,
      "interpretation": "small"
    },
    "thompson_vs_random": {
      "cohens_d": 1.45,
      "interpretation": "large"
    },
    "linucb_vs_epsilon": {
      "cohens_d": 0.08,
      "interpretation": "negligible"
    }
  }
}
```

**Interpretation**: Thompson vs Random shows a large effect (d=1.45), meaning Thompson substantially outperforms Random.

---

## Bootstrap Confidence Intervals

### Purpose

Estimate the **uncertainty** in our quality and cost metrics without assuming a specific distribution.

### Method

1. Resample data with replacement $B$ times (default: 10,000)
2. Compute statistic (mean) for each resample
3. Take percentiles for confidence interval

### Formula

For 95% confidence interval:
- Lower bound: 2.5th percentile of bootstrap means
- Upper bound: 97.5th percentile of bootstrap means

### Implementation

```python
import numpy as np

def bootstrap_ci(
    data: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
) -> tuple[float, float]:
    """Calculate bootstrap confidence interval.

    Args:
        data: Sample data (e.g., quality scores)
        confidence: Confidence level (default: 0.95)
        n_bootstrap: Number of bootstrap samples

    Returns:
        (lower_bound, upper_bound)
    """
    if not data:
        return (0.0, 0.0)

    data_array = np.array(data)
    bootstrap_means = []

    rng = np.random.default_rng(42)  # Reproducible
    for _ in range(n_bootstrap):
        sample = rng.choice(data_array, size=len(data_array), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    return (float(lower), float(upper))
```

### Example Output

```json
{
  "thompson": {
    "average_quality": 0.912,
    "quality_ci": [0.897, 0.926]
  },
  "ucb1": {
    "average_quality": 0.905,
    "quality_ci": [0.889, 0.920]
  }
}
```

**Interpretation**: Thompson's true mean quality is between 0.897 and 0.926 with 95% confidence. Since the CIs overlap with UCB1, the difference may not be significant.

---

## Pareto Frontier Analysis

### Purpose

Identify algorithms that represent **optimal trade-offs** between cost and quality.

### Definition

An algorithm is **Pareto optimal** (non-dominated) if no other algorithm:
- Has lower cost AND equal or higher quality, OR
- Has equal or lower cost AND higher quality

### Visual Interpretation

```
Quality
  ^
  |     * A (Pareto optimal - highest quality)
  |   *   B (Pareto optimal)
  |  *      C (dominated by B)
  | *         D (Pareto optimal)
  |*            E (Pareto optimal - lowest cost)
  +----------------> Cost
```

Algorithms A, B, D, E are Pareto optimal; C is dominated.

### Implementation

```python
def identify_pareto_frontier(
    algorithms: dict[str, tuple[float, float]]  # name -> (cost, quality)
) -> list[str]:
    """Identify Pareto optimal algorithms.

    Args:
        algorithms: Dict mapping algorithm name to (cost, quality)

    Returns:
        List of Pareto optimal algorithm names
    """
    pareto_optimal = []

    for name1, (cost1, quality1) in algorithms.items():
        is_dominated = False

        for name2, (cost2, quality2) in algorithms.items():
            if name1 == name2:
                continue

            # Check if name2 dominates name1
            if (cost2 < cost1 and quality2 >= quality1) or \
               (cost2 <= cost1 and quality2 > quality1):
                is_dominated = True
                break

        if not is_dominated:
            pareto_optimal.append(name1)

    return pareto_optimal
```

### Example Output

```json
{
  "pareto_frontier": [
    "always_best",        // Highest quality
    "hybrid_thompson",    // Good balance
    "linucb",            // Cost-efficient
    "always_cheapest"    // Lowest cost
  ]
}
```

**Interpretation**: These 4 algorithms represent optimal trade-offs. Other algorithms are dominated (better alternatives exist).

---

## When to Use Each Test

### Decision Tree

```
Question: Are algorithms significantly different?
├── Yes (Friedman p < 0.05)
│   └── Question: How different is each pair?
│       └── Use Cohen's d for pairwise comparisons
│           ├── Large effect (|d| > 0.8): Meaningful difference
│           ├── Medium effect (0.5-0.8): Notable difference
│           ├── Small effect (0.2-0.5): Minor difference
│           └── Negligible (|d| < 0.2): No practical difference
│
└── No (Friedman p ≥ 0.05)
    └── Algorithms perform similarly
        └── Choose based on other factors (cost, latency)
```

### Summary Table

| Question | Test | Result |
|----------|------|--------|
| "Are algorithms different overall?" | Friedman Test | p-value |
| "How big is the difference?" | Cohen's d | Effect size |
| "What's our uncertainty?" | Bootstrap CI | 95% interval |
| "Which are best for cost/quality?" | Pareto Frontier | Optimal set |

---

## Implementation References

| Method | File | Function |
|--------|------|----------|
| Friedman Test | `conduit_bench/analysis/metrics.py` | `friedman_test()` |
| Cohen's d | `conduit_bench/analysis/metrics.py` | `calculate_cohens_d()` |
| Bootstrap CI | `conduit_bench/analysis/metrics.py` | `bootstrap_ci()` |
| Pareto Frontier | `conduit_bench/analysis/metrics.py` | `identify_pareto_frontier()` |
| Full Analysis | `conduit_bench/analysis/metrics.py` | `analyze_benchmark_results()` |

### Dependencies

```python
from scipy.stats import friedmanchisquare
import numpy as np
```

---

## Appendix A: Complete Analysis Output

```json
{
  "summary": {
    "num_algorithms": 11,
    "best_quality_algorithm": "hybrid_thompson_linucb",
    "best_cost_algorithm": "always_cheapest",
    "quality_rankings": ["hybrid_thompson_linucb", "thompson", "ucb1", ...],
    "cost_rankings": ["always_cheapest", "random", "epsilon", ...]
  },
  "statistical_tests": {
    "friedman": {
      "statistic": 847.32,
      "p_value": 1.2e-175,
      "significant": true
    },
    "pairwise_effect_sizes": {
      "thompson_vs_ucb1": {"cohens_d": 0.23, "interpretation": "small"},
      "thompson_vs_random": {"cohens_d": 1.45, "interpretation": "large"}
    }
  },
  "algorithms": {
    "thompson": {
      "average_quality": 0.912,
      "quality_ci": [0.897, 0.926],
      "total_cost": 1.847
    }
  },
  "pareto_frontier": ["always_best", "hybrid_thompson", "always_cheapest"]
}
```

---

## References

1. **Friedman Test**: Friedman, M. (1937). "The Use of Ranks to Avoid the Assumption of Normality Implicit in the Analysis of Variance." *JASA*.

2. **Cohen's d**: Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*. Lawrence Erlbaum Associates.

3. **Bootstrap Methods**: Efron, B. & Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.

4. **Pareto Optimality**: Multi-criteria decision analysis literature; originally from Pareto, V. (1896). *Cours d'économie politique*.
