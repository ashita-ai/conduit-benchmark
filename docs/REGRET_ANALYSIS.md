# Regret Analysis Methodology

**Last Updated**: 2025-12-07

Documentation of how regret is calculated and analyzed in conduit-benchmark.

---

## Table of Contents

1. [Overview](#overview)
2. [Regret Definitions](#regret-definitions)
3. [Oracle Baseline](#oracle-baseline)
4. [Regret Calculation](#regret-calculation)
5. [Theoretical Regret Bounds](#theoretical-regret-bounds)
6. [Regret Visualization](#regret-visualization)
7. [Practical Considerations](#practical-considerations)

---

## Overview

**Regret** measures the cost of learning - the gap between an algorithm's cumulative performance and the optimal performance achievable with perfect knowledge.

### Why Regret Matters

1. **Learning efficiency**: Lower regret = faster learning
2. **Sample complexity**: How many queries until near-optimal?
3. **Algorithm comparison**: Standardized performance metric
4. **Production impact**: Real cost of exploration

---

## Regret Definitions

### Instantaneous Regret

The regret at a single time step $t$:

$$r_t = r^* - r_t^{(a)}$$

Where:
- $r^*$: Optimal reward (best model for this query)
- $r_t^{(a)}$: Reward from selected arm $a$

### Cumulative Regret

Total regret accumulated over $T$ queries:

$$R_T = \sum_{t=1}^{T} r_t = \sum_{t=1}^{T} (r^* - r_t^{(a)})$$

This is the primary metric for algorithm comparison.

### Normalized Regret

Per-query average regret:

$$\bar{R}_T = \frac{R_T}{T} = \frac{1}{T} \sum_{t=1}^{T} (r^* - r_t^{(a)})$$

Useful for comparing runs of different lengths.

### Expected Regret

The expected cumulative regret under the algorithm's policy:

$$\mathbb{E}[R_T] = \sum_{t=1}^{T} \mathbb{E}[r^* - r_t^{(a)}]$$

Theoretical bounds are typically stated in terms of expected regret.

---

## Oracle Baseline

### Definition

The **Oracle** has perfect hindsight knowledge - it knows which model is best for each specific query before selection.

$$r^*_t = \max_{a \in \mathcal{A}} r_t^{(a)}$$

### Oracle Implementation

In our benchmark, Oracle achieves perfect knowledge by:

1. **Executing all $K$ models** for each query
2. **Recording all rewards** before selection
3. **Selecting the best** based on observed results

```python
# Oracle execution (simplified)
for query in queries:
    all_rewards = {}
    for model in models:
        response = execute(model, query)
        all_rewards[model] = evaluate(response)

    best_model = max(all_rewards, key=all_rewards.get)
    oracle_selection = best_model
```

### Oracle Properties

| Property | Value |
|----------|-------|
| Regret | 0 (by definition) |
| Cost | $K \times$ standard (runs all models) |
| Practical? | No (requires executing all models) |
| Use case | Theoretical upper bound |

### Why Oracle is Excluded from Default Benchmarks

- **6x cost increase** (8 models × each query)
- **Not achievable** in production
- **Separate analysis** when needed

---

## Regret Calculation

### In Practice: Quality-Based Regret

Since we don't run Oracle by default, we estimate regret using:

$$r_t = q^*_{\text{expected}} - q_t$$

Where:
- $q^*_{\text{expected}}$: Expected quality of best model (from `always_best` baseline)
- $q_t$: Actual quality achieved

### Cost-Adjusted Regret

For cost-aware analysis:

$$r_t^{\text{cost}} = \frac{q^*_{\text{expected}} - q_t}{c_t}$$

Where $c_t$ is the cost of the selected model.

### Implementation

```python
def calculate_regret(
    algorithm_results: dict,
    oracle_quality: float,  # or always_best average
) -> dict:
    """Calculate regret metrics for an algorithm run.

    Args:
        algorithm_results: Results from benchmark run
        oracle_quality: Reference quality (oracle or always_best)

    Returns:
        Dict with cumulative_regret, normalized_regret, regret_history
    """
    regret_history = []
    cumulative_regret = 0.0

    for query_result in algorithm_results['feedback']:
        instant_regret = oracle_quality - query_result['quality_score']
        instant_regret = max(0, instant_regret)  # Non-negative

        cumulative_regret += instant_regret
        regret_history.append(cumulative_regret)

    return {
        'cumulative_regret': cumulative_regret,
        'normalized_regret': cumulative_regret / len(regret_history),
        'regret_history': regret_history,
    }
```

---

## Theoretical Regret Bounds

### Non-Contextual Algorithms

| Algorithm | Expected Regret Bound | Notes |
|-----------|----------------------|-------|
| **Thompson Sampling** | $O(\sqrt{KT \log T})$ | Optimal for Bernoulli rewards |
| **UCB1** | $O(\sqrt{KT \log T})$ | Optimal, deterministic |
| **Epsilon-Greedy** | $O(K \log T / \Delta)$ | Suboptimal, $\Delta$ = gap |

### Contextual Algorithms

| Algorithm | Expected Regret Bound | Notes |
|-----------|----------------------|-------|
| **LinUCB** | $O(d\sqrt{T \log T})$ | $d$ = feature dimension |
| **Contextual Thompson** | $O(d\sqrt{T} \log T)$ | Bayesian optimal |

### Interpretation

- **$O(\sqrt{T})$**: Sublinear regret (learning works)
- **$O(T)$**: Linear regret (no better than random)
- **$O(\log T)$**: Logarithmic regret (optimal gap-dependent)

### Regret vs. Sample Size

For $T = 1000$ queries with $K = 8$ models:

| Algorithm | Theoretical Regret | Normalized |
|-----------|-------------------|------------|
| Thompson | $\sqrt{8 \times 1000 \times 7} \approx 237$ | 0.237 |
| UCB1 | $\sqrt{8 \times 1000 \times 7} \approx 237$ | 0.237 |
| Random | $1000 \times 0.15 \approx 150$ | 0.150 |

*Note: These are order-of-magnitude estimates, not exact predictions.*

---

## Regret Visualization

### Cumulative Regret Curve

```
Cumulative
Regret
    ^
    |                    ______ Random (linear)
    |               ____/
    |          ____/
    |     ____/____--------- UCB1 (sublinear)
    |____/----
    |___---------- Thompson (sublinear)
    +--------------------------> Queries (t)
```

**Interpretation**:
- **Flattening curve**: Algorithm converging (exploitation phase)
- **Linear curve**: Constant regret per query (no learning)
- **Early steep, later flat**: Exploration→exploitation transition

### Regret Per Query Over Time

```
Instant
Regret
    ^
    |****
    | ***
    |  ****
    |    ****
    |      *****
    |          ********
    |                  ************
    +---------------------------------> Queries (t)
```

**Interpretation**: Decreasing instant regret indicates learning.

---

## Practical Considerations

### Why We Don't Always Use Oracle

1. **Cost**: 6-8x more expensive per query
2. **Time**: Parallel execution still takes longer
3. **Comparison**: `always_best` provides similar reference

### Proxy for Oracle Quality

We use `always_best` baseline as a practical proxy:

| Metric | Oracle | Always-Best |
|--------|--------|-------------|
| Definition | Best model per query | Best model on average |
| Quality | Optimal | Near-optimal |
| Cost | $K \times$ normal | $1 \times$ (premium model) |
| Practical | No | Yes |

### Regret Limitations

1. **Stochastic rewards**: Same query can have different optimal models
2. **Non-stationarity**: Model quality may drift over time
3. **Missing oracle**: Without Oracle, regret is estimated

### When to Use Full Oracle Analysis

- Academic papers requiring theoretical bounds
- Small-scale experiments (< 100 queries)
- Validating algorithm correctness

---

## Summary

| Metric | Formula | Use Case |
|--------|---------|----------|
| Instant regret | $r^* - r_t$ | Per-query analysis |
| Cumulative regret | $\sum_t (r^* - r_t)$ | Algorithm comparison |
| Normalized regret | $R_T / T$ | Cross-run comparison |
| Cost-adjusted | $(r^* - r_t) / c_t$ | Cost-aware analysis |

**Key insight**: Sublinear regret ($o(T)$) proves the algorithm is learning. Our learning algorithms (Thompson, UCB1, LinUCB) achieve this, while Random has linear regret.
