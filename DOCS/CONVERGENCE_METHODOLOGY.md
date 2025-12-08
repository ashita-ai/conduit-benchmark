# Convergence Detection Methodology

**Last Updated**: 2025-12-07

Technical documentation of the convergence detection algorithm used in conduit-benchmark for measuring when multi-armed bandit algorithms stabilize their model selection policies.

---

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Algorithm-Specific Parameter Extraction](#algorithm-specific-parameter-extraction)
4. [Convergence Detection Algorithm](#convergence-detection-algorithm)
5. [Special Cases](#special-cases)
6. [Configuration Parameters](#configuration-parameters)
7. [Implementation Details](#implementation-details)
8. [Validation and Interpretation](#validation-and-interpretation)

---

## Overview

Convergence detection determines when a bandit algorithm has learned a stable model selection policy. This is critical for:

1. **Benchmarking**: Comparing how quickly different algorithms learn
2. **Production Deployment**: Knowing when to trust routing decisions
3. **Sample Efficiency**: Measuring data efficiency across algorithms

### Key Insight

Rather than tracking noisy quality scores (which depend on question difficulty), we track **internal algorithm parameters** that represent the algorithm's learned beliefs about model quality.

---

## Mathematical Foundation

### Coefficient of Variation (CV)

Convergence is detected when the **Coefficient of Variation** of algorithm parameters drops below a threshold:

$$CV = \frac{\sigma}{\mu}$$

Where:
- $\sigma$ = standard deviation of parameter values in sliding window
- $\mu$ = mean of parameter values in sliding window

**Convergence criterion**: $CV < \theta$ (default $\theta = 0.10$, i.e., 10%)

### Why CV?

1. **Scale-invariant**: Works regardless of absolute parameter magnitudes
2. **Interpretable**: 10% CV means parameters vary within ±10% of mean
3. **Robust**: Insensitive to the specific parameter being tracked

### Sliding Window Approach

We compute CV over a sliding window to detect when parameters stabilize:

$$CV_t = \frac{\text{std}(P_{t-w:t})}{\text{mean}(P_{t-w:t})}$$

Where:
- $P_{t-w:t}$ = parameter values from time $t-w$ to $t$
- $w$ = window size (adaptive, default ~10% of dataset size)

---

## Algorithm-Specific Parameter Extraction

Different bandit algorithms maintain different internal state. We extract the most informative parameters for each:

### Thompson Sampling

**State tracked**: `arm_distributions` containing Beta distribution parameters per model

**Parameter extracted**: Mean expected reward across all arms

$$\bar{\mu}_t = \frac{1}{K} \sum_{k=1}^{K} \frac{\alpha_k}{\alpha_k + \beta_k}$$

Where:
- $K$ = number of models (arms)
- $\alpha_k, \beta_k$ = Beta distribution parameters for model $k$

**Rationale**: As Thompson Sampling learns, the mean expected rewards stabilize, reflecting confident beliefs about model quality.

### UCB1 (Upper Confidence Bound)

**State tracked**: `arm_mean_reward` containing empirical mean rewards per model

**Parameter extracted**: Mean reward across all arms

$$\bar{r}_t = \frac{1}{K} \sum_{k=1}^{K} \bar{r}_k$$

Where $\bar{r}_k$ is the empirical mean reward for model $k$.

**Rationale**: UCB1's mean rewards converge to true model quality as samples accumulate.

### Epsilon-Greedy

**State tracked**: `arm_mean_reward` (same structure as UCB1)

**Parameter extracted**: Mean reward across all arms (identical to UCB1)

**Rationale**: Epsilon-greedy maintains the same empirical mean estimates as UCB1.

### LinUCB (Contextual Linear UCB)

**State tracked**: `arm_success_rates` containing success rates per model

**Parameter extracted**: Mean success rate across all arms

$$\bar{s}_t = \frac{1}{K} \sum_{k=1}^{K} s_k$$

Where $s_k$ is the success rate for model $k$.

**Rationale**: LinUCB's linear regression parameters are high-dimensional; success rates provide a scalar summary of learned policy quality.

### Contextual Thompson Sampling

**State tracked**: `arm_distributions` (same as Thompson Sampling)

**Parameter extracted**: Mean expected reward (identical to Thompson Sampling)

### Hybrid Algorithms (Thompson+LinUCB, UCB1+LinUCB)

**State tracked**: Depends on which phase is active

**Parameter extracted**: Uses Thompson/UCB1 parameters during warmup, LinUCB parameters after switch

---

## Convergence Detection Algorithm

### Pseudocode

```python
def detect_convergence(algorithm_state_history, window, threshold, min_samples):
    """
    Detect when algorithm parameters have stabilized.

    Args:
        algorithm_state_history: List of internal state dicts from algorithm.get_stats()
        window: Sliding window size for CV calculation
        threshold: CV threshold for declaring convergence (default: 0.10)
        min_samples: Minimum observations before checking convergence

    Returns:
        ConvergenceMetrics with:
        - converged: bool
        - convergence_point: int (1-indexed query number)
        - coefficient_of_variation: float (final CV)
    """

    # Extract representative parameter from each state
    param_values = []
    for state in algorithm_state_history:
        if 'arm_distributions' in state:
            # Thompson Sampling: use mean rewards
            means = [d['mean'] for d in state['arm_distributions'].values()]
            param_values.append(mean(means))
        elif 'arm_mean_reward' in state:
            # UCB1, Epsilon-Greedy: use mean rewards
            param_values.append(mean(state['arm_mean_reward'].values()))
        elif 'arm_success_rates' in state:
            # LinUCB: use success rates
            param_values.append(mean(state['arm_success_rates'].values()))

    # Need sufficient data
    if len(param_values) < min_samples:
        return ConvergenceMetrics(converged=False, convergence_point=None)

    # Detect convergence point
    convergence_point = None
    for i in range(min_samples, len(param_values)):
        window_data = param_values[max(0, i-window):i]

        if len(window_data) >= window // 2:
            cv = std(window_data) / mean(window_data)

            if cv < threshold:
                convergence_point = i + 1  # 1-indexed
                break

    return ConvergenceMetrics(
        converged=(convergence_point is not None),
        convergence_point=convergence_point,
        coefficient_of_variation=final_cv
    )
```

### Step-by-Step Process

1. **Extract Parameters**: For each query $t$, extract the algorithm's internal state and compute a scalar summary (mean expected reward or success rate)

2. **Build Parameter History**: Accumulate $P = [p_1, p_2, ..., p_n]$ where $p_t$ is the scalar parameter at time $t$

3. **Check Minimum Samples**: Wait until we have at least `min_samples` observations (default: 10-15, adaptive to dataset size)

4. **Sliding Window CV**: For each time $t \geq \text{min\_samples}$:
   - Compute $CV_t = \text{std}(P_{t-w:t}) / \text{mean}(P_{t-w:t})$
   - If $CV_t < \theta$, declare convergence at query $t$

5. **Return Results**: Report convergence status, point, and final CV

---

## Special Cases

### Fixed Strategies

**Algorithms**: `always_best`, `always_cheapest`, `oracle`

**Behavior**: Converge immediately (convergence_point = 1)

**Rationale**: These algorithms don't learn; they always select the same model. Their "policy" is stable from the first query.

```python
if algorithm_name in ['always_best', 'always_cheapest', 'oracle']:
    return ConvergenceMetrics(
        converged=True,
        convergence_point=1,
        coefficient_of_variation=0.0
    )
```

### Random Baseline

**Algorithm**: `random`

**Behavior**: Never converges (converged = False, CV = infinity)

**Rationale**: Random selection has no learning; parameters don't stabilize. This provides a baseline for comparison.

```python
if algorithm_name == 'random':
    return ConvergenceMetrics(
        converged=False,
        convergence_point=None,
        coefficient_of_variation=float('inf')
    )
```

### Insufficient Data

**Condition**: `len(param_values) < min_samples`

**Behavior**: Report non-convergence

**Rationale**: Cannot reliably assess stability with too few observations.

---

## Configuration Parameters

### Default Values

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` | 0.10 | CV threshold (10% variation) |
| `window` | Adaptive | `min(20, max(10, 0.10 × dataset_size))` |
| `min_samples` | Adaptive | `min(15, max(10, 0.05 × dataset_size))` |

### Recommended Settings by Dataset Size

| Dataset Size | Window | Min Samples | Rationale |
|--------------|--------|-------------|-----------|
| 100 queries | 10 | 10 | Quick convergence detection |
| 500 queries | 20 | 15 | Balanced approach |
| 1000 queries | 20 | 15 | Standard benchmark |
| 5000+ queries | 20 | 15 | Capped for efficiency |

### Threshold Selection

- **0.05 (5%)**: Strict convergence, may under-report
- **0.10 (10%)**: Recommended default, balances sensitivity
- **0.15 (15%)**: Lenient, may over-report convergence

---

## Implementation Details

### Data Structure: Algorithm State History

Each query stores the algorithm's internal state in the feedback metadata:

```json
{
  "feedback": [
    {
      "query": "What is the capital of France?",
      "model": "gpt-5.1-2025-11-13",
      "quality_score": 1.0,
      "metadata": {
        "algorithm_state": {
          "arm_distributions": {
            "gpt-5.1-2025-11-13": {"alpha": 15.2, "beta": 2.1, "mean": 0.879},
            "claude-sonnet-4.5": {"alpha": 12.8, "beta": 3.4, "mean": 0.790},
            "gemini-2.5-pro": {"alpha": 10.1, "beta": 2.8, "mean": 0.783}
          },
          "total_queries": 42,
          "arm_counts": {
            "gpt-5.1-2025-11-13": 18,
            "claude-sonnet-4.5": 14,
            "gemini-2.5-pro": 10
          }
        }
      }
    }
  ]
}
```

### State Extraction Logic

```python
def extract_parameter(state: dict) -> float:
    """Extract scalar parameter from algorithm state."""

    # Thompson Sampling: arm_distributions with Beta params
    if 'arm_distributions' in state and state['arm_distributions']:
        means = [dist.get('mean', 0.0)
                 for dist in state['arm_distributions'].values()]
        return np.mean(means) if means else 0.0

    # UCB1, Epsilon-Greedy: arm_mean_reward
    if 'arm_mean_reward' in state and state['arm_mean_reward']:
        return np.mean(list(state['arm_mean_reward'].values()))

    # LinUCB: arm_success_rates
    if 'arm_success_rates' in state and state['arm_success_rates']:
        return np.mean(list(state['arm_success_rates'].values()))

    return 0.0
```

### Fallback: Quality-Based Convergence

When algorithm state history is unavailable (e.g., legacy data), we fall back to quality score-based detection using slope analysis:

```python
def detect_quality_convergence(quality_scores, window, threshold, min_samples):
    """Fallback convergence detection using quality scores."""

    # Smooth the curve
    smoothed = convolve(quality_scores, smoothing_kernel)

    for i in range(min_samples, len(smoothed)):
        segment = smoothed[max(0, i-window):i]

        # Calculate normalized slope
        slope = linear_regression_slope(segment)
        normalized_slope = abs(slope / mean(segment))

        if normalized_slope < threshold:
            return convergence_point = i + 1

    return None
```

---

## Validation and Interpretation

### Interpreting Convergence Results

| Converged | CV | Interpretation |
|-----------|-----|----------------|
| True | < 0.05 | Strong convergence, highly stable policy |
| True | 0.05-0.10 | Normal convergence, stable policy |
| True | 0.10-0.15 | Marginal convergence, some instability |
| False | > 0.10 | Not converged, still learning |
| False | > 0.50 | Highly unstable, may need more data |

### Convergence Speed Comparison

Lower convergence points indicate faster learning:

```
Algorithm Rankings (1000-query MMLU benchmark):
1. always_best:            1 query   (fixed strategy)
2. always_cheapest:        1 query   (fixed strategy)
3. dueling_bandit:        47 queries (fast learner)
4. thompson_sampling:     89 queries (efficient)
5. ucb1:                 124 queries (moderate)
6. epsilon_greedy:       156 queries (slower)
7. linucb:               312 queries (contextual, needs more data)
8. random:               Never      (no learning)
```

### Statistical Significance

Convergence points can be compared using:

1. **Bootstrap CI**: Resample and compute convergence point distribution
2. **Wilcoxon Test**: Compare convergence points across repeated runs
3. **Effect Size**: Cohen's d on convergence point distributions

---

## References

- **Multi-Armed Bandits**: Lattimore, T. & Szepesvári, C. (2020). *Bandit Algorithms*. Cambridge University Press.
- **Thompson Sampling**: Russo, D. et al. (2018). "A Tutorial on Thompson Sampling." *Foundations and Trends in Machine Learning*.
- **UCB1**: Auer, P. et al. (2002). "Finite-time Analysis of the Multiarmed Bandit Problem." *Machine Learning*.
- **LinUCB**: Li, L. et al. (2010). "A Contextual-Bandit Approach to Personalized News Article Recommendation." *WWW*.
- **Convergence Diagnostics**: Cowles, M.K. & Carlin, B.P. (1996). "Markov Chain Monte Carlo Convergence Diagnostics: A Comparative Review." *JASA*.

---

## Appendix A: Full Convergence Metrics Output

```json
{
  "convergence": {
    "converged": true,
    "convergence_point": 89,
    "coefficient_of_variation": 0.073,
    "window_size": 20,
    "threshold": 0.10
  }
}
```

## Appendix B: Code Reference

Implementation: `conduit_bench/analysis/metrics.py`

Key functions:
- `calculate_convergence()` (lines 103-165): Main entry point
- `_detect_parameter_convergence()` (lines 168-252): Parameter-based detection
- `_detect_quality_convergence()` (lines 255-310): Fallback quality-based detection
