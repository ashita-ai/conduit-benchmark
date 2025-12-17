# Cost Model and Pricing Methodology

**Last Updated**: 2025-12-07

Documentation of how costs are calculated, normalized, and compared across models in conduit-benchmark.

---

## Table of Contents

1. [Overview](#overview)
2. [Pricing Data Source](#pricing-data-source)
3. [Cost Calculation](#cost-calculation)
4. [Model Pricing Table](#model-pricing-table)
5. [Cost Normalization](#cost-normalization)
6. [Cost Efficiency Metrics](#cost-efficiency-metrics)
7. [Budget-Based Filtering](#budget-based-filtering)
8. [Implementation References](#implementation-references)

---

## Overview

Cost tracking is essential for evaluating LLM routing algorithms. The goal is to find algorithms that achieve high quality at low cost - the **cost-quality Pareto frontier**.

### Key Principles

1. **Actual costs**: Use real per-token pricing from LiteLLM's database
2. **Per-query tracking**: Record cost for each individual query
3. **Cumulative analysis**: Track cost accumulation over time
4. **Normalization**: Enable fair comparison across different run lengths

---

## Pricing Data Source

### Primary Source: LiteLLM

Pricing is retrieved from **LiteLLM's bundled `model_cost` database**, which is updated with each LiteLLM release.

```python
import litellm

def get_model_pricing(model_id: str) -> ModelPricing | None:
    """Get pricing from LiteLLM's bundled database."""
    model_info = litellm.model_cost.get(model_id)

    if model_info is None:
        return None

    return ModelPricing(
        model_id=model_id,
        input_cost_per_million=model_info["input_cost_per_token"] * 1_000_000,
        output_cost_per_million=model_info["output_cost_per_token"] * 1_000_000,
        source="litellm",
    )
```

### Fallback Pricing

When a model is not found in LiteLLM's database:

| Fallback | Input Cost | Output Cost |
|----------|------------|-------------|
| Conservative (GPT-4 tier) | $10.00 / 1M tokens | $30.00 / 1M tokens |

---

## Cost Calculation

### Formula

For each model execution:

$$\text{Cost} = (n_{\text{input}} \times c_{\text{input}}) + (n_{\text{output}} \times c_{\text{output}})$$

Where:
- $n_{\text{input}}$: Number of input tokens (prompt + query)
- $n_{\text{output}}$: Number of output tokens (model response)
- $c_{\text{input}}$: Cost per input token ($/token)
- $c_{\text{output}}$: Cost per output token ($/token)

### With Caching Support

Some providers (Anthropic, OpenAI) support prompt caching:

$$\text{Cost} = (n_{\text{input}} \times c_{\text{input}}) + (n_{\text{output}} \times c_{\text{output}}) + (n_{\text{cache\_read}} \times c_{\text{cache\_read}}) + (n_{\text{cache\_create}} \times c_{\text{cache\_create}})$$

### Implementation

```python
def compute_cost(
    input_tokens: int,
    output_tokens: int,
    model_id: str,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float:
    """Compute total cost for a model call in USD."""

    pricing = get_model_pricing(model_id)

    if pricing is None:
        # Conservative fallback
        return (input_tokens * 10.0 / 1_000_000
                + output_tokens * 30.0 / 1_000_000)

    cost = (input_tokens * pricing.input_cost_per_token
            + output_tokens * pricing.output_cost_per_token)

    # Add cache costs if applicable
    if cache_read_tokens > 0 and pricing.cached_input_cost_per_token:
        cost += cache_read_tokens * pricing.cached_input_cost_per_token

    if cache_creation_tokens > 0 and pricing.cache_creation_cost_per_million:
        cost += cache_creation_tokens * pricing.cache_creation_cost_per_million / 1_000_000

    return cost
```

---

## Model Pricing Table

Current pricing for benchmark models (as of December 2025):

### Tier 1: Premium Models

| Model | Provider | Input ($/1M) | Output ($/1M) | Notes |
|-------|----------|--------------|---------------|-------|
| claude-opus-4-5-20251101 | Anthropic | $15.00 | $75.00 | Highest quality |
| gemini-3-pro-preview | Google | $7.00 | $21.00 | Latest Gemini |
| gpt-5.1-2025-11-13 | OpenAI | $5.00 | $15.00 | GPT-5.1 |

### Tier 2: Balanced Models

| Model | Provider | Input ($/1M) | Output ($/1M) | Notes |
|-------|----------|--------------|---------------|-------|
| claude-sonnet-4-5-20250929 | Anthropic | $3.00 | $15.00 | Best code |
| gemini-2.5-pro | Google | $3.50 | $10.50 | Strong reasoning |

### Tier 3: Cost-Effective Models

| Model | Provider | Input ($/1M) | Output ($/1M) | Notes |
|-------|----------|--------------|---------------|-------|
| gpt-5-mini-2025-08-07 | OpenAI | $0.40 | $1.60 | Fast, cheap |
| claude-haiku-4-5-20251001 | Anthropic | $0.80 | $4.00 | Budget Anthropic |
| gemini-2.5-flash | Google | $0.15 | $0.60 | Fastest |

### Tier 4: Minimal Models

| Model | Provider | Input ($/1M) | Output ($/1M) | Notes |
|-------|----------|--------------|---------------|-------|
| gpt-5-nano-2025-08-07 | OpenAI | $0.10 | $0.40 | Smallest |

### Cost Ratio Analysis

Relative to cheapest model (gpt-5-nano):

| Model | Input Ratio | Output Ratio | Overall Ratio |
|-------|-------------|--------------|---------------|
| gpt-5-nano | 1.0x | 1.0x | 1.0x |
| gemini-2.5-flash | 1.5x | 1.5x | 1.5x |
| gpt-5-mini | 4.0x | 4.0x | 4.0x |
| claude-haiku-4.5 | 8.0x | 10.0x | 9.0x |
| claude-sonnet-4.5 | 30.0x | 37.5x | 33.8x |
| gpt-5.1 | 50.0x | 37.5x | 43.8x |
| gemini-3-pro | 70.0x | 52.5x | 61.3x |
| claude-opus-4.5 | 150.0x | 187.5x | 168.8x |

---

## Cost Normalization

### Per-Query Normalization

To compare algorithms across different run lengths:

$$\text{Cost per query} = \frac{\text{Total cost}}{\text{Number of queries}}$$

### Cumulative Cost Tracking

For convergence analysis, we track cumulative cost at each step:

$$C_t = \sum_{i=1}^{t} c_i$$

Where $c_i$ is the cost of query $i$.

### Implementation

```python
@dataclass
class AlgorithmMetrics:
    """Cost metrics for algorithm evaluation."""

    total_cost: float           # Sum of all query costs
    cumulative_cost: float      # Final cumulative value
    normalized_cost: float      # total_cost / total_queries
    cost_per_query: float       # Same as normalized_cost
```

---

## Cost Efficiency Metrics

### Quality-Cost Ratio

$$\text{Efficiency} = \frac{\text{Average Quality}}{\text{Cost per Query}}$$

Higher is better (more quality per dollar).

### Pareto Frontier

An algorithm is **Pareto optimal** if no other algorithm achieves:
- Lower cost AND equal or higher quality, OR
- Equal or lower cost AND higher quality

```python
def identify_pareto_frontier(
    algorithms: dict[str, tuple[float, float]]  # name -> (cost, quality)
) -> list[str]:
    """Find Pareto optimal algorithms."""

    pareto_optimal = []

    for name1, (cost1, quality1) in algorithms.items():
        dominated = False

        for name2, (cost2, quality2) in algorithms.items():
            if name1 == name2:
                continue

            # Check if name2 dominates name1
            if (cost2 < cost1 and quality2 >= quality1) or \
               (cost2 <= cost1 and quality2 > quality1):
                dominated = True
                break

        if not dominated:
            pareto_optimal.append(name1)

    return pareto_optimal
```

### Example Pareto Analysis

| Algorithm | Avg Quality | Total Cost | Pareto Optimal? |
|-----------|-------------|------------|-----------------|
| always_best | 0.95 | $2.50 | Yes (highest quality) |
| thompson | 0.92 | $1.80 | Yes |
| linucb | 0.90 | $1.20 | Yes |
| epsilon | 0.88 | $1.50 | No (dominated by linucb) |
| always_cheapest | 0.75 | $0.30 | Yes (lowest cost) |

---

## Budget-Based Filtering

### Pre-Selection Cost Estimation

Before bandit selection, models can be filtered by budget:

```python
def estimate_cost(model: ModelArm, input_tokens: int) -> float:
    """Estimate cost before execution."""
    output_tokens = int(input_tokens * output_ratio)  # Default 1.0

    pricing = get_model_pricing(model.model_id)

    input_cost = input_tokens * pricing.input_cost_per_token
    output_cost = output_tokens * pricing.output_cost_per_token

    return input_cost + output_cost
```

### Budget Filter

```python
def filter_by_budget(
    models: list[ModelArm],
    max_cost: float,
    query_text: str,
) -> list[ModelArm]:
    """Return models within budget."""

    input_tokens = estimate_tokens(query_text)

    within_budget = [
        model for model in models
        if estimate_cost(model, input_tokens) <= max_cost
    ]

    # Fallback: if none fit, use cheapest
    if not within_budget:
        return [min(models, key=lambda m: estimate_cost(m, input_tokens))]

    return within_budget
```

---

## Implementation References

| Component | File |
|-----------|------|
| Pricing retrieval | `conduit/core/pricing.py` |
| Cost calculation | `conduit/core/pricing.py:compute_cost()` |
| Cost filter | `conduit/engines/cost_filter.py` |
| Metrics calculation | `conduit_bench/analysis/metrics.py` |
| Pareto frontier | `conduit_bench/analysis/metrics.py:identify_pareto_frontier()` |

---

## Appendix A: Unit Conversions

| From | To | Formula |
|------|-----|---------|
| $/1M tokens | $/token | `cost_per_token = cost_per_million / 1,000,000` |
| $/token | $/1K tokens | `cost_per_1k = cost_per_token × 1,000` |
| $/1M tokens | $/1K tokens | `cost_per_1k = cost_per_million / 1,000` |

---

## Appendix B: Typical Query Costs

For a typical MMLU query (~150 input tokens, ~50 output tokens):

| Model | Input Cost | Output Cost | Total |
|-------|------------|-------------|-------|
| gpt-5-nano | $0.000015 | $0.000020 | $0.000035 |
| gemini-2.5-flash | $0.000023 | $0.000030 | $0.000053 |
| gpt-5-mini | $0.000060 | $0.000080 | $0.000140 |
| claude-haiku-4.5 | $0.000120 | $0.000200 | $0.000320 |
| claude-sonnet-4.5 | $0.000450 | $0.000750 | $0.001200 |
| gpt-5.1 | $0.000750 | $0.000750 | $0.001500 |
| claude-opus-4.5 | $0.002250 | $0.003750 | $0.006000 |

**1000-query benchmark cost range**: $0.035 (all gpt-5-nano) to $6.00 (all claude-opus-4.5)
