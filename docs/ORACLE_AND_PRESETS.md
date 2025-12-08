# Oracle Baseline and Algorithm Presets

## Oracle Baseline

### Why Oracle is Not Benchmarked by Default

The **Oracle** baseline is excluded from all preset configurations and default benchmarks due to its **6x cost increase**.

**How Oracle works:**
- Executes **all 6 models** for each query to gain perfect knowledge
- Selects the best-performing model after seeing all results
- Achieves near-perfect quality (1.0) by definition

**Cost implications:**
- Standard algorithm: 1 model execution per query
- Oracle algorithm: 6 model executions per query = **6x cost**

**When to use Oracle:**
- Establishing theoretical upper bound on quality
- Academic research requiring perfect baseline
- Small-scale experiments (< 100 queries) where cost is acceptable

**How to run Oracle:**
```bash
# Explicit Oracle runs (with cost warning)
conduit-bench run --dataset mmlu --mmlu-limit 10 --algorithms oracle

# Oracle available but excluded from presets
conduit-bench run --dataset mmlu --preset balanced  # No Oracle ✓
```

### Oracle Implementation

Oracle is implemented in `/Users/evan/Documents/gh/conduit-benchmark/conduit_bench/runners/benchmark_runner.py` (lines 246-325):

```python
if algorithm.name == "oracle":
    # Execute ALL arms and feed results to Oracle
    all_results = []
    for arm in algorithm.arms.values():
        exec_result = await self.executor.execute_with_fallback(...)
        quality_score = await self._evaluate_with_evaluator(...)
        oracle_feedback = BanditFeedback(...)
        await algorithm.update(oracle_feedback, features)
        all_results.append((exec_result, quality_score))

    # Now Oracle has perfect knowledge - select best arm
    selected_arm = await algorithm.select_arm(features)
```

**Result:** Oracle metadata includes `oracle_knowledge_size: N` where N = num_queries × 6 models.

---

## Algorithm Presets

Preset configurations provide convenient algorithm selections for different optimization goals.

### Available Presets

#### `--preset balanced`
**Best mix of learning algorithms**

Algorithms: `thompson,linucb,contextual_thompson,hybrid_thompson_linucb,random`

- **Thompson Sampling**: Non-contextual Bayesian exploration
- **LinUCB**: Contextual linear upper confidence bound
- **Contextual Thompson**: Contextual Bayesian optimization
- **Hybrid Thompson-LinUCB**: Combines exploration strategies
- **Random**: Baseline for comparison

**Use case:** General-purpose benchmark comparing diverse learning approaches

#### `--preset quality`
**Prioritize accuracy over cost**

Algorithms: `contextual_thompson,linucb,dueling,always_best`

- **Contextual Thompson**: Best contextual exploration
- **LinUCB**: Strong contextual learner
- **Dueling Bandit**: Pairwise preference learning
- **Always Best**: Upper bound on quality

**Use case:** Quality-critical applications where accuracy matters most

#### `--preset cost`
**Minimize inference costs**

Algorithms: `linucb,always_cheapest,random`

- **LinUCB**: Efficient contextual learning
- **Always Cheapest**: Lower bound on cost
- **Random**: Baseline

**Use case:** Cost-sensitive deployments, high-volume applications

#### `--preset speed`
**Fast non-contextual algorithms**

Algorithms: `thompson,ucb1,epsilon,random`

- **Thompson Sampling**: Fast Bayesian sampling
- **UCB1**: Fast upper confidence bound
- **Epsilon Greedy**: Simple exploration/exploitation
- **Random**: Baseline

**Use case:** Low-latency requirements, simple query distributions

### Usage Examples

```bash
# Use preset configuration
conduit-bench run --dataset mmlu --mmlu-limit 1000 --preset balanced --parallel

# Override preset with custom algorithms
conduit-bench run --dataset mmlu --preset quality --algorithms custom,list,here

# Explicit algorithm list (no preset)
conduit-bench run --dataset mmlu --algorithms thompson,ucb1,linucb --parallel
```

### Preset Algorithm Matrix

| Preset     | Thompson | UCB1 | Epsilon | LinUCB | Ctx-Thompson | Dueling | Hybrid-TL | Random | Always-Best | Always-Cheapest |
|------------|----------|------|---------|--------|--------------|---------|-----------|--------|-------------|-----------------|
| balanced   | ✓        |      |         | ✓      | ✓            |         | ✓         | ✓      |             |                 |
| quality    |          |      |         | ✓      | ✓            | ✓       |           |        | ✓           |                 |
| cost       |          |      |         | ✓      |              |         |           | ✓      |             | ✓               |
| speed      | ✓        | ✓    | ✓       |        |              |         |           | ✓      |             |                 |

**Note:** Oracle is excluded from ALL presets due to 6x cost.

---

## Complete Algorithm List

All 12 available algorithms (Oracle requires explicit specification):

### Learning Algorithms (6)
- `thompson` - Thompson Sampling (non-contextual)
- `ucb1` - Upper Confidence Bound (non-contextual)
- `epsilon` - Epsilon Greedy (non-contextual)
- `linucb` - Linear UCB (contextual)
- `contextual_thompson` - Contextual Thompson Sampling
- `dueling` - Dueling Bandit (pairwise learning)

### Hybrid Algorithms (2)
- `hybrid_thompson_linucb` - Combines Thompson + LinUCB
- `hybrid_ucb1_linucb` - Combines UCB1 + LinUCB

### Baseline Algorithms (4)
- `random` - Uniform random selection
- `always_best` - Always select highest quality model
- `always_cheapest` - Always select cheapest model
- `oracle` - Perfect knowledge (6x cost, explicit use only)

---

## Implementation Details

### Preset Configuration (cli.py:444-449)

```python
ALGORITHM_PRESETS = {
    "balanced": "thompson,linucb,contextual_thompson,hybrid_thompson_linucb,random",
    "quality": "contextual_thompson,linucb,dueling,always_best",
    "cost": "linucb,always_cheapest,random",
    "speed": "thompson,ucb1,epsilon,random",
}
```

### CLI Integration

```python
@click.option(
    "--preset",
    "-p",
    type=click.Choice(["balanced", "quality", "cost", "speed"]),
    help=(
        "Algorithm preset configuration:\n"
        "  balanced: Best mix of learning algorithms\n"
        "  quality: Prioritize accuracy over cost\n"
        "  cost: Minimize inference costs\n"
        "  speed: Fast non-contextual algorithms\n"
        "Note: Oracle excluded from all presets (6x cost)"
    ),
)
```

---

## Testing Recommendations

### Preset Validation

```bash
# Test all presets on small dataset
conduit-bench run --dataset mmlu --mmlu-limit 10 --preset balanced
conduit-bench run --dataset mmlu --mmlu-limit 10 --preset quality
conduit-bench run --dataset mmlu --mmlu-limit 10 --preset cost
conduit-bench run --dataset mmlu --mmlu-limit 10 --preset speed

# Oracle explicit test (understand 6x cost)
conduit-bench run --dataset mmlu --mmlu-limit 5 --algorithms oracle
```

### Expected Behavior

- Preset message displays: `Using '[preset]' preset: [algorithm,list]`
- Oracle excluded unless explicitly specified in `--algorithms`
- `--algorithms` overrides `--preset` if both provided
