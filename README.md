# Conduit Bench

**Multi-armed bandit algorithm benchmarking for the Conduit Router**

Benchmark bandit algorithms using real evaluation datasets (GSM8K, MMLU, HumanEval) and synthetic queries to validate Conduit's intelligent LLM routing.

---

## Purpose

**Research Question**: Which bandit algorithm achieves the best cost/quality trade-off for LLM routing?

**What We Test**:
- **11 Algorithms** (all 12 except Oracle): 3 contextual, 3 non-contextual, 2 hybrid, 3 baselines
  - Oracle excluded from default benchmarks due to 6x cost (executes all 6 models per query)
- **6 Models**: OpenAI (o4-mini, gpt-5, gpt-5.1), Anthropic (claude-sonnet-4.5, claude-opus-4.5), Google (gemini-2.5-pro)
- **Real Datasets**: MMLU (~1,319 obs), GSM8K (~14,000 obs)
  - **Sampling**: Randomly sample 1,000 observations from each dataset for experiments
- **Synthetic Queries**: Diverse categories with complexity variation
- **Algorithm Presets**: balanced, quality, cost, speed (see [ORACLE_AND_PRESETS.md](DOCS/ORACLE_AND_PRESETS.md))

**Goal**: Validate that Thompson Sampling and contextual bandits achieve 40-50% cost savings while maintaining 95%+ quality.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BENCHMARK DESIGN                          │
│                                                              │
│  Conduit (source of truth)                                  │
│  ├─ conduit.engines.bandits    → All 10 algorithms          │
│  └─ conduit.models             → Model registry + pricing   │
│                                                              │
│  Conduit-Bench (benchmark runner)                           │
│  ├─ datasets/                  → GSM8K, MMLU, HumanEval     │
│  ├─ generators/                → Synthetic query generation │
│  ├─ runners/                   → Benchmark execution        │
│  ├─ evaluators/                → Exact match, code exec     │
│  ├─ adapters/                  → HybridRouter adapter       │
│  └─ analysis/                  → Metrics + visualization    │
│                                                              │
│  Arbiter (quality evaluation)                               │
│  └─ Semantic similarity + custom criteria scoring           │
└─────────────────────────────────────────────────────────────┘
```

---

## Bandit Algorithms (from Conduit)

All algorithms are imported from `conduit.engines.bandits`. No duplication.

**Total: 12 algorithms** (11 tested by default, Oracle excluded)

### Contextual Algorithms (use query features)

| Algorithm | CLI Name | Description | Best For |
|-----------|----------|-------------|----------|
| **LinUCBBandit** | `linucb` | Linear UCB with ridge regression | Production routing with query features |
| **ContextualThompsonSamplingBandit** | `contextual_thompson` | Bayesian linear regression | Complex query-dependent routing |
| **DuelingBandit** | `dueling` | Pairwise preference learning | High sample efficiency |

### Non-Contextual Algorithms

| Algorithm | CLI Name | Description | Best For |
|-----------|----------|-------------|----------|
| **ThompsonSamplingBandit** | `thompson` | Beta-Bernoulli Bayesian | Cold start, default in Conduit |
| **UCB1Bandit** | `ucb1` | Upper confidence bound | Fast convergence, deterministic |
| **EpsilonGreedyBandit** | `epsilon` | Simple exploration/exploitation | Baseline comparison |

### Hybrid Algorithms (combine strategies)

| Algorithm | CLI Name | Description | Best For |
|-----------|----------|-------------|----------|
| **HybridThompsonLinUCBBandit** | `hybrid_thompson_linucb` | Thompson → LinUCB transition | Production hybrid routing |
| **HybridUCB1LinUCBBandit** | `hybrid_ucb1_linucb` | UCB1 → LinUCB transition | Fast hybrid routing |

### Baselines

| Algorithm | CLI Name | Description | Purpose |
|-----------|----------|-------------|---------|
| **RandomBaseline** | `random` | Uniform random selection | Lower bound |
| **OracleBaseline** | `oracle` | Perfect knowledge (6x cost) | Upper bound (NOT tested by default) |
| **AlwaysBestBaseline** | `always_best` | Always highest quality model | Quality ceiling |
| **AlwaysCheapestBaseline** | `always_cheapest` | Always lowest cost model | Cost floor |

---

## Model Pool & Configuration (from Conduit)

**Dynamic Pricing**: Pricing loaded from LiteLLM's bundled pricing database via `conduit.core.pricing`:
- No external API calls or database required
- Pricing updates with `uv update litellm`
- Optional: Use `--sync-pricing` flag to sync to database for historical tracking

**Quality Priors**: Context-specific quality expectations loaded from `conduit.yaml`:

| Context | Description | Best Models |
|---------|-------------|-------------|
| `code` | Programming tasks | claude-sonnet-4.5 (0.92), claude-opus-4.5 (0.91) |
| `creative` | Creative writing | claude-opus-4.5 (0.94), claude-sonnet-4.5 (0.90) |
| `analysis` | Analytical reasoning | claude-opus-4.5 (0.92), gpt-5.1 (0.89) |
| `simple_qa` | Simple questions | o4-mini (0.90), gemini-2.0-flash (0.88) |
| `general` | General purpose | gpt-5.1 (0.88), claude-opus-4.5 (0.87) |

**Model Pool** (7 models across 3 providers):

| Model | Provider | Input $/1M | Output $/1M |
|-------|----------|------------|-------------|
| o4-mini | OpenAI | $0.15 | $0.60 |
| gpt-5 | OpenAI | $2.50 | $10.00 |
| gpt-5.1 | OpenAI | $2.50 | $10.00 |
| claude-sonnet-4.5 | Anthropic | $3.00 | $15.00 |
| claude-opus-4.5 | Anthropic | $15.00 | $75.00 |
| gemini-2.5-pro | Google | $1.25 | $5.00 |
| gemini-2.5-flash | Google | $0.075 | $0.30 |

**Price Range**: $0.075 - $75.00 per 1M tokens (100x spread)
**Quality Context**: Set via `CONDUIT_QUALITY_CONTEXT=code` (default: `general`)
**Cost Optimization**: Route simple queries to gemini-2.5-flash, complex to opus/gpt-5.1

---

## Datasets

### Real Evaluation Datasets

**MMLU** (Massive Multitask Language Understanding):
- **Total observations**: ~1,319 test problems
- **Sampling**: Randomly sample 1,000 observations for benchmark experiments
- **Format**: Multiple choice (A/B/C/D)
- **Coverage**: 57 subjects across STEM, humanities, social sciences
- **Evaluation**: Exact match on letter choice
- **Source**: `huggingface.co/datasets/cais/mmlu`

**GSM8K** (Grade School Math):
- **Total observations**: ~14,000 problems (8,792 train + 1,319 test)
- **Sampling**: Randomly sample 1,000 observations for benchmark experiments
- **Format**: Multi-step math reasoning problems
- **Evaluation**: Exact match on numeric answer after `#### `
- **Source**: `huggingface.co/datasets/openai/gsm8k`

**HumanEval** (Code Generation):
- 164 Python programming problems
- Function signature + docstring provided
- Evaluation: Code execution against test cases

### Synthetic Queries

Generated via `conduit_bench.generators.SyntheticQueryGenerator`:
- 17 categories (technical, creative, business, philosophical, etc.)
- Complexity levels mapped per category
- Optional reference answers from GPT-4o

---

## Quick Start

### Prerequisites

- Python 3.10+
- Conduit installed (`pip install -e ../conduit`)
- Arbiter installed (`pip install -e ../arbiter`)
- API keys for LLM providers (OpenAI, Anthropic, Google)
- PostgreSQL database (optional - for pricing database mode)
- Redis (optional - for pricing cache mode)

### Installation

```bash
cd /Users/evan/Documents/gh/conduit-benchmark
uv sync
cp .env.example .env
# Add API keys to .env

# Configure quality priors context (optional, default: general)
export CONDUIT_QUALITY_CONTEXT=code  # Options: code, creative, analysis, simple_qa, general
```

### Run Benchmark

```bash
# Run with algorithm preset (recommended)
uv run conduit-bench run --dataset mmlu --mmlu-limit 1000 --preset balanced --parallel

# Run all 11 algorithms (excluding Oracle)
uv run conduit-bench run --dataset mmlu --mmlu-limit 1000 \
  --algorithms thompson,ucb1,epsilon,linucb,contextual_thompson,dueling,\
hybrid_thompson_linucb,hybrid_ucb1_linucb,random,always_best,always_cheapest \
  --parallel

# Run with GSM8K dataset (1000 sample)
uv run conduit-bench run --dataset gsm8k --limit 1000 --preset balanced

# Run with specific quality context
CONDUIT_QUALITY_CONTEXT=code uv run conduit-bench run --dataset gsm8k --limit 500 --preset quality

# Run specific algorithms (custom selection)
uv run conduit-bench run --dataset mmlu --mmlu-limit 1000 --algorithms thompson,linucb,ucb1

# Include Oracle (explicit opt-in, 6x cost)
uv run conduit-bench run --dataset mmlu --mmlu-limit 100 --algorithms oracle,thompson,random

# Analyze results
uv run conduit-bench analyze --results results/experiment_001/
```

**Algorithm Presets** (see [ORACLE_AND_PRESETS.md](DOCS/ORACLE_AND_PRESETS.md) for details):
- `--preset balanced`: Best mix of learning algorithms (5 algorithms)
- `--preset quality`: Prioritize accuracy over cost (4 algorithms)
- `--preset cost`: Minimize inference costs (3 algorithms)
- `--preset speed`: Fast non-contextual algorithms (4 algorithms)

---

## Integration

### Conduit (Algorithm Source)

```python
from conduit_bench.algorithms import (
    ThompsonSamplingBandit,
    LinUCBBandit,
    UCB1Bandit,
    ContextualThompsonSamplingBandit,
    DuelingBandit,
    EpsilonGreedyBandit,
    RandomBaseline,
    OracleBaseline,
    AlwaysBestBaseline,
    AlwaysCheapestBaseline,
)

from conduit_bench.models import DEFAULT_REGISTRY, PRICING
```

All algorithms and models are re-exported from Conduit for convenience.

### Arbiter (Quality Evaluation)

```python
from arbiter_ai import evaluate

result = await evaluate(
    output=model_response,
    reference=expected_answer,
    evaluators=["semantic"],
    model="o4-mini"
)
quality_score = result.overall_score
```

### HybridRouter Adapter

For benchmarking Conduit's HybridRouter (Thompson -> LinUCB transition):

```python
from conduit.engines.hybrid_router import HybridRouter
from conduit_bench.adapters import HybridRouterBanditAdapter

router = HybridRouter(
    models=["o4-mini", "gpt-5", "claude-sonnet-4.5"],
    phase1_algorithm="thompson_sampling",
    phase2_algorithm="linucb",
    switch_threshold=2000,
)
adapter = HybridRouterBanditAdapter(router)

# Now adapter has BanditAlgorithm interface for benchmarking
arm = await adapter.select_arm(features)
```

---

## Expected Results

| Algorithm | Cost Savings | Quality Maintained | Convergence |
|-----------|--------------|-------------------|-------------|
| Thompson Sampling | 42-48% | 94-96% | 2,500-3,500 queries |
| LinUCB | 45-50% | 95-97% | 2,000-3,000 queries |
| UCB1 | 40-46% | 93-95% | 1,500-2,500 queries |
| Epsilon-Greedy | 35-42% | 91-94% | 4,000-6,000 queries |
| Random | 15-25% | 85-88% | Never |
| Oracle | 50-55% | 98%+ | N/A |

---

## Repository Structure

```
conduit-benchmark/
├── conduit_bench/
│   ├── algorithms/        # Re-exports from conduit.engines.bandits
│   ├── models/            # Re-exports from conduit.models
│   ├── datasets/          # GSM8K, MMLU, HumanEval loaders
│   ├── generators/        # Synthetic query generation
│   ├── evaluators/        # Exact match, code execution
│   ├── adapters/          # HybridRouter adapter
│   ├── runners/           # Benchmark execution
│   ├── analysis/          # Metrics and visualization
│   └── cli.py             # Command-line interface
├── data/                  # Generated datasets (git-ignored)
├── results/               # Experiment results (git-ignored)
├── tests/
├── AGENTS.md
├── README.md
└── pyproject.toml
```

---

## Development

```bash
uv run pytest
uv run mypy conduit_bench/
uv run ruff check conduit_bench/
uv run black conduit_bench/
```

---

## Related Projects

- **[Conduit](../conduit)**: ML-powered LLM routing with bandit algorithms
- **[Arbiter](../arbiter)**: LLM evaluation framework with cost tracking
