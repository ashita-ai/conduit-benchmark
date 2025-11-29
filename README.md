# Conduit Bench

**Multi-armed bandit algorithm benchmarking for the Conduit Router**

Benchmark bandit algorithms using real evaluation datasets (GSM8K, MMLU, HumanEval) and synthetic queries to validate Conduit's intelligent LLM routing.

---

## Purpose

**Research Question**: Which bandit algorithm achieves the best cost/quality trade-off for LLM routing?

**What We Test**:
- **10 Algorithms** from Conduit: 3 contextual, 3 non-contextual, 4 baselines
- **7 Models**: OpenAI (o4-mini, gpt-5, gpt-5.1), Anthropic (claude-sonnet-4.5, claude-opus-4.5), Google (gemini-2.5-pro, gemini-2.5-flash)
- **Real Datasets**: GSM8K (math), MMLU (knowledge), HumanEval (code)
- **Synthetic Queries**: Diverse categories with complexity variation

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

### Contextual Algorithms (use query features)

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| **LinUCBBandit** | Linear UCB with ridge regression | Production routing with query features |
| **ContextualThompsonSamplingBandit** | Bayesian linear regression | Complex query-dependent routing |
| **DuelingBandit** | Pairwise preference learning | High sample efficiency |

### Non-Contextual Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| **ThompsonSamplingBandit** | Beta-Bernoulli Bayesian | Cold start, default in Conduit |
| **UCB1Bandit** | Upper confidence bound | Fast convergence, deterministic |
| **EpsilonGreedyBandit** | Simple exploration/exploitation | Baseline comparison |

### Baselines

| Algorithm | Description | Purpose |
|-----------|-------------|---------|
| **RandomBaseline** | Uniform random selection | Lower bound |
| **OracleBaseline** | Perfect knowledge | Upper bound (expensive) |
| **AlwaysBestBaseline** | Always highest quality model | Quality ceiling |
| **AlwaysCheapestBaseline** | Always lowest cost model | Cost floor |

---

## Model Pool & Configuration (from Conduit)

**Dynamic Pricing**: Pricing loaded from Conduit's PricingManager with three-tier fallback:
1. **Database**: Latest pricing snapshots from `model_prices` table
2. **Cache**: Redis-cached pricing (24-hour TTL)
3. **Live Fetch**: llm-prices.com API for latest pricing

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

**GSM8K** (Grade School Math):
- 8,792 train + 1,319 test problems
- Multi-step reasoning required
- Evaluation: Exact match on numeric answer after `#### `
- Source: `huggingface.co/datasets/openai/gsm8k`

**MMLU** (Massive Multitask Language Understanding):
- 57 subjects across STEM, humanities, social sciences
- Multiple choice format
- Evaluation: Exact match on letter choice (A/B/C/D)

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
# Run with GSM8K dataset
uv run conduit-bench run --dataset gsm8k --limit 1000

# Run with specific quality context
CONDUIT_QUALITY_CONTEXT=code uv run conduit-bench run --dataset gsm8k --limit 500

# Run with synthetic queries
uv run conduit-bench generate --queries 1000 --seed 42
uv run conduit-bench run --dataset data/queries_1000.jsonl

# Run specific algorithms
uv run conduit-bench run --dataset gsm8k --algorithms thompson_sampling,linucb,ucb1

# Analyze results
uv run conduit-bench analyze --results results/experiment_001/
```

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
