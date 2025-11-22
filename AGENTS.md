---
name: conduit-bench-agent
description: Bandit algorithm benchmarking researcher - evaluating LLM routing strategies
---

# AGENTS.md - AI Agent Guide

**Purpose**: Development guidelines for Conduit Bench bandit benchmarking
**Last Updated**: 2025-01-22
**Status**: ✅ Synced with Conduit test improvements (85% coverage, bandit fixes)

---

## Quick Orientation

**Conduit Bench**: Multi-armed bandit algorithm benchmarking for LLM routing
**Stack**: Python 3.10+, PydanticAI, Arbiter (evaluation), 17 models across 6 providers
**Purpose**: Compare bandit algorithms (Thompson Sampling, UCB1, Epsilon-Greedy) to identify optimal cost/quality trade-off

**Research Methodology**: See [RESEARCH.md](RESEARCH.md) for experimental design, sample sizes, and expected results

---

## Critical Rules

### 1. Integration Strategy

**Conduit (ALGORITHM SOURCE - ESSENTIAL)**:
- **Role**: Contains ALL bandit algorithm implementations
- **Location**: `conduit.engines.bandits` (7 algorithms) + `conduit.models` (registry)
- **Import**: `from conduit.engines.bandits import ThompsonSamplingBandit, UCB1Bandit, ...`
- **Why**: Single source of truth, no code duplication, algorithms usable by Router and benchmark

**Arbiter (EVALUATION - ESSENTIAL)**:
- **Role**: Quality evaluation for model responses
- **Usage**: `evaluate(output, reference, evaluators=["semantic"])`
- **Integration**: Used by benchmark runner for quality scoring
- **Why**: Provider-agnostic, battle-tested, handles all 17 models

**Loom (NOT NEEDED)**:
- Too heavyweight for research benchmark
- Adds unnecessary complexity (pipelines, quality gates)
- Use for production later, not for algorithm comparison

**Architecture**:
```
Conduit (algorithms + model registry)
    ↑
    | imports from
    |
conduit-bench (benchmark runner + analysis)
```

### 2. Type Safety (Strict Mypy)

All functions require type hints, no `Any` without justification.

### 3. No Placeholders

Production-grade code only. Complete implementations or nothing.

---

## Boundaries

### 🚫 NEVER Do

**CRITICAL SECURITY VIOLATION** ⚠️:
- **NEVER EVER COMMIT CREDENTIALS TO GITHUB**
- No API keys, tokens, passwords, secrets in ANY file
- No credentials in code, documentation, examples, tests, or configuration files
- Use environment variables (.env files in .gitignore) ONLY
- This is NON-NEGOTIABLE - violating this rule has serious security consequences

**Code Quality Violations**:
- Skip tests to make builds pass
- Disable type checking or linting errors
- Leave TODO comments in production code
- Create placeholder/stub implementations

**Destructive Actions**:
- Work directly on main/master branch
- Force push to shared branches
- Delete failing tests instead of fixing them
- Remove error handling to "fix" issues

---

## Development Workflow

### Before Starting

1. **Check dependencies**:
   ```bash
   # Ensure Arbiter exists
   ls ../arbiter
   ```

2. **Create feature branch**:
   ```bash
   git checkout -b feature/benchmark-runner
   ```

### During Development

```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest

# Type check
poetry run mypy conduit_bench/

# Format
poetry run black conduit_bench/
```

### Before Committing

```bash
poetry run pytest --cov=conduit_bench   # Tests pass
poetry run mypy conduit_bench/          # Type checking clean
poetry run ruff check conduit_bench/    # Linting clean
poetry run black conduit_bench/         # Formatted
```

---

## Tech Stack

### Core Dependencies ✅
- **Python**: 3.10+ (modern type hints, async/await)
- **Pydantic**: 2.12+ (data validation and serialization)
- **PydanticAI**: 1.20+ (unified LLM interface)
- **Arbiter**: Latest via git (evaluation engine)
- **NumPy**: 1.24+ (Beta distribution sampling)

### Data Processing
- **Pandas**: 2.2+ (DataFrame operations)
- **Polars**: 0.19+ (Fast data processing)

### LLM Providers (via PydanticAI)
- OpenAI, Anthropic, Google, Groq, Mistral, Cohere

### Analysis & Visualization
- **Matplotlib**: 3.9+ (Static plots)
- **Seaborn**: 0.13+ (Statistical visualizations)
- **Plotly**: 5.24+ (Interactive dashboards)

### CLI
- **Click**: 8.1+ (Command-line interface)
- **Rich**: 14.0+ (Beautiful terminal output)

---

## Key Implementation Patterns

### Bandit Algorithm Interface

```python
from conduit_bench.algorithms import BanditAlgorithm, BanditContext, BanditFeedback

class MyBandit(BanditAlgorithm):
    def __init__(self, arms: list[ModelArm]) -> None:
        super().__init__(name="my_bandit", arms=arms)
        # Initialize algorithm state

    async def select_arm(self, context: BanditContext) -> ModelArm:
        """Select which model to use for this query."""
        # Algorithm logic
        return selected_arm

    async def update(self, feedback: BanditFeedback, context: BanditContext) -> None:
        """Update algorithm state with feedback."""
        # Learning logic

    def reset(self) -> None:
        """Reset to initial state."""
        # Clear state
```

### Model Registry Usage

```python
from conduit_bench.models import DEFAULT_REGISTRY, filter_models

# Get all 17 models
all_models = DEFAULT_REGISTRY

# Filter models
high_quality_models = filter_models(
    DEFAULT_REGISTRY,
    min_quality=0.85,
    max_cost=0.005,
    providers=["openai", "anthropic"]
)
```

### Running Benchmark

```python
# Import from Conduit (algorithms live here)
from conduit.engines.bandits import (
    ThompsonSamplingBandit,
    UCB1Bandit,
    EpsilonGreedyBandit,
    RandomBaseline,
    OracleBaseline,
    AlwaysBestBaseline,
    AlwaysCheapestBaseline,
)
from conduit.models import DEFAULT_REGISTRY

# Or import from conduit_bench (re-exports for convenience)
# from conduit_bench.algorithms import ThompsonSamplingBandit, ...
# from conduit_bench.models import DEFAULT_REGISTRY

# Create all 7 algorithms
algorithms = [
    ThompsonSamplingBandit(DEFAULT_REGISTRY),
    UCB1Bandit(DEFAULT_REGISTRY, c=1.5),
    EpsilonGreedyBandit(DEFAULT_REGISTRY, epsilon=0.1),
    RandomBaseline(DEFAULT_REGISTRY),
    OracleBaseline(DEFAULT_REGISTRY),
    AlwaysBestBaseline(DEFAULT_REGISTRY),
    AlwaysCheapestBaseline(DEFAULT_REGISTRY),
]

# Run experiment (to be implemented)
# results = await run_benchmark(dataset, algorithms)
```

---

## Directory Structure

```
conduit-bench/
├── conduit_bench/
│   ├── algorithms/              # ✅ COMPLETE
│   │   ├── base.py              # Base classes and interfaces
│   │   ├── thompson_sampling.py # Bayesian approach
│   │   ├── ucb.py               # Upper Confidence Bound
│   │   ├── epsilon_greedy.py    # Simple exploration
│   │   └── baselines.py         # Random, Oracle, Always-*
│   ├── models/                  # ✅ COMPLETE
│   │   └── registry.py          # 17 models with pricing
│   ├── generators/              # ⚠️ TO BUILD
│   │   └── synthetic.py         # Generate diverse queries
│   ├── runners/                 # ⚠️ TO BUILD
│   │   ├── model_executor.py    # Direct PydanticAI calls
│   │   └── benchmark_runner.py  # Algorithm comparison
│   ├── analysis/                # ⚠️ TO BUILD
│   │   ├── metrics.py           # Regret, cost, quality
│   │   └── visualize.py         # Charts and plots
│   └── cli.py                   # ⚠️ TO BUILD
├── tests/                       # ⚠️ TO BUILD
│   ├── test_algorithms.py       # Test bandit algorithms
│   ├── test_models.py           # Test model registry
│   └── test_benchmark.py        # Integration tests
├── AGENTS.md                    # This file
├── README.md                    # User documentation
└── pyproject.toml               # Dependencies
```

---

## Testing Strategy

### Unit Tests

```python
# Test algorithm selection
async def test_thompson_sampling_selection():
    arms = create_test_arms(n=5)
    bandit = ThompsonSamplingBandit(arms, random_seed=42)

    context = BanditContext(query_text="Test query")
    arm = await bandit.select_arm(context)

    assert arm in arms
    assert bandit.total_queries == 1

# Test algorithm update
async def test_thompson_sampling_update():
    arms = create_test_arms(n=5)
    bandit = ThompsonSamplingBandit(arms)

    context = BanditContext(query_text="Test query")
    arm = await bandit.select_arm(context)

    feedback = BanditFeedback(
        model_id=arm.model_id,
        cost=0.001,
        quality_score=0.95,
        latency=1.2
    )

    await bandit.update(feedback, context)

    # Verify Beta distribution updated
    assert bandit.alpha[arm.model_id] > 1.0
```

### Integration Tests

```python
# Test full benchmark flow
async def test_benchmark_runner():
    # Generate small dataset
    dataset = generate_test_dataset(n=100)

    # Create algorithms
    algorithms = [
        ThompsonSamplingBandit(TEST_ARMS),
        UCB1Bandit(TEST_ARMS),
        RandomBaseline(TEST_ARMS),
    ]

    # Run benchmark
    results = await run_benchmark(dataset, algorithms)

    # Verify results
    assert len(results) == len(algorithms)
    assert all(r.total_queries == 100 for r in results)
```

---

## Common Tasks

### Generate Dataset

```bash
poetry run conduit-bench generate --queries 10000 --seed 42
```

### Run Benchmark

```bash
# Single run
poetry run conduit-bench run --dataset data/queries_10000.jsonl

# Multiple runs for statistical significance
poetry run conduit-bench run --dataset data/queries_10000.jsonl --runs 10
```

### Analyze Results

```bash
poetry run conduit-bench analyze --results results/experiment_001/
poetry run conduit-bench visualize --results results/experiment_001/
```

---

## Environment Variables

```bash
# LLM Provider API Keys (all 6 providers for full benchmark)
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GEMINI_API_KEY=...       # NOTE: GEMINI_API_KEY (not GOOGLE_API_KEY)
GROQ_API_KEY=...
MISTRAL_API_KEY=...
COHERE_API_KEY=...

# Benchmarking
BENCHMARK_QUERY_COUNT=10000
BENCHMARK_RUNS=10
BENCHMARK_SEED=42
```

---

## Working with AI Agents

### Task Management
**TodoWrite enforcement (MANDATORY)**: For ANY task with 3+ distinct steps, use TodoWrite to track progress.

### Output Quality
**Full data display**: Show complete data structures, not summaries. Examples should display real output.

### Audience & Context Recognition
**Auto-detect technical audiences**: No marketing language in engineering contexts (code examples, technical docs).

### Quality & Testing
**Test output quality, not just functionality**: Verify examples produce useful results.

### Workflow Patterns
**Iterate fast**: Ship → test → get feedback → fix → commit.

### Git & Commit Hygiene
**Clean workflow**: Feature branches, meaningful commits.

---

## Quick Reference

### Run Full Benchmark

```bash
# Generate dataset (10,000 queries)
poetry run conduit-bench generate --queries 10000 --seed 42

# Run all 7 algorithms (10 independent runs)
poetry run conduit-bench run --dataset data/queries_10000.jsonl --runs 10

# Analyze results
poetry run conduit-bench analyze --results results/experiment_001/

# Generate visualizations
poetry run conduit-bench visualize --results results/experiment_001/
```

### Development

```bash
# Tests
poetry run pytest --cov=conduit_bench

# Type check
poetry run mypy conduit_bench/

# Format
poetry run black conduit_bench/

# Lint
poetry run ruff check conduit_bench/
```

---

## Related Documents

- **[RESEARCH.md](RESEARCH.md)**: Experimental design and methodology
- **[README.md](README.md)**: User documentation, algorithm explanations
- **[conduit_bench/algorithms/](conduit_bench/algorithms/)**: Algorithm implementations

## Related Projects

- **[Conduit](../conduit/)**: ML-powered LLM routing (algorithm source)
- **[Arbiter](../arbiter/)**: LLM evaluation framework (essential)
- **[Loom](../loom/)**: AI pipeline orchestration (optional)

---

**Last Updated**: 2025-01-22
