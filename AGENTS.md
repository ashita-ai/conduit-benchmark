# AGENTS.md - AI Agent Guide

**Purpose**: Development guidelines for Conduit Bench bandit benchmarking
**Last Updated**: 2025-01-19

---

## Quick Orientation

**Conduit Bench**: Multi-armed bandit algorithm benchmarking for LLM routing
**Stack**: Python 3.10+, PydanticAI, Arbiter (evaluation), 17 models across 6 providers
**Purpose**: Compare bandit algorithms (Thompson Sampling, UCB1, Epsilon-Greedy) to identify optimal cost/quality trade-off

### Research Question

**Which bandit algorithm achieves the best cost/quality trade-off for LLM routing across multiple providers?**

---

## Critical Rules

### 1. Single Experiment Design (NOT 3 Rounds)

**Rule**: One large experiment with all algorithms running in parallel on same dataset

**Rationale**:
- Bandits learn continuously from every query, not in discrete "rounds"
- We're COMPARING algorithms, not iteratively improving one algorithm
- This is algorithm research, not deployment validation

**Experiment Structure**:
```
Generate Dataset (10,000 queries)
    ↓
Run All 7 Algorithms in Parallel on Same Dataset:
    - Thompson Sampling
    - UCB1
    - Epsilon-Greedy
    - Random
    - Oracle
    - AlwaysBest
    - AlwaysCheapest
    ↓
Evaluate All Responses with Arbiter
    ↓
Calculate Metrics:
    - Cumulative Regret
    - Cost Savings
    - Quality Maintained
    - Convergence Speed
    ↓
Compare Results (statistical significance tests)
```

**What We Removed**: The confusing "Round 1 (5K) → Round 2 (1K) → Round 3 (500)" structure that suggested iterative improvement.

### 2. Sample Size Requirements

**Main Experiment**: **10,000 queries**
- Per-Arm: 10,000 / 17 models ≈ 590 samples per model
- Per-Category: 10,000 / 10 categories = 1,000 per category
- Convergence: Sufficient for detection (algorithms stabilize in 2-5K queries)

**Quick Validation**: **1,000 queries**
- Per-Arm: ~59 samples per model (marginal but usable)
- Use Case: Rapid prototyping, parameter tuning

**Statistical Rigor**: **10 independent runs**
- Report: mean ± 95% confidence interval
- Total: 10,000 × 10 = 100,000 queries

**Minimum Viable**:
- 17 models × 30 samples (CLT) = 510 queries
- Recommended: 17 × 100 = 1,700 queries
- Robust: 17 × 500 = 8,500 queries

### 3. Integration Strategy

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

### 4. Model Pool

**17 Models Across 6 Providers**:
- OpenAI (4): gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
- Anthropic (3): claude-3-5-sonnet, claude-3-opus, claude-3-haiku
- Google (3): gemini-1.5-pro, gemini-1.5-flash, gemini-1.0-pro
- Groq (3): llama-3.1-70b, llama-3.1-8b, mixtral-8x7b
- Mistral (3): mistral-large, mistral-medium, mistral-small
- Cohere (2): command-r-plus, command-r

**Pricing Range**: $0.00005 - $0.075 per 1K tokens (1,500× difference!)
**Quality Range**: 0.72 - 0.97 (25% difference)

### 5. Bandit Algorithms

**Learning Algorithms (3)**:
1. **Thompson Sampling** (Bayesian probability matching)
   - Regret: O(√(K × T × ln T)) - optimal
   - Convergence: 2,500-3,500 queries
   - Complexity: Medium (Beta sampling)

2. **UCB1** (Optimistic estimates)
   - Regret: O(√(K × T × ln T)) - optimal
   - Convergence: 1,500-2,500 queries (fastest)
   - Complexity: Low (arithmetic only)

3. **Epsilon-Greedy** (Simple exploration)
   - Regret: O(K × T^(2/3)) - suboptimal
   - Convergence: 4,000-6,000 queries (slowest)
   - Complexity: Very low

**Baselines (4)**:
1. **Random**: Lower bound (uniform random selection)
2. **Oracle**: Upper bound (perfect knowledge - zero regret)
3. **AlwaysBest**: Quality ceiling (always use Claude-3-Opus)
4. **AlwaysCheapest**: Cost floor (always use Llama-3.1-8B)

### 6. Type Safety (Strict Mypy)

All functions require type hints, no `Any` without justification.

### 7. No Placeholders

Production-grade code only. Complete implementations or nothing.

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

## Expected Metrics

### After 10,000 Queries (10 Runs)

**Thompson Sampling**:
- Cumulative Regret: Low (near-optimal)
- Cost Savings: 42-48% vs AlwaysBest
- Quality Maintained: 94-96%
- Convergence: 2,500-3,500 queries

**UCB1**:
- Cumulative Regret: Low (near-optimal)
- Cost Savings: 40-46% vs AlwaysBest
- Quality Maintained: 93-95%
- Convergence: 1,500-2,500 queries (fastest)

**Epsilon-Greedy**:
- Cumulative Regret: Medium (suboptimal)
- Cost Savings: 35-42% vs AlwaysBest
- Quality Maintained: 91-94%
- Convergence: 4,000-6,000 queries (slowest)

**Random**:
- Cumulative Regret: High
- Cost Savings: 15-25%
- Quality Maintained: 85-88%
- Convergence: Never

**Oracle**:
- Cumulative Regret: 0 (by definition)
- Cost Savings: 50-55% (theoretical maximum)
- Quality Maintained: 98%+
- Note: Requires 170,000 LLM calls (10K queries × 17 models)

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

- **[README.md](README.md)**: User documentation, algorithm explanations
- **[conduit_bench/algorithms/](conduit_bench/algorithms/)**: Algorithm implementations

## Related Projects

- **[Conduit](../conduit/)**: ML-powered LLM routing (reference)
- **[Arbiter](../arbiter/)**: LLM evaluation framework (essential)
- **[Loom](../loom/)**: AI pipeline orchestration (optional)

---

**Last Updated**: 2025-01-19
