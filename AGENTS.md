---
name: conduit-bench-agent
description: Bandit algorithm benchmarking researcher - evaluating LLM routing strategies
---

# AGENTS.md - AI Agent Guide

**Purpose**: Development guidelines for Conduit Bench bandit benchmarking
**Last Updated**: 2025-01-22
**Status**: ✅ Synced with Conduit test improvements (85% coverage, bandit fixes)

**Design Philosophy**: Simplicity wins, use good defaults, YAML config where needed, no hardcoded assumptions.

---

## Quick Start (First Session Commands)

**New to this repo? Run these 5 commands first:**

```bash
# 1. Verify you're on a feature branch (NEVER work on main)
git status && git branch

# 2. Install dependencies and run tests
poetry install
poetry run pytest --cov=conduit_bench

# 3. Run specific algorithm test to verify environment
poetry run pytest tests/test_algorithms.py -v

# 4. Check for any TODOs or placeholders (should be NONE)
grep -r "TODO\|FIXME\|NotImplementedError" conduit_bench/ || echo "✅ No placeholders found"

# 5. Verify type checking and linting
poetry run mypy conduit_bench/
poetry run ruff check conduit_bench/
```

---

## Quick Orientation

**Conduit Bench**: Multi-armed bandit algorithm benchmarking for LLM routing
**Stack**: Python 3.10+, PydanticAI, Arbiter (evaluation), 17 models across 6 providers
**Purpose**: Compare bandit algorithms (Thompson Sampling, UCB1, Epsilon-Greedy) to identify optimal cost/quality trade-off

**Research Methodology**: See [RESEARCH.md](RESEARCH.md) for experimental design, sample sizes, and expected results

---

## Session Analysis & Continuous Improvement

**When to Analyze** (Multiple Triggers):
- During active sessions: After completing major tasks or every 30-60 minutes
- When failures occur: Immediately analyze and update rules
- Session end: Review entire session for patterns before closing
- User corrections: Any time user points out a mistake

**Identify Failures**:
- Framework violations (boundaries crossed, rules ignored)
- Repeated patterns (same mistake multiple times)
- Rules that didn't prevent failures
- User corrections (what needed fixing)

**Analyze Each Failure**:
- What rule should have prevented this?
- Why didn't it work? (too vague, wrong priority, missing detection pattern)
- What would have caught this earlier?

**Update AGENTS.md** (In Real-Time):
- Add new rules or strengthen existing rules immediately
- Add detection patterns (git commands, test patterns, code patterns)
- Include examples of violations and corrections
- Update priority if rule was underweighted
- Propose updates to user during session (don't wait until end)

**Priority Levels**:
- 🔴 **CRITICAL**: Security, credentials, production breaks → Update immediately, stop work
- 🟡 **IMPORTANT**: Framework violations, repeated patterns → Update with detection patterns, continue work
- 🟢 **RECOMMENDED**: Code quality, style issues → Update with examples, lowest priority

**Example Pattern**:
```
Failure: Committed TODO comments in production code (violated "No Partial Features" rule)
Detection: `grep -r "TODO" src/` before commit
Rule Update: Add pre-commit check pattern to Boundaries section
Priority: 🟡 IMPORTANT
Action Taken: Proposed rule update to user mid-session, updated AGENTS.md
```

**Proactive Analysis**:
- Before risky operations: Check if existing rules cover this scenario
- After 3+ similar operations: Look for pattern that should be codified
- When uncertainty arises: Document the decision-making gap

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

### ✅ Always Do (No Permission Needed)
- Run tests: `poetry run pytest`, `pytest --cov=conduit_bench`, `pytest -v`
- Format code: `poetry run black conduit_bench/`
- Lint code: `poetry run ruff check conduit_bench/`
- Type check: `poetry run mypy conduit_bench/` (strict mode required)
- Add unit tests for new algorithms in `tests/`
- Update docstrings when changing function signatures
- Add examples to `examples/` for new benchmarks

### ⚠️ Ask First

**Core Integration** (Why: Breaks benchmark foundation):
- Modify Conduit algorithm imports - Source of truth for all algorithms
- Change benchmark runner architecture - Affects all experiments
- Modify model registry - Changes which models are benchmarked
- Add new bandit algorithms - Must follow established patterns

**Dependencies & Analysis** (Why: Affects research validity):
- Add/update dependencies in `pyproject.toml` - Increases attack surface
- Change analysis metrics - Research comparability at risk
- Modify visualization approaches - Results presentation affected

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

**Detection Commands** (Run before committing):
```bash
# Check for security violations
grep -r "API_KEY\|SECRET\|PASSWORD" conduit_bench/ tests/ && echo "🚨 CREDENTIALS FOUND" || echo "✅ No credentials"

# Check for code quality violations
grep -r "TODO\|FIXME" conduit_bench/ && echo "🚨 TODO comments found" || echo "✅ No TODOs"

# Check for incomplete features
grep -r "NotImplementedError\|pass  # TODO" conduit_bench/ && echo "🚨 Placeholder code found" || echo "✅ No placeholders"

# Verify on feature branch
git branch --show-current | grep -E "^(main|master)$" && echo "🚨 ON MAIN BRANCH - CREATE FEATURE BRANCH" || echo "✅ On feature branch"

# Verify type checking
poetry run mypy conduit_bench/ && echo "✅ Type checking passed" || echo "🚨 TYPE ERRORS"
```

---

## Common Mistakes & How to Avoid Them

### Mistake 1: Not Importing Algorithms from Conduit
**Detection**: Algorithm implementations duplicated in conduit_bench
**Prevention**: Always import from `conduit.engines.bandits`
**Fix**: Remove duplicated code, import from Conduit
**Why It Matters**: Conduit is source of truth for all algorithms

### Mistake 2: Hardcoding Model Lists
**Detection**: Model lists defined directly in benchmark code
**Prevention**: Use `conduit.models.DEFAULT_REGISTRY`
**Fix**: Import and use model registry
**Why It Matters**: Model updates should happen in one place

### Mistake 3: Skipping Statistical Significance Tests
**Detection**: Single-run benchmarks without error bars
**Prevention**: Run multiple independent trials (10+)
**Fix**: Add `--runs N` parameter to benchmarks
**Why It Matters**: Research validity requires statistical rigor

### Mistake 4: Not Mocking LLM Calls in Tests
**Detection**: Tests hitting real LLM APIs
**Prevention**: Mock all LLM calls in unit tests
**Fix**: Use `mocker.patch` for API calls
**Why It Matters**: Tests should be fast and free

### Mistake 5: Inconsistent Metric Calculations
**Detection**: Metrics calculated differently across analysis scripts
**Prevention**: Centralize metric calculations
**Fix**: Create `conduit_bench/analysis/metrics.py`
**Why It Matters**: Consistent metrics enable fair comparison

### Mistake 6: Using Wrong API Key Environment Variable
**Detection**: Tests fail with "Invalid API key"
**Prevention**: Use `GEMINI_API_KEY` not `GOOGLE_API_KEY`
**Fix**: Update environment variable names
**Why It Matters**: Google renamed their API key variable

### Mistake 7: Not Seeding Random Number Generators
**Detection**: Non-reproducible benchmark results
**Prevention**: Always set random seed for experiments
**Fix**: Add `--seed N` parameter and use consistently
**Why It Matters**: Research reproducibility is critical

---

## Testing Decision Matrix

**When to Mock:**
- LLM API calls (all 6 providers) - Use mocked responses to avoid costs
- Arbiter evaluation calls - Mock for unit tests
- Random number generation - Use fixed seeds for determinism

**When to Use Real Dependencies:**
- Bandit algorithm logic - Real algorithm implementations from Conduit
- Statistical calculations - Real math operations
- Data processing - Real pandas/polars operations

**Example:**
```python
# ✅ GOOD - Mock LLM calls
@pytest.mark.asyncio
async def test_benchmark_runner_mocked(mocker):
    mocker.patch("conduit_bench.runners.model_executor.Agent.run")
    runner = BenchmarkRunner(algorithms=[...], dataset=[...])
    # Test logic without hitting real APIs

# ✅ GOOD - Real algorithm logic
def test_thompson_sampling():
    bandit = ThompsonSamplingBandit(arms=test_arms, random_seed=42)
    arm = await bandit.select_arm(context)
    assert arm in test_arms  # Real algorithm selection

# ❌ BAD - Using real API in tests
async def test_runner():
    results = await run_benchmark(dataset, algorithms)  # Costs $$!
```

---

## Pre-Commit Validation

```bash
# 1. Tests pass
poetry run pytest --cov=conduit_bench
if [ $? -ne 0 ]; then echo "🚨 TESTS FAILED"; exit 1; fi

# 2. Type checking clean
poetry run mypy conduit_bench/
if [ $? -ne 0 ]; then echo "🚨 TYPE ERRORS - FIX BEFORE COMMIT"; exit 1; fi

# 3. Linting clean
poetry run ruff check conduit_bench/
if [ $? -ne 0 ]; then echo "🚨 LINT ERRORS - FIX BEFORE COMMIT"; exit 1; fi

# 4. Formatted
poetry run black conduit_bench/

# 5. No TODOs or placeholders
grep -r "TODO\|FIXME\|NotImplementedError" conduit_bench/ && echo "🚨 REMOVE TODOs" && exit 1

# 6. No credentials
grep -r "API_KEY\|SECRET\|PASSWORD" conduit_bench/ tests/ && echo "🚨 CREDENTIALS FOUND" && exit 1

# All checks passed
echo "✅ All checks passed - ready to commit"
git add <files>
git commit -m "Clear message"
```

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
