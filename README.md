# Conduit Bench 🎯

**Multi-armed bandit algorithm benchmarking for the Conduit Router**

Compare Thompson Sampling, UCB, and Epsilon-Greedy algorithms across 17 models from 6 providers to identify the optimal cost/quality trade-off for the Conduit Router's intelligent LLM routing.

---

## Purpose

**Research Question**: Which bandit algorithm achieves the best cost/quality trade-off for LLM routing across multiple providers?

**What We Test**:
- **3 Learning Algorithms**: Thompson Sampling, UCB1, Epsilon-Greedy
- **4 Baselines**: Random, Oracle, AlwaysBest, AlwaysCheapest
- **17 Models**: OpenAI, Anthropic, Google, Groq, Mistral, Cohere
- **10,000 Queries**: Diverse synthetic dataset across 10 categories

**Goal**: Identify the algorithm that achieves 40-50% cost savings while maintaining 95%+ quality compared to always using GPT-4o.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              SINGLE EXPERIMENT DESIGN                        │
│                                                              │
│  1. Generate Dataset (10,000 queries)                       │
│     └─ 10 categories × 3 complexity levels                  │
│     └─ Reference answers from GPT-4o                        │
│                                                              │
│  2. Run All Algorithms in Parallel                          │
│     ├─ Thompson Sampling (Bayesian)                         │
│     ├─ UCB1 (Optimistic)                                    │
│     ├─ Epsilon-Greedy (Simple)                              │
│     ├─ Random (Lower Bound)                                 │
│     ├─ Oracle (Upper Bound)                                 │
│     ├─ AlwaysBest (Quality Ceiling)                         │
│     └─ AlwaysCheapest (Cost Floor)                          │
│                                                              │
│  3. Evaluate with Arbiter                                   │
│     └─ Semantic similarity + custom criteria                │
│                                                              │
│  4. Calculate Metrics                                       │
│     ├─ Cumulative Regret vs Oracle                          │
│     ├─ Cost Savings vs AlwaysBest                           │
│     ├─ Convergence Speed                                    │
│     └─ Quality Maintained (% above threshold)               │
│                                                              │
│  5. Visualize & Compare                                     │
│     ├─ Regret curves over time                              │
│     ├─ Cost-quality Pareto frontier                         │
│     └─ Model selection heatmaps                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Bandit Algorithms Explained

### Learning Algorithms

#### 1. Thompson Sampling (Bayesian)
**Philosophy**: Probability matching - select arms proportional to probability they're optimal

**How It Works**:
- Maintain Beta(α, β) distribution for each model's quality
- α = successes, β = failures
- Sample from each Beta distribution
- Select model with highest sample

**Pros**:
- Theoretically optimal regret: O(√(K × T × ln T))
- Natural exploration/exploitation balance
- Works well with sparse rewards
- Handles non-stationary environments

**Cons**:
- More complex implementation
- Requires Beta distribution sampling
- Sensitive to prior choice

**Best For**: Complex environments, sparse feedback

#### 2. UCB1 (Upper Confidence Bound)
**Philosophy**: Optimism in the face of uncertainty

**How It Works**:
- UCB(model) = mean_quality + c × √(ln(total_queries) / queries_to_model)
- Select model with highest upper confidence bound
- c parameter controls exploration (higher = more exploration)

**Pros**:
- Theoretically optimal regret: O(√(K × T × ln T))
- Deterministic (given exploration parameter)
- Fast convergence in stationary environments
- Simple, principled approach

**Cons**:
- Can over-explore if c is too high
- Deterministic can be suboptimal in adversarial environments
- Sensitive to exploration parameter choice

**Best For**: Stationary environments, need deterministic policy

#### 3. Epsilon-Greedy
**Philosophy**: Simple exploration/exploitation trade-off

**How It Works**:
- With probability ε: explore (select random model)
- With probability (1-ε): exploit (select best model by mean quality)
- Optional: decay ε over time for better convergence

**Pros**:
- Very simple to implement and understand
- Intuitive parameter (ε = exploration rate)
- Can adapt exploration with decay

**Cons**:
- Explores uniformly (ignores uncertainty)
- Inefficient (explores bad models equally)
- Suboptimal regret: O(K × T^(2/3))

**Best For**: Simple baselines, quick prototyping

### Baseline Algorithms

#### 4. Random Baseline
**Purpose**: Lower bound on performance (what you get with no learning)

**Method**: Uniform random selection across all 17 models

**Expected**: Flat regret curve, average cost/quality across all models

#### 5. Oracle Baseline
**Purpose**: Upper bound (theoretical optimum with perfect knowledge)

**Method**: Always select the best model for each specific query

**Note**: Requires running all 17 models to determine optimal (expensive!)

**Expected**: Zero regret by definition

#### 6. AlwaysBest Baseline
**Purpose**: Quality ceiling, cost floor

**Method**: Always use highest quality model (e.g., Claude-3-Opus)

**Expected**: Best quality, highest cost, illustrates cost/quality trade-off

#### 7. AlwaysCheapest Baseline
**Purpose**: Cost ceiling (maximum savings), quality floor

**Method**: Always use cheapest model (e.g., Llama-3.1-8B via Groq)

**Expected**: Lowest cost, potentially poor quality

### Algorithm Comparison

| Algorithm | Regret Bound | Convergence | Exploration | Complexity |
|-----------|--------------|-------------|-------------|------------|
| Thompson Sampling | O(√KT ln T) | Fast | Probabilistic | Medium |
| UCB1 | O(√KT ln T) | Fastest | Optimistic | Low |
| Epsilon-Greedy | O(KT^(2/3)) | Slow | Uniform | Very Low |
| Random | O(T) | Never | Full | Trivial |
| Oracle | 0 | N/A | None | N/A |

**Key Takeaway**: Thompson Sampling and UCB1 are theoretically optimal. Epsilon-Greedy is suboptimal but simple. The benchmark determines which works best in practice for LLM routing.

---

## Sample Size Requirements

### Statistical Justification

**10,000 Queries (Main Experiment)**:
- **Per-Arm Samples**: 10,000 / 17 models ≈ 590 samples per model
- **Convergence Detection**: Sufficient for algorithms to stabilize (2-5K typical)
- **Per-Category Analysis**: 10,000 / 10 categories = 1,000 per category
- **Statistical Power**: Adequate for detecting 5-10% difference in metrics

**1,000 Queries (Quick Validation)**:
- **Per-Arm Samples**: ~59 samples per model (marginal but usable)
- **Use Case**: Rapid prototyping, parameter tuning
- **Limitation**: May not detect convergence for slower algorithms

**Multiple Runs**:
- **10 Independent Runs**: Standard for reporting mean ± 95% CI
- **Total Experiment**: 10,000 queries × 10 runs = 100,000 queries
- **Per Algorithm**: ~14,300 queries per model across all runs

### Minimum Viable

For algorithm to converge: **≥ 30 samples per arm** (Central Limit Theorem)
- Minimum: 17 models × 30 = **510 queries**
- Recommended: 17 models × 100 = **1,700 queries**
- Robust: 17 models × 500 = **8,500 queries**

**Our Choice**: **10,000 queries** provides robust convergence detection and per-category analysis.

---

## Model Pool (17 Models)

### OpenAI (4 models)
- gpt-4o ($0.0025/1K in, $0.010/1K out, quality: 0.95)
- gpt-4o-mini ($0.00015/1K in, $0.0006/1K out, quality: 0.85)
- gpt-4-turbo ($0.010/1K in, $0.030/1K out, quality: 0.93)
- gpt-3.5-turbo ($0.0005/1K in, $0.0015/1K out, quality: 0.75)

### Anthropic (3 models)
- claude-3-5-sonnet ($0.003/1K in, $0.015/1K out, quality: 0.96)
- claude-3-opus ($0.015/1K in, $0.075/1K out, quality: 0.97)
- claude-3-haiku ($0.00025/1K in, $0.00125/1K out, quality: 0.80)

### Google (3 models)
- gemini-1.5-pro ($0.00125/1K in, $0.005/1K out, quality: 0.92)
- gemini-1.5-flash ($0.000075/1K in, $0.0003/1K out, quality: 0.82)
- gemini-1.0-pro ($0.0005/1K in, $0.0015/1K out, quality: 0.78)

### Groq (3 models)
- llama-3.1-70b ($0.00059/1K in, $0.00079/1K out, quality: 0.88)
- llama-3.1-8b ($0.00005/1K in, $0.00008/1K out, quality: 0.72)
- mixtral-8x7b ($0.00024/1K in, $0.00024/1K out, quality: 0.85)

### Mistral (3 models)
- mistral-large ($0.002/1K in, $0.006/1K out, quality: 0.91)
- mistral-medium ($0.0007/1K in, $0.0021/1K out, quality: 0.86)
- mistral-small ($0.0002/1K in, $0.0006/1K out, quality: 0.79)

### Cohere (2 models)
- command-r-plus ($0.003/1K in, $0.015/1K out, quality: 0.90)
- command-r ($0.0005/1K in, $0.0015/1K out, quality: 0.83)

**Price Range**: $0.00005 - $0.075 per 1K tokens (1,500× difference!)
**Quality Range**: 0.72 - 0.97 (25% difference)
**Optimal Trade-off**: To be determined by benchmark!

---

## Integration with Other Projects

### Arbiter (ESSENTIAL)

**What We Use**:
- `evaluate(output, reference, evaluators=["semantic"])` - Quality scoring
- `batch_evaluate(items)` - Parallel evaluation
- `result.total_llm_cost()` - Automatic cost tracking
- Multiple evaluators (semantic similarity, custom criteria)

**Why Essential**:
- Provides objective quality assessment (0-1 scale)
- Tracks costs automatically for regret calculation
- Provider-agnostic (works with all PydanticAI models)
- Battle-tested evaluation logic

**Integration**: Direct Python import, no configuration needed

### Conduit (ALGORITHM SOURCE - ESSENTIAL)

**What We Use**:
- `from conduit.engines.bandits import ThompsonSamplingBandit, UCB1Bandit, ...` - All 7 bandit algorithms
- `from conduit.models import DEFAULT_REGISTRY` - 17 models with pricing data
- Single source of truth for all algorithm implementations

**Why Essential**:
- All bandit algorithms implemented in Conduit (not duplicated in benchmark)
- Model registry with real pricing/quality data lives in Conduit
- Benchmark imports from Conduit and runs experiments
- Enables Router to use same algorithms validated by benchmark

**Architecture**:
```
Conduit (algorithms + model registry)
    ↑
    | imports from
    |
conduit-bench (benchmark runner + analysis)
```

**Integration**: Direct Python import, no configuration needed

### Loom (NOT NEEDED)

**Why Not Using**:
- Too heavyweight for research benchmark
- Adds complexity (pipelines, quality gates, orchestration)
- Batch evaluation can be done directly with Arbiter
- No need for Extract → Transform → Evaluate → Load pattern

**When It Would Help**:
- Production deployment of winning algorithm
- Continuous benchmarking with real user queries
- Integration with existing data pipelines

**Decision**: Keep benchmark simple, use Loom for production later

---

## Quick Start

### Prerequisites

- Python 3.10+
- Arbiter (sibling directory: `../arbiter`)
- LLM API keys (at least OpenAI for reference answers)

### Installation

```bash
# Clone repo
cd /Users/evan/Documents/gh
git clone https://github.com/yourusername/conduit-bench.git
cd conduit-bench

# Install dependencies
poetry install

# Setup environment
cp .env.example .env
# Edit .env with your API keys (all 6 providers for full benchmark)
```

### Run Benchmark

```bash
# Generate dataset (10,000 queries)
poetry run conduit-bench generate --queries 10000 --seed 42

# Run all algorithms (single run)
poetry run conduit-bench run --dataset data/queries_10000.jsonl

# Run with multiple runs for statistical significance
poetry run conduit-bench run --dataset data/queries_10000.jsonl --runs 10

# Analyze results
poetry run conduit-bench analyze --results results/experiment_001/

# Generate visualizations
poetry run conduit-bench visualize --results results/experiment_001/
```

### Quick Validation (1,000 queries)

```bash
# Fast validation for development
poetry run conduit-bench generate --queries 1000 --seed 42
poetry run conduit-bench run --dataset data/queries_1000.jsonl --runs 3
poetry run conduit-bench analyze --results results/validation_001/
```

---

## Expected Results

### Main Findings (10,000 queries, 10 runs)

**Thompson Sampling**:
- Cumulative Regret: **Low** (near-optimal)
- Cost Savings: **42-48%** vs AlwaysBest
- Quality Maintained: **94-96%**
- Convergence: **2,500-3,500 queries**

**UCB1**:
- Cumulative Regret: **Low** (near-optimal)
- Cost Savings: **40-46%** vs AlwaysBest
- Quality Maintained: **93-95%**
- Convergence: **1,500-2,500 queries** (fastest)

**Epsilon-Greedy**:
- Cumulative Regret: **Medium** (suboptimal)
- Cost Savings: **35-42%** vs AlwaysBest
- Quality Maintained: **91-94%**
- Convergence: **4,000-6,000 queries** (slowest)

**Random**:
- Cumulative Regret: **High**
- Cost Savings: **15-25%** (not optimal)
- Quality Maintained: **85-88%**
- Convergence: **Never**

**Oracle**:
- Cumulative Regret: **0** (by definition)
- Cost Savings: **50-55%** (theoretical maximum)
- Quality Maintained: **98%+**
- Note: Requires running all 17 models (170,000 LLM calls!)

### Visualizations

1. **Regret Curves**: Cumulative regret over 10,000 queries
   - Thompson and UCB converge quickly
   - Epsilon-Greedy slower but eventually stabilizes
   - Random grows linearly (no learning)

2. **Cost-Quality Scatter**: Pareto frontier
   - Oracle at top-right (perfect)
   - Thompson/UCB near Oracle
   - Epsilon-Greedy slightly worse
   - Random far from optimal

3. **Model Selection Heatmap**: Category × Model usage
   - Technical queries → GPT-4o, Claude-3.5-Sonnet
   - Creative writing → GPT-4o, Claude-3-Opus
   - Simple queries → GPT-4o-mini, Gemini-Flash
   - Math/Code → GPT-4o, Claude-3.5-Sonnet

4. **Convergence Detection**: When does performance stabilize?
   - UCB: ~2,000 queries
   - Thompson: ~3,000 queries
   - Epsilon-Greedy: ~5,000 queries

---

## Metrics

### Cumulative Regret
**Definition**: Total cost above optimal (Oracle) strategy
**Formula**: Σ(cost_actual - cost_optimal)
**Interpretation**: Lower is better, 0 is perfect

### Cost Savings
**Definition**: Cost reduction vs AlwaysBest baseline
**Formula**: (cost_always_best - cost_actual) / cost_always_best
**Target**: 40-50% savings

### Quality Maintained
**Definition**: Percentage of queries meeting quality threshold (0.85)
**Formula**: queries_above_threshold / total_queries
**Target**: 95%+ maintained

### Convergence Speed
**Definition**: Number of queries until performance stabilizes
**Method**: Detect when moving average plateaus
**Interpretation**: Faster = more efficient learning

---

## Repository Structure

```
conduit-bench/
├── conduit_bench/
│   ├── algorithms/              # Bandit algorithms (NEW)
│   │   ├── base.py              # Base classes and interfaces
│   │   ├── thompson_sampling.py # Bayesian approach
│   │   ├── ucb.py               # Upper Confidence Bound
│   │   ├── epsilon_greedy.py    # Simple exploration
│   │   └── baselines.py         # Random, Oracle, Always-*
│   ├── models/                  # Model registry (NEW)
│   │   └── registry.py          # 17 models with pricing
│   ├── generators/              # Query generation
│   │   └── synthetic.py         # Generate diverse queries
│   ├── runners/                 # Execution
│   │   ├── model_executor.py    # Direct PydanticAI calls
│   │   └── benchmark_runner.py  # Algorithm comparison
│   ├── analysis/                # Results analysis
│   │   ├── metrics.py           # Regret, cost, quality
│   │   └── visualize.py         # Charts and plots
│   └── cli.py                   # Command-line interface
├── data/                        # Datasets (git-ignored)
│   └── queries_10000.jsonl      # Generated queries
├── results/                     # Results (git-ignored)
│   └── experiment_001/          # Per-experiment results
├── tests/                       # Tests
├── AGENTS.md                    # AI agent guide
├── README.md                    # This file
└── pyproject.toml               # Dependencies
```

---

## Development

```bash
# Run tests
poetry run pytest

# Type checking
poetry run mypy conduit_bench/

# Linting
poetry run ruff check conduit_bench/

# Formatting
poetry run black conduit_bench/
```

---

## Related Projects

- **[Conduit](https://github.com/yourusername/conduit)**: ML-powered LLM routing (Thompson Sampling implementation)
- **[Arbiter](https://github.com/yourusername/arbiter)**: LLM evaluation framework (used for quality scoring)
- **[Loom](https://github.com/yourusername/loom)**: AI pipeline orchestration (optional for production)

---

**Research Goal**: Identify the optimal bandit algorithm for LLM routing across multiple providers, achieving 40-50% cost savings while maintaining 95%+ quality 🎯
