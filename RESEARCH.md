# Research Methodology - Conduit Bench

**Purpose**: Research design and experimental methodology for bandit algorithm benchmarking
**Last Updated**: 2025-01-22

---

## Research Question

**Which bandit algorithm achieves the best cost/quality trade-off for LLM routing across multiple providers?**

---

## Experiment Design

### Single Experiment Structure (NOT 3 Rounds)

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

---

## Sample Size Requirements

### Main Experiment: 10,000 queries
- Per-Arm: 10,000 / 17 models ≈ 590 samples per model
- Per-Category: 10,000 / 10 categories = 1,000 per category
- Convergence: Sufficient for detection (algorithms stabilize in 2-5K queries)

### Quick Validation: 1,000 queries
- Per-Arm: ~59 samples per model (marginal but usable)
- Use Case: Rapid prototyping, parameter tuning

### Statistical Rigor: 10 independent runs
- Report: mean ± 95% confidence interval
- Total: 10,000 × 10 = 100,000 queries

### Minimum Viable Sample Sizes
- 17 models × 30 samples (CLT) = 510 queries
- Recommended: 17 × 100 = 1,700 queries
- Robust: 17 × 500 = 8,500 queries

---

## Algorithms Under Test

### Learning Algorithms (3)

**1. Thompson Sampling** (Bayesian probability matching)
- Regret: O(√(K × T × ln T)) - optimal
- Convergence: 2,500-3,500 queries
- Complexity: Medium (Beta sampling)

**2. UCB1** (Optimistic estimates)
- Regret: O(√(K × T × ln T)) - optimal
- Convergence: 1,500-2,500 queries (fastest)
- Complexity: Low (arithmetic only)

**3. Epsilon-Greedy** (Simple exploration)
- Regret: O(K × T^(2/3)) - suboptimal
- Convergence: 4,000-6,000 queries (slowest)
- Complexity: Very low

### Baselines (4)

**1. Random**: Lower bound (uniform random selection)
**2. Oracle**: Upper bound (perfect knowledge - zero regret)
**3. AlwaysBest**: Quality ceiling (always use Claude-3-Opus)
**4. AlwaysCheapest**: Cost floor (always use Llama-3.1-8B)

---

## Model Pool

**17 Models Across 6 Providers**:
- OpenAI (4): gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo
- Anthropic (3): claude-3-5-sonnet, claude-3-opus, claude-3-haiku
- Google (3): gemini-1.5-pro, gemini-1.5-flash, gemini-1.0-pro
- Groq (3): llama-3.1-70b, llama-3.1-8b, mixtral-8x7b
- Mistral (3): mistral-large, mistral-medium, mistral-small
- Cohere (2): command-r-plus, command-r

**Pricing Range**: $0.00005 - $0.075 per 1K tokens (1,500× difference!)
**Quality Range**: 0.72 - 0.97 (25% difference)

---

## Expected Results

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

## Evaluation Methodology

### Quality Assessment
- Arbiter semantic evaluator for quality scoring
- Reference answers for each query category
- Threshold: 0.8 for "acceptable" quality

### Cost Calculation
- Actual API pricing from LLM providers
- Token counting for input + output
- Total cost per algorithm run

### Metrics

**1. Cumulative Regret**
```
regret_t = Σ(optimal_reward - actual_reward)
```

**2. Cost Savings**
```
savings = (AlwaysBest_cost - algorithm_cost) / AlwaysBest_cost
```

**3. Quality Maintained**
```
quality_rate = (queries_above_threshold) / total_queries
```

**4. Convergence Speed**
```
convergence_query = first query where regret_rate < 0.05
```

---

## Statistical Analysis

### Comparison Tests
- Paired t-tests between algorithms
- 95% confidence intervals
- Effect size (Cohen's d)
- Multiple comparison correction (Bonferroni)

### Visualization
- Cumulative regret over time
- Cost vs quality scatter plots
- Convergence curves
- Algorithm comparison tables

---

## Experimental Controls

### Fixed Variables
- Same query dataset for all algorithms
- Same model pool (17 models)
- Same quality threshold (0.8)
- Same random seed per run

### Independent Variables
- Algorithm type
- Algorithm hyperparameters (epsilon, c, alpha)

### Dependent Variables
- Cumulative regret
- Total cost
- Quality scores
- Convergence speed

---

## Related Documents
- [AGENTS.md](AGENTS.md) - Development guidelines
- [README.md](README.md) - User documentation
- [../conduit/](../conduit/) - Algorithm implementations
- [../arbiter/](../arbiter/) - Evaluation framework

---

**Last Updated**: 2025-01-22
