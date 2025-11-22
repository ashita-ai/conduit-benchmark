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

### Overview by Use Case

| Sample Size | What You Can Prove | Confidence | Use Case | Cost |
|-------------|-------------------|------------|----------|------|
| **100** | Priors help early decisions | Low | Quick check | $5 |
| **500** | Bandits beat random | Medium | Proof of concept | $25 |
| **1,000** | Convergence starting (with priors) | Medium | Customer demo | $50 |
| **2,000** | Algorithm comparison | High | Blog post | $100 |
| **5,000** | Full convergence (with priors) | High | Product launch | $250 |
| **10,000** | Full convergence (no priors) | Very High | Research paper | $500 |

### Sample Size by Prior Strategy

#### Without Priors (Cold Start Baseline)

**Per-Model Requirements**:
- Minimum: 30 samples per model (Central Limit Theorem)
- Recommended: 100 samples per model (95% confidence intervals)
- Robust: 500 samples per model (low variance estimates)

**Total Requirements**:
```
Minimum: 30 × 17 models = 510 queries
Recommended: 100 × 17 = 1,700 queries
Robust: 500 × 17 = 8,500 queries
Convergence proof: 2× expected convergence = 5,000 queries
```

**Convergence Timeline**:
- Thompson Sampling: 2,500-3,500 queries
- UCB1: 1,500-2,500 queries
- Epsilon-Greedy: 4,000-6,000 queries

#### With Industry Priors (Recommended)

**Starting Knowledge**: Each model has ~1,000 samples from industry data
- Effective sample size: ~10,000 queries of pre-existing knowledge
- Equivalent to Beta(8500, 1500) for high-quality models

**New Samples Required**:
```
Proof priors help: 100-200 queries (show early guidance)
Show convergence: 500-1,000 queries (5× faster than no priors)
Validate long-term: 5,000 queries (same final performance)
```

**Convergence Timeline** (with priors):
- Thompson Sampling: 500 queries (5× faster)
- UCB1: 300 queries (5× faster)
- Epsilon-Greedy: 1,000 queries (4× faster)

**Key Insight**: Priors reduce sample requirements by **5-10×** for same confidence level

#### With Context-Specific Priors (code/creative/analysis/simple)

**Per-Context Requirements**:
- Minimum: 50 queries per context (early decisions)
- Recommended: 200 queries per context (convergence)
- Robust: 500 queries per context (statistical confidence)

**Total for 4 Contexts**:
```
Minimum: 50 × 4 = 200 queries
Recommended: 200 × 4 = 800 queries
Robust: 500 × 4 = 2,000 queries
```

**Note**: With context priors, each context converges independently at 200 queries (vs 500 with general priors)

#### With Customer Segment Priors (SaaS/legal/etc.)

**Starting Knowledge**: Segment priors from ~50 similar customers = 50,000 queries of knowledge

**New Samples Required**:
```
Single customer validation: 100-200 queries
Segment validation: 2,000 queries (10 customers × 200 each)
Cross-segment comparison: 10,000 queries (5 segments × 2,000 each)
```

**Convergence**: 100-150 queries per customer (10× faster than industry priors)

### Statistical Power Analysis

#### Hypothesis 1: "Bandits work better than random"

**Assumptions**:
- Random baseline quality: 0.85 (average across models)
- Thompson Sampling expected: 0.92 (best models)
- Effect size: 0.07 (7 percentage points)
- Significance: α = 0.05, Power: 1 - β = 0.80

**Sample size calculation**:
```
σ = 0.15  # Standard deviation of quality scores
δ = 0.07  # Effect size
n ≈ 2 × (1.96 + 0.84)² × 0.15² / 0.07²
n ≈ 72 queries per algorithm

With 10 runs: 720 queries minimum
Recommended: 1,000 queries
```

#### Hypothesis 2: "Priors reduce convergence time"

**Comparison**: With priors (500 convergence) vs without (2,500 convergence)

**Sample size**:
```
Run both to same length for fair comparison:
- Without priors: 5,000 queries (2× convergence time)
- With priors: 5,000 queries (same length)
- Clear difference visible in convergence curves

Minimum for proof: 1,000 queries (show convergence starting)
Recommended: 5,000 queries (show full convergence + stability)
```

### Main Experiment Design: 10,000 queries

**Per-Arm Distribution**:
- 10,000 / 17 models ≈ 590 samples per model
- Sufficient for convergence detection (2-5K queries)
- Robust statistical estimates

**Per-Category Distribution** (if using contexts):
- 10,000 / 4 contexts = 2,500 per context
- 2,500 / 17 models ≈ 147 per model per context
- Sufficient for context-specific learning

**Statistical Rigor: 10 independent runs**
- Report: mean ± 95% confidence interval
- Total: 10,000 × 10 = 100,000 queries
- Cost: ~$5,000 (for all algorithms + Oracle baseline)

### Quick Validation: 1,000 queries

**With Priors** (Recommended):
- Per-Arm: ~59 samples per model
- Use Case: Rapid validation, customer demos
- Convergence: Visible for Thompson/UCB1 with priors
- Cost: ~$50

**Without Priors**:
- Per-Arm: ~59 samples per model
- Use Case: Baseline comparison only
- Convergence: Not yet visible
- Cost: ~$60

### Cost Analysis

#### Without Priors (10,000 queries)
```
Oracle: 10,000 × 17 × $0.015 = $2,550
Thompson: 10,000 × $0.008 = $80
Random: 10,000 × $0.012 = $120

Total (3 algorithms × 10 runs): ~$27,500
```

#### With Industry Priors (1,000 queries)
```
Thompson with priors: 1,000 × $0.005 = $5
Thompson no priors: 1,000 × $0.008 = $8
Random: 1,000 × $0.012 = $12

Total (3 algorithms × 10 runs): ~$250

Cost reduction: 100× cheaper for same statistical confidence!
```

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
- Prior strategy (none, industry, context, segment)

### Dependent Variables
- Cumulative regret
- Total cost
- Quality scores
- Convergence speed

---

## Three-Phase Experimental Design

### Phase 1: Quick Proof (Week 1)

**Goal**: Prove bandits work and priors help

**Configuration**:
```python
experiment_1 = {
    "queries": 500,
    "runs": 3,
    "algorithms": [
        "thompson_with_industry_priors",
        "random"
    ],
    "evaluation": "arbiter_semantic",
    "cost_estimate": "$25",
    "execution_time": "1 hour",
}
```

**What You Can Prove**:
- ✅ Thompson Sampling > Random (statistical significance)
- ✅ Priors improve early performance
- ✅ System works end-to-end
- ⚠️ NOT enough for convergence proof
- ⚠️ NOT enough for algorithm comparison

**Deliverable**: Internal demo, proof of concept

**Success Criteria**:
- Thompson quality > Random quality (p < 0.05)
- Thompson cost < Random cost
- First 100 queries show learning (quality improving)

---

### Phase 2: Solid Validation (Week 2-3)

**Goal**: Show convergence, compare algorithms, validate priors

**Configuration**:
```python
experiment_2 = {
    "queries": 2000,
    "runs": 5,
    "algorithms": [
        "thompson_no_priors",
        "thompson_industry_priors",
        "thompson_context_priors",
        "ucb1",
        "epsilon_greedy",
        "random"
    ],
    "evaluation": "arbiter_semantic",
    "cost_estimate": "$150-200",
    "execution_time": "3-4 hours",
}
```

**What You Can Prove**:
- ✅ Algorithm comparison (Thompson vs UCB1 vs Epsilon-Greedy)
- ✅ Prior comparison (none vs industry vs context)
- ✅ Convergence starting (visible in regret curves)
- ✅ Cost/quality trade-offs quantified
- ✅ Context-specific priors outperform general priors

**Deliverable**: Customer demo, blog post, product marketing

**Success Criteria**:
- Thompson/UCB1 converge by query 1,000 (with priors)
- Context priors show 2× faster convergence vs industry priors
- Cost savings: 40-45% vs AlwaysBest baseline
- Quality maintained: >90% above threshold

**Visualizations**:
- Cumulative regret over time (all algorithms)
- Convergence comparison (no priors vs industry vs context)
- Cost vs quality scatter plots
- Per-context performance breakdown

---

### Phase 3: Research Quality (Week 4)

**Goal**: Publication-ready results, statistical rigor

**Configuration**:
```python
experiment_3 = {
    "queries": 10000,
    "runs": 10,
    "algorithms": [
        # Learning algorithms
        "thompson_no_priors",
        "thompson_industry_priors",
        "thompson_context_priors",
        "ucb1_no_priors",
        "ucb1_context_priors",
        "epsilon_greedy",
        "linucb",

        # Baselines
        "random",
        "oracle",
        "always_best",
        "always_cheapest"
    ],
    "evaluation": "arbiter_semantic",
    "cost_estimate": "$500-1,000",
    "execution_time": "1 day",
}
```

**What You Can Prove**:
- ✅ Full convergence (all algorithms reach plateau)
- ✅ Statistical significance (p < 0.05, Bonferroni corrected)
- ✅ Effect sizes (Cohen's d)
- ✅ Regret bounds match theory
- ✅ Prior effectiveness quantified (5-10× sample reduction)
- ✅ Cost/quality Pareto frontier
- ✅ Ready for academic publication

**Deliverable**: Research paper, arXiv preprint, Hacker News launch

**Success Criteria**:
- Thompson/UCB1: O(√(K × T × ln T)) regret (matches theory)
- Epsilon-Greedy: O(K × T^(2/3)) regret (suboptimal as expected)
- Oracle establishes upper bound (~50-55% cost savings)
- All algorithms statistically distinguishable (p < 0.05)
- Priors reduce convergence: 2,500 → 500 queries (5× improvement)

**Statistical Tests**:
- Paired t-tests between all algorithm pairs
- 95% confidence intervals on all metrics
- Effect size (Cohen's d) for each comparison
- Multiple comparison correction (Bonferroni)
- Power analysis (achieved power > 0.80)

**Visualizations**:
- Cumulative regret curves (all 11 algorithms)
- Convergence speed comparison (with/without priors)
- Cost vs quality Pareto frontier
- Per-algorithm confidence intervals
- Regret rate over time (should approach 0)
- Per-context learning curves (for context priors)
- Algorithm comparison heatmap (win/loss matrix)

---

## Progressive Validation Strategy

### Recommended Approach: Build Progressively

**Week 1**: Phase 1 (500 queries)
- Prove basic functionality
- Validate infrastructure
- Identify issues early
- Cost: $25

**Week 2**: Phase 2 if Phase 1 successful (2,000 queries)
- Expand algorithm coverage
- Test prior strategies
- Generate marketing materials
- Cost: $150-200

**Week 3**: Analyze Phase 2, prepare Phase 3

**Week 4**: Phase 3 if publishing (10,000 queries)
- Full statistical rigor
- Publication-ready results
- Cost: $500-1,000

**Total Investment**: $700-1,200 over 4 weeks

**Risk Mitigation**:
- Don't commit to Phase 3 until Phase 2 shows promise
- Can stop after Phase 2 for product validation
- Phase 3 only needed for academic publication or HN launch

---

## Related Documents
- [AGENTS.md](AGENTS.md) - Development guidelines
- [README.md](README.md) - User documentation
- [../conduit/](../conduit/) - Algorithm implementations
- [../arbiter/](../arbiter/) - Evaluation framework

---

**Last Updated**: 2025-01-22
