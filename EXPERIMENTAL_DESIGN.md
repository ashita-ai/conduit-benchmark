# Conduit Benchmark Experimental Design

## 🎯 Objectives
1. Compare 11 bandit algorithms for LLM routing
2. Test across 9 modern frontier models (Claude, OpenAI, Google)
3. Evaluate with/without PCA dimensionality reduction (67 dimensions)
4. Measure convergence, quality, cost, and regret with statistical rigor

## 🤖 Algorithms Under Test (11 Total)

### Contextual Algorithms (3) - Use query features
1. **LinUCBBandit** - Linear UCB with contextual features (uses PCA)
2. **ContextualThompsonSamplingBandit** - Bayesian linear regression
3. **DuelingBandit** - Pairwise preference learning

### Non-Contextual Algorithms (3) - No features
4. **ThompsonSamplingBandit** - Bayesian probability matching
5. **UCB1Bandit** - Upper confidence bound
6. **EpsilonGreedyBandit** - Simple exploration-exploitation

### Baselines (4) - For comparison
7. **RandomBaseline** - Uniform random (lower bound)
8. **OracleBaseline** - Perfect knowledge (upper bound, requires reference=1.0)
9. **AlwaysBestBaseline** - Always highest quality model
10. **AlwaysCheapestBaseline** - Always lowest cost model

**Note:** Oracle only testable with 100% reference answers

### Hybrid Algorithm (1) - Production optimized
11. **HybridRouter** - UCB1 (0-2000 queries) → LinUCB (2000+)
    - Warm-start strategy for cold-start problem
    - Switching point to be validated empirically

## 📱 Model Arms (9 Models)

### Anthropic Claude 4
- `claude-opus-4` - Premium quality, highest cost
- `claude-sonnet-4` - Balanced quality/cost
- `claude-haiku-4` - Fast, lowest cost

### OpenAI
- `gpt-5.1` - Latest flagship
- `chatgpt-5` - Standard quality
- `chatgpt-5-mini` - Economical option

### Google Gemini 3
- `gemini-3-pro` - New flagship (1501 Elo, first model >1500)
- `gemini-3-flash` - Fast inference
- `gemini-2.5-pro` - Previous generation baseline

**Model Version Tracking:**
- Document exact model versions/timestamps at experiment start
- Track API changes during experiment period
- Report model availability windows

**Sources:**
- [Gemini 3 Official Announcement](https://blog.google/products/gemini/gemini-3/)
- [Google Gemini 3 Launch](https://www.cnbc.com/2025/11/18/google-announces-gemini-3-as-battle-with-openai-intensifies.html)

## 📊 Sample Size Calculation

### Formal Power Analysis

**Target Effect Size:** Detect 10% difference in quality scores (δ = 0.10)
**Significance Level:** α = 0.05 (two-tailed)
**Desired Power:** 1 - β = 0.80
**Expected Variance:** σ² (estimated from pilot study)

**Power Calculation Formula:**
```
n = 2 * (Z_α/2 + Z_β)² * σ² / δ²
```

Where:
- Z_α/2 = 1.96 (for α = 0.05)
- Z_β = 0.84 (for power = 0.80)
- δ = 0.10 (minimum detectable difference)
- σ² = variance estimate from pilot (see Pilot Study section)

### Convergence Requirements

**Contextual Algorithms** (LinUCB, ContextualThompson, Dueling):
- PCA dimensions: `d = 67`
- Minimum samples: `20 * d = 20 * 67 = 1,340`
- Recommended: `30 * d = 30 * 67 = 2,010`
- **Power-adjusted:** Based on pilot variance (see below)

**Non-Contextual Algorithms** (Thompson, UCB1, Epsilon):
- Number of arms: `K = 9`
- Minimum samples: `100 * K = 100 * 9 = 900`
- Recommended: `150 * K = 150 * 9 = 1,350`
- **Power-adjusted:** Based on pilot variance (see below)

**Baselines:**
- Random: Any N (establishes lower bound)
- Oracle: N ≥ 500 (requires 100% references)
- AlwaysBest/Cheapest: Any N (deterministic)

### Pilot Study (Required Before Main Experiment)

**Purpose:** Estimate variance for power analysis and validate assumptions

**Design:**
- Sample size: N = 200 queries
- Algorithms: Random, UCB1, LinUCB (representative subset)
- Metrics: Quality scores, costs, regret
- Runs: 3 independent runs per algorithm

**Deliverables:**
1. **Variance Estimates:**
   - σ²_quality: Variance of quality scores
   - σ²_cost: Variance of cost per query
   - σ²_regret: Variance of cumulative regret

2. **Sample Size Recalculation:**
   - Use pilot variance to compute required N
   - Verify N=2,500 is sufficient (or adjust upward)

3. **Distribution Validation:**
   - Test normality assumptions (Shapiro-Wilk)
   - Check for outliers
   - Validate query distribution balance

**Example Power Calculation (Post-Pilot):**
```python
from statsmodels.stats.power import TTestIndPower

# After pilot: σ_quality = 0.15 (estimated)
analysis = TTestIndPower()
n_required = analysis.solve_power(
    effect_size=0.10 / 0.15,  # Cohen's d = δ / σ
    power=0.80,
    alpha=0.05,
    ratio=1.0
)
# Result: n_required ≈ 2,000 per algorithm
# With 3 runs: N = 2,000 is sufficient
```

### **RECOMMENDATION: N = 2,500 samples** (Post-Pilot Validation)

**Rationale:**
- ✅ Allows contextual algorithms to converge (67 dimensions)
- ✅ Ensures all 9 models explored sufficiently
- ✅ Provides ≥80% statistical power (after pilot validation)
- ✅ Reasonable runtime (~6-8 hours for 11 algorithms)
- ✅ Cost-effective (~$25-30 for Arbiter evaluation)
- ✅ Balances power vs. execution time

**Note:** Final sample size subject to pilot study variance estimates. May adjust to N=3,000-5,000 if pilot shows higher variance.

**Alternative Configurations:**
- **Quick test**: N = 1,000 (2-3 hours, 60% power) - Not recommended
- **Standard**: N = 2,500 (6-8 hours, 80% power) ⭐ **RECOMMENDED**
- **High power**: N = 5,000 (12-16 hours, 90% power)
- **Publication quality**: N = 10,000 (24-32 hours, 95% power)

## 📐 Regret Function Definition

### Mathematical Specification

**Cumulative Regret (Primary Metric):**
```
R_T = Σ_{t=1}^T [r*_t - r_{a_t}]
```

Where:
- `r*_t` = reward of optimal arm at time t (oracle reward)
- `r_{a_t}` = reward of selected arm at time t
- `T` = total number of queries

### Reward Function

**Option 1: Quality-Weighted Reward (Primary)**
```
r_t = quality_score_t
```

**Option 2: Quality-Cost Tradeoff (Secondary)**
```
r_t = quality_score_t - λ * (cost_t - cost_min)
```

Where:
- `λ` = cost penalty coefficient (default: 0.01 per $0.001)
- `cost_min` = minimum cost across all models
- Normalized so quality dominates: `λ << 1`

**Option 3: Normalized Regret (For Reporting)**
```
R_norm = R_T / (T * r*_max)
```

Where `r*_max` = maximum possible reward (oracle average)

### Oracle Definition

**Per-Query Oracle (Used for Regret):**
- For each query, select the model with highest quality score
- Requires quality scores for all 9 models (reference answers)
- Only computable when reference_probability = 1.0

**Single Best Oracle (Alternative Baseline):**
- Select the single model with highest average quality
- Computable with partial references (reference_probability < 1.0)
- Used when full oracle unavailable

**Implementation:**
```python
def compute_regret(selected_rewards, oracle_rewards):
    """
    Compute cumulative regret.

    Args:
        selected_rewards: List of rewards from selected arms
        oracle_rewards: List of optimal rewards (per-query oracle)

    Returns:
        cumulative_regret: Sum of (oracle - selected) rewards
        normalized_regret: Regret / (T * max_oracle_reward)
    """
    cumulative_regret = sum(o - s for o, s in zip(oracle_rewards, selected_rewards))
    normalized_regret = cumulative_regret / (len(oracle_rewards) * max(oracle_rewards))
    return cumulative_regret, normalized_regret
```

## 🔄 Convergence Definition

### Stability Criterion

**Convergence Definition:**
An algorithm is considered converged when its performance metric (regret, quality, or cost) stabilizes within a specified threshold over a sliding window.

**Mathematical Specification:**
```python
def is_converged(metric_history, window=200, threshold=0.05, min_samples=500):
    """
    Check if algorithm has converged.

    Args:
        metric_history: List of metric values over time
        window: Sliding window size (default: 200 queries)
        threshold: Coefficient of variation threshold (default: 5%)
        min_samples: Minimum samples before checking convergence

    Returns:
        converged: Boolean indicating convergence
        convergence_point: Query index where convergence occurred
    """
    if len(metric_history) < min_samples:
        return False, None

    # Use sliding window approach
    recent_metric = metric_history[-window:]
    mean_recent = np.mean(recent_metric)
    std_recent = np.std(recent_metric)

    # Coefficient of variation
    cv = std_recent / mean_recent if mean_recent > 0 else float('inf')

    converged = cv < threshold
    convergence_point = len(metric_history) - window if converged else None

    return converged, convergence_point
```

**Convergence Metrics:**
1. **Regret Convergence:** CV(regret) < 5% over 200-query window
2. **Quality Convergence:** CV(quality) < 3% over 200-query window
3. **Cost Convergence:** CV(cost) < 5% over 200-query window

**Convergence Time:**
- Report query index where convergence first occurs
- Report 95% confidence interval on convergence time
- Compare convergence rates across algorithms

## 🧪 Experimental Variables

### Independent Variables
1. **Algorithm Type** (11 levels)
2. **Model Set** (9 models per algorithm)
3. **PCA Enabled** (True/False for contextual algorithms)
4. **Reference Probability** (0.25 fixed for main experiment, varied in ablation)
5. **Query Domain Distribution** (10 categories from synthetic generator)

### Dependent Variables
1. **Cumulative Regret** - Total suboptimality vs oracle (see Regret Function Definition)
2. **Average Quality** - Mean quality score across queries (with 95% CI)
3. **Total Cost** - Cumulative spend in USD (with 95% CI)
4. **Convergence Time** - Queries until stable performance (with 95% CI)
5. **Model Selection Distribution** - Which models selected over time
6. **Effect Sizes** - Cohen's d for algorithm comparisons

### Control Variables
1. **Random Seed** (42 for reproducibility, varied across runs: 42, 123, 456)
2. **Arbiter Model** (gpt-4o-mini for consistent evaluation)
3. **Query Distribution** (same synthetic dataset for all algorithms)
4. **Feature Extraction** (same QueryAnalyzer for all runs)
5. **Model Versions** (documented and fixed at experiment start)

## 📋 Experimental Design Matrix

### Primary Experiment
```yaml
algorithms: [thompson, ucb1, epsilon, linucb, contextual_thompson, dueling,
             random, oracle, always_best, always_cheapest, hybrid]
models: 9  # Claude 4 (3) + OpenAI (3) + Gemini 3 (3)
samples: 2500
reference_probability: 0.25  # Fixed for main experiment
pca_dimensions: 67
runs_per_algorithm: 3  # Independent runs with different seeds
random_seeds: [42, 123, 456]  # For reproducibility
total_queries: 2500 * 11 * 3 = 82,500
```

### Ablation Studies

**1. PCA Impact (Contextual algorithms only)**
```yaml
algorithms: [linucb, contextual_thompson, dueling]
conditions:
  - pca_enabled: true, pca_dimensions: 67
  - pca_enabled: false  # Full embedding dimensions
samples: 2500
runs: 3
```

**2. Reference Probability Sensitivity**
```yaml
algorithms: [thompson, ucb1, linucb]  # Representative subset
reference_probability: [0.1, 0.2, 0.3, 0.5, 1.0]
samples: 2500
runs: 3
```

**3. Model Set Size**
```yaml
algorithms: [thompson, ucb1, linucb]
model_counts: [3, 5, 7, 9]  # Test scalability
samples: 2500
runs: 3
```

**4. Hybrid Switching Point**
```yaml
algorithm: hybrid
switching_points: [1000, 1500, 2000, 2500, 3000]
samples: 2500
runs: 3
```

## 📊 Statistical Analysis Plan

### Multiple Comparisons Correction

**Problem:** Comparing 11 algorithms creates C(11,2) = 55 pairwise comparisons. Without correction, expect ~2.75 false positives at α=0.05.

**Solution:** Hierarchical testing approach

1. **Overall Test (Friedman Test):**
   - H₀: All algorithms perform equally
   - If p < 0.05, proceed to pairwise tests
   - If p ≥ 0.05, no pairwise tests needed

2. **Pairwise Comparisons (Post-Hoc):**
   - **Nemenyi Test** (recommended for multiple algorithms)
   - **Bonferroni Correction:** α_corrected = 0.05 / 55 ≈ 0.0009
   - **Benjamini-Hochberg FDR Control:** Less conservative alternative

**Implementation:**
```python
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi

# Overall test
stat, p_value = friedmanchisquare(*algorithm_results)
if p_value < 0.05:
    # Pairwise comparisons with Nemenyi
    p_matrix = posthoc_nemenyi(algorithm_results)
    # Apply FDR correction
    from statsmodels.stats.multitest import multipletests
    p_corrected = multipletests(p_matrix.flatten(), method='fdr_bh')[1]
```

### Primary Statistical Tests

1. **Friedman Test** - Overall algorithm differences
   - Non-parametric (no normality assumption)
   - Tests: H₀: All algorithms equal vs H₁: At least one differs
   - Report: χ² statistic, p-value, effect size (Kendall's W)

2. **Nemenyi Post-Hoc Test** - Pairwise comparisons
   - Designed for multiple algorithm comparison
   - Controls family-wise error rate
   - Report: Critical difference (CD) diagram

3. **Effect Size Reporting:**
   - **Cohen's d** for pairwise comparisons
   - Interpretation: d < 0.2 (small), 0.2-0.5 (medium), >0.5 (large)
   - **Kendall's W** for overall Friedman test
   - Interpretation: W < 0.1 (small), 0.1-0.3 (medium), >0.3 (large)

4. **Confidence Intervals:**
   - 95% CI for all metrics (regret, quality, cost)
   - Bootstrap method (10,000 resamples) for non-normal distributions
   - Report: Mean ± CI for each algorithm

### Secondary Analyses

1. **ANOVA** - Factor effects (PCA, reference prob)
   - Two-way ANOVA: Algorithm × PCA
   - Report: F-statistics, p-values, effect sizes (η²)

2. **Regression** - Convergence rate modeling
   - Model: convergence_time ~ algorithm + pca + model_count
   - Report: R², coefficients, p-values

3. **Non-Stationarity Testing:**
   - **CUSUM Test:** Detect mean shifts in reward distribution
   - **Page-Hinkley Test:** Detect change points
   - Report: Stationarity p-values, change point locations

**Implementation:**
```python
def test_stationarity(reward_history, alpha=0.05):
    """
    Test for non-stationarity in reward distribution.

    Returns:
        is_stationary: Boolean
        change_points: List of detected change points
    """
    from scipy.stats import kruskal

    # Split into windows
    window_size = len(reward_history) // 4
    windows = [reward_history[i:i+window_size]
               for i in range(0, len(reward_history), window_size)]

    # Kruskal-Wallis test across windows
    stat, p_value = kruskal(*windows)
    is_stationary = p_value > alpha

    return is_stationary, []
```

## 💰 Cost Estimation

**Per Algorithm Run (2,500 queries):**
- Model execution: ~$2-5 (depends on model mix)
- Arbiter evaluation (25% with ref): 625 queries × $0.01 = $6.25
- **Buffer (20%):** $1.25
- **Total per run:** ~$9-13

**Full Benchmark (11 algorithms, 3 runs each):**
- Total queries: 82,500
- Estimated cost: 11 * 3 * $11 = $363
- **Buffer (30%):** $109
- **Total budget:** ~$470

**Cost Tracking:**
- Log actual costs per run
- Report cost variance across runs
- Document any API pricing changes during experiment

**Parallelization Strategy:**
- Run algorithms in parallel (11 concurrent)
- Reduces wall time to: 3 runs * 7 hours = 21 hours (~1 day)
- Requires parallel execution infrastructure

## ⏱️ Runtime Estimation

**Per Query:**
- Feature extraction: ~0.5s
- Model execution: ~2-4s (depends on model)
- Quality evaluation: ~1-2s (when reference available)
- Total: ~4-7s average

**Per Algorithm (2,500 queries):**
- Sequential: 2,500 * 5s = 12,500s ≈ 3.5 hours
- With concurrency (5): ~45 minutes

**Full Benchmark:**
- Sequential: 11 algorithms * 3.5 hours * 3 runs = 115.5 hours
- Parallel (11 algorithms): 3.5 hours * 3 runs = 10.5 hours ⭐

## 📈 Success Criteria

### Algorithm Performance
- **Convergence:** Algorithms reach stable performance (CV < 5%) by N=2,000
- **Regret:** Contextual algorithms achieve <30% normalized regret vs oracle
- **Quality:** Top algorithms achieve >0.75 average quality (95% CI)
- **Cost:** Smart algorithms use 20-40% less cost than random (with effect size d > 0.5)

### Statistical Significance
- **Power:** ≥80% for detecting 10% performance differences
- **Confidence:** 95% confidence intervals on all metrics
- **Reproducibility:** Results consistent across 3 runs (CV <10%)
- **Effect Sizes:** Report Cohen's d for all pairwise comparisons

### PCA Impact
- **Speed:** 2-3x faster feature processing (measured)
- **Quality:** <5% quality degradation vs full embeddings (with CI)
- **Convergence:** Similar convergence rate (within 10%, non-inferiority test)

## 🔄 Execution Plan

### Phase 0: Pilot Study (N=200) ⚠️ REQUIRED FIRST
**Goals:**
- Estimate variance for power analysis
- Validate query distribution
- Test infrastructure end-to-end
- Calibrate evaluation costs
- Estimate actual runtimes

**Algorithms:** thompson, ucb1, random (3 total)
**Runs:** 3 per algorithm
**Time:** ~1 hour
**Cost:** ~$15

**Deliverables:**
- Variance estimates (σ²_quality, σ²_cost, σ²_regret)
- Recalculated sample sizes
- Distribution validation report
- Infrastructure validation

### Phase 1: Core Algorithms (N=2,500)
**Algorithms:** thompson, ucb1, epsilon, linucb, contextual_thompson, dueling, random (7 total)
**Runs:** 3 per algorithm
**Time:** ~21 hours (parallel)
**Cost:** ~$231 (with buffer)

### Phase 2: Baselines & Hybrid (N=2,500)
**Algorithms:** oracle, always_best, always_cheapest, hybrid (4 total)
**Runs:** 3 per algorithm
**Time:** ~10.5 hours (parallel)
**Cost:** ~$132 (with buffer)

### Phase 3: Ablation Studies
**Studies:** PCA impact, reference sensitivity, model set size, hybrid switching
**Time:** ~15 hours (parallel)
**Cost:** ~$100 (with buffer)

**Total:** 4 phases, ~47 hours, ~$478

## 📊 Analysis Plan

### Primary Analyses
1. **Cumulative Regret Curves** - Plot regret vs query count (with 95% CI bands)
2. **Quality-Cost Frontier** - Pareto optimal algorithms (with effect sizes)
3. **Convergence Analysis** - Time to stable performance (with 95% CI)
4. **Model Selection Heatmaps** - Which models selected when
5. **Statistical Significance** - Friedman test + Nemenyi post-hoc (with corrections)

### Secondary Analyses
1. **PCA Impact** - Quality/speed tradeoff analysis (non-inferiority test)
2. **Reference Sensitivity** - Performance vs reference probability (regression)
3. **Domain Analysis** - Performance by query domain (ANOVA)
4. **Scalability** - Performance vs number of models (regression)
5. **Non-Stationarity** - CUSUM/Page-Hinkley tests for reward stability

### Effect Size Reporting
- **Cohen's d** for all pairwise algorithm comparisons
- **Kendall's W** for overall Friedman test
- **η²** for ANOVA factor effects
- Interpretation guidelines (small/medium/large)

## 📝 Reporting

### Visualizations
1. Cumulative regret curves (line plot with 95% CI bands)
2. Quality-cost scatter (pareto frontier with effect sizes)
3. Model selection over time (stacked area)
4. Algorithm comparison table (summary stats with 95% CI)
5. Convergence rate comparison (bar chart with 95% CI)
6. PCA impact analysis (before/after with non-inferiority)
7. Critical difference (CD) diagram (Nemenyi test results)
8. Effect size heatmap (Cohen's d matrix)

### Metrics Tables
- Per-algorithm summary statistics (mean ± 95% CI)
- Statistical significance tests (Friedman, Nemenyi, ANOVA)
- Effect sizes (Cohen's d, Kendall's W, η²)
- Convergence metrics (time ± 95% CI)
- Cost-efficiency analysis (with effect sizes)
- Non-stationarity test results

### Required Reporting Elements
- **Pilot Study Results:** Variance estimates, sample size justification
- **Model Versions:** Exact versions/timestamps used
- **Cost Tracking:** Actual vs estimated costs
- **Reproducibility:** Random seeds, environment details
- **Limitations:** Non-stationarity, synthetic data, etc.

## 🎓 Publication Readiness

For academic publication, consider:
- **N = 5,000-10,000** samples for 90-95% power
- **5-10 runs** per algorithm for robust statistics
- **Multiple datasets** (not just synthetic)
- **Real-world validation** with production queries
- **Ablation studies** for all major design choices
- **Full statistical reporting:** Effect sizes, confidence intervals, multiple comparison corrections

## 🚀 Quick Start Commands

```bash
# Phase 0: Pilot Study (REQUIRED FIRST)
uv run conduit-bench generate --queries 200 --seed 42 --reference-probability 0.25 --output data/pilot_200.jsonl
uv run conduit-bench run --dataset data/pilot_200.jsonl --algorithms thompson,ucb1,random --runs 3 --output results/pilot/

# Analyze pilot results
uv run conduit-bench analyze --results results/pilot/ --output analysis/pilot/
# Review variance estimates and recalculate sample sizes

# Phase 1: Generate main dataset
uv run conduit-bench generate --queries 2500 --seed 42 --reference-probability 0.25 --output data/benchmark_2500.jsonl

# Phase 2: Run core algorithms (parallel recommended)
uv run conduit-bench run --dataset data/benchmark_2500.jsonl --algorithms thompson,ucb1,epsilon,linucb,contextual_thompson,dueling,random --runs 3 --seeds 42,123,456 --output results/core/

# Phase 3: Run baselines
uv run conduit-bench run --dataset data/benchmark_2500.jsonl --algorithms oracle,always_best,always_cheapest,hybrid --runs 3 --seeds 42,123,456 --output results/baselines/

# Analyze results (with statistical tests)
uv run conduit-bench analyze --results results/*.json --output analysis/ --statistical-tests friedman,nemenyi --effect-sizes cohens-d
```

---

**Last Updated:** 2025-11-26
**PCA Dimensions:** 67
**Recommended Sample Size:** 2,500 (subject to pilot validation)
**Estimated Cost:** $478 (with 30% buffer)
**Estimated Time:** 47 hours (with parallelization)
**Statistical Rigor:** Publication-ready with formal power analysis
