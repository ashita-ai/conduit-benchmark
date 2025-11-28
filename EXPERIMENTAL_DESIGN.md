# Conduit Benchmark Experimental Design

## 🎯 Objectives

1. **Validate Thompson Sampling as optimal default** - research shows Thompson > LinUCB/hybrids for LLM routing
2. **Use objective evaluation** - exact-match and code execution, NOT LLM-as-judge
3. **Benchmark on established datasets** - GSM8K, MMLU, HumanEval (credible, reproducible)
4. Measure accuracy, cost, latency across multiple domains
5. Compare Thompson against baselines (UCB1, Epsilon-Greedy, Random)

**Research Context**: Recent findings (BayesianRouter, arXiv 2510.02850) show Thompson Sampling outperforms LinUCB and hybrid routing for LLM domains. This benchmark validates those findings and supports changing conduit's default from HybridRouter to pure Thompson Sampling (GitHub Issue #169).

## 📊 Benchmark Suite (3 Datasets)

### Overview

| Dataset | Size | Domain | Evaluation | Est. Cost | Headline |
|---------|------|--------|------------|-----------|----------|
| **GSM8K** | 1,319 | Math reasoning | Exact match (`#### N`) | ~$100-150 | "85% accuracy at 40% cost" |
| **MMLU** | 1,000 | Knowledge (57 subjects) | Exact match (A/B/C/D) | ~$80-100 | "Matches GPT-4 at 50% cost" |
| **HumanEval** | 164 | Python coding | Code execution | ~$20-30 | "75% pass rate, 60% savings" |
| **Total** | **2,483** | | | **~$200-300** | |

### Why These Datasets?

**GSM8K** (Grade School Math 8K)
- Source: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)
- 1,319 test problems with step-by-step solutions
- Answer format: `#### 72` - objectively correct/incorrect
- No LLM-as-judge needed - eliminates circular dependency

**MMLU** (Massive Multitask Language Understanding)
- Source: [cais/mmlu](https://huggingface.co/datasets/cais/mmlu)
- 14,042 questions across 57 subjects (using 1k subset)
- Multiple choice (A/B/C/D) - exact match on answer
- Broad coverage: STEM, humanities, social sciences

**HumanEval** (OpenAI Code Benchmark)
- Source: [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval)
- 164 Python function completion problems
- Execute code + run unit tests - pass/fail evaluation
- Most credible to developers - executable tests > LLM judges

## 🔧 Evaluation Methods

### Exact Match (GSM8K, MMLU)

**No LLM-as-judge.** Extract answer, compare to ground truth.

```python
# GSM8K: Extract "#### N" from response
def extract_gsm8k_answer(text: str) -> str | None:
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    return match.group(1).replace(',', '') if match else None

# MMLU: Extract A/B/C/D from response
def extract_mmlu_answer(text: str) -> str | None:
    match = re.search(r'\b([ABCD])\b', text.upper())
    return match.group(1) if match else None

# Evaluation: 1.0 if correct, 0.0 if incorrect
reward = 1.0 if predicted == expected else 0.0
```

### Code Execution (HumanEval)

**Execute code, run tests.** Pass/fail based on actual execution.

```python
# Combine prompt + model response + test cases
full_code = f"{prompt}{response}\n\n{test_code}\ncheck({entry_point})"

# Execute in sandboxed subprocess with timeout
result = subprocess.run(['python', temp_file], timeout=10, capture_output=True)
reward = 1.0 if result.returncode == 0 else 0.0
```

### Arbiter (Production Use Only)

Arbiter (LLM-as-judge) is still available for:
- Production routing where no ground truth exists
- Open-ended query quality evaluation
- A/B testing in deployment

**NOT used for benchmarking** due to circular dependency (LLM judging LLM).

## 🤖 Algorithms Under Test (4 Core Algorithms)

**Tier 1 - Core Validation** (Data: 2,483 queries)

### Primary Algorithm - Proposed Default ⭐
1. **ThompsonSamplingBandit** - Bayesian probability matching
   - **This is the algorithm we're validating as new default**
   - Works from query 1, converges in 100-500 queries
   - Research-backed: BayesianRouter shows Thompson > LinUCB/hybrids

### Comparison Algorithms (2) - Alternative non-contextual bandits
2. **UCB1Bandit** - Upper confidence bound (non-contextual)
   - Converges in 100-500 queries
3. **EpsilonGreedyBandit** - Simple ε-greedy exploration
   - Converges in 100-500 queries

### Baseline (1) - Lower bound
4. **RandomBaseline** - Uniform random selection (lower bound)
   - No learning, immediate performance

**Note**: Oracle baseline can be added optionally for regret calculation upper bound.

**Why Not Test Hybrids/Contextual/Dueling?**
- **Data constraint**: Hybrids need 3,000-4,000 queries, we have 2,483
- **Research evidence**: BayesianRouter (arXiv 2510.02850) shows Thompson > LinUCB/hybrids
- **Dueling bandits**: Incompatible feedback mechanism (pairwise comparisons vs absolute scores)
- **Focus**: Validate Thompson as new default, not test algorithms we can't properly evaluate

**Note on Dueling Bandits**: Conduit supports dueling bandits (`algorithm="dueling"`), which uses pairwise comparisons (A vs B) instead of absolute quality scores. This requires a different experimental setup and is not directly comparable with other algorithms. See DESIGN_DECISIONS.md for details.

## 📱 Model Arms (6 Models)

🚨 **CANONICAL MODEL LIST - Matches conduit/core/config.py default_models** 🚨

These exact API model IDs match conduit's default configuration.
Changing models invalidates benchmark results and conduit parity.

| API Model ID | Conduit ID | Provider | Tier | Notes |
|--------------|------------|----------|------|-------|
| `gpt-4o-mini` | `o4-mini` | OpenAI | Budget | Fast, cheap reasoning |
| `gpt-4o` | `gpt-5` | OpenAI | Mid-tier | Strong reasoning |
| `gpt-4-turbo` | `gpt-5.1` | OpenAI | Flagship | Latest flagship |
| `claude-sonnet-4-5-20250929` | `claude-sonnet-4.5` | Anthropic | Balanced | Best for code |
| `claude-opus-4-5-20251101` | `claude-opus-4.5` | Anthropic | Premium | Highest quality |
| `gemini-2.5-pro` | `gemini-2.5-pro` | Google | Flagship | Competitive flagship |

**Source of truth**: `../conduit/conduit/core/config.py` (default_models field, lines 82-92)

API Documentation:
- OpenAI: https://platform.openai.com/docs/models
- Anthropic: https://platform.claude.com/docs/en/about-claude/models/all-models
- Google: https://ai.google.dev/gemini-api/docs/models

## 🔧 Embedding Configuration

**Provider**: OpenAI `text-embedding-3-small` (1536 dimensions)

| Configuration | Embedding | Metadata | Total Features |
|---------------|-----------|----------|----------------|
| Without PCA | 1536 | 2 | **1538** |
| With PCA (128 components) | 128 | 2 | **130** |

**Metadata Features** (2 total):
- `token_count` - Query length normalization
- `complexity_score` - Estimated query complexity

## 📈 Metrics

### Primary Metrics
- **Accuracy**: % of correct answers (exact match or pass rate)
- **Cost**: Total USD spent on API calls
- **Cost per correct answer**: cost / correct_answers (efficiency)
- **Latency**: Average response time

### Secondary Metrics
- **Convergence time**: Queries until stable performance
- **Model selection distribution**: Which models selected over time
- **Pareto efficiency**: Accuracy vs cost tradeoff

### Regret Calculation
For each query:
- If algorithm selects correctly: `regret = cost_selected - cost_cheapest_correct`
- If algorithm selects incorrectly: `regret = quality_penalty + cost_selected`

## 🚀 Quick Start Commands

```bash
# GSM8K Benchmark (Math Reasoning)
uv run conduit-bench run \
  --dataset gsm8k \
  --algorithms thompson,ucb1,epsilon,random \
  --evaluator exact_match \
  --output results/gsm8k.json \
  --parallel

# MMLU Benchmark (Knowledge)
uv run conduit-bench run \
  --dataset mmlu \
  --mmlu-limit 1000 \
  --algorithms thompson,ucb1,epsilon,random \
  --evaluator exact_match \
  --output results/mmlu.json \
  --parallel

# HumanEval Benchmark (Coding)
uv run conduit-bench run \
  --dataset humaneval \
  --algorithms thompson,ucb1,epsilon,random \
  --evaluator code_execution \
  --output results/humaneval.json \
  --parallel

# Generate combined analysis
uv run conduit-bench analyze \
  --results results/gsm8k.json results/mmlu.json results/humaneval.json \
  --output analysis/
```

## 💰 Cost Estimation

### Per Benchmark (4 Algorithms × 3 Runs)

| Benchmark | Queries | Algorithms × Runs | LLM Calls | Est. Cost per Run | Total (3 runs) |
|-----------|---------|-------------------|-----------|-------------------|----------------|
| GSM8K | 1,319 | 4 × 3 | ~5,300 | ~$31-44 | ~$93-132 |
| MMLU | 1,000 | 4 × 3 | ~4,000 | ~$24-33 | ~$72-99 |
| HumanEval | 164 | 4 × 3 | ~660 | ~$10-13 | ~$30-39 |
| **Total** | **2,483** | **12 runs** | **~9,960** | **~$65-90** | **~$195-270** |

**64% cost savings vs original 11-algorithm plan** (~$600-840)

### Cost Breakdown
- 4 bandit algorithms: Thompson, UCB1, Epsilon-Greedy, Random
- Each algorithm: 1 model selected per query
- 3 independent runs for statistical significance
- Embeddings: ~$1 total (negligible)
- **No Arbiter costs** - exact match/code execution is free
- Optional: Add Oracle baseline (+$140-200) for regret upper bound

## ⏱️ Runtime Estimation

**Per Query:**
- Embedding: ~0.3s (Thompson, UCB1, Epsilon don't need embeddings - even faster)
- Model execution: ~2-4s
- Evaluation: ~0s (exact match) or ~2s (code execution)

**Total Runtime (with concurrency=10, 4 algorithms × 3 runs):**
- GSM8K: ~1.5-2 hours (faster without embedding overhead)
- MMLU: ~1-1.5 hours
- HumanEval: ~20-30 minutes
- **Total: ~2.5-4 hours (65% time savings vs 11-algorithm plan)**

## 📊 Success Criteria

### Thompson Sampling Validation (Primary Goal)
- **Accuracy**: Thompson outperforms or matches UCB1/Epsilon baselines
- **Cost efficiency**: Achieves competitive cost-per-correct-answer vs baselines
- **Convergence**: Shows clear learning curve improvement over Random baseline
- **Consistency**: Performs well across all 3 domains (math, knowledge, code)

**Target**: Empirical validation that Thompson Sampling is the right default for conduit (supporting GitHub Issue #169)

### Headline Results (Target)
1. **GSM8K**: "Thompson Sampling achieves 85%+ accuracy while learning optimal model selection"
2. **MMLU**: "Learns to route across 57 knowledge domains intelligently"
3. **HumanEval**: "Smart model selection for code generation with minimal exploration overhead"

### Statistical Requirements
- Report mean ± 95% CI for all metrics
- Friedman test for overall algorithm differences
- Nemenyi post-hoc for pairwise comparisons

## ⚠️ Known Limitations

### Dataset Limitations
- **GSM8K**: Math-only, may not generalize to other domains
- **MMLU**: Multiple choice format may not reflect real usage
- **HumanEval**: Python-only, limited to function completion

### Evaluation Limitations
- **Exact match**: Strict - partial credit not possible
- **Code execution**: Sensitive to output format requirements
- Binary reward signal (may slow bandit learning)

### Experimental Limitations
- **Single run**: For stronger claims, run multiple replications
- **Point-in-time**: Model capabilities change over time
- **Synthetic features**: Embeddings may not capture all query properties

### What This Doesn't Prove
- Performance on open-ended queries (no ground truth)
- Production cost savings (requires real traffic)
- Generalization to other model providers
- Hybrid algorithm performance with sufficient data (need 3,000-4,000+ queries)

## 🔬 Future Research

### MATH Dataset - Comprehensive Hybrid Validation

**Why not now?**
- Current plan (2,483 queries) insufficient for hybrids (need 3,000-4,000+)
- Research evidence (BayesianRouter) already shows Thompson > hybrids
- Not required for conduit default decision (GitHub #169)

**Future work** (if comprehensive hybrid validation needed):

**Dataset**: MATH (12,500 test queries)
- Source: [hendrycks/math](https://huggingface.co/datasets/hendrycks/math)
- Size: 12,500 test problems with step-by-step solutions
- Domain: Competition-level mathematics (AMC 8/10/12, AIME)
- Evaluation: Exact match on final answer
- **Sufficient data**: 12,500 >> 4,000 needed for hybrid convergence

**Scope**: All 11 algorithms with proper convergence data
- 5 hybrid variants (Thompson→LinUCB, UCB1→LinUCB, etc.)
- 4 standalone bandits (Thompson, UCB1, LinUCB, Epsilon)
- 2 baselines (Random, Oracle)
- 3 independent runs for statistical significance

**Research Questions**:
1. Do hybrids outperform pure Thompson with sufficient data?
2. What is the optimal switch_threshold for production? (test 1000, 2000, 3000)
3. Does LinUCB phase provide value over pure Thompson?
4. How does PCA dimensionality reduction impact convergence?

**Cost & Time**:
- Full suite: 11 algorithms × 12,500 queries × 3 runs = $1,265-1,771, 16-24 hours
- Hybrids-focused: 7 algorithms × 12,500 queries × 3 runs = $690-966, 10-14 hours

**Expected Outcome** (based on BayesianRouter research):
- Thompson likely matches or exceeds hybrid performance
- Validates that simpler algorithm (Thompson) is better choice
- Provides definitive empirical evidence even with ample data

## 📝 Comparison to Other Routers

### RouteLLM (Berkeley)
- [GitHub](https://github.com/lm-sys/RouteLLM)
- Also benchmarks on GSM8K, MMLU
- Direct comparison possible using same methodology

### Martian Router
- Commercial solution
- No public benchmark results for comparison

---

**Last Updated:** 2025-11-27 (Revised for Thompson Sampling validation)
**Evaluation Strategy:** Exact match (GSM8K, MMLU) + Code execution (HumanEval)
**Total Benchmark Size:** 2,483 queries across 3 datasets
**Algorithms**: 4 core algorithms (Thompson, UCB1, Epsilon, Random)
**Estimated Cost:** ~$195-270 (64% savings vs 11-algorithm plan)
**Estimated Time:** ~2.5-4 hours (65% time savings)
**Primary Goal:** Validate Thompson Sampling as optimal default for conduit (GitHub #169)
**Research Backing:** BayesianRouter (arXiv 2510.02850) - Thompson > LinUCB/hybrids for LLM routing
