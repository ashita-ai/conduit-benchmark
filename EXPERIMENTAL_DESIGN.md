# Conduit Benchmark Experimental Design

## 🎯 Objectives

1. **Validate HybridRouter** - the production routing algorithm conduit ships
2. **Use objective evaluation** - exact-match and code execution, NOT LLM-as-judge
3. **Benchmark on established datasets** - GSM8K, MMLU, HumanEval (credible, reproducible)
4. Measure accuracy, cost, latency across multiple domains
5. Compare against baselines and alternative algorithms

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

## 🤖 Algorithms Under Test (7 Total)

### Production Algorithm (1) - What conduit ships ⭐
1. **HybridRouter** - UCB1 (0-2000 queries) → LinUCB (2000+)
   - Cold-start: UCB1 explores efficiently without needing context
   - Warm: LinUCB exploits query features after sufficient data
   - **This is the algorithm we're validating**

### Component Algorithms (2) - HybridRouter building blocks
2. **LinUCBBandit** - Linear UCB with contextual features
3. **UCB1Bandit** - Upper confidence bound (non-contextual)

### Alternative Algorithms (2) - Comparison
4. **ThompsonSamplingBandit** - Bayesian probability matching
5. **EpsilonGreedyBandit** - Simple exploration-exploitation

### Baselines (2) - Upper/lower bounds
6. **RandomBaseline** - Uniform random selection (lower bound)
7. **OracleBaseline** - Best model per query (upper bound, requires all models run)

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
- `gemini-3-pro` - New flagship (1501 Elo)
- `gemini-3-flash` - Fast inference
- `gemini-2.5-pro` - Previous generation baseline

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
  --algorithms hybrid,linucb,ucb1,thompson,epsilon,random \
  --evaluator exact_match \
  --output results/gsm8k.json

# MMLU Benchmark (Knowledge)
uv run conduit-bench run \
  --dataset mmlu \
  --mmlu-limit 1000 \
  --algorithms hybrid,linucb,ucb1,thompson,epsilon,random \
  --evaluator exact_match \
  --output results/mmlu.json

# HumanEval Benchmark (Coding)
uv run conduit-bench run \
  --dataset humaneval \
  --algorithms hybrid,linucb,ucb1,thompson,epsilon,random \
  --evaluator code_execution \
  --output results/humaneval.json

# Generate combined analysis
uv run conduit-bench analyze \
  --results results/gsm8k.json results/mmlu.json results/humaneval.json \
  --output analysis/
```

## 💰 Cost Estimation

### Per Benchmark

| Benchmark | Queries | LLM Calls (bandits) | Oracle Calls | Total Calls | Est. Cost |
|-----------|---------|---------------------|--------------|-------------|-----------|
| GSM8K | 1,319 | ~9,200 | ~12,000 | ~21,000 | ~$100-150 |
| MMLU | 1,000 | ~7,000 | ~9,000 | ~16,000 | ~$80-100 |
| HumanEval | 164 | ~1,150 | ~1,500 | ~2,650 | ~$20-30 |
| **Total** | **2,483** | **~17,000** | **~22,500** | **~40,000** | **~$200-300** |

### Cost Breakdown
- Bandit algorithms: Each selects 1 model per query
- Oracle: Runs all 9 models per query (expensive but needed for regret)
- Embeddings: ~$1 total (negligible)
- **No Arbiter costs** - exact match/code execution is free

## ⏱️ Runtime Estimation

**Per Query:**
- Embedding: ~0.3s
- Model execution: ~2-4s
- Evaluation: ~0s (exact match) or ~2s (code execution)

**Total Runtime (with concurrency=30):**
- GSM8K: ~2-3 hours
- MMLU: ~1.5-2 hours
- HumanEval: ~30-45 minutes
- **Total: ~4-6 hours**

## 📊 Success Criteria

### HybridRouter Validation (Primary Goal)
- **Accuracy**: Matches or exceeds standalone LinUCB/UCB1
- **Cost efficiency**: Achieves >85% of oracle accuracy at <50% oracle cost
- **Convergence**: Shows clear learning curve improvement

### Headline Results (Target)
1. **GSM8K**: "85% accuracy at 40% the cost of always using best model"
2. **MMLU**: "Matches top-tier accuracy while cutting costs in half"
3. **HumanEval**: "75% pass rate with intelligent model selection"

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

## 📝 Comparison to Other Routers

### RouteLLM (Berkeley)
- [GitHub](https://github.com/lm-sys/RouteLLM)
- Also benchmarks on GSM8K, MMLU
- Direct comparison possible using same methodology

### Martian Router
- Commercial solution
- No public benchmark results for comparison

---

**Last Updated:** 2025-11-27
**Evaluation Strategy:** Exact match (GSM8K, MMLU) + Code execution (HumanEval)
**Total Benchmark Size:** 2,483 queries across 3 datasets
**Estimated Cost:** ~$200-300
**Estimated Time:** ~4-6 hours
**Primary Goal:** Validate HybridRouter with objective, reproducible benchmarks
