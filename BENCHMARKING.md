# Benchmarking Guide

Complete guide for running bandit algorithm benchmarks with conduit-benchmark.

## Quick Start

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

## Datasets (3 Benchmarks)

### Overview

| Dataset | Size | Domain | Evaluation | Est. Cost |
|---------|------|--------|------------|-----------|
| **GSM8K** | 1,319 | Math reasoning | Exact match | ~$100-150 |
| **MMLU** | 1,000 | Knowledge (57 subjects) | Exact match | ~$80-100 |
| **HumanEval** | 164 | Python coding | Code execution | ~$20-30 |
| **Total** | **2,483** | | | **~$200-300** |

### 1. GSM8K (Grade School Math)

**Source**: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)
**Size**: 1,319 test problems
**Evaluation**: Exact match on `#### N` answer format

**Why GSM8K**:
- Step-by-step reasoning with verifiable final answer
- Answer format: `#### 72` - objectively correct/incorrect
- No LLM-as-judge needed - eliminates circular dependency
- Well-established benchmark for math reasoning

**Example**:
```
Question: Natalia sold clips to 48 friends in April and half as many in May.
          How many clips did she sell altogether?
Answer: "Natalia sold 48/2 = <<48/2=24>>24 clips in May.
        Natalia sold 48+24 = <<48+24=72>>72 clips altogether.
        #### 72"
```

**Extraction**:
```python
def extract_gsm8k_answer(text: str) -> str | None:
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    return match.group(1).replace(',', '') if match else None
```

### 2. MMLU (Massive Multitask Language Understanding)

**Source**: [cais/mmlu](https://huggingface.co/datasets/cais/mmlu)
**Size**: 1,000 questions (subset of 14,042)
**Evaluation**: Exact match on A/B/C/D answer

**Why MMLU**:
- 57 subjects: STEM, humanities, social sciences
- Multiple choice - unambiguous correct answer
- Broad coverage tests generalization
- Standard benchmark for knowledge evaluation

**Example**:
```
Question: What is the capital of France?
(A) London (B) Berlin (C) Paris (D) Madrid
Answer: C
```

**Extraction**:
```python
def extract_mmlu_answer(text: str) -> str | None:
    match = re.search(r'\b([ABCD])\b', text.upper())
    return match.group(1) if match else None
```

### 3. HumanEval (Code Generation)

**Source**: [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval)
**Size**: 164 Python function problems
**Evaluation**: Code execution with unit tests

**Why HumanEval**:
- Most credible to developers - executable tests > LLM judges
- Pass/fail based on actual code execution
- Sandboxed subprocess with timeout
- Standard benchmark for code generation

**Example**:
```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Check if any two numbers in list are closer than threshold."""
    # Model generates implementation here

# Test:
assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False
assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0], 0.3) == True
```

**Evaluation**:
```python
full_code = f"{prompt}{response}\n\n{test_code}\ncheck({entry_point})"
result = subprocess.run(['python', temp_file], timeout=10, capture_output=True)
reward = 1.0 if result.returncode == 0 else 0.0
```

## Running Benchmarks

### Basic Usage

```bash
# Run a single benchmark
uv run conduit-bench run \
  --dataset gsm8k \
  --algorithms hybrid,linucb,ucb1 \
  --evaluator exact_match \
  --output results/gsm8k.json

# Run with oracle baseline (for regret calculation)
uv run conduit-bench run \
  --dataset gsm8k \
  --algorithms hybrid,linucb,ucb1,oracle \
  --evaluator exact_match \
  --output results/gsm8k_with_oracle.json
```

### Advanced Options

```bash
uv run conduit-bench run \
  --dataset gsm8k \
  --algorithms hybrid,linucb,thompson,ucb1,epsilon,random,oracle \
  --evaluator exact_match \
  --output results/gsm8k_full.json \
  --parallel \                      # Enable parallel LLM calls
  --max-concurrency 30 \            # Max concurrent requests
  --oracle-reference results/oracle.json # Reuse oracle results
```

### Algorithm Options

**Production Algorithm:**
- `hybrid`: HybridRouter - UCB1 (0-2000 queries) → LinUCB (2000+) ⭐

**Component Algorithms:**
- `linucb`: Contextual bandit with linear upper confidence bound
- `ucb1`: Upper Confidence Bound (non-contextual)

**Alternative Algorithms:**
- `thompson`: Thompson Sampling (Bayesian approach)
- `epsilon`: ε-greedy exploration

**Baselines:**
- `oracle`: Always select best model (upper bound)
- `random`: Random selection (lower bound)

### Evaluator Options

- `exact_match`: For GSM8K and MMLU - extract and compare answers
- `code_execution`: For HumanEval - run code and check tests pass

### Performance Optimization

**Oracle Caching** (saves API calls on repeat runs):
```bash
# Step 1: Generate oracle results once
uv run conduit-bench run \
  --dataset gsm8k \
  --algorithms oracle \
  --evaluator exact_match \
  --output results/gsm8k_oracle.json \
  --max-concurrency 30

# Step 2: Reuse oracle results for bandit algorithms
uv run conduit-bench run \
  --dataset gsm8k \
  --algorithms hybrid,linucb,thompson,ucb1,epsilon \
  --evaluator exact_match \
  --oracle-reference results/gsm8k_oracle.json \
  --parallel \
  --max-concurrency 30 \
  --output results/gsm8k_bandits.json
```

**Parallelization**:
- `--parallel`: Enable concurrent LLM calls
- `--max-concurrency N`: Limit concurrent requests (default: 10)
- Recommended: 20-30 for most API providers

## Time and Cost Estimation

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

### Runtime Estimation

**Per Query:**
- Embedding: ~0.3s
- Model execution: ~2-4s
- Evaluation: ~0s (exact match) or ~2s (code execution)

**Total Runtime (with concurrency=30):**
- GSM8K: ~2-3 hours
- MMLU: ~1.5-2 hours
- HumanEval: ~30-45 minutes
- **Total: ~4-6 hours**

## Analysis

### Basic Analysis

```bash
uv run conduit-bench analyze \
  --results results/gsm8k.json \
  --output analysis/gsm8k_metrics.json
```

**Output Metrics**:
- Accuracy: % of correct answers (exact match or pass rate)
- Cost: Total USD spent on API calls
- Cost per correct answer: cost / correct_answers (efficiency)
- Latency: Average response time
- Regret: Cumulative cost vs oracle
- Learning curves: Performance over time

### Combined Analysis

```bash
# Analyze all benchmarks together
uv run conduit-bench analyze \
  --results results/gsm8k.json results/mmlu.json results/humaneval.json \
  --output analysis/combined_metrics.json
```

### Statistical Testing

```bash
# Statistical significance testing
uv run conduit-bench significance \
  --results results/gsm8k.json \
  --baseline oracle \
  --alpha 0.05 \
  --output analysis/significance.json
```

**Statistical Methods**:
- Friedman test for overall algorithm differences
- Nemenyi post-hoc for pairwise comparisons
- Report mean ± 95% CI for all metrics

## File Organization

```
conduit-benchmark/
├── data/
│   └── (datasets auto-downloaded from HuggingFace)
├── results/
│   ├── gsm8k.json                       # GSM8K benchmark results
│   ├── mmlu.json                        # MMLU benchmark results
│   └── humaneval.json                   # HumanEval benchmark results
├── analysis/
│   ├── combined_metrics.json            # Cross-benchmark analysis
│   ├── learning_curves.png              # Visualization
│   └── significance.json                # Statistical tests
└── conduit_bench/
    ├── evaluators/                      # Pluggable evaluators
    │   ├── base.py                      # Abstract evaluator interface
    │   ├── exact_match.py               # GSM8K/MMLU evaluator
    │   └── code_execution.py            # HumanEval evaluator
    └── datasets/                        # Dataset loaders
        ├── gsm8k.py                     # GSM8K loader
        ├── mmlu.py                      # MMLU loader
        └── humaneval.py                 # HumanEval loader
```

## Best Practices

### 1. Start with HumanEval

HumanEval has only 164 problems - cheapest way to validate setup:
```bash
uv run conduit-bench run \
  --dataset humaneval \
  --algorithms hybrid,ucb1,random \
  --evaluator code_execution \
  --output results/humaneval_test.json
```

### 2. Use Oracle Caching

Generate oracle results separately and reuse:
```bash
# Generate oracle once per dataset
uv run conduit-bench run \
  --dataset gsm8k \
  --algorithms oracle \
  --evaluator exact_match \
  --output results/gsm8k_oracle.json

# Reuse for all bandit runs
uv run conduit-bench run \
  --dataset gsm8k \
  --algorithms hybrid,linucb,ucb1 \
  --evaluator exact_match \
  --oracle-reference results/gsm8k_oracle.json \
  --output results/gsm8k_bandits.json
```

### 3. Monitor Costs

Check estimated costs before running:
```bash
# Preview run cost (dry run mode)
uv run conduit-bench run \
  --dataset gsm8k \
  --algorithms hybrid,linucb,thompson,ucb1,oracle \
  --evaluator exact_match \
  --dry-run
```

### 4. Save Intermediate Results

Benchmark runs can be interrupted:
```bash
# Results auto-save every 100 queries to output file
# Resume from checkpoint on interruption
```

### 5. Replicate for Statistical Significance

Run multiple replications with different seeds:
```bash
for seed in {1..5}; do
  uv run conduit-bench run \
    --dataset gsm8k \
    --algorithms hybrid,linucb,thompson,ucb1 \
    --evaluator exact_match \
    --oracle-reference results/gsm8k_oracle.json \
    --seed $seed \
    --output results/gsm8k_seed_${seed}.json
done

# Analyze statistical significance
uv run conduit-bench significance \
  --results results/gsm8k_seed_*.json \
  --output analysis/gsm8k_significance.json
```

## Troubleshooting

### API Rate Limits

If hitting rate limits:
```bash
# Reduce concurrency
--max-concurrency 5

# Add delays between requests
--request-delay 0.1
```

### Code Execution Timeout

For HumanEval problems taking too long:
```bash
# Increase timeout (default 10s)
--code-timeout 30
```

### Missing Dependencies

For HumanEval execution:
```bash
# Ensure numpy, typing_extensions available in sandbox
uv add numpy typing_extensions
```

## Next Steps

1. **Start with HumanEval**: 164 problems, lowest cost (~$20-30)
2. **Then GSM8K**: 1,319 problems, math reasoning
3. **Then MMLU**: 1,000 problems, knowledge breadth
4. **Analyze combined results**: Generate headline numbers
5. **Prepare HN post**: Use results for launch

## References

- **GSM8K Dataset**: https://huggingface.co/datasets/openai/gsm8k
- **MMLU Dataset**: https://huggingface.co/datasets/cais/mmlu
- **HumanEval Dataset**: https://huggingface.co/datasets/openai/openai_humaneval
- **RouteLLM (Berkeley)**: https://github.com/lm-sys/RouteLLM
- **Experimental Design**: EXPERIMENTAL_DESIGN.md
