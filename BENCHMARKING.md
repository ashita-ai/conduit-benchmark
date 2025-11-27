# Benchmarking Guide

Complete guide for running bandit algorithm benchmarks with conduit-benchmark.

## Quick Start

```bash
# 1. Generate synthetic dataset (10k queries for training)
uv run python -m conduit_bench.cli generate \
  --queries 10000 \
  --seed 42 \
  --output data/synthetic_10k.jsonl \
  --reference-probability 1.0

# 2. Run benchmark with oracle caching
# Step 2a: Generate oracle baseline
uv run python -m conduit_bench.cli run \
  --dataset data/synthetic_10k.jsonl \
  --algorithms oracle \
  --output results/oracle.json \
  --max-concurrency 30

# Step 2b: Run bandit algorithms with oracle cache
uv run python -m conduit_bench.cli run \
  --dataset data/synthetic_10k.jsonl \
  --algorithms linucb,thompson,ucb1,epsilon_greedy \
  --oracle-reference results/oracle.json \
  --output results/benchmark_results.json \
  --parallel \
  --max-concurrency 30

# 3. Analyze results
uv run python -m conduit_bench.cli analyze \
  --results results/benchmark_results.json \
  --output analysis/metrics.json
```

## Datasets

### 1. Synthetic Dataset (Training)

**Source**: Template-based generation with 10 categories
**Size**: 10,000 queries recommended for training
**Purpose**: Primary training dataset for bandit algorithm development

**Advantages**:
- Fast generation (no API calls, instant)
- Controllable difficulty and diversity
- Full text reference answers (Arbiter-compatible)
- Cost-free dataset creation
- 10 categories, 200+ templates

**Generate**:
```bash
uv run python -m conduit_bench.cli generate \
  --queries 10000 \
  --seed 42 \
  --output data/synthetic_10k.jsonl \
  --reference-probability 1.0
```

**Parameters**:
- `--queries`: Number of queries to generate
- `--seed`: Random seed for reproducibility
- `--reference-probability`: Fraction with reference answers (0.0-1.0)
  - 1.0 = all queries have answers (recommended for oracle comparison)
  - 0.5 = balanced (oracle + non-oracle evaluation)
  - 0.0 = no reference answers (routing-only benchmarks)

**Categories** (200+ templates):
- Code generation, debugging, review, architecture
- Technical writing, documentation
- Data analysis, SQL queries
- System design, API design
- Testing, deployment, monitoring
- Math, reasoning, general knowledge

### 2. GSM8K Dataset (Validation)

**Source**: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) - Grade school math word problems
**Size**: 1,000 test queries recommended
**Purpose**: Out-of-distribution validation to test router generalization

**Advantages**:
- Full step-by-step reasoning in reference answers (Arbiter-compatible)
- Well-established benchmark (8.5K problems)
- Different domain from synthetic (math vs code-heavy)
- Tests router generalization beyond training distribution
- Validates transfer learning capability

**Example Reference Answer**:
```
"Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72"
```

**Validation Strategy**:
Train router on synthetic queries (code-heavy) → Validate on GSM8K (math) to test if router learned:
- ✅ **Routing strategy** (generalizable pattern recognition)
- ❌ **Query patterns** (domain-specific overfitting)

**Future**: ELI5 dataset for extreme generalization testing (Reddit-style explanations)

### 3. Three-Phase Validation Strategy

**Phase 1 - Development (Synthetic)**:
- Train/tune router on 10,000 synthetic queries
- Fast iteration, controlled diversity
- Validates router learns on known distribution

**Phase 2 - In-Distribution Validation (Held-out Synthetic)**:
- Test on 1,000 held-out synthetic queries
- Confirms router works on same distribution

**Phase 3 - Out-of-Distribution Validation (GSM8K)**:
- Test on 1,000 GSM8K test queries
- Confirms router generalizes beyond training distribution
- Stronger signal of router quality

## Running Benchmarks

### Basic Usage

```bash
uv run python -m conduit_bench.cli run \
  --dataset data/synthetic_10k.jsonl \
  --algorithms linucb,thompson,ucb1,oracle \
  --output results/benchmark.json
```

### Advanced Options

```bash
uv run python -m conduit_bench.cli run \
  --dataset data/synthetic_10k.jsonl \
  --algorithms linucb,thompson,ucb1,epsilon_greedy,oracle \
  --output results/full_benchmark.json \
  --parallel \                      # Enable parallel LLM calls
  --max-concurrency 30 \             # Max concurrent requests
  --oracle-reference results/oracle.json # Reuse oracle results
```

### Algorithm Options

- `linucb`: Contextual bandit with linear upper confidence bound
- `thompson`: Thompson Sampling (Bayesian approach)
- `ucb1`: Upper Confidence Bound (stateless)
- `epsilon_greedy`: ε-greedy exploration
- `oracle`: Always select best model (baseline)
- `random`: Random selection (baseline)

### Performance Optimization

**Oracle Caching** (saves 50% of API calls):
```bash
# Step 1: Generate oracle results once
uv run python -m conduit_bench.cli run \
  --dataset data/synthetic_10k.jsonl \
  --algorithms oracle \
  --output results/oracle_reference.json \
  --max-concurrency 30

# Step 2: Reuse oracle results for bandit algorithms
uv run python -m conduit_bench.cli run \
  --dataset data/synthetic_10k.jsonl \
  --algorithms linucb,thompson,ucb1,epsilon_greedy \
  --oracle-reference results/oracle_reference.json \
  --parallel \
  --max-concurrency 30 \
  --output results/bandits_with_oracle_cache.json
```

**Parallelization**:
- `--parallel`: Enable concurrent LLM calls
- `--max-concurrency N`: Limit concurrent requests (default: 10)
- Recommended: 20-30 for most API providers

## Time and Cost Estimation

### Synthetic Dataset (10k queries)

**With Oracle Caching**:
- Oracle run: 10,000 queries × 9 models = 90,000 LLM calls
- Bandit algorithms: 10,000 queries × 1 model each = 10,000 LLM calls
- **Total**: 100,000 LLM calls

**Without Oracle Caching**:
- Each algorithm: 10,000 queries × 9 models = 90,000 LLM calls
- 5 algorithms: 450,000 LLM calls

**Estimated Time** (with concurrency=30):
- Oracle: ~2-3 hours
- Each bandit: ~20-30 minutes
- **Total with caching**: ~4 hours
- **Total without caching**: ~15-20 hours

**Estimated Cost** (gpt-4o-mini at $0.15/$0.60 per 1M tokens):
- Average query: ~500 input + 100 output tokens
- 100,000 calls × 600 tokens = 60M tokens
- **Cost with caching**: ~$15
- **Cost without caching**: ~$60

### GSM8K Validation Dataset (1k queries)

**With Oracle Caching**:
- Oracle run: 1,000 queries × 9 models = 9,000 LLM calls
- Bandit algorithms: 1,000 queries × 1 model each = 1,000 LLM calls
- **Total**: 10,000 LLM calls

**Estimated Time** (with concurrency=30):
- Oracle: ~20-30 minutes
- Each bandit: ~3-5 minutes
- **Total with caching**: ~40 minutes

**Estimated Cost**:
- **Cost with caching**: ~$1.50

## Analysis

### Basic Analysis

```bash
uv run python -m conduit_bench.cli analyze \
  --results results/benchmark.json \
  --output analysis/metrics.json
```

**Output Metrics**:
- Accuracy (vs oracle)
- Average quality score
- Total cost
- Average latency
- Regret (cumulative cost vs oracle)
- Learning curves

### Advanced Analysis

```bash
# Compare multiple benchmark runs
uv run python -m conduit_bench.cli compare \
  --results results/run1.json results/run2.json results/run3.json \
  --output analysis/comparison.json

# Statistical significance testing
uv run python -m conduit_bench.cli significance \
  --results results/benchmark.json \
  --baseline oracle \
  --alpha 0.05 \
  --output analysis/significance.json
```

## Experimental Design (Issue #141)

For validating Scalable LinUCB implementation:

```bash
# 1. Generate synthetic dataset with sufficient convergence samples
uv run python -m conduit_bench.cli generate \
  --queries 10000 \
  --seed 42 \
  --output data/synthetic_10k.jsonl \
  --reference-probability 1.0

# 2. Benchmark standard LinUCB
uv run python -m conduit_bench.cli run \
  --dataset data/synthetic_10k.jsonl \
  --algorithms linucb \
  --output results/standard_linucb.json

# 3. Benchmark Scalable LinUCB (after implementation)
uv run python -m conduit_bench.cli run \
  --dataset data/synthetic_10k.jsonl \
  --algorithms scalable_linucb \
  --output results/scalable_linucb.json

# 4. Compare convergence and performance
uv run python -m conduit_bench.cli compare \
  --results results/standard_linucb.json results/scalable_linucb.json \
  --output analysis/scalable_comparison.json
```

**Expected Results** (from arXiv:2510.19349):
- **Computation**: 59x faster matrix operations (387³ → 387×64²)
- **Memory**: 3.1x less memory (387² → 387×64×2)
- **Convergence**: Similar or faster learning (fewer samples needed)
- **Quality**: Preserves original LinUCB algorithm quality

## File Organization

```
conduit-benchmark/
├── data/
│   ├── synthetic_10k.jsonl              # Training dataset (synthetic)
│   ├── synthetic_test_1k.jsonl          # Test dataset (held-out synthetic)
│   ├── gsm8k_1k.jsonl                   # Validation dataset (GSM8K)
│   └── pilot_200.jsonl                  # Small test dataset
├── results/
│   ├── oracle_reference.json            # Oracle baseline
│   ├── linucb_results.json              # LinUCB results
│   ├── thompson_results.json            # Thompson Sampling results
│   └── ucb1_results.json                # UCB1 results
├── analysis/
│   ├── metrics.json                     # Performance metrics
│   ├── comparison.json                  # Algorithm comparison
│   └── learning_curves.png              # Visualization
└── archive/
    └── routerbench/                     # Archived RouterBench code
        └── routerbench_loader.py        # RouterBench adapter (archived)
```

## Best Practices

### 1. Start with Small Test Run

```bash
# Generate small test dataset (10 queries)
uv run python -m conduit_bench.cli generate \
  --queries 10 \
  --seed 99 \
  --output data/test_10.jsonl \
  --reference-probability 1.0

# Test with small dataset
uv run python -m conduit_bench.cli run \
  --dataset data/test_10.jsonl \
  --algorithms ucb1,oracle \
  --output results/test_10.json
```

### 2. Use Oracle Caching

Always generate oracle results separately and reuse:
```bash
# Generate oracle once
uv run python -m conduit_bench.cli run --dataset DATA --algorithms oracle --output oracle.json

# Reuse for all bandit runs
uv run python -m conduit_bench.cli run --dataset DATA --oracle-reference oracle.json ...
```

### 3. Monitor Costs

Check estimated costs before running:
```bash
# Preview run cost (dry run mode)
uv run python -m conduit_bench.cli run \
  --dataset data/hybrid_10k.jsonl \
  --algorithms linucb,thompson,ucb1,oracle \
  --dry-run
```

### 4. Save Intermediate Results

Benchmark runs can be interrupted. Save results periodically:
```bash
# Results auto-save every 100 queries to output file
# Resume from checkpoint on interruption
```

### 5. Replicate for Statistical Significance

Run multiple replications with different seeds:
```bash
for seed in {1..10}; do
  uv run python -m conduit_bench.cli run \
    --dataset data/synthetic_10k.jsonl \
    --algorithms linucb,thompson,ucb1 \
    --oracle-reference results/oracle.json \
    --seed $seed \
    --output results/replication_${seed}.json
done

# Analyze statistical significance
uv run python -m conduit_bench.cli significance \
  --results results/replication_*.json \
  --output analysis/significance.json
```

## Troubleshooting

### API Rate Limits

If hitting rate limits:
```bash
# Reduce concurrency
--max-concurrency 5

# Add delays between requests (if supported)
--request-delay 0.1
```

### Out of Memory

For large datasets:
```bash
# Process in batches
--batch-size 1000

# Reduce feature dimensions
--use-pca \
--pca-dimensions 67
```

### Slow Convergence

If algorithms aren't converging:
```bash
# Increase dataset size to 10k queries
uv run python -m conduit_bench.cli generate \
  --queries 10000 \
  --seed 42 \
  --output data/synthetic_10k.jsonl \
  --reference-probability 1.0
```

## Next Steps

1. **Generate synthetic dataset**: Create 10k queries for training
2. **Run quick test**: Use 10-query test dataset to validate setup
3. **Generate oracle**: Create oracle baseline with caching enabled
4. **Benchmark algorithms**: Compare all bandit algorithms on synthetic data
5. **In-distribution validation**: Test on held-out synthetic queries
6. **Out-of-distribution validation**: Test on GSM8K to evaluate generalization
7. **Analyze results**: Generate learning curves and statistical comparison
8. **Future validation**: Test on ELI5 dataset for extreme generalization

## References

- **GSM8K Dataset**: https://huggingface.co/datasets/openai/gsm8k
- **ELI5 Dataset**: https://huggingface.co/datasets/eli5
- **Issue #141**: Scalable LinUCB implementation (arXiv:2510.19349)
- **Design Decisions**: DESIGN_DECISIONS.md
- **Experimental Design**: EXPERIMENTAL_DESIGN.md
