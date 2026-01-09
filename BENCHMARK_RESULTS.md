# Benchmark Results: LLM Routing Algorithms

**Date**: November 30, 2025
**Benchmark Version**: Conduit v0.1.0
**Dataset**: MMLU (1000 queries)

## Executive Summary

I evaluated 11 routing algorithms on MMLU (1000 multiple-choice questions across 57 subjects). **Dueling Bandit achieved the highest quality (93.2%)**, outperforming all other algorithms including the "always best" baseline.

### Key Findings

1. **Dueling Bandit**: Best quality at 93.2%, $1.97 total cost
2. **Hybrid UCB1+LinUCB**: Strong quality (91.1%) at lower cost ($1.81)
3. **Learning algorithms outperform static baselines**: The "always best" model only achieved 82.0%
4. **All algorithms converge quickly**: Most converge within 16-30 queries

## Methodology

### Experimental Setup

- **Model Pool**: 6 LLMs (GPT-4o-mini, GPT-4o, GPT-4-turbo, Claude Sonnet 4.5, Claude Opus 4.5, Gemini 2.5 Pro)
- **Dataset**: MMLU - 1,000 multiple-choice questions across 57 subjects
- **Evaluation**: ExactMatch scoring
- **Infrastructure**: Parallel execution (10 concurrent workers)

### Algorithms Tested (11 total)

**Bandit Algorithms**:
- Thompson Sampling
- UCB1
- Epsilon-Greedy (ε=0.1)
- Dueling Bandit

**Contextual Bandits**:
- LinUCB
- Contextual Thompson Sampling

**Hybrid Algorithms**:
- Hybrid Thompson+LinUCB
- Hybrid UCB1+LinUCB

**Baselines**:
- Random
- Static model (GPT-4-turbo)
- Always Cheapest

### Cost Model

Actual API pricing (November 2025):
- **GPT-4o-mini**: $1.10/M input, $4.40/M output
- **GPT-4o**: $1.25/M input, $10/M output
- **GPT-4-turbo**: $1.25/M input, $10/M output
- **Claude Sonnet 4.5**: $3/M input, $15/M output
- **Claude Opus 4.5**: $2/M input, $8/M output
- **Gemini 2.5 Pro**: $1.25/M input, $10/M output

## Results

### MMLU (1000 queries)

| Algorithm | Total Cost | Avg Quality | Rank |
|-----------|------------|-------------|------|
| Dueling Bandit | $1.97 | **93.2%** | 1 |
| Hybrid UCB1+LinUCB | $1.81 | 91.1% | 2 |
| Epsilon-Greedy | $1.72 | 90.5% | 3 |
| Thompson Sampling | $2.61 | 90.3% | 4 |
| Contextual Thompson | $2.45 | 87.7% | 5 |
| Random | $2.20 | 86.9% | 6 |
| UCB1 | $2.47 | 86.7% | 7 |
| Hybrid Thompson+LinUCB | $0.37 | 82.6% | 8 |
| LinUCB | $0.19 | 82.3% | 9 |
| Static model (GPT-4-turbo) | $0.20 | 82.0% | 10 |
| Always Cheapest | **$0.17** | 76.4% | 11 |

**Key Insights**:
- **Dueling Bandit achieves 93.2% quality** - highest of all algorithms
- Learning algorithms significantly outperform static model selection
- Static routing to GPT-4-turbo (82.0%) is beaten by 7 different algorithms
- Cost-quality tradeoffs vary widely: Dueling Bandit costs 10x more than LinUCB but gains 11% quality

### Why Learning Algorithms Win

Static model selection assumes one model dominates all queries. In practice, different models excel at different question types:

1. **No single model is universally best** - performance varies by subject area
2. **Learning algorithms adapt** - they discover which models work for which query types
3. **Dueling Bandit's pairwise comparisons** - effective at finding the best model per-context

### Convergence Speed

| Algorithm | Convergence Point |
|-----------|-------------------|
| Hybrid Thompson+LinUCB | 16 queries |
| Hybrid UCB1+LinUCB | 16 queries |
| Thompson Sampling | 16 queries |
| UCB1 | 16 queries |
| Epsilon-Greedy | 16 queries |
| LinUCB | 27 queries |
| Contextual Thompson | 30 queries |

### Pareto Frontier

The following algorithms are Pareto-optimal (no other algorithm dominates them on both cost and quality):

- epsilon_greedy
- linucb
- dueling_bandit
- hybrid_thompson_linucb
- hybrid_ucb1_linucb
- always_cheapest

## Production Recommendations

### Best Overall: Dueling Bandit

- Highest quality (93.2%)
- Reasonable cost ($1.97 for 1000 queries)
- Good for quality-critical applications

### Best Cost-Quality Balance: Hybrid UCB1+LinUCB

- Strong quality (91.1%)
- Lower cost than Dueling Bandit ($1.81)
- Fast convergence (16 queries)

### Best for Cost-Sensitive: LinUCB or Hybrid Thompson+LinUCB

- Very low cost ($0.19-$0.37)
- Reasonable quality (82-83%)
- Good for high-volume, cost-constrained workloads

### When to Use Each Algorithm

| Use Case | Recommended Algorithm |
|----------|----------------------|
| Quality-critical | Dueling Bandit |
| Balanced | Hybrid UCB1+LinUCB |
| Cost-sensitive | LinUCB |
| Simple workloads | Epsilon-Greedy |

## Reproducibility

### Requirements
```bash
uv sync
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
export GOOGLE_API_KEY="..."
```

### Run Benchmarks
```bash
# MMLU 1000 queries, 11 algorithms
conduit-bench run \
  --dataset mmlu \
  --mmlu-limit 1000 \
  --algorithms all \
  --evaluator exact_match \
  --output results/mmlu_full_11algorithms.json \
  --parallel \
  --max-concurrency 10
```

### Generate Analysis
```bash
conduit-bench analyze \
  --results results/mmlu_full_11algorithms.json \
  --output analysis/mmlu_full_11algorithms.json
```

## Conclusion

**Learning algorithms consistently outperform static model selection** for LLM routing:

1. **Quality**: Dueling Bandit achieves 93.2% vs 82.0% for static GPT-4-turbo
2. **Adaptability**: Algorithms learn query-specific routing patterns
3. **Fast Learning**: Most converge within 16-30 queries
4. **Flexibility**: Different algorithms suit different cost-quality tradeoffs

The key insight: **no single model is best for all queries**. Adaptive routing discovers this automatically.

---

**Benchmark Cost**: ~$15 total (11 algorithms × 1000 queries)
**Runtime**: ~30 minutes (parallel execution)
