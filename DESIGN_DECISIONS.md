# Design Decisions

Critical architectural and dataset decisions for conduit-benchmark.

---

## Dataset Strategy

### Decision: Synthetic Generation + GSM8K Validation

**Date**: 2024-11-26

**Context**: Need benchmark dataset to evaluate bandit routing algorithms with semantic evaluation (Arbiter).

**Options Considered**:
1. **RouterBench** (withmartian/routerbench)
2. **Synthetic generation** (template-based)
3. **GSM8K** (openai/gsm8k)
4. **Hybrid** (RouterBench + Synthetic)

**Decision**: Use **synthetic generation for development**, **GSM8K for validation**

**Rationale**:

#### Why NOT RouterBench

| Aspect | RouterBench | Required for Arbiter |
|--------|-------------|---------------------|
| Reference answers | "A", "B", "C", "D" | Full text responses |
| Evaluation method | Binary correctness (0.0/1.0) | Semantic similarity |
| Query type | Multiple choice | Open-ended |
| Arbiter compatibility | ❌ Meaningless | ✅ Required |

**Core problem**: Semantic similarity between a model's explanation ("Based on the context, B is correct because...") and reference answer ("B") is essentially random. Arbiter cannot extract quality signals from letter-only answers.

**Example incompatibility**:
```
Query: "Which option continues the sentence correctly?"
Model response: "Based on the context, option B makes the most sense because it maintains narrative consistency and logical flow."
Reference answer: "B"
Semantic similarity: ~0.15 (meaningless)
```

RouterBench is fundamentally incompatible with semantic evaluation.

#### Why Synthetic Generation

**Advantages**:
- Full text reference answers compatible with Arbiter
- Open-ended queries matching production use cases
- Controllable diversity (code, debugging, docs, etc.)
- Fast generation (no API calls, instant)
- Cost-free dataset creation
- 10 categories, 200+ templates

**Production alignment**:
- Code generation and debugging
- Technical explanations
- Documentation writing
- System design
- Data analysis

**Disadvantages**:
- Template-based (less natural than real queries)
- Potential pattern overfitting
- Limited domain coverage (only what we template)

#### Why GSM8K for Validation

**Advantages**:
- Full step-by-step reasoning in reference answers
- Semantic evaluation compatible (2-8 solution steps)
- Well-established benchmark (8.5K problems)
- Out-of-distribution test (math vs code-heavy synthetic)
- Validates router generalization

**Example GSM8K answer**:
```
"Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72"
```

**Disadvantages**:
- Math-only domain (narrow)
- Different distribution from production (transfer learning question)

#### Transfer Learning Strategy

**Question**: If router trains on synthetic (code-heavy), will it generalize to GSM8K (math)?

**Answer**: This is a **feature, not a bug**. We want to test generalization.

**Three-phase validation strategy**:

1. **Development (Synthetic)**:
   - Train/tune router on synthetic queries
   - Fast iteration, controlled diversity
   - Validates router learns on known distribution

2. **In-distribution validation (Held-out Synthetic)**:
   - Test on held-out synthetic test set
   - Confirms router works on same distribution

3. **Out-of-distribution validation (GSM8K)**:
   - Test on GSM8K (different domain)
   - Confirms router generalizes beyond training distribution
   - Stronger signal of router quality

**Why this works**:
- Router learns from **embeddings**, not query content
- Math queries and code queries have different embedding patterns
- If router learns to route code queries well, GSM8K tests if it learned **routing strategy** (generalizable) vs **code patterns** (overfitting)

**Final dataset mix**:
- **Training**: 10,000 synthetic queries (all 10 categories)
- **In-distribution validation**: 1,000 held-out synthetic queries
- **Out-of-distribution validation**: 1,000 GSM8K test queries

---

## Evaluation Architecture

### Decision: Arbiter-based Semantic Evaluation

**Date**: 2024-11-25

**Context**: Need to evaluate response quality for routing decisions.

**Options Considered**:
1. Binary correctness (0.0/1.0)
2. Exact match
3. Semantic similarity (Arbiter)
4. LLM-as-judge

**Decision**: Semantic similarity via **Arbiter** (embedding-based)

**Rationale**:
- Arbiter provides continuous quality scores (0.0-1.0)
- Semantic similarity captures meaning, not just string matching
- Faster and cheaper than LLM-as-judge
- Compatible with bandit algorithms (need continuous rewards)

**Implementation**:
```python
from arbiter_ai import Arbiter

arbiter = Arbiter()
quality_score = arbiter.evaluate(
    query=query.query_text,
    response=model_response,
    reference=query.reference_answer,
)
```

---

## Bandit Algorithm Selection

### Decision: Support LinUCB, Thompson Sampling, UCB1, Epsilon-Greedy

**Date**: 2024-11-23

**Context**: Need to benchmark multiple bandit algorithms for routing.

**Algorithms included**:
1. **LinUCB**: Contextual bandit (uses query embeddings)
2. **Thompson Sampling**: Bayesian approach
3. **UCB1**: Upper confidence bound (stateless)
4. **Epsilon-Greedy**: Simple baseline
5. **Oracle**: Always select best model (upper bound)
6. **Random**: Random selection (lower bound)

**Rationale**:
- LinUCB: Best for contextual routing (uses embeddings)
- Thompson Sampling: Alternative Bayesian approach
- UCB1: Simple baseline, fast convergence
- Epsilon-Greedy: Industry standard baseline
- Oracle: Upper bound for comparison
- Random: Lower bound for sanity check

**Convergence requirements** (from experimental design):
- LinUCB: ~5,000 queries (high-dimensional context)
- Thompson Sampling: ~3,000 queries
- UCB1: ~2,000 queries (stateless)

**Dataset size target**: 10,000 queries (sufficient for all algorithms)

---

## Performance Optimization

### Decision: Oracle Caching for Cost Reduction

**Date**: 2024-11-24

**Context**: Running benchmarks is expensive (9 models × N queries).

**Strategy**:
1. Generate oracle results once (9 models × N queries)
2. Cache oracle routing decisions
3. Reuse oracle results for bandit algorithm comparison

**Cost savings**:
- Without caching: 225,000 LLM calls (5 algorithms × 9 models × 5,000 queries)
- With caching: 50,000 LLM calls (oracle + 4 bandits × 5,000)
- **Cost reduction: 78%**

**Time savings**:
- Without caching: ~10 hours
- With caching: ~2 hours
- **Time reduction: 80%**

**Implementation**:
```bash
# Step 1: Generate oracle once
uv run conduit-bench run \
  --dataset data/synthetic_10k.jsonl \
  --algorithms oracle \
  --output results/oracle.json

# Step 2: Reuse oracle for bandits
uv run conduit-bench run \
  --dataset data/synthetic_10k.jsonl \
  --algorithms linucb,thompson,ucb1,epsilon_greedy \
  --oracle-reference results/oracle.json \
  --output results/bandits.json
```

---

## Scalability Considerations

### Decision: Implement Scalable LinUCB (Future)

**Date**: 2024-11-26

**Context**: Standard LinUCB has O(d³) complexity with 387-dim features.

**Problem**:
- Computation: O(387³) = 59x slower than necessary
- Memory: O(387²) = 3.1x more than necessary
- Convergence: Requires 30×387 = 11,610 samples

**Proposed solution** (Issue #141):
- Implement Scalable LinUCB (arXiv:2510.19349)
- Low-rank parametrization (rank=64)
- 59x faster, 3.1x less memory
- Same quality as standard LinUCB

**Timeline**: After initial benchmarking validates standard LinUCB works

**Reference**: https://github.com/ashita-ai/conduit/issues/141

---

## Archive

### Rejected: RouterBench Dataset

**Date**: 2024-11-26

**Reason**: Incompatible with semantic evaluation (Arbiter)

**Archived files**:
- `archive/routerbench/routerbench_loader.py`
- `archive/routerbench/test_routerbench_adapter.py`
- `archive/routerbench/generate_routerbench_dataset.py`

**Kept for reference**: Implementation may be useful for future binary correctness benchmarks.

---

## Future Considerations

### Open Questions

1. **Template diversity**: Should we expand beyond 200 templates?
   - Current: 10 categories, 200+ templates
   - Risk: Pattern overfitting on templates
   - Mitigation: GSM8K validation catches overfitting

2. **Domain expansion**: Should we add more domain-specific datasets?
   - Candidates: HumanEval (code), MATH (advanced math), TriviaQA (knowledge)
   - **ELI5 (Planned)**: Reddit-style explanations for extreme generalization testing
     - Transfer distance: Code/Math → General explanations (maximum shift)
     - Tests if router learned truly generalizable strategies vs domain patterns
     - Expected: <40% performance drop indicates exceptional robustness
   - Benefit: Validates generalization across domains
   - Cost: Integration effort, API costs

3. **Real production data**: Should we benchmark on actual usage?
   - Benefit: Most realistic validation
   - Challenge: Privacy, labeling, diversity

4. **Multi-metric evaluation**: Should we use multiple evaluators?
   - Current: Arbiter (semantic similarity)
   - Alternatives: BLEU, ROUGE, BERTScore, LLM-as-judge
   - Trade-off: More robust vs more complex

### Decision Framework

For future dataset decisions, evaluate:
1. **Arbiter compatibility**: Full text reference answers required
2. **Production alignment**: Matches real use cases
3. **Diversity**: Covers multiple domains/categories
4. **Cost**: Generation/acquisition cost
5. **Validation value**: Tests generalization vs overfitting

---

## Change Log

- **2024-11-26**: Initial design decisions documented
- **2024-11-26**: Rejected RouterBench, archived implementation
- **2024-11-26**: Defined synthetic + GSM8K strategy
- **2024-11-26**: Documented transfer learning approach
