# Design Decisions

Critical architectural and dataset decisions for conduit-benchmark.

---

## 🔄 MAJOR REDESIGN (2025-11-27)

### Decision: Real Benchmarks + Objective Evaluation (Thompson Sampling Validation)

**Date**: 2025-11-27

**Context**: Redesigned experiment based on two critical findings:
1. **Data constraint**: Total 2,483 queries < 3,000-4,000 needed for hybrid algorithm convergence
2. **Research evidence**: BayesianRouter (arXiv 2510.02850) shows Thompson Sampling > LinUCB/hybrids for LLM routing

**Previous approach** (DEPRECATED):
- Synthetic data generation + Arbiter evaluation
- 11 algorithms including 5 hybrid variants
- 10,000 synthetic queries + GSM8K validation

**Current approach**:
- Real benchmark datasets with objective evaluation
- 4 core algorithms (Thompson validation focus)
- 2,483 queries across 3 established benchmarks

---

## Dataset Strategy

### Decision: Established Benchmarks (GSM8K, MMLU, HumanEval)

**Date**: 2025-11-27

**Context**: Need credible, reproducible benchmarks with objective evaluation methods.

**Decision**: Use **established benchmarks** with **objective evaluation**

**Datasets selected**:

| Dataset | Size | Domain | Evaluation Method | Rationale |
|---------|------|--------|-------------------|-----------|
| **GSM8K** | 1,319 | Math reasoning | Exact match (`#### N`) | Objective, no LLM-as-judge needed |
| **MMLU** | 1,000 | Knowledge (57 subjects) | Exact match (A/B/C/D) | Broad domain coverage |
| **HumanEval** | 164 | Python coding | Code execution | Most credible to developers |
| **Total** | **2,483** | Multi-domain | Objective only | Reproducible, unbiased |

**Rationale**:

#### Why Established Benchmarks

**Advantages**:
- **Credibility**: Well-known, peer-reviewed datasets
- **Reproducibility**: Other researchers can replicate results
- **Objective evaluation**: No LLM-as-judge circular dependency
- **Domain diversity**: Math, knowledge, code
- **No overfitting risk**: Real-world problems, not synthetic templates

**Key constraint**: 2,483 total queries limits algorithm testing (see "Algorithm Selection" below)

#### Why GSM8K (Grade School Math)

**Source**: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)

**Advantages**:
- 1,319 test problems with step-by-step solutions
- Answer format: `#### 72` - objectively correct/incorrect
- No LLM-as-judge needed - eliminates circular dependency
- Tests reasoning capability across models

**Evaluation**: Extract `#### N` from response, exact match against ground truth

#### Why MMLU (Massive Multitask Language Understanding)

**Source**: [cais/mmlu](https://huggingface.co/datasets/cais/mmlu)

**Advantages**:
- 14,042 questions across 57 subjects (using 1k subset)
- Multiple choice (A/B/C/D) - exact match on answer
- Broad coverage: STEM, humanities, social sciences
- Tests knowledge breadth across models

**Evaluation**: Extract A/B/C/D from response, exact match against ground truth

#### Why HumanEval (OpenAI Code Benchmark)

**Source**: [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval)

**Advantages**:
- 164 Python function completion problems
- Execute code + run unit tests - pass/fail evaluation
- Most credible to developers - executable tests > LLM judges
- Tests practical coding ability

**Evaluation**: Execute generated code with test cases, pass (1.0) or fail (0.0)

#### Why NOT Synthetic Data (for benchmarking)

**Rejected approach**: Template-based synthetic generation

**Reasons**:
- Pattern overfitting risk on templates
- Less credible than established benchmarks
- Not reproducible by other researchers
- Required Arbiter (LLM-as-judge) - circular dependency

**Note**: Synthetic data may still be useful for production usage where no ground truth exists

---

## Evaluation Architecture

### Decision: Objective Evaluation (Exact Match + Code Execution)

**Date**: 2025-11-27

**Context**: Need unbiased, reproducible evaluation methods without circular dependency.

**Options Considered**:
1. **Arbiter** (LLM-as-judge via semantic similarity)
2. **Exact match** (string comparison)
3. **Code execution** (run tests)
4. **Human evaluation** (manual grading)

**Decision**: **Objective evaluation only** - Exact match for GSM8K/MMLU, Code execution for HumanEval

**Rationale**:

#### Why NOT Arbiter (LLM-as-judge)

**Problem**: Circular dependency - using LLM to judge LLM routing creates bias

**Issues**:
- LLM-as-judge may favor certain model styles over others
- Not reproducible (evaluation model versions change)
- Adds cost and latency to benchmarking
- Introduces evaluation noise and bias

**Conclusion**: Arbiter is valuable for **production use** (no ground truth), but inappropriate for **benchmarking**

#### Why Exact Match (GSM8K, MMLU)

**Implementation**:
```python
# GSM8K: Extract "#### N" from response
def extract_gsm8k_answer(text: str) -> str | None:
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    return match.group(1).replace(',', '') if match else None

# MMLU: Extract A/B/C/D from response
def extract_mmlu_answer(text: str) -> str | None:
    match = re.search(r'\b([ABCD])\b', text.upper())
    return match.group(1) if match else None

# Evaluation: Binary reward
reward = 1.0 if predicted == expected else 0.0
```

**Advantages**:
- Completely objective - no bias
- Reproducible - same results every run
- Fast - no additional API calls
- Free - no evaluation costs

**Limitation**: Binary reward signal (may slow bandit learning vs continuous rewards)

#### Why Code Execution (HumanEval)

**Implementation**:
```python
# Combine prompt + model response + test cases
full_code = f"{prompt}{response}\n\n{test_code}\ncheck({entry_point})"

# Execute in sandboxed subprocess with timeout
result = subprocess.run(['python', temp_file], timeout=10, capture_output=True)
reward = 1.0 if result.returncode == 0 else 0.0
```

**Advantages**:
- Objective - code either works or doesn't
- Credible - developers trust executable tests
- Realistic - matches production code evaluation
- Comprehensive - tests functionality, not just style

**Safety**: Executed in subprocess with timeout to prevent infinite loops or system harm

---

## Bandit Algorithm Selection

### Decision: Focus on Thompson Sampling Validation (4 Core Algorithms)

**Date**: 2025-11-27

**Context**: Two critical constraints emerged:
1. **Data limitation**: 2,483 total queries < 3,000-4,000 needed for hybrid/contextual algorithms
2. **Research evidence**: BayesianRouter (arXiv 2510.02850) shows Thompson Sampling > LinUCB/hybrids for LLM routing

**Previous plan** (DEPRECATED):
- 11 algorithms (5 hybrids + 4 standalone + 2 baselines)
- Intended to validate HybridRouter (Thompson → LinUCB)
- Required 3,000-4,000 queries for proper hybrid convergence

**Current plan**:
- **4 core algorithms** focused on Thompson Sampling validation
- Align with conduit default change (GitHub conduit#169)

**Algorithms selected**:

| Algorithm | Type | Convergence Queries | Role |
|-----------|------|-------------------|------|
| **Thompson Sampling** | Non-contextual Bayesian | 100-500 | **PRIMARY** - Proposed default |
| **UCB1** | Non-contextual UCB | 100-500 | Comparison baseline |
| **Epsilon-Greedy** | Non-contextual ε-greedy | 100-500 | Comparison baseline |
| **Random** | Uniform selection | 0 (no learning) | Lower bound |

**Rationale**:

#### Why Thompson Sampling as Primary

**Research backing** (BayesianRouter, arXiv 2510.02850):
- Thompson Sampling SUPERIOR to LinUCB for LLM routing
- LinUCB problems with LLM domains:
  - Premature exploitation with delayed feedback
  - Over-exploitation leads to collapse to single arm
  - Requires significantly more data to converge
- Thompson advantages:
  - Robust to delayed feedback
  - Maintains exploration-exploitation balance
  - Works effectively from query 1
  - No high-dimensional context needed

**Convergence**: 100-500 queries (well within 2,483 available)

**Goal**: Validate Thompson as optimal default for conduit (supports GitHub conduit#169)

#### Why NOT Test Hybrids/LinUCB

**Data constraint**:
- Hybrid algorithms (e.g., Thompson → LinUCB) need 3,000-4,000 queries
- LinUCB alone needs 650-1,300 queries (5-10× feature dimensionality with PCA)
- We have 2,483 queries (insufficient for proper evaluation)

**Research evidence**:
- BayesianRouter already shows Thompson > LinUCB/hybrids
- No need to re-validate what research has established
- Focus on empirical Thompson validation instead

**Conclusion**: Not worth testing algorithms we can't properly evaluate

#### Future Work: MATH Dataset

**If comprehensive hybrid validation needed**:
- Dataset: MATH (12,500 test queries)
- Sufficient data: 12,500 >> 4,000 needed for hybrid convergence
- Scope: All 11 algorithms with proper convergence
- Cost: $1,265-1,771 (full suite)
- Timeline: Post-Thompson validation (if needed)

**Expected outcome**: Thompson likely matches or exceeds hybrids even with ample data (research-backed)

---

## Model Canonicalization

### Decision: Use Current-Generation Models with Actual API Names

**Date**: 2025-11-27

**Context**: Need to benchmark against current production models, not outdated versions.

**Models selected** (6 current-generation):

| API Model ID | Provider | Tier | Notes |
|--------------|----------|------|-------|
| `gpt-4o-mini` | OpenAI | Budget | Fast, cheap |
| `gpt-4-turbo` | OpenAI | Flagship | High quality |
| `claude-sonnet-4-5-20250929` | Anthropic | Balanced | Best for code |
| `claude-opus-4-5-20251101` | Anthropic | Premium | Highest quality |
| `gemini-2.5-pro` | Google | Flagship | Stable |
| `gemini-3-pro-preview` | Google | Cutting Edge | Preview (may change) |

**Documentation sources**:
- OpenAI: https://platform.openai.com/docs/models
- Anthropic: https://platform.claude.com/docs/en/about-claude/models/all-models
- Google: https://ai.google.dev/gemini-api/docs/models

**Rationale**:

#### Why These Models

**Tier distribution**:
- Budget: 1 model (gpt-4o-mini)
- Balanced: 2 models (gpt-4-turbo, claude-sonnet-4-5)
- Premium: 2 models (claude-opus-4-5, gemini-2.5-pro)
- Cutting edge: 1 model (gemini-3-pro-preview)

**Provider diversity**: 2 OpenAI, 2 Anthropic, 2 Google (balanced representation)

**Cost variation**: Wide range from cheap (gpt-4o-mini) to expensive (claude-opus-4-5) - tests router's cost-quality tradeoff learning

#### Why Actual API Names (Not Aliases)

**Problem with aliases** (e.g., using "gpt-5" instead of "gpt-4o"):
- Research reproducibility issues (aliases change over time)
- Documentation confusion (official docs use API names)
- Version ambiguity (which model was actually tested?)

**Solution**: Use exact API model IDs from provider documentation

**Frozen for reproducibility**: These exact model IDs are frozen in EXPERIMENTAL_DESIGN.md - changing models invalidates all benchmark results

## Performance Optimization

### Decision: Parallel Execution + Database Streaming

**Date**: 2025-11-27

**Context**: Minimize benchmark runtime while maintaining data integrity.

**Optimizations implemented**:

1. **Parallel algorithm execution**:
   - All algorithms run concurrently (not sequential)
   - Max concurrency: 10 (configurable)
   - Time savings: ~65% vs sequential

2. **Database streaming writes**:
   - Results written to database as they complete
   - No catastrophic data loss if benchmark interrupted
   - Enables real-time monitoring

3. **Algorithm reduction** (11 → 4):
   - Focus on Thompson validation only
   - Cost savings: 64% ($195-270 vs $600-840)
   - Time savings: 65% (2.5-4 hours vs 12-17 hours)

**Total efficiency gain**:
- **Original plan**: 11 algorithms × 2,483 queries = $600-840, 12-17 hours
- **Optimized plan**: 4 algorithms × 2,483 queries = $195-270, 2.5-4 hours
- **Savings**: 64% cost, 65% time

---

## Production Concerns (Conduit Library)

### PCA Persistence Architecture Issue

**Date**: 2025-11-27 (Identified, not yet resolved)

**Problem**: PCA models currently stored in filesystem (`~/.cache/conduit/`), should be in database

**Issues**:
- Cache loss on instance restart
- No multi-instance sharing (each instance trains own PCA)
- No versioning or rollback capability
- No auditability of PCA changes

**Impact**: Not blocking benchmark execution, but affects production deployment

**Resolution**: Future work for conduit library (separate from benchmark project)

---

## Future Considerations

### MATH Dataset - Comprehensive Hybrid Validation

**Date**: 2025-11-27

**Context**: Current benchmark (2,483 queries) insufficient for hybrid algorithm testing.

**Proposal**:
- **Dataset**: MATH (12,500 test queries)
- **Source**: [hendrycks/math](https://huggingface.co/datasets/hendrycks/math)
- **Evaluation**: Exact match on final answer
- **Sufficient data**: 12,500 >> 4,000 needed for hybrid convergence

**Scope**:
- All 11 algorithms (5 hybrids + 4 standalone + 2 baselines)
- 3 independent runs for statistical significance
- Cost: $1,265-1,771 (full suite)
- Time: 16-24 hours

**Research questions**:
1. Do hybrids outperform pure Thompson with sufficient data?
2. What is the optimal switch_threshold for production?
3. Does LinUCB phase provide value over pure Thompson?
4. How does PCA dimensionality reduction impact convergence?

**Expected outcome**: Thompson likely matches or exceeds hybrids even with ample data (research-backed)

**Priority**: Low - not required for conduit default decision (GitHub conduit#169)

### Domain Expansion Candidates

**Post-Thompson validation**, consider adding:
1. **TriviaQA**: Knowledge retrieval and factual accuracy
2. **HellaSwag**: Commonsense reasoning
3. **DROP**: Reading comprehension and numerical reasoning

**Criteria for inclusion**:
1. **Objective evaluation**: Exact match or code execution (no LLM-as-judge)
2. **Production alignment**: Matches real use cases
3. **Domain diversity**: Tests different model strengths
4. **Cost effectiveness**: Reasonable API costs

---

## Archive

### Deprecated: Synthetic Data + Arbiter Evaluation

**Date**: 2025-11-27

**Reason**: Replaced with established benchmarks + objective evaluation

**Why deprecated**:
- Synthetic data risks pattern overfitting
- Arbiter (LLM-as-judge) creates circular dependency
- Established benchmarks more credible and reproducible

**Original approach** (archived):
- 10,000 synthetic queries (10 categories, 200+ templates)
- Semantic similarity evaluation via Arbiter
- Transfer learning validation with GSM8K

**Current approach**: Real benchmarks (GSM8K, MMLU, HumanEval) with objective evaluation

### Rejected: RouterBench Dataset

**Date**: 2024-11-26

**Reason**: Incompatible with both Arbiter and objective evaluation needs

**Why rejected**:
- Multiple choice format (A/B/C/D reference answers)
- Not suitable for semantic similarity evaluation
- Better alternatives available (MMLU for multiple choice)

---

## Change Log

- **2025-11-27**: MAJOR REDESIGN - Real benchmarks + objective evaluation
- **2025-11-27**: Algorithm reduction: 11 → 4 (Thompson validation focus)
- **2025-11-27**: Model canonicalization (6 current-gen models with API names)
- **2025-11-27**: Added MATH dataset as future research direction
- **2025-11-27**: Deprecated synthetic data + Arbiter approach
- **2024-11-26**: Initial design decisions documented (DEPRECATED)
- **2024-11-26**: Rejected RouterBench, archived implementation
- **2024-11-26**: Defined synthetic + GSM8K strategy (DEPRECATED)
- **2024-11-26**: Documented transfer learning approach (DEPRECATED)
