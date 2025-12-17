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

#### Why NOT Test Dueling Bandits

**What is Dueling Bandits**:
- **Algorithm**: FGTS.CDB (Fast Gradient Thompson Sampling for Contextual Dueling Bandits)
- **Feedback mechanism**: Pairwise comparisons (A vs B) instead of absolute quality scores
- **Availability**: Conduit supports via `algorithm="dueling"` parameter
- **Claims**: 30-50% faster convergence, better for human evaluation

**Why excluded from this benchmark**:
1. **Incompatible feedback mechanism**:
   - Dueling requires selecting 2 models and judging which is better (pairwise)
   - All other algorithms use absolute quality scores (0.0-1.0)
   - Not directly comparable in same experiment
2. **Infrastructure complexity**:
   - Would require modifying benchmark to evaluate model pairs
   - Different data flow: select_pair() vs select_arm()
   - Different update: DuelingFeedback vs BanditFeedback
3. **Evaluation complexity**:
   - Need to generate response from BOTH models for each query
   - Need pairwise comparison judge (2× the evaluation work)
   - Binary preference scores harder to compare with absolute rewards
4. **Research scope**:
   - Current experiment focuses on Thompson Sampling validation
   - Dueling bandits are a different research question entirely

**When Dueling makes sense**:
- Human-in-the-loop evaluation (comparing outputs more natural than rating)
- Subjective quality assessment (creative writing, style preferences)
- Relative performance matters more than absolute scores
- AB testing scenarios

**Future work**:
- Separate experiment comparing pairwise vs absolute feedback mechanisms
- Research question: "Do dueling bandits converge faster with human feedback?"
- Not a priority for current Thompson Sampling validation

#### Future Work: MATH Dataset

**If comprehensive hybrid validation needed**:
- Dataset: MATH (12,500 test queries)
- Sufficient data: 12,500 >> 4,000 needed for hybrid convergence
- Scope: All 11 algorithms with proper convergence
- Cost: $1,265-1,771 (full suite)
- Timeline: Post-Thompson validation (if needed)

**Expected outcome**: Thompson likely matches or exceeds hybrids even with ample data (research-backed)

---

## Reward Weights Configuration

### Decision: Use `user_facing` Preset for Benchmarks

**Date**: 2025-11-28

**Context**: Multi-objective routing requires balancing quality, latency, and cost. Conduit provides 5 production presets in `conduit.yaml`.

**Decision**: Benchmarks use **`user_facing` preset** (quality: 60%, latency: 30%, cost: 10%)

**Available presets**:

| Preset | Quality | Latency | Cost | Use Case |
|--------|---------|---------|------|----------|
| **`user_facing`** | 60% | 30% | 10% | Most production apps (DEFAULT) |
| `internal_tools` | 55% | 25% | 20% | Dev tools, automation, benchmarks |
| `realtime` | 50% | 40% | 10% | Live chat, search, autocomplete |
| `batch` | 50% | 10% | 40% | Background jobs, reports |
| `critical` | 85% | 10% | 5% | Medical, legal, financial |

**Rationale**:

#### Why `user_facing` for Benchmarks

**Represents 80% of use cases**:
- Customer-facing applications where quality matters
- Users notice and hate waiting (latency important)
- Cost is secondary to user experience

**Aligns with production defaults**:
- Conduit's default optimization is `user_facing`
- Benchmark results reflect typical production behavior
- Most users won't change default weights

**Balances all three objectives**:
- Quality (60%): Correct answers critical
- Latency (30%): Users notice delays
- Cost (10%): Optimize after UX is good

#### Future Research Area

**Research question**: How do different reward weights affect algorithm performance?

**Hypothesis**: Algorithms may converge differently with alternative presets:
- `critical` (85% quality): May favor expensive models earlier
- `batch` (40% cost): May converge faster to cheap models
- `realtime` (40% latency): May favor fast models over accurate ones

**Experiment design**:
- Run benchmarks with all 5 presets
- Compare convergence speed and final performance
- Identify preset-specific algorithm advantages

**Cost**: 5× current benchmark cost (~$1,000-1,500)
**Timeline**: Post-Thompson validation
**Priority**: Low - not required for default algorithm decision

**Why not now**:
- Current focus is Thompson Sampling validation
- `user_facing` represents typical production use
- Research can wait until after primary validation complete

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

## Model Failure Handling

### Decision: Automatic Fallback on Model Refusal/Failure

**Date**: 2025-12-04

**Context**: Some LLM models may refuse certain queries (e.g., sensitive topics, safety guardrails) or fail due to validation errors. This can cause benchmark runs to abort.

**Problem discovered**: Query `mmlu_test_5` (about biological weapons weaponization) caused `claude-opus-4-5-20251101` to fail with `UnexpectedModelBehavior: Exceeded maximum retries for output validation`. Algorithms that happened to select this model for this query failed entirely.

**Decision**: Implement **automatic fallback chain** when primary model fails

**Behavior**:
1. Execute query against bandit-selected model (primary)
2. If primary fails (refusal, validation error, timeout), try fallback models
3. Fallback order: sorted by expected quality descending (top 3 alternatives)
4. Log all failures and successful fallback for analysis
5. Penalize failed models with quality=0.0 in bandit feedback
6. Only abort if ALL models (primary + 3 fallbacks) fail

**Implementation**:
```python
# Default fallback when algorithm doesn't provide get_fallback_chain()
other_arms = [arm for arm in algorithm.arm_list if arm.model_id != selected_arm.model_id]
fallback_arms = sorted(other_arms, key=lambda a: a.expected_quality, reverse=True)[:3]

# Execute with fallback support
execution_result = await executor.execute_with_fallback(
    primary_arm=selected_arm,
    fallback_arms=fallback_arms,  # Now populated, not empty
    query_text=query.query_text,
)

# Log fallback usage
if execution_result.was_fallback and execution_result.success:
    console.print(f"⚠ Primary model {execution_result.primary_model} failed, used {execution_result.model_id}")
```

**Rationale**:

#### Why Fallback (Not Abort)

**Problems with abort-on-failure**:
- Single model refusal kills entire benchmark run
- Wastes compute time and API costs from completed queries
- Different algorithms affected by luck of model selection
- Unfair comparison (some algorithms hit problematic queries, others don't)

**Advantages of fallback**:
- Benchmark continues despite individual model issues
- All algorithms evaluated on same query set
- Failed models still penalized (quality=0.0 feedback)
- Refusal patterns captured in logs for analysis

#### Why Penalize Failed Models

**When primary model fails**:
- Record `quality_score=0.0` for the failed model
- Bandit algorithm learns to avoid that model for similar queries
- Successful fallback model gets actual quality score

**This is fair because**:
- A model that refuses to answer IS worse than one that answers
- Production routing should avoid unreliable models
- Captures real-world model behavior differences

#### Known Sensitive Query Patterns

Based on benchmark runs, these query types trigger model refusals:
- **MMLU security_studies**: Questions about weapons, biological agents, military strategy
- **Code generation**: Potentially harmful code (exploits, malware patterns)
- **Medical/legal advice**: Queries that could be interpreted as professional advice

**Mitigation**: Fallback chain ensures benchmark continues; logs capture patterns for analysis.

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

- **2025-12-04**: Implemented automatic fallback chain for model refusals/failures
- **2025-12-04**: Fixed parallel execution to use `return_exceptions=True` for resilience
- **2025-12-04**: Fixed LinUCB feature dimension mismatch (1538 vs 386) by passing QueryAnalyzer feature_dim
- **2025-11-28**: Documented reward weights configuration (`user_facing` preset)
- **2025-11-28**: Added future research area: multi-preset benchmarking
- **2025-11-27**: Documented dueling bandits support and exclusion rationale
- **2025-11-27**: MAJOR REDESIGN - Real benchmarks + objective evaluation
- **2025-11-27**: Algorithm reduction: 11 → 4 (Thompson validation focus)
- **2025-11-27**: Model canonicalization (6 current-gen models with API names)
- **2025-11-27**: Added MATH dataset as future research direction
- **2025-11-27**: Deprecated synthetic data + Arbiter approach
- **2024-11-26**: Initial design decisions documented (DEPRECATED)
- **2024-11-26**: Rejected RouterBench, archived implementation
- **2024-11-26**: Defined synthetic + GSM8K strategy (DEPRECATED)
- **2024-11-26**: Documented transfer learning approach (DEPRECATED)
