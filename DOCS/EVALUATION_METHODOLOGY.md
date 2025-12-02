# Evaluation Methodology

**Last Updated**: 2025-12-01

Comprehensive documentation of evaluation strategies, dataset characteristics, and quality assessment methods for conduit-benchmark.

---

## Table of Contents

1. [Evaluation Strategy Overview](#evaluation-strategy-overview)
2. [Dataset Comparison](#dataset-comparison)
3. [Evaluator Types](#evaluator-types)
4. [Why Arbiter for GSM8K](#why-arbiter-for-gsm8k)
5. [Quality Measurement](#quality-measurement)
6. [Cost & Performance](#cost--performance)

---

## Evaluation Strategy Overview

**Core Principle**: Use the most appropriate quality measurement for each dataset's domain and answer format.

| Dataset | Domain | Evaluator | Reason |
|---------|--------|-----------|--------|
| **MMLU** | Knowledge (57 subjects) | `exact_match` | Multiple choice (A/B/C/D) → binary correctness |
| **GSM8K** | Math reasoning | `arbiter` | Step-by-step reasoning → nuanced quality scoring |
| **HumanEval** | Code generation | `code_execution` | Executable tests → objective pass/fail |

---

## Dataset Comparison

### MMLU (Massive Multitask Language Understanding)

**Source**: [cais/mmlu](https://huggingface.co/datasets/cais/mmlu)
**Size**: 1,000 questions (subset of 14,042)
**Domain**: Knowledge across 57 subjects (STEM, humanities, social sciences)

**Answer Format**:
```
Question: What is the capital of France?
(A) London
(B) Berlin
(C) Paris
(D) Madrid

Correct Answer: C
```

**Why Exact Match**:
- **Unambiguous correctness**: Only one answer is correct
- **Binary evaluation**: Either matches ground truth or doesn't
- **Zero cost**: No LLM calls needed for evaluation
- **Instant results**: Regex extraction + string comparison
- **No circular dependency**: Doesn't use LLMs to judge LLMs

**Quality Score**: 0.0 (incorrect) or 1.0 (correct)

---

### GSM8K (Grade School Math)

**Source**: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)
**Size**: 1,319 test problems
**Domain**: Multi-step math word problems requiring reasoning

**Answer Format**:
```
Question: Natalia sold clips to 48 friends in April and half as many in May.
          How many clips did she sell altogether?

Expected Answer: "Natalia sold 48/2 = 24 clips in May.
                  Natalia sold 48+24 = 72 clips altogether.
                  #### 72"

Model Response: "Let's solve step by step:
                 - April sales: 48 clips
                 - May sales: 48 ÷ 2 = 24 clips
                 - Total: 48 + 24 = 72 clips

                 The answer is 72."
```

**Why Arbiter (Not Exact Match)**:

❌ **Exact Match Fails**:
- **Format variance**: Models don't reliably output `#### N` format
- **Expression differences**: "72" vs "72 clips" vs "seventy-two"
- **Missing reasoning**: Short answer misses quality of explanation
- **Binary scoring**: Can't distinguish partial credit scenarios

✅ **Arbiter Succeeds**:
- **Semantic understanding**: Recognizes "72 clips" = "#### 72"
- **Reasoning quality**: Evaluates step-by-step logic, not just final answer
- **Partial credit**: 0.0-1.0 scoring for partially correct reasoning
- **Robust extraction**: Handles format variations gracefully

**Quality Score**: 0.0-1.0 (continuous)
- 1.0: Correct answer with sound reasoning
- 0.6-0.8: Correct answer with minor logical gaps
- 0.3-0.5: Partially correct or right approach, wrong answer
- 0.0: Completely incorrect

---

### HumanEval (Code Generation)

**Source**: [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval)
**Size**: 164 Python function problems
**Domain**: Function implementation with unit tests

**Answer Format**:
```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Check if any two numbers in list are closer than threshold."""
    # Model generates implementation
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False

# Evaluation via unit tests:
assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False
assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0], 0.3) == True
```

**Why Code Execution**:
- **Objective correctness**: Code either passes tests or doesn't
- **No ambiguity**: Executable unit tests provide ground truth
- **Developer credibility**: Actual execution > LLM judgment
- **Standard benchmark**: Industry-standard evaluation method

**Quality Score**: 0.0 (any test fails) or 1.0 (all tests pass)

---

## Evaluator Types

### 1. Exact Match Evaluator

**Use Cases**: MMLU (multiple choice)

**How It Works**:
```python
def evaluate_exact_match(response: str, ground_truth: str) -> float:
    """Extract answer and compare with ground truth."""
    # Extract answer (A, B, C, or D)
    match = re.search(r'\b([ABCD])\b', response.upper())
    predicted = match.group(1) if match else None

    # Binary comparison
    return 1.0 if predicted == ground_truth else 0.0
```

**Characteristics**:
- ✅ Zero cost (no API calls)
- ✅ Instant evaluation (<1ms)
- ✅ Deterministic (same input = same output)
- ✅ No circular dependency (doesn't use LLMs)
- ❌ Binary only (no partial credit)
- ❌ Requires unambiguous answer format

---

### 2. Arbiter Evaluator

**Use Cases**: GSM8K (math reasoning)

**How It Works**:

Arbiter is an **LLM-as-judge framework** that evaluates responses using multiple criteria:

```python
from arbiter_ai import evaluate

async def evaluate_arbiter(
    query: str,
    response: str,
    ground_truth: str
) -> float:
    """Evaluate using Arbiter's semantic + factuality judges."""

    # Run evaluation with multiple judges
    results = await evaluate(
        query=query,
        response=response,
        reference=ground_truth,
        evaluators=["semantic", "factuality"],
        model="gpt-4o-mini"  # Judge model
    )

    # Aggregate scores (average of judges)
    return (results["semantic"] + results["factuality"]) / 2
```

**Evaluator Components**:

1. **Semantic Evaluator**: Query-response similarity
   - Measures if response addresses the question
   - Checks logical flow and reasoning
   - Scores 0.0-1.0

2. **Factuality Evaluator**: Correctness vs. ground truth
   - Compares final answer to expected result
   - Handles format variations ("72" vs "#### 72")
   - Scores 0.0-1.0

**Characteristics**:
- ✅ Handles format variance (robust extraction)
- ✅ Partial credit (0.0-1.0 continuous scoring)
- ✅ Evaluates reasoning quality (not just final answer)
- ✅ Semantic understanding (recognizes equivalent expressions)
- ⚠️ **Costs ~$0.001 per evaluation** (judge LLM calls)
- ⚠️ **Slower ~5-10 sec per evaluation** (API latency)
- ⚠️ **API reliability dependency** (timeouts possible)
- ⚠️ Non-deterministic (small variance in scores)

**Configuration Settings** (`conduit.yaml`):

```yaml
arbiter:
  sample_rate: 0.1                # Evaluate 10% of responses (cost control)
  daily_budget: 10.0              # Maximum $10/day on evaluations
  model: "gpt-5"                  # Judge model for evaluation
  evaluators:                     # Active evaluator types
    - semantic                    # Query-response alignment
    - factuality                  # Ground truth accuracy
```

**Rationale for Configuration**:
- **`model: gpt-5`**: High-quality judge model for accurate evaluations
- **`evaluators: [semantic, factuality]`**: Dual-judge approach balances reasoning quality and correctness
- **`sample_rate: 0.1`**: Budget-conscious setting (not used in benchmarks - we evaluate 100%)
- **`daily_budget: 10.0`**: Safety limit to prevent runaway costs

**Judge Model Selection**:
| Model | Cost (per eval) | Speed | Quality | Choice |
|-------|----------------|-------|---------|--------|
| gpt-5.1 | ~$0.005 | 5-10s | Excellent | Premium option |
| **gpt-5** | **~$0.002** | **5-10s** | **Excellent** | ✅ **Selected** |
| gpt-5-mini | ~$0.001 | 3-5s | Very Good | Faster alternative |

---

### 3. Code Execution Evaluator

**Use Cases**: HumanEval (code generation)

**How It Works**:
```python
def evaluate_code_execution(
    prompt: str,
    response: str,
    test_code: str,
    entry_point: str
) -> float:
    """Execute code and run unit tests."""

    # Build complete code
    full_code = f"{prompt}{response}\n\n{test_code}\ncheck({entry_point})"

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as f:
        f.write(full_code)
        f.flush()

        # Execute in subprocess with timeout
        result = subprocess.run(
            ['python', f.name],
            timeout=10,
            capture_output=True
        )

    # Return 1.0 if all tests passed, 0.0 otherwise
    return 1.0 if result.returncode == 0 else 0.0
```

**Characteristics**:
- ✅ Objective (executable tests)
- ✅ Zero cost (local execution)
- ✅ Fast (~2 seconds including execution)
- ✅ Deterministic
- ⚠️ Requires sandbox security (subprocess isolation)
- ⚠️ Binary only (no partial credit)

---

## Why Arbiter for GSM8K

### The Problem with Exact Match

**Original approach** (documented in BENCHMARKING.md):
```python
def extract_gsm8k_answer(text: str) -> str | None:
    """Extract #### N format answer."""
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    return match.group(1).replace(',', '') if match else None
```

**Failure modes**:

1. **Format non-compliance**:
   ```
   Expected: "#### 72"
   Got:      "The answer is 72 clips"  ❌ No match
   Got:      "72"                       ❌ No #### marker
   Got:      "seventy-two"              ❌ Non-numeric
   ```

2. **Quality blindness**:
   ```
   Response A: "48 + 24 = 72. Correct reasoning! #### 72"  ✅ (1.0)
   Response B: "I'll guess #### 72"                        ✅ (1.0)
   ```
   Both get 1.0, but B has no reasoning!

3. **Lost nuance**:
   ```
   Response: "48 ÷ 2 = 20 (wrong), 48 + 20 = 68. #### 68"
   ```
   - Shows problem-solving attempt
   - Arithmetic error only
   - Deserves partial credit (0.4-0.6)
   - Exact match: 0.0 ❌

### Arbiter's Advantages

**Example evaluation**:

```python
Query: "Natalia sold 48 clips in April, half as many in May. Total?"
Ground Truth: "#### 72"

Response 1: "48 ÷ 2 = 24, 48 + 24 = 72. Answer: 72 clips"
Arbiter Score: 1.0
  - Semantic: 1.0 (addresses question, clear reasoning)
  - Factuality: 1.0 (correct answer)

Response 2: "I think the answer is 72"
Arbiter Score: 0.6
  - Semantic: 0.2 (no reasoning shown)
  - Factuality: 1.0 (correct answer)

Response 3: "48 ÷ 2 = 20, 48 + 20 = 68"
Arbiter Score: 0.4
  - Semantic: 0.8 (correct approach, arithmetic error)
  - Factuality: 0.0 (wrong answer)
```

**Benefits**:
- ✅ Captures reasoning quality
- ✅ Handles format variations
- ✅ Provides nuanced scoring (0.0-1.0)
- ✅ Aligns with educational assessment (partial credit)

---

## Quality Measurement

### MMLU Quality Distribution

From 1,000-query benchmark:

| Algorithm | Avg Quality | Distribution |
|-----------|-------------|--------------|
| dueling_bandit | 0.932 | Binary (0 or 1) |
| epsilon_greedy | 0.905 | Binary (0 or 1) |
| thompson_sampling | 0.903 | Binary (0 or 1) |

**Interpretation**: 90%+ accuracy on knowledge questions

---

### GSM8K Quality Distribution

From 10-query test:

| Algorithm | Avg Quality | Distribution |
|-----------|-------------|--------------|
| thompson_sampling | 0.633 | 0.3, 0.6, 1.0 (varied) |
| ucb1 | 0.573 | 0.4, 0.6, 0.7, 0.8, 1.0 |
| epsilon_greedy | 0.520 | 0.4, 0.6 (mixed) |

**Interpretation**:
- Math reasoning is harder (60% vs 90%)
- Partial credit reveals learning quality
- More granular quality assessment

---

## Cost & Performance

### Per-Evaluation Costs

| Evaluator | Cost | Latency | API Calls |
|-----------|------|---------|-----------|
| exact_match | $0.000 | <1ms | 0 |
| arbiter | ~$0.001 | ~5-10s | 2 (semantic + factuality) |
| code_execution | $0.000 | ~2s | 0 |

### Full Benchmark Costs

**MMLU (1,000 queries, 11 algorithms)**:
- Model calls: 11,000 × $0.002 avg = **$22**
- Evaluation: 11,000 × $0 = **$0**
- **Total: ~$22**

**GSM8K (1,319 queries, 11 algorithms)**:
- Model calls: 14,509 × $0.003 avg = **$44**
- Arbiter evaluation: 14,509 × $0.001 = **$15**
- **Total: ~$59**

**Trade-off**: GSM8K costs 2.7× more but provides nuanced quality assessment for math reasoning.

---

## Methodology Decisions

### Why Not Exact Match for All?

**Exact match works when**:
- Answers have standardized format (A/B/C/D)
- Correctness is unambiguous
- No partial credit needed

**Exact match fails when**:
- Answers have format variance
- Reasoning quality matters
- Partial credit improves signal

### Why Not Arbiter for All?

**Arbiter adds value when**:
- Reasoning quality matters (math, essays)
- Format varies significantly
- Partial credit is meaningful

**Arbiter is overkill when**:
- Binary correctness suffices (multiple choice)
- Evaluation cost outweighs benefit
- Speed is critical

### Current Assignment Rationale

| Dataset | Answer Type | Variance | Quality Granularity | Evaluator |
|---------|-------------|----------|---------------------|-----------|
| MMLU | Fixed (A/B/C/D) | None | Binary | exact_match ✅ |
| GSM8K | Free-form text | High | Continuous | arbiter ✅ |
| HumanEval | Code | Varies | Binary | code_execution ✅ |

---

## References

- **Arbiter AI Framework**: https://arbiter-ai.readthedocs.io
- **GSM8K Paper**: https://arxiv.org/abs/2110.14168
- **MMLU Paper**: https://arxiv.org/abs/2009.03300
- **HumanEval Paper**: https://arxiv.org/abs/2107.03374
- **LLM-as-Judge Survey**: https://arxiv.org/abs/2306.05685
