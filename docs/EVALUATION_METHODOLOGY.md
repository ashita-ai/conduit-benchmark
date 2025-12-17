# Evaluation Methodology

**Last Updated**: 2025-12-17

Comprehensive documentation of evaluation strategies, dataset characteristics, and quality assessment methods for conduit-benchmark.

---

## Table of Contents

1. [Evaluation Strategy Overview](#evaluation-strategy-overview)
2. [Dataset Comparison](#dataset-comparison)
3. [Evaluator Types](#evaluator-types)
4. [Quality Measurement](#quality-measurement)
5. [Cost & Performance](#cost--performance)

---

## Evaluation Strategy Overview

**Core Principle**: Use **objective evaluation only** - no LLM-as-judge to avoid circular dependencies.

| Dataset | Domain | Evaluator | Reason |
|---------|--------|-----------|--------|
| **MMLU** | Knowledge (57 subjects) | `exact_match` | Multiple choice (A/B/C/D) → binary correctness |
| **GSM8K** | Math reasoning | `exact_match` | Extract `#### N` format → binary correctness |
| **HumanEval** | Code generation | `code_execution` | Executable tests → objective pass/fail |

**Why No LLM-as-Judge (Arbiter)?**
- **Circular dependency**: Using LLMs to judge LLMs creates bias
- **Cost**: Each evaluation costs ~$0.001-$0.005
- **Non-determinism**: LLM judges produce variable scores
- **Reproducibility**: Exact match/code execution are fully reproducible

> **Note**: Arbiter is available for **production use** where no ground truth exists, but is NOT used for benchmarking.

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

**Evaluation Method**: Extract single letter (A/B/C/D) and compare to ground truth.

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

Ground Truth: "Natalia sold 48/2 = 24 clips in May.
              Natalia sold 48+24 = 72 clips altogether.
              #### 72"
```

**Evaluation Method**: Extract the numeric answer after `#### ` marker and compare to ground truth.

```python
def extract_gsm8k_answer(text: str) -> str | None:
    """Extract #### N format answer."""
    match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
    return match.group(1).replace(',', '') if match else None
```

**Why Exact Match (Not Arbiter)**:
- **Objective**: The `#### N` format provides unambiguous ground truth
- **Zero cost**: No LLM API calls needed
- **Reproducible**: Same input always produces same evaluation
- **No circular dependency**: Avoids LLM-as-judge bias
- **Fast**: Regex extraction is instant

**Quality Score**: 0.0 (incorrect or no `####` marker found) or 1.0 (correct)

**Note**: Models are prompted to use the `#### N` format. If a model doesn't follow this format, it receives a 0.0 score, which is a valid signal about instruction-following capability.

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

**Evaluation Method**: Execute generated code against unit tests.

**Why Code Execution**:
- **Objective correctness**: Code either passes tests or doesn't
- **No ambiguity**: Executable unit tests provide ground truth
- **Developer credibility**: Actual execution > LLM judgment
- **Standard benchmark**: Industry-standard evaluation method

**Quality Score**: 0.0 (any test fails) or 1.0 (all tests pass)

---

## Evaluator Types

### 1. Exact Match Evaluator

**Use Cases**: MMLU (multiple choice), GSM8K (math)

**How It Works**:
```python
class ExactMatchEvaluator(BaseEvaluator):
    def __init__(self, dataset_type: Literal["gsm8k", "mmlu"]):
        self.dataset_type = dataset_type

    def extract_answer(self, response: str) -> str | None:
        if self.dataset_type == "gsm8k":
            # Extract number after #### marker
            match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', response)
            return match.group(1).replace(',', '') if match else None
        else:  # mmlu
            # Extract A/B/C/D choice
            match = re.search(r'\b([ABCD])\b', response.upper())
            return match.group(1) if match else None

    def evaluate(self, response: str, ground_truth: str) -> float:
        predicted = self.extract_answer(response)
        expected = self._normalize_answer(ground_truth)
        return 1.0 if predicted == expected else 0.0
```

**Characteristics**:
- ✅ Zero cost (no API calls)
- ✅ Instant evaluation (<1ms)
- ✅ Deterministic (same input = same output)
- ✅ No circular dependency (doesn't use LLMs)
- ✅ Fully reproducible

---

### 2. Code Execution Evaluator

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

    # Execute in subprocess with timeout
    result = subprocess.run(
        ['python', '-c', full_code],
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

---

### 3. Arbiter Evaluator (Production Only)

**Use Cases**: Production routing where no ground truth exists

> **Important**: Arbiter is NOT used for benchmarking due to circular dependency concerns. It's available for production use where ground truth answers don't exist.

**How It Works**:
```python
from arbiter_ai import evaluate

async def evaluate_arbiter(response: str, reference: str) -> float:
    """Evaluate using Arbiter's semantic judges."""
    results = await evaluate(
        output=response,
        reference=reference,
        evaluators=["semantic", "factuality"],
        model="gpt-4o-mini"
    )
    return results.overall_score
```

**Characteristics**:
- ✅ Handles format variance
- ✅ Partial credit (0.0-1.0 continuous)
- ⚠️ **NOT used in benchmarks** - creates circular dependency
- ⚠️ Costs ~$0.001 per evaluation
- ⚠️ Non-deterministic

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

From full benchmark:

| Algorithm | Avg Quality | Distribution |
|-----------|-------------|--------------|
| thompson_sampling | ~0.70 | Binary (0 or 1) |
| ucb1 | ~0.68 | Binary (0 or 1) |
| epsilon_greedy | ~0.65 | Binary (0 or 1) |

**Interpretation**: Math reasoning is harder than knowledge recall (~70% vs ~90%)

---

## Cost & Performance

### Per-Evaluation Costs

| Evaluator | Cost | Latency | API Calls |
|-----------|------|---------|-----------|
| exact_match | $0.000 | <1ms | 0 |
| code_execution | $0.000 | ~2s | 0 |

### Full Benchmark Costs

**MMLU (1,000 queries, 11 algorithms)**:
- Model calls: 11,000 × $0.002 avg = **$22**
- Evaluation: 11,000 × $0 = **$0**
- **Total: ~$22**

**GSM8K (1,319 queries, 11 algorithms)**:
- Model calls: 14,509 × $0.003 avg = **$44**
- Evaluation: 14,509 × $0 = **$0** (exact match)
- **Total: ~$44**

**HumanEval (164 queries, 11 algorithms)**:
- Model calls: 1,804 × $0.003 avg = **$5**
- Evaluation: 1,804 × $0 = **$0** (code execution)
- **Total: ~$5**

---

## References

- **GSM8K Paper**: https://arxiv.org/abs/2110.14168
- **MMLU Paper**: https://arxiv.org/abs/2009.03300
- **HumanEval Paper**: https://arxiv.org/abs/2107.03374
