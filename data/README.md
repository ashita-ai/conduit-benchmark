# Synthetic Dataset Generation Methodology

This directory contains synthetically generated programming challenge datasets used for benchmarking LLM routing algorithms.

## Generation Method

All synthetic datasets in this directory were generated using the standardized prompt defined in [`GENERATE_SYNTHETIC_DATASET_PROMPT.md`](./GENERATE_SYNTHETIC_DATASET_PROMPT.md).

### Process

1. **Prompt Template**: A detailed prompt specifying format, requirements, and quality standards
2. **Model Execution**: Different LLMs generate 100 programming questions following the template
3. **Filename Convention**: `synthetic_{model_name}.jsonl` where model name indicates the generator
4. **Quality Control**: Generated datasets follow strict schema and validation requirements

### Dataset Schema

Each entry in the JSONL files contains:

```json
{
  "query_id": "synthetic-001",
  "query_text": "Programming question with language and function name specified",
  "reference_answer": "Complete, runnable code solution",
  "assertions": "Runnable test code to verify correctness",
  "metadata": {
    "language": "Python|TypeScript|Go|Rust|etc",
    "function_name": "function_to_implement",
    "difficulty": "medium|advanced|very_advanced",
    "category": "algorithms|data_structures|concurrency|etc",
    "complexity": 0.75
  }
}
```

### Dataset Requirements

- **Format**: Valid JSONL (one JSON object per line)
- **Size**: 100 programming challenges per dataset
- **Language Distribution**:
  - 40% Python
  - 40% TypeScript
  - 7% Go
  - 3% Rust
  - 10% Other languages (Java, JavaScript, C++, Ruby, Swift, Kotlin, C#, PHP, Scala, Haskell)
- **Difficulty Distribution**: Evenly split across medium, advanced, and very advanced (~33 each)
- **Quality Standards**:
  - All code must be syntactically correct and runnable
  - Assertions must be executable and comprehensive
  - Problems must be unique and non-trivial
  - Solutions must handle edge cases appropriately

## Current Datasets

| File | Generator Model | Entries | Status |
|------|----------------|---------|--------|
| `synthetic_gpt5.jsonl` | GPT-4o (gpt-5) | 90 | In progress |

## Usage

These synthetic datasets complement our real-world benchmarks (GSM8K, MMLU, HumanEval) by providing:
- **Diverse programming challenges** across multiple languages
- **Controlled difficulty distribution** for testing algorithm learning curves
- **Code generation evaluation** with executable tests
- **Multi-language routing** to test language-specific model strengths

## Validation

All generated datasets should pass these checks:
- ✅ Valid JSONL format
- ✅ Exactly 100 entries (when complete)
- ✅ Language distribution matches requirements
- ✅ All required fields present in each entry
- ✅ Code is syntactically correct
- ✅ Assertions are runnable
- ✅ Function names match between query_text and metadata
- ✅ No duplicate problems

## Generation Cost

Estimated cost per 100-question dataset: ~$0.50-2.00 depending on model used
- GPT-4o: ~$0.50-1.00
- Claude Sonnet 3.5: ~$1.00-1.50
- GPT-4: ~$1.50-2.00

## Future Work

- Complete remaining datasets (claude-sonnet-3.5, gemini-pro, etc.)
- Add cross-validation comparing model-generated solutions
- Expand to 200-500 questions for more comprehensive testing
- Add domain-specific datasets (web dev, systems programming, ML/AI)
