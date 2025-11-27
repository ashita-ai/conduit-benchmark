"""Dataset loaders for established benchmarks.

Loads datasets from HuggingFace and converts them to BenchmarkQuery format:
- GSM8K: Grade school math (1,319 test problems)
- MMLU: Massive multitask language understanding (14,042 questions)
- HumanEval: Python code generation (164 problems)
"""

from conduit_bench.datasets.gsm8k import GSM8KLoader
from conduit_bench.datasets.mmlu import MMLULoader
from conduit_bench.datasets.humaneval import HumanEvalLoader

__all__ = [
    "GSM8KLoader",
    "MMLULoader",
    "HumanEvalLoader",
]
