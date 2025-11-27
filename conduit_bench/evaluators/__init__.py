"""Pluggable evaluators for benchmark quality assessment.

This module provides objective evaluation methods that don't rely on LLM-as-judge:
- ExactMatchEvaluator: For GSM8K and MMLU (extract answer, compare to ground truth)
- CodeExecutionEvaluator: For HumanEval (run code, check tests pass)
- ArbiterEvaluator: Legacy LLM-as-judge (kept for production routing, not benchmarks)
"""

from conduit_bench.evaluators.base import BaseEvaluator, EvaluationResult
from conduit_bench.evaluators.exact_match import ExactMatchEvaluator
from conduit_bench.evaluators.code_execution import CodeExecutionEvaluator

__all__ = [
    "BaseEvaluator",
    "EvaluationResult",
    "ExactMatchEvaluator",
    "CodeExecutionEvaluator",
]
