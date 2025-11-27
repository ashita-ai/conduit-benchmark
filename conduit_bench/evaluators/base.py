"""Base evaluator interface for pluggable benchmark evaluation.

Defines the abstract interface that all evaluators must implement,
enabling objective evaluation without LLM-as-judge circular dependency.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationResult:
    """Result of evaluating a model response.

    Attributes:
        score: Quality score (0.0 = incorrect, 1.0 = correct)
        correct: Boolean indicating if answer was correct
        predicted: The extracted/predicted answer
        expected: The expected ground truth answer
        metadata: Additional evaluation-specific metadata
    """

    score: float
    correct: bool
    predicted: str | None
    expected: str | None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "score": self.score,
            "correct": self.correct,
            "predicted": self.predicted,
            "expected": self.expected,
            "metadata": self.metadata or {},
        }


class BaseEvaluator(ABC):
    """Abstract base class for benchmark evaluators.

    Evaluators provide objective assessment of model responses without
    using LLM-as-judge approaches that create circular dependencies.

    Subclasses must implement:
    - evaluate(): Score a single response against ground truth
    - extract_answer(): Extract the answer from model response (optional override)

    Example:
        >>> evaluator = ExactMatchEvaluator(dataset_type="gsm8k")
        >>> result = evaluator.evaluate(
        ...     response="The answer is #### 42",
        ...     ground_truth="42"
        ... )
        >>> print(result.correct)  # True
        >>> print(result.score)    # 1.0
    """

    @abstractmethod
    def evaluate(
        self,
        response: str,
        ground_truth: str,
        query: str | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate a model response against ground truth.

        Args:
            response: The model's response text
            ground_truth: The correct answer (format depends on dataset)
            query: Original query text (optional, for context)
            **kwargs: Additional dataset-specific parameters

        Returns:
            EvaluationResult with score, correctness, and extracted answers
        """
        pass

    def extract_answer(self, response: str, **kwargs: Any) -> str | None:
        """Extract the answer from a model response.

        Override this method for dataset-specific answer extraction logic.
        Default implementation returns the response as-is.

        Args:
            response: The model's response text
            **kwargs: Additional parameters for extraction

        Returns:
            Extracted answer string, or None if extraction failed
        """
        return response.strip() if response else None

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this evaluator."""
        pass

    @property
    def description(self) -> str:
        """Description of the evaluation method."""
        return f"{self.name} evaluator"
