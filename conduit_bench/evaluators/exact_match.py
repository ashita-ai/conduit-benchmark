"""Exact match evaluator for GSM8K and MMLU benchmarks.

Provides objective evaluation by extracting answers from model responses
and comparing to ground truth using dataset-specific patterns.

No LLM-as-judge - eliminates circular dependency in benchmark evaluation.
"""

import re
from typing import Any, Literal

from conduit_bench.evaluators.base import BaseEvaluator, EvaluationResult


class ExactMatchEvaluator(BaseEvaluator):
    """Exact match evaluator for GSM8K and MMLU datasets.

    Extracts answers from model responses using dataset-specific patterns:
    - GSM8K: Extract numeric answer after "#### " marker
    - MMLU: Extract single letter A/B/C/D choice

    Example:
        >>> evaluator = ExactMatchEvaluator(dataset_type="gsm8k")
        >>> result = evaluator.evaluate(
        ...     response="Let me solve this. 48/2 = 24. 48 + 24 = 72. #### 72",
        ...     ground_truth="72"
        ... )
        >>> print(result.correct)  # True

        >>> evaluator = ExactMatchEvaluator(dataset_type="mmlu")
        >>> result = evaluator.evaluate(
        ...     response="The answer is C because Paris is the capital.",
        ...     ground_truth="C"
        ... )
        >>> print(result.correct)  # True
    """

    def __init__(self, dataset_type: Literal["gsm8k", "mmlu"] = "gsm8k") -> None:
        """Initialize the exact match evaluator.

        Args:
            dataset_type: Type of dataset ("gsm8k" or "mmlu")
        """
        self.dataset_type = dataset_type

    @property
    def name(self) -> str:
        """Human-readable name for this evaluator."""
        return f"ExactMatch ({self.dataset_type.upper()})"

    @property
    def description(self) -> str:
        """Description of the evaluation method."""
        if self.dataset_type == "gsm8k":
            return "Extracts numeric answer after '#### ' marker and compares to ground truth"
        else:
            return "Extracts A/B/C/D choice and compares to ground truth"

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
            ground_truth: The correct answer
            query: Original query text (unused, for interface compatibility)
            **kwargs: Additional parameters (unused)

        Returns:
            EvaluationResult with score (1.0 if correct, 0.0 if incorrect)
        """
        predicted = self.extract_answer(response)
        expected = self._normalize_answer(ground_truth)

        # Compare normalized answers
        correct = predicted is not None and predicted == expected

        return EvaluationResult(
            score=1.0 if correct else 0.0,
            correct=correct,
            predicted=predicted,
            expected=expected,
            metadata={
                "dataset_type": self.dataset_type,
                "raw_response_length": len(response) if response else 0,
            },
        )

    def extract_answer(self, response: str, **kwargs: Any) -> str | None:
        """Extract the answer from a model response.

        Args:
            response: The model's response text
            **kwargs: Additional parameters (unused)

        Returns:
            Extracted answer string, or None if extraction failed
        """
        if not response:
            return None

        if self.dataset_type == "gsm8k":
            return self._extract_gsm8k_answer(response)
        else:
            return self._extract_mmlu_answer(response)

    def _extract_gsm8k_answer(self, text: str) -> str | None:
        """Extract numeric answer from GSM8K format.

        GSM8K answers follow the format: "#### N" where N is the numeric answer.
        Handles integers, decimals, and comma-separated numbers.

        Args:
            text: Model response text

        Returns:
            Normalized numeric string, or None if not found
        """
        # Primary pattern: #### followed by number
        match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
        if match:
            return match.group(1).replace(",", "")

        # Fallback: Look for "answer is N" or "= N" at end
        fallback_patterns = [
            r"(?:answer|result)\s+is\s+(-?\d+(?:,\d+)*(?:\.\d+)?)\s*\.?\s*$",
            r"=\s*(-?\d+(?:,\d+)*(?:\.\d+)?)\s*\.?\s*$",
            r"(-?\d+(?:,\d+)*(?:\.\d+)?)\s*\.?\s*$",
        ]

        for pattern in fallback_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).replace(",", "")

        return None

    def _extract_mmlu_answer(self, text: str) -> str | None:
        """Extract multiple choice answer from MMLU format.

        MMLU answers are single letters: A, B, C, or D.

        Args:
            text: Model response text

        Returns:
            Uppercase letter (A/B/C/D), or None if not found
        """
        # Look for explicit answer patterns first
        explicit_patterns = [
            r"(?:answer|choice)\s+is\s+[:\s]*([ABCD])\b",
            r"(?:correct|right)\s+answer\s+is\s+[:\s]*([ABCD])\b",
            r"\b([ABCD])\s*[.):]\s*(?:is\s+)?(?:correct|right)",
            r"^\s*([ABCD])\s*$",  # Just the letter alone
        ]

        for pattern in explicit_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        # Fallback: First standalone A/B/C/D in the response
        match = re.search(r"\b([ABCD])\b", text.upper())
        if match:
            return match.group(1)

        return None

    def _normalize_answer(self, answer: str) -> str:
        """Normalize ground truth answer for comparison.

        Args:
            answer: Raw ground truth answer

        Returns:
            Normalized answer string
        """
        if not answer:
            return ""

        answer = answer.strip()

        if self.dataset_type == "gsm8k":
            # Remove commas and normalize numeric format
            answer = answer.replace(",", "")
            # Handle GSM8K format where answer might include "#### "
            if "####" in answer:
                match = re.search(r"####\s*(-?\d+(?:\.\d+)?)", answer)
                if match:
                    answer = match.group(1)
        else:
            # MMLU: Uppercase letter only
            answer = answer.upper()
            if len(answer) == 1 and answer in "ABCD":
                return answer
            # Extract letter if wrapped in other text
            match = re.search(r"([ABCD])", answer)
            if match:
                return match.group(1)

        return answer
