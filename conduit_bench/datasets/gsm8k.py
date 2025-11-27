"""GSM8K dataset loader.

Loads the Grade School Math 8K dataset from HuggingFace and converts
to BenchmarkQuery format for bandit algorithm evaluation.

Source: https://huggingface.co/datasets/openai/gsm8k
Size: 8,792 train + 1,319 test problems
Evaluation: Exact match on "#### N" answer format
"""

import re
from typing import Literal

from datasets import load_dataset

from conduit_bench.benchmark_models import BenchmarkQuery


class GSM8KLoader:
    """Loader for GSM8K grade school math dataset.

    Loads problems from HuggingFace and formats them as BenchmarkQuery objects.
    Ground truth is extracted from the "#### N" answer format.

    Example:
        >>> loader = GSM8KLoader()
        >>> queries = loader.load(split="test", limit=100)
        >>> print(len(queries))  # 100
        >>> print(queries[0].query_text)  # "Natalia sold clips..."
        >>> print(queries[0].reference_answer)  # "72"
    """

    DATASET_NAME = "openai/gsm8k"
    DATASET_CONFIG = "main"

    def __init__(self) -> None:
        """Initialize the GSM8K loader."""
        self._dataset = None

    def load(
        self,
        split: Literal["train", "test"] = "test",
        limit: int | None = None,
        seed: int = 42,
    ) -> list[BenchmarkQuery]:
        """Load GSM8K dataset and convert to BenchmarkQuery format.

        Args:
            split: Dataset split ("train" or "test", default: "test")
            limit: Maximum number of queries to load (default: all)
            seed: Random seed for shuffling (if limit < full dataset)

        Returns:
            List of BenchmarkQuery objects
        """
        # Load from HuggingFace
        dataset = load_dataset(
            self.DATASET_NAME,
            self.DATASET_CONFIG,
            split=split,
        )

        # Shuffle and limit if requested
        if limit and limit < len(dataset):
            dataset = dataset.shuffle(seed=seed).select(range(limit))

        # Convert to BenchmarkQuery format
        queries = []
        for idx, item in enumerate(dataset):
            query = self._convert_item(item, idx, split)
            queries.append(query)

        return queries

    def _convert_item(
        self,
        item: dict,
        idx: int,
        split: str,
    ) -> BenchmarkQuery:
        """Convert a single GSM8K item to BenchmarkQuery.

        Args:
            item: Raw dataset item with 'question' and 'answer' fields
            idx: Item index for ID generation
            split: Dataset split name

        Returns:
            BenchmarkQuery object
        """
        question = item["question"]
        full_answer = item["answer"]

        # Extract numeric answer from "#### N" format
        ground_truth = self._extract_answer(full_answer)

        # Format the query to encourage step-by-step reasoning with #### answer
        query_text = (
            f"{question}\n\n"
            "Please solve this step by step. Show your work and provide "
            "the final answer in the format: #### [answer]"
        )

        return BenchmarkQuery(
            query_id=f"gsm8k_{split}_{idx}",
            query_text=query_text,
            reference_answer=ground_truth,
            metadata={
                "dataset": "gsm8k",
                "split": split,
                "full_solution": full_answer,
                "original_question": question,
            },
        )

    def _extract_answer(self, answer_text: str) -> str:
        """Extract numeric answer from GSM8K answer format.

        GSM8K answers include step-by-step solution followed by "#### N"
        where N is the final numeric answer.

        Args:
            answer_text: Full answer text including solution steps

        Returns:
            Numeric answer string
        """
        match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", answer_text)
        if match:
            return match.group(1).replace(",", "")

        # Fallback: last number in the text
        numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", answer_text)
        if numbers:
            return numbers[-1].replace(",", "")

        return ""

    @property
    def description(self) -> str:
        """Human-readable description of the dataset."""
        return (
            "GSM8K (Grade School Math 8K): 8,792 train + 1,319 test grade school "
            "math word problems requiring multi-step reasoning. "
            "Evaluation: Exact match on numeric answer after '#### ' marker."
        )
