"""MMLU dataset loader.

Loads the Massive Multitask Language Understanding dataset from HuggingFace
and converts to BenchmarkQuery format for bandit algorithm evaluation.

Source: https://huggingface.co/datasets/cais/mmlu
Size: 14,042 questions across 57 subjects
Evaluation: Exact match on A/B/C/D answer
"""

from typing import Literal

from datasets import load_dataset

from conduit_bench.benchmark_models import BenchmarkQuery


class MMLULoader:
    """Loader for MMLU multi-subject knowledge dataset.

    Loads questions from HuggingFace and formats them as BenchmarkQuery objects.
    Ground truth is the correct answer letter (A/B/C/D).

    Example:
        >>> loader = MMLULoader()
        >>> queries = loader.load(split="test", limit=100)
        >>> print(len(queries))  # 100
        >>> print(queries[0].query_text)  # "What is the capital of France?..."
        >>> print(queries[0].reference_answer)  # "C"
    """

    DATASET_NAME = "cais/mmlu"
    DATASET_CONFIG = "all"

    # MMLU subjects (57 total)
    SUBJECTS = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology",
        "high_school_statistics", "high_school_us_history",
        "high_school_world_history", "human_aging", "human_sexuality",
        "international_law", "jurisprudence", "logical_fallacies",
        "machine_learning", "management", "marketing", "medical_genetics",
        "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
        "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
        "virology", "world_religions",
    ]

    ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}

    def __init__(self) -> None:
        """Initialize the MMLU loader."""
        self._dataset = None

    def load(
        self,
        split: Literal["validation", "test"] = "test",
        limit: int | None = None,
        seed: int = 42,
        subjects: list[str] | None = None,
    ) -> list[BenchmarkQuery]:
        """Load MMLU dataset and convert to BenchmarkQuery format.

        Args:
            split: Dataset split ("validation" or "test", default: "test")
            limit: Maximum number of queries to load (default: all)
            seed: Random seed for shuffling (if limit < full dataset)
            subjects: List of subjects to include (default: all 57)

        Returns:
            List of BenchmarkQuery objects
        """
        # Load from HuggingFace
        dataset = load_dataset(
            self.DATASET_NAME,
            self.DATASET_CONFIG,
            split=split,
        )

        # Filter by subjects if specified
        if subjects:
            valid_subjects = [s for s in subjects if s in self.SUBJECTS]
            if valid_subjects:
                dataset = dataset.filter(
                    lambda x: x["subject"] in valid_subjects
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
        """Convert a single MMLU item to BenchmarkQuery.

        Args:
            item: Raw dataset item with question, choices, and answer
            idx: Item index for ID generation
            split: Dataset split name

        Returns:
            BenchmarkQuery object
        """
        question = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]
        subject = item.get("subject", "unknown")

        # Convert numeric answer to letter
        ground_truth = self.ANSWER_MAP.get(answer_idx, "A")

        # Format as multiple choice question
        query_text = self._format_question(question, choices)

        return BenchmarkQuery(
            query_id=f"mmlu_{split}_{idx}",
            query_text=query_text,
            reference_answer=ground_truth,
            metadata={
                "dataset": "mmlu",
                "split": split,
                "subject": subject,
                "original_question": question,
                "choices": choices,
                "answer_index": answer_idx,
            },
        )

    def _format_question(self, question: str, choices: list[str]) -> str:
        """Format question with answer choices.

        Args:
            question: The question text
            choices: List of 4 answer choices

        Returns:
            Formatted question string
        """
        formatted = f"{question}\n\n"

        for i, choice in enumerate(choices):
            letter = self.ANSWER_MAP[i]
            formatted += f"({letter}) {choice}\n"

        formatted += (
            "\nPlease select the correct answer (A, B, C, or D). "
            "Provide your answer as a single letter."
        )

        return formatted

    def get_subjects(self) -> list[str]:
        """Get list of all available subjects.

        Returns:
            List of 57 MMLU subject names
        """
        return self.SUBJECTS.copy()

    @property
    def description(self) -> str:
        """Human-readable description of the dataset."""
        return (
            "MMLU (Massive Multitask Language Understanding): 14,042 questions "
            "across 57 subjects covering STEM, humanities, and social sciences. "
            "Evaluation: Exact match on answer letter (A/B/C/D)."
        )
