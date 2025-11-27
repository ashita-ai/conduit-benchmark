"""HumanEval dataset loader.

Loads the OpenAI HumanEval dataset from HuggingFace and converts
to BenchmarkQuery format for bandit algorithm evaluation.

Source: https://huggingface.co/datasets/openai/openai_humaneval
Size: 164 Python function completion problems
Evaluation: Code execution with unit tests (pass/fail)
"""

from datasets import load_dataset

from conduit_bench.benchmark_models import BenchmarkQuery


class HumanEvalLoader:
    """Loader for HumanEval code generation dataset.

    Loads Python function completion problems from HuggingFace and formats
    them as BenchmarkQuery objects. Ground truth is the test code.

    Example:
        >>> loader = HumanEvalLoader()
        >>> queries = loader.load(limit=10)
        >>> print(len(queries))  # 10
        >>> print(queries[0].metadata["entry_point"])  # "has_close_elements"
        >>> print(queries[0].reference_answer)  # Test assertions
    """

    DATASET_NAME = "openai/openai_humaneval"

    def __init__(self) -> None:
        """Initialize the HumanEval loader."""
        self._dataset = None

    def load(
        self,
        limit: int | None = None,
        seed: int = 42,
    ) -> list[BenchmarkQuery]:
        """Load HumanEval dataset and convert to BenchmarkQuery format.

        Args:
            limit: Maximum number of problems to load (default: all 164)
            seed: Random seed for shuffling (if limit < full dataset)

        Returns:
            List of BenchmarkQuery objects
        """
        # Load from HuggingFace (HumanEval only has 'test' split)
        dataset = load_dataset(
            self.DATASET_NAME,
            split="test",
        )

        # Shuffle and limit if requested
        if limit and limit < len(dataset):
            dataset = dataset.shuffle(seed=seed).select(range(limit))

        # Convert to BenchmarkQuery format
        queries = []
        for idx, item in enumerate(dataset):
            query = self._convert_item(item, idx)
            queries.append(query)

        return queries

    def _convert_item(
        self,
        item: dict,
        idx: int,
    ) -> BenchmarkQuery:
        """Convert a single HumanEval item to BenchmarkQuery.

        Args:
            item: Raw dataset item with prompt, test, entry_point, etc.
            idx: Item index for ID generation

        Returns:
            BenchmarkQuery object
        """
        task_id = item["task_id"]  # e.g., "HumanEval/0"
        prompt = item["prompt"]  # Function signature + docstring
        test_code = item["test"]  # Unit test assertions
        entry_point = item["entry_point"]  # Function name
        canonical_solution = item.get("canonical_solution", "")

        # Format as code completion request
        query_text = self._format_prompt(prompt, entry_point)

        return BenchmarkQuery(
            query_id=f"humaneval_{idx}",
            query_text=query_text,
            reference_answer=test_code,  # Test code for execution
            metadata={
                "dataset": "humaneval",
                "task_id": task_id,
                "prompt": prompt,  # Original function signature
                "entry_point": entry_point,
                "canonical_solution": canonical_solution,
                "test_code": test_code,
            },
        )

    def _format_prompt(self, prompt: str, entry_point: str) -> str:
        """Format the code completion prompt.

        Args:
            prompt: Original function signature and docstring
            entry_point: Function name

        Returns:
            Formatted prompt for code completion
        """
        formatted = (
            "Complete the following Python function. Provide only the function "
            "body (the implementation), not the signature or docstring.\n\n"
            "```python\n"
            f"{prompt}"
            "```\n\n"
            f"Implement the `{entry_point}` function according to the docstring. "
            "Do not include the function signature - just provide the body."
        )

        return formatted

    @property
    def description(self) -> str:
        """Human-readable description of the dataset."""
        return (
            "HumanEval: 164 Python function completion problems from OpenAI. "
            "Each problem includes a function signature, docstring, and unit tests. "
            "Evaluation: Execute generated code with unit tests (pass/fail)."
        )
