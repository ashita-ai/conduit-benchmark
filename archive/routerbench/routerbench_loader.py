"""RouterBench dataset loader and adapter.

Loads RouterBench dataset (withmartian/routerbench) and converts to
BenchmarkQuery format for conduit-benchmark evaluation.

RouterBench provides 36,497 prompts across 86 benchmark sources with
binary correctness scores (0.0/1.0) for 11 LLMs, plus model responses
and costs. We extract reference answers from correct model responses.

Example:
    >>> loader = RouterBenchLoader()
    >>> queries = await loader.load(sample_size=1000, seed=42)
    >>> print(len(queries))
    1000
    >>> print(queries[0].reference_answer)
    "To solve this problem..."
"""

import pickle
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

from conduit_bench.benchmark_models import BenchmarkQuery


# RouterBench model columns (11 models tested)
MODELS = [
    "WizardLM/WizardLM-13B-V1.2",
    "claude-instant-v1",
    "claude-v1",
    "claude-v2",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
    "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat",
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
]


class RouterBenchLoader:
    """Load and convert RouterBench dataset to BenchmarkQuery format.

    RouterBench contains 36,497 prompts from 86 benchmark sources with
    binary correctness scores for 11 LLMs. We use correct model responses
    as reference answers for evaluation.

    Example:
        >>> loader = RouterBenchLoader(dataset_version="0shot")
        >>> queries = await loader.load(sample_size=5000, seed=42)
        >>> # Filters to factual-only benchmarks
        >>> factual_queries = await loader.load(
        ...     factual_only=True,
        ...     sample_size=5000
        ... )
    """

    def __init__(
        self,
        dataset_version: str = "0shot",
        cache_dir: str | None = None,
    ) -> None:
        """Initialize the loader.

        Args:
            dataset_version: RouterBench version ("0shot" or "5shot")
            cache_dir: Directory to cache downloaded dataset (default: HF cache)
        """
        self.dataset_version = dataset_version
        self.cache_dir = cache_dir
        self.dataset_path: Path | None = None

    async def load(
        self,
        sample_size: int | None = None,
        seed: int | None = None,
        exclude_no_correct: bool = True,
        factual_only: bool = False,
        min_correct_models: int = 1,
    ) -> list[BenchmarkQuery]:
        """Load RouterBench and convert to BenchmarkQuery format.

        Args:
            sample_size: Number of queries to sample (None = all)
            seed: Random seed for reproducible sampling
            exclude_no_correct: Skip samples where no model got it right
            factual_only: Filter to factual benchmarks only (exclude creative)
            min_correct_models: Minimum number of models that must be correct (default: 1)

        Returns:
            List of BenchmarkQuery objects with reference answers

        Reference Answer Strategy:
            - Use the correct model's response as reference_answer
            - If multiple models correct, use first correct model alphabetically
            - If no_model_correct and exclude_no_correct=False, reference_answer=None
        """
        # Download dataset if not cached
        if self.dataset_path is None:
            self.dataset_path = Path(
                hf_hub_download(
                    repo_id="withmartian/routerbench",
                    filename=f"routerbench_{self.dataset_version}.pkl",
                    repo_type="dataset",
                    cache_dir=self.cache_dir,
                )
            )

        # Load pandas DataFrame
        with open(self.dataset_path, "rb") as f:
            df = pickle.load(f)

        # Filter to factual benchmarks if requested
        if factual_only:
            factual_benchmarks = self._get_factual_benchmarks()
            df = df[df["eval_name"].isin(factual_benchmarks)]

        # Convert to BenchmarkQuery objects
        queries: list[BenchmarkQuery] = []

        for _, row in df.iterrows():
            # Extract query text from prompt list
            prompt = row["prompt"]
            if isinstance(prompt, list):
                query_text = "\n".join(str(p) for p in prompt)
            else:
                query_text = str(prompt)

            # Find correct model(s)
            correct_models = []
            for model in MODELS:
                if row.get(model) == 1.0:
                    correct_models.append(model)

            # Skip if no correct models and exclude_no_correct=True
            if len(correct_models) < min_correct_models:
                if exclude_no_correct:
                    continue
                else:
                    # Include with no reference answer
                    reference_answer = None
            else:
                # Use first correct model's response (alphabetically)
                correct_model = sorted(correct_models)[0]
                reference_answer = row.get(f"{correct_model}|model_response")
                if reference_answer is not None:
                    reference_answer = str(reference_answer)

            # Create BenchmarkQuery
            query = BenchmarkQuery(
                query_id=f"routerbench_{row['sample_id']}",
                query_text=query_text,
                reference_answer=reference_answer,
                metadata={
                    "source": "routerbench",
                    "eval_name": row["eval_name"],
                    "oracle_model": row.get("oracle_model_to_route_to"),
                    "num_correct_models": len(correct_models),
                    "correct_models": correct_models,
                },
            )
            queries.append(query)

        # Sample if requested
        if sample_size is not None and sample_size < len(queries):
            if seed is not None:
                random.seed(seed)
            queries = random.sample(queries, sample_size)

        # Shuffle for diversity (mix different eval_name sources)
        if seed is not None:
            random.seed(seed)
        random.shuffle(queries)

        return queries

    def _get_factual_benchmarks(self) -> set[str]:
        """Get list of factual (non-creative) benchmark sources.

        Returns:
            Set of benchmark names that are factual/objective

        Creative/Subjective Benchmarks to Exclude:
            - MT-Bench (subjective quality judgments)
            - Creative writing tasks
            - Open-ended generation tasks
        """
        # Based on RouterBench analysis - these are factual with clear correctness
        factual_benchmarks = {
            # Code generation (verifiable correctness)
            "mbpp",
            "humaneval",

            # Math (objective answers)
            "grade-school-math",
            "math",

            # Multiple choice QA (ground truth)
            "hellaswag",
            "arc-challenge",
            "arc-easy",
            "winogrande",
            "mmlu-professional-law",
            "mmlu-high-school-government-and-politics",
            "mmlu-jurisprudence",
            "mmlu-professional-accounting",
            "mmlu-high-school-macroeconomics",
            "mmlu-international-law",
            "mmlu-professional-medicine",
            "mmlu-high-school-us-history",
            "mmlu-high-school-world-history",
            "mmlu-us-foreign-policy",
            "mmlu-world-religions",
            "mmlu-philosophy",
            "mmlu-prehistory",
            "mmlu-formal-logic",
            "mmlu-moral-scenarios",
            "mmlu-moral-disputes",

            # Knowledge QA (factual)
            "truthfulqa",
            "natural-questions",
            "hotpotqa",
            "trivia-qa",

            # Reasoning (objective logic)
            "gsm8k",
            "strategy-qa",
            "commonsenseqa",

            # Science (factual)
            "sciq",
            "mmlu-high-school-chemistry",
            "mmlu-high-school-physics",
            "mmlu-high-school-biology",
            "mmlu-college-chemistry",
            "mmlu-college-physics",
            "mmlu-college-biology",

            # Other factual domains
            "mmlu-high-school-geography",
            "mmlu-high-school-computer-science",
            "mmlu-college-computer-science",
            "mmlu-machine-learning",
            "mmlu-astronomy",
            "mmlu-nutrition",
            "mmlu-clinical-knowledge",
            "mmlu-medical-genetics",
            "mmlu-anatomy",
            "mmlu-college-mathematics",
            "mmlu-high-school-mathematics",
            "mmlu-high-school-statistics",
            "mmlu-abstract-algebra",
            "mmlu-electrical-engineering",
            "mmlu-computer-security",
            "mmlu-econometrics",
            "mmlu-public-relations",
            "mmlu-marketing",
            "mmlu-management",
            "mmlu-business-ethics",
            "mmlu-global-facts",
            "mmlu-logical-fallacies",
            "mmlu-miscellaneous",
            "mmlu-security-studies",
            "mmlu-sociology",
            "mmlu-virology",
        }

        # Exclude these creative/subjective benchmarks
        creative_benchmarks = {
            "mt-bench",  # Subjective quality judgments
            "alpaca-eval",  # Open-ended generation
            # Add more as identified
        }

        return factual_benchmarks

    def get_dataset_stats(self) -> dict[str, int | dict[str, int]]:
        """Get statistics about the loaded dataset.

        Returns:
            Dictionary with dataset statistics including:
            - total_samples: Total number of samples
            - eval_name_counts: Number of samples per benchmark source
            - model_win_rates: Number of times each model was correct
        """
        if self.dataset_path is None:
            # Download if needed
            self.dataset_path = Path(
                hf_hub_download(
                    repo_id="withmartian/routerbench",
                    filename=f"routerbench_{self.dataset_version}.pkl",
                    repo_type="dataset",
                    cache_dir=self.cache_dir,
                )
            )

        # Load dataset
        with open(self.dataset_path, "rb") as f:
            df = pickle.load(f)

        # Calculate statistics
        stats = {
            "total_samples": len(df),
            "eval_name_counts": df["eval_name"].value_counts().to_dict(),
            "oracle_routing_counts": df["oracle_model_to_route_to"].value_counts().to_dict(),
            "no_model_correct": (df["oracle_model_to_route_to"] == "no_model_correct").sum(),
        }

        return stats
