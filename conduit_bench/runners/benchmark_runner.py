"""BenchmarkRunner for orchestrating bandit algorithm evaluation.

Executes queries across multiple bandit algorithms, collects feedback,
and generates comprehensive benchmark results.
"""

from datetime import datetime, timezone

from arbiter import evaluate
from rich.console import Console
from tqdm.asyncio import tqdm_asyncio

from conduit.core.models import QueryFeatures
from conduit.engines.bandits import BanditAlgorithm, BanditFeedback

from conduit_bench.benchmark_models import (
    AlgorithmRun,
    BenchmarkQuery,
    BenchmarkResult,
    QueryEvaluation,
)
from conduit_bench.runners.model_executor import ModelExecutor


console = Console()


class BenchmarkRunner:
    """Orchestrates benchmark execution across multiple bandit algorithms.

    The BenchmarkRunner executes a dataset of queries using multiple bandit
    algorithms in parallel, collecting feedback and generating comprehensive
    comparison results.

    Example:
        >>> from conduit.engines.bandits import ThompsonSamplingBandit, UCB1Bandit
        >>> from conduit.models import DEFAULT_ARMS
        >>>
        >>> algorithms = [
        ...     ThompsonSamplingBandit(DEFAULT_ARMS),
        ...     UCB1Bandit(DEFAULT_ARMS, c=1.5),
        ... ]
        >>> runner = BenchmarkRunner(algorithms=algorithms)
        >>> result = await runner.run(dataset=queries)
        >>> print(f"Thompson: ${result.get_algorithm('Thompson Sampling').total_cost:.2f}")
        Thompson: $12.45
    """

    def __init__(
        self,
        algorithms: list[BanditAlgorithm],
        executor: ModelExecutor | None = None,
        arbiter_model: str = "gpt-4o-mini",
        max_concurrency: int = 5,
    ) -> None:
        """Initialize BenchmarkRunner.

        Args:
            algorithms: List of bandit algorithms to benchmark
            executor: ModelExecutor for LLM execution (creates default if None)
            arbiter_model: Model to use for quality evaluation
            max_concurrency: Maximum number of parallel query executions
        """
        self.algorithms = algorithms
        self.executor = executor or ModelExecutor()
        self.arbiter_model = arbiter_model
        self.max_concurrency = max_concurrency

    async def run(
        self,
        dataset: list[BenchmarkQuery],
        show_progress: bool = True,
    ) -> BenchmarkResult:
        """Run benchmark across all algorithms.

        Args:
            dataset: List of queries to benchmark
            show_progress: Whether to show progress bars

        Returns:
            Complete benchmark results with all algorithm runs
        """
        console.print(f"\n[bold cyan]Starting benchmark with {len(self.algorithms)} algorithms[/bold cyan]")
        console.print(f"Dataset size: {len(dataset)} queries\n")

        # Reset all algorithms to initial state
        for algo in self.algorithms:
            algo.reset()

        # Run each algorithm independently
        algorithm_runs: list[AlgorithmRun] = []

        for algo in self.algorithms:
            console.print(f"[yellow]Running {algo.name}...[/yellow]")
            run = await self._run_algorithm(algo, dataset, show_progress)
            algorithm_runs.append(run)
            console.print(f"[green]✓ {algo.name} complete: ${run.total_cost:.4f}, quality={run.average_quality:.3f}[/green]\n")

        # Create benchmark result
        result = BenchmarkResult(
            dataset_size=len(dataset),
            algorithms=algorithm_runs,
        )

        console.print("[bold green]Benchmark complete![/bold green]\n")
        return result

    async def _run_algorithm(
        self,
        algorithm: BanditAlgorithm,
        dataset: list[BenchmarkQuery],
        show_progress: bool,
    ) -> AlgorithmRun:
        """Run a single algorithm on the dataset.

        Args:
            algorithm: Bandit algorithm to run
            dataset: List of queries
            show_progress: Whether to show progress

        Returns:
            Algorithm run results
        """
        started_at = datetime.now(timezone.utc)
        selections: list[tuple[str, str]] = []
        feedback_list: list[QueryEvaluation] = []
        cumulative_regret: list[float] = []
        total_cost = 0.0
        total_quality = 0.0

        # Process queries sequentially (bandit algorithms need feedback before next selection)
        iterator = tqdm_asyncio(dataset, desc=algorithm.name, disable=not show_progress)

        for query in iterator:
            # Create simplified query features (for benchmarking, we use basic features)
            features = self._create_query_features(query)

            # Algorithm selects model
            selected_arm = await algorithm.select_arm(features)
            selections.append((query.query_id, selected_arm.model_id))

            # Execute query with selected model
            execution_result = await self.executor.execute(
                arm=selected_arm,
                query_text=query.query_text,
                system_prompt="You are a helpful assistant.",
            )

            # Evaluate quality using Arbiter
            quality_score = 0.0
            if execution_result.success and query.reference_answer:
                quality_score = await self._evaluate_quality(
                    output=execution_result.response_text,
                    reference=query.reference_answer,
                )
            elif not execution_result.success:
                quality_score = 0.0  # Failed execution = zero quality
            else:
                quality_score = 0.5  # No reference answer = neutral quality

            # Create evaluation record
            evaluation = QueryEvaluation(
                query_id=query.query_id,
                model_id=selected_arm.model_id,
                response_text=execution_result.response_text,
                quality_score=quality_score,
                cost=execution_result.cost,
                latency=execution_result.latency,
                success=execution_result.success,
            )
            feedback_list.append(evaluation)

            # Update algorithm with feedback
            bandit_feedback = BanditFeedback(
                model_id=selected_arm.model_id,
                cost=execution_result.cost,
                quality_score=quality_score,
                latency=execution_result.latency,
                success=execution_result.success,
            )
            await algorithm.update(bandit_feedback, features)

            # Track cumulative metrics
            total_cost += execution_result.cost
            total_quality += quality_score
            cumulative_regret.append(total_cost)  # Simplified regret (actual cumulative cost)

        completed_at = datetime.now(timezone.utc)

        return AlgorithmRun(
            algorithm_name=algorithm.name,
            total_queries=len(dataset),
            total_cost=total_cost,
            average_quality=total_quality / len(dataset) if dataset else 0.0,
            selections=selections,
            feedback=feedback_list,
            cumulative_regret=cumulative_regret,
            started_at=started_at,
            completed_at=completed_at,
            metadata=algorithm.get_stats(),
        )

    def _create_query_features(self, query: BenchmarkQuery) -> QueryFeatures:
        """Create QueryFeatures from BenchmarkQuery.

        For benchmarking, we create simplified features since we don't need
        the full query analyzer pipeline.

        Args:
            query: Benchmark query

        Returns:
            QueryFeatures for bandit algorithm
        """
        # Create zero embedding (bandit algorithms that don't use features will ignore this)
        embedding = [0.0] * 384

        return QueryFeatures(
            embedding=embedding,
            token_count=len(query.query_text.split()),  # Rough approximation
            complexity_score=query.complexity,
            domain=query.category,
            domain_confidence=1.0,  # High confidence since we explicitly set the category
        )

    async def _evaluate_quality(self, output: str, reference: str) -> float:
        """Evaluate response quality using Arbiter.

        Args:
            output: Model's response
            reference: Reference answer

        Returns:
            Quality score (0-1 scale)
        """
        try:
            result = await evaluate(
                output=output,
                reference=reference,
                evaluators=["semantic"],
                model=self.arbiter_model,
            )
            return result.overall_score
        except Exception as e:
            console.print(f"[red]Warning: Arbiter evaluation failed: {e}[/red]")
            return 0.5  # Neutral score on evaluation failure
