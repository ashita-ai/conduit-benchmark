"""BenchmarkRunner for orchestrating bandit algorithm evaluation.

Executes queries across multiple bandit algorithms, collects feedback,
and generates comprehensive benchmark results.
"""

from datetime import datetime, timezone

from arbiter_ai import evaluate
from rich.console import Console
from tqdm.asyncio import tqdm_asyncio

from conduit.core.models import QueryFeatures
from conduit.engines.analyzer import QueryAnalyzer
from conduit.engines.bandits import BanditAlgorithm, BanditFeedback

from conduit_bench.benchmark_models import (
    AlgorithmRun,
    BenchmarkQuery,
    BenchmarkResult,
    QueryEvaluation,
)
from conduit_bench.database import BenchmarkDatabase
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
        database: BenchmarkDatabase | None = None,
        enable_db_write: bool = True,
        analyzer: QueryAnalyzer | None = None,
    ) -> None:
        """Initialize BenchmarkRunner.

        Args:
            algorithms: List of bandit algorithms to benchmark
            executor: ModelExecutor for LLM execution (creates default if None)
            arbiter_model: Model to use for quality evaluation
            max_concurrency: Maximum number of parallel query executions
            database: BenchmarkDatabase instance (creates default if None)
            enable_db_write: Whether to enable streaming database writes
            analyzer: QueryAnalyzer for feature extraction (creates default if None)
        """
        self.algorithms = algorithms
        self.executor = executor or ModelExecutor()
        self.arbiter_model = arbiter_model
        self.max_concurrency = max_concurrency
        self.database = database
        self.enable_db_write = enable_db_write
        self.analyzer = analyzer or QueryAnalyzer()

    async def run(
        self,
        dataset: list[BenchmarkQuery],
        show_progress: bool = True,
        parallel: bool = False,
        benchmark_id: str | None = None,
    ) -> BenchmarkResult:
        """Run benchmark across all algorithms.

        Args:
            dataset: List of queries to benchmark
            show_progress: Whether to show progress bars
            parallel: Run algorithms in parallel (3x faster for 3 algorithms)
            benchmark_id: Optional benchmark ID (auto-generated if None)

        Returns:
            Complete benchmark results with all algorithm runs
        """
        console.print(f"\n[bold cyan]Starting benchmark with {len(self.algorithms)} algorithms[/bold cyan]")
        console.print(f"Dataset size: {len(dataset)} queries")
        if parallel:
            console.print("[bold yellow]Running algorithms in parallel[/bold yellow]\n")
        else:
            console.print()

        # Initialize database connection if enabled
        if self.enable_db_write and not self.database:
            self.database = BenchmarkDatabase()

        db_connected = False
        if self.enable_db_write and self.database:
            try:
                await self.database.connect()
                db_connected = True
                console.print("[green]Database connected for streaming writes[/green]")
            except Exception as e:
                console.print(f"[yellow]Warning: Database connection failed: {e}[/yellow]")
                console.print("[yellow]Continuing without database writes[/yellow]")
                self.enable_db_write = False

        try:
            # Create benchmark result with ID
            from uuid import uuid4
            _benchmark_id = benchmark_id or str(uuid4())

            # Create benchmark run record in database
            if db_connected and self.database:
                try:
                    await self.database.create_benchmark_run(
                        benchmark_id=_benchmark_id,
                        dataset_size=len(dataset),
                        metadata={
                            "algorithm_names": [algo.name for algo in self.algorithms],
                            "parallel": parallel,
                        },
                    )
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to create benchmark run: {e}[/yellow]")

            # Reset all algorithms to initial state
            for algo in self.algorithms:
                algo.reset()

            # Run algorithms in parallel or sequentially
            if parallel:
                # Run all algorithms concurrently
                import asyncio

                console.print("[yellow]Starting all algorithms in parallel...[/yellow]\n")
                tasks = [self._run_algorithm(algo, dataset, show_progress, _benchmark_id) for algo in self.algorithms]
                algorithm_runs = await asyncio.gather(*tasks)

                # Print completion summary
                for run in algorithm_runs:
                    console.print(f"[green]✓ {run.algorithm_name} complete: ${run.total_cost:.4f}, quality={run.average_quality:.3f}[/green]")
            else:
                # Run each algorithm sequentially (original behavior)
                algorithm_runs: list[AlgorithmRun] = []

                for algo in self.algorithms:
                    console.print(f"[yellow]Running {algo.name}...[/yellow]")
                    run = await self._run_algorithm(algo, dataset, show_progress, _benchmark_id)
                    algorithm_runs.append(run)
                    console.print(f"[green]✓ {algo.name} complete: ${run.total_cost:.4f}, quality={run.average_quality:.3f}[/green]\n")

            # Create benchmark result
            result = BenchmarkResult(
                benchmark_id=_benchmark_id,
                dataset_size=len(dataset),
                algorithms=algorithm_runs,
            )

            console.print("\n[bold green]Benchmark complete![/bold green]\n")
            return result
        finally:
            # ALWAYS close database connection, even on error
            if db_connected and self.database:
                try:
                    await self.database.disconnect()
                except Exception as e:
                    console.print(f"[yellow]Warning: Database disconnect failed: {e}[/yellow]")

    async def _run_algorithm(
        self,
        algorithm: BanditAlgorithm,
        dataset: list[BenchmarkQuery],
        show_progress: bool,
        benchmark_id: str,
    ) -> AlgorithmRun:
        """Run a single algorithm on the dataset.

        Args:
            algorithm: Bandit algorithm to run
            dataset: List of queries
            show_progress: Whether to show progress
            benchmark_id: Parent benchmark ID for database linkage

        Returns:
            Algorithm run results
        """
        from uuid import uuid4

        started_at = datetime.now(timezone.utc)
        run_id = str(uuid4())
        selections: list[tuple[str, str]] = []
        feedback_list: list[QueryEvaluation] = []
        cumulative_regret: list[float] = []
        total_cost = 0.0
        total_quality = 0.0

        # Create algorithm run record in database
        if self.enable_db_write and self.database:
            try:
                await self.database.create_algorithm_run(
                    run_id=run_id,
                    benchmark_id=benchmark_id,
                    algorithm_name=algorithm.name,
                    started_at=started_at,
                    metadata=algorithm.get_stats(),
                )
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to create algorithm run: {e}[/yellow]")

        # Process queries sequentially (bandit algorithms need feedback before next selection)
        iterator = tqdm_asyncio(dataset, desc=algorithm.name, disable=not show_progress)

        for query in iterator:
            # Use Conduit's QueryAnalyzer to extract real features (NO CHEATING!)
            features = await self.analyzer.analyze(query.query_text)

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
                error=execution_result.error,
            )
            feedback_list.append(evaluation)

            # Write query evaluation to database (streaming)
            if self.enable_db_write and self.database:
                try:
                    await self.database.write_query_evaluation(
                        run_id=run_id,
                        query_id=query.query_id,
                        model_id=selected_arm.model_id,
                        quality_score=quality_score,
                        cost=execution_result.cost,
                        latency=execution_result.latency,
                        success=execution_result.success,
                        metadata={"response_length": len(execution_result.response_text)},
                    )
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to write query evaluation: {e}[/yellow]")

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

        # Calculate final metrics
        average_quality = total_quality / len(dataset) if dataset else 0.0

        # Update algorithm run with final metrics in database
        if self.enable_db_write and self.database:
            try:
                await self.database.update_algorithm_run(
                    run_id=run_id,
                    total_cost=total_cost,
                    average_quality=average_quality,
                    total_queries=len(dataset),
                    completed_at=completed_at,
                )
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to update algorithm run: {e}[/yellow]")

        return AlgorithmRun(
            algorithm_name=algorithm.name,
            run_id=run_id,
            total_queries=len(dataset),
            total_cost=total_cost,
            average_quality=average_quality,
            selections=selections,
            feedback=feedback_list,
            cumulative_regret=cumulative_regret,
            started_at=started_at,
            completed_at=completed_at,
            metadata=algorithm.get_stats(),
        )


    async def _evaluate_quality(self, output: str, reference: str) -> float:
        """Evaluate response quality using Arbiter with mixed evaluators.

        Uses a probabilistic mix of evaluators:
        - 30% semantic (embedding similarity)
        - 30% groundedness (factual grounding)
        - 10% semantic + groundedness (comprehensive grounding)
        - 10% semantic + factuality (comprehensive fact-checking)
        - 20% factuality (pure fact checking)

        Args:
            output: Model's response
            reference: Reference answer

        Returns:
            Quality score (0-1 scale)
        """
        import random

        # Probabilistic evaluator selection
        rand = random.random()
        if rand < 0.3:
            evaluators = ["semantic"]
        elif rand < 0.6:
            evaluators = ["groundedness"]
        elif rand < 0.7:
            evaluators = ["semantic", "groundedness"]
        elif rand < 0.8:
            evaluators = ["semantic", "factuality"]
        else:
            evaluators = ["factuality"]

        try:
            result = await evaluate(
                output=output,
                reference=reference,
                evaluators=evaluators,
                model=self.arbiter_model,
            )
            return result.overall_score
        except Exception as e:
            console.print(f"[red]Warning: Arbiter evaluation failed: {e}[/red]")
            return 0.5  # Neutral score on evaluation failure
