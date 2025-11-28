"""BenchmarkRunner for orchestrating bandit algorithm evaluation.

Executes queries across multiple bandit algorithms, collects feedback,
and generates comprehensive benchmark results.

Supports pluggable evaluation:
- Arbiter (legacy): LLM-as-judge for production routing
- ExactMatch: For GSM8K/MMLU (objective, no LLM judge)
- CodeExecution: For HumanEval (pass/fail based on test execution)
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from arbiter_ai import evaluate
from rich.console import Console
from tqdm.asyncio import tqdm_asyncio

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

if TYPE_CHECKING:
    from conduit_bench.evaluators.base import BaseEvaluator


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
        evaluator: "BaseEvaluator | None" = None,
    ) -> None:
        """Initialize BenchmarkRunner.

        Args:
            algorithms: List of bandit algorithms to benchmark
            executor: ModelExecutor for LLM execution (creates default if None)
            arbiter_model: Model to use for quality evaluation (legacy Arbiter mode)
            max_concurrency: Maximum number of parallel query executions
            database: BenchmarkDatabase instance (creates default if None)
            enable_db_write: Whether to enable streaming database writes
            analyzer: QueryAnalyzer for feature extraction (creates default if None)
            evaluator: Pluggable evaluator (ExactMatch, CodeExecution). If None,
                       uses legacy Arbiter LLM-as-judge evaluation.
        """
        self.algorithms = algorithms
        self.executor = executor or ModelExecutor()
        self.arbiter_model = arbiter_model
        self.max_concurrency = max_concurrency
        self.database = database
        self.enable_db_write = enable_db_write
        self.analyzer = analyzer or QueryAnalyzer()
        self.evaluator = evaluator  # None means use Arbiter (legacy)

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
        cumulative_cost: list[float] = []
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

            # Get fallback chain from algorithm
            fallback_arms = algorithm.get_fallback_chain(features, exclude=selected_arm, max_fallbacks=3)

            # Execute query with fallback support
            execution_result = await self.executor.execute_with_fallback(
                primary_arm=selected_arm,
                fallback_arms=fallback_arms,
                query_text=query.query_text,
                system_prompt="You are a helpful assistant.",
            )

            # Evaluate quality using pluggable evaluator or Arbiter (legacy)
            has_reference = bool(query.reference_answer)

            if execution_result.success:
                if self.evaluator is not None:
                    # Use pluggable evaluator (ExactMatch, CodeExecution)
                    quality_score = await self._evaluate_with_evaluator(
                        response=execution_result.response_text,
                        query=query,
                    )
                elif has_reference:
                    # Legacy: Arbiter with reference
                    quality_score = await self._evaluate_with_reference(
                        output=execution_result.response_text,
                        query=query.query_text,
                        reference=query.reference_answer,
                    )
                else:
                    # Legacy: Arbiter without reference
                    quality_score = await self._evaluate_without_reference(
                        output=execution_result.response_text,
                        query=query.query_text,
                    )
            else:
                quality_score = 0.0  # Failed execution = zero quality

            # Create evaluation record (all queries now evaluated)
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
                        model_id=execution_result.model_id,  # Use actually used model (may be fallback)
                        quality_score=quality_score,
                        cost=execution_result.cost,
                        latency=execution_result.latency,
                        success=execution_result.success,
                        metadata={
                            "response_length": len(execution_result.response_text),
                            "response_text": execution_result.response_text,  # Store actual response
                            "has_reference": has_reference,
                            "was_fallback": execution_result.was_fallback,
                            "primary_model": execution_result.primary_model,
                            "failed_models": execution_result.failed_models,
                            "error_details": execution_result.error,  # Error messages from failed models
                            "query_text": execution_result.query_text,  # For debugging
                        },
                    )
                except Exception as e:
                    console.print(f"[yellow]Warning: Failed to write query evaluation: {e}[/yellow]")

            # Update algorithm with feedback - penalize failures, reward success
            if execution_result.was_fallback and execution_result.failed_models:
                # Penalize all failed models with quality=0.0
                for failed_model_id in execution_result.failed_models:
                    failed_feedback = BanditFeedback(
                        model_id=failed_model_id,
                        cost=0.0,
                        quality_score=0.0,
                        latency=0.0,
                        success=False,
                    )
                    await algorithm.update(failed_feedback, features)

            # Reward the successful model (or penalize if all failed)
            bandit_feedback = BanditFeedback(
                model_id=execution_result.model_id,
                cost=execution_result.cost,
                quality_score=quality_score,
                latency=execution_result.latency,
                success=execution_result.success,
            )
            await algorithm.update(bandit_feedback, features)

            # Track cumulative metrics
            total_quality += quality_score
            total_cost += execution_result.cost
            cumulative_cost.append(total_cost)

        completed_at = datetime.now(timezone.utc)

        # Calculate final metrics (all queries are now evaluated)
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
            cumulative_cost=cumulative_cost,
            started_at=started_at,
            completed_at=completed_at,
            metadata=algorithm.get_stats(),
        )


    async def _evaluate_with_reference(self, output: str, query: str, reference: str) -> float:
        """Evaluate response quality when reference answer is available.

        Always uses ALL reference-based evaluators to maximize evaluation quality:
        - semantic: Embedding similarity to reference
        - groundedness: Output grounded in reference
        - factuality: Facts verified against reference

        Args:
            output: Model's response
            query: Original query (unused but kept for consistency)
            reference: Reference answer

        Returns:
            Quality score (0-1 scale)
        """
        try:
            result = await evaluate(
                output=output,
                reference=reference,
                evaluators=["semantic", "groundedness", "factuality"],
                model=self.arbiter_model,
            )
            return result.overall_score
        except Exception as e:
            console.print(f"[red]Warning: Reference-based evaluation failed: {e}[/red]")
            return 0.5  # Neutral score on evaluation failure

    async def _evaluate_without_reference(self, output: str, query: str) -> float:
        """Evaluate response quality when NO reference answer is available.

        Uses probabilistic mix of query-based evaluators:
        - 50% relevance + semantic (query alignment + semantic similarity to query)
        - 30% factuality + semantic (fact-checking + query topic alignment)
        - 20% relevance + factuality + semantic (comprehensive check)

        Args:
            output: Model's response
            query: Original query

        Returns:
            Quality score (0-1 scale)
        """
        import random

        # Probabilistic evaluator selection
        rand = random.random()
        if rand < 0.5:
            evaluators = ["relevance", "semantic"]
        elif rand < 0.8:
            evaluators = ["factuality", "semantic"]
        else:
            evaluators = ["relevance", "factuality", "semantic"]

        try:
            result = await evaluate(
                output=output,
                reference=query,  # Use query as reference for relevance/semantic
                evaluators=evaluators,
                model=self.arbiter_model,
            )
            return result.overall_score
        except Exception as e:
            console.print(f"[red]Warning: Query-based evaluation failed: {e}[/red]")
            return 0.5  # Neutral score on evaluation failure

    async def _evaluate_with_evaluator(
        self,
        response: str,
        query: BenchmarkQuery,
    ) -> float:
        """Evaluate using pluggable evaluator (ExactMatch, CodeExecution).

        This method uses objective evaluation without LLM-as-judge:
        - ExactMatch: Extract answer from response, compare to ground truth
        - CodeExecution: Run code, check if tests pass

        Args:
            response: Model's response text
            query: BenchmarkQuery with ground truth in reference_answer

        Returns:
            Quality score (1.0 if correct, 0.0 if incorrect)
        """
        if self.evaluator is None:
            console.print("[red]Warning: No evaluator configured[/red]")
            return 0.5

        try:
            # Get additional kwargs from query metadata for code execution
            kwargs = {}
            if "prompt" in query.metadata:
                kwargs["prompt"] = query.metadata["prompt"]
            if "entry_point" in query.metadata:
                kwargs["entry_point"] = query.metadata["entry_point"]

            result = self.evaluator.evaluate(
                response=response,
                ground_truth=query.reference_answer or "",
                query=query.query_text,
                **kwargs,
            )
            return result.score
        except Exception as e:
            console.print(f"[red]Warning: Evaluator failed: {e}[/red]")
            return 0.0  # Evaluation failure = incorrect
