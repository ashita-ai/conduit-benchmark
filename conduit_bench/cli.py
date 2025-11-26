"""Command-line interface for conduit-bench.

Provides commands for generating datasets, running benchmarks, and analyzing results.
"""

import asyncio
import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from conduit.engines.bandits import (
    EpsilonGreedyBandit,
    ModelArm,
    ThompsonSamplingBandit,
    UCB1Bandit,
)

from conduit_bench.benchmark_models import BenchmarkQuery, BenchmarkResult
from conduit_bench.generators.synthetic import SyntheticQueryGenerator
from conduit_bench.runners.benchmark_runner import BenchmarkRunner

console = Console()

# Default model arms for benchmarking (real current models)
DEFAULT_ARMS = [
    ModelArm(
        model_id="gpt-4o-mini",
        model_name="gpt-4o-mini",
        provider="openai",
        cost_per_input_token=0.00000015,  # $0.150/1M tokens
        cost_per_output_token=0.0000006,  # $0.600/1M tokens
        expected_quality=0.85,
    ),
    ModelArm(
        model_id="gpt-4o",
        model_name="gpt-4o",
        provider="openai",
        cost_per_input_token=0.0000025,  # $2.50/1M tokens
        cost_per_output_token=0.00001,  # $10.00/1M tokens
        expected_quality=0.95,
    ),
    ModelArm(
        model_id="claude-3-5-haiku-20241022",
        model_name="claude-3-5-haiku-20241022",
        provider="anthropic",
        cost_per_input_token=0.000001,  # $1.00/1M tokens
        cost_per_output_token=0.000005,  # $5.00/1M tokens
        expected_quality=0.80,
    ),
]


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """Conduit-Bench: Benchmark bandit algorithms for the Conduit Router.

    Compare Thompson Sampling, UCB, and Epsilon-Greedy algorithms across
    multiple LLM models to identify optimal cost/quality trade-offs.
    """
    pass


@main.command()
@click.option(
    "--queries",
    "-n",
    type=int,
    default=1000,
    help="Number of queries to generate",
    show_default=True,
)
@click.option(
    "--seed",
    "-s",
    type=int,
    default=42,
    help="Random seed for reproducibility",
    show_default=True,
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="data/queries.jsonl",
    help="Output file path (JSONL format)",
    show_default=True,
)
@click.option(
    "--no-reference",
    is_flag=True,
    help="Skip reference answer generation (faster, for testing)",
)
@click.option(
    "--categories",
    type=str,
    help="Comma-separated list of categories (default: all 10)",
)
@click.option(
    "--reference-probability",
    "-p",
    type=float,
    default=0.7,
    help="Probability of generating reference answers (0.0-1.0)",
    show_default=True,
)
def generate(
    queries: int,
    seed: int,
    output: Path,
    no_reference: bool,
    categories: str | None,
    reference_probability: float,
) -> None:
    """Generate synthetic benchmark dataset.

    Creates diverse, complex queries across categories (technical, data engineering, etc.)
    with multipart questions and probabilistic reference answers from GPT-4o.

    Example:
        conduit-bench generate --queries 1000 --seed 42 --reference-probability 0.7
    """

    async def _generate() -> None:
        console.print(f"\n[bold cyan]Generating {queries} synthetic queries[/bold cyan]")

        # Parse categories
        category_list = None
        if categories:
            category_list = [c.strip() for c in categories.split(",")]
            console.print(f"Categories: {', '.join(category_list)}")

        # Initialize generator with reference probability
        # If --no-reference is used, set probability to 0.0
        actual_probability = 0.0 if no_reference else reference_probability
        generator = SyntheticQueryGenerator(
            seed=seed,
            reference_probability=actual_probability,
        )

        # Generate queries
        if no_reference:
            console.print("[yellow]Skipping reference answers (--no-reference)[/yellow]")
        elif reference_probability < 1.0:
            console.print(f"Generating with {int(reference_probability * 100)}% reference answers from GPT-4o...")
        else:
            console.print("Generating with reference answers from GPT-4o...")

        query_list = await generator.generate(
            n_queries=queries,
            categories=category_list,
            show_progress=True,
        )

        # Ensure output directory exists
        output.parent.mkdir(parents=True, exist_ok=True)

        # Save to JSONL
        with open(output, "w") as f:
            for query in query_list:
                f.write(query.model_dump_json() + "\n")

        console.print(
            f"\n[bold green]✓ Generated {len(query_list)} queries → {output}[/bold green]"
        )

        # Show sample
        console.print("\n[bold]Sample queries:[/bold]")
        for i, query in enumerate(query_list[:3], 1):
            category = query.metadata.get("category", "unknown")
            console.print(f"{i}. [{category}] {query.query_text[:80]}...")

    asyncio.run(_generate())


@main.command()
@click.option(
    "--dataset",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to JSONL dataset file",
)
@click.option(
    "--algorithms",
    "-a",
    type=str,
    default="thompson,ucb1,epsilon",
    help="Comma-separated list of algorithms to run",
    show_default=True,
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="results/benchmark.json",
    help="Output file for results",
    show_default=True,
)
@click.option(
    "--max-queries",
    type=int,
    help="Limit number of queries to process (for testing)",
)
@click.option(
    "--parallel",
    is_flag=True,
    help="Run algorithms in parallel (3x faster for large datasets)",
)
def run(
    dataset: Path,
    algorithms: str,
    output: Path,
    max_queries: int | None,
    parallel: bool,
) -> None:
    """Run benchmark experiments.

    Executes multiple bandit algorithms on the same dataset and compares
    their performance in terms of cost, quality, and regret.

    Example:
        conduit-bench run --dataset data/queries.jsonl --algorithms thompson,ucb1
    """

    async def _run() -> None:
        console.print("\n[bold cyan]Loading dataset...[/bold cyan]")

        # Load queries from JSONL
        queries: list[BenchmarkQuery] = []
        with open(dataset) as f:
            for line in f:
                if line.strip():
                    queries.append(BenchmarkQuery.model_validate_json(line))

        if max_queries:
            queries = queries[:max_queries]
            console.print(f"[yellow]Limited to first {max_queries} queries[/yellow]")

        console.print(f"Loaded {len(queries)} queries from {dataset}")

        # Parse algorithm list
        algo_names = [a.strip().lower() for a in algorithms.split(",")]
        console.print(f"Algorithms: {', '.join(algo_names)}\n")

        # Create algorithm instances
        algorithm_map = {
            "thompson": ThompsonSamplingBandit(DEFAULT_ARMS),
            "ucb1": UCB1Bandit(DEFAULT_ARMS, c=1.5),
            "epsilon": EpsilonGreedyBandit(DEFAULT_ARMS, epsilon=0.1),
        }

        selected_algorithms = []
        for name in algo_names:
            if name in algorithm_map:
                selected_algorithms.append(algorithm_map[name])
            else:
                console.print(
                    f"[red]Warning: Unknown algorithm '{name}', skipping[/red]"
                )

        if not selected_algorithms:
            console.print("[red]Error: No valid algorithms selected[/red]")
            return

        # Run benchmark
        runner = BenchmarkRunner(
            algorithms=selected_algorithms,
            arbiter_model="gpt-4o-mini",
            max_concurrency=5,
        )

        result: BenchmarkResult = await runner.run(
            dataset=queries,
            show_progress=True,
            parallel=parallel,
        )

        # Ensure output directory exists
        output.parent.mkdir(parents=True, exist_ok=True)

        # Save results
        with open(output, "w") as f:
            f.write(result.model_dump_json(indent=2))

        console.print(f"\n[bold green]✓ Results saved to {output}[/bold green]")

        # Display summary table
        _display_summary(result)

    asyncio.run(_run())


def _display_summary(result: BenchmarkResult) -> None:
    """Display benchmark results summary table."""
    console.print("\n[bold]Benchmark Results Summary[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Algorithm", style="yellow")
    table.add_column("Total Cost", justify="right")
    table.add_column("Avg Quality", justify="right")
    table.add_column("Queries", justify="right")

    for algo_run in result.algorithms:
        table.add_row(
            algo_run.algorithm_name,
            f"${algo_run.total_cost:.4f}",
            f"{algo_run.average_quality:.3f}",
            str(algo_run.total_queries),
        )

    console.print(table)


@main.command()
@click.option(
    "--results",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to benchmark results JSON file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="analysis/metrics.json",
    help="Output file for metrics",
    show_default=True,
)
def analyze(results: Path, output: Path) -> None:
    """Analyze benchmark results and calculate metrics.

    Calculates cumulative regret, cost savings, quality metrics,
    and convergence analysis.

    NOTE: Full implementation coming in Issue #17.
    Currently provides basic metrics from BenchmarkResult.

    Example:
        conduit-bench analyze --results results/benchmark.json
    """
    console.print("\n[bold cyan]Analyzing benchmark results...[/bold cyan]")

    # Load results
    with open(results) as f:
        result = BenchmarkResult.model_validate_json(f.read())

    console.print(f"Dataset size: {result.dataset_size} queries")
    console.print(f"Algorithms analyzed: {len(result.algorithms)}\n")

    # Calculate basic metrics
    metrics = {
        "benchmark_id": result.benchmark_id,
        "dataset_size": result.dataset_size,
        "algorithms": {},
    }

    for algo_run in result.algorithms:
        metrics["algorithms"][algo_run.algorithm_name] = {
            "total_cost": algo_run.total_cost,
            "average_quality": algo_run.average_quality,
            "total_queries": algo_run.total_queries,
            "cumulative_regret": algo_run.cumulative_regret[-1]
            if algo_run.cumulative_regret
            else 0.0,
        }

    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Save metrics
    with open(output, "w") as f:
        json.dump(metrics, f, indent=2)

    console.print(f"[bold green]✓ Basic metrics saved to {output}[/bold green]")
    console.print(
        "\n[yellow]ℹ Full metrics implementation coming in Issue #17:[/yellow]"
    )
    console.print("  - Statistical significance tests")
    console.print("  - Per-category breakdown")
    console.print("  - Convergence speed detection")
    console.print("  - Cost savings vs baselines")


@main.command()
@click.option(
    "--results",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to benchmark results JSON file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="charts/",
    help="Output directory for visualizations",
    show_default=True,
)
def visualize(results: Path, output: Path) -> None:
    """Generate visualizations from benchmark results.

    Creates regret curves, cost-quality plots, heatmaps, and HTML reports.

    NOTE: Full implementation coming in Issue #18.
    Currently a placeholder.

    Example:
        conduit-bench visualize --results results/benchmark.json
    """
    console.print("\n[bold yellow]Visualization module placeholder[/bold yellow]")
    console.print(f"Results: {results}")
    console.print(f"Output directory: {output}\n")

    console.print("[yellow]ℹ Full visualization implementation coming in Issue #18:[/yellow]")
    console.print("  - Regret curves over time")
    console.print("  - Cost-quality Pareto frontier")
    console.print("  - Model selection heatmaps")
    console.print("  - Convergence detection plots")
    console.print("  - HTML report generation")
    console.print("\n[dim]For now, use the analyze command to get JSON metrics.[/dim]")


if __name__ == "__main__":
    main()
