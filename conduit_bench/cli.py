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
    ContextualThompsonSamplingBandit,
    DuelingBandit,
    EpsilonGreedyBandit,
    LinUCBBandit,
    ModelArm,
    ThompsonSamplingBandit,
    UCB1Bandit,
)
from conduit.engines.bandits.baselines import (
    AlwaysBestBaseline,
    AlwaysCheapestBaseline,
    OracleBaseline,
    RandomBaseline,
)

from conduit_bench.benchmark_models import BenchmarkQuery, BenchmarkResult
from conduit_bench.generators.synthetic import SyntheticQueryGenerator
from conduit_bench.runners.benchmark_runner import BenchmarkRunner

console = Console()

# Default model arms for benchmarking (Latest models as of November 2025)
# 9 arms across 3 providers at 3 price tiers
DEFAULT_ARMS = [
    # BUDGET TIER
    ModelArm(
        model_id="gpt-5-nano",
        model_name="gpt-5-nano",
        provider="openai",
        cost_per_input_token=0.00000005,  # $0.05/1M tokens
        cost_per_output_token=0.0000004,  # $0.40/1M tokens
        expected_quality=0.78,
    ),
    ModelArm(
        model_id="claude-haiku-4-5",
        model_name="claude-haiku-4-5",
        provider="anthropic",
        cost_per_input_token=0.000001,  # $1.00/1M tokens
        cost_per_output_token=0.000005,  # $5.00/1M tokens
        expected_quality=0.82,
    ),
    ModelArm(
        model_id="gemini-2.0-flash",
        model_name="gemini-2.0-flash",
        provider="google",
        cost_per_input_token=0.0000001,  # $0.10/1M tokens
        cost_per_output_token=0.0000004,  # $0.40/1M tokens
        expected_quality=0.80,
    ),
    # MID TIER
    ModelArm(
        model_id="gpt-5-mini",
        model_name="gpt-5-mini",
        provider="openai",
        cost_per_input_token=0.00000025,  # $0.25/1M tokens
        cost_per_output_token=0.000002,  # $2.00/1M tokens
        expected_quality=0.86,
    ),
    ModelArm(
        model_id="claude-sonnet-4-5",
        model_name="claude-sonnet-4-5",
        provider="anthropic",
        cost_per_input_token=0.000003,  # $3.00/1M tokens
        cost_per_output_token=0.000015,  # $15.00/1M tokens
        expected_quality=0.92,
    ),
    ModelArm(
        model_id="gemini-2.5-pro",
        model_name="gemini-2.5-pro",
        provider="google",
        cost_per_input_token=0.00000125,  # $1.25/1M tokens
        cost_per_output_token=0.000005,  # $5.00/1M tokens
        expected_quality=0.88,
    ),
    # PREMIUM TIER
    ModelArm(
        model_id="gpt-5",
        model_name="gpt-5",
        provider="openai",
        cost_per_input_token=0.00000125,  # $1.25/1M tokens
        cost_per_output_token=0.00001,  # $10.00/1M tokens
        expected_quality=0.95,
    ),
    ModelArm(
        model_id="claude-opus-4-5",
        model_name="claude-opus-4-5",
        provider="anthropic",
        cost_per_input_token=0.000005,  # $5.00/1M tokens
        cost_per_output_token=0.000025,  # $25.00/1M tokens
        expected_quality=0.97,
    ),
    ModelArm(
        model_id="gemini-3-pro",
        model_name="gemini-3-pro",
        provider="google",
        cost_per_input_token=0.000002,  # $2.00/1M tokens
        cost_per_output_token=0.000012,  # $12.00/1M tokens
        expected_quality=0.94,
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
@click.option(
    "--max-concurrency",
    type=int,
    default=5,
    help="Maximum number of parallel LLM calls (default: 5, recommended: 20-30 for speed)",
    show_default=True,
)
@click.option(
    "--oracle-reference",
    type=click.Path(exists=True, path_type=Path),
    help="Path to pre-computed Oracle results (skip Oracle run, use cached results)",
)
def run(
    dataset: Path,
    algorithms: str,
    output: Path,
    max_queries: int | None,
    parallel: bool,
    max_concurrency: int,
    oracle_reference: Path | None,
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

        # If oracle reference provided, skip oracle in algorithm list
        if oracle_reference:
            if "oracle" in algo_names:
                algo_names.remove("oracle")
                console.print(f"[yellow]Using cached Oracle results from {oracle_reference}[/yellow]")

        console.print(f"Algorithms: {', '.join(algo_names)}\n")

        # Create algorithm instances
        algorithm_map = {
            # Standard (non-contextual) bandits
            "thompson": ThompsonSamplingBandit(DEFAULT_ARMS),
            "ucb1": UCB1Bandit(DEFAULT_ARMS, c=1.5),
            "epsilon": EpsilonGreedyBandit(DEFAULT_ARMS, epsilon=0.1),
            # Contextual bandits
            "linucb": LinUCBBandit(DEFAULT_ARMS, alpha=1.0, feature_dim=5),
            "contextual_thompson": ContextualThompsonSamplingBandit(
                DEFAULT_ARMS, lambda_reg=1.0, feature_dim=5
            ),
            "dueling": DuelingBandit(DEFAULT_ARMS, feature_dim=5),
            # Baselines
            "random": RandomBaseline(DEFAULT_ARMS, random_seed=42),
            "oracle": OracleBaseline(DEFAULT_ARMS),
            "always_best": AlwaysBestBaseline(DEFAULT_ARMS),
            "always_cheapest": AlwaysCheapestBaseline(DEFAULT_ARMS),
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
            max_concurrency=max_concurrency,
        )

        # Store oracle reference path if provided
        if oracle_reference:
            console.print(f"[cyan]Oracle reference: {oracle_reference}[/cyan]")

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
    """Analyze benchmark results with comprehensive statistical metrics.

    Calculates:
    - Cumulative regret and convergence analysis
    - Statistical significance tests (Friedman, effect sizes)
    - Cost-quality Pareto frontier
    - Confidence intervals (95% via bootstrap)
    - Algorithm rankings and comparisons

    Example:
        conduit-bench analyze --results results/benchmark.json
    """
    from conduit_bench.analysis.metrics import analyze_benchmark_results

    console.print("\n[bold cyan]Analyzing benchmark results...[/bold cyan]")

    # Load results
    with open(results) as f:
        result_data = json.load(f)

    console.print(f"Dataset size: {result_data['dataset_size']} queries")
    console.print(f"Algorithms analyzed: {len(result_data['algorithms'])}\n")

    # Calculate comprehensive metrics
    analysis = analyze_benchmark_results(result_data)

    # Ensure output directory exists
    output.parent.mkdir(parents=True, exist_ok=True)

    # Save metrics
    with open(output, "w") as f:
        json.dump(analysis, f, indent=2)

    console.print(f"[bold green]✓ Comprehensive metrics saved to {output}[/bold green]\n")

    # Display summary
    _display_analysis_summary(analysis)


def _display_analysis_summary(analysis: dict) -> None:
    """Display analysis summary table."""
    console.print("[bold]Algorithm Performance Summary[/bold]\n")

    # Quality ranking table
    table = Table(show_header=True, header_style="bold cyan", title="Quality Ranking")
    table.add_column("Rank", style="yellow", width=6)
    table.add_column("Algorithm", style="green")
    table.add_column("Avg Quality", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("Converged", justify="center")

    comp = analysis["comparative_analysis"]
    for rank, (algo_name, quality) in enumerate(comp["quality_ranking"], 1):
        algo_metrics = analysis["algorithms"][algo_name]
        ci_lower = algo_metrics["quality_ci_lower"]
        ci_upper = algo_metrics["quality_ci_upper"]
        converged = "✓" if algo_metrics["converged"] else "✗"

        table.add_row(
            str(rank),
            algo_name,
            f"{quality:.3f}",
            f"[{ci_lower:.3f}, {ci_upper:.3f}]",
            converged,
        )

    console.print(table)

    # Cost ranking
    console.print("\n[bold]Cost Ranking[/bold]")
    for rank, (algo_name, cost) in enumerate(comp["cost_ranking"], 1):
        console.print(f"  {rank}. {algo_name}: ${cost:.4f}")

    # Pareto frontier
    console.print(f"\n[bold]Pareto Optimal Algorithms:[/bold] {', '.join(comp['pareto_optimal'])}")

    # Statistical significance
    friedman = comp["friedman_test"]
    if friedman["significant"]:
        console.print(
            f"\n[bold green]✓ Friedman test: Significant differences detected (p={friedman['p_value']:.4f})[/bold green]"
        )
    else:
        console.print(
            f"\n[yellow]Friedman test: No significant differences (p={friedman['p_value']:.4f})[/yellow]"
        )


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
@click.option(
    "--analysis",
    "-a",
    type=click.Path(exists=True, path_type=Path),
    help="Path to analysis metrics JSON (if separate from results)",
)
def visualize(results: Path, output: Path, analysis: Path | None) -> None:
    """Generate publication-quality visualizations from benchmark results.

    Creates:
    - Regret curves with 95% CI bands
    - Cost-quality scatter plot (Pareto frontier)
    - Convergence speed comparison
    - Quality ranking with error bars
    - Comprehensive HTML report

    Example:
        conduit-bench visualize --results results/benchmark.json --output charts/
    """
    from conduit_bench.analysis.metrics import analyze_benchmark_results
    from conduit_bench.analysis.visualize import create_all_visualizations

    console.print("\n[bold cyan]Generating visualizations...[/bold cyan]")

    # Load results
    with open(results) as f:
        result_data = json.load(f)

    # Get analysis (either from separate file or compute from results)
    if analysis:
        with open(analysis) as f:
            analysis_data = json.load(f)
    else:
        console.print("Computing metrics from results...")
        analysis_data = analyze_benchmark_results(result_data)

    # Create output directory
    output.mkdir(parents=True, exist_ok=True)

    # Generate all visualizations
    console.print(f"Output directory: {output}\n")
    console.print("Creating visualizations:")

    chart_paths = create_all_visualizations(analysis_data, output, result_data)

    # Display created files
    for viz_type, path in chart_paths.items():
        if path.suffix == ".html":
            console.print(f"  📄 {viz_type}: {path}")
        else:
            console.print(f"  📊 {viz_type}: {path}")

    console.print(f"\n[bold green]✓ All visualizations saved to {output}[/bold green]")
    console.print(f"[bold]View HTML report:[/bold] {output / 'benchmark_report.html'}")


if __name__ == "__main__":
    main()
