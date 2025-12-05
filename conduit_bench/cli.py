"""Command-line interface for conduit-bench.

Provides commands for generating datasets, running benchmarks, and analyzing results.

Supports three benchmark datasets with objective evaluation:
- GSM8K: Math reasoning (exact match on #### N)
- MMLU: Knowledge (exact match on A/B/C/D)
- HumanEval: Code generation (pass/fail via execution)
"""

import asyncio
import json
import os
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from conduit.core.config import load_context_priors, settings as conduit_settings
from conduit.core.pricing_manager import PricingManager
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
from conduit.engines.hybrid_router import HybridRouter

from conduit_bench.adapters import HybridRouterBanditAdapter
from conduit_bench.benchmark_models import BenchmarkQuery, BenchmarkResult
from conduit_bench.generators.synthetic import SyntheticQueryGenerator
from conduit_bench.runners.benchmark_runner import BenchmarkRunner

# Dataset loaders
from conduit_bench.datasets import GSM8KLoader, MMLULoader, HumanEvalLoader

# Evaluators
from conduit_bench.evaluators import ExactMatchEvaluator, CodeExecutionEvaluator

console = Console()


# ============================================================================
# 🚨 CANONICAL MODEL LIST - DO NOT MODIFY WITHOUT EXPLICIT USER APPROVAL 🚨
# ============================================================================
# These are the ACTUAL API model names used for benchmarking.
# Conduit's config now uses these directly (no confusing internal mapping).
#
# Sources:
#   - Anthropic: https://platform.claude.com/docs/en/about-claude/models/all-models
#   - Google: https://ai.google.dev/gemini-api/docs/models
#   - OpenAI: https://platform.openai.com/docs/models
#
# The mapping below is for backwards compatibility with old conduit configs.
# New conduit configs use API names directly, so mapping is identity.
# ============================================================================
CONDUIT_TO_API_MODEL = {
    # Identity mappings (conduit now uses API names directly)
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4-turbo": "gpt-4-turbo",
    "claude-sonnet-4-5-20250929": "claude-sonnet-4-5-20250929",
    "claude-opus-4-5-20251101": "claude-opus-4-5-20251101",
    "claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-3-pro-preview": "gemini-3-pro-preview",  # Cutting edge (preview)
    # Legacy mappings (backwards compat with old conduit internal names)
    "o4-mini": "gpt-4o-mini",
    "gpt-5": "gpt-4o",
    "gpt-5.1": "gpt-4-turbo",
    "claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
    "claude-opus-4.5": "claude-opus-4-5-20251101",
    "claude-haiku-4.5": "claude-haiku-4-5-20251001",
}


def _load_quality_priors(context: str = "general") -> dict[str, float]:
    """Load quality priors from Conduit's context-specific configuration.

    Args:
        context: Quality prior context (code, creative, analysis, simple_qa, general)

    Returns:
        Dictionary mapping model_id to expected quality (0.0-1.0)
    """
    # Load priors from conduit.yaml (returns dict[str, tuple[float, float]])
    priors_beta = load_context_priors(context)

    # Convert Beta distribution (alpha, beta) to quality scores
    quality_priors = {}
    for model_id, (alpha, beta) in priors_beta.items():
        quality = alpha / (alpha + beta)
        quality_priors[model_id] = quality

    return quality_priors


# Load quality priors at module import time (using "general" context by default)
# Contexts available: code, creative, analysis, simple_qa, general
# To use different context, set CONDUIT_QUALITY_CONTEXT env var before import
_QUALITY_CONTEXT = os.getenv("CONDUIT_QUALITY_CONTEXT", "general")
_QUALITY_PRIORS = _load_quality_priors(_QUALITY_CONTEXT)


async def _load_pricing_from_conduit():
    """Load pricing from Conduit's PricingManager.

    Returns dict mapping model_id to pricing info with keys:
    - input_cost_per_1m: Input cost per 1M tokens
    - output_cost_per_1m: Output cost per 1M tokens
    """
    pricing_manager = PricingManager(database=None)  # Use cache-only mode
    pricing_data = await pricing_manager.get_pricing()

    # Convert ModelPricing objects to dict format needed by get_default_arms()
    return {
        model_id: {
            "input_cost_per_1m": pricing.input_cost_per_million,
            "output_cost_per_1m": pricing.output_cost_per_million,
        }
        for model_id, pricing in pricing_data.items()
    }


# Load pricing at module import time
_PRICING_CACHE = asyncio.run(_load_pricing_from_conduit())


def _detect_provider(model_id: str) -> str:
    """Detect provider from model ID.

    Maps model IDs to PydanticAI provider format.
    """
    model_lower = model_id.lower()
    if any(x in model_lower for x in ["gpt", "o1", "o3", "o4", "davinci", "turbo"]):
        return "openai"
    elif any(x in model_lower for x in ["claude", "opus", "sonnet", "haiku"]):
        return "anthropic"
    elif "gemini" in model_lower:
        return "google-gla"  # Google AI Studio (uses API keys)
    elif "mistral" in model_lower or "mixtral" in model_lower:
        return "mistral"
    elif "grok" in model_lower:
        return "xai"
    elif "deepseek" in model_lower:
        return "deepseek"
    else:
        return "openai"  # Default fallback


def get_default_arms() -> list[ModelArm]:
    """Build ModelArm list from conduit's configuration.

    Uses conduit.core.config.settings.default_models as the single source of truth.
    Maps conduit's internal model names to actual API model names.

    Pricing is loaded dynamically from Conduit's PricingManager at module import time.
    Quality priors are used by baseline algorithms (AlwaysBest/AlwaysCheapest) to make
    informed decisions. Actual execution costs are calculated dynamically by Arbiter.

    Returns:
        List of ModelArm objects for benchmarking.
    """
    arms = []
    for conduit_model_id in conduit_settings.default_models:
        # Map conduit name to API name (fallback to same name if not mapped)
        api_model_name = CONDUIT_TO_API_MODEL.get(conduit_model_id, conduit_model_id)
        provider = _detect_provider(api_model_name)

        # Get pricing from Conduit's PricingManager (fallback to defaults if not found)
        pricing = _PRICING_CACHE.get(conduit_model_id, {
            "input_cost_per_1m": 2.0,
            "output_cost_per_1m": 8.0,
        })

        # Get quality prior (fallback to 0.80 if not found)
        expected_quality = _QUALITY_PRIORS.get(conduit_model_id, 0.80)

        # Convert $/1M to $/token for ModelArm
        input_cost_per_token = pricing["input_cost_per_1m"] / 1_000_000
        output_cost_per_token = pricing["output_cost_per_1m"] / 1_000_000

        arms.append(ModelArm(
            model_id=conduit_model_id,  # Keep conduit ID for tracking
            model_name=api_model_name,  # Use API name for execution
            provider=provider,
            cost_per_input_token=input_cost_per_token,  # Prior for baselines
            cost_per_output_token=output_cost_per_token,  # Prior for baselines
            expected_quality=expected_quality,  # Prior for baselines
        ))
    return arms


# Default arms from conduit's configuration
DEFAULT_ARMS = get_default_arms()


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
    type=str,
    required=True,
    help="Dataset: 'gsm8k', 'mmlu', 'humaneval', or path to JSONL file",
)
@click.option(
    "--preset",
    "-p",
    type=click.Choice(["balanced", "quality", "cost", "speed"]),
    help=(
        "Algorithm preset configuration:\n"
        "  balanced: Best mix of learning algorithms\n"
        "  quality: Prioritize accuracy over cost\n"
        "  cost: Minimize inference costs\n"
        "  speed: Fast non-contextual algorithms\n"
        "Note: Oracle excluded from all presets (6x cost)"
    ),
)
@click.option(
    "--algorithms",
    "-a",
    type=str,
    default="hybrid_thompson_linucb,ucb1,random",
    help="Comma-separated list of algorithms to run (overrides --preset)",
    show_default=True,
)
@click.option(
    "--evaluator",
    "-e",
    type=click.Choice(["exact_match", "code_execution", "arbiter"]),
    default=None,
    help="Evaluator type (auto-detected from dataset if not specified)",
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
@click.option(
    "--mmlu-limit",
    type=int,
    default=1000,
    help="Number of MMLU questions to use (default: 1000 of 14k)",
    show_default=True,
)
@click.option(
    "--code-timeout",
    type=int,
    default=10,
    help="Timeout in seconds for HumanEval code execution",
    show_default=True,
)
def run(
    dataset: str,
    preset: str | None,
    algorithms: str,
    evaluator: str | None,
    output: Path,
    max_queries: int | None,
    parallel: bool,
    max_concurrency: int,
    oracle_reference: Path | None,
    mmlu_limit: int,
    code_timeout: int,
) -> None:
    """Run benchmark experiments.

    Executes multiple bandit algorithms on established benchmarks and compares
    their performance in terms of accuracy, cost, and regret.

    Datasets:
      - gsm8k: Grade school math (1,319 problems, exact match)
      - mmlu: Multi-subject knowledge (1k of 14k, exact match)
      - humaneval: Python coding (164 problems, code execution)
      - /path/to/file.jsonl: Custom JSONL dataset (Arbiter evaluation)

    Example:
        conduit-bench run --dataset gsm8k --algorithms hybrid,ucb1,random
        conduit-bench run --dataset mmlu --mmlu-limit 500
        conduit-bench run --dataset humaneval --code-timeout 15
    """

    async def _run() -> None:
        # Handle preset configurations
        # Oracle excluded from all presets due to 6x cost (executes all 6 models per query)
        ALGORITHM_PRESETS = {
            "balanced": "thompson,linucb,contextual_thompson,hybrid_thompson_linucb,random",
            "quality": "contextual_thompson,linucb,dueling,always_best",
            "cost": "linucb,always_cheapest,random",
            "speed": "thompson,ucb1,epsilon,random",
        }

        # Use preset if specified, otherwise use --algorithms value
        selected_algorithms = algorithms
        if preset:
            selected_algorithms = ALGORITHM_PRESETS[preset]
            console.print(
                f"\n[bold cyan]Using '{preset}' preset:[/bold cyan] {selected_algorithms}"
            )

        console.print("\n[bold cyan]Loading dataset...[/bold cyan]")

        # Determine dataset type and load accordingly
        queries: list[BenchmarkQuery] = []
        selected_evaluator = None
        dataset_name = dataset.lower()

        if dataset_name == "gsm8k":
            # Load GSM8K from HuggingFace
            console.print("[cyan]Loading GSM8K from HuggingFace...[/cyan]")
            loader = GSM8KLoader()
            queries = loader.load(split="test", limit=max_queries)
            selected_evaluator = ExactMatchEvaluator(dataset_type="gsm8k")
            console.print(f"[green]Loaded {len(queries)} GSM8K test problems[/green]")

        elif dataset_name == "mmlu":
            # Load MMLU from HuggingFace
            console.print("[cyan]Loading MMLU from HuggingFace...[/cyan]")
            loader = MMLULoader()
            limit = max_queries or mmlu_limit
            queries = loader.load(split="test", limit=limit)
            selected_evaluator = ExactMatchEvaluator(dataset_type="mmlu")
            console.print(f"[green]Loaded {len(queries)} MMLU questions[/green]")

        elif dataset_name == "humaneval":
            # Load HumanEval from HuggingFace
            console.print("[cyan]Loading HumanEval from HuggingFace...[/cyan]")
            loader = HumanEvalLoader()
            queries = loader.load(limit=max_queries)
            selected_evaluator = CodeExecutionEvaluator(timeout=code_timeout)
            console.print(f"[green]Loaded {len(queries)} HumanEval problems[/green]")

        else:
            # Custom JSONL file (legacy behavior)
            dataset_path = Path(dataset)
            if not dataset_path.exists():
                console.print(f"[red]Error: Dataset '{dataset}' not found[/red]")
                console.print("Available datasets: gsm8k, mmlu, humaneval, or path to JSONL file")
                return

            console.print(f"[cyan]Loading custom dataset from {dataset_path}...[/cyan]")
            with open(dataset_path) as f:
                for line in f:
                    if line.strip():
                        queries.append(BenchmarkQuery.model_validate_json(line))

            if max_queries:
                queries = queries[:max_queries]

            # Use Arbiter for custom datasets (legacy behavior)
            selected_evaluator = None  # Triggers Arbiter fallback
            console.print(f"[green]Loaded {len(queries)} queries from {dataset_path}[/green]")
            console.print("[yellow]Using Arbiter LLM-as-judge evaluation[/yellow]")

        # Override evaluator if explicitly specified
        if evaluator:
            if evaluator == "exact_match":
                # Determine type from dataset
                if dataset_name == "mmlu":
                    selected_evaluator = ExactMatchEvaluator(dataset_type="mmlu")
                else:
                    selected_evaluator = ExactMatchEvaluator(dataset_type="gsm8k")
            elif evaluator == "code_execution":
                selected_evaluator = CodeExecutionEvaluator(timeout=code_timeout)
            elif evaluator == "arbiter":
                selected_evaluator = None  # Arbiter fallback

        if selected_evaluator:
            console.print(f"[cyan]Evaluator: {selected_evaluator.name}[/cyan]")

        # Parse algorithm list
        algo_names = [a.strip().lower() for a in selected_algorithms.split(",")]

        # If oracle reference provided, skip oracle in algorithm list
        if oracle_reference:
            if "oracle" in algo_names:
                algo_names.remove("oracle")
                console.print(f"[yellow]Using cached Oracle results from {oracle_reference}[/yellow]")

        console.print(f"Algorithms: {', '.join(algo_names)}\n")

        # Create QueryAnalyzer to get the correct feature dimension for contextual bandits
        # The analyzer's feature_dim depends on the embedding provider (OpenAI=1538, FastEmbed=386)
        from conduit.engines.analyzer import QueryAnalyzer
        analyzer = QueryAnalyzer()
        feature_dim = analyzer.feature_dim
        console.print(f"[cyan]Feature dimension: {feature_dim}[/cyan]")

        # Create algorithm instances
        # HybridRouter takes model names (API names like "gpt-4o-mini", not conduit IDs like "o4-mini")
        # This is needed because HybridRouter._infer_provider() pattern-matches on the model name
        model_names = [arm.model_name for arm in DEFAULT_ARMS]
        algorithm_map = {
            # Hybrid Router Configurations (2 core variants)
            # Thompson → LinUCB (quality-first cold start)
            "hybrid_thompson_linucb": HybridRouterBanditAdapter(
                HybridRouter(
                    model_names,
                    switch_threshold=50,
                    phase1_algorithm="thompson_sampling",
                    phase2_algorithm="linucb",
                    feature_dim=feature_dim,  # Pass correct feature dimension
                )
            ),
            # UCB1 → LinUCB (fast convergence)
            "hybrid_ucb1_linucb": HybridRouterBanditAdapter(
                HybridRouter(
                    model_names,
                    switch_threshold=50,
                    phase1_algorithm="ucb1",
                    phase2_algorithm="linucb",
                    feature_dim=feature_dim,  # Pass correct feature dimension
                )
            ),
            # Standard (non-contextual) bandits
            "thompson": ThompsonSamplingBandit(DEFAULT_ARMS),
            "ucb1": UCB1Bandit(DEFAULT_ARMS, c=1.5),
            "epsilon": EpsilonGreedyBandit(DEFAULT_ARMS, epsilon=0.1),
            # Contextual bandits - use feature_dim from QueryAnalyzer
            "linucb": LinUCBBandit(DEFAULT_ARMS, alpha=1.0, feature_dim=feature_dim),
            "contextual_thompson": ContextualThompsonSamplingBandit(
                DEFAULT_ARMS, lambda_reg=1.0, feature_dim=feature_dim
            ),
            "dueling": DuelingBandit(DEFAULT_ARMS, feature_dim=feature_dim),
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
            evaluator=selected_evaluator,  # Pluggable evaluator (None = Arbiter)
            analyzer=analyzer,  # Use the same analyzer for consistent feature dimensions
        )

        # Store oracle reference path if provided
        if oracle_reference:
            console.print(f"[cyan]Oracle reference: {oracle_reference}[/cyan]")

        # Build comprehensive metadata for benchmark run
        benchmark_metadata = {
            # Dataset information
            "dataset": {
                "name": dataset,
                "size": len(queries),
                "max_queries": max_queries,
            },
            # Evaluator configuration
            "evaluator": {
                "type": evaluator if evaluator else ("exact_match" if dataset in ["mmlu", "gsm8k"] else "code_execution" if dataset == "humaneval" else "arbiter"),
                "model": "gpt-4o-mini" if not selected_evaluator else None,
            },
            # Model pool configuration
            "model_pool": {
                "arms": [
                    {
                        "model_id": arm.model_id,
                        "model_name": arm.model_name,
                        "provider": arm.provider,
                        "input_price_per_m": arm.cost_per_input_token * 1000,  # Convert per-1K to per-1M
                        "output_price_per_m": arm.cost_per_output_token * 1000,
                        "expected_quality": arm.expected_quality,
                    }
                    for arm in DEFAULT_ARMS
                ]
            },
            # Benchmark configuration
            "config": {
                "max_concurrency": max_concurrency,
                "timeout_per_query": 30,  # Default from ModelExecutor
            },
        }

        # Add dataset-specific configuration
        if dataset == "mmlu":
            benchmark_metadata["dataset"]["mmlu_limit"] = mmlu_limit
        elif dataset == "humaneval":
            benchmark_metadata["config"]["code_timeout"] = code_timeout

        result: BenchmarkResult = await runner.run(
            dataset=queries,
            show_progress=True,
            parallel=parallel,
            benchmark_metadata=benchmark_metadata,
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
    from dataclasses import asdict, is_dataclass
    from conduit_bench.analysis.metrics import analyze_benchmark_results

    def convert_to_dict(obj):
        """Recursively convert dataclasses to dicts for JSON serialization."""
        import numpy as np

        # Handle numpy types
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        # Handle dataclasses
        elif is_dataclass(obj) and not isinstance(obj, type):
            return {k: convert_to_dict(v) for k, v in asdict(obj).items()}
        # Handle dictionaries
        elif isinstance(obj, dict):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        # Handle lists and tuples
        elif isinstance(obj, (list, tuple)):
            return [convert_to_dict(item) for item in obj]
        # Handle booleans explicitly
        elif isinstance(obj, bool):
            return obj
        # Handle None, strings, numbers
        elif obj is None or isinstance(obj, (str, int, float)):
            return obj
        # For anything else, try to convert to a string
        else:
            return str(obj)

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

    # Convert dataclasses to dicts for JSON serialization
    analysis_dict = convert_to_dict(analysis)

    # Save metrics
    with open(output, "w") as f:
        json.dump(analysis_dict, f, indent=2)

    console.print(f"[bold green]✓ Comprehensive metrics saved to {output}[/bold green]\n")

    # Display summary
    _display_analysis_summary(analysis)


def _display_analysis_summary(analysis: dict) -> None:
    """Display analysis summary table."""
    console.print("[bold]Algorithm Performance Summary[/bold]\n")

    # Handle empty algorithms
    if not analysis.get("algorithms"):
        console.print("[yellow]No algorithms to display[/yellow]")
        return

    # Quality ranking table
    table = Table(show_header=True, header_style="bold cyan", title="Quality Ranking")
    table.add_column("Rank", style="yellow", width=6)
    table.add_column("Algorithm", style="green")
    table.add_column("Avg Quality", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("Converged", justify="center")

    # Use summary for rankings
    quality_rankings = analysis.get("summary", {}).get("quality_rankings", [])
    for rank, algo_name in enumerate(quality_rankings, 1):
        algo_metrics = analysis["algorithms"].get(algo_name, {})
        avg_quality = algo_metrics.get("average_quality", 0.0)

        # Handle both tuple and separate ci fields
        quality_ci = algo_metrics.get("quality_ci")
        if quality_ci:
            ci_lower, ci_upper = quality_ci
        else:
            ci_lower = algo_metrics.get("quality_ci_lower", 0.0)
            ci_upper = algo_metrics.get("quality_ci_upper", 0.0)

        # Handle both nested and flat convergence
        convergence = algo_metrics.get("convergence", {})
        if isinstance(convergence, dict):
            converged = convergence.get("converged", False)
        else:
            converged = algo_metrics.get("converged", False)
        converged_str = "✓" if converged else "✗"

        table.add_row(
            str(rank),
            algo_name,
            f"{avg_quality:.3f}",
            f"[{ci_lower:.3f}, {ci_upper:.3f}]",
            converged_str,
        )

    console.print(table)

    # Cost ranking
    cost_rankings = analysis.get("summary", {}).get("cost_rankings", [])
    if cost_rankings:
        console.print("\n[bold]Cost Ranking[/bold]")
        for rank, algo_name in enumerate(cost_rankings, 1):
            algo_metrics = analysis["algorithms"].get(algo_name, {})
            cost = algo_metrics.get("total_cost", 0.0)
            console.print(f"  {rank}. {algo_name}: ${cost:.4f}")

    # Pareto frontier
    pareto = analysis.get("pareto_frontier", [])
    if pareto:
        console.print(f"\n[bold]Pareto Optimal Algorithms:[/bold] {', '.join(pareto)}")

    # Statistical significance
    stats = analysis.get("statistical_tests", {})
    friedman = stats.get("friedman", {})
    if friedman:
        p_value = friedman.get("p_value", 1.0)
        if friedman.get("significant", False):
            console.print(
                f"\n[bold green]✓ Friedman test: Significant differences detected (p={p_value:.4f})[/bold green]"
            )
        else:
            console.print(
                f"\n[yellow]Friedman test: No significant differences (p={p_value:.4f})[/yellow]"
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
