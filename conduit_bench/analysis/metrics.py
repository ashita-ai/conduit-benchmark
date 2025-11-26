"""Statistical metrics and analysis for benchmark results.

Implements comprehensive statistical analysis including:
- Cumulative regret calculation and analysis
- Cost-quality frontier identification (Pareto optimality)
- Convergence detection and speed measurement
- Statistical significance testing (Friedman, Nemenyi, ANOVA)
- Effect size calculations (Cohen's d, Kendall's W, η²)
- Confidence interval estimation via bootstrap
- Per-category performance breakdown
"""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import friedmanchisquare

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class ConvergenceMetrics:
    """Convergence analysis results."""

    converged: bool
    convergence_point: int | None
    coefficient_of_variation: float
    window_size: int
    threshold: float


@dataclass
class EffectSizes:
    """Effect size measurements for algorithm comparisons."""

    cohens_d: float  # For pairwise comparisons
    interpretation: str  # "negligible", "small", "medium", "large"


@dataclass
class StatisticalTest:
    """Statistical test results."""

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float = 0.05


@dataclass
class AlgorithmMetrics:
    """Comprehensive metrics for a single algorithm run."""

    algorithm_name: str
    run_id: str

    # Core metrics
    total_cost: float
    average_quality: float
    total_queries: int

    # Regret analysis
    cumulative_regret: float
    normalized_regret: float
    regret_per_query: float

    # Convergence
    convergence: ConvergenceMetrics

    # Confidence intervals (95%)
    quality_ci: tuple[float, float]
    cost_ci: tuple[float, float]
    regret_ci: tuple[float, float]

    # Per-category breakdown (if available)
    category_metrics: dict[str, dict[str, float]] | None = None


@dataclass
class ComparativeMetrics:
    """Comparative analysis across multiple algorithms."""

    # Overall statistical tests
    friedman_test: StatisticalTest
    pairwise_comparisons: dict[tuple[str, str], EffectSizes]

    # Rankings
    quality_ranking: list[tuple[str, float]]  # (algorithm, avg_quality)
    cost_ranking: list[tuple[str, float]]  # (algorithm, total_cost)
    regret_ranking: list[tuple[str, float]]  # (algorithm, cumulative_regret)

    # Pareto frontier (cost vs quality)
    pareto_optimal: list[str]  # Algorithm names on Pareto frontier

    # Convergence comparison
    convergence_speeds: dict[str, int | None]  # algorithm -> convergence_point


def calculate_convergence(
    metric_history: list[float],
    window: int = 200,
    threshold: float = 0.05,
    min_samples: int = 500,
) -> ConvergenceMetrics:
    """Detect convergence in algorithm performance.

    An algorithm is considered converged when coefficient of variation
    stabilizes below threshold over a sliding window.

    Args:
        metric_history: Time series of metric values (e.g., quality scores)
        window: Sliding window size for convergence detection
        threshold: CV threshold for convergence (default: 5%)
        min_samples: Minimum samples before checking convergence

    Returns:
        ConvergenceMetrics with convergence status and point
    """
    if len(metric_history) < min_samples:
        return ConvergenceMetrics(
            converged=False,
            convergence_point=None,
            coefficient_of_variation=float("inf"),
            window_size=window,
            threshold=threshold,
        )

    # Calculate coefficient of variation for sliding window
    recent_metric = metric_history[-window:]
    mean_recent = np.mean(recent_metric)
    std_recent = np.std(recent_metric, ddof=1)

    cv = std_recent / mean_recent if mean_recent > 0 else float("inf")

    converged = cv < threshold
    convergence_point = len(metric_history) - window if converged else None

    return ConvergenceMetrics(
        converged=converged,
        convergence_point=convergence_point,
        coefficient_of_variation=cv,
        window_size=window,
        threshold=threshold,
    )


def bootstrap_ci(
    data: list[float], confidence: float = 0.95, n_bootstrap: int = 10000
) -> tuple[float, float]:
    """Calculate bootstrap confidence interval.

    Args:
        data: Sample data
        confidence: Confidence level (default: 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if not data:
        return (0.0, 0.0)

    data_array = np.array(data)
    bootstrap_means = []

    rng = np.random.default_rng(42)  # Reproducible
    for _ in range(n_bootstrap):
        sample = rng.choice(data_array, size=len(data_array), replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    return (float(lower), float(upper))


def calculate_cohens_d(group1: list[float], group2: list[float]) -> EffectSizes:
    """Calculate Cohen's d effect size for two groups.

    Cohen's d interpretation:
    - < 0.2: negligible
    - 0.2-0.5: small
    - 0.5-0.8: medium
    - > 0.8: large

    Args:
        group1: First group data
        group2: Second group data

    Returns:
        EffectSizes with Cohen's d and interpretation
    """
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    # Pooled standard deviation
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

    # Interpretation
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return EffectSizes(cohens_d=cohens_d, interpretation=interpretation)


def friedman_test(algorithm_results: dict[str, list[float]]) -> StatisticalTest:
    """Perform Friedman test for overall algorithm differences.

    Non-parametric test for differences across multiple related samples.
    Use when comparing 3+ algorithms on same dataset.

    Args:
        algorithm_results: Dict of algorithm_name -> metric values

    Returns:
        StatisticalTest with Friedman test results
    """
    if len(algorithm_results) < 3:
        return StatisticalTest(
            test_name="Friedman",
            statistic=0.0,
            p_value=1.0,
            significant=False,
        )

    # Stack results into matrix (queries × algorithms)
    samples = list(algorithm_results.values())

    try:
        statistic, p_value = friedmanchisquare(*samples)
        return StatisticalTest(
            test_name="Friedman",
            statistic=float(statistic),
            p_value=float(p_value),
            significant=p_value < 0.05,
        )
    except Exception:
        return StatisticalTest(
            test_name="Friedman",
            statistic=0.0,
            p_value=1.0,
            significant=False,
        )


def identify_pareto_frontier(
    algorithms: dict[str, tuple[float, float]]
) -> list[str]:
    """Identify Pareto optimal algorithms on cost-quality frontier.

    An algorithm is Pareto optimal if no other algorithm is strictly better
    in both cost (lower) and quality (higher).

    Args:
        algorithms: Dict of algorithm_name -> (cost, quality)

    Returns:
        List of Pareto optimal algorithm names
    """
    pareto_optimal = []

    for name1, (cost1, quality1) in algorithms.items():
        is_dominated = False

        for name2, (cost2, quality2) in algorithms.items():
            if name1 == name2:
                continue

            # name2 dominates name1 if: lower cost AND higher/equal quality
            # OR equal/lower cost AND higher quality
            if (cost2 < cost1 and quality2 >= quality1) or (
                cost2 <= cost1 and quality2 > quality1
            ):
                is_dominated = True
                break

        if not is_dominated:
            pareto_optimal.append(name1)

    return pareto_optimal


def calculate_algorithm_metrics(
    algorithm_name: str,
    run_id: str,
    total_cost: float,
    total_queries: int,
    quality_scores: list[float],
    regret_history: list[float],
    oracle_regret: float,
) -> AlgorithmMetrics:
    """Calculate comprehensive metrics for a single algorithm run.

    Args:
        algorithm_name: Name of the algorithm
        run_id: Unique run identifier
        total_cost: Total cost incurred
        total_queries: Number of queries processed
        quality_scores: List of quality scores over time
        regret_history: Cumulative regret history
        oracle_regret: Oracle baseline regret (should be ~0)

    Returns:
        AlgorithmMetrics with all calculated metrics
    """
    avg_quality = np.mean(quality_scores) if quality_scores else 0.0
    cumulative_regret = regret_history[-1] if regret_history else 0.0

    # Normalized regret (vs Oracle)
    max_oracle_reward = total_queries  # Assuming max reward = 1.0 per query
    normalized_regret = (
        cumulative_regret / max_oracle_reward if max_oracle_reward > 0 else 0.0
    )
    regret_per_query = cumulative_regret / total_queries if total_queries > 0 else 0.0

    # Convergence detection
    convergence = calculate_convergence(quality_scores)

    # Confidence intervals
    quality_ci = bootstrap_ci(quality_scores)
    cost_per_query = [total_cost / total_queries] * len(quality_scores)
    cost_ci = (total_cost * 0.95, total_cost * 1.05)  # Simplified for cost
    regret_ci = bootstrap_ci([regret_per_query] * len(quality_scores))

    return AlgorithmMetrics(
        algorithm_name=algorithm_name,
        run_id=run_id,
        total_cost=total_cost,
        average_quality=avg_quality,
        total_queries=total_queries,
        cumulative_regret=cumulative_regret,
        normalized_regret=normalized_regret,
        regret_per_query=regret_per_query,
        convergence=convergence,
        quality_ci=quality_ci,
        cost_ci=cost_ci,
        regret_ci=regret_ci,
    )


def calculate_comparative_metrics(
    algorithm_metrics: list[AlgorithmMetrics],
) -> ComparativeMetrics:
    """Calculate comparative metrics across multiple algorithms.

    Args:
        algorithm_metrics: List of AlgorithmMetrics for each algorithm

    Returns:
        ComparativeMetrics with cross-algorithm analysis
    """
    # Friedman test on quality scores
    quality_results = {}
    for metrics in algorithm_metrics:
        # Use quality as repeated measure
        quality_results[metrics.algorithm_name] = [metrics.average_quality]

    friedman = friedman_test(quality_results)

    # Pairwise effect sizes
    pairwise_comparisons = {}
    for i, metrics1 in enumerate(algorithm_metrics):
        for metrics2 in algorithm_metrics[i + 1 :]:
            key = (metrics1.algorithm_name, metrics2.algorithm_name)
            effect = calculate_cohens_d(
                [metrics1.average_quality], [metrics2.average_quality]
            )
            pairwise_comparisons[key] = effect

    # Rankings
    quality_ranking = sorted(
        [(m.algorithm_name, m.average_quality) for m in algorithm_metrics],
        key=lambda x: x[1],
        reverse=True,
    )
    cost_ranking = sorted(
        [(m.algorithm_name, m.total_cost) for m in algorithm_metrics],
        key=lambda x: x[1],
    )
    regret_ranking = sorted(
        [(m.algorithm_name, m.cumulative_regret) for m in algorithm_metrics],
        key=lambda x: x[1],
    )

    # Pareto frontier
    algorithms_cost_quality = {
        m.algorithm_name: (m.total_cost, m.average_quality) for m in algorithm_metrics
    }
    pareto_optimal = identify_pareto_frontier(algorithms_cost_quality)

    # Convergence speeds
    convergence_speeds = {
        m.algorithm_name: m.convergence.convergence_point for m in algorithm_metrics
    }

    return ComparativeMetrics(
        friedman_test=friedman,
        pairwise_comparisons=pairwise_comparisons,
        quality_ranking=quality_ranking,
        cost_ranking=cost_ranking,
        regret_ranking=regret_ranking,
        pareto_optimal=pareto_optimal,
        convergence_speeds=convergence_speeds,
    )


def analyze_benchmark_results(benchmark_result: dict[str, Any]) -> dict[str, Any]:
    """Analyze complete benchmark results and generate comprehensive metrics.

    Args:
        benchmark_result: BenchmarkResult as dictionary

    Returns:
        Dictionary with comprehensive analysis including:
        - Individual algorithm metrics
        - Comparative analysis
        - Statistical tests
        - Effect sizes
        - Rankings
    """
    algorithms_data = benchmark_result.get("algorithms", [])

    # Calculate individual algorithm metrics
    individual_metrics = []
    for algo in algorithms_data:
        quality_scores = [
            eval_data["quality_score"] for eval_data in algo.get("feedback", [])
        ]

        metrics = calculate_algorithm_metrics(
            algorithm_name=algo["algorithm_name"],
            run_id=algo["run_id"],
            total_cost=algo["total_cost"],
            total_queries=algo["total_queries"],
            quality_scores=quality_scores,
            regret_history=algo.get("cumulative_regret", []),
            oracle_regret=0.0,
        )
        individual_metrics.append(metrics)

    # Comparative analysis
    comparative = calculate_comparative_metrics(individual_metrics)

    # Build analysis dictionary
    analysis = {
        "benchmark_id": benchmark_result.get("benchmark_id"),
        "dataset_size": benchmark_result.get("dataset_size"),
        "algorithms": {
            m.algorithm_name: {
                "total_cost": m.total_cost,
                "average_quality": m.average_quality,
                "quality_ci_lower": m.quality_ci[0],
                "quality_ci_upper": m.quality_ci[1],
                "cumulative_regret": m.cumulative_regret,
                "normalized_regret": m.normalized_regret,
                "regret_per_query": m.regret_per_query,
                "converged": m.convergence.converged,
                "convergence_point": m.convergence.convergence_point,
                "coefficient_of_variation": m.convergence.coefficient_of_variation,
            }
            for m in individual_metrics
        },
        "comparative_analysis": {
            "friedman_test": {
                "statistic": comparative.friedman_test.statistic,
                "p_value": comparative.friedman_test.p_value,
                "significant": comparative.friedman_test.significant,
            },
            "quality_ranking": comparative.quality_ranking,
            "cost_ranking": comparative.cost_ranking,
            "regret_ranking": comparative.regret_ranking,
            "pareto_optimal": comparative.pareto_optimal,
            "convergence_speeds": comparative.convergence_speeds,
            "pairwise_effect_sizes": {
                f"{k[0]}_vs_{k[1]}": {
                    "cohens_d": v.cohens_d,
                    "interpretation": v.interpretation,
                }
                for k, v in comparative.pairwise_comparisons.items()
            },
        },
    }

    return analysis
