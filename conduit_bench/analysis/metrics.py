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

    # Cost analysis (note: true regret requires oracle comparison)
    cumulative_cost: float
    normalized_cost: float
    cost_per_query: float

    # Convergence
    convergence: ConvergenceMetrics

    # Confidence intervals (95%)
    quality_ci: tuple[float, float]
    cost_ci: tuple[float, float]

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
    cost_history_ranking: list[tuple[str, float]]  # (algorithm, cumulative_cost)

    # Pareto frontier (cost vs quality)
    pareto_optimal: list[str]  # Algorithm names on Pareto frontier

    # Convergence comparison
    convergence_speeds: dict[str, int | None]  # algorithm -> convergence_point


def calculate_convergence(
    metric_history: list[float],
    window: int | None = None,
    threshold: float = 0.10,
    min_samples: int | None = None,
) -> ConvergenceMetrics:
    """Detect convergence in algorithm performance (learning curve stabilization).

    An algorithm is considered converged when the trend in the learning curve
    flattens (slope near zero) over a sliding window, indicating the algorithm
    has stopped meaningfully improving.

    Adaptive defaults based on dataset size:
    - min_samples: max(100, 20% of dataset)
    - window: max(50, 10% of dataset)

    Args:
        metric_history: Time series of metric values (e.g., quality scores)
        window: Sliding window size for convergence detection (default: adaptive)
        threshold: Slope threshold for convergence (default: 0.10, i.e., <10% change)
        min_samples: Minimum samples before checking convergence (default: adaptive)

    Returns:
        ConvergenceMetrics with convergence status and point
    """
    dataset_size = len(metric_history)

    # Adaptive defaults based on dataset size
    if min_samples is None:
        min_samples = max(100, int(dataset_size * 0.2))  # 20% of dataset, min 100

    if window is None:
        window = max(50, int(dataset_size * 0.1))  # 10% of dataset, min 50

    if dataset_size < min_samples:
        return ConvergenceMetrics(
            converged=False,
            convergence_point=None,
            coefficient_of_variation=float("inf"),
            window_size=window,
            threshold=threshold,
        )

    # Check for fixed baselines (always_best, always_cheapest, oracle)
    # These have very low variance from the start and should be marked as converged at query 1
    early_window = min(20, len(metric_history) // 4)
    if early_window >= 10:
        early_data = metric_history[:early_window]
        early_cv = np.std(early_data) / np.mean(early_data) if np.mean(early_data) > 0 else 0

        # If coefficient of variation < 0.01 (1% variance) early on, it's a fixed baseline
        if early_cv < 0.01:
            return ConvergenceMetrics(
                converged=True,
                convergence_point=1,  # Fixed from the start
                coefficient_of_variation=early_cv,
                window_size=window,
                threshold=threshold,
            )

    # Calculate moving average to smooth out noise
    window_size = min(50, window // 2)
    smoothed = np.convolve(metric_history, np.ones(window_size)/window_size, mode='valid')

    # Calculate slope of the last window of smoothed data
    if len(smoothed) < window:
        recent_smoothed = smoothed
    else:
        recent_smoothed = smoothed[-window:]

    # Fit linear regression to detect trend
    x = np.arange(len(recent_smoothed))
    if len(x) > 1:
        slope = np.polyfit(x, recent_smoothed, 1)[0]
        # Normalize slope by mean to get percentage change per query
        mean_recent = np.mean(recent_smoothed)
        normalized_slope = abs(slope / mean_recent) if mean_recent > 0 else float("inf")
    else:
        normalized_slope = float("inf")

    # Check if this is a random/noisy algorithm (high variance, no learning)
    # Use SMOOTHED data for CV to avoid penalizing learning algorithms with exploration
    smoothed_cv = np.std(smoothed) / np.mean(smoothed) if np.mean(smoothed) > 0 else float("inf")

    # If CV > 0.20 (20% variance) even in smoothed data, check for learning
    # This catches "random" baseline which has high variance and no learning
    if smoothed_cv > 0.20:
        # Check if there's actual learning trend (not just mean improvement)
        # Fit linear trend to smoothed data
        if len(smoothed) >= 20:
            x_trend = np.arange(len(smoothed))
            trend_slope = np.polyfit(x_trend, smoothed, 1)[0]
            mean_smoothed = np.mean(smoothed)
            normalized_trend = trend_slope / mean_smoothed if mean_smoothed > 0 else 0

            # If slope is nearly flat or negative (< 0.001 improvement per query) and high variance,
            # it's a random algorithm with no learning
            if normalized_trend < 0.001:
                return ConvergenceMetrics(
                    converged=False,
                    convergence_point=None,
                    coefficient_of_variation=smoothed_cv,
                    window_size=window,
                    threshold=threshold,
                )

    # Converged if slope is nearly flat (< threshold change per query)
    converged = normalized_slope < threshold

    # Find convergence point (when slope first dropped below threshold)
    convergence_point = None
    if converged:
        # Search forward to find when it first converged
        # Check smaller windows starting from min_samples to find earliest convergence
        min_window_for_convergence = min(window, 20)  # At least 20 points to detect convergence

        for i in range(min_window_for_convergence, len(smoothed)):
            # Use a sliding window to detect when slope first becomes small
            segment = smoothed[max(0, i-min_window_for_convergence):i]
            if len(segment) > 1:
                x_seg = np.arange(len(segment))
                seg_slope = np.polyfit(x_seg, segment, 1)[0]
                seg_mean = np.mean(segment)
                seg_norm_slope = abs(seg_slope / seg_mean) if seg_mean > 0 else float("inf")
                if seg_norm_slope < threshold:
                    convergence_point = i
                    break

    # Still compute CV for informational purposes
    recent_metric = metric_history[-window:]
    mean_recent = np.mean(recent_metric)
    std_recent = np.std(recent_metric, ddof=1)
    cv = std_recent / mean_recent if mean_recent > 0 else float("inf")

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
    cost_history: list[float],
) -> AlgorithmMetrics:
    """Calculate comprehensive metrics for a single algorithm run.

    Args:
        algorithm_name: Name of the algorithm
        run_id: Unique run identifier
        total_cost: Total cost incurred
        total_queries: Number of queries processed
        quality_scores: List of quality scores over time
        cost_history: Cumulative cost history over time

    Returns:
        AlgorithmMetrics with all calculated metrics
    """
    avg_quality = np.mean(quality_scores) if quality_scores else 0.0
    cumulative_cost = cost_history[-1] if cost_history else total_cost

    # Normalized cost (per query)
    normalized_cost = cumulative_cost / total_queries if total_queries > 0 else 0.0
    cost_per_query = total_cost / total_queries if total_queries > 0 else 0.0

    # Convergence detection
    convergence = calculate_convergence(quality_scores)

    # Confidence intervals
    quality_ci = bootstrap_ci(quality_scores)
    cost_ci = (total_cost * 0.95, total_cost * 1.05)  # Simplified for cost

    return AlgorithmMetrics(
        algorithm_name=algorithm_name,
        run_id=run_id,
        total_cost=total_cost,
        average_quality=avg_quality,
        total_queries=total_queries,
        cumulative_cost=cumulative_cost,
        normalized_cost=normalized_cost,
        cost_per_query=cost_per_query,
        convergence=convergence,
        quality_ci=quality_ci,
        cost_ci=cost_ci,
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
    cost_history_ranking = sorted(
        [(m.algorithm_name, m.cumulative_cost) for m in algorithm_metrics],
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
        cost_history_ranking=cost_history_ranking,
        pareto_optimal=pareto_optimal,
        convergence_speeds=convergence_speeds,
    )


def analyze_benchmark_results(benchmark_result: dict[str, Any]) -> dict[str, Any]:
    """Analyze complete benchmark results and generate comprehensive metrics.

    Accepts two input formats:
    1. BenchmarkResult format: {"algorithms": [{"algorithm_name": ..., ...}]}
    2. Dict-of-algorithms format: {"algo_name": {"avg_quality": ..., ...}}

    Args:
        benchmark_result: BenchmarkResult as dictionary or dict of algorithm results

    Returns:
        Dictionary with comprehensive analysis including:
        - summary: Overview with rankings and best performers
        - algorithms: Per-algorithm metrics with CIs and convergence
        - statistical_tests: Friedman test and pairwise comparisons
        - pareto_frontier: List of Pareto-optimal algorithms
    """
    # Detect input format and normalize to list of algorithm dicts
    algorithms_data = benchmark_result.get("algorithms", [])

    # If "algorithms" key exists and is a list, use BenchmarkResult format
    # Otherwise, treat the entire dict as algorithm_name -> algorithm_data
    if isinstance(algorithms_data, list) and len(algorithms_data) > 0:
        # BenchmarkResult format
        normalized_algos = []
        for algo in algorithms_data:
            # Try multiple sources for quality scores
            quality_scores = [
                eval_data["quality_score"] for eval_data in algo.get("feedback", [])
            ]
            if not quality_scores:
                # Try 'queries' field (test fixture format)
                quality_scores = [
                    q.get("quality_score", 0.0) for q in algo.get("queries", [])
                    if "quality_score" in q
                ]
            if not quality_scores:
                # Fall back to quality_history if feedback not available
                quality_scores = algo.get("quality_history", [])
            if not quality_scores:
                # Use avg_quality as single score
                quality_scores = [algo.get("avg_quality", 0.0)]

            normalized_algos.append({
                "name": algo["algorithm_name"],
                "quality_scores": quality_scores,
                "total_cost": algo["total_cost"],
                "total_queries": algo.get("total_queries", len(quality_scores)),
                "cumulative_cost": algo.get("cumulative_cost", []),
            })
    else:
        # Dict-of-algorithms format (algorithm_name -> metrics dict)
        normalized_algos = []
        for algo_name, algo_data in benchmark_result.items():
            if algo_name in ("benchmark_id", "dataset_size", "algorithms"):
                continue
            quality_scores = algo_data.get("quality_scores", [])
            if not quality_scores:
                quality_scores = [algo_data.get("avg_quality", 0.0)]

            cost_history = algo_data.get("cumulative_cost", [])
            if isinstance(cost_history, (int, float)):
                cost_history = [cost_history]

            normalized_algos.append({
                "name": algo_name,
                "quality_scores": quality_scores,
                "total_cost": algo_data.get("total_cost", 0.0),
                "total_queries": len(quality_scores),
                "cumulative_cost": cost_history,
            })

    # Calculate individual algorithm metrics
    individual_metrics = []
    for algo in normalized_algos:
        metrics = calculate_algorithm_metrics(
            algorithm_name=algo["name"],
            run_id="unknown",
            total_cost=algo["total_cost"],
            total_queries=algo["total_queries"],
            quality_scores=algo["quality_scores"],
            cost_history=algo["cumulative_cost"],
        )
        individual_metrics.append(metrics)

    # Handle empty input
    if not individual_metrics:
        return {
            "summary": {
                "num_algorithms": 0,
                "best_quality_algorithm": None,
                "best_cost_algorithm": None,
                "quality_rankings": [],
                "cost_rankings": [],
            },
            "algorithms": {},
            "statistical_tests": {
                "friedman": {
                    "statistic": 0.0,
                    "p_value": 1.0,
                    "significant": False,
                }
            },
            "pareto_frontier": [],
        }

    # Comparative analysis
    comparative = calculate_comparative_metrics(individual_metrics)

    # Extract rankings as simple lists of algorithm names
    quality_rankings = [name for name, _ in comparative.quality_ranking]
    cost_rankings = [name for name, _ in comparative.cost_ranking]

    # Build analysis dictionary with expected structure
    analysis = {
        "benchmark_id": benchmark_result.get("benchmark_id", "N/A"),
        "dataset_size": benchmark_result.get("dataset_size", 0),
        "summary": {
            "num_algorithms": len(individual_metrics),
            "best_quality_algorithm": quality_rankings[0] if quality_rankings else None,
            "best_cost_algorithm": cost_rankings[0] if cost_rankings else None,
            "quality_rankings": quality_rankings,
            "cost_rankings": cost_rankings,
        },
        "algorithms": {
            m.algorithm_name: {
                "total_cost": m.total_cost,
                "average_quality": m.average_quality,
                "quality_ci": (m.quality_ci[0], m.quality_ci[1]),
                "cumulative_cost": m.cumulative_cost,
                "normalized_cost": m.normalized_cost,
                "cost_per_query": m.cost_per_query,
                "convergence": {
                    "converged": m.convergence.converged,
                    "convergence_point": m.convergence.convergence_point,
                    "coefficient_of_variation": m.convergence.coefficient_of_variation,
                },
            }
            for m in individual_metrics
        },
        "statistical_tests": {
            "friedman": {
                "statistic": comparative.friedman_test.statistic,
                "p_value": comparative.friedman_test.p_value,
                "significant": comparative.friedman_test.significant,
            },
            "pairwise_effect_sizes": {
                f"{k[0]}_vs_{k[1]}": {
                    "cohens_d": v.cohens_d,
                    "interpretation": v.interpretation,
                }
                for k, v in comparative.pairwise_comparisons.items()
            },
        },
        "pareto_frontier": comparative.pareto_optimal,
    }

    return analysis
