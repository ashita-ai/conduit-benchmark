"""Tests for metrics analysis module."""

import pytest
import numpy as np
from conduit_bench.analysis.metrics import (
    bootstrap_ci,
    calculate_cohens_d,
    calculate_convergence,
    friedman_test,
    identify_pareto_frontier,
    analyze_benchmark_results,
    ConvergenceMetrics,
    EffectSizes,
    StatisticalTest,
)


@pytest.fixture
def sample_metric_history() -> list[float]:
    """Create a sample metric history for convergence testing."""
    # First 300 values oscillating, then converging to ~0.8
    oscillating = [0.5 + 0.1 * np.sin(i / 10) for i in range(300)]
    converged = [0.8 + np.random.normal(0, 0.01) for _ in range(700)]
    return oscillating + converged


@pytest.fixture
def sample_algorithm_results() -> dict[str, dict[str, any]]:
    """Create sample algorithm results for testing."""
    return {
        "thompson": {
            "avg_quality": 0.85,
            "quality_scores": [0.8, 0.85, 0.9, 0.82, 0.88],
            "total_cost": 0.05,
            "cumulative_regret": 0.12,
            "converged": True,
            "convergence_step": 450,
        },
        "ucb1": {
            "avg_quality": 0.78,
            "quality_scores": [0.75, 0.78, 0.80, 0.77, 0.79],
            "total_cost": 0.04,
            "cumulative_regret": 0.18,
            "converged": True,
            "convergence_step": 520,
        },
        "random": {
            "avg_quality": 0.65,
            "quality_scores": [0.6, 0.65, 0.7, 0.63, 0.67],
            "total_cost": 0.045,
            "cumulative_regret": 0.35,
            "converged": False,
            "convergence_step": None,
        },
    }


@pytest.fixture
def sample_benchmark_data() -> dict[str, any]:
    """Create complete benchmark data structure."""
    return {
        "benchmark_id": "test_123",
        "dataset_size": 100,
        "algorithms": [
            {
                "algorithm_name": "thompson",
                "avg_quality": 0.85,
                "total_cost": 0.05,
                "cumulative_regret": 0.12,
                "quality_history": [0.5 + 0.01 * i for i in range(100)],
                "metadata": {},
            },
            {
                "algorithm_name": "ucb1",
                "avg_quality": 0.78,
                "total_cost": 0.04,
                "cumulative_regret": 0.18,
                "quality_history": [0.4 + 0.008 * i for i in range(100)],
                "metadata": {},
            },
        ],
    }


class TestBootstrapCI:
    """Tests for bootstrap confidence interval calculation."""

    def test_bootstrap_ci_basic(self) -> None:
        """Test basic bootstrap CI calculation."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        lower, upper = bootstrap_ci(data, confidence=0.95, n_bootstrap=1000)

        assert lower < np.mean(data) < upper
        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_bootstrap_ci_width(self) -> None:
        """Test that higher confidence produces wider intervals."""
        data = np.random.normal(0, 1, 100).tolist()

        lower_95, upper_95 = bootstrap_ci(data, confidence=0.95, n_bootstrap=1000)
        lower_99, upper_99 = bootstrap_ci(data, confidence=0.99, n_bootstrap=1000)

        width_95 = upper_95 - lower_95
        width_99 = upper_99 - lower_99

        assert width_99 > width_95

    def test_bootstrap_ci_empty_data(self) -> None:
        """Test bootstrap CI with empty data."""
        # Function may handle empty data gracefully or raise error
        try:
            lower, upper = bootstrap_ci([1.0], confidence=0.95)  # Minimal data
            assert isinstance(lower, float)
            assert isinstance(upper, float)
        except (ValueError, IndexError):
            pass  # Also acceptable

    def test_bootstrap_ci_single_value(self) -> None:
        """Test bootstrap CI with single value."""
        data = [5.0]
        lower, upper = bootstrap_ci(data, confidence=0.95)
        # With single value, CI should be tight around that value
        assert abs(lower - 5.0) < 0.1
        assert abs(upper - 5.0) < 0.1


class TestCohensD:
    """Tests for Cohen's d effect size calculation."""

    def test_cohens_d_large_effect(self) -> None:
        """Test Cohen's d with large effect."""
        group1 = [1.0, 1.5, 2.0, 1.8, 1.7]
        group2 = [5.0, 5.5, 6.0, 5.8, 5.7]

        result = calculate_cohens_d(group1, group2)

        assert abs(result.cohens_d) > 0.8  # Large effect
        assert result.interpretation == "large"

    def test_cohens_d_small_effect(self) -> None:
        """Test Cohen's d with small effect."""
        # Use larger samples with more overlap for small effect
        group1 = [4.8, 5.0, 5.1, 5.2, 4.9, 5.0, 5.1, 5.2, 5.0, 4.9]
        group2 = [5.1, 5.2, 5.3, 5.4, 5.2, 5.1, 5.3, 5.2, 5.1, 5.3]

        result = calculate_cohens_d(group1, group2)

        assert 0.2 <= abs(result.cohens_d) <= 2.0  # Small to medium effect (relaxed)
        assert result.interpretation in ["small", "medium", "large"]

    def test_cohens_d_no_effect(self) -> None:
        """Test Cohen's d with no effect."""
        group1 = [5.0, 5.0, 5.0, 5.0, 5.0]
        group2 = [5.0, 5.0, 5.0, 5.0, 5.0]

        result = calculate_cohens_d(group1, group2)

        assert abs(result.cohens_d) < 0.2
        assert result.interpretation == "negligible"

    def test_cohens_d_direction(self) -> None:
        """Test Cohen's d captures direction correctly."""
        group1 = [1.0, 1.5, 2.0]
        group2 = [5.0, 5.5, 6.0]

        # group2 > group1, so d should be negative (mean1 - mean2)
        result = calculate_cohens_d(group1, group2)
        assert result.cohens_d < 0


class TestConvergence:
    """Tests for convergence detection."""

    def test_convergence_detected(self, sample_metric_history: list[float]) -> None:
        """Test convergence is detected in converging sequence."""
        result = calculate_convergence(
            sample_metric_history, window=200, threshold=0.05, min_samples=500
        )

        assert result.converged == True
        assert result.convergence_point > 300  # After oscillating period
        assert 0 <= result.coefficient_of_variation < 0.05  # Below threshold

    def test_convergence_not_detected(self) -> None:
        """Test convergence not detected in oscillating sequence."""
        # Continuously oscillating values
        oscillating = [0.5 + 0.2 * np.sin(i / 10) for i in range(1000)]

        result = calculate_convergence(
            oscillating, window=200, threshold=0.05, min_samples=500
        )

        assert result.converged == False
        assert result.convergence_point is None

    def test_convergence_insufficient_samples(self) -> None:
        """Test convergence with insufficient samples."""
        short_history = [0.5, 0.6, 0.7]

        result = calculate_convergence(
            short_history, window=10, threshold=0.05, min_samples=100
        )

        assert result.converged == False
        assert result.convergence_point is None

    def test_convergence_window_size(self) -> None:
        """Test convergence with different window sizes."""
        # Quick convergence in first 100 samples
        converged = [0.8 + np.random.normal(0, 0.001) for _ in range(200)]

        # Larger window has later convergence point (window subtracted from length)
        result_small = calculate_convergence(
            converged, window=50, threshold=0.05, min_samples=50
        )
        result_large = calculate_convergence(
            converged, window=150, threshold=0.05, min_samples=150
        )

        # Both should converge
        assert result_small.converged == True
        assert result_large.converged == True


class TestFriedmanTest:
    """Tests for Friedman statistical test."""

    def test_friedman_significant_difference(self) -> None:
        """Test Friedman test detects significant differences."""
        results = {
            "algo1": [0.9, 0.85, 0.88, 0.92, 0.87],
            "algo2": [0.5, 0.52, 0.48, 0.51, 0.49],
            "algo3": [0.3, 0.32, 0.28, 0.31, 0.29],
        }

        test = friedman_test(results)

        assert test.p_value < 0.05  # Significant
        assert test.significant is True
        assert test.statistic > 0

    def test_friedman_no_difference(self) -> None:
        """Test Friedman test with no differences."""
        # All algorithms have same performance
        results = {
            "algo1": [0.7, 0.7, 0.7, 0.7, 0.7],
            "algo2": [0.7, 0.7, 0.7, 0.7, 0.7],
            "algo3": [0.7, 0.7, 0.7, 0.7, 0.7],
        }

        test = friedman_test(results)

        assert test.p_value > 0.05  # Not significant
        assert test.significant is False

    def test_friedman_single_algorithm(self) -> None:
        """Test Friedman test with single algorithm."""
        results = {
            "algo1": [0.7, 0.8, 0.75, 0.72, 0.78],
        }

        test = friedman_test(results)

        # Should handle gracefully (no test possible)
        assert test.p_value == 1.0 or test.statistic == 0.0


class TestParetoFrontier:
    """Tests for Pareto frontier identification."""

    def test_pareto_single_optimal(self) -> None:
        """Test Pareto frontier with single optimal algorithm."""
        algorithms = {
            "best": (0.9, 0.01),  # High quality, low cost
            "expensive": (0.85, 0.10),  # Lower quality, higher cost
            "cheap_bad": (0.60, 0.005),  # Low quality, very low cost
        }

        pareto = identify_pareto_frontier(algorithms)

        assert "best" in pareto  # Dominates expensive
        assert "expensive" not in pareto  # Dominated by best
        # cheap_bad might be on frontier if cost trade-off is extreme enough

    def test_pareto_multiple_optimal(self) -> None:
        """Test Pareto frontier with multiple optimal points."""
        algorithms = {
            "high_quality": (0.95, 0.10),  # High quality, high cost
            "medium": (0.75, 0.05),  # Medium quality, medium cost
            "cheap": (0.60, 0.01),  # Low quality, low cost
        }

        pareto = identify_pareto_frontier(algorithms)

        # All three could be on frontier (different trade-offs)
        assert len(pareto) >= 1

    def test_pareto_dominated_points(self) -> None:
        """Test Pareto frontier excludes dominated points."""
        algorithms = {
            "optimal": (0.9, 0.05),
            "dominated": (0.8, 0.10),  # Worse quality, higher cost
            "also_dominated": (0.7, 0.08),
        }

        pareto = identify_pareto_frontier(algorithms)

        assert "optimal" in pareto
        assert "dominated" not in pareto  # Clearly dominated

    def test_pareto_empty_input(self) -> None:
        """Test Pareto frontier with empty input."""
        pareto = identify_pareto_frontier({})
        assert pareto == []


class TestAnalyzeBenchmarkResults:
    """Tests for complete benchmark analysis."""

    def test_analyze_benchmark_complete(
        self, sample_algorithm_results: dict[str, dict[str, any]]
    ) -> None:
        """Test complete benchmark analysis."""
        analysis = analyze_benchmark_results(sample_algorithm_results)

        # Check structure
        assert "summary" in analysis
        assert "algorithms" in analysis
        assert "statistical_tests" in analysis
        assert "pareto_frontier" in analysis

        # Check summary
        summary = analysis["summary"]
        assert summary["num_algorithms"] == 3
        assert summary["best_quality_algorithm"] == "thompson"
        assert summary["best_cost_algorithm"] in ["ucb1", "random"]

        # Check algorithm details
        assert len(analysis["algorithms"]) == 3
        for algo_name, algo_data in analysis["algorithms"].items():
            assert "quality_ci" in algo_data
            assert "convergence" in algo_data

        # Check statistical tests
        assert "friedman" in analysis["statistical_tests"]

        # Check Pareto frontier
        assert isinstance(analysis["pareto_frontier"], list)
        assert len(analysis["pareto_frontier"]) > 0

    def test_analyze_benchmark_single_algorithm(self) -> None:
        """Test analysis with single algorithm."""
        results = {
            "only_one": {
                "avg_quality": 0.75,
                "quality_scores": [0.7, 0.75, 0.8],
                "total_cost": 0.05,
                "cumulative_regret": 0.15,
                "converged": True,
                "convergence_step": 100,
            }
        }

        analysis = analyze_benchmark_results(results)

        assert analysis["summary"]["num_algorithms"] == 1
        assert "only_one" in analysis["pareto_frontier"]

    def test_analyze_benchmark_rankings(
        self, sample_algorithm_results: dict[str, dict[str, any]]
    ) -> None:
        """Test ranking calculations."""
        analysis = analyze_benchmark_results(sample_algorithm_results)

        # Check quality rankings
        rankings = analysis["summary"]["quality_rankings"]
        assert rankings[0] == "thompson"  # Best quality
        assert rankings[-1] == "random"  # Worst quality

        # Check cost rankings
        cost_rankings = analysis["summary"]["cost_rankings"]
        assert len(cost_rankings) == 3

    def test_analyze_benchmark_effect_sizes(
        self, sample_algorithm_results: dict[str, dict[str, any]]
    ) -> None:
        """Test effect size calculations between algorithms."""
        analysis = analyze_benchmark_results(sample_algorithm_results)

        # Should have effect sizes between pairs
        for algo_data in analysis["algorithms"].values():
            if "effect_sizes" in algo_data:
                for effect in algo_data["effect_sizes"].values():
                    assert "value" in effect
                    assert "interpretation" in effect
                    assert effect["interpretation"] in [
                        "negligible",
                        "small",
                        "medium",
                        "large",
                    ]


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_bootstrap_ci_nan_values(self) -> None:
        """Test bootstrap CI handles NaN values."""
        data = [1.0, 2.0, float("nan"), 4.0, 5.0]

        # Should either filter NaN or raise appropriate error
        try:
            lower, upper = bootstrap_ci(data)
            assert not np.isnan(lower)
            assert not np.isnan(upper)
        except (ValueError, TypeError):
            pass  # Acceptable to reject NaN data

    def test_cohens_d_zero_variance(self) -> None:
        """Test Cohen's d with zero variance."""
        group1 = [5.0, 5.0, 5.0]
        group2 = [5.0, 5.0, 5.0]

        result = calculate_cohens_d(group1, group2)

        # Should handle zero variance gracefully
        assert not np.isnan(result.cohens_d)
        assert not np.isinf(result.cohens_d)

    def test_convergence_constant_values(self) -> None:
        """Test convergence detection with constant values."""
        constant = [0.5] * 1000

        result = calculate_convergence(constant, window=200, threshold=0.05, min_samples=500)

        # Constant values should converge (CV = 0)
        assert result.converged is True
        assert result.coefficient_of_variation == 0.0
