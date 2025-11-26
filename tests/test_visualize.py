"""Tests for visualization module."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # Non-interactive backend for testing

from conduit_bench.analysis.visualize import (
    plot_regret_curves,
    plot_cost_quality_scatter,
    plot_convergence_comparison,
    plot_quality_ranking,
    generate_html_report,
)


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create temporary directory for test outputs."""
    output_dir = tmp_path / "test_charts"
    output_dir.mkdir()
    yield output_dir
    # Cleanup handled by tmp_path fixture


@pytest.fixture
def sample_algorithms_data() -> dict[str, dict[str, any]]:
    """Create sample algorithm data for visualization testing."""
    return {
        "thompson": {
            "avg_quality": 0.85,
            "quality_ci": (0.80, 0.90),
            "total_cost": 0.050,
            "cumulative_regret": 0.12,
            "converged": True,
            "convergence_step": 450,
            "convergence": {"converged": True, "convergence_step": 450},
        },
        "ucb1": {
            "avg_quality": 0.78,
            "quality_ci": (0.73, 0.83),
            "total_cost": 0.040,
            "cumulative_regret": 0.18,
            "converged": True,
            "convergence_step": 520,
            "convergence": {"converged": True, "convergence_step": 520},
        },
        "random": {
            "avg_quality": 0.65,
            "quality_ci": (0.60, 0.70),
            "total_cost": 0.045,
            "cumulative_regret": 0.35,
            "converged": False,
            "convergence_step": None,
            "convergence": {"converged": False, "convergence_step": None},
        },
    }


@pytest.fixture
def sample_benchmark_data() -> dict[str, any]:
    """Create complete benchmark data with time series."""
    return {
        "benchmark_id": "test_123",
        "dataset_size": 100,
        "algorithms": [
            {
                "algorithm_name": "thompson",
                "avg_quality": 0.85,
                "total_cost": 0.05,
                "cumulative_regret": [0.01 * i for i in range(100)],
                "quality_history": [0.5 + 0.005 * i for i in range(100)],
            },
            {
                "algorithm_name": "ucb1",
                "avg_quality": 0.78,
                "total_cost": 0.04,
                "cumulative_regret": [0.015 * i for i in range(100)],
                "quality_history": [0.4 + 0.004 * i for i in range(100)],
            },
        ],
    }


@pytest.fixture
def sample_analysis() -> dict[str, any]:
    """Create sample analysis results for HTML report testing."""
    return {
        "summary": {
            "num_algorithms": 2,
            "best_quality_algorithm": "thompson",
            "best_cost_algorithm": "ucb1",
            "quality_rankings": ["thompson", "ucb1"],
            "cost_rankings": ["ucb1", "thompson"],
        },
        "algorithms": {
            "thompson": {
                "avg_quality": 0.85,
                "quality_ci": (0.80, 0.90),
                "total_cost": 0.050,
                "cumulative_regret": 0.12,
                "converged": True,
                "convergence_step": 450,
            },
            "ucb1": {
                "avg_quality": 0.78,
                "quality_ci": (0.73, 0.83),
                "total_cost": 0.040,
                "cumulative_regret": 0.18,
                "converged": True,
                "convergence_step": 520,
            },
        },
        "statistical_tests": {
            "friedman": {
                "statistic": 5.2,
                "p_value": 0.02,
                "significant": True,
            }
        },
        "pareto_frontier": ["thompson", "ucb1"],
    }


class TestPlotRegretCurves:
    """Tests for regret curve plotting."""

    def test_plot_regret_curves_basic(
        self,
        sample_algorithms_data: dict[str, dict[str, any]],
        temp_output_dir: Path,
    ) -> None:
        """Test basic regret curve plotting."""
        output_path = temp_output_dir / "regret.png"

        fig = plot_regret_curves(
            sample_algorithms_data, output_path=str(output_path), show_ci=False
        )

        assert fig is not None
        assert output_path.exists()
        plt.close(fig)

    def test_plot_regret_curves_with_ci(
        self,
        sample_algorithms_data: dict[str, dict[str, any]],
        sample_benchmark_data: dict[str, any],
        temp_output_dir: Path,
    ) -> None:
        """Test regret curves with confidence intervals."""
        output_path = temp_output_dir / "regret_ci.png"

        fig = plot_regret_curves(
            sample_algorithms_data,
            benchmark_data=sample_benchmark_data,
            output_path=str(output_path),
            show_ci=True,
        )

        assert fig is not None
        assert output_path.exists()
        plt.close(fig)

    def test_plot_regret_curves_no_output_path(
        self, sample_algorithms_data: dict[str, dict[str, any]]
    ) -> None:
        """Test regret curves without saving to file."""
        fig = plot_regret_curves(sample_algorithms_data, show_ci=False)

        assert fig is not None
        plt.close(fig)

    def test_plot_regret_curves_single_algorithm(
        self, temp_output_dir: Path
    ) -> None:
        """Test regret curves with single algorithm."""
        data = {
            "only_one": {
                "avg_quality": 0.75,
                "quality_ci": (0.70, 0.80),
                "total_cost": 0.05,
                "cumulative_regret": 0.15,
            }
        }

        fig = plot_regret_curves(data, show_ci=False)
        assert fig is not None
        plt.close(fig)


class TestPlotCostQualityScatter:
    """Tests for cost-quality scatter plot."""

    def test_plot_cost_quality_basic(
        self,
        sample_algorithms_data: dict[str, dict[str, any]],
        temp_output_dir: Path,
    ) -> None:
        """Test basic cost-quality scatter plot."""
        output_path = temp_output_dir / "scatter.png"

        fig = plot_cost_quality_scatter(
            sample_algorithms_data, output_path=str(output_path)
        )

        assert fig is not None
        assert output_path.exists()
        plt.close(fig)

    def test_plot_cost_quality_with_pareto(
        self,
        sample_algorithms_data: dict[str, dict[str, any]],
        temp_output_dir: Path,
    ) -> None:
        """Test scatter plot with Pareto frontier highlighted."""
        output_path = temp_output_dir / "scatter_pareto.png"

        fig = plot_cost_quality_scatter(
            sample_algorithms_data,
            pareto_optimal=["thompson"],
            output_path=str(output_path),
        )

        assert fig is not None
        assert output_path.exists()
        plt.close(fig)

    def test_plot_cost_quality_no_output(
        self, sample_algorithms_data: dict[str, dict[str, any]]
    ) -> None:
        """Test scatter plot without saving."""
        fig = plot_cost_quality_scatter(sample_algorithms_data)

        assert fig is not None
        plt.close(fig)

    def test_plot_cost_quality_single_point(self) -> None:
        """Test scatter plot with single algorithm."""
        data = {
            "only_one": {
                "avg_quality": 0.75,
                "total_cost": 0.05,
            }
        }

        fig = plot_cost_quality_scatter(data)
        assert fig is not None
        plt.close(fig)


class TestPlotConvergenceComparison:
    """Tests for convergence comparison plot."""

    def test_plot_convergence_basic(
        self,
        sample_algorithms_data: dict[str, dict[str, any]],
        temp_output_dir: Path,
    ) -> None:
        """Test basic convergence comparison plot."""
        output_path = temp_output_dir / "convergence.png"

        fig = plot_convergence_comparison(
            sample_algorithms_data, output_path=str(output_path)
        )

        assert fig is not None
        assert output_path.exists()
        plt.close(fig)

    def test_plot_convergence_mixed_states(
        self, temp_output_dir: Path
    ) -> None:
        """Test convergence plot with mixed convergence states."""
        data = {
            "converged1": {
                "converged": True,
                "convergence_step": 100,
                "convergence": {"converged": True, "convergence_step": 100},
            },
            "converged2": {
                "converged": True,
                "convergence_step": 200,
                "convergence": {"converged": True, "convergence_step": 200},
            },
            "not_converged": {
                "converged": False,
                "convergence_step": None,
                "convergence": {"converged": False, "convergence_step": None},
            },
        }

        fig = plot_convergence_comparison(data)
        assert fig is not None
        plt.close(fig)

    def test_plot_convergence_all_converged(self) -> None:
        """Test convergence plot when all algorithms converged."""
        data = {
            "algo1": {
                "converged": True,
                "convergence_step": 150,
                "convergence": {"converged": True, "convergence_step": 150},
            },
            "algo2": {
                "converged": True,
                "convergence_step": 300,
                "convergence": {"converged": True, "convergence_step": 300},
            },
        }

        fig = plot_convergence_comparison(data)
        assert fig is not None
        plt.close(fig)

    def test_plot_convergence_none_converged(self) -> None:
        """Test convergence plot when no algorithms converged."""
        data = {
            "algo1": {
                "converged": False,
                "convergence_step": None,
                "convergence": {"converged": False, "convergence_step": None},
            },
            "algo2": {
                "converged": False,
                "convergence_step": None,
                "convergence": {"converged": False, "convergence_step": None},
            },
        }

        fig = plot_convergence_comparison(data)
        assert fig is not None
        plt.close(fig)


class TestPlotQualityRanking:
    """Tests for quality ranking plot."""

    def test_plot_quality_ranking_basic(
        self,
        sample_algorithms_data: dict[str, dict[str, any]],
        temp_output_dir: Path,
    ) -> None:
        """Test basic quality ranking plot."""
        output_path = temp_output_dir / "ranking.png"

        fig = plot_quality_ranking(
            sample_algorithms_data, output_path=str(output_path)
        )

        assert fig is not None
        assert output_path.exists()
        plt.close(fig)

    def test_plot_quality_ranking_with_ci(
        self,
        sample_algorithms_data: dict[str, dict[str, any]],
    ) -> None:
        """Test quality ranking includes confidence intervals."""
        fig = plot_quality_ranking(sample_algorithms_data)

        assert fig is not None
        # Visual inspection would verify error bars are present
        plt.close(fig)

    def test_plot_quality_ranking_sorted(
        self, sample_algorithms_data: dict[str, dict[str, any]]
    ) -> None:
        """Test that quality ranking is sorted correctly."""
        fig = plot_quality_ranking(sample_algorithms_data)

        # Extract data from figure to verify sorting
        ax = fig.axes[0]
        y_data = [bar.get_height() for bar in ax.patches]

        # Should be sorted in descending order
        assert all(y_data[i] >= y_data[i + 1] for i in range(len(y_data) - 1))

        plt.close(fig)

    def test_plot_quality_ranking_single_algorithm(self) -> None:
        """Test quality ranking with single algorithm."""
        data = {
            "only_one": {
                "avg_quality": 0.75,
                "quality_ci": (0.70, 0.80),
            }
        }

        fig = plot_quality_ranking(data)
        assert fig is not None
        plt.close(fig)


class TestGenerateHTMLReport:
    """Tests for HTML report generation."""

    def test_generate_html_report_basic(
        self, sample_analysis: dict[str, any], temp_output_dir: Path
    ) -> None:
        """Test basic HTML report generation."""
        report_path = generate_html_report(sample_analysis, temp_output_dir)

        assert report_path.exists()
        assert report_path.name == "benchmark_report.html"

        # Verify HTML content
        html_content = report_path.read_text()
        assert "<!DOCTYPE html>" in html_content
        assert "Conduit Benchmark Report" in html_content
        assert "thompson" in html_content
        assert "ucb1" in html_content

    def test_generate_html_report_with_charts(
        self,
        sample_analysis: dict[str, any],
        temp_output_dir: Path,
    ) -> None:
        """Test HTML report with chart references."""
        chart_paths = {
            "regret_curves": "regret_curves.png",
            "cost_quality_scatter": "cost_quality.png",
            "convergence_comparison": "convergence.png",
            "quality_ranking": "quality.png",
        }

        report_path = generate_html_report(
            sample_analysis, temp_output_dir, chart_paths=chart_paths
        )

        html_content = report_path.read_text()

        # Verify chart references in HTML
        assert "regret_curves.png" in html_content
        assert "cost_quality.png" in html_content
        assert "convergence.png" in html_content
        assert "quality.png" in html_content

    def test_generate_html_report_summary_section(
        self, sample_analysis: dict[str, any], temp_output_dir: Path
    ) -> None:
        """Test HTML report includes summary section."""
        report_path = generate_html_report(sample_analysis, temp_output_dir)
        html_content = report_path.read_text()

        assert "Executive Summary" in html_content
        assert "Dataset Size" in html_content
        assert "Algorithms Tested" in html_content

    def test_generate_html_report_pareto_section(
        self, sample_analysis: dict[str, any], temp_output_dir: Path
    ) -> None:
        """Test HTML report includes Pareto optimal section."""
        report_path = generate_html_report(sample_analysis, temp_output_dir)
        html_content = report_path.read_text()

        assert "Pareto Optimal" in html_content
        # Should list Pareto optimal algorithms
        for algo in sample_analysis["pareto_frontier"]:
            assert algo in html_content

    def test_generate_html_report_statistical_section(
        self, sample_analysis: dict[str, any], temp_output_dir: Path
    ) -> None:
        """Test HTML report includes statistical analysis."""
        report_path = generate_html_report(sample_analysis, temp_output_dir)
        html_content = report_path.read_text()

        assert "Statistical Analysis" in html_content
        assert "Friedman Test" in html_content

    def test_generate_html_report_algorithm_table(
        self, sample_analysis: dict[str, any], temp_output_dir: Path
    ) -> None:
        """Test HTML report includes algorithm performance table."""
        report_path = generate_html_report(sample_analysis, temp_output_dir)
        html_content = report_path.read_text()

        assert "Algorithm Performance" in html_content
        assert "<table>" in html_content
        assert "Avg Quality" in html_content
        assert "Total Cost" in html_content
        assert "Converged" in html_content


class TestVisualizationEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_algorithms_data(self) -> None:
        """Test visualization functions handle empty data."""
        empty_data = {}

        # Should not crash, but might return None or empty figure
        try:
            fig = plot_quality_ranking(empty_data)
            if fig is not None:
                plt.close(fig)
        except (ValueError, KeyError):
            pass  # Acceptable to reject empty data

    def test_missing_quality_ci(self, temp_output_dir: Path) -> None:
        """Test quality ranking handles missing CI data."""
        data = {
            "algo1": {
                "avg_quality": 0.75,
                # Missing quality_ci
            }
        }

        # Should handle gracefully (plot without error bars)
        fig = plot_quality_ranking(data)
        assert fig is not None
        plt.close(fig)

    def test_invalid_output_path(
        self, sample_algorithms_data: dict[str, dict[str, any]]
    ) -> None:
        """Test visualization handles invalid output paths."""
        invalid_path = "/nonexistent/directory/chart.png"

        with pytest.raises((OSError, FileNotFoundError)):
            plot_quality_ranking(
                sample_algorithms_data, output_path=invalid_path
            )

    def test_html_report_no_pareto_frontier(
        self, temp_output_dir: Path
    ) -> None:
        """Test HTML report handles missing Pareto frontier."""
        analysis = {
            "summary": {
                "num_algorithms": 1,
                "best_quality_algorithm": "algo1",
                "quality_rankings": ["algo1"],
            },
            "algorithms": {
                "algo1": {
                    "avg_quality": 0.75,
                    "total_cost": 0.05,
                }
            },
            "statistical_tests": {},
            "pareto_frontier": [],  # Empty
        }

        report_path = generate_html_report(analysis, temp_output_dir)
        assert report_path.exists()

    def test_convergence_plot_missing_data(self) -> None:
        """Test convergence plot handles missing convergence data."""
        data = {
            "algo1": {
                # Missing convergence data
                "avg_quality": 0.75,
            }
        }

        # Should handle gracefully
        try:
            fig = plot_convergence_comparison(data)
            if fig is not None:
                plt.close(fig)
        except KeyError:
            pass  # Acceptable to require convergence data
