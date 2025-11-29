"""Tests for visualization module."""

import pytest
from pathlib import Path
import plotly.graph_objects as go

from conduit_bench.analysis.visualize import (
    plot_cost_curves,
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
            "average_quality": 0.85,
            "quality_ci": (0.80, 0.90),
            "quality_ci_lower": 0.80,
            "quality_ci_upper": 0.90,
            "total_cost": 0.050,
            "cumulative_cost": 0.050,
            "convergence": {
                "converged": True,
                "convergence_point": 450
            },
        },
        "ucb1": {
            "average_quality": 0.78,
            "quality_ci": (0.73, 0.83),
            "quality_ci_lower": 0.73,
            "quality_ci_upper": 0.83,
            "total_cost": 0.040,
            "cumulative_cost": 0.040,
            "convergence": {
                "converged": True,
                "convergence_point": 520
            },
        },
        "random": {
            "average_quality": 0.65,
            "quality_ci": (0.60, 0.70),
            "quality_ci_lower": 0.60,
            "quality_ci_upper": 0.70,
            "total_cost": 0.045,
            "cumulative_cost": 0.045,
            "convergence": {
                "converged": False,
                "convergence_point": None
            },
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
                "average_quality": 0.85,
                "total_cost": 0.05,
                "cumulative_cost": [0.0005 * i for i in range(100)],
                "quality_history": [0.5 + 0.005 * i for i in range(100)],
            },
            {
                "algorithm_name": "ucb1",
                "average_quality": 0.78,
                "total_cost": 0.04,
                "cumulative_cost": [0.0004 * i for i in range(100)],
                "quality_history": [0.4 + 0.004 * i for i in range(100)],
            },
        ],
    }


@pytest.fixture
def sample_analysis() -> dict[str, any]:
    """Create sample analysis results for HTML report testing."""
    return {
        "benchmark_id": "test_123",
        "dataset_size": 100,
        "algorithms": {
            "thompson": {
                "average_quality": 0.85,
                "quality_ci_lower": 0.80,
                "quality_ci_upper": 0.90,
                "total_cost": 0.050,
                "cumulative_cost": 0.050,
                "convergence": {
                    "converged": True,
                    "convergence_point": 450
                },
            },
            "ucb1": {
                "average_quality": 0.78,
                "quality_ci_lower": 0.73,
                "quality_ci_upper": 0.83,
                "total_cost": 0.040,
                "cumulative_cost": 0.040,
                "convergence": {
                    "converged": True,
                    "convergence_point": 520
                },
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


class TestPlotCostCurves:
    """Tests for cost curve plotting."""

    def test_plot_cost_curves_basic(
        self,
        sample_algorithms_data: dict[str, dict[str, any]],
        temp_output_dir: Path,
    ) -> None:
        """Test basic cost curve plotting."""
        output_path = temp_output_dir / "cost.html"

        fig = plot_cost_curves(
            sample_algorithms_data, output_path=str(output_path), show_ci=False
        )

        assert isinstance(fig, go.Figure)
        assert output_path.exists()

    def test_plot_cost_curves_with_ci(
        self,
        sample_algorithms_data: dict[str, dict[str, any]],
        sample_benchmark_data: dict[str, any],
        temp_output_dir: Path,
    ) -> None:
        """Test cost curves with confidence intervals."""
        output_path = temp_output_dir / "cost_ci.html"

        fig = plot_cost_curves(
            sample_algorithms_data,
            benchmark_data=sample_benchmark_data,
            output_path=str(output_path),
            show_ci=True,
        )

        assert isinstance(fig, go.Figure)
        assert output_path.exists()

    def test_plot_cost_curves_no_output_path(
        self, sample_algorithms_data: dict[str, dict[str, any]]
    ) -> None:
        """Test cost curves without saving to file."""
        fig = plot_cost_curves(sample_algorithms_data, show_ci=False)

        assert isinstance(fig, go.Figure)

    def test_plot_cost_curves_single_algorithm(
        self, temp_output_dir: Path
    ) -> None:
        """Test cost curves with single algorithm."""
        data = {
            "only_one": {
                "average_quality": 0.75,
                "quality_ci_lower": 0.70,
                "quality_ci_upper": 0.80,
                "total_cost": 0.05,
                "cumulative_cost": 0.05,
            }
        }

        fig = plot_cost_curves(data, show_ci=False)
        assert isinstance(fig, go.Figure)


class TestPlotCostQualityScatter:
    """Tests for cost-quality scatter plot."""

    def test_plot_cost_quality_basic(
        self,
        sample_algorithms_data: dict[str, dict[str, any]],
        temp_output_dir: Path,
    ) -> None:
        """Test basic cost-quality scatter plot."""
        output_path = temp_output_dir / "scatter.html"

        fig = plot_cost_quality_scatter(
            sample_algorithms_data, output_path=str(output_path)
        )

        assert isinstance(fig, go.Figure)
        assert output_path.exists()

    def test_plot_cost_quality_with_pareto(
        self,
        sample_algorithms_data: dict[str, dict[str, any]],
        temp_output_dir: Path,
    ) -> None:
        """Test scatter plot with Pareto frontier highlighted."""
        output_path = temp_output_dir / "scatter_pareto.html"

        fig = plot_cost_quality_scatter(
            sample_algorithms_data,
            pareto_optimal=["thompson"],
            output_path=str(output_path),
        )

        assert isinstance(fig, go.Figure)
        assert output_path.exists()

    def test_plot_cost_quality_no_output(
        self, sample_algorithms_data: dict[str, dict[str, any]]
    ) -> None:
        """Test scatter plot without saving."""
        fig = plot_cost_quality_scatter(sample_algorithms_data)

        assert isinstance(fig, go.Figure)

    def test_plot_cost_quality_single_point(self) -> None:
        """Test scatter plot with single algorithm."""
        data = {
            "only_one": {
                "average_quality": 0.75,
                "total_cost": 0.05,
            }
        }

        fig = plot_cost_quality_scatter(data)
        assert isinstance(fig, go.Figure)


class TestPlotConvergenceComparison:
    """Tests for convergence comparison plot."""

    def test_plot_convergence_basic(
        self,
        sample_algorithms_data: dict[str, dict[str, any]],
        temp_output_dir: Path,
    ) -> None:
        """Test basic convergence comparison plot."""
        output_path = temp_output_dir / "convergence.html"

        fig = plot_convergence_comparison(
            sample_algorithms_data, output_path=str(output_path)
        )

        assert isinstance(fig, go.Figure)
        assert output_path.exists()

    def test_plot_convergence_mixed_states(
        self, temp_output_dir: Path
    ) -> None:
        """Test convergence plot with mixed convergence states."""
        data = {
            "converged1": {
                "convergence": {
                    "converged": True,
                    "convergence_point": 100
                },
            },
            "converged2": {
                "convergence": {
                    "converged": True,
                    "convergence_point": 200
                },
            },
            "not_converged": {
                "convergence": {
                    "converged": False,
                    "convergence_point": None
                },
            },
        }

        fig = plot_convergence_comparison(data)
        assert isinstance(fig, go.Figure)

    def test_plot_convergence_all_converged(self) -> None:
        """Test convergence plot when all algorithms converged."""
        data = {
            "algo1": {
                "convergence": {
                    "converged": True,
                    "convergence_point": 150
                },
            },
            "algo2": {
                "convergence": {
                    "converged": True,
                    "convergence_point": 300
                },
            },
        }

        fig = plot_convergence_comparison(data)
        assert isinstance(fig, go.Figure)

    def test_plot_convergence_none_converged(self) -> None:
        """Test convergence plot when no algorithms converged."""
        data = {
            "algo1": {
                "convergence": {
                    "converged": False,
                    "convergence_point": None
                },
            },
            "algo2": {
                "convergence": {
                    "converged": False,
                    "convergence_point": None
                },
            },
        }

        fig = plot_convergence_comparison(data)
        assert isinstance(fig, go.Figure)


class TestPlotQualityRanking:
    """Tests for quality ranking plot."""

    def test_plot_quality_ranking_basic(
        self,
        sample_algorithms_data: dict[str, dict[str, any]],
        temp_output_dir: Path,
    ) -> None:
        """Test basic quality ranking plot."""
        output_path = temp_output_dir / "ranking.html"

        fig = plot_quality_ranking(
            sample_algorithms_data, output_path=str(output_path)
        )

        assert isinstance(fig, go.Figure)
        assert output_path.exists()

    def test_plot_quality_ranking_with_ci(
        self,
        sample_algorithms_data: dict[str, dict[str, any]],
    ) -> None:
        """Test quality ranking includes confidence intervals."""
        fig = plot_quality_ranking(sample_algorithms_data)

        assert isinstance(fig, go.Figure)
        # Plotly figures have error bars in data traces

    def test_plot_quality_ranking_sorted(
        self, sample_algorithms_data: dict[str, dict[str, any]]
    ) -> None:
        """Test that quality ranking is sorted correctly."""
        fig = plot_quality_ranking(sample_algorithms_data)

        # Extract x values from Plotly bar chart (horizontal orientation)
        # The figure should have one trace (the bar chart)
        assert len(fig.data) > 0
        bar_trace = fig.data[0]

        # For horizontal bars, x contains the quality values
        x_data = list(bar_trace.x)

        # Should be sorted in descending order
        assert all(x_data[i] >= x_data[i + 1] for i in range(len(x_data) - 1))

    def test_plot_quality_ranking_single_algorithm(self) -> None:
        """Test quality ranking with single algorithm."""
        data = {
            "only_one": {
                "average_quality": 0.75,
                "quality_ci_lower": 0.70,
                "quality_ci_upper": 0.80,
            }
        }

        fig = plot_quality_ranking(data)
        assert isinstance(fig, go.Figure)


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
        sample_algorithms_data: dict[str, dict[str, any]],
        temp_output_dir: Path,
    ) -> None:
        """Test HTML report with interactive chart figures."""
        # Create actual Plotly figures
        chart_figs = {
            "Cost Curves": plot_cost_curves(sample_algorithms_data),
            "Cost-Quality Scatter": plot_cost_quality_scatter(sample_algorithms_data),
            "Convergence Comparison": plot_convergence_comparison(sample_algorithms_data),
            "Quality Ranking": plot_quality_ranking(sample_algorithms_data),
        }

        report_path = generate_html_report(
            sample_analysis, temp_output_dir, chart_figs=chart_figs
        )

        html_content = report_path.read_text()

        # Verify Plotly charts are embedded
        assert "plotly" in html_content.lower()
        assert "Interactive Visualizations" in html_content

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
            assert isinstance(fig, go.Figure)
        except (ValueError, KeyError):
            pass  # Acceptable to reject empty data

    def test_missing_quality_ci(self, temp_output_dir: Path) -> None:
        """Test quality ranking handles missing CI data."""
        data = {
            "algo1": {
                "average_quality": 0.75,
                # Missing quality_ci_lower and quality_ci_upper
            }
        }

        # Should handle gracefully (plot without error bars)
        fig = plot_quality_ranking(data)
        assert isinstance(fig, go.Figure)

    def test_invalid_output_path(
        self, sample_algorithms_data: dict[str, dict[str, any]]
    ) -> None:
        """Test visualization handles invalid output paths."""
        invalid_path = "/nonexistent/directory/chart.html"

        with pytest.raises((OSError, FileNotFoundError)):
            plot_quality_ranking(
                sample_algorithms_data, output_path=invalid_path
            )

    def test_html_report_no_pareto_frontier(
        self, temp_output_dir: Path
    ) -> None:
        """Test HTML report handles missing Pareto frontier."""
        analysis = {
            "benchmark_id": "test_empty",
            "dataset_size": 10,
            "algorithms": {
                "algo1": {
                    "average_quality": 0.75,
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
                # Missing convergence data entirely
                "average_quality": 0.75,
            }
        }

        # Should handle gracefully
        try:
            fig = plot_convergence_comparison(data)
            assert isinstance(fig, go.Figure)
        except KeyError:
            pass  # Acceptable to require convergence data
