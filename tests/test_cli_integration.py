"""Integration tests for CLI commands."""

import pytest
import json
import tempfile
from pathlib import Path
from click.testing import CliRunner
from unittest.mock import patch, MagicMock, AsyncMock

from conduit_bench.cli import main as cli


@pytest.fixture
def runner() -> CliRunner:
    """Create CLI test runner."""
    return CliRunner()


@pytest.fixture
def temp_test_dir(tmp_path: Path) -> Path:
    """Create temporary directory for test files."""
    test_dir = tmp_path / "cli_test"
    test_dir.mkdir()
    (test_dir / "data").mkdir()
    (test_dir / "results").mkdir()
    (test_dir / "analysis").mkdir()
    return test_dir


@pytest.fixture
def sample_dataset_file(temp_test_dir: Path) -> Path:
    """Create sample dataset JSONL file."""
    dataset_path = temp_test_dir / "data" / "test_queries.jsonl"

    queries = [
        {
            "query_text": "What is Python?",
            "category": "technical",
            "complexity": 0.3,
        },
        {
            "query_text": "Explain quantum computing",
            "category": "technical",
            "complexity": 0.8,
        },
        {
            "query_text": "Write a haiku about coding",
            "category": "creative",
            "complexity": 0.5,
        },
    ]

    with open(dataset_path, "w") as f:
        for query in queries:
            f.write(json.dumps(query) + "\n")

    return dataset_path


@pytest.fixture
def sample_results_file(temp_test_dir: Path) -> Path:
    """Create sample benchmark results file."""
    results_path = temp_test_dir / "results" / "test_results.json"

    results = {
        "benchmark_id": "test_benchmark_123",
        "dataset_size": 3,
        "algorithms": [
            {
                "algorithm_name": "thompson",
                "avg_quality": 0.85,
                "total_cost": 0.05,
                "cumulative_cost": [0.01, 0.03, 0.05],
                "quality_history": [0.7, 0.8, 0.85],
                "cost_history": [0.01, 0.02, 0.02],
                "queries": [
                    {
                        "query_id": 0,
                        "query_text": "What is Python?",
                        "model_used": "gpt-4o-mini",
                        "quality_score": 0.7,
                        "cost": 0.01,
                    },
                    {
                        "query_id": 1,
                        "query_text": "Explain quantum computing",
                        "model_used": "gpt-4o",
                        "quality_score": 0.8,
                        "cost": 0.02,
                    },
                    {
                        "query_id": 2,
                        "query_text": "Write a haiku about coding",
                        "model_used": "gpt-4o",
                        "quality_score": 0.85,
                        "cost": 0.02,
                    },
                ],
            },
            {
                "algorithm_name": "ucb1",
                "avg_quality": 0.78,
                "total_cost": 0.04,
                "cumulative_cost": [0.01, 0.025, 0.04],
                "quality_history": [0.65, 0.75, 0.78],
                "cost_history": [0.01, 0.015, 0.015],
                "queries": [
                    {
                        "query_id": 0,
                        "query_text": "What is Python?",
                        "model_used": "gpt-4o-mini",
                        "quality_score": 0.65,
                        "cost": 0.01,
                    },
                    {
                        "query_id": 1,
                        "query_text": "Explain quantum computing",
                        "model_used": "gpt-4o-mini",
                        "quality_score": 0.75,
                        "cost": 0.015,
                    },
                    {
                        "query_id": 2,
                        "query_text": "Write a haiku about coding",
                        "model_used": "gpt-4o-mini",
                        "quality_score": 0.78,
                        "cost": 0.015,
                    },
                ],
            },
        ],
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results_path


class TestCLIGenerate:
    """Tests for generate command."""

    @patch("conduit_bench.cli.SyntheticQueryGenerator")
    def test_generate_basic(
        self,
        mock_generator_class: MagicMock,
        runner: CliRunner,
        temp_test_dir: Path,
    ) -> None:
        """Test basic generate command."""
        output_path = temp_test_dir / "data" / "generated.jsonl"

        # Mock generator
        mock_generator = MagicMock()
        mock_query = MagicMock()
        mock_query.query_text = "Test query"
        mock_query.category = "technical"
        mock_query.complexity = 0.5
        mock_query.reference_answer = None
        mock_generator.generate_simple = AsyncMock(return_value=[mock_query])
        mock_generator_class.return_value = mock_generator

        result = runner.invoke(
            cli,
            [
                "generate",
                "--queries",
                "1",
                "--seed",
                "42",
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        assert output_path.exists()

    @patch("conduit_bench.cli.SyntheticQueryGenerator")
    def test_generate_with_reference(
        self,
        mock_generator_class: MagicMock,
        runner: CliRunner,
        temp_test_dir: Path,
    ) -> None:
        """Test generate command with reference answers."""
        output_path = temp_test_dir / "data" / "generated_ref.jsonl"

        mock_generator = MagicMock()
        mock_query = MagicMock()
        mock_query.query_text = "Test query"
        mock_query.category = "technical"
        mock_query.complexity = 0.5
        mock_query.reference_answer = "Reference answer"
        mock_generator.generate = AsyncMock(return_value=[mock_query])
        mock_generator_class.return_value = mock_generator

        result = runner.invoke(
            cli,
            [
                "generate",
                "--queries",
                "1",
                "--seed",
                "42",
                "--output",
                str(output_path),
                "--reference-probability",
                "1.0",
            ],
        )

        assert result.exit_code == 0

    def test_generate_invalid_queries(self, runner: CliRunner) -> None:
        """Test generate with invalid number of queries."""
        result = runner.invoke(
            cli, ["generate", "--queries", "-1", "--output", "out.jsonl"]
        )

        assert result.exit_code != 0


class TestCLIRun:
    """Tests for run command."""

    @patch("conduit_bench.cli.BenchmarkRunner")
    @patch("conduit_bench.cli.PostgreSQLDatabase")
    def test_run_basic(
        self,
        mock_db_class: MagicMock,
        mock_runner_class: MagicMock,
        runner: CliRunner,
        sample_dataset_file: Path,
        temp_test_dir: Path,
    ) -> None:
        """Test basic run command."""
        output_path = temp_test_dir / "results" / "run_output.json"

        # Mock database
        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        # Mock runner
        mock_runner = MagicMock()
        mock_result = {
            "benchmark_id": "test_123",
            "dataset_size": 3,
            "algorithms": [],
        }
        mock_runner.run_benchmark = AsyncMock(return_value=mock_result)
        mock_runner_class.return_value = mock_runner

        result = runner.invoke(
            cli,
            [
                "run",
                "--dataset",
                str(sample_dataset_file),
                "--algorithms",
                "thompson",
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0

    @patch("conduit_bench.cli.BenchmarkRunner")
    def test_run_multiple_algorithms(
        self,
        mock_runner_class: MagicMock,
        runner: CliRunner,
        sample_dataset_file: Path,
        temp_test_dir: Path,
    ) -> None:
        """Test run with multiple algorithms."""
        output_path = temp_test_dir / "results" / "multi_algo.json"

        mock_runner = MagicMock()
        mock_runner.run_benchmark = AsyncMock(
            return_value={"algorithms": []}
        )
        mock_runner_class.return_value = mock_runner

        result = runner.invoke(
            cli,
            [
                "run",
                "--dataset",
                str(sample_dataset_file),
                "--algorithms",
                "thompson,ucb1,random",
                "--output",
                str(output_path),
            ],
        )

        # Should parse algorithms correctly
        assert "thompson" in result.output or result.exit_code == 0

    def test_run_missing_dataset(self, runner: CliRunner) -> None:
        """Test run with missing dataset file."""
        result = runner.invoke(
            cli,
            [
                "run",
                "--dataset",
                "/nonexistent/file.jsonl",
                "--algorithms",
                "thompson",
            ],
        )

        assert result.exit_code != 0


class TestCLIAnalyze:
    """Tests for analyze command."""

    def test_analyze_basic(
        self,
        runner: CliRunner,
        sample_results_file: Path,
        temp_test_dir: Path,
    ) -> None:
        """Test basic analyze command."""
        output_path = temp_test_dir / "analysis" / "metrics.json"

        result = runner.invoke(
            cli,
            [
                "analyze",
                "--results",
                str(sample_results_file),
                "--output",
                str(output_path),
            ],
        )

        assert result.exit_code == 0
        assert output_path.exists()

        # Verify analysis output structure
        with open(output_path) as f:
            analysis = json.load(f)
            assert "summary" in analysis
            assert "algorithms" in analysis
            assert "statistical_tests" in analysis

    def test_analyze_displays_summary(
        self, runner: CliRunner, sample_results_file: Path
    ) -> None:
        """Test analyze displays summary to console."""
        result = runner.invoke(
            cli, ["analyze", "--results", str(sample_results_file)]
        )

        assert result.exit_code == 0
        # Should show quality rankings
        assert "thompson" in result.output or "ucb1" in result.output

    def test_analyze_missing_results(self, runner: CliRunner) -> None:
        """Test analyze with missing results file."""
        result = runner.invoke(
            cli, ["analyze", "--results", "/nonexistent/results.json"]
        )

        assert result.exit_code != 0

    def test_analyze_invalid_json(
        self, runner: CliRunner, temp_test_dir: Path
    ) -> None:
        """Test analyze with invalid JSON file."""
        invalid_file = temp_test_dir / "invalid.json"
        invalid_file.write_text("not valid json {")

        result = runner.invoke(
            cli, ["analyze", "--results", str(invalid_file)]
        )

        assert result.exit_code != 0


class TestCLIVisualize:
    """Tests for visualize command."""

    def test_visualize_basic(
        self,
        runner: CliRunner,
        sample_results_file: Path,
        temp_test_dir: Path,
    ) -> None:
        """Test basic visualize command."""
        output_dir = temp_test_dir / "charts"
        output_dir.mkdir()

        result = runner.invoke(
            cli,
            [
                "visualize",
                "--results",
                str(sample_results_file),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0

        # Check that charts were created
        expected_files = [
            "regret_curves.png",
            "cost_quality_scatter.png",
            "convergence_comparison.png",
            "quality_ranking.png",
            "benchmark_report.html",
        ]

        for filename in expected_files:
            assert (output_dir / filename).exists()

    def test_visualize_html_only(
        self,
        runner: CliRunner,
        sample_results_file: Path,
        temp_test_dir: Path,
    ) -> None:
        """Test visualize creates HTML report."""
        output_dir = temp_test_dir / "reports"
        output_dir.mkdir()

        result = runner.invoke(
            cli,
            [
                "visualize",
                "--results",
                str(sample_results_file),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0

        report_path = output_dir / "benchmark_report.html"
        assert report_path.exists()

        # Verify HTML content
        html_content = report_path.read_text()
        assert "Conduit Benchmark Report" in html_content

    def test_visualize_missing_results(self, runner: CliRunner) -> None:
        """Test visualize with missing results file."""
        result = runner.invoke(
            cli, ["visualize", "--results", "/nonexistent/results.json"]
        )

        assert result.exit_code != 0


class TestCLIIntegration:
    """End-to-end integration tests."""

    @patch("conduit_bench.cli.SyntheticQueryGenerator")
    @patch("conduit_bench.cli.BenchmarkRunner")
    @patch("conduit_bench.cli.PostgreSQLDatabase")
    def test_full_pipeline(
        self,
        mock_db_class: MagicMock,
        mock_runner_class: MagicMock,
        mock_generator_class: MagicMock,
        runner: CliRunner,
        temp_test_dir: Path,
    ) -> None:
        """Test full generate → run → analyze → visualize pipeline."""
        # Setup mocks
        mock_query = MagicMock()
        mock_query.query_text = "Test query"
        mock_query.category = "technical"
        mock_query.complexity = 0.5
        mock_query.reference_answer = None

        mock_generator = MagicMock()
        mock_generator.generate_simple = AsyncMock(
            return_value=[mock_query, mock_query]
        )
        mock_generator_class.return_value = mock_generator

        mock_db = MagicMock()
        mock_db_class.return_value = mock_db

        mock_runner = MagicMock()
        mock_result = {
            "benchmark_id": "integration_test",
            "dataset_size": 2,
            "algorithms": [
                {
                    "algorithm_name": "thompson",
                    "avg_quality": 0.85,
                    "total_cost": 0.05,
                    "cumulative_cost": [0.02, 0.05],
                    "quality_history": [0.8, 0.85],
                    "queries": [],
                }
            ],
        }
        mock_runner.run_benchmark = AsyncMock(return_value=mock_result)
        mock_runner_class.return_value = mock_runner

        # Step 1: Generate dataset
        dataset_path = temp_test_dir / "data" / "integration.jsonl"
        result = runner.invoke(
            cli,
            [
                "generate",
                "--queries",
                "2",
                "--seed",
                "42",
                "--output",
                str(dataset_path),
            ],
        )
        assert result.exit_code == 0

        # Step 2: Run benchmark
        results_path = temp_test_dir / "results" / "integration.json"
        result = runner.invoke(
            cli,
            [
                "run",
                "--dataset",
                str(dataset_path),
                "--algorithms",
                "thompson",
                "--output",
                str(results_path),
            ],
        )
        assert result.exit_code == 0

        # Step 3: Analyze results
        analysis_path = temp_test_dir / "analysis" / "integration.json"
        result = runner.invoke(
            cli,
            [
                "analyze",
                "--results",
                str(results_path),
                "--output",
                str(analysis_path),
            ],
        )
        assert result.exit_code == 0

        # Step 4: Visualize
        charts_dir = temp_test_dir / "charts"
        charts_dir.mkdir()
        result = runner.invoke(
            cli,
            [
                "visualize",
                "--results",
                str(results_path),
                "--output-dir",
                str(charts_dir),
            ],
        )
        assert result.exit_code == 0


class TestCLIEdgeCases:
    """Tests for CLI edge cases and error handling."""

    def test_cli_help(self, runner: CliRunner) -> None:
        """Test CLI help command."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Conduit Benchmark CLI" in result.output

    def test_generate_help(self, runner: CliRunner) -> None:
        """Test generate command help."""
        result = runner.invoke(cli, ["generate", "--help"])
        assert result.exit_code == 0
        assert "queries" in result.output

    def test_invalid_command(self, runner: CliRunner) -> None:
        """Test invalid CLI command."""
        result = runner.invoke(cli, ["invalid_command"])
        assert result.exit_code != 0

    def test_analyze_empty_algorithms(
        self, runner: CliRunner, temp_test_dir: Path
    ) -> None:
        """Test analyze with empty algorithms list."""
        empty_results = temp_test_dir / "empty.json"
        empty_results.write_text(
            json.dumps({"benchmark_id": "test", "algorithms": []})
        )

        result = runner.invoke(
            cli, ["analyze", "--results", str(empty_results)]
        )

        # Should handle gracefully
        assert result.exit_code == 0 or "No algorithms" in result.output
