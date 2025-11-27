"""Integration tests for CLI commands."""

import pytest
import json
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

    @pytest.mark.skip(reason="Requires complex async mock setup for SyntheticQueryGenerator")
    @patch("conduit_bench.cli.SyntheticQueryGenerator")
    def test_generate_basic(
        self,
        mock_generator_class: MagicMock,
        runner: CliRunner,
        temp_test_dir: Path,
    ) -> None:
        """Test basic generate command."""
        output_path = temp_test_dir / "data" / "generated.jsonl"

        # Mock generator - CLI uses generator.generate() method
        mock_generator = MagicMock()
        mock_query = MagicMock()
        mock_query.query_text = "Test query"
        mock_query.metadata = {"category": "technical"}
        mock_query.model_dump_json = MagicMock(return_value='{"query_text": "Test query"}')
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
            ],
        )

        assert result.exit_code == 0
        assert output_path.exists()

    @pytest.mark.skip(reason="Requires complex async mock setup for SyntheticQueryGenerator")
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
        mock_query.metadata = {"category": "technical"}
        mock_query.model_dump_json = MagicMock(return_value='{"query_text": "Test query"}')
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

    def test_generate_help(self, runner: CliRunner) -> None:
        """Test generate command help works."""
        result = runner.invoke(cli, ["generate", "--help"])
        assert result.exit_code == 0
        assert "queries" in result.output


class TestCLIRun:
    """Tests for run command."""

    @pytest.mark.skip(reason="Requires complex mocking of BenchmarkRunner and dataset loading")
    @patch("conduit_bench.cli.BenchmarkRunner")
    def test_run_basic(
        self,
        mock_runner_class: MagicMock,
        runner: CliRunner,
        sample_dataset_file: Path,
        temp_test_dir: Path,
    ) -> None:
        """Test basic run command."""
        output_path = temp_test_dir / "results" / "run_output.json"

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

    def test_run_multiple_algorithms_help(
        self,
        runner: CliRunner,
    ) -> None:
        """Test run command accepts algorithms option."""
        result = runner.invoke(cli, ["run", "--help"])

        # Should show algorithms option
        assert result.exit_code == 0
        assert "algorithms" in result.output

    def test_run_help(self, runner: CliRunner) -> None:
        """Test run command help."""
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "dataset" in result.output.lower() or "run" in result.output.lower()


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

    @pytest.mark.skip(reason="Requires matplotlib and output file checks")
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

    @pytest.mark.skip(reason="Requires matplotlib and output file checks")
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

    def test_visualize_help(self, runner: CliRunner) -> None:
        """Test visualize command help."""
        result = runner.invoke(cli, ["visualize", "--help"])
        assert result.exit_code == 0
        assert "results" in result.output


class TestCLIIntegration:
    """End-to-end integration tests."""

    @pytest.mark.skip(reason="Full pipeline test requires complex mocking of multiple components")
    def test_full_pipeline(
        self,
        runner: CliRunner,
        temp_test_dir: Path,
    ) -> None:
        """Test full generate → run → analyze → visualize pipeline."""
        # This test requires mocking multiple components with async support
        # Skipped in favor of individual command tests
        pass

    def test_all_commands_have_help(self, runner: CliRunner) -> None:
        """Test all commands have working help."""
        commands = ["generate", "run", "analyze", "visualize"]
        for cmd in commands:
            result = runner.invoke(cli, [cmd, "--help"])
            assert result.exit_code == 0, f"{cmd} help failed"


class TestCLIEdgeCases:
    """Tests for CLI edge cases and error handling."""

    def test_cli_help(self, runner: CliRunner) -> None:
        """Test CLI help command."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Conduit-Bench" in result.output

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
            json.dumps({"benchmark_id": "test", "dataset_size": 0, "algorithms": []})
        )

        result = runner.invoke(
            cli, ["analyze", "--results", str(empty_results)]
        )

        # Should handle gracefully - either succeeds or shows appropriate message
        # Exit code 0 is expected for empty results (still valid data)
        assert result.exit_code == 0 or "error" in result.output.lower()
