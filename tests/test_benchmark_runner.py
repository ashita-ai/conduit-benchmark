"""Tests for BenchmarkRunner."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from conduit.engines.bandits import ModelArm, ThompsonSamplingBandit, UCB1Bandit
from conduit_bench.benchmark_models import BenchmarkQuery
from conduit_bench.runners import BenchmarkRunner, ModelExecutor, ModelExecutionResult


@pytest.fixture
def test_arms() -> list[ModelArm]:
    """Create test model arms."""
    return [
        ModelArm(
            model_id="openai:gpt-4o-mini",
            provider="openai",
            model_name="gpt-4o-mini",
            cost_per_input_token=0.00015,
            cost_per_output_token=0.0006,
            expected_quality=0.85,
        ),
        ModelArm(
            model_id="anthropic:claude-3-haiku",
            provider="anthropic",
            model_name="claude-3-haiku-20240307",
            cost_per_input_token=0.00025,
            cost_per_output_token=0.00125,
            expected_quality=0.80,
        ),
    ]


@pytest.fixture
def test_dataset() -> list[BenchmarkQuery]:
    """Create a small test dataset."""
    return [
        BenchmarkQuery(
            query_text="What is 2+2?",
            category="math",
            complexity=0.3,
            reference_answer="2+2 equals 4.",
        ),
        BenchmarkQuery(
            query_text="Explain recursion in Python",
            category="technical",
            complexity=0.7,
            reference_answer="Recursion is when a function calls itself.",
        ),
        BenchmarkQuery(
            query_text="Write a haiku about AI",
            category="creative",
            complexity=0.5,
            reference_answer="Silicon minds think / Patterns emerge from the void / Future awakens",
        ),
    ]


@pytest.mark.asyncio
async def test_benchmark_runner_initialization(test_arms: list[ModelArm]) -> None:
    """Test BenchmarkRunner initialization."""
    algorithms = [
        ThompsonSamplingBandit(arms=test_arms),
        UCB1Bandit(arms=test_arms, c=1.5),
    ]

    runner = BenchmarkRunner(algorithms=algorithms)

    assert len(runner.algorithms) == 2
    assert runner.arbiter_model == "gpt-4o-mini"
    assert runner.max_concurrency == 5


@pytest.mark.asyncio
async def test_benchmark_runner_mocked(
    test_arms: list[ModelArm], test_dataset: list[BenchmarkQuery]
) -> None:
    """Test benchmark runner with mocked LLM and Arbiter calls."""
    algorithms = [ThompsonSamplingBandit(arms=test_arms)]

    # Mock ModelExecutor
    mock_executor = MagicMock(spec=ModelExecutor)
    mock_result = ModelExecutionResult(
        model_id="openai:gpt-4o-mini",
        response_text="Mocked response",
        cost=0.0001,
        latency=1.0,
        input_tokens=10,
        output_tokens=5,
        success=True,
        was_fallback=False,
        primary_model=None,
        failed_models=[],
    )
    mock_executor.execute = AsyncMock(return_value=mock_result)
    mock_executor.execute_with_fallback = AsyncMock(return_value=mock_result)

    runner = BenchmarkRunner(algorithms=algorithms, executor=mock_executor)

    # Mock Arbiter evaluation
    with patch("conduit_bench.runners.benchmark_runner.evaluate") as mock_evaluate:
        mock_eval_result = MagicMock()
        mock_eval_result.overall_score = 0.9
        mock_evaluate.return_value = mock_eval_result

        result = await runner.run(dataset=test_dataset, show_progress=False)

    assert result.dataset_size == 3
    assert len(result.algorithms) == 1

    algo_run = result.algorithms[0]
    assert algo_run.algorithm_name == "thompson_sampling"
    assert algo_run.total_queries == 3
    assert algo_run.total_cost > 0
    assert algo_run.average_quality > 0
    assert len(algo_run.selections) == 3
    assert len(algo_run.feedback) == 3


@pytest.mark.skip(reason="Method _create_query_features no longer exists in BenchmarkRunner")
@pytest.mark.asyncio
async def test_create_query_features(test_arms: list[ModelArm]) -> None:
    """Test query features creation."""
    algorithms = [ThompsonSamplingBandit(arms=test_arms)]
    runner = BenchmarkRunner(algorithms=algorithms)

    query = BenchmarkQuery(
        query_text="Test query with multiple words",
        category="technical",
        complexity=0.5,
        reference_answer="Test answer",
    )

    features = runner._create_query_features(query)

    assert len(features.embedding) == 384
    assert features.complexity_score == 0.5
    assert features.domain == "technical"
    assert features.domain_confidence == 1.0
    assert features.token_count > 0


@pytest.mark.skip(reason="Method _evaluate_quality no longer exists in BenchmarkRunner")
@pytest.mark.asyncio
async def test_evaluate_quality_success(test_arms: list[ModelArm]) -> None:
    """Test quality evaluation with Arbiter."""
    algorithms = [ThompsonSamplingBandit(arms=test_arms)]
    runner = BenchmarkRunner(algorithms=algorithms)

    with patch("conduit_bench.runners.benchmark_runner.evaluate") as mock_evaluate:
        mock_result = MagicMock()
        mock_result.overall_score = 0.95
        mock_evaluate.return_value = mock_result

        score = await runner._evaluate_quality(
            output="Good response", reference="Reference answer"
        )

    assert score == 0.95
    mock_evaluate.assert_called_once()


@pytest.mark.skip(reason="Method _evaluate_quality no longer exists in BenchmarkRunner")
@pytest.mark.asyncio
async def test_evaluate_quality_failure(test_arms: list[ModelArm]) -> None:
    """Test quality evaluation when Arbiter fails."""
    algorithms = [ThompsonSamplingBandit(arms=test_arms)]
    runner = BenchmarkRunner(algorithms=algorithms)

    with patch("conduit_bench.runners.benchmark_runner.evaluate") as mock_evaluate:
        mock_evaluate.side_effect = Exception("Arbiter error")

        score = await runner._evaluate_quality(
            output="Response", reference="Reference"
        )

    assert score == 0.5  # Neutral score on failure


@pytest.mark.asyncio
async def test_multiple_algorithms(
    test_arms: list[ModelArm], test_dataset: list[BenchmarkQuery]
) -> None:
    """Test running multiple algorithms in parallel."""
    algorithms = [
        ThompsonSamplingBandit(arms=test_arms),
        UCB1Bandit(arms=test_arms, c=1.5),
    ]

    mock_executor = MagicMock(spec=ModelExecutor)
    mock_result = ModelExecutionResult(
        model_id="openai:gpt-4o-mini",
        response_text="Response",
        cost=0.0001,
        latency=1.0,
        input_tokens=10,
        output_tokens=5,
        success=True,
        was_fallback=False,
        primary_model=None,
        failed_models=[],
    )
    mock_executor.execute = AsyncMock(return_value=mock_result)
    mock_executor.execute_with_fallback = AsyncMock(return_value=mock_result)

    runner = BenchmarkRunner(algorithms=algorithms, executor=mock_executor)

    with patch("conduit_bench.runners.benchmark_runner.evaluate") as mock_evaluate:
        mock_eval_result = MagicMock()
        mock_eval_result.overall_score = 0.85
        mock_evaluate.return_value = mock_eval_result

        result = await runner.run(dataset=test_dataset, show_progress=False)

    assert len(result.algorithms) == 2
    assert result.get_algorithm("thompson_sampling") is not None
    assert result.get_algorithm("ucb1") is not None
    assert result.get_algorithm("NonExistent") is None


@pytest.mark.asyncio
async def test_benchmark_runner_parallel_execution(
    test_arms: list[ModelArm], test_dataset: list[BenchmarkQuery]
) -> None:
    """Test running algorithms in parallel mode."""
    algorithms = [
        ThompsonSamplingBandit(arms=test_arms),
        UCB1Bandit(arms=test_arms, c=1.5),
    ]

    mock_executor = MagicMock(spec=ModelExecutor)
    mock_result = ModelExecutionResult(
        model_id="openai:gpt-4o-mini",
        response_text="Response",
        cost=0.0001,
        latency=1.0,
        input_tokens=10,
        output_tokens=5,
        success=True,
        was_fallback=False,
        primary_model=None,
        failed_models=[],
    )
    mock_executor.execute = AsyncMock(return_value=mock_result)
    mock_executor.execute_with_fallback = AsyncMock(return_value=mock_result)

    runner = BenchmarkRunner(algorithms=algorithms, executor=mock_executor)

    with patch("conduit_bench.runners.benchmark_runner.evaluate") as mock_evaluate:
        mock_eval_result = MagicMock()
        mock_eval_result.overall_score = 0.85
        mock_evaluate.return_value = mock_eval_result

        result = await runner.run(dataset=test_dataset, show_progress=False, parallel=True)

    assert len(result.algorithms) == 2
    assert result.metadata.get("parallel") is True


@pytest.mark.asyncio
async def test_benchmark_runner_with_fallback(
    test_arms: list[ModelArm], test_dataset: list[BenchmarkQuery]
) -> None:
    """Test benchmark runner handles fallback results correctly."""
    algorithms = [ThompsonSamplingBandit(arms=test_arms)]

    mock_executor = MagicMock(spec=ModelExecutor)
    # Simulate a fallback scenario
    mock_result = ModelExecutionResult(
        model_id="anthropic:claude-3-haiku",
        response_text="Fallback response",
        cost=0.0001,
        latency=1.0,
        input_tokens=10,
        output_tokens=5,
        success=True,
        was_fallback=True,
        primary_model="openai:gpt-4o-mini",
        failed_models=["openai:gpt-4o-mini"],
    )
    mock_executor.execute = AsyncMock(return_value=mock_result)
    mock_executor.execute_with_fallback = AsyncMock(return_value=mock_result)

    runner = BenchmarkRunner(algorithms=algorithms, executor=mock_executor)

    with patch("conduit_bench.runners.benchmark_runner.evaluate") as mock_evaluate:
        mock_eval_result = MagicMock()
        mock_eval_result.overall_score = 0.9
        mock_evaluate.return_value = mock_eval_result

        result = await runner.run(dataset=test_dataset, show_progress=False)

    assert result.dataset_size == 3
    algo_run = result.algorithms[0]
    assert algo_run.total_queries == 3


@pytest.mark.asyncio
async def test_benchmark_runner_with_pluggable_evaluator(
    test_arms: list[ModelArm], test_dataset: list[BenchmarkQuery]
) -> None:
    """Test benchmark runner with custom evaluator instead of Arbiter."""
    from conduit_bench.evaluators.base import BaseEvaluator, EvaluationResult

    # Create a mock evaluator
    class MockEvaluator(BaseEvaluator):
        @property
        def name(self) -> str:
            return "mock_evaluator"

        def evaluate(self, response: str, ground_truth: str, query: str = "", **kwargs) -> EvaluationResult:
            is_correct = "correct" in response.lower()
            return EvaluationResult(
                score=1.0 if is_correct else 0.0,
                correct=is_correct,
                predicted=response,
                expected=ground_truth,
            )

    algorithms = [ThompsonSamplingBandit(arms=test_arms)]

    mock_executor = MagicMock(spec=ModelExecutor)
    mock_result = ModelExecutionResult(
        model_id="openai:gpt-4o-mini",
        response_text="This is the correct answer",
        cost=0.0001,
        latency=1.0,
        input_tokens=10,
        output_tokens=5,
        success=True,
        was_fallback=False,
        primary_model=None,
        failed_models=[],
    )
    mock_executor.execute = AsyncMock(return_value=mock_result)
    mock_executor.execute_with_fallback = AsyncMock(return_value=mock_result)

    runner = BenchmarkRunner(
        algorithms=algorithms,
        executor=mock_executor,
        evaluator=MockEvaluator(),
    )

    result = await runner.run(dataset=test_dataset, show_progress=False)

    algo_run = result.algorithms[0]
    assert algo_run.average_quality == 1.0  # MockEvaluator returns 1.0 for "correct"


@pytest.mark.asyncio
async def test_benchmark_runner_convergence_tracking(
    test_arms: list[ModelArm],
) -> None:
    """Test that benchmark runner tracks convergence with algorithm state history."""
    # Create a larger dataset for convergence testing
    dataset = [
        BenchmarkQuery(
            query_text=f"Query {i}",
            category="test",
            complexity=0.5,
            reference_answer=f"Answer {i}",
        )
        for i in range(10)
    ]

    algorithms = [ThompsonSamplingBandit(arms=test_arms)]

    mock_executor = MagicMock(spec=ModelExecutor)
    mock_result = ModelExecutionResult(
        model_id="openai:gpt-4o-mini",
        response_text="Response",
        cost=0.0001,
        latency=1.0,
        input_tokens=10,
        output_tokens=5,
        success=True,
        was_fallback=False,
        primary_model=None,
        failed_models=[],
    )
    mock_executor.execute = AsyncMock(return_value=mock_result)
    mock_executor.execute_with_fallback = AsyncMock(return_value=mock_result)

    runner = BenchmarkRunner(algorithms=algorithms, executor=mock_executor)

    with patch("conduit_bench.runners.benchmark_runner.evaluate") as mock_evaluate:
        mock_eval_result = MagicMock()
        mock_eval_result.overall_score = 0.85
        mock_evaluate.return_value = mock_eval_result

        result = await runner.run(dataset=dataset, show_progress=False)

    algo_run = result.algorithms[0]
    # Check that convergence is tracked
    assert hasattr(algo_run, "converged")
    assert hasattr(algo_run, "convergence_point")
    # Feedback should contain algorithm_state metadata
    assert len(algo_run.feedback) == 10
    for feedback in algo_run.feedback:
        assert "algorithm_state" in feedback.metadata


@pytest.mark.asyncio
async def test_benchmark_result_get_algorithm(
    test_arms: list[ModelArm], test_dataset: list[BenchmarkQuery]
) -> None:
    """Test BenchmarkResult.get_algorithm method."""
    algorithms = [
        ThompsonSamplingBandit(arms=test_arms),
        UCB1Bandit(arms=test_arms, c=1.5),
    ]

    mock_executor = MagicMock(spec=ModelExecutor)
    mock_result = ModelExecutionResult(
        model_id="openai:gpt-4o-mini",
        response_text="Response",
        cost=0.0001,
        latency=1.0,
        input_tokens=10,
        output_tokens=5,
        success=True,
        was_fallback=False,
        primary_model=None,
        failed_models=[],
    )
    mock_executor.execute = AsyncMock(return_value=mock_result)
    mock_executor.execute_with_fallback = AsyncMock(return_value=mock_result)

    runner = BenchmarkRunner(algorithms=algorithms, executor=mock_executor)

    with patch("conduit_bench.runners.benchmark_runner.evaluate") as mock_evaluate:
        mock_eval_result = MagicMock()
        mock_eval_result.overall_score = 0.85
        mock_evaluate.return_value = mock_eval_result

        result = await runner.run(dataset=test_dataset, show_progress=False)

    # Test exact match (get_algorithm uses exact string matching)
    assert result.get_algorithm("thompson_sampling") is not None
    assert result.get_algorithm("ucb1") is not None

    # Test non-existent algorithms
    assert result.get_algorithm("random") is None
    assert result.get_algorithm("") is None
    assert result.get_algorithm("Thompson_Sampling") is None  # Case sensitive
