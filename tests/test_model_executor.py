"""Tests for ModelExecutor."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from conduit.engines.bandits import ModelArm
from conduit_bench.runners import ModelExecutor


@pytest.fixture
def test_arm() -> ModelArm:
    """Create a test model arm."""
    return ModelArm(
        model_id="openai:gpt-4o-mini",
        provider="openai",
        model_name="gpt-4o-mini",
        cost_per_input_token=0.00015,
        cost_per_output_token=0.0006,
        expected_quality=0.85,
    )


@pytest.fixture
def executor() -> ModelExecutor:
    """Create a ModelExecutor instance."""
    return ModelExecutor(timeout=30.0, max_retries=1)


@pytest.mark.asyncio
async def test_execute_success(test_arm: ModelArm, executor: ModelExecutor) -> None:
    """Test successful model execution."""
    # Mock PydanticAI Agent
    mock_result = MagicMock()
    mock_result.data = "2+2 equals 4."
    mock_cost = MagicMock()
    mock_cost.request_tokens = 10
    mock_cost.response_tokens = 5
    mock_result.cost.return_value = mock_cost

    with patch("conduit_bench.runners.model_executor.Agent") as MockAgent:
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        MockAgent.return_value = mock_agent

        result = await executor.execute(
            arm=test_arm,
            query_text="What is 2+2?",
            system_prompt="You are a helpful assistant.",
        )

    assert result.success
    assert result.model_id == "openai:gpt-4o-mini"
    assert result.response_text == "2+2 equals 4."
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.cost > 0
    assert result.latency > 0
    assert result.error is None


@pytest.mark.asyncio
async def test_execute_with_retry(test_arm: ModelArm, executor: ModelExecutor) -> None:
    """Test execution with retry on failure."""
    # Mock PydanticAI Agent to fail once, then succeed
    mock_result = MagicMock()
    mock_result.data = "Success after retry"
    mock_cost = MagicMock()
    mock_cost.request_tokens = 10
    mock_cost.response_tokens = 5
    mock_result.cost.return_value = mock_cost

    with patch("conduit_bench.runners.model_executor.Agent") as MockAgent:
        mock_agent = MagicMock()
        # First call fails, second succeeds
        mock_agent.run = AsyncMock(
            side_effect=[Exception("Temporary error"), mock_result]
        )
        MockAgent.return_value = mock_agent

        result = await executor.execute(
            arm=test_arm,
            query_text="Test query",
        )

    assert result.success
    assert result.response_text == "Success after retry"
    assert mock_agent.run.call_count == 2  # Initial + 1 retry


@pytest.mark.asyncio
async def test_execute_failure_after_retries(
    test_arm: ModelArm, executor: ModelExecutor
) -> None:
    """Test execution failure after all retries exhausted."""
    with patch("conduit_bench.runners.model_executor.Agent") as MockAgent:
        mock_agent = MagicMock()
        # Always fail
        mock_agent.run = AsyncMock(side_effect=Exception("Persistent error"))
        MockAgent.return_value = mock_agent

        result = await executor.execute(
            arm=test_arm,
            query_text="Test query",
        )

    assert not result.success
    assert result.response_text == ""
    assert result.cost == 0.0
    assert result.error is not None
    assert "Persistent error" in result.error
    assert mock_agent.run.call_count == 2  # Initial + 1 retry


@pytest.mark.asyncio
async def test_execute_batch(executor: ModelExecutor) -> None:
    """Test batch execution of multiple models."""
    arms = [
        ModelArm(
            model_id="openai:gpt-4o-mini",
            provider="openai",
            model_name="gpt-4o-mini",
            cost_per_input_token=0.00015,
            cost_per_output_token=0.0006,
        ),
        ModelArm(
            model_id="anthropic:claude-3-haiku",
            provider="anthropic",
            model_name="claude-3-haiku-20240307",
            cost_per_input_token=0.00025,
            cost_per_output_token=0.00125,
        ),
    ]

    # Mock PydanticAI Agent
    mock_result = MagicMock()
    mock_result.data = "Test response"
    mock_cost = MagicMock()
    mock_cost.request_tokens = 10
    mock_cost.response_tokens = 5
    mock_result.cost.return_value = mock_cost

    with patch("conduit_bench.runners.model_executor.Agent") as MockAgent:
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        MockAgent.return_value = mock_agent

        results = await executor.execute_batch(
            arms=arms,
            query_text="Test query",
            max_concurrency=2,
        )

    assert len(results) == 2
    assert "openai:gpt-4o-mini" in results
    assert "anthropic:claude-3-haiku" in results
    assert all(r.success for r in results.values())


@pytest.mark.asyncio
async def test_cost_calculation(test_arm: ModelArm, executor: ModelExecutor) -> None:
    """Test cost calculation with different token counts."""
    mock_result = MagicMock()
    mock_result.data = "Response"
    mock_cost = MagicMock()
    mock_cost.request_tokens = 1000  # 1K input tokens
    mock_cost.response_tokens = 500  # 500 output tokens
    mock_result.cost.return_value = mock_cost

    with patch("conduit_bench.runners.model_executor.Agent") as MockAgent:
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        MockAgent.return_value = mock_agent

        result = await executor.execute(
            arm=test_arm,
            query_text="Test",
        )

    # Cost = (1000 * 0.00015 / 1000) + (500 * 0.0006 / 1000)
    # Cost = 0.00015 + 0.0003 = 0.00045
    expected_cost = 0.00045
    assert abs(result.cost - expected_cost) < 1e-6
