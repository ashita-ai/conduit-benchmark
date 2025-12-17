"""Tests for ModelExecutor."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from conduit.engines.bandits import ModelArm
from conduit_bench.runners import ModelExecutor, ModelExecutionResult


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
def fallback_arm() -> ModelArm:
    """Create a fallback model arm."""
    return ModelArm(
        model_id="anthropic:claude-3-haiku-20240307",
        provider="anthropic",
        model_name="claude-3-haiku-20240307",
        cost_per_input_token=0.00025,
        cost_per_output_token=0.00125,
        expected_quality=0.80,
    )


@pytest.fixture
def executor() -> ModelExecutor:
    """Create a ModelExecutor instance."""
    return ModelExecutor(timeout=30.0, max_retries=1)


def _create_mock_agent_result(response: str, input_tokens: int = 10, output_tokens: int = 5):
    """Helper to create a properly mocked PydanticAI agent result."""
    mock_result = MagicMock()
    mock_result.data = response
    mock_result.output = response

    # Mock usage() method
    mock_usage = MagicMock()
    mock_usage.input_tokens = input_tokens
    mock_usage.output_tokens = output_tokens
    mock_usage.request_tokens = input_tokens  # Fallback
    mock_usage.response_tokens = output_tokens  # Fallback
    mock_result.usage = MagicMock(return_value=mock_usage)

    return mock_result


@pytest.mark.asyncio
async def test_execute_success(test_arm: ModelArm, executor: ModelExecutor) -> None:
    """Test successful model execution."""
    mock_result = _create_mock_agent_result("2+2 equals 4.", input_tokens=10, output_tokens=5)

    with patch("conduit_bench.runners.model_executor.Agent") as MockAgent, \
         patch("conduit_bench.runners.model_executor.compute_cost", return_value=0.00045):
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
    assert result.cost == 0.00045
    assert result.latency > 0
    assert result.error is None


@pytest.mark.asyncio
async def test_execute_with_retry(test_arm: ModelArm, executor: ModelExecutor) -> None:
    """Test execution with retry on failure."""
    mock_result = _create_mock_agent_result("Success after retry", input_tokens=10, output_tokens=5)

    with patch("conduit_bench.runners.model_executor.Agent") as MockAgent, \
         patch("conduit_bench.runners.model_executor.compute_cost", return_value=0.0001):
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
            model_id="anthropic:claude-3-haiku-20240307",
            provider="anthropic",
            model_name="claude-3-haiku-20240307",
            cost_per_input_token=0.00025,
            cost_per_output_token=0.00125,
        ),
    ]

    mock_result = _create_mock_agent_result("Test response", input_tokens=10, output_tokens=5)

    with patch("conduit_bench.runners.model_executor.Agent") as MockAgent, \
         patch("conduit_bench.runners.model_executor.compute_cost", return_value=0.0001):
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
    assert "anthropic:claude-3-haiku-20240307" in results
    assert all(r.success for r in results.values())


@pytest.mark.asyncio
async def test_cost_calculation(test_arm: ModelArm, executor: ModelExecutor) -> None:
    """Test cost calculation uses compute_cost from conduit.core.pricing."""
    mock_result = _create_mock_agent_result("Response", input_tokens=1000, output_tokens=500)

    with patch("conduit_bench.runners.model_executor.Agent") as MockAgent, \
         patch("conduit_bench.runners.model_executor.compute_cost", return_value=0.00045) as mock_cost:
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        MockAgent.return_value = mock_agent

        result = await executor.execute(
            arm=test_arm,
            query_text="Test",
        )

    # Verify compute_cost was called with correct parameters
    mock_cost.assert_called_once_with(
        input_tokens=1000,
        output_tokens=500,
        model_id="openai:gpt-4o-mini",
    )
    assert result.cost == 0.00045


@pytest.mark.asyncio
async def test_execute_with_fallback_primary_succeeds(
    test_arm: ModelArm, fallback_arm: ModelArm, executor: ModelExecutor
) -> None:
    """Test execute_with_fallback when primary model succeeds."""
    mock_result = _create_mock_agent_result("Primary response", input_tokens=10, output_tokens=5)

    with patch("conduit_bench.runners.model_executor.Agent") as MockAgent, \
         patch("conduit_bench.runners.model_executor.compute_cost", return_value=0.0001):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        MockAgent.return_value = mock_agent

        result = await executor.execute_with_fallback(
            primary_arm=test_arm,
            fallback_arms=[fallback_arm],
            query_text="Test query",
        )

    assert result.success
    assert result.model_id == "openai:gpt-4o-mini"
    assert result.response_text == "Primary response"
    assert not result.was_fallback
    assert result.primary_model is None
    assert result.failed_models == []


@pytest.mark.asyncio
async def test_execute_with_fallback_uses_fallback(
    test_arm: ModelArm, fallback_arm: ModelArm, executor: ModelExecutor
) -> None:
    """Test execute_with_fallback when primary fails and fallback succeeds."""
    mock_success = _create_mock_agent_result("Fallback response", input_tokens=10, output_tokens=5)

    call_count = 0

    async def mock_run(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # First two calls (primary + 1 retry) fail, then fallback succeeds
        if call_count <= 2:
            raise Exception("Primary model error")
        return mock_success

    with patch("conduit_bench.runners.model_executor.Agent") as MockAgent, \
         patch("conduit_bench.runners.model_executor.compute_cost", return_value=0.0001):
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=mock_run)
        MockAgent.return_value = mock_agent

        result = await executor.execute_with_fallback(
            primary_arm=test_arm,
            fallback_arms=[fallback_arm],
            query_text="Test query",
        )

    assert result.success
    assert result.model_id == "anthropic:claude-3-haiku-20240307"
    assert result.response_text == "Fallback response"
    assert result.was_fallback
    assert result.primary_model == "openai:gpt-4o-mini"
    assert "openai:gpt-4o-mini" in result.failed_models


@pytest.mark.asyncio
async def test_execute_with_fallback_all_fail(
    test_arm: ModelArm, fallback_arm: ModelArm, executor: ModelExecutor
) -> None:
    """Test execute_with_fallback when all models fail."""
    with patch("conduit_bench.runners.model_executor.Agent") as MockAgent:
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=Exception("All models failed"))
        MockAgent.return_value = mock_agent

        result = await executor.execute_with_fallback(
            primary_arm=test_arm,
            fallback_arms=[fallback_arm],
            query_text="Test query",
        )

    assert not result.success
    assert result.was_fallback
    assert result.primary_model == "openai:gpt-4o-mini"
    assert "openai:gpt-4o-mini" in result.failed_models
    assert "anthropic:claude-3-haiku-20240307" in result.failed_models


@pytest.mark.asyncio
async def test_execute_empty_response_treated_as_failure(
    test_arm: ModelArm, executor: ModelExecutor
) -> None:
    """Test that empty responses are treated as failures."""
    mock_result = _create_mock_agent_result("", input_tokens=10, output_tokens=0)

    with patch("conduit_bench.runners.model_executor.Agent") as MockAgent:
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_result)
        MockAgent.return_value = mock_agent

        result = await executor.execute(
            arm=test_arm,
            query_text="Test query",
        )

    assert not result.success
    assert "Empty response" in result.error


@pytest.mark.asyncio
async def test_execute_timeout(test_arm: ModelArm) -> None:
    """Test execution timeout handling."""
    import asyncio

    executor = ModelExecutor(timeout=0.1, max_retries=0)

    async def slow_response(*args, **kwargs):
        await asyncio.sleep(1.0)
        return _create_mock_agent_result("Too slow")

    with patch("conduit_bench.runners.model_executor.Agent") as MockAgent:
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=slow_response)
        MockAgent.return_value = mock_agent

        result = await executor.execute(
            arm=test_arm,
            query_text="Test query",
        )

    assert not result.success
    assert "TimeoutError" in result.error or "timeout" in result.error.lower()
