"""ModelExecutor for LLM query execution.

Handles direct PydanticAI calls to LLM models with cost tracking,
latency measurement, and error handling.
"""

import asyncio
import time

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName

from conduit.engines.bandits import ModelArm


class ModelExecutionResult(BaseModel):
    """Result from executing a single model on a query.

    Attributes:
        model_id: Model identifier (e.g., "openai:gpt-4o-mini")
        response_text: Generated response text
        cost: Actual cost incurred in USD
        latency: Response time in seconds
        input_tokens: Number of input tokens consumed
        output_tokens: Number of output tokens generated
        success: Whether execution succeeded
        error: Error message if execution failed
    """

    model_id: str
    response_text: str
    cost: float = Field(..., ge=0.0)
    latency: float = Field(..., ge=0.0)
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    success: bool = True
    error: str | None = None


class ModelExecutor:
    """Executes LLM queries using PydanticAI with cost and latency tracking.

    This class handles the direct execution of queries against LLM models,
    measuring costs, latency, and token usage. Used by BenchmarkRunner
    to execute queries for bandit algorithm evaluation.

    Example:
        >>> executor = ModelExecutor()
        >>> arm = ModelArm(
        ...     model_id="openai:gpt-4o-mini",
        ...     provider="openai",
        ...     model_name="gpt-4o-mini",
        ...     cost_per_input_token=0.00015,
        ...     cost_per_output_token=0.0006
        ... )
        >>> result = await executor.execute(
        ...     arm=arm,
        ...     query_text="What is 2+2?",
        ...     system_prompt="You are a helpful assistant."
        ... )
        >>> print(result.response_text)
        "2+2 equals 4."
        >>> print(f"Cost: ${result.cost:.6f}")
        Cost: $0.000012
    """

    def __init__(self, timeout: float = 60.0, max_retries: int = 2) -> None:
        """Initialize ModelExecutor.

        Args:
            timeout: Maximum execution time per query in seconds
            max_retries: Number of retry attempts on failure
        """
        self.timeout = timeout
        self.max_retries = max_retries

    async def execute(
        self,
        arm: ModelArm,
        query_text: str,
        system_prompt: str | None = None,
    ) -> ModelExecutionResult:
        """Execute a query against a specific model.

        Args:
            arm: Model arm with pricing and configuration
            query_text: User query text
            system_prompt: Optional system prompt for the model

        Returns:
            Execution result with response, cost, and metrics

        Raises:
            asyncio.TimeoutError: If execution exceeds timeout
            Exception: For other execution failures (captured in result.error)
        """
        start_time = time.time()

        # Build model name for PydanticAI
        model_name: KnownModelName = arm.full_name  # type: ignore

        # Create agent with system prompt
        agent: Agent[None, str] = Agent(
            model_name,
            system_prompt=system_prompt or "You are a helpful assistant.",
        )

        # Retry logic
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    agent.run(query_text),
                    timeout=self.timeout,
                )

                # Calculate cost from token usage
                cost_result = result.cost()
                input_tokens = cost_result.request_tokens or 0
                output_tokens = cost_result.response_tokens or 0

                # Calculate cost in USD
                cost_usd = (
                    input_tokens * arm.cost_per_input_token / 1000.0
                    + output_tokens * arm.cost_per_output_token / 1000.0
                )

                latency = time.time() - start_time

                return ModelExecutionResult(
                    model_id=arm.model_id,
                    response_text=result.data,
                    cost=cost_usd,
                    latency=latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=True,
                    error=None,
                )

            except asyncio.TimeoutError as e:
                last_error = e
                if attempt < self.max_retries:
                    await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                    continue
                break

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff
                    continue
                break

        # All retries failed
        latency = time.time() - start_time
        error_msg = f"{type(last_error).__name__}: {str(last_error)}" if last_error else "Unknown error"

        return ModelExecutionResult(
            model_id=arm.model_id,
            response_text="",
            cost=0.0,
            latency=latency,
            input_tokens=0,
            output_tokens=0,
            success=False,
            error=error_msg,
        )

    async def execute_batch(
        self,
        arms: list[ModelArm],
        query_text: str,
        system_prompt: str | None = None,
        max_concurrency: int = 5,
    ) -> dict[str, ModelExecutionResult]:
        """Execute a query against multiple models in parallel.

        Args:
            arms: List of model arms to execute
            query_text: User query text
            system_prompt: Optional system prompt
            max_concurrency: Maximum number of parallel executions

        Returns:
            Dictionary mapping model_id to execution result
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def execute_with_semaphore(arm: ModelArm) -> tuple[str, ModelExecutionResult]:
            async with semaphore:
                result = await self.execute(arm, query_text, system_prompt)
                return arm.model_id, result

        # Execute all models in parallel (limited by semaphore)
        tasks = [execute_with_semaphore(arm) for arm in arms]
        results = await asyncio.gather(*tasks)

        return dict(results)
