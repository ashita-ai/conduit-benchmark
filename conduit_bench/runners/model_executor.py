"""ModelExecutor for LLM query execution.

Handles direct PydanticAI calls to LLM models with cost tracking,
latency measurement, and error handling.

Uses Conduit's LiteLLM-based pricing for accurate cost calculation.
"""

import asyncio
import time

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import KnownModelName
from conduit.core.pricing import compute_cost

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
        was_fallback: Whether a fallback model was used
        primary_model: The originally selected model (if fallback was used)
        failed_models: List of models that failed before success
        query_text: The query text (for debugging)
        system_prompt: The system prompt used (for debugging)
    """

    model_id: str
    response_text: str
    cost: float = Field(..., ge=0.0)
    latency: float = Field(..., ge=0.0)
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    success: bool = True
    error: str | None = None
    was_fallback: bool = False
    primary_model: str | None = None
    failed_models: list[str] = Field(default_factory=list)
    query_text: str | None = None
    system_prompt: str | None = None


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

                # Get token usage from result
                usage = result.usage()

                # Extract tokens (use new naming, fallback to deprecated)
                input_tokens = getattr(usage, "input_tokens", None)
                output_tokens = getattr(usage, "output_tokens", None)
                if input_tokens is None:
                    input_tokens = getattr(usage, "request_tokens", 0) or 0
                    output_tokens = getattr(usage, "response_tokens", 0) or 0

                # Calculate cost using Conduit's LiteLLM-based pricing
                cost_usd = compute_cost(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model_id=arm.model_id,
                )

                latency = time.time() - start_time

                response_text = result.output if hasattr(result, 'output') else str(result.data)

                # Detect empty responses (API returned but with no content)
                if not response_text or response_text.strip() == "":
                    raise ValueError("Empty response from API")

                return ModelExecutionResult(
                    model_id=arm.model_id,
                    response_text=response_text,
                    cost=cost_usd,
                    latency=latency,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    success=True,
                    error=None,
                    query_text=query_text,
                    system_prompt=system_prompt,
                )

            except asyncio.TimeoutError as e:
                last_error = e
                if attempt < self.max_retries:
                    # Exponential backoff with jitter: 2^attempt + random(0-1)
                    import random
                    backoff = (2 ** attempt) + random.random()
                    await asyncio.sleep(backoff)
                    continue
                break

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    # Exponential backoff with jitter: 2^attempt + random(0-1)
                    import random
                    backoff = (2 ** attempt) + random.random()
                    await asyncio.sleep(backoff)
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
            query_text=query_text,
            system_prompt=system_prompt,
        )

    async def execute_with_fallback(
        self,
        primary_arm: ModelArm,
        fallback_arms: list[ModelArm],
        query_text: str,
        system_prompt: str | None = None,
    ) -> ModelExecutionResult:
        """Execute query with automatic fallback to alternative models on failure.

        Tries the primary model first, then fallback models in order until success.
        Designed to handle empty responses and API failures gracefully.

        Args:
            primary_arm: Primary model to try first
            fallback_arms: Ordered list of fallback models to try on failure
            query_text: User query text
            system_prompt: Optional system prompt

        Returns:
            ModelExecutionResult with was_fallback, failed_models, and error messages

        Example:
            >>> result = await executor.execute_with_fallback(
            ...     primary_arm=opus_arm,
            ...     fallback_arms=[sonnet_arm, haiku_arm],
            ...     query_text="What is 2+2?",
            ... )
            >>> if result.was_fallback:
            ...     print(f"Primary {result.primary_model} failed, used {result.model_id}")
            ...     print(f"Failed models: {result.failed_models}")
        """
        failed_models: list[str] = []
        failed_errors: dict[str, str] = {}  # Map model_id → error message

        # Try primary model
        result = await self.execute(primary_arm, query_text, system_prompt)

        if result.success:
            return result

        # Primary failed, record it with error
        failed_models.append(primary_arm.model_id)
        if result.error:
            failed_errors[primary_arm.model_id] = result.error

        # Try fallback models in order
        for fallback_arm in fallback_arms:
            result = await self.execute(fallback_arm, query_text, system_prompt)

            if result.success:
                # Fallback succeeded!
                result.was_fallback = True
                result.primary_model = primary_arm.model_id
                result.failed_models = failed_models.copy()
                # Store error messages as JSON string in result.error for logging
                if failed_errors:
                    import json
                    result.error = json.dumps(failed_errors)
                return result

            # Fallback also failed
            failed_models.append(fallback_arm.model_id)
            if result.error:
                failed_errors[fallback_arm.model_id] = result.error

        # All models failed - return the last failure with metadata
        result.was_fallback = True  # Attempted fallback but all failed
        result.primary_model = primary_arm.model_id
        result.failed_models = failed_models.copy()
        # Store all error messages
        if failed_errors:
            import json
            result.error = json.dumps(failed_errors)
        return result

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
