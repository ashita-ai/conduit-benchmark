"""Base classes and interfaces for bandit algorithms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel, Field


class BanditContext(BaseModel):
    """Context information for bandit decision-making.

    Attributes:
        query_text: The query text
        query_embedding: Optional pre-computed embedding (768-dim vector)
        category: Query category (e.g., "technical_qa", "creative_writing")
        complexity: Query complexity ("simple", "moderate", "complex")
        metadata: Additional context features
    """

    query_text: str
    query_embedding: Optional[list[float]] = None
    category: Optional[str] = None
    complexity: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelArm(BaseModel):
    """Represents a model (arm) in the multi-armed bandit.

    Attributes:
        model_id: Unique identifier (e.g., "openai:gpt-4o-mini")
        provider: Provider name ("openai", "anthropic", "google", etc.)
        model_name: Model name within provider
        cost_per_input_token: Cost in USD per 1K input tokens
        cost_per_output_token: Cost in USD per 1K output tokens
        expected_quality: Prior estimate of quality (0-1 scale)
        metadata: Additional model characteristics
    """

    model_id: str
    provider: str
    model_name: str
    cost_per_input_token: float
    cost_per_output_token: float
    expected_quality: float = 0.5
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def full_name(self) -> str:
        """Get full model name for PydanticAI."""
        return f"{self.provider}:{self.model_name}"


class BanditFeedback(BaseModel):
    """Feedback from executing a model selection.

    Attributes:
        model_id: Which model was selected
        cost: Actual cost incurred (USD)
        quality_score: Quality score from evaluation (0-1 scale)
        latency: Response latency in seconds
        success: Whether execution succeeded
        metadata: Additional feedback data (token counts, etc.)
    """

    model_id: str
    cost: float
    quality_score: float
    latency: float
    success: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class BanditAlgorithm(ABC):
    """Abstract base class for multi-armed bandit algorithms.

    All bandit algorithms must implement:
    1. select_arm: Choose which model to use for a query
    2. update: Update internal state with feedback
    3. reset: Reset algorithm state
    """

    def __init__(self, name: str, arms: list[ModelArm]) -> None:
        """Initialize bandit algorithm.

        Args:
            name: Algorithm name for identification
            arms: List of available model arms
        """
        self.name = name
        self.arms = {arm.model_id: arm for arm in arms}
        self.arm_list = arms
        self.n_arms = len(arms)
        self.total_queries = 0

    @abstractmethod
    async def select_arm(self, context: BanditContext) -> ModelArm:
        """Select which model arm to pull for this query.

        Args:
            context: Query context information

        Returns:
            Selected model arm

        Example:
            >>> context = BanditContext(query_text="What is 2+2?")
            >>> arm = await algorithm.select_arm(context)
            >>> print(arm.model_id)
            "openai:gpt-4o-mini"
        """
        pass

    @abstractmethod
    async def update(self, feedback: BanditFeedback, context: BanditContext) -> None:
        """Update algorithm state with feedback from arm pull.

        Args:
            feedback: Feedback from model execution
            context: Original query context

        Example:
            >>> feedback = BanditFeedback(
            ...     model_id="openai:gpt-4o-mini",
            ...     cost=0.0001,
            ...     quality_score=0.95,
            ...     latency=1.2
            ... )
            >>> await algorithm.update(feedback, context)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset algorithm to initial state.

        Clears all learned parameters and history.
        Useful for running multiple independent experiments.

        Example:
            >>> algorithm.reset()
            >>> algorithm.total_queries
            0
        """
        pass

    def get_stats(self) -> dict[str, Any]:
        """Get algorithm statistics and state.

        Returns:
            Dictionary with algorithm-specific statistics

        Example:
            >>> stats = algorithm.get_stats()
            >>> print(stats["total_queries"])
            1000
        """
        return {
            "name": self.name,
            "total_queries": self.total_queries,
            "n_arms": self.n_arms,
        }


@dataclass
class BanditMetrics:
    """Metrics for evaluating bandit algorithm performance.

    Attributes:
        cumulative_regret: Total regret vs oracle (perfect knowledge)
        cumulative_cost: Total cost incurred
        cumulative_reward: Total reward accumulated
        average_quality: Average quality score
        convergence_point: Query number where performance stabilized
        arm_selection_counts: How many times each arm was selected
    """

    cumulative_regret: float
    cumulative_cost: float
    cumulative_reward: float
    average_quality: float
    convergence_point: Optional[int]
    arm_selection_counts: dict[str, int]
