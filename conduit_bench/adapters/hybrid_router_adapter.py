"""Adapter to make HybridRouter compatible with BanditAlgorithm interface.

HybridRouter uses route(query) -> RoutingDecision, but BenchmarkRunner
expects select_arm(features) -> ModelArm. This adapter bridges the gap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from conduit.engines.hybrid_router import HybridRouter
    from conduit.engines.bandits.base import BanditFeedback, ModelArm
    from conduit.core.models import QueryFeatures


class HybridRouterBanditAdapter:
    """Adapter that wraps HybridRouter to expose BanditAlgorithm interface.

    BenchmarkRunner expects algorithms with:
    - name: str property
    - select_arm(features) -> ModelArm
    - update(feedback, features) -> None
    - get_stats() -> dict
    - reset() -> None

    HybridRouter has:
    - route(query) -> RoutingDecision (different interface)
    - update(feedback, features) -> None (compatible)
    - get_stats() -> dict (compatible)
    - reset() -> None (compatible)
    - phase1_bandit / phase2_bandit with select_arm() (internal)

    This adapter:
    1. Generates a descriptive name from phase1/phase2 algorithms
    2. Delegates select_arm() to the current phase's internal bandit
    3. Handles phase transitions by tracking query count
    4. Passes through update(), get_stats(), reset()

    Example:
        >>> from conduit.engines.hybrid_router import HybridRouter
        >>> router = HybridRouter(
        ...     models=["gpt-4o-mini", "gpt-4o"],
        ...     phase1_algorithm="thompson_sampling",
        ...     phase2_algorithm="linucb",
        ...     switch_threshold=100,
        ... )
        >>> adapter = HybridRouterBanditAdapter(router)
        >>> adapter.name
        'hybrid_thompson_linucb'
        >>> arm = await adapter.select_arm(features)
    """

    def __init__(self, router: HybridRouter) -> None:
        """Initialize adapter with a HybridRouter instance.

        Args:
            router: The HybridRouter to adapt
        """
        self._router = router
        # Generate name from algorithm combination
        p1 = router.phase1_algorithm.replace("_sampling", "")
        p2 = router.phase2_algorithm.replace("contextual_", "c_").replace("_sampling", "")
        self._name = f"hybrid_{p1}_{p2}"

    @property
    def name(self) -> str:
        """Algorithm name for display and tracking."""
        return self._name

    @property
    def arms(self) -> dict[str, Any]:
        """Expose arms from underlying router."""
        return self._router.phase1_bandit.arms

    async def select_arm(self, features: QueryFeatures) -> ModelArm:
        """Select an arm using the current phase's bandit.

        This bypasses HybridRouter's route() method and goes directly
        to the internal bandits, maintaining the query count and
        handling phase transitions.

        Args:
            features: Query features extracted by QueryAnalyzer

        Returns:
            Selected ModelArm
        """
        # Increment query count (normally done by route())
        self._router.query_count += 1

        # Check if should transition to phase2
        if (
            self._router.current_phase == self._router.phase1_algorithm
            and self._router.query_count >= self._router.switch_threshold
        ):
            await self._router._transition_to_phase2()

        # Delegate to current phase's bandit
        if self._router.current_phase == self._router.phase1_algorithm:
            return await self._router.phase1_bandit.select_arm(features)
        else:
            return await self._router.phase2_bandit.select_arm(features)

    async def update(self, feedback: BanditFeedback, features: QueryFeatures) -> None:
        """Update the current phase's bandit with feedback.

        Args:
            feedback: Bandit feedback with quality/cost/latency
            features: Query features
        """
        await self._router.update(feedback, features)

    def get_stats(self) -> dict[str, Any]:
        """Get statistics from the router.

        Returns:
            Dictionary with phase info, query count, and bandit stats
        """
        return self._router.get_stats()

    def reset(self) -> None:
        """Reset the router to initial state."""
        self._router.reset()

    def to_state(self) -> Any:
        """Serialize router state for persistence."""
        return self._router.to_state()

    def from_state(self, state: Any, allow_conversion: bool = True) -> None:
        """Restore router state from persisted data."""
        self._router.from_state(state, allow_conversion=allow_conversion)
