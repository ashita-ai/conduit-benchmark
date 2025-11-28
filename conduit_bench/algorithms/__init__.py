"""Bandit algorithms - imported from Conduit.

All bandit algorithm implementations now live in Conduit.
This module re-exports them for backward compatibility.
"""

from conduit.engines.bandits import (
    AlwaysBestBaseline,
    AlwaysCheapestBaseline,
    BanditAlgorithm,
    BanditFeedback,
    ContextualThompsonSamplingBandit,
    DuelingBandit,
    EpsilonGreedyBandit,
    LinUCBBandit,
    ModelArm,
    OracleBaseline,
    RandomBaseline,
    ThompsonSamplingBandit,
    UCB1Bandit,
)

__all__ = [
    "BanditAlgorithm",
    "ModelArm",
    "BanditFeedback",
    # Contextual algorithms
    "LinUCBBandit",
    "ContextualThompsonSamplingBandit",
    "DuelingBandit",
    # Non-contextual algorithms
    "ThompsonSamplingBandit",
    "UCB1Bandit",
    "EpsilonGreedyBandit",
    # Baselines
    "RandomBaseline",
    "OracleBaseline",
    "AlwaysBestBaseline",
    "AlwaysCheapestBaseline",
]
