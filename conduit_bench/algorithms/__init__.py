"""Bandit algorithms for LLM routing comparison."""

from .base import BanditAlgorithm, ModelArm, BanditContext, BanditFeedback
from .thompson_sampling import ThompsonSamplingBandit
from .ucb import UCB1Bandit
from .epsilon_greedy import EpsilonGreedyBandit
from .baselines import RandomBaseline, OracleBaseline, AlwaysBestBaseline, AlwaysCheapestBaseline

__all__ = [
    "BanditAlgorithm",
    "ModelArm",
    "BanditContext",
    "BanditFeedback",
    "ThompsonSamplingBandit",
    "UCB1Bandit",
    "EpsilonGreedyBandit",
    "RandomBaseline",
    "OracleBaseline",
    "AlwaysBestBaseline",
    "AlwaysCheapestBaseline",
]
