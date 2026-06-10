from .base import Agent, Decision
from .classical import ClassicalAgent

__all__ = ["Agent", "Decision", "ClassicalAgent", "RLAgent"]


def __getattr__(name):
    # Lazy import: keeps torch/stable-baselines3 out of classical-only runs.
    if name == "RLAgent":
        from .rl_agent import RLAgent
        return RLAgent
    raise AttributeError(name)
