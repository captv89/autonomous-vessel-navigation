"""
Common agent interface.

Both navigation approaches implement `Agent`. The contract that makes the
comparison transparent: every step, an agent returns a `Decision` carrying
not just the helm order (desired heading + speed) but a JSON-serializable
`explanation` of *why* — the active COLREGs rule and CPA/TCPA numbers for
the classical agent, the action probability distribution and value estimate
for the RL agent. Decisions are logged verbatim by the episode recorder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

from src.sim.engine import Observation


@dataclass
class Decision:
    desired_heading: float            # radians
    desired_speed: float              # cells/s
    explanation: Dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        return {"desired_heading": float(self.desired_heading),
                "desired_speed": float(self.desired_speed),
                **self.explanation}


class Agent(ABC):
    """A navigation policy driving the simulation engine."""

    name: str = "agent"

    @abstractmethod
    def reset(self, obs: Observation) -> None:
        """Prepare for a new episode (e.g. plan a route)."""

    @abstractmethod
    def decide(self, obs: Observation) -> Decision:
        """Return the helm order for this step."""

    def metadata(self) -> Dict[str, Any]:
        """Static description logged in the episode header."""
        return {"name": self.name, "type": type(self).__name__}
