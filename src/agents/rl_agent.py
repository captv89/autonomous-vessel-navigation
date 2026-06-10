"""
RL navigation agent: wraps a trained Stable-Baselines3 PPO policy behind
the common Agent interface so it runs through the exact evaluation loop the
classical agent uses.

Transparency: the network weights are a black box, but every decision
records what the policy saw (the labeled observation vector), the full
action probability distribution, the value estimate, and the chosen helm
order — auditable step by step in the episode logs and the replay viewer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.config import Config
from src.agents.base import Agent, Decision
from src.sim.engine import Observation
from src.rl.observation import ObservationBuilder


class RLAgent(Agent):
    name = "rl_ppo"

    def __init__(self, config: Config, model_path: str,
                 deterministic: bool = True):
        import torch  # local import: keep torch out of classical-only runs
        from stable_baselines3 import PPO

        self.config = config
        self.model_path = model_path
        self.deterministic = deterministic
        self.model = PPO.load(model_path, device="cpu")
        self.builder = ObservationBuilder(config)
        self.heading_actions = [np.radians(a)
                                for a in config.rl.heading_actions_deg]
        if self.model.observation_space.shape != (self.builder.size,):
            raise ValueError(
                f"Model expects observation shape "
                f"{self.model.observation_space.shape}, but config produces "
                f"({self.builder.size},). Re-train or fix the config.")
        self._torch = torch

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name, "type": type(self).__name__,
            "algorithm": "PPO (stable-baselines3)",
            "model_path": str(self.model_path),
            "deterministic": self.deterministic,
            "observation_labels": self.builder.labels(),
            "action_labels": self._action_labels(),
        }

    def _action_labels(self):
        return [f"{'stbd' if a < 0 else 'port' if a > 0 else 'steady'}"
                f"{abs(a):g}" for a in self.config.rl.heading_actions_deg]

    # --------------------------------------------------------------- episode

    def reset(self, obs: Observation) -> None:
        # The policy network is stateless; only the action-hold timer needs
        # clearing (decisions are re-evaluated at the training cadence:
        # every `rl.action_repeat` engine steps).
        self._held: Optional[Decision] = None
        self._held_until = -1.0

    def decide(self, obs: Observation) -> Decision:
        if self._held is not None and obs.t < self._held_until:
            held = Decision(self._held.desired_heading,
                            self._held.desired_speed,
                            {**self._held.explanation, "held": True})
            return held
        decision = self._evaluate_policy(obs)
        self._held = decision
        self._held_until = obs.t + (self.config.rl.action_repeat
                                    * self.config.simulation.dt) - 1e-9
        return decision

    def _evaluate_policy(self, obs: Observation) -> Decision:
        torch = self._torch
        vec = self.builder.build(obs)

        with torch.no_grad():
            tensor, _ = self.model.policy.obs_to_tensor(vec)
            dist = self.model.policy.get_distribution(tensor)
            probs = dist.distribution.probs.cpu().numpy().flatten()
            value = float(self.model.policy.predict_values(tensor).item())

        if self.deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(len(probs), p=probs))

        heading_change = self.heading_actions[action]
        desired_heading = obs.vessel["heading"] + heading_change

        labels = self._action_labels()
        explanation = {
            "mode": "policy",
            "action": labels[action],
            "action_index": action,
            "heading_change_deg":
                float(self.config.rl.heading_actions_deg[action]),
            "action_probs": {l: round(float(p), 4)
                             for l, p in zip(labels, probs)},
            "value_estimate": round(value, 2),
            "observation": self.builder.to_dict(vec),
        }
        return Decision(desired_heading=float(desired_heading),
                        desired_speed=float(self.config.vessel.cruise_speed),
                        explanation=explanation)
