"""
Shielded RL agent: a learned policy with a classical runtime safety filter.

The RL policy proposes a helm order each step; the shield rolls the proposal
out with the *real* vessel dynamics against constant-velocity traffic
predictions and the static grid. If the proposal is predicted to stay safe,
it passes through untouched. If not, the shield substitutes the nearest safe
course alteration (the same predictive machinery the classical agent uses),
holding it with hysteresis until the policy's own intent is safe again.

This is the architecture the maritime-autonomy literature converges on
(learned decision-making inside a certifiable safety envelope), and the
benchmark logs make the division of labor measurable: every decision records
whether the shield intervened and why, so reports can state exactly how much
of the final safety is attributable to the shield versus the policy.

The shield's safety threshold is deliberately tighter than the classical
agent's comfort distance (see `avoidance.shield_safe_distance`): it is a
last-resort envelope, not a second navigator — otherwise the comparison
would measure the shield, not the policy.
"""

from __future__ import annotations

import copy
from typing import Any, Dict

from src.config import Config
from src.agents.base import Agent, Decision
from src.agents.avoidance import PredictiveAvoider
from src.agents.rl_agent import RLAgent
from src.sim.engine import Observation


class ShieldedRLAgent(Agent):
    name = "rl_ppo_shielded"

    def __init__(self, config: Config, model_path: str):
        self.config = config
        self.policy = RLAgent(config, model_path)
        # The shield reuses the predictive avoider with a tighter envelope.
        shield_config = copy.deepcopy(config)
        shield_config.avoidance.safe_distance = \
            config.avoidance.shield_safe_distance
        self._shield_config = shield_config
        self.shield: PredictiveAvoider | None = None
        self.interventions = 0
        self.decisions = 0

    def metadata(self) -> Dict[str, Any]:
        meta = self.policy.metadata()
        meta.update({
            "name": self.name, "type": type(self).__name__,
            "family": "hybrid (learning + runtime safety filter)",
            "author": "V. Ravendranathan (VesselNav-Bench baseline)",
            "summary": "The PPO policy proposes; a classical predictive "
                       "filter vets every proposal with real-dynamics "
                       "rollouts and substitutes the nearest safe course "
                       "when needed. Every intervention is logged.",
            "shield": "predictive rollout filter",
            "shield_safe_distance":
                self.config.avoidance.shield_safe_distance,
        })
        return meta

    # --------------------------------------------------------------- episode

    def reset(self, obs: Observation) -> None:
        self.policy.reset(obs)
        self.shield = PredictiveAvoider(self._shield_config, obs.world)
        self.shield.engage_without_traffic = True
        self.shield.always_check_track = True
        self.interventions = 0
        self.decisions = 0

    def decide(self, obs: Observation) -> Decision:
        proposal = self.policy.decide(obs)
        self.shield.current = obs.current

        # The shield treats the policy's intent as the "track": it stays
        # inactive while the intent is predicted safe and otherwise picks
        # the nearest safe course alteration around it.
        verdict = self.shield.update(
            t=obs.t, vessel_state=obs.vessel,
            track_heading=proposal.desired_heading,
            obstacles=obs.obstacles)

        self.decisions += 1
        if not verdict.active:
            explanation = dict(proposal.explanation)
            explanation["shield"] = {"intervened": False}
            return Decision(proposal.desired_heading, proposal.desired_speed,
                            explanation)

        self.interventions += 1
        explanation = dict(proposal.explanation)
        explanation["mode"] = "shield"
        explanation["shield"] = {
            "intervened": True,
            "reason": verdict.reason,
            "policy_heading": float(proposal.desired_heading),
            "override_heading": float(verdict.heading),
            "offset_deg": verdict.offset_deg,
            "candidates": [c.to_record() for c in verdict.candidates],
        }
        return Decision(float(verdict.heading),
                        proposal.desired_speed * verdict.speed_factor,
                        explanation)
