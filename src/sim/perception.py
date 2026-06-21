"""
Degraded-perception model for the benchmark's `degraded` condition.

Real perception is not AIS-grade ground truth: target positions and courses
carry noise, fixes arrive at a finite interval (so a moving target's reported
position lags reality), and targets are occasionally lost. This model applies
those effects to the *perceived traffic* an agent decides on, while the engine
and the episode recorder keep ground truth — so collision/CPA/COLREGs scoring
stays physically correct and only the agent's information is degraded.

Applied at the shared observation funnel in `run_episode`, every agent
(classical, MPC, RL) sees the same degraded traffic. Seeded from the scenario
seed, so the noise realization is byte-identical across agents for a given
(scenario, seed, condition) triple — the benchmark's fairness guarantee.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Optional

import numpy as np

from src.config import Config
from src.sim.engine import Observation

# Decouples the perception RNG stream from the engine's disturbance RNG, which
# is seeded from the same scenario seed.
_PERCEPTION_SALT = 0x9E3779B9


class Perception:
    """Per-episode, seeded view of traffic for one agent run."""

    def __init__(self, config: Config, seed: Optional[int]):
        self.cfg = config.perception
        self.rng = np.random.default_rng((seed or 0) ^ _PERCEPTION_SALT)
        # id -> {"t": time of last fix, "ob": reported dict or None if lost}
        self._fixes: Dict[Any, Dict[str, Any]] = {}

    def perceive(self, obs: Observation) -> Observation:
        if not self.cfg.enabled:
            return obs

        course_noise = np.radians(self.cfg.course_noise_deg)
        reported = []
        for ob in obs.obstacles:
            last = self._fixes.get(ob["id"])
            due = last is None or (obs.t - last["t"]) >= self.cfg.update_interval
            if due:
                if self.rng.random() < self.cfg.dropout_prob:
                    fix = None                      # target momentarily lost
                else:
                    fix = dict(ob)
                    fix["x"] += self.rng.normal(0.0, self.cfg.position_noise)
                    fix["y"] += self.rng.normal(0.0, self.cfg.position_noise)
                    fix["heading"] += self.rng.normal(0.0, course_noise)
                    fix["speed"] = max(
                        0.0, ob["speed"] + self.rng.normal(0.0, self.cfg.speed_noise))
                self._fixes[ob["id"]] = {"t": obs.t, "ob": fix}
            else:
                fix = last["ob"]                    # stale: hold the last fix
            if fix is not None:
                reported.append(fix)

        return dataclasses.replace(obs, obstacles=reported)
