"""
Gymnasium environment for RL training.

Thin wrapper around the shared SimulationEngine — the RL agent trains on
exactly the physics the classical agent is evaluated on. Actions are
relative course changes (the same "helm order" abstraction the classical
agent uses); the engine's PD autopilot turns them into rudder movements.

Transparency: `info` carries a full reward decomposition every step, plus
the engine events, so training behavior can be audited after the fact.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import Config
from src.sim.engine import SimulationEngine, OUTCOME_GOAL
from src.sim.scenarios import build_scenario
from src.rl.observation import ObservationBuilder


class VesselNavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Optional[Config] = None,
                 scenario_name: Optional[str] = None,
                 seed: Optional[int] = None):
        super().__init__()
        self.config = config or Config()
        # scenario_name may be a comma-separated mix; one is sampled per
        # episode (e.g. "random,random_encounter").
        self.scenario_name = scenario_name or self.config.training.scenario
        self._scenario_pool = [n.strip()
                               for n in self.scenario_name.split(",")]
        self.builder = ObservationBuilder(self.config)
        self.heading_actions = [np.radians(a)
                                for a in self.config.rl.heading_actions_deg]

        self.action_space = spaces.Discrete(len(self.heading_actions))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.builder.size,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self.engine: Optional[SimulationEngine] = None
        self._prev_goal_distance = 0.0

    # ------------------------------------------------------------------- API

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        # Benchmark seeds (1000..) are held out by convention: train seeds
        # start above them.
        scenario_seed = int(self._rng.integers(10_000, 2**31 - 1))
        name = self._scenario_pool[
            int(self._rng.integers(len(self._scenario_pool)))]
        scenario = build_scenario(name, seed=scenario_seed)
        self.engine = SimulationEngine(self.config, scenario)
        obs = self.engine.reset()
        self._prev_goal_distance = obs.distance_to_goal
        return self.builder.build(obs), {"scenario_seed": scenario_seed}

    def step(self, action: int
             ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """One decision = `rl.action_repeat` engine steps holding the same
        helm order (rewards accumulate across the held steps)."""
        obs_before = self.engine.observe()
        heading_change = self.heading_actions[int(action)]
        desired_heading = obs_before.vessel["heading"] + heading_change

        reward_parts: Dict[str, float] = {
            "heading_change":
                self.config.rl.reward.heading_change_penalty
                * abs(heading_change)}
        events: List[str] = []
        for _ in range(max(self.config.rl.action_repeat, 1)):
            result = self.engine.step(desired_heading,
                                      self.config.vessel.cruise_speed)
            events.extend(result.events)
            for key, val in self._reward(result.obs, result).items():
                reward_parts[key] = reward_parts.get(key, 0.0) + val
            if result.done:
                break
        obs = result.obs
        obs_vec = self.builder.build(obs)

        # Shaped land/wall proximity penalty from the lidar the agent sees
        # (terminal grounding alone is too sparse a signal to learn from).
        rw = self.config.rl.reward
        n0 = self.builder.n_scalar
        lidar = obs_vec[n0:n0 + self.config.rl.n_lidar_rays]
        min_land = float(lidar.min()) * self.config.rl.lidar_range
        if min_land < rw.land_proximity_range:
            reward_parts["land_proximity"] = rw.land_proximity * (
                1.0 - min_land / rw.land_proximity_range)

        reward = float(sum(reward_parts.values()))

        terminated = result.done and result.outcome != "timeout"
        truncated = result.done and result.outcome == "timeout"
        info: Dict[str, Any] = {
            "reward_components": reward_parts,
            "events": events,
            "outcome": result.outcome,
        }
        if result.done:
            info["is_success"] = result.outcome == OUTCOME_GOAL
        return (obs_vec, reward, terminated, truncated, info)

    # ---------------------------------------------------------------- reward

    def _reward(self, obs, result) -> Dict[str, float]:
        rw = self.config.rl.reward
        parts: Dict[str, float] = {"time": rw.time_penalty}

        goal_distance = obs.distance_to_goal
        parts["progress"] = rw.progress_scale * (
            self._prev_goal_distance - goal_distance)
        self._prev_goal_distance = goal_distance

        near_miss_at = self.config.avoidance.detector_safe_distance * 1.5
        min_sep = min((np.hypot(ob["x"] - obs.vessel["x"],
                                ob["y"] - obs.vessel["y"])
                       for ob in obs.obstacles), default=float("inf"))
        if min_sep < near_miss_at:
            parts["near_miss"] = rw.near_miss * (1.0 - min_sep / near_miss_at)

        if result.outcome == "goal":
            parts["goal"] = rw.goal_reached
        elif result.outcome == "collision":
            parts["collision"] = rw.collision
        elif result.outcome == "grounding":
            parts["grounding"] = rw.grounding
        elif result.outcome == "out_of_bounds":
            parts["out_of_bounds"] = rw.out_of_bounds
        return parts
