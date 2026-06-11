"""
Observation builder shared by the training environment and the RLAgent.

Encodes the engine's ground-truth Observation into a fixed-size, normalized
vector. Built here once so the policy sees byte-identical features in
training and in evaluation, and so every feature has a human-readable label
(`labels()`) for logging and reports — no anonymous black-box inputs.

Layout (all values normalized to roughly [-1, 1]):
    0  goal distance (clipped by world diagonal)
    1  goal bearing relative to heading (sin)
    2  goal bearing relative to heading (cos)
    3  speed (surge) / max_speed
    4  sway velocity / max_speed
    5  turn rate / max plausible turn rate
    6  rudder angle / max rudder
    7..7+R-1                lidar: land/bounds distance per ray (1 = clear)
    7+R..7+R+4*K-1          per tracked vessel (K nearest):
                              distance, relative bearing (sin, cos),
                              closing speed (positive = approaching)
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from src.config import Config
from src.sim.engine import Observation


class ObservationBuilder:
    def __init__(self, config: Config):
        self.config = config
        self.n_rays = config.rl.n_lidar_rays
        self.lidar_range = config.rl.lidar_range
        self.n_tracked = config.rl.n_tracked_obstacles
        self.size = 7 + self.n_rays + 4 * self.n_tracked
        self.n_scalar = 7        # features before the lidar block

    def labels(self) -> List[str]:
        out = ["goal_distance", "goal_bearing_sin", "goal_bearing_cos",
               "speed", "sway", "turn_rate", "rudder_angle"]
        out += [f"lidar_{i}" for i in range(self.n_rays)]
        for k in range(self.n_tracked):
            out += [f"vessel{k}_distance", f"vessel{k}_bearing_sin",
                    f"vessel{k}_bearing_cos", f"vessel{k}_closing_speed"]
        return out

    def build(self, obs: Observation) -> np.ndarray:
        cfg = self.config
        v = obs.vessel
        x, y, heading = v["x"], v["y"], v["heading"]
        world = obs.world

        diag = float(np.hypot(world.width, world.height))
        goal_dist = min(obs.distance_to_goal / diag, 1.0)
        goal_bearing = np.arctan2(obs.goal[1] - y, obs.goal[0] - x) - heading

        max_turn = cfg.vessel.nomoto_K * cfg.vessel.max_rudder
        vec = [
            goal_dist,
            np.sin(goal_bearing),
            np.cos(goal_bearing),
            v["speed"] / cfg.vessel.max_speed,
            np.clip(v.get("sway", 0.0) / cfg.vessel.max_speed, -1.0, 1.0),
            np.clip(v["turn_rate"] / max_turn, -1.0, 1.0),
            v["rudder_angle"] / cfg.vessel.max_rudder,
        ]
        vec.extend(self._lidar(world, x, y, heading))
        vec.extend(self._tracked_vessels(obs))
        return np.asarray(vec, dtype=np.float32)

    # --------------------------------------------------------------- features

    def _lidar(self, world, x: float, y: float, heading: float) -> List[float]:
        """Distance to land or world edge per ray, 1.0 = clear at range."""
        out = []
        step = 1.0
        for i in range(self.n_rays):
            ang = heading + 2.0 * np.pi * i / self.n_rays
            dx, dy = np.cos(ang) * step, np.sin(ang) * step
            d = self.lidar_range
            px, py = x, y
            for j in range(1, int(self.lidar_range / step) + 1):
                px += dx
                py += dy
                if not (0 <= px < world.width and 0 <= py < world.height):
                    d = j * step
                    break
                if world.grid[int(py), int(px)] > 0.5:
                    d = j * step
                    break
            out.append(d / self.lidar_range)
        return out

    def _tracked_vessels(self, obs: Observation) -> List[float]:
        v = obs.vessel
        x, y, heading = v["x"], v["y"], v["heading"]
        vx = v["speed"] * np.cos(heading)
        vy = v["speed"] * np.sin(heading)
        diag = float(np.hypot(obs.world.width, obs.world.height))

        ranked = sorted(obs.obstacles,
                        key=lambda ob: np.hypot(ob["x"] - x, ob["y"] - y))
        out: List[float] = []
        for k in range(self.n_tracked):
            if k < len(ranked):
                ob = ranked[k]
                rx, ry = ob["x"] - x, ob["y"] - y
                dist = float(np.hypot(rx, ry))
                bearing = np.arctan2(ry, rx) - heading
                rvx = ob["speed"] * np.cos(ob["heading"]) - vx
                rvy = ob["speed"] * np.sin(ob["heading"]) - vy
                # Positive when the range is shrinking
                closing = (-(rx * rvx + ry * rvy) / max(dist, 1e-6))
                out += [min(dist / diag, 1.0), float(np.sin(bearing)),
                        float(np.cos(bearing)),
                        float(np.clip(closing / (2 * self.config.vessel.max_speed),
                                      -1.0, 1.0))]
            else:
                # Padding for "no vessel": far away, no closing speed
                out += [1.0, 0.0, 1.0, 0.0]
        return out

    def to_dict(self, vec: np.ndarray) -> Dict[str, float]:
        return {label: round(float(val), 4)
                for label, val in zip(self.labels(), vec)}
