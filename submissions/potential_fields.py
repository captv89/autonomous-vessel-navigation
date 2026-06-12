"""
Artificial Potential Fields (APF) — example external submission.

Method: O. Khatib, "Real-Time Obstacle Avoidance for Manipulators and
Mobile Robots", The International Journal of Robotics Research, 5(1),
1986. Independent reference implementation against the VesselNav-Bench
`Agent` contract:

    python main.py benchmark --suite benchmarks/v1.yaml \\
        --agent submissions.potential_fields:APFAgent

The vessel moves in an artificial force field: an attractive force pulls
toward a target on the planned route, repulsive forces (active inside an
influence radius) push away from traffic vessels and nearby land; the helm
order is the direction of the resultant force. APF is famous both for its
elegance and for its known weaknesses (local minima, oscillation in narrow
passages) — it is included as a historically important reference point, not
as a competitive entry.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from src.config import Config
from src.agents.base import Agent, Decision
from src.sim.engine import Observation
from src.pathfinding.astar import AStar
from src.vessel.path_follower import LOSController


class APFAgent(Agent):
    name = "potential-fields"

    K_ATT = 1.0                  # attractive gain (unit vector toward target)
    K_REP = 60.0                 # repulsive gain, Khatib's (1/d - 1/d0)/d^2
    D0_TRAFFIC = 18.0            # influence radius around traffic (cells)
    D0_LAND = 8.0                # influence radius around land (cells)
    N_LAND_RAYS = 16

    def __init__(self, config: Config):
        self.config = config
        self.waypoints: List[Tuple[float, float]] = []

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": type(self).__name__,
            "family": "reactive (force field)",
            "author": "Khatib (1986); reference implementation",
            "summary": "Steers along the resultant of an attractive force "
                       "toward the route and repulsive forces from traffic "
                       "and land. Historically foundational; known to "
                       "suffer local minima and oscillation.",
            "waypoints": [list(wp) for wp in self.waypoints],
        }

    def reset(self, obs: Observation) -> None:
        cfg = self.config
        planner = AStar(obs.world, allow_diagonal=True,
                        safety_margin_cells=cfg.planner.safety_margin)
        start = (int(round(obs.vessel["x"])), int(round(obs.vessel["y"])))
        goal = (int(round(obs.goal[0])), int(round(obs.goal[1])))
        path = planner.find_path(start, goal, simplify=True,
                                 max_segment_length=cfg.planner.max_segment_length)
        self.waypoints = ([(float(x), float(y)) for x, y in path]
                          if path else [tuple(map(float, start)),
                                        tuple(map(float, goal))])
        self.world = obs.world
        fcfg = cfg.follower
        self.follower = LOSController(lookahead_distance=fcfg.lookahead,
                                      path_tolerance=fcfg.path_tolerance,
                                      integral_gain=fcfg.integral_gain)
        self.follower.reset()

    def decide(self, obs: Observation) -> Decision:
        cfg = self.config
        pos = np.array(obs.position)

        # Attractive force: unit vector toward the route's virtual target
        if obs.distance_to_goal < cfg.follower.lookahead:
            target_heading = np.arctan2(obs.goal[1] - pos[1],
                                        obs.goal[0] - pos[0])
        else:
            target_heading = self.follower.compute_desired_heading(
                pos, self.waypoints)
            if target_heading is None:
                target_heading = np.arctan2(obs.goal[1] - pos[1],
                                            obs.goal[0] - pos[0])
        force = self.K_ATT * np.array([np.cos(target_heading),
                                       np.sin(target_heading)])

        contributions = []
        # Repulsion from traffic (Khatib's FIRAS potential gradient)
        for ob in obs.obstacles:
            r = pos - np.array([ob["x"], ob["y"]])
            d = float(np.linalg.norm(r))
            if 1e-6 < d < self.D0_TRAFFIC:
                mag = self.K_REP * (1.0 / d - 1.0 / self.D0_TRAFFIC) / d**2
                force += mag * r / d
                contributions.append({"source": f"vessel{ob['id']}",
                                      "distance": round(d, 1),
                                      "magnitude": round(float(mag), 3)})
        # Repulsion from land: nearest land hit along radial rays
        for k in range(self.N_LAND_RAYS):
            ang = 2 * np.pi * k / self.N_LAND_RAYS
            d = self._land_distance(pos, ang, self.D0_LAND)
            if d is not None and d > 1e-6:
                mag = self.K_REP * (1.0 / d - 1.0 / self.D0_LAND) / d**2
                force += mag * np.array([-np.cos(ang), -np.sin(ang)])

        heading = float(np.arctan2(force[1], force[0]))
        err = abs(np.degrees(np.arctan2(
            np.sin(heading - obs.vessel["heading"]),
            np.cos(heading - obs.vessel["heading"]))))
        factor = 1.0 if err <= 15 else max(0.3, 1.0 - (err - 15) / 60)
        return Decision(
            desired_heading=heading,
            desired_speed=float(cfg.vessel.cruise_speed * factor),
            explanation={
                "mode": "apf",
                "force_heading_deg": round(np.degrees(heading), 1),
                "attractive_heading_deg": round(np.degrees(
                    float(target_heading)), 1),
                "repulsors": contributions[:6],
            })

    def _land_distance(self, pos, angle: float, max_d: float):
        dx, dy = np.cos(angle), np.sin(angle)
        for step in range(1, int(max_d) + 1):
            x, y = pos[0] + dx * step, pos[1] + dy * step
            if not (0 <= x < self.world.width and 0 <= y < self.world.height):
                return float(step)     # boundary repels like land
            if self.world.grid[int(y), int(x)] > 0.5:
                return float(step)
        return None
