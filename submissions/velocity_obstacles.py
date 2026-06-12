"""
Velocity Obstacles (VO) — example external submission to VesselNav-Bench.

Method: P. Fiorini and Z. Shiller, "Motion Planning in Dynamic Environments
Using Velocity Obstacles", The International Journal of Robotics Research,
17(7), 1998. This file is an independent reference implementation of that
published method, packaged exactly the way any third party would submit a
model to the benchmark: a standalone module implementing the `Agent`
contract, run with

    python main.py benchmark --suite benchmarks/v1.yaml \\
        --agent submissions.velocity_obstacles:VOAgent

The method in one paragraph: for each traffic vessel, the set of own-ship
velocities that would lead to a collision within a time horizon (given the
obstacle holds its current velocity) forms a cone in velocity space — the
"velocity obstacle". The vessel picks, each step, the velocity closest to
its preferred velocity (here: cruise speed along an A*-planned route) that
lies outside every cone. We add a starboard tie-preference so the selection
is COLREGs-leaning, and a simple land ray-check, both standard adaptations
in the maritime VO literature.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import Config
from src.agents.base import Agent, Decision
from src.sim.engine import Observation
from src.pathfinding.astar import AStar
from src.vessel.path_follower import LOSController

# Candidate headings relative to the preferred course; starboard first so
# equal-deviation ties resolve to starboard (COLREGs-leaning).
CANDIDATE_OFFSETS_DEG = [0] + [s * d for d in
                               (10, 20, 30, 45, 60, 75, 90, 120, 150, 180)
                               for s in (-1, 1)]


class VOAgent(Agent):
    name = "velocity-obstacles"

    def __init__(self, config: Config):
        self.config = config
        self.horizon = 60.0                  # VO time horizon tau (s)
        self.combined_radius = config.avoidance.safe_distance
        self.land_lookahead = 25.0           # cells of land ray-check

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": type(self).__name__,
            "family": "reactive (velocity space)",
            "author": "Fiorini & Shiller (1998); reference implementation",
            "summary": "Picks the velocity closest to the preferred course "
                       "that lies outside every obstacle's collision cone "
                       "(starboard-preferring, A* route reference).",
            "waypoints": [list(wp) for wp in self.waypoints],
        }

    # --------------------------------------------------------------- episode

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

    # ------------------------------------------------------------------ step

    def decide(self, obs: Observation) -> Decision:
        cfg = self.config
        pos = obs.position

        if obs.distance_to_goal < cfg.follower.lookahead:
            preferred = np.arctan2(obs.goal[1] - pos[1], obs.goal[0] - pos[0])
        else:
            preferred = self.follower.compute_desired_heading(
                pos, self.waypoints)
            if preferred is None:
                preferred = np.arctan2(obs.goal[1] - pos[1],
                                       obs.goal[0] - pos[0])
        preferred = float(preferred)
        speed = cfg.vessel.cruise_speed

        chosen = None
        evaluations: List[Dict[str, Any]] = []
        for off in CANDIDATE_OFFSETS_DEG:
            heading = preferred + np.radians(off)
            in_vo = self._in_any_velocity_obstacle(obs, heading, speed)
            hits_land = self._ray_hits_land(pos, heading)
            evaluations.append({"offset_deg": off, "in_vo": in_vo,
                                "hits_land": hits_land})
            if chosen is None and not in_vo and not hits_land:
                chosen = heading
                chosen_off = off

        if chosen is None:
            # Every candidate conflicts: take the velocity with the largest
            # predicted closest approach (least-bad), as in the paper's
            # escape-maneuver discussion.
            best = max(
                (h for h in evaluations if not h["hits_land"]),
                key=lambda h: self._min_dcpa(
                    obs, preferred + np.radians(h["offset_deg"]), speed),
                default=evaluations[0])
            chosen_off = best["offset_deg"]
            chosen = preferred + np.radians(chosen_off)

        explanation = {
            "mode": "vo_avoid" if chosen_off != 0 else "route",
            "preferred_heading_deg": round(np.degrees(preferred), 1),
            "chosen_offset_deg": chosen_off,
            "candidates": evaluations[:12],
        }
        # Slow down through large alterations (same seamanship convention
        # the benchmark's own baselines use).
        err = abs(np.degrees(np.arctan2(
            np.sin(chosen - obs.vessel["heading"]),
            np.cos(chosen - obs.vessel["heading"]))))
        factor = 1.0 if err <= 15 else max(0.3, 1.0 - (err - 15) / 60)
        return Decision(desired_heading=float(chosen),
                        desired_speed=float(speed * factor),
                        explanation=explanation)

    # --------------------------------------------------------------- VO test

    def _in_any_velocity_obstacle(self, obs: Observation, heading: float,
                                  speed: float) -> bool:
        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)
        for ob in obs.obstacles:
            rx = ob["x"] - obs.vessel["x"]
            ry = ob["y"] - obs.vessel["y"]
            rvx = vx - ob["speed"] * np.cos(ob["heading"])
            rvy = vy - ob["speed"] * np.sin(ob["heading"])
            v2 = rvx * rvx + rvy * rvy
            if v2 < 1e-9:
                continue
            tcpa = (rx * rvx + ry * rvy) / v2
            if not (0.0 < tcpa < self.horizon):
                continue
            dcpa = np.hypot(rx - rvx * tcpa, ry - rvy * tcpa)
            if dcpa < self.combined_radius:
                return True
        return False

    def _min_dcpa(self, obs: Observation, heading: float,
                  speed: float) -> float:
        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)
        worst = float("inf")
        for ob in obs.obstacles:
            rx = ob["x"] - obs.vessel["x"]
            ry = ob["y"] - obs.vessel["y"]
            rvx = vx - ob["speed"] * np.cos(ob["heading"])
            rvy = vy - ob["speed"] * np.sin(ob["heading"])
            v2 = rvx * rvx + rvy * rvy
            tcpa = (rx * rvx + ry * rvy) / v2 if v2 > 1e-9 else 0.0
            tcpa = float(np.clip(tcpa, 0.0, self.horizon))
            worst = min(worst, float(np.hypot(rx - rvx * tcpa,
                                              ry - rvy * tcpa)))
        return worst

    def _ray_hits_land(self, pos: Tuple[float, float],
                       heading: float) -> bool:
        dx, dy = np.cos(heading), np.sin(heading)
        for step in range(1, int(self.land_lookahead) + 1):
            x = pos[0] + dx * step
            y = pos[1] + dy * step
            if not (0 <= x < self.world.width and 0 <= y < self.world.height):
                return step < 12
            if self.world.grid[int(y), int(x)] > 0.5:
                return True
        return False
