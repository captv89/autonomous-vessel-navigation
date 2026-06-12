"""
Sampling-based Model Predictive Control (MPC) agent.

The third approach family on the benchmark, alongside the layered
rule-based pipeline (classical) and the learned policy (rl_ppo):

- A* provides the global route (land is non-convex; every local method
  needs a global reference).
- Instead of a follower plus a separate avoidance override, ONE local
  optimal-control problem is solved each replan interval: candidate plans
  are two-phase command sequences (hold a maneuver heading/speed for a
  switch time, then steer back along the route), rolled out with the real
  vessel dynamics against constant-velocity traffic predictions, and
  scored by a single cost:

      J = - w_progress * progress along route
          + w_separation * sum of squared safety-margin violations
          + w_effort * |maneuver offset|
          + w_speed * commanded slowdown
          + w_port * port-side maneuver penalty (COLREGs prior)
          (candidates that hit land / leave the world are infeasible)

- The minimizer is executed until the next replan. The decision record
  carries the full ranked candidate table with the cost decomposition, so
  every choice is auditable like the other agents'.

This is scenario-based MPC (optimization by enumeration over a command
lattice) rather than gradient-based: with rate-limited rudder dynamics and
non-convex obstacles the lattice is robust, derivative-free, and exactly
as transparent as the benchmark requires.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import Config
from src.agents.base import Agent, Decision
from src.sim.engine import Observation
from src.pathfinding.astar import AStar
from src.vessel.dynamics import make_vessel
from src.vessel.path_follower import LOSController

logger = logging.getLogger(__name__)

# Candidate lattice: heading offsets are relative to the route bearing,
# negative = starboard (tried first: COLREGs preference).
OFFSETS_DEG = [0, -15, -30, -45, -60, -90, 15, 30, 45, 60, 90]
SPEED_FACTORS = [1.0, 0.6]
SWITCH_TIMES = [10.0, 20.0]          # s holding the maneuver before returning


@dataclass
class PlanCandidate:
    offset_deg: float
    speed_factor: float
    switch_time: float
    cost: float
    feasible: bool
    min_separation: float
    progress: float
    costs: Dict[str, float] = field(default_factory=dict)
    heading: float = 0.0

    def label(self) -> str:
        side = ("hold" if self.offset_deg == 0
                else f"stbd{abs(self.offset_deg):.0f}" if self.offset_deg < 0
                else f"port{self.offset_deg:.0f}")
        tag = f"{side}/{self.switch_time:.0f}s"
        if self.speed_factor < 1.0:
            tag += f"@{self.speed_factor:.1f}u"
        return tag

    def to_record(self) -> Dict[str, Any]:
        return {"candidate": self.label(),
                "cost": round(self.cost, 2) if np.isfinite(self.cost) else None,
                "feasible": self.feasible,
                "min_separation": (round(self.min_separation, 2)
                                   if np.isfinite(self.min_separation)
                                   else None),
                "costs": {k: round(v, 2) for k, v in self.costs.items()}}


class MPCAgent(Agent):
    name = "mpc"

    # Cost weights (per rollout; distances in cells, times in seconds)
    W_PROGRESS = 1.0          # per cell of along-route progress (reward)
    W_SEPARATION = 0.08       # per (cell of violation)^2-second; soft margin
    W_EFFORT = 0.05           # per degree of maneuver offset
    W_SPEED = 12.0            # for commanding a slowdown
    W_PORT = 3.0              # COLREGs prior against port maneuvers

    def __init__(self, config: Config):
        self.config = config
        self.waypoints: List[Tuple[float, float]] = []
        self.horizon = 45.0
        self.replan_interval = 1.0
        self.rollout_dt = 0.5
        self.land_buffer = 2.0
        self.safe_distance = config.avoidance.safe_distance

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name, "type": type(self).__name__,
            "family": "optimization (model predictive control)",
            "author": "V. Ravendranathan (VesselNav-Bench baseline)",
            "summary": "Solves a local optimal-control problem each second: "
                       "candidate maneuver plans are rolled out with the "
                       "real dynamics and scored by one cost combining "
                       "progress, separation, effort, and a COLREGs prior.",
            "planner": "A* (global) + sampling MPC (local)",
            "horizon_s": self.horizon,
            "lattice": f"{len(OFFSETS_DEG)} offsets x "
                       f"{len(SPEED_FACTORS)} speeds x "
                       f"{len(SWITCH_TIMES)} switch times",
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
        if path is None:
            logger.warning("MPC: A* found no path %s -> %s; direct line",
                           start, goal)
            path = [start, goal]
        self.waypoints = [(float(x), float(y)) for x, y in path]
        self.world = obs.world

        # Follower used for (a) the route bearing reference and (b) the
        # "return to route" phase inside rollouts.
        fcfg = cfg.follower
        self.follower = LOSController(lookahead_distance=fcfg.lookahead,
                                      path_tolerance=fcfg.path_tolerance,
                                      integral_gain=fcfg.integral_gain)
        self.follower.reset()

        self._plan: Optional[PlanCandidate] = None
        self._plan_t = -np.inf        # last (re)optimization time
        self._plan_start = -np.inf    # when the committed plan began
        self._candidates: List[PlanCandidate] = []

    # ------------------------------------------------------------------ step

    def decide(self, obs: Observation) -> Decision:
        cfg = self.config
        v = obs.vessel
        pos = obs.position

        # Route bearing from the (stateful) follower; final approach aims
        # straight at the goal like the other agents.
        if obs.distance_to_goal < cfg.follower.lookahead:
            route_heading = np.arctan2(obs.goal[1] - pos[1],
                                       obs.goal[0] - pos[0])
        else:
            route_heading = self.follower.compute_desired_heading(
                pos, self.waypoints)
            if route_heading is None:
                route_heading = np.arctan2(obs.goal[1] - pos[1],
                                           obs.goal[0] - pos[0])
        route_heading = float(route_heading)

        if obs.t - self._plan_t >= self.replan_interval:
            self._replan(obs, route_heading)

        plan = self._plan
        # Execute the committed plan. The pure-route plan tracks the
        # follower live; maneuver plans hold their heading until the
        # switch time, then return to the route.
        on_maneuver = (plan.offset_deg != 0 or plan.speed_factor < 1.0)
        if on_maneuver and obs.t - self._plan_start < plan.switch_time:
            desired_heading = plan.heading
            speed_factor = plan.speed_factor
        else:
            desired_heading = route_heading
            speed_factor = 1.0

        desired_speed = cfg.vessel.cruise_speed * speed_factor
        # Controlled arrival (same convention as the classical agent)
        if obs.distance_to_goal < 3.0 * cfg.simulation.goal_tolerance:
            desired_speed *= max(0.4, obs.distance_to_goal
                                 / (3.0 * cfg.simulation.goal_tolerance))

        explanation = {
            "mode": "mpc" if plan.offset_deg != 0 or plan.speed_factor < 1.0
                    else "route",
            "plan": plan.label(),
            "plan_cost": round(plan.cost, 2),
            "plan_age_s": round(obs.t - self._plan_t, 1),
            "route_heading_deg": round(np.degrees(route_heading), 1),
            "candidates": [c.to_record() for c in
                           sorted(self._candidates, key=lambda c: c.cost)[:8]],
        }
        return Decision(desired_heading=float(desired_heading),
                        desired_speed=float(desired_speed),
                        explanation=explanation)

    # ------------------------------------------------------------ optimizer

    KEEP_MARGIN = 1.0     # hysteresis: keep the committed plan unless a new
                          # one beats it by this much (prevents chattering).
                          # Kept small: a committed plan re-evaluated over its
                          # REMAINING hold time is naturally cheaper than
                          # full-length candidates, so a large margin makes
                          # maneuvers sticky.

    def _replan(self, obs: Observation, route_heading: float) -> None:
        self._candidates = self._optimize(obs, route_heading)
        feasible = [c for c in self._candidates if c.feasible]
        best = (min(feasible, key=lambda c: c.cost) if feasible
                else max(self._candidates, key=lambda c: c.min_separation))
        self._plan_t = obs.t

        committed = self._plan
        if committed is not None and (committed.offset_deg != 0
                                      or committed.speed_factor < 1.0):
            remaining = committed.switch_time - (obs.t - self._plan_start)
            if remaining > self.rollout_dt:
                # Re-evaluate the committed maneuver from the current state
                # (same absolute heading, remaining hold time).
                re_eval = self._evaluate(
                    obs, route_heading, committed.offset_deg,
                    committed.speed_factor, remaining,
                    maneuver_heading=committed.heading)
                if re_eval.feasible and re_eval.cost <= best.cost +                         self.KEEP_MARGIN:
                    return  # hold the committed plan
        self._plan = best
        self._plan_start = obs.t

    def _optimize(self, obs: Observation,
                  route_heading: float) -> List[PlanCandidate]:
        # Evaluate the pure-route plan first; the full maneuver lattice is
        # only worth the compute when the route itself predicts a safety
        # violation somewhere in the horizon (this is also what gives MPC
        # its early engagement: the rollout sees conflicts long before any
        # distance threshold would).
        route_plan = self._evaluate(obs, route_heading, 0, 1.0,
                                    SWITCH_TIMES[0])
        candidates: List[PlanCandidate] = [route_plan]
        route_unsafe = (not route_plan.feasible
                        or route_plan.costs.get("separation", 0.0) > 0.0)
        if not route_unsafe:
            return candidates
        for off in OFFSETS_DEG:
            for speed_factor in SPEED_FACTORS:
                for switch in SWITCH_TIMES:
                    if off == 0 and speed_factor == 1.0:
                        continue  # already evaluated as the route plan
                    candidates.append(self._evaluate(
                        obs, route_heading, off, speed_factor, switch))
        return candidates

    def _evaluate(self, obs: Observation, route_heading: float,
                  offset_deg: float, speed_factor: float,
                  switch_time: float,
                  maneuver_heading: Optional[float] = None) -> PlanCandidate:
        cfg = self.config
        v = obs.vessel
        dt = self.rollout_dt
        n = int(self.horizon / dt)
        if maneuver_heading is None:
            maneuver_heading = route_heading + np.radians(offset_deg)

        model = make_vessel(cfg, x=v["x"], y=v["y"], heading=v["heading"],
                            speed=v["speed"], current=obs.current)
        if hasattr(model, "state"):
            model.state.turn_rate = v["turn_rate"]
        else:
            model.r = v["turn_rate"]
            model.v = v.get("sway", 0.0)
        model.rudder_angle = v["rudder_angle"]

        # Return-phase steering: clone of the follower state
        clone = LOSController(lookahead_distance=cfg.follower.lookahead,
                              path_tolerance=cfg.follower.path_tolerance,
                              integral_gain=cfg.follower.integral_gain)
        clone.current_wp_idx = self.follower.current_wp_idx
        clone.sigma = self.follower.sigma

        goal = obs.goal
        goal_tol = cfg.simulation.goal_tolerance
        d0 = obs.distance_to_goal
        min_sep = float("inf")
        violation = 0.0
        feasible = True
        cruise = cfg.vessel.cruise_speed

        for i in range(1, n + 1):
            t = i * dt
            if t < switch_time:
                target = maneuver_heading
                u_cmd = cruise * speed_factor
            else:
                steered = clone.compute_desired_heading(
                    model.get_position(), self.waypoints, dt=dt)
                target = steered if steered is not None else np.arctan2(
                    goal[1] - model.get_position()[1],
                    goal[0] - model.get_position()[0])
                u_cmd = cruise
            err = target - model.get_heading()
            err = np.arctan2(np.sin(err), np.cos(err))
            rudder = (cfg.control.heading_kp * err
                      - cfg.control.heading_kd * model.get_turn_rate())
            model.update(dt, rudder_command=rudder, desired_speed=u_cmd)

            x, y = model.get_position()
            if np.hypot(goal[0] - x, goal[1] - y) < goal_tol:
                break                 # episode would end here: rollout done
            if not (0 <= x < self.world.width and 0 <= y < self.world.height):
                feasible = t > 30.0   # late exits are post-encounter noise
                break
            if self._near_land(x, y):
                feasible = False
                break
            for ob in obs.obstacles:
                ox = ob["x"] + ob["speed"] * np.cos(ob["heading"]) * t
                oy = ob["y"] + ob["speed"] * np.sin(ob["heading"]) * t
                d = float(np.hypot(ox - x, oy - y))
                min_sep = min(min_sep, d)
                if d < self.safe_distance:
                    violation += (self.safe_distance - d) ** 2 * dt
                if d < cfg.simulation.collision_radius * 2.0:
                    feasible = False

        x, y = model.get_position()
        progress = d0 - float(np.hypot(goal[0] - x, goal[1] - y))

        costs = {
            "progress": -self.W_PROGRESS * progress,
            "separation": self.W_SEPARATION * violation,
            "effort": self.W_EFFORT * abs(offset_deg),
            "speed": self.W_SPEED * (1.0 - speed_factor),
            "port": self.W_PORT if offset_deg > 0 else 0.0,
        }
        cost = sum(costs.values()) if feasible else float("inf")
        return PlanCandidate(offset_deg=offset_deg,
                             speed_factor=speed_factor,
                             switch_time=switch_time, cost=cost,
                             feasible=feasible, min_separation=min_sep,
                             progress=progress, costs=costs,
                             heading=float(maneuver_heading))

    def _near_land(self, x: float, y: float) -> bool:
        gx, gy = int(x), int(y)
        b = int(np.ceil(self.land_buffer))
        for dx in range(-b, b + 1):
            for dy in range(-b, b + 1):
                cx, cy = gx + dx, gy + dy
                if (0 <= cx < self.world.width and 0 <= cy < self.world.height
                        and self.world.grid[cy, cx] > 0.5
                        and np.hypot(dx, dy) <= self.land_buffer):
                    return True
        return False
