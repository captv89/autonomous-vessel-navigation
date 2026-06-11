"""
Classical navigation agent.

Pipeline per episode:
    1. A* on the inflated grid plans a route start -> goal (with string
       pulling and segment-length capping).
    2. ILOS (or pure pursuit) tracks the route each step.
    3. CPA/TCPA risk assessment + predictive COLREGs-inspired collision
       avoidance may override the follower's heading.

Two avoidance implementations are available via config
(`avoidance.implementation`): "predictive" (default, see
src/agents/avoidance.py) and "legacy" (the original
src/vessel/collision_avoidance.py module, kept for comparison).

Every decision records which layer produced the final heading plus the full
risk picture, so episodes can be audited step by step.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from src.config import Config
from src.agents.base import Agent, Decision
from src.agents.avoidance import PredictiveAvoider
from src.sim.engine import Observation
from src.pathfinding.astar import AStar
from src.vessel.path_follower import LOSController, PurePursuitController
from src.vessel.collision_avoidance import CollisionAvoidance
from src.environment.collision_detection import CollisionDetector

logger = logging.getLogger(__name__)


class ClassicalAgent(Agent):
    name = "classical"

    def __init__(self, config: Config):
        self.config = config
        self.waypoints: List[Tuple[float, float]] = []
        self.follower = None
        self.detector = None
        self.avoider = None

    def metadata(self) -> Dict[str, Any]:
        cfg = self.config
        return {
            "name": self.name, "type": type(self).__name__,
            "planner": "A* + string pulling",
            "follower": cfg.follower.type,
            "avoidance": cfg.avoidance.implementation,
            "waypoints": [list(wp) for wp in self.waypoints],
        }

    # --------------------------------------------------------------- episode

    def reset(self, obs: Observation) -> None:
        cfg = self.config
        planner = AStar(obs.world, allow_diagonal=True,
                        safety_margin_cells=cfg.planner.safety_margin)
        start = (int(round(obs.vessel["x"])), int(round(obs.vessel["y"])))
        goal = (int(round(obs.goal[0])), int(round(obs.goal[1])))
        path = planner.find_path(
            start, goal, simplify=True,
            max_segment_length=cfg.planner.max_segment_length)
        if path is None:
            logger.warning("A* found no path %s -> %s; falling back to "
                           "direct line", start, goal)
            path = [start, goal]
        self.waypoints = [(float(x), float(y)) for x, y in path]

        if cfg.follower.type == "pure_pursuit":
            self.follower = PurePursuitController(
                lookahead_distance=cfg.follower.lookahead)
        else:
            self.follower = LOSController(
                lookahead_distance=cfg.follower.lookahead,
                path_tolerance=cfg.follower.path_tolerance,
                integral_gain=cfg.follower.integral_gain)
        self.follower.reset()

        self.detector = CollisionDetector(
            safe_distance=cfg.avoidance.detector_safe_distance,
            warning_distance=cfg.avoidance.detector_warning_distance,
            time_horizon=cfg.avoidance.time_horizon)

        if cfg.avoidance.implementation == "legacy":
            self.avoider = CollisionAvoidance(
                safe_distance=cfg.avoidance.safe_distance,
                warning_distance=cfg.avoidance.warning_distance,
                avoidance_angle=np.radians(cfg.avoidance.avoidance_angle_deg),
                grid_world=obs.world,
                vessel_K=cfg.vessel.nomoto_K,
                vessel_T=cfg.vessel.nomoto_T,
                rudder_rate=cfg.vessel.rudder_rate,
                max_rudder=cfg.vessel.max_rudder)
        else:
            self.avoider = PredictiveAvoider(cfg, obs.world)

    # ------------------------------------------------------------------ step

    def decide(self, obs: Observation) -> Decision:
        v = obs.vessel
        pos = obs.position

        # 1. Path following; inside the final lookahead, steer straight at
        # the goal so route-acceptance radii cannot cause a near miss.
        dist_to_goal = obs.distance_to_goal
        if dist_to_goal < self.config.follower.lookahead:
            desired_heading = np.arctan2(obs.goal[1] - pos[1],
                                         obs.goal[0] - pos[0])
        else:
            desired_heading = self.follower.compute_desired_heading(
                pos, self.waypoints)
            if desired_heading is None:
                desired_heading = np.arctan2(obs.goal[1] - pos[1],
                                             obs.goal[0] - pos[0])
        desired_speed = self.config.vessel.cruise_speed
        follower_heading = float(desired_heading)

        # 2. Risk assessment for every traffic vessel (logged every step)
        risks = []
        collision_infos = []
        obstacles = [{"x": ob["x"], "y": ob["y"], "heading": ob["heading"],
                      "speed": ob["speed"]} for ob in obs.obstacles]
        for ob in obs.obstacles:
            info = self.detector.assess_collision_risk(
                pos1=pos, heading1=v["heading"], speed1=v["speed"],
                pos2=(ob["x"], ob["y"]), heading2=ob["heading"],
                speed2=ob["speed"], vessel1_id=0, vessel2_id=ob["id"])
            encounter = self.detector.determine_encounter_type(
                info.relative_bearing, v["heading"], ob["heading"])
            collision_infos.append({
                "info": info, "encounter_type": encounter,
                "vessel_id": ob["id"], "obs_x": ob["x"], "obs_y": ob["y"],
                "obs_heading": ob["heading"], "obs_speed": ob["speed"]})
            risks.append({
                "vessel_id": ob["id"], "encounter": encounter,
                "cpa": round(float(info.cpa_distance), 2),
                "tcpa": round(float(info.tcpa), 1),
                "distance": round(float(info.current_distance), 2),
                "risk_level": int(info.risk_level)})

        # 3. Collision avoidance may override the follower heading
        if isinstance(self.avoider, PredictiveAvoider):
            self.avoider.current = obs.current
            desired_heading, mode, avoidance_info = self._avoid_predictive(
                obs, desired_heading)
        else:
            desired_heading, mode, avoidance_info = self._avoid_legacy(
                obs, desired_heading, obstacles, collision_infos)

        explanation = {
            "mode": mode,
            "controller": self.config.follower.type,
            "follower_heading_deg": round(np.degrees(follower_heading), 1),
            "cross_track_error": round(self._cross_track_error(pos), 2),
            "target_waypoint_index": int(getattr(
                self.follower, "current_wp_idx",
                getattr(self.follower, "current_idx", 0))),
            "n_waypoints": len(self.waypoints),
            "risks": risks,
        }
        if avoidance_info:
            explanation["avoidance"] = avoidance_info

        # 4. Slow down through large course changes (reduces tracking drift)
        # and on final approach (controlled arrival).
        speed_factor = self._turn_speed_factor(
            desired_heading - v["heading"])
        if dist_to_goal < 3.0 * self.config.simulation.goal_tolerance:
            speed_factor = min(speed_factor, max(
                0.4, dist_to_goal / (3.0 * self.config.simulation.goal_tolerance)))
        desired_speed *= speed_factor
        explanation["speed_factor"] = round(speed_factor, 2)

        return Decision(desired_heading=float(desired_heading),
                        desired_speed=float(desired_speed),
                        explanation=explanation)

    # ------------------------------------------------------ avoidance flavors

    def _make_track_steerer(self):
        """Factory producing fresh follower clones for avoidance rollouts."""
        waypoints = self.waypoints
        fcfg = self.config.follower
        follower = self.follower

        def factory():
            if isinstance(follower, PurePursuitController):
                clone = PurePursuitController(lookahead_distance=fcfg.lookahead)
                clone.current_idx = follower.current_idx
                return lambda pos: clone.compute_desired_heading(pos, waypoints)
            clone = LOSController(lookahead_distance=fcfg.lookahead,
                                  path_tolerance=fcfg.path_tolerance,
                                  integral_gain=fcfg.integral_gain)
            clone.current_wp_idx = follower.current_wp_idx
            clone.sigma = follower.sigma
            return lambda pos: clone.compute_desired_heading(pos, waypoints,
                                                             dt=0.5)
        return factory

    def _avoid_predictive(self, obs: Observation, desired_heading: float):
        decision = self.avoider.update(
            t=obs.t, vessel_state=obs.vessel, track_heading=desired_heading,
            obstacles=obs.obstacles,
            track_steerer=self._make_track_steerer())
        if not decision.active:
            return desired_heading, "path_following", None
        info = {
            "reason": decision.reason,
            "offset_deg": decision.offset_deg,
            "heading_change_deg": round(float(np.degrees(
                self._angle_diff(decision.heading, desired_heading))), 1),
            "committed_heading_deg": round(float(np.degrees(decision.heading)), 1),
            "candidates": [c.to_record() for c in decision.candidates],
        }
        return decision.heading, "avoidance", info

    def _avoid_legacy(self, obs: Observation, desired_heading: float,
                      obstacles, collision_infos):
        v = obs.vessel
        avoidance_action, returning = None, False
        if obstacles:
            avoidance_action, returning = self.avoider.get_avoidance_action(
                current_time=obs.t, our_pos=obs.position,
                our_heading=v["heading"], our_speed=v["speed"],
                current_turn_rate=v["turn_rate"],
                current_rudder=v["rudder_angle"], obstacles=obstacles,
                collision_infos=collision_infos,
                desired_heading=desired_heading)
        if avoidance_action is None:
            if returning:
                self.avoider.return_to_track_heading = None
            return desired_heading, "path_following", None
        new_heading, _ = self.avoider.apply_avoidance(
            desired_heading, self.config.vessel.cruise_speed,
            [avoidance_action])
        info = {
            "action_type": avoidance_action.type,
            "reason": avoidance_action.reason,
            "priority": int(avoidance_action.priority),
            "heading_change_deg": round(float(np.degrees(
                self._angle_diff(new_heading, desired_heading))), 1),
            "committed_heading_deg": round(float(np.degrees(
                self.avoider.committed_absolute_heading)), 1)
                if self.avoider.committed_absolute_heading is not None else None,
        }
        return new_heading, "avoidance", info

    # --------------------------------------------------------------- helpers

    def _turn_speed_factor(self, heading_error: float) -> float:
        ctl = self.config.control
        err = abs(np.degrees(self._angle_diff(heading_error, 0.0)))
        if err <= ctl.slowdown_start_deg:
            return 1.0
        frac = min(1.0, (err - ctl.slowdown_start_deg)
                   / (ctl.slowdown_full_deg - ctl.slowdown_start_deg))
        return 1.0 - frac * (1.0 - ctl.min_speed_factor)

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        return np.arctan2(np.sin(a - b), np.cos(a - b))

    def _cross_track_error(self, pos: Tuple[float, float]) -> float:
        """Signed distance from the active route segment."""
        idx = getattr(self.follower, "current_wp_idx",
                      getattr(self.follower, "current_idx", 1))
        idx = max(1, min(idx, len(self.waypoints) - 1))
        if len(self.waypoints) < 2:
            return 0.0
        (x1, y1), (x2, y2) = self.waypoints[idx - 1], self.waypoints[idx]
        seg = np.hypot(x2 - x1, y2 - y1)
        if seg < 1e-6:
            return 0.0
        chi = np.arctan2(y2 - y1, x2 - x1)
        return float(-(pos[0] - x1) * np.sin(chi) + (pos[1] - y1) * np.cos(chi))
