"""
Headless simulation engine.

The engine owns the ground truth: world grid, own ship (Nomoto dynamics),
and traffic vessels. Agents receive an `Observation` snapshot each step and
return a `Decision` (desired heading + speed); the engine's PD autopilot
converts that into a rate-limited rudder command, integrates the dynamics,
moves traffic, and reports any events.

The engine is deliberately agent-agnostic and render-free so the classical
pipeline, the RL gymnasium environment, the evaluation runner, and the
pygame viewer all drive the exact same physics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import Config
from src.environment.grid_world import GridWorld
from src.environment.dynamic_obstacles import DynamicObstacleManager
from src.vessel.dynamics import make_vessel
from src.sim.reactive_traffic import steer_reactive_traffic
from src.sim.scenarios import Scenario

logger = logging.getLogger(__name__)

OUTCOME_GOAL = "goal"
OUTCOME_COLLISION = "collision"
OUTCOME_GROUNDING = "grounding"
OUTCOME_OUT_OF_BOUNDS = "out_of_bounds"
OUTCOME_TIMEOUT = "timeout"


@dataclass
class Observation:
    """Ground-truth snapshot handed to agents each step."""
    t: float
    vessel: Dict[str, float]          # x, y, heading, speed, sway, turn_rate, rudder_angle, rudder_command
    obstacles: List[Dict[str, float]] # id, x, y, heading, speed
    goal: Tuple[float, float]
    world: GridWorld                  # reference (read-only by convention)
    current: Tuple[float, float] = (0.0, 0.0)  # water current (drift estimate)

    @property
    def position(self) -> Tuple[float, float]:
        return (self.vessel["x"], self.vessel["y"])

    @property
    def distance_to_goal(self) -> float:
        return float(np.hypot(self.goal[0] - self.vessel["x"],
                              self.goal[1] - self.vessel["y"]))


@dataclass
class StepResult:
    obs: Observation                  # observation AFTER the step
    done: bool
    outcome: Optional[str]            # one of OUTCOME_* when done (except None on running)
    events: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)


class SimulationEngine:
    """Steps one navigation episode defined by a Scenario."""

    def __init__(self, config: Config, scenario: Scenario):
        self.config = config
        self.scenario = scenario
        self.world: GridWorld = scenario.make_world(config)
        self.goal = (float(scenario.goal[0]), float(scenario.goal[1]))
        self.reset()

    # ------------------------------------------------------------- lifecycle

    def reset(self) -> Observation:
        cfg = self.config
        sc = self.scenario
        self.t = 0.0
        self.steps = 0
        self.outcome: Optional[str] = None

        # Environmental disturbances: fixed from config, or drawn from the
        # scenario seed when randomize is on (deterministic per scenario).
        rng = np.random.default_rng(sc.seed if sc.seed is not None else 0)
        env = cfg.environment
        if env.randomize:
            speed = rng.uniform(0.0, env.max_random_current)
            direction = rng.uniform(-np.pi, np.pi)
            self.current = (float(speed * np.cos(direction)),
                            float(speed * np.sin(direction)))
        else:
            cur = env.current_vector()
            self.current = (float(cur[0]), float(cur[1]))

        # Depart at half cruise speed: vessels align with their route before
        # coming up to speed, which keeps the initial turn transient from
        # building large cross-track error (same condition for all agents).
        self.vessel = make_vessel(
            cfg, x=float(sc.start[0]), y=float(sc.start[1]),
            heading=np.radians(sc.start_heading_deg),
            speed=0.5 * cfg.vessel.cruise_speed,
            current=self.current, wind_gust_accel=env.wind_gust_accel,
            rng=rng)
        self.traffic: DynamicObstacleManager = sc.make_traffic()
        self.collision_count = 0
        self.grounding_count = 0
        self._colliding_ids: set = set()   # debounce per-vessel collision events
        self._grounded = False
        return self.observe()

    # ------------------------------------------------------------ observation

    def observe(self) -> Observation:
        v = self.vessel
        x, y = v.get_position()
        return Observation(
            t=self.t,
            vessel={
                "x": x, "y": y,
                "heading": v.get_heading(),
                "speed": v.get_speed(),
                "sway": getattr(v, "get_sway", lambda: 0.0)(),
                "turn_rate": v.get_turn_rate(),
                "rudder_angle": v.get_rudder_angle(),
                "rudder_command": v.get_rudder_command(),
            },
            obstacles=[{
                "id": mv.obstacle.id,
                "x": mv.obstacle.x, "y": mv.obstacle.y,
                "heading": mv.obstacle.heading,
                "speed": mv.obstacle.speed,
            } for mv in self.traffic.obstacles],
            goal=self.goal,
            world=self.world,
            current=self.current)

    # ------------------------------------------------------------------ step

    def step(self, desired_heading: float, desired_speed: float) -> StepResult:
        """Advance one timestep toward the commanded heading/speed."""
        cfg = self.config
        dt = cfg.simulation.dt

        # PD autopilot steering course-over-ground: the commanded heading
        # from agents is really a desired track direction, so steering the
        # velocity vector compensates crab angle from sideslip and current.
        # Falls back to heading when nearly stationary (course undefined).
        psi = self.vessel.get_heading()
        u = self.vessel.get_speed()
        own_x, own_y = self.vessel.get_position()
        sway = getattr(self.vessel, "get_sway", lambda: 0.0)()
        vx = u * np.cos(psi) - sway * np.sin(psi) + self.current[0]
        vy = u * np.sin(psi) + sway * np.cos(psi) + self.current[1]
        if np.hypot(vx, vy) > 0.3:
            reference = np.arctan2(vy, vx)
        else:
            reference = psi
        heading_error = desired_heading - reference
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        rudder_command = (cfg.control.heading_kp * heading_error
                          - cfg.control.heading_kd * self.vessel.get_turn_rate())

        self.vessel.update(dt, rudder_command=rudder_command,
                           desired_speed=desired_speed)
        # Reactive (give-way) targets steer off the pre-step world snapshot
        # before they integrate; non-reactive traffic is untouched.
        steer_reactive_traffic(
            self.traffic,
            {"id": -1, "x": own_x, "y": own_y, "heading": psi, "speed": u},
            cfg, dt)
        self.traffic.update_all(dt)
        # Traffic drifts with the same water current as the own ship.
        if self.current != (0.0, 0.0):
            for mv in self.traffic.obstacles:
                mv.obstacle.x += self.current[0] * dt
                mv.obstacle.y += self.current[1] * dt
        self.t += dt
        self.steps += 1

        events: List[str] = []
        obs = self.observe()
        x, y = obs.position

        # --- event detection -------------------------------------------------
        out_of_bounds = not (0 <= x < self.world.width and 0 <= y < self.world.height)
        if out_of_bounds:
            events.append("out_of_bounds")

        grounded = (not out_of_bounds
                    and self.world.grid[int(y), int(x)] > 0.5)
        if grounded and not self._grounded:
            self.grounding_count += 1
            events.append("grounding")
        self._grounded = grounded

        colliding_now = set()
        for ob in obs.obstacles:
            if np.hypot(ob["x"] - x, ob["y"] - y) < cfg.simulation.collision_radius:
                colliding_now.add(ob["id"])
                if ob["id"] not in self._colliding_ids:
                    self.collision_count += 1
                    events.append(f"collision:{ob['id']}")
        self._colliding_ids = colliding_now

        goal_reached = obs.distance_to_goal < cfg.simulation.goal_tolerance
        if goal_reached:
            events.append("goal_reached")

        # --- termination ------------------------------------------------------
        done = False
        if goal_reached:
            done, self.outcome = True, OUTCOME_GOAL
        elif out_of_bounds:
            done, self.outcome = True, OUTCOME_OUT_OF_BOUNDS
        elif grounded and cfg.simulation.terminate_on_grounding:
            done, self.outcome = True, OUTCOME_GROUNDING
        elif colliding_now and cfg.simulation.terminate_on_collision:
            done, self.outcome = True, OUTCOME_COLLISION
        elif self.t >= cfg.simulation.max_duration:
            done, self.outcome = True, OUTCOME_TIMEOUT

        return StepResult(obs=obs, done=done, outcome=self.outcome,
                          events=events,
                          info={"rudder_command": float(rudder_command)})
