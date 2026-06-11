"""
Central configuration for the simulator.

All tunable parameters live here (or in a YAML file with the same structure).
World units are grid cells; `world.cell_size` gives the real-world scale in
meters per cell. Angles are radians internally, but config files use degrees
where noted (`*_deg`).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


@dataclass
class SimulationConfig:
    dt: float = 0.1                      # integration timestep (s)
    max_duration: float = 600.0          # episode timeout (s)
    goal_tolerance: float = 5.0          # distance considered "arrived" (cells)
    collision_radius: float = 2.0        # vessel-to-vessel collision distance (cells)
    terminate_on_collision: bool = True
    terminate_on_grounding: bool = True


@dataclass
class WorldConfig:
    width: int = 100                     # grid cells
    height: int = 100
    cell_size: float = 10.0              # meters per cell (for reporting only)


@dataclass
class VesselConfig:
    model: str = "fossen3"               # "fossen3" (3-DOF) or "nomoto"
    cruise_speed: float = 2.5            # cells/s (world is 100 cells across)
    max_speed: float = 4.0
    nomoto_K: float = 0.4                # yaw gain (1/s per rad of rudder)
    nomoto_T: float = 4.0                # yaw time constant (s)
    max_rudder_deg: float = 35.0         # IMO standard
    rudder_rate_deg: float = 70.0 / 11.0 # IMO: 70 deg in 11 s
    # fossen3-only parameters
    sway_time_constant: float = 2.0      # s, sideslip build-up
    sideslip_gain: float = 1.2           # v_steady = -gain * u * r
    turn_speed_loss: float = 0.4         # u_dot -= loss * |r| * u
    surge_time_constant: float = 4.0     # s, speed response

    @property
    def max_rudder(self) -> float:
        return np.radians(self.max_rudder_deg)

    @property
    def rudder_rate(self) -> float:
        return np.radians(self.rudder_rate_deg)


@dataclass
class EnvironmentConfig:
    """Environmental disturbances (fossen3 model only)."""
    current_speed: float = 0.0           # cells/s water current
    current_direction_deg: float = 0.0   # direction the current flows TOWARD
    wind_gust_accel: float = 0.0         # std of seeded gust accel (cells/s^2)
    randomize: bool = False              # scenario-seeded random current/gusts
    max_random_current: float = 0.3      # cells/s when randomize is on

    def current_vector(self):
        rad = np.radians(self.current_direction_deg)
        return (self.current_speed * np.cos(rad),
                self.current_speed * np.sin(rad))


@dataclass
class ControlConfig:
    """PD autopilot mapping desired heading -> rudder command."""
    heading_kp: float = 1.0
    heading_kd: float = 1.5
    # Slow down in large course changes (good seamanship; reduces drift):
    # speed factor ramps from 1.0 at slowdown_start_deg to min_speed_factor
    # at slowdown_full_deg of heading error.
    slowdown_start_deg: float = 15.0
    slowdown_full_deg: float = 60.0
    min_speed_factor: float = 0.3


@dataclass
class PlannerConfig:
    safety_margin: int = 7               # obstacle inflation for A* (cells)
    max_segment_length: float = 15.0     # waypoint injection limit (cells)


@dataclass
class FollowerConfig:
    type: str = "ilos"                   # "ilos" or "pure_pursuit"
    lookahead: float = 18.0
    path_tolerance: float = 4.0
    integral_gain: float = 0.5


@dataclass
class AvoidanceConfig:
    implementation: str = "predictive"   # "predictive" (v2) or "legacy"
    safe_distance: float = 10.0          # minimum acceptable CPA (cells)
    shield_safe_distance: float = 6.0    # tighter envelope for the RL shield
    warning_distance: float = 18.0       # distance at which avoidance engages
    detector_safe_distance: float = 8.0  # CPA risk-assessment threshold
    detector_warning_distance: float = 15.0
    time_horizon: float = 60.0           # detector look-ahead (s)
    avoidance_angle_deg: float = 30.0    # default course alteration


@dataclass
class RewardConfig:
    goal_reached: float = 100.0
    collision: float = -100.0
    grounding: float = -100.0
    out_of_bounds: float = -50.0
    time_penalty: float = -0.05          # per step
    progress_scale: float = 1.0          # x (reduction in goal distance, cells)
    near_miss: float = -0.5              # per step inside detector safe_distance * 1.5
    heading_change_penalty: float = -0.02 # x |action heading change| in rad, discourages zigzag
    land_proximity: float = -0.3         # per decision, scaled by closeness inside the range below
    land_proximity_range: float = 8.0    # cells; shaping starts when land/wall is nearer than this


@dataclass
class RLConfig:
    action_repeat: int = 5               # engine steps each decision is held
    n_lidar_rays: int = 16
    lidar_range: float = 40.0            # cells
    n_tracked_obstacles: int = 3         # nearest dynamic obstacles in observation
    heading_actions_deg: List[float] = field(
        default_factory=lambda: [-30.0, -15.0, -5.0, 0.0, 5.0, 15.0, 30.0])
    reward: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class TrainingConfig:
    total_timesteps: int = 300_000
    n_envs: int = 8
    seed: int = 42
    learning_rate: float = 3.0e-4
    n_steps: int = 2048                  # PPO rollout length per env
    batch_size: int = 512
    gamma: float = 0.995
    ent_coef: float = 0.01
    scenario: str = "random"             # scenario used for training episodes
    model_dir: str = "models"
    log_dir: str = "runs"


@dataclass
class Config:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    world: WorldConfig = field(default_factory=WorldConfig)
    vessel: VesselConfig = field(default_factory=VesselConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    follower: FollowerConfig = field(default_factory=FollowerConfig)
    avoidance: AvoidanceConfig = field(default_factory=AvoidanceConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # ------------------------------------------------------------------ I/O

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        def build(dc_type, values):
            if values is None:
                return dc_type()
            kwargs = {}
            for f in dataclasses.fields(dc_type):
                if f.name not in values:
                    continue
                v = values[f.name]
                if dataclasses.is_dataclass(f.type) or f.name == "reward":
                    sub_type = {"reward": RewardConfig}.get(f.name, f.type)
                    v = build(sub_type, v)
                kwargs[f.name] = v
            return dc_type(**kwargs)

        sections = {
            "simulation": SimulationConfig, "world": WorldConfig,
            "vessel": VesselConfig, "environment": EnvironmentConfig,
            "control": ControlConfig,
            "planner": PlannerConfig, "follower": FollowerConfig,
            "avoidance": AvoidanceConfig, "rl": RLConfig,
            "training": TrainingConfig,
        }
        kwargs = {name: build(t, data.get(name)) for name, t in sections.items()}
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Load from YAML; missing keys fall back to defaults."""
        if path is None:
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
