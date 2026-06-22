"""
Vessel dynamics models behind one interface.

Two models, selected by `vessel.model` in the config:

- "nomoto": the original first-order yaw model (kept for ablation studies
  and backward compatibility).
- "fossen3": a 3-DOF linear maneuvering model (surge-sway-yaw). The yaw
  channel is the same first-order Nomoto response (the classical result
  that Nomoto's model is the yaw transfer function of the linear
  sway-yaw system with N_v = 0), so K and T keep their meaning, while the
  model adds what reviewers expect of ship dynamics:
    * sway velocity / sideslip in turns (first-order toward a steady
      sideslip proportional to u*r),
    * speed loss while turning and first-order surge response (no more
      instantaneous speed changes),
    * water current as a kinematic drift and stochastic wind gust forces.

Every consumer (engine, avoidance rollouts) constructs vessels through
`make_vessel`, so simulation and prediction physics can never diverge.
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Optional, Protocol, Tuple

import numpy as np

from src.config import Config, VesselConfig
from src.vessel.vessel_model import NomotoVessel


# Maneuvering parameters jittered by hull randomization (G11). The Nomoto
# channel (K, T) is shared by both models; the rest are fossen3-only but
# scaling them when the nomoto model is selected is harmless.
HULL_RANDOM_PARAMS = ("nomoto_K", "nomoto_T", "sway_time_constant",
                      "sideslip_gain", "turn_speed_loss",
                      "surge_time_constant")


def sample_hull(vc: VesselConfig,
                rng: np.random.Generator) -> Tuple[VesselConfig, Dict[str, float]]:
    """Return a per-episode VesselConfig with jittered hull parameters.

    Each parameter in HULL_RANDOM_PARAMS is scaled by an independent factor
    drawn uniformly from [1 - hull_jitter, 1 + hull_jitter]. Deterministic
    given `rng`; the sampled values are returned for logging/inspection.
    """
    j = vc.hull_jitter
    sampled = {name: float(getattr(vc, name) * rng.uniform(1.0 - j, 1.0 + j))
               for name in HULL_RANDOM_PARAMS}
    return dataclasses.replace(vc, **sampled), sampled


class VesselDynamics(Protocol):
    def update(self, dt: float, rudder_command: float = 0.0,
               desired_speed: Optional[float] = None) -> None: ...
    def get_position(self) -> Tuple[float, float]: ...
    def get_heading(self) -> float: ...
    def get_speed(self) -> float: ...
    def get_turn_rate(self) -> float: ...
    def get_rudder_angle(self) -> float: ...
    def get_rudder_command(self) -> float: ...


class Fossen3DOFVessel:
    """Surge-sway-yaw maneuvering model with environmental disturbances.

    States: position (x, y), heading psi, body velocities (u surge,
    v sway, r yaw rate), actual rudder angle (rate-limited actuator).

    Yaw:    T * r_dot + r = K * delta            (Nomoto channel)
    Sway:   T_sway * v_dot + v = -sideslip_gain * u * r
    Surge:  T_surge * u_dot + u = u_cmd, minus turn-induced speed loss
            u_dot -= turn_speed_loss * |r| * u
    Kinematics include the water current; wind enters as a seeded gust
    acceleration so runs stay reproducible.
    """

    def __init__(self, x: float, y: float, heading: float, speed: float,
                 *, max_speed: float, K: float, T: float,
                 max_rudder: float, rudder_rate: Optional[float],
                 sway_time_constant: float = 2.0,
                 sideslip_gain: float = 1.2,
                 turn_speed_loss: float = 0.4,
                 surge_time_constant: float = 4.0,
                 current: Tuple[float, float] = (0.0, 0.0),
                 wind_gust_accel: float = 0.0,
                 rng: Optional[np.random.Generator] = None):
        self.x, self.y, self.psi = float(x), float(y), float(heading)
        self.u, self.v, self.r = float(speed), 0.0, 0.0
        self.max_speed = max_speed
        self.K, self.T = K, T
        self.max_rudder = max_rudder
        self.rudder_rate = rudder_rate
        self.T_sway = sway_time_constant
        self.k_beta = sideslip_gain
        self.c_loss = turn_speed_loss
        self.T_surge = surge_time_constant
        self.current = (float(current[0]), float(current[1]))
        self.wind_gust_accel = wind_gust_accel
        self.rng = rng or np.random.default_rng(0)

        self.rudder_command = 0.0
        self.rudder_angle = 0.0
        self._u_cmd = float(speed)

    def update(self, dt: float, rudder_command: float = 0.0,
               desired_speed: Optional[float] = None) -> None:
        if desired_speed is not None:
            self._u_cmd = float(np.clip(desired_speed, 0.0, self.max_speed))

        # Rudder actuator: saturation + slew-rate limit
        self.rudder_command = float(np.clip(rudder_command,
                                            -self.max_rudder, self.max_rudder))
        if self.rudder_rate is not None:
            max_delta = self.rudder_rate * dt
            err = self.rudder_command - self.rudder_angle
            self.rudder_angle += float(np.clip(err, -max_delta, max_delta))
        else:
            self.rudder_angle = self.rudder_command

        # Yaw (Nomoto channel)
        self.r += (self.K * self.rudder_angle - self.r) / self.T * dt

        # Sway: first-order toward steady sideslip in the turn
        v_steady = -self.k_beta * self.u * self.r
        self.v += (v_steady - self.v) / self.T_sway * dt

        # Surge: first-order toward commanded speed, with turn loss
        self.u += ((self._u_cmd - self.u) / self.T_surge
                   - self.c_loss * abs(self.r) * self.u) * dt
        self.u = float(np.clip(self.u, 0.0, self.max_speed))

        # Wind gusts: seeded random walk acceleration on body velocities
        if self.wind_gust_accel > 0.0:
            gu, gv = self.rng.normal(0.0, self.wind_gust_accel, size=2)
            self.u = float(np.clip(self.u + gu * dt, 0.0, self.max_speed))
            self.v += gv * dt

        # Kinematics (body -> world) + current drift
        self.psi += self.r * dt
        self.psi = float(np.arctan2(np.sin(self.psi), np.cos(self.psi)))
        c, s = np.cos(self.psi), np.sin(self.psi)
        self.x += (self.u * c - self.v * s + self.current[0]) * dt
        self.y += (self.u * s + self.v * c + self.current[1]) * dt

    # ------------------------------------------------------------- getters

    def get_position(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def get_heading(self) -> float:
        return self.psi

    def get_speed(self) -> float:
        return self.u

    def get_sway(self) -> float:
        return self.v

    def get_turn_rate(self) -> float:
        return self.r

    def get_rudder_angle(self) -> float:
        return self.rudder_angle

    def get_rudder_command(self) -> float:
        return self.rudder_command


def make_vessel(config: Config, *, x: float, y: float, heading: float,
                speed: float, current: Tuple[float, float] = (0.0, 0.0),
                wind_gust_accel: float = 0.0,
                rng: Optional[np.random.Generator] = None) -> VesselDynamics:
    """Single construction point for own-ship dynamics (engine + rollouts)."""
    vc = config.vessel
    if vc.model == "nomoto":
        return NomotoVessel(x=x, y=y, heading=heading, speed=speed,
                            max_speed=vc.max_speed, K=vc.nomoto_K,
                            T=vc.nomoto_T, max_rudder=vc.max_rudder,
                            rudder_rate=vc.rudder_rate)
    if vc.model == "fossen3":
        return Fossen3DOFVessel(
            x, y, heading, speed,
            max_speed=vc.max_speed, K=vc.nomoto_K, T=vc.nomoto_T,
            max_rudder=vc.max_rudder, rudder_rate=vc.rudder_rate,
            sway_time_constant=vc.sway_time_constant,
            sideslip_gain=vc.sideslip_gain,
            turn_speed_loss=vc.turn_speed_loss,
            surge_time_constant=vc.surge_time_constant,
            current=current, wind_gust_accel=wind_gust_accel, rng=rng)
    raise ValueError(f"Unknown vessel.model '{vc.model}' "
                     f"(expected 'nomoto' or 'fossen3')")
