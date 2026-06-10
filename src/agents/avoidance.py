"""
Predictive COLREGs-inspired collision avoidance (v2).

Design:
- The "stay on route" baseline is rolled out by steering along the actual
  planned waypoints (a cloned path follower), so risk is judged against
  what the ship would really do, not a straight-line projection.
- Candidate maneuvers are absolute headings: the track heading plus a set
  of offsets, starboard first (COLREGs Rules 14/15/17 preference).
- Rollouts use the real NomotoVessel model and the same PD autopilot the
  engine uses, so prediction physics never drift from simulation physics.
- A chosen maneuver is kept (hysteresis) while it still predicts a safe
  passage; track is resumed once the route itself is predicted safe again.
- Every candidate evaluation is included in the decision explanation, with
  predicted miss distance and rejection reason, so each decision is fully
  auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.config import Config
from src.environment.grid_world import GridWorld
from src.vessel.vessel_model import NomotoVessel

# Steering callback: position -> desired heading (None = keep last)
SteerFn = Callable[[Tuple[float, float]], Optional[float]]

# Candidate course offsets in degrees; negative = starboard (heading 0=East,
# positive angles counter-clockwise). Starboard candidates come first so
# they win ties, encoding the COLREGs preference.
DEFAULT_OFFSETS_DEG = [-15, -30, -45, -60, -90, 15, 30, 45, 60, 90]


@dataclass
class CandidateResult:
    label: str
    min_separation: float
    hits_land: bool
    leaves_world: bool
    safe: bool
    heading: Optional[float] = None       # None for the track candidate
    offset_deg: Optional[float] = None
    reject_reason: str = ""

    def to_record(self) -> Dict[str, Any]:
        return {
            "candidate": self.label,
            "min_separation": (round(self.min_separation, 2)
                               if np.isfinite(self.min_separation) else None),
            "safe": self.safe,
            "reject_reason": self.reject_reason,
        }


@dataclass
class AvoidanceDecision:
    active: bool
    heading: Optional[float] = None       # absolute commanded heading when active
    offset_deg: Optional[float] = None
    reason: str = ""
    candidates: List[CandidateResult] = field(default_factory=list)


class PredictiveAvoider:
    """Stateful collision avoidance layered over a path follower."""

    def __init__(self, config: Config, world: GridWorld,
                 offsets_deg: Optional[List[float]] = None):
        self.config = config
        self.world = world
        self.offsets_deg = offsets_deg or list(DEFAULT_OFFSETS_DEG)
        av = config.avoidance
        self.safe_distance = av.safe_distance
        self.horizon = av.time_horizon              # rollout length (s)
        self.rollout_dt = 0.5
        self.replan_interval = 1.0                  # s between re-evaluations
        self.land_buffer = 2.0                      # cells of clearance to land
        self.reset()

    def reset(self) -> None:
        self.committed_heading: Optional[float] = None
        self.committed_offset: Optional[float] = None
        self.committed_reason = ""
        self._last_plan_t = -np.inf
        self._last_candidates: List[CandidateResult] = []

    @property
    def active(self) -> bool:
        return self.committed_heading is not None

    # ------------------------------------------------------------------ API

    def update(self, t: float, vessel_state: Dict[str, float],
               track_heading: float, obstacles: List[Dict[str, float]],
               track_steerer: Optional[Callable[[], SteerFn]] = None
               ) -> AvoidanceDecision:
        """Decide whether to override the follower this step.

        track_steerer: zero-arg factory returning a fresh steering function
        that follows the planned route (used to roll out the baseline).
        Falls back to constant `track_heading` when not provided.
        """
        if not obstacles:
            self.reset()
            return AvoidanceDecision(active=False, reason="no traffic")

        if t - self._last_plan_t < self.replan_interval:
            return self._hold()
        self._last_plan_t = t

        # Out-of-bounds matters only on the encounter timescale: a candidate
        # may exit the (finite) world long after traffic is cleared, because
        # by then the ship will be back on route.
        max_tcpa = self._max_positive_tcpa(vessel_state, obstacles)
        oob_grace = float(np.clip(max_tcpa + 10.0, 15.0, self.horizon))

        def track_rollout() -> CandidateResult:
            steer = track_steerer() if track_steerer else None
            result = self._rollout(vessel_state, obstacles,
                                   target_heading=track_heading,
                                   steer_fn=steer, oob_grace=oob_grace)
            result.label = "track"
            return result

        track_result = track_rollout()

        if not self.active:
            if track_result.min_separation >= self.safe_distance:
                return AvoidanceDecision(
                    active=False,
                    reason=f"route safe (predicted miss "
                           f"{track_result.min_separation:.1f})",
                    candidates=[track_result])
            return self._plan(vessel_state, track_heading, obstacles,
                              track_result, oob_grace,
                              trigger=f"route miss "
                                      f"{track_result.min_separation:.1f} < "
                                      f"{self.safe_distance:.1f}")

        # Currently committed: resume the route as soon as it is safe again
        # (separation AND land both clear, since we may be well off-path).
        if track_result.safe:
            self.reset()
            self._last_plan_t = t
            return AvoidanceDecision(
                active=False, reason="route predicted safe; resuming",
                candidates=[track_result])

        # Still avoiding: is the committed maneuver still good?
        committed = self._rollout(vessel_state, obstacles,
                                  target_heading=self.committed_heading,
                                  oob_grace=oob_grace)
        if committed.safe:
            return self._hold()
        return self._plan(vessel_state, track_heading, obstacles,
                          track_result, oob_grace,
                          trigger="committed maneuver no longer safe")

    # ------------------------------------------------------------- internals

    def _hold(self) -> AvoidanceDecision:
        if not self.active:
            return AvoidanceDecision(active=False, reason="holding: on track")
        return AvoidanceDecision(
            active=True, heading=self.committed_heading,
            offset_deg=self.committed_offset,
            reason=self.committed_reason,
            candidates=self._last_candidates)

    def _max_positive_tcpa(self, vs: Dict[str, float],
                           obstacles: List[Dict[str, float]]) -> float:
        vx = vs["speed"] * np.cos(vs["heading"])
        vy = vs["speed"] * np.sin(vs["heading"])
        worst = 0.0
        for ob in obstacles:
            rx, ry = ob["x"] - vs["x"], ob["y"] - vs["y"]
            rvx = ob["speed"] * np.cos(ob["heading"]) - vx
            rvy = ob["speed"] * np.sin(ob["heading"]) - vy
            v2 = rvx * rvx + rvy * rvy
            if v2 > 1e-9:
                worst = max(worst, -(rx * rvx + ry * rvy) / v2)
        return worst

    def _plan(self, vs: Dict[str, float], track_heading: float,
              obstacles: List[Dict[str, float]],
              track_result: CandidateResult, oob_grace: float,
              trigger: str) -> AvoidanceDecision:
        results: List[CandidateResult] = [track_result]
        chosen: Optional[CandidateResult] = None
        for off in self.offsets_deg:
            heading = track_heading + np.radians(off)
            result = self._rollout(vs, obstacles, target_heading=heading,
                                   oob_grace=oob_grace)
            result.offset_deg = off
            side = "stbd" if off < 0 else "port"
            result.label = f"{side}{abs(off):.0f}"
            results.append(result)
            if chosen is None and result.safe:
                chosen = result
        self._last_candidates = results

        if chosen is None:
            # Nothing predicted fully safe: take the largest predicted miss
            # among candidates that stay in the world and off the land.
            pool = [r for r in results[1:]
                    if not r.hits_land and not r.leaves_world] or results[1:]
            chosen = max(pool, key=lambda r: r.min_separation)
            reason = (f"EMERGENCY ({trigger}): no safe candidate, best is "
                      f"{chosen.label} miss {chosen.min_separation:.1f}")
        else:
            reason = (f"avoid ({trigger}): {chosen.label}, predicted miss "
                      f"{chosen.min_separation:.1f}")

        self.committed_heading = chosen.heading
        self.committed_offset = chosen.offset_deg
        self.committed_reason = reason
        return AvoidanceDecision(active=True, heading=chosen.heading,
                                 offset_deg=chosen.offset_deg, reason=reason,
                                 candidates=results)

    def _rollout(self, vs: Dict[str, float],
                 obstacles: List[Dict[str, float]],
                 target_heading: float,
                 steer_fn: Optional[SteerFn] = None,
                 oob_grace: float = 40.0) -> CandidateResult:
        """Simulate a candidate and report the predicted outcome."""
        cfg = self.config
        dt = self.rollout_dt
        n = int(self.horizon / dt)

        model = NomotoVessel(
            x=vs["x"], y=vs["y"], heading=vs["heading"], speed=vs["speed"],
            max_speed=cfg.vessel.max_speed, K=cfg.vessel.nomoto_K,
            T=cfg.vessel.nomoto_T, max_rudder=cfg.vessel.max_rudder,
            rudder_rate=cfg.vessel.rudder_rate)
        model.state.turn_rate = vs.get("turn_rate", 0.0)
        model.rudder_angle = vs.get("rudder_angle", 0.0)

        target = target_heading
        min_sep = float("inf")
        hits_land = leaves_world = False
        for i in range(1, n + 1):
            if steer_fn is not None:
                steered = steer_fn(model.get_position())
                if steered is not None:
                    target = steered
            err = target - model.get_heading()
            err = np.arctan2(np.sin(err), np.cos(err))
            rudder = (cfg.control.heading_kp * err
                      - cfg.control.heading_kd * model.get_turn_rate())
            model.update(dt, rudder_command=rudder)
            x, y = model.get_position()
            t = i * dt

            if not (0 <= x < self.world.width and 0 <= y < self.world.height):
                leaves_world = t < oob_grace
                break
            if self._near_land(x, y):
                hits_land = True
                break
            for ob in obstacles:
                ox = ob["x"] + ob["speed"] * np.cos(ob["heading"]) * t
                oy = ob["y"] + ob["speed"] * np.sin(ob["heading"]) * t
                min_sep = min(min_sep, float(np.hypot(ox - x, oy - y)))

        safe = (not hits_land and not leaves_world
                and min_sep >= self.safe_distance)
        reject = ("hits land" if hits_land
                  else "leaves world" if leaves_world
                  else f"miss {min_sep:.1f} < {self.safe_distance:.1f}"
                  if min_sep < self.safe_distance else "")
        return CandidateResult(label="", heading=target_heading,
                               min_separation=min_sep, hits_land=hits_land,
                               leaves_world=leaves_world, safe=safe,
                               reject_reason=reject)

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
