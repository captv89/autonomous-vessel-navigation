"""Reactive (COLREGs give-way) traffic behaviour — simulator gap G1.

Historically every traffic vessel held its course no matter what, so a
"stand-on" scenario only half-tested the rule: the own ship was stood on by
a target that would never have acted anyway. This layer lets a target ship
take its own avoiding action.

A reactive target classifies its encounter with every other vessel (the own
ship and the other targets) using the project's own COLREGs classifier — the
same `CollisionDetector` the benchmark scores the own ship with — so target
behaviour is consistent with how compliance is judged. When the target is the
give-way vessel in a risky encounter it alters course to starboard until the
encounter clears, then resumes its nominal course. A stand-on target holds,
taking last-resort starboard action only in extremis (Rule 17(b)).

Compliance levels (`TrafficSpec.compliance` / `MovingVessel.compliance`):

- ``none`` / ``rogue`` — never give way (default; the historical behaviour).
- ``compliant``        — give way early (full horizon) with a clear alteration.
- ``partial``          — give way late (half horizon) with a smaller alteration.

The decision is a pure function of the pre-step world snapshot, so it is
order-independent and fully deterministic (scenario-seeded).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import Config
from src.environment.collision_detection import CollisionDetector

# Per-level tuning: how early the target reacts (fraction of the avoidance
# time horizon) and how hard it turns to starboard (degrees off its course).
_PARAMS = {
    "compliant": {"react_horizon_frac": 1.0, "alter_deg": 45.0},
    "partial":   {"react_horizon_frac": 0.5, "alter_deg": 20.0},
}
REACTIVE_LEVELS = frozenset(_PARAMS)            # levels that take action
VALID_COMPLIANCE = REACTIVE_LEVELS | {"none", "rogue"}

_EXTREMIS_ALTER_DEG = 60.0                      # last-resort starboard swing
_TARGET_TURN_RATE = np.radians(4.0)            # rad/s; smooth target turn

# Give-way encounters for the *classifying* vessel: it must keep clear when
# the other ship is reciprocal (head-on) or crossing from its own starboard.
_GIVE_WAY_ENCOUNTERS = ("head-on", "crossing-starboard")


def steer_reactive_traffic(manager, own_state: Dict[str, float],
                           cfg: Config, dt: float) -> None:
    """Adjust the heading of every reactive target in `manager` for one step.

    `own_state` is the own ship as {id, x, y, heading, speed} (id distinct
    from any target id). Non-reactive targets are left untouched, so worlds
    without reactive traffic are unaffected.
    """
    reactive = [mv for mv in manager.obstacles if mv.compliance in REACTIVE_LEVELS]
    if not reactive:
        return

    detector = CollisionDetector(
        safe_distance=cfg.avoidance.safe_distance,
        warning_distance=cfg.avoidance.safe_distance * 2.0,
        time_horizon=cfg.avoidance.time_horizon)

    # Snapshot all vessels *before* any heading change so each target decides
    # against the same world — order-independent and deterministic.
    states = [own_state] + [_state(mv) for mv in manager.obstacles]

    for mv in reactive:
        others = [s for s in states if s["id"] != mv.obstacle.id]
        desired, engaged = _decide(mv, others, cfg, detector)
        mv.giveway_engaged = engaged
        _rotate_toward(mv, desired, dt)


def _state(mv) -> Dict[str, float]:
    o = mv.obstacle
    return {"id": o.id, "x": o.x, "y": o.y, "heading": o.heading,
            "speed": o.speed}


def _decide(mv, others: List[Dict[str, float]], cfg: Config,
            detector: CollisionDetector) -> Tuple[float, bool]:
    """Return (desired absolute heading, engaged) for one reactive vessel."""
    p = _PARAMS[mv.compliance]
    safe = cfg.avoidance.safe_distance
    horizon = cfg.avoidance.time_horizon * p["react_horizon_frac"]
    coll = cfg.simulation.collision_radius

    o = mv.obstacle
    vvel = (o.speed * np.cos(o.heading), o.speed * np.sin(o.heading))

    # While already engaged, hold until the threat is genuinely past: relax
    # the CPA margin and drop the look-ahead cap so we don't disengage the
    # instant the starboard turn nudges CPA above the safe distance.
    cpa_limit = safe * (1.3 if mv.giveway_engaged else 1.0)

    give_way = extremis = False
    for s in others:
        ovel = (s["speed"] * np.cos(s["heading"]),
                s["speed"] * np.sin(s["heading"]))
        cpa, tcpa = detector.calculate_cpa_tcpa((o.x, o.y), vvel,
                                                (s["x"], s["y"]), ovel)
        dist = float(np.hypot(s["x"] - o.x, s["y"] - o.y))

        # In extremis: close aboard and still closing — act whatever the role.
        if dist < safe and 0.0 < tcpa and cpa < coll * 1.5:
            extremis = True

        within = 0.0 < tcpa < horizon if not mv.giveway_engaged else tcpa > 0.0
        if not (within and cpa < cpa_limit):
            continue
        bearing = detector.calculate_relative_bearing(
            (o.x, o.y), o.heading, (s["x"], s["y"]))
        if detector.determine_encounter_type(
                bearing, o.heading, s["heading"]) in _GIVE_WAY_ENCOUNTERS:
            give_way = True

    if give_way:
        return mv.nominal_heading - np.radians(p["alter_deg"]), True
    if extremis:
        return mv.nominal_heading - np.radians(_EXTREMIS_ALTER_DEG), True
    return mv.nominal_heading, False


def _rotate_toward(mv, desired: float, dt: float) -> None:
    err = np.arctan2(np.sin(desired - mv.obstacle.heading),
                     np.cos(desired - mv.obstacle.heading))
    step = float(np.clip(err, -_TARGET_TURN_RATE * dt, _TARGET_TURN_RATE * dt))
    mv.obstacle.heading = float(np.arctan2(
        np.sin(mv.obstacle.heading + step),
        np.cos(mv.obstacle.heading + step)))
