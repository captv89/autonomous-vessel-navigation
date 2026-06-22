"""
COLREGs compliance scoring from recorded episodes.

A simplified, quantitative rule-adherence score in the spirit of Woerner et
al. (2019), computed purely from ground-truth state in the episode logs so
it applies identically to any agent.

For every traffic vessel we find the encounter onset (first step where the
straight-line CPA prediction falls below the safety threshold with positive
TCPA), classify the encounter geometry at onset, then score the own ship's
behavior over the encounter window:

- head-on (Rule 14):        initial turn to starboard + safe passing distance
- crossing, give-way        early action + initial turn to starboard +
  (Rule 15/16):             safe passing distance
- crossing, stand-on        hold course while it remains safe to do so
  (Rule 17):                (maneuvering is not penalized once unsafe)
- overtaking (Rule 13):     safe passing distance (either side permitted)

Each encounter scores in [0, 1]; a collision with that vessel scores 0.
The limitations are documented in the report: constant-velocity CPA at
onset, no multi-vessel rule interaction, simplified Rule 17.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import ShipDomain
from src.environment.collision_detection import CollisionDetector
from src.sim.ship_domain import domain_distance


@dataclass
class EncounterScore:
    vessel_id: int
    encounter: str                    # geometry class at onset
    role: str                         # give_way | stand_on | any
    onset_t: float
    min_separation: float
    initial_turn: Optional[str]       # starboard | port | None
    score: float
    components: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"vessel_id": self.vessel_id, "encounter": self.encounter,
                "role": self.role, "onset_t": round(self.onset_t, 1),
                "min_separation": round(self.min_separation, 2),
                "initial_turn": self.initial_turn,
                "score": round(self.score, 3),
                "components": {k: round(v, 3)
                               for k, v in self.components.items()}}


def _cpa_tcpa(own, ob) -> tuple:
    vx = own["speed"] * np.cos(own["heading"])
    vy = own["speed"] * np.sin(own["heading"])
    rvx = ob["speed"] * np.cos(ob["heading"]) - vx
    rvy = ob["speed"] * np.sin(ob["heading"]) - vy
    rx, ry = ob["x"] - own["x"], ob["y"] - own["y"]
    v2 = rvx * rvx + rvy * rvy
    if v2 < 1e-9:
        return float(np.hypot(rx, ry)), 0.0
    tcpa = -(rx * rvx + ry * rvy) / v2
    cpa = float(np.hypot(rx + rvx * max(tcpa, 0.0),
                         ry + rvy * max(tcpa, 0.0)))
    return cpa, float(tcpa)


def _heading_series_turn(steps: List[dict], start: int, end: int,
                         threshold_deg: float = 10.0) -> Optional[str]:
    """Direction of the first significant course change after onset."""
    base = steps[start]["vessel"]["heading"]
    for s in steps[start:end]:
        diff = s["vessel"]["heading"] - base
        diff = np.arctan2(np.sin(diff), np.cos(diff))
        if abs(np.degrees(diff)) >= threshold_deg:
            return "starboard" if diff < 0 else "port"
    return None


def score_episode(steps: List[dict], safe_distance: float,
                  collision_radius: float,
                  domain: Optional[ShipDomain] = None,
                  restricted_visibility: bool = False) -> List[EncounterScore]:
    """Score every close-quarters encounter in one recorded episode.

    The passing-distance component is judged against `domain` (an asymmetric
    ship domain); the default circular domain reproduces the legacy
    range/safe_distance test exactly.

    When `restricted_visibility` is set, every encounter is scored under
    Rule 19 instead of Rules 14-17 (gap G10): there are no give-way/stand-on
    roles; compliance rewards action in ample time and, for a vessel forward
    of the beam, avoiding an alteration to port (Rule 19(d)(i)).
    """
    if not steps:
        return []
    if domain is None:
        domain = ShipDomain()
    detector = CollisionDetector(safe_distance=safe_distance,
                                 warning_distance=safe_distance * 2)
    vessel_ids = sorted({ob["id"] for s in steps for ob in s["obstacles"]})
    out: List[EncounterScore] = []

    for vid in vessel_ids:
        series = [(i, s, next((ob for ob in s["obstacles"]
                               if ob["id"] == vid), None))
                  for i, s in enumerate(steps)]
        series = [(i, s, ob) for i, s, ob in series if ob is not None]
        if not series:
            continue

        # Encounter onset: predicted CPA below threshold, closing
        onset = None
        for i, s, ob in series:
            cpa, tcpa = _cpa_tcpa(s["vessel"], ob)
            if cpa < safe_distance and 0.0 < tcpa < 120.0:
                onset = (i, s, ob, tcpa)
                break
        if onset is None:
            continue
        onset_i, onset_s, onset_ob, onset_tcpa = onset

        # Window: onset until actual closest approach (+ a short tail)
        distances = [float(np.hypot(ob["x"] - s["vessel"]["x"],
                                    ob["y"] - s["vessel"]["y"]))
                     for _, s, ob in series]
        min_idx = int(np.argmin(distances))
        min_sep = distances[min_idx]
        end_i = min(len(steps) - 1,
                    series[min(min_idx, len(series) - 1)][0] + 50)

        own = onset_s["vessel"]
        bearing = detector.calculate_relative_bearing(
            (own["x"], own["y"]), own["heading"],
            (onset_ob["x"], onset_ob["y"]))
        encounter = detector.determine_encounter_type(
            bearing, own["heading"], onset_ob["heading"])

        turn = _heading_series_turn(steps, onset_i, end_i)
        collided = min_sep < collision_radius
        # Passing-distance score uses the deepest ship-domain intrusion over
        # the encounter (a close target ahead counts worse than one astern).
        # Circular domain => min_sep / safe_distance, the legacy behavior.
        sep_score = float(np.clip(
            min(domain_distance(s["vessel"]["x"], s["vessel"]["y"],
                                s["vessel"]["heading"], ob["x"], ob["y"],
                                safe_distance, domain)
                for _, s, ob in series), 0.0, 1.0))

        comp: Dict[str, float] = {}
        if restricted_visibility:
            # Rule 19: in restricted visibility the give-way/stand-on roles do
            # not apply; the test is action in ample time and, for a target
            # forward of the beam, avoiding an alteration to port.
            role = "any"
            forward_of_beam = abs(bearing) <= np.pi / 2
            early_end = onset_i + max(
                1, int(0.5 * onset_tcpa
                       / max(steps[1]["t"] - steps[0]["t"], 1e-6)))
            early_turn = _heading_series_turn(
                steps, onset_i, min(early_end, end_i))
            comp["early_action"] = 1.0 if early_turn is not None else 0.0
            comp["safe_distance"] = sep_score
            if forward_of_beam:
                # Rule 19(d)(i): avoid altering to port for a vessel forward of
                # the beam (a starboard alteration, or holding, is compliant).
                comp["avoid_port_turn"] = 0.0 if turn == "port" else 1.0
                score = (0.3 * comp["early_action"]
                         + 0.3 * comp["avoid_port_turn"]
                         + 0.4 * comp["safe_distance"])
            else:
                score = (0.4 * comp["early_action"]
                         + 0.6 * comp["safe_distance"])
        elif encounter == "head-on":
            role = "give_way"
            comp["starboard_turn"] = 1.0 if turn == "starboard" else 0.0
            comp["safe_distance"] = sep_score
            score = 0.5 * comp["starboard_turn"] + 0.5 * comp["safe_distance"]
        elif encounter == "crossing-starboard":
            role = "give_way"
            # Early action: significant turn before half the onset TCPA elapsed
            early_end = onset_i + max(
                1, int(0.5 * onset_tcpa
                       / max(steps[1]["t"] - steps[0]["t"], 1e-6)))
            early_turn = _heading_series_turn(
                steps, onset_i, min(early_end, end_i))
            comp["early_action"] = 1.0 if early_turn is not None else 0.0
            comp["starboard_turn"] = 1.0 if turn == "starboard" else 0.0
            comp["safe_distance"] = sep_score
            score = (0.3 * comp["early_action"]
                     + 0.3 * comp["starboard_turn"]
                     + 0.4 * comp["safe_distance"])
        elif encounter == "crossing-port":
            role = "stand_on"
            # Hold course while safe; once separation is compromised,
            # maneuvering is permitted (Rule 17(a)(ii)) so only score the
            # held-course component when the passage stayed safe.
            held = turn is None
            comp["safe_distance"] = sep_score
            if min_sep >= safe_distance * 0.8:
                comp["held_course"] = 1.0 if held else 0.4
                score = (0.4 * comp["held_course"]
                         + 0.6 * comp["safe_distance"])
            else:
                score = sep_score
        elif encounter == "overtaking":
            role = "give_way"
            comp["safe_distance"] = sep_score
            score = sep_score
        else:
            role = "any"
            comp["safe_distance"] = sep_score
            score = sep_score

        if collided:
            score = 0.0
        out.append(EncounterScore(
            vessel_id=vid, encounter=encounter, role=role,
            onset_t=steps[onset_i]["t"], min_separation=min_sep,
            initial_turn=turn, score=float(score), components=comp))
    return out


# Rule 10 (traffic separation scheme) scoring, gap G9.
_TSS_FLOW_TOL_DEG = 90.0     # within this of a lane's flow counts as "with it"
_TSS_ZONE_CAP = 0.30         # fraction of episode time in the zone scoring 0


def _tss_lane_flow(tss, x: float) -> Optional[float]:
    for x0, x1, flow_deg in tss.lanes:
        if x0 <= x <= x1:
            return flow_deg
    return None


def score_tss(steps: List[dict], tss) -> Dict[str, float]:
    """Episode-level Rule 10 (traffic separation scheme) compliance.

    The own ship's task is inferred from its net displacement relative to the
    lane axis: motion *along* the axis is a transit (scored on proceeding with
    the lane flow and keeping clear of the separation zone); motion *across* it
    is a crossing (scored on crossing near right angles, Rule 10(c), and not
    lingering in the zone). Components and the combined `score` are in [0, 1].
    Returns {} when there is no scheme or no track.
    """
    if tss is None or not steps:
        return {}
    xs = np.array([s["vessel"]["x"] for s in steps])
    ys = np.array([s["vessel"]["y"] for s in steps])
    course = np.arctan2(ys[-1] - ys[0], xs[-1] - xs[0])
    axis = np.radians(tss.axis_deg)
    rel = abs(np.arctan2(np.sin(course - axis), np.cos(course - axis)))
    rel = min(rel, np.pi - rel)                  # 0 = along axis, pi/2 = across

    z0, z1 = tss.zone
    frac_zone = float(np.mean((xs >= z0) & (xs <= z1)))

    comp: Dict[str, float] = {}
    if rel > np.pi / 4:                          # crossing the scheme
        # A perpendicular crossing must pass through the zone; only penalize
        # time beyond a direct crossing (lingering / paralleling the lanes).
        span = abs(xs[-1] - xs[0])               # across-axis distance covered
        expected = (z1 - z0) / span if span > 1e-6 else 0.0
        excess = max(0.0, frac_zone - expected)
        zone_clear = float(np.clip(1.0 - excess / _TSS_ZONE_CAP, 0.0, 1.0))
        crossing_angle = float(np.clip(np.degrees(rel) / 90.0, 0.0, 1.0))
        comp["crossing_angle"] = round(crossing_angle, 3)
        comp["zone_clear"] = round(zone_clear, 3)
        comp["score"] = round(0.6 * crossing_angle + 0.4 * zone_clear, 3)
    else:                                        # transiting a lane
        # A transit should never enter the zone: any time in it is penalized.
        zone_clear = float(np.clip(1.0 - frac_zone / _TSS_ZONE_CAP, 0.0, 1.0))
        comp["zone_clear"] = round(zone_clear, 3)
        with_flow_n = total = 0
        for s in steps:
            v = s["vessel"]
            flow = _tss_lane_flow(tss, v["x"])
            if flow is None or v["speed"] <= 1e-3:
                continue
            total += 1
            d = np.arctan2(np.sin(v["heading"] - np.radians(flow)),
                           np.cos(v["heading"] - np.radians(flow)))
            if abs(np.degrees(d)) <= _TSS_FLOW_TOL_DEG:
                with_flow_n += 1
        with_flow = with_flow_n / total if total else 1.0
        comp["with_flow"] = round(with_flow, 3)
        comp["score"] = round(0.6 * with_flow + 0.4 * zone_clear, 3)
    return comp


def aggregate_tss(per_episode: List[Dict[str, float]]) -> Dict[str, Any]:
    """Mean Rule 10 score over the episodes that ran under a TSS."""
    scored = [r for r in per_episode if r]
    if not scored:
        return {"episodes": 0}
    return {"episodes": len(scored),
            "mean_score": round(
                float(np.mean([r["score"] for r in scored])), 3)}


def aggregate_compliance(per_episode: List[List[EncounterScore]]
                         ) -> Dict[str, Any]:
    """Mean compliance overall and per encounter type."""
    flat = [e for ep in per_episode for e in ep]
    if not flat:
        return {"encounters": 0}
    by_type: Dict[str, List[float]] = {}
    for e in flat:
        by_type.setdefault(e.encounter, []).append(e.score)
    return {
        "encounters": len(flat),
        "mean_score": round(float(np.mean([e.score for e in flat])), 3),
        "by_encounter": {k: {"n": len(v),
                             "mean_score": round(float(np.mean(v)), 3)}
                         for k, v in sorted(by_type.items())},
    }
