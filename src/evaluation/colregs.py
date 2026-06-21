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
                  domain: Optional[ShipDomain] = None) -> List[EncounterScore]:
    """Score every close-quarters encounter in one recorded episode.

    The passing-distance component is judged against `domain` (an asymmetric
    ship domain); the default circular domain reproduces the legacy
    range/safe_distance test exactly.
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
        if encounter == "head-on":
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
