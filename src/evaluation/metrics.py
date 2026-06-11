"""
Episode metrics.

All metrics are computed from recorded step data (ground truth), never from
agent self-reports, so classical and RL episodes are scored identically.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def compute_metrics(steps: List[Dict[str, Any]], outcome: str,
                    goal: tuple, near_miss_distance: float) -> Dict[str, Any]:
    """Score one episode from its recorded steps."""
    if not steps:
        return {"outcome": outcome, "success": False}

    xs = np.array([s["vessel"]["x"] for s in steps])
    ys = np.array([s["vessel"]["y"] for s in steps])
    rudders = np.array([abs(s["vessel"]["rudder_angle"]) for s in steps])
    speeds = np.array([s["vessel"]["speed"] for s in steps])

    path_length = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
    straight = float(np.hypot(goal[0] - xs[0], goal[1] - ys[0]))

    min_separation = float("inf")
    near_miss_steps = 0
    for s in steps:
        if not s["obstacles"]:
            continue
        d = min(np.hypot(o["x"] - s["vessel"]["x"], o["y"] - s["vessel"]["y"])
                for o in s["obstacles"])
        min_separation = min(min_separation, d)
        if d < near_miss_distance:
            near_miss_steps += 1

    collisions = sum(1 for s in steps
                     for e in s["events"] if e.startswith("collision"))
    groundings = sum(1 for s in steps if "grounding" in s["events"])

    metrics: Dict[str, Any] = {
        "outcome": outcome,
        "success": outcome == "goal",
        "duration_s": steps[-1]["t"],
        "n_steps": len(steps),
        "path_length": round(path_length, 1),
        "straight_line_distance": round(straight, 1),
        "path_efficiency": round(straight / path_length, 3) if path_length > 0 else 0.0,
        "min_separation": round(min_separation, 2) if min_separation != float("inf") else None,
        "near_miss_steps": near_miss_steps,
        "collisions": collisions,
        "groundings": groundings,
        "mean_speed": round(float(speeds.mean()), 2),
        "mean_abs_rudder_deg": round(float(np.degrees(rudders.mean())), 2),
        "max_abs_rudder_deg": round(float(np.degrees(rudders.max())), 2),
    }

    # Agent-specific extras derived from logged decisions (when present).
    decisions = [s.get("decision", {}) for s in steps]
    modes = [d.get("mode") for d in decisions if "mode" in d]
    if modes:
        metrics["avoidance_step_fraction"] = round(
            sum(1 for m in modes if m == "avoidance") / len(modes), 3)
    shield_flags = [d["shield"]["intervened"] for d in decisions
                    if isinstance(d.get("shield"), dict)]
    if shield_flags:
        metrics["shield_intervention_fraction"] = round(
            sum(shield_flags) / len(shield_flags), 3)
    ctes = [abs(d["cross_track_error"]) for d in decisions
            if "cross_track_error" in d]
    if ctes:
        metrics["mean_abs_cross_track"] = round(float(np.mean(ctes)), 2)
    # COLREGs tendency: of all heading-change decisions during avoidance,
    # what fraction altered course to starboard (negative change)?
    stbd, total = 0, 0
    for d in decisions:
        av = d.get("avoidance")
        if av and abs(av.get("heading_change_deg", 0.0)) > 1.0:
            total += 1
            if av["heading_change_deg"] < 0:
                stbd += 1
    if total:
        metrics["starboard_turn_fraction"] = round(stbd / total, 3)

    return metrics


def aggregate(per_episode: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a list of episode metric dicts (means over episodes)."""
    if not per_episode:
        return {}
    out: Dict[str, Any] = {
        "episodes": len(per_episode),
        "success_rate": round(
            sum(1 for m in per_episode if m.get("success")) / len(per_episode), 3),
        "collision_rate": round(
            sum(1 for m in per_episode if m.get("outcome") == "collision")
            / len(per_episode), 3),
        "grounding_rate": round(
            sum(1 for m in per_episode if m.get("outcome") == "grounding")
            / len(per_episode), 3),
        "timeout_rate": round(
            sum(1 for m in per_episode if m.get("outcome") == "timeout")
            / len(per_episode), 3),
    }
    # Route-quality metrics are only meaningful for episodes that finished
    # the route; failed episodes end early and would skew them optimistic.
    success_only = {"duration_s", "path_length", "path_efficiency"}
    for key in ("duration_s", "path_length", "path_efficiency",
                "min_separation", "near_miss_steps", "mean_abs_rudder_deg",
                "mean_abs_cross_track", "avoidance_step_fraction",
                "starboard_turn_fraction", "shield_intervention_fraction"):
        pool = ([m for m in per_episode if m.get("success")]
                if key in success_only else per_episode)
        values = [m[key] for m in pool
                  if m.get(key) is not None and key in m]
        if values:
            out[f"mean_{key}"] = round(float(np.mean(values)), 3)
    return out
