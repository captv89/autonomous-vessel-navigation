"""
Phase 1 vs Phase 2 physics comparison figure for the blog/paper.

Drives the *same* commanded maneuver — a straight run into a hard 90 deg
starboard turn — through both own-ship dynamics models on the shared
SimulationEngine, then plots the two (x, y) traces on one axis:

    Nomoto-only (Phase 1)   first-order yaw only, no sideslip/speed loss
    Fossen 3-DOF (Phase 2)  surge-sway-yaw: drifts outward (sideslip) and
                            loses speed in the turn, so the track bows out

Both runs use identical control (the engine's PD course-keeping autopilot),
the identical heading/speed command schedule, and the same start state — the
only thing that changes is `vessel.model`, so the difference on the plot is
purely the physics upgrade.

Usage:
    python scripts/make_physics_comparison.py [--out animations/physics_comparison.png]
"""

from __future__ import annotations

# Allow running as `python scripts/<name>.py` from the repo root
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import Config
from src.sim.engine import SimulationEngine
from src.sim.scenarios import Scenario

# palette (matches the blog spec): old/baseline cool blue, new in orange
NOMOTO = "#1f3a5f"
FOSSEN = "#d97706"

STRAIGHT_S = 20.0          # run straight (lets Fossen surge settle to cruise)
TURN_DEG = -90.0           # hard starboard alteration (course over ground)
DURATION_S = 70.0          # total simulated time to trace


def run_model(model: str):
    """Trace the scripted maneuver under one dynamics model.

    Returns (xs, ys, times, speeds).
    """
    config = Config()
    config.vessel.model = model
    # A blank, large-clearance scenario: we only want the open-water turn, so
    # place the goal far away (never reached during the trace) and start with
    # plenty of room for the post-turn legs to separate.
    scenario = Scenario(
        name="physics_demo",
        start=(20.0, 85.0), start_heading_deg=0.0, goal=(99.0, 1.0),
        description="Scripted hard-turn maneuver for the physics comparison.")
    engine = SimulationEngine(config, scenario)
    engine.reset()

    cruise = config.vessel.cruise_speed
    turn_rad = np.radians(TURN_DEG)
    xs, ys, ts, spd = [], [], [], []
    n_steps = int(DURATION_S / config.simulation.dt)
    for i in range(n_steps):
        t = i * config.simulation.dt
        desired_heading = 0.0 if t < STRAIGHT_S else turn_rad
        result = engine.step(desired_heading, cruise)   # done is ignored: we
        obs = result.obs                                #  trace a fixed window
        xs.append(obs.vessel["x"])
        ys.append(obs.vessel["y"])
        ts.append(t)
        spd.append(obs.vessel["speed"])
    return xs, ys, ts, spd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="animations/physics_comparison.png")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    nx, ny, nt, ns = run_model("nomoto")
    fx, fy, ft, fs = run_model("fossen3")
    turn_idx = int(STRAIGHT_S / Config().simulation.dt)

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.45, 1.0]})

    # --- left: ground track --------------------------------------------------
    ax.plot(nx, ny, color=NOMOTO, linewidth=2.6, label="Nomoto-only (Phase 1)")
    ax.plot(fx, fy, color=FOSSEN, linewidth=2.6, label="Fossen 3-DOF (Phase 2)")
    ax.plot(nx[0], ny[0], marker="o", color="#2c3e50", markersize=8, zorder=5)
    ax.annotate("start", (nx[0], ny[0]), textcoords="offset points",
                xytext=(8, 8), fontsize=10, color="#2c3e50")
    ax.plot(nx[turn_idx], ny[turn_idx], marker="s", color="#7f8c8d",
            markersize=7, zorder=5)
    ax.annotate("hard starboard\nordered here",
                (nx[turn_idx], ny[turn_idx]), textcoords="offset points",
                xytext=(8, -30), fontsize=9, color="#566573")
    ax.set_aspect("equal")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")
    ax.set_title("Ground track through a hard 90° turn", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=9)

    # --- right: speed over time (the turn-induced loss Fossen adds) ----------
    ax2.plot(nt, ns, color=NOMOTO, linewidth=2.4, label="Nomoto-only (Phase 1)")
    ax2.plot(ft, fs, color=FOSSEN, linewidth=2.4, label="Fossen 3-DOF (Phase 2)")
    ax2.axvline(STRAIGHT_S, color="#7f8c8d", linestyle="--", linewidth=1.2)
    ax2.annotate("turn ordered", (STRAIGHT_S, ax2.get_ylim()[0]),
                 textcoords="offset points", xytext=(4, 6), fontsize=9,
                 color="#566573")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("speed (cells/s)")
    ax2.set_title("Speed: Fossen loses way in the turn", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.4)
    ax2.legend(loc="lower right", framealpha=0.95, fontsize=9)

    fig.suptitle("Same command, two physics models — Nomoto holds a clean rail; "
                 "Fossen 3-DOF adds sideslip + speed loss",
                 fontsize=12.5, y=1.0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
