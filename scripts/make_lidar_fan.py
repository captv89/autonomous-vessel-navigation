"""
Lidar-fan observation figure for the blog/paper.

Navigates a scenario with the classical agent on the shared engine, then
renders one frame where the own ship is near land: the vessel, the coastline,
and all 16 lidar rays the RL observation actually casts (the same
`ObservationBuilder._lidar` used in training), drawn from the ship out to each
ray's measured range. Short rays (land nearby) are warm/red, long rays (open
water) are cool/blue, so the contrast that feeds the policy is visible.

Usage:
    python scripts/make_lidar_fan.py [--scenario coastal] [--seed 1000]
                                     [--out animations/lidar_fan.png]
"""

from __future__ import annotations

# Allow running as `python scripts/<name>.py` from the repo root
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib import cm
from matplotlib.colors import Normalize

from src.config import Config
from src.sim.engine import SimulationEngine
from src.sim.scenarios import build_scenario
from src.rl.observation import ObservationBuilder
from src.evaluation.benchmark import make_agent_factory

OCEAN = "#E8F4FD"
LAND = "#8B7355"
LAND_EDGE = "#6E5A43"
OWN = "#1F4E79"


def best_frame(config: Config, scenario_name: str, seed: int):
    """Run the classical agent and return the most lidar-informative frame.

    Picks the step with the widest spread between its shortest and longest ray
    (i.e. land close on one side, open water on another) — the frame that best
    shows what the lidar block encodes.
    """
    scenario = build_scenario(scenario_name, seed=seed)
    engine = SimulationEngine(config, scenario)
    builder = ObservationBuilder(config)
    agent = make_agent_factory("classical", config)()

    obs = engine.reset()
    agent.reset(obs)
    best = None
    best_spread = -1.0
    for _ in range(int(config.simulation.max_duration / config.simulation.dt)):
        decision = agent.decide(obs)
        result = engine.step(decision.desired_heading, decision.desired_speed)
        obs = result.obs
        v = obs.vessel
        rays = np.asarray(builder._lidar(obs.world, v["x"], v["y"],
                                         v["heading"]))
        spread = float(rays.max() - rays.min())
        # Require some genuinely short rays so the frame shows land proximity.
        if rays.min() < 0.5 and spread > best_spread:
            best_spread = spread
            best = (v["x"], v["y"], v["heading"], rays, obs.world)
        if result.done:
            break
    if best is None:
        raise SystemExit(f"no near-land frame found for {scenario_name}")
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="coastal")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", default="animations/lidar_fan.png")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    config = Config()
    x, y, heading, rays, world = best_frame(config, args.scenario, args.seed)
    n_rays = config.rl.n_lidar_rays
    lidar_range = config.rl.lidar_range

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor(OCEAN)
    ax.set_xlim(0, world.width)
    ax.set_ylim(0, world.height)
    ax.set_aspect("equal")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("y (cells)")

    # land
    for rx, ry, rw, rh in getattr(world, "rect_obstacles", []) or []:
        ax.add_patch(Rectangle((rx, ry), rw, rh, facecolor=LAND,
                               edgecolor=LAND_EDGE, linewidth=1.0, zorder=2))
    # the engine rasterizes obstacles into world.grid; draw that as the
    # ground truth the rays are actually cast against (covers any shape)
    ax.imshow(np.ma.masked_less(world.grid, 0.5), origin="lower",
              extent=(0, world.width, 0, world.height),
              cmap="copper_r", alpha=0.85, zorder=2,
              interpolation="nearest")

    # lidar rays, colored by measured range (short = red, long = blue)
    cmap = plt.get_cmap("RdYlBu")
    norm = Normalize(vmin=0.0, vmax=1.0)
    for i in range(n_rays):
        ang = heading + 2.0 * np.pi * i / n_rays
        d = float(rays[i]) * lidar_range
        ex, ey = x + np.cos(ang) * d, y + np.sin(ang) * d
        color = cmap(norm(float(rays[i])))
        ax.plot([x, ex], [y, ey], color=color, linewidth=1.8, alpha=0.95,
                zorder=4)
        ax.plot(ex, ey, marker="o", color=color, markersize=4, zorder=5)

    # own vessel (triangle pointing along heading)
    rot_deg = np.degrees(heading) - 90.0
    ax.plot(x, y, marker=(3, 0, rot_deg), markersize=20, color=OWN,
            markeredgecolor="white", markeredgewidth=1.0, linestyle="None",
            zorder=6)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"ray range  (0 = land at ship, 1 = clear at {lidar_range:.0f} cells)")

    ax.set_title(f"The 16-ray lidar the policy sees — "
                 f"{args.scenario} (seed {args.seed})\n"
                 "short rays = land close; long rays = open water",
                 fontsize=12)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
