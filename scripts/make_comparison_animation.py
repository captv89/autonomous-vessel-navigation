"""
Side-by-side Classical vs RL-PPO navigation animation for the blog/paper.

Renders two episode logs (same scenario + seed, different agent) as a single
side-by-side MP4. Pure matplotlib + numpy + stdlib: no pygame, torch, or SB3,
so it runs in a plain headless environment.

Usage:
    python scripts/make_comparison_animation.py \
        --left  logs/classical_head_on_seed1000.jsonl \
        --right logs/rl_head_on_seed1000.jsonl \
        --out   animations/classical_vs_rl_head_on.mp4

Options:
    --left/--right   episode JSONL logs (left is usually classical, right RL)
    --out            output .mp4 path (default derived from scenario name)
    --fps            output frame rate (default 15)
    --speed          playback speed vs real time (default 8x; sim dt is 0.1s,
                     so 1x = 10 sim-steps/sec which makes 3-minute episodes
                     unwatchable -- compress time instead)
    --freeze         seconds to hold the final outcome frame (default 1.5)
    --trail          number of past positions in the wake trail (default 40)
    --dpi            output DPI (default 100 -> 1600x800 at figsize 16x8)

If ffmpeg is unavailable the script falls back to writing PNG frames under
animations/frames/<name>/ and prints the ffmpeg command to stitch them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless: never try to open a window
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle

# --- palette (matches the blog spec) -------------------------------------
OCEAN = "#E8F4FD"
LAND = "#8B7355"
LAND_EDGE = "#6E5A43"
OWN = "#1F4E79"
ROUTE = "#2ECC71"
CPA = "#E74C3C"
TRAFFIC_COLORS = {
    "give-way": "#C0392B",
    "stand-on": "#E67E22",
    "neutral": "#7F8C8D",
}
OUTCOME_STYLE = {
    "goal": ("✓ SUCCESS", "#2ECC71"),
    "success": ("✓ SUCCESS", "#2ECC71"),
    "collision": ("✗ COLLISION", "#C0392B"),
    "grounding": ("✗ GROUNDED", "#C0392B"),
    "grounded": ("✗ GROUNDED", "#C0392B"),
    "timeout": ("✗ TIMEOUT", "#E67E22"),
}

# encounter classification (from the classical decision log) -> traffic role
GIVE_WAY = {"crossing-starboard", "head-on", "overtaking"}
STAND_ON = {"crossing-port"}


@dataclass
class Episode:
    """Parsed episode log indexed by simulated time for frame lookup."""

    label: str
    outcome: str
    end_time: float
    times: List[float]
    steps: List[Dict[str, Any]]
    waypoints: Optional[List[List[float]]]
    rect_obstacles: List[List[float]]
    circle_obstacles: List[List[float]]
    goal: Optional[List[float]]
    width: float
    height: float

    def state_at(self, t: float) -> Dict[str, Any]:
        """Last step at or before time t (clamped to the episode bounds)."""
        if not self.steps:
            return {}
        i = bisect_right(self.times, t) - 1
        i = max(0, min(i, len(self.steps) - 1))
        return self.steps[i]

    def finished_at(self, t: float) -> bool:
        return t >= self.end_time


def _label_for(agent: Dict[str, Any]) -> str:
    name = (agent or {}).get("name", "agent")
    pretty = {
        "classical": "Classical",
        "classical-legacy": "Classical (legacy)",
        "mpc": "MPC",
        "rl": "RL-PPO (10M)",
        "rl_ppo": "RL-PPO (10M)",
        "rl-shielded": "RL-PPO + Shield",
        "rl_ppo_shielded": "RL-PPO + Shield",
    }
    return pretty.get(name, name)


# friendlier scenario titles for the three blog clips
SCENARIO_TITLES = {
    "head_on": "Head-On (Rule 14)",
    "crossing_starboard": "Crossing — Give-Way (Rule 15)",
    "crossing_port": "Crossing — Stand-On (Rule 17)",
    "coastal": "Coastal Channel",
    "overtaking": "Overtaking (Rule 13)",
}


def _pretty_scenario(name: str) -> str:
    return SCENARIO_TITLES.get(name, name.replace("_", " ").title())


def load_episode(path: str) -> Episode:
    header: Dict[str, Any] = {}
    summary: Dict[str, Any] = {}
    steps: List[Dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            kind = rec.get("type")
            if kind == "header":
                header = rec
            elif kind == "step":
                steps.append(rec)
            elif kind == "summary":
                summary = rec

    if not steps:
        sys.exit(f"no step records found in {path}")

    scenario = header.get("scenario", {})
    agent = header.get("agent", {})
    config = header.get("config", {})
    world = config.get("world", {})

    times = [float(s.get("t", i)) for i, s in enumerate(steps)]
    outcome = summary.get("outcome") or "timeout"

    return Episode(
        label=_label_for(agent),
        outcome=outcome,
        end_time=times[-1],
        times=times,
        steps=steps,
        waypoints=agent.get("waypoints"),  # classical only; RL has none
        rect_obstacles=scenario.get("rect_obstacles", []) or [],
        circle_obstacles=scenario.get("circle_obstacles", []) or [],
        goal=scenario.get("goal"),
        width=float(world.get("width", 100)),
        height=float(world.get("height", 100)),
    )


def _traffic_role(step: Dict[str, Any]) -> Dict[int, str]:
    """Map traffic id -> COLREGs role using the classical risk log, if present.

    RL logs carry no risk classification, so everything falls back to neutral.
    """
    roles: Dict[int, str] = {}
    for risk in step.get("decision", {}).get("risks", []) or []:
        vid = risk.get("vessel_id")
        enc = risk.get("encounter", "")
        if vid is None:
            continue
        if enc in GIVE_WAY:
            roles[vid] = "give-way"
        elif enc in STAND_ON:
            roles[vid] = "stand-on"
        else:
            roles[vid] = "neutral"
    return roles


def _vessel_triangle(ax, x, y, heading_rad, color, size, zorder=5):
    """A filled triangle marker pointing along heading (0 rad = +x / East)."""
    rot_deg = math.degrees(heading_rad) - 90.0  # marker (3,0,0) points up (+y)
    return ax.plot(
        x, y,
        marker=(3, 0, rot_deg),
        markersize=size,
        color=color,
        markeredgecolor="white",
        markeredgewidth=0.8,
        linestyle="None",
        zorder=zorder,
    )


def _draw_static(ax, ep: Episode):
    """Land, goal and (classical) planned route -- the parts that never move."""
    ax.set_facecolor(OCEAN)
    ax.set_xlim(0, ep.width)
    ax.set_ylim(0, ep.height)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#B0C4D4")

    for rx, ry, rw, rh in ep.rect_obstacles:
        ax.add_patch(Rectangle((rx, ry), rw, rh, facecolor=LAND,
                               edgecolor=LAND_EDGE, linewidth=1.0, zorder=2))
    for cx, cy, cr in ep.circle_obstacles:
        ax.add_patch(Circle((cx, cy), cr, facecolor=LAND,
                            edgecolor=LAND_EDGE, linewidth=1.0, zorder=2))

    if ep.waypoints and len(ep.waypoints) > 1:
        wx = [p[0] for p in ep.waypoints]
        wy = [p[1] for p in ep.waypoints]
        ax.plot(wx, wy, color=ROUTE, linestyle="--", linewidth=1.6,
                alpha=0.45, zorder=3, label="planned route")

    if ep.goal:
        ax.plot(ep.goal[0], ep.goal[1], marker="*", markersize=20,
                color="#F1C40F", markeredgecolor="#B7950B",
                markeredgewidth=0.8, zorder=4)


class Panel:
    """One agent's subplot: static layer drawn once, dynamic layer per frame."""

    def __init__(self, ax, ep: Episode, trail_len: int):
        self.ax = ax
        self.ep = ep
        self.trail_len = trail_len
        _draw_static(ax, ep)
        self._dynamic: List[Any] = []

    def _clear_dynamic(self):
        for artist in self._dynamic:
            artist.remove()
        self._dynamic = []

    def draw(self, t: float):
        self._clear_dynamic()
        ep = self.ep
        step = ep.state_at(t)
        if not step:
            return

        vessel = step.get("vessel", {})
        vx, vy = vessel.get("x", 0.0), vessel.get("y", 0.0)
        vh = vessel.get("heading", 0.0)
        vspeed = vessel.get("speed", 0.0)

        # wake trail: last N own-vessel positions, fading
        idx = bisect_right(ep.times, t) - 1
        idx = max(0, min(idx, len(ep.steps) - 1))
        lo = max(0, idx - self.trail_len)
        txs = [s["vessel"]["x"] for s in ep.steps[lo:idx + 1]]
        tys = [s["vessel"]["y"] for s in ep.steps[lo:idx + 1]]
        if len(txs) > 1:
            (line,) = self.ax.plot(txs, tys, color=OWN, linewidth=1.6,
                                   alpha=0.5, zorder=4)
            self._dynamic.append(line)

        # traffic vessels (variable count per step)
        roles = _traffic_role(step)
        for ob in step.get("obstacles", []) or []:
            oid = ob.get("id")
            role = roles.get(oid, "neutral")
            color = TRAFFIC_COLORS[role]
            self._dynamic.extend(
                _vessel_triangle(self.ax, ob.get("x", 0.0), ob.get("y", 0.0),
                                 ob.get("heading", 0.0), color, 14, zorder=5)
            )
            # CPA line for at-risk vessels (classical logs only)
            for risk in step.get("decision", {}).get("risks", []) or []:
                if risk.get("vessel_id") == oid and risk.get("risk_level", 0) >= 2:
                    (cl,) = self.ax.plot([vx, ob.get("x", 0.0)],
                                         [vy, ob.get("y", 0.0)],
                                         color=CPA, linestyle=":", linewidth=1.2,
                                         alpha=0.6, zorder=4)
                    self._dynamic.append(cl)

        # own vessel on top
        self._dynamic.extend(
            _vessel_triangle(self.ax, vx, vy, vh, OWN, 18, zorder=6)
        )

        # status line
        mode = step.get("decision", {}).get("mode", "")
        mode_txt = {"path_following": "PATH FOLLOWING",
                    "avoidance": "AVOIDANCE",
                    "policy": "POLICY"}.get(mode, mode.upper())
        status = (f"{ep.label}    t = {step.get('t', t):.1f}s    {mode_txt}\n"
                  f"speed {vspeed:.2f}   hdg {math.degrees(vh) % 360:.0f}°")
        txt = self.ax.text(0.02, 0.03, status, transform=self.ax.transAxes,
                           fontsize=11, va="bottom", ha="left", color="#1B2631",
                           family="monospace",
                           bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                                     edgecolor="#B0C4D4", alpha=0.85), zorder=7)
        self._dynamic.append(txt)

        # outcome banner once the episode has ended
        if ep.finished_at(t):
            label, color = OUTCOME_STYLE.get(ep.outcome, (ep.outcome.upper(), "#7F8C8D"))
            banner = self.ax.text(0.5, 0.5, label, transform=self.ax.transAxes,
                                  fontsize=30, fontweight="bold", color="white",
                                  ha="center", va="center", zorder=8,
                                  bbox=dict(boxstyle="round,pad=0.6", facecolor=color,
                                            edgecolor="white", alpha=0.85))
            self._dynamic.append(banner)


def build_animation(left: Episode, right: Episode, args) -> FuncAnimation:
    scenario_name = _pretty_scenario(args.scenario or "scenario")
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(16, 8))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04, wspace=0.06)

    panels = [Panel(axl, left, args.trail), Panel(axr, right, args.trail)]

    title = f"Scenario: {scenario_name}"
    if args.seed is not None:
        title += f"  (Seed {args.seed})"
    suptitle = fig.suptitle(title, fontsize=17, fontweight="bold", y=0.975)

    # timeline: align both episodes by SIMULATED time, then hold the last frame
    total_sim = max(left.end_time, right.end_time)
    sim_per_frame = args.speed / args.fps
    n_play = int(math.ceil(total_sim / sim_per_frame)) + 1
    n_freeze = int(round(args.freeze * args.fps))
    n_frames = n_play + n_freeze
    frame_time = fig.text(0.5, 0.925, "", ha="center", fontsize=11, color="#566573")

    def update(frame: int):
        t = min(frame, n_play - 1) * sim_per_frame
        for panel in panels:
            panel.draw(t)
        frame_time.set_text(f"t = {min(t, total_sim):.1f}s  /  {total_sim:.1f}s")
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / args.fps,
                         blit=False, repeat=False)
    anim._fig = fig  # keep a handle alive
    return anim, fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", required=True, help="left-panel episode log (classical)")
    ap.add_argument("--right", required=True, help="right-panel episode log (RL)")
    ap.add_argument("--scenario", default=None, help="scenario name for the title")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None, help="output mp4 path")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--speed", type=float, default=8.0,
                    help="playback speed vs real time (sim dt is 0.1s)")
    ap.add_argument("--freeze", type=float, default=1.5,
                    help="seconds to hold the outcome frame")
    ap.add_argument("--trail", type=int, default=40)
    ap.add_argument("--dpi", type=int, default=100)
    args = ap.parse_args()

    left = load_episode(args.left)
    right = load_episode(args.right)

    # infer scenario name from the log filename if not given
    if args.scenario is None:
        stem = Path(args.left).stem
        args.scenario = stem.replace("classical_", "").replace("_seed" + str(args.seed or ""), "")

    out = args.out or f"animations/classical_vs_rl_{args.scenario}.mp4"
    Path(out).parent.mkdir(parents=True, exist_ok=True)

    anim, fig = build_animation(left, right, args)

    from matplotlib.animation import writers
    if writers.is_available("ffmpeg"):
        Writer = writers["ffmpeg"]
        writer = Writer(fps=args.fps, codec="h264", bitrate=1800,
                        extra_args=["-pix_fmt", "yuv420p"])
        anim.save(out, writer=writer, dpi=args.dpi)
        print(f"wrote {out}")
    else:
        frame_dir = Path("animations/frames") / args.scenario
        frame_dir.mkdir(parents=True, exist_ok=True)
        anim.save(str(frame_dir / "frame_%05d.png"), writer="pillow", dpi=args.dpi)
        print(f"ffmpeg not found; wrote PNG frames to {frame_dir}")
        print("Stitch them with:")
        print(f"  ffmpeg -framerate {args.fps} -i {frame_dir}/frame_%05d.png "
              f"-c:v libx264 -pix_fmt yuv420p -b:v 1800k {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
