"""
Training-curves figure for the blog/paper.

Reads the PPO run's `progress.csv` (written by src/rl/train.py to a timestamped
runs/ppo_* directory) and plots episode reward, success rate, and grounding
rate against timesteps on a shared x-axis, with the 4M-step point marked (the
post calls out the 4M-vs-10M plateau).

These are real logged scalars — run a training first, e.g.:
    python -m src.rl.train --timesteps 10000000 \
        --scenario coastal_random,random_encounter,random --out models/ppo_vessel

Usage:
    python scripts/make_training_curves.py [--csv runs/ppo_XXedit/progress.csv]
                                           [--out animations/training_curves.png]
    # with no --csv, the most recent runs/ppo_*/progress.csv is used.
"""

from __future__ import annotations

# Allow running as `python scripts/<name>.py` from the repo root
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import glob
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REWARD = "#1f3a5f"
SUCCESS = "#2e7d32"
GROUND = "#c0392b"
MARK = "#7f8c8d"


def latest_csv() -> str:
    runs = sorted(glob.glob("runs/ppo_*/progress.csv"), key=os.path.getmtime)
    if not runs:
        raise SystemExit("no runs/ppo_*/progress.csv found — train first "
                         "(python -m src.rl.train ...)")
    return runs[-1]


def load(csv_path: str):
    cols: dict = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for key in reader.fieldnames or []:
            cols[key] = []
        for row in reader:
            for key, val in row.items():
                cols[key].append(val)

    def series(name):
        return np.array([float(v) if v not in ("", None) else np.nan
                         for v in cols.get(name, [])])

    t = series("time/total_timesteps")
    return {
        "t": t,
        "reward": series("rollout/ep_rew_mean"),
        "success": series("rollout/success_rate"),
        "grounding": series("rollout/grounding_rate"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None, help="path to progress.csv")
    ap.add_argument("--out", default="animations/training_curves.png")
    ap.add_argument("--mark", type=float, default=4_000_000,
                    help="timestep to annotate (default 4M)")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    csv_path = args.csv or latest_csv()
    d = load(csv_path)
    if d["t"].size == 0:
        raise SystemExit(f"no rows in {csv_path}")

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    panels = [
        (axes[0], d["reward"], "Episode reward (mean)", REWARD, None),
        (axes[1], d["success"] * 100, "Success rate (%)", SUCCESS, (0, 100)),
        (axes[2], d["grounding"] * 100, "Grounding rate (%)", GROUND, (0, None)),
    ]
    for ax, y, label, color, ylim in panels:
        ax.plot(d["t"], y, color=color, linewidth=2.0)
        ax.set_ylabel(label, fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.4)
        if ylim:
            ax.set_ylim(*ylim)
        if args.mark and d["t"].max() >= args.mark:
            ax.axvline(args.mark, color=MARK, linestyle="--", linewidth=1.2)
    axes[0].axvline(args.mark, color=MARK, linestyle="--", linewidth=1.2,
                    label=f"{args.mark/1e6:.0f}M steps")
    axes[0].legend(loc="lower right", fontsize=9, framealpha=0.95)

    axes[-1].set_xlabel("timesteps")
    axes[-1].xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{v/1e6:.0f}M"))
    fig.suptitle(f"PPO training curves  ({Path(csv_path).parent.name})",
                 fontsize=13, y=0.995)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out}  (from {csv_path})")


if __name__ == "__main__":
    main()
