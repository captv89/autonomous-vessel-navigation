"""
Grounding-rate-by-landmass-geometry figure for the blog/paper.

Reads a benchmark results.json, groups the RL policy's episodes by the
landmass-geometry class of their scenario (Scenario.geometry, e.g. "narrow
channel", "channel + islands", "scattered islands", "open water"), and plots
grounding rate per geometry as a sorted bar chart.

Every value comes from the benchmark's per-episode outcomes — run it first:
    python main.py benchmark --suite benchmarks/v1.yaml \
        --agent classical --agent rl:models/ppo_vessel --out reports/ppo_10m_bench

Usage:
    python scripts/make_grounding_by_geometry.py \
        [--results reports/ppo_10m_bench/results.json] \
        [--agent rl_ppo] [--condition all] \
        [--out animations/grounding_by_geometry.png]
"""

from __future__ import annotations

# Allow running as `python scripts/<name>.py` from the repo root
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.sim.scenarios import scenario_geometry

GROUND = "#c0392b"


def pick_agent(results: dict, requested: str | None) -> str:
    agents = list(results["agents"])
    if requested:
        if requested not in agents:
            raise SystemExit(f"agent '{requested}' not in {agents}")
        return requested
    rl = [a for a in agents if "rl" in a.lower()]
    if not rl:
        raise SystemExit(f"no RL agent found in {agents}; pass --agent")
    return rl[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="reports/ppo_10m_bench/results.json")
    ap.add_argument("--agent", default=None, help="agent name (default: first RL)")
    ap.add_argument("--condition", default="all",
                    help="benchmark condition, or 'all' to pool (default)")
    ap.add_argument("--out", default="animations/grounding_by_geometry.png")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text())
    agent = pick_agent(results, args.agent)
    conditions = results["agents"][agent]["conditions"]
    cond_names = (list(conditions) if args.condition == "all"
                  else [args.condition])

    # grounding count / total per geometry
    total: dict = defaultdict(int)
    grounded: dict = defaultdict(int)
    for cond in cond_names:
        for ep in conditions[cond]["episodes"]:
            geom = scenario_geometry(ep["scenario"])
            total[geom] += 1
            if ep["outcome"] == "grounding":
                grounded[geom] += 1

    geoms = sorted(total, key=lambda g: grounded[g] / total[g], reverse=True)
    rates = [100 * grounded[g] / total[g] for g in geoms]
    counts = [total[g] for g in geoms]

    overall = 100 * sum(grounded.values()) / sum(total.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(geoms)), rates, color=GROUND, width=0.6)
    for bar, rate, n in zip(bars, rates, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{rate:.1f}%\n(n={n})", ha="center", va="bottom", fontsize=9)
    ax.axhline(overall, color="#7f8c8d", linestyle="--", linewidth=1.2,
               label=f"overall {overall:.1f}%")
    ax.set_xticks(range(len(geoms)))
    ax.set_xticklabels(geoms, fontsize=10)
    ax.set_ylabel("grounding rate (%)")
    ax.set_ylim(0, max(rates) * 1.25 + 1)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    cond_txt = "all conditions" if args.condition == "all" else args.condition
    ax.set_title(f"{agent}: grounding rate by landmass geometry ({cond_txt})",
                 fontsize=12)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out}  (agent={agent}, conditions={cond_names})")


if __name__ == "__main__":
    main()
