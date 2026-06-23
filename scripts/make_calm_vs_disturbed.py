"""
Calm-vs-disturbed robustness figure for the blog/paper.

Reads a benchmark results.json and plots each agent's benchmark score under
calm and disturbed (current + wind gust) conditions as a grouped bar chart,
with a tight y-axis so the degradation is visible rather than flattened by a
0-100 scale. The four numbers are printed to stdout so they can be checked
against the post's published values (it does NOT edit the post).

Run the benchmark first with both agents and conditions:
    python main.py benchmark --suite benchmarks/v1.yaml \
        --agent classical --agent rl:models/ppo_vessel --out reports/ppo_10m_bench

Usage:
    python scripts/make_calm_vs_disturbed.py \
        [--results reports/ppo_10m_bench/results.json] \
        [--agents classical,rl_ppo] \
        [--out animations/calm_vs_disturbed.png]
"""

from __future__ import annotations

# Allow running as `python scripts/<name>.py` from the repo root
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CALM = "#1f3a5f"
DISTURBED = "#d97706"
PRETTY = {"classical": "Classical", "rl_ppo": "RL-PPO", "rl": "RL-PPO",
          "mpc": "MPC", "rl_ppo_shielded": "RL-PPO + Shield"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="reports/ppo_10m_bench/results.json")
    ap.add_argument("--agents", default=None,
                    help="comma-separated agent names (default: all in file)")
    ap.add_argument("--out", default="animations/calm_vs_disturbed.png")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    results = json.loads(Path(args.results).read_text())
    available = list(results["agents"])
    agents = (args.agents.split(",") if args.agents else available)
    for a in agents:
        if a not in available:
            raise SystemExit(f"agent '{a}' not in {available}")

    # benchmark_score per (agent, condition)
    scores = {}
    print("Benchmark scores (verify against the post before publishing):")
    for a in agents:
        conds = results["agents"][a]["conditions"]
        calm = conds.get("calm", {}).get("benchmark_score")
        dist = conds.get("disturbed", {}).get("benchmark_score")
        scores[a] = (calm, dist)
        print(f"  {a:18s} calm={calm}  disturbed={dist}")

    x = np.arange(len(agents))
    w = 0.36
    calm_vals = [scores[a][0] for a in agents]
    dist_vals = [scores[a][1] for a in agents]

    lo = min(v for v in calm_vals + dist_vals if v is not None)
    hi = max(v for v in calm_vals + dist_vals if v is not None)
    pad = max((hi - lo) * 0.25, 2.0)

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w / 2, calm_vals, w, label="Calm", color=CALM)
    b2 = ax.bar(x + w / 2, dist_vals, w, label="Disturbed", color=DISTURBED)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + pad * 0.05,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY.get(a, a) for a in agents])
    ax.set_ylabel("benchmark score")
    ax.set_ylim(max(0, lo - pad), hi + pad)        # tight: show the gap
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.set_title("Robustness: benchmark score, calm vs disturbed", fontsize=12)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
