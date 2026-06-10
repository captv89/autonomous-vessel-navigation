"""
Comparison report: runs multiple agents over the same seeded scenario suite
and writes a markdown report plus trajectory/metric figures.

Fairness guarantee: agents are evaluated on identical (scenario, seed)
pairs through the identical engine and scoring, so differences in the
tables are differences between the approaches, not between test setups.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import Config
from src.evaluation.runner import run_suite
from src.sim.recorder import read_episode

logger = logging.getLogger(__name__)

TABLE_ROWS = [
    ("success_rate", "Success rate"),
    ("collision_rate", "Collision rate"),
    ("grounding_rate", "Grounding rate"),
    ("timeout_rate", "Timeout rate"),
    ("mean_duration_s", "Mean time to goal (s)"),
    ("mean_path_efficiency", "Path efficiency"),
    ("mean_min_separation", "Mean min separation"),
    ("mean_near_miss_steps", "Near-miss steps"),
    ("mean_mean_abs_rudder_deg", "Mean |rudder| (deg)"),
    ("mean_avoidance_step_fraction", "Time avoiding (frac)"),
    ("mean_starboard_turn_fraction", "Starboard turns (frac)"),
]


def generate_report(config: Config,
                    agent_factories: Dict[str, Callable],
                    scenario_names: List[str],
                    episodes_per_scenario: int,
                    base_seed: int,
                    out_dir: str | Path) -> Path:
    out = Path(out_dir)
    logs = out / "episodes"
    out.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, Any]] = {}
    for agent_name, factory in agent_factories.items():
        logger.info("Evaluating agent '%s' ...", agent_name)
        results[agent_name] = run_suite(
            config, factory, scenario_names,
            episodes_per_scenario=episodes_per_scenario,
            base_seed=base_seed, log_dir=logs)

    (out / "results.json").write_text(json.dumps(results, indent=1))
    _plot_outcomes(results, out / "outcomes.png")
    for name in scenario_names:
        _plot_trajectories(results, name, base_seed, logs,
                           out / f"trajectories_{name}.png")
    report_path = out / "report.md"
    report_path.write_text(_markdown(config, results, scenario_names,
                                     episodes_per_scenario, base_seed))
    logger.info("Report written to %s", report_path)
    return report_path


# ----------------------------------------------------------------- markdown

def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def _markdown(config: Config, results: Dict[str, Any],
              scenario_names: List[str], episodes: int,
              base_seed: int) -> str:
    agents = list(results)
    lines = [
        "# Classical vs RL Navigation - Comparison Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}  ",
        f"Scenarios: {', '.join(scenario_names)}  ",
        f"Episodes per scenario: {episodes} (seeds {base_seed}.."
        f"{base_seed + episodes - 1}, identical for every agent)",
        "",
        "![outcomes](outcomes.png)",
        "",
        "## Overall",
        "",
        "| Metric | " + " | ".join(agents) + " |",
        "|---|" + "---|" * len(agents),
    ]
    for key, label in TABLE_ROWS:
        row = [_fmt(results[a]["overall"].get(key)) for a in agents]
        lines.append(f"| {label} | " + " | ".join(row) + " |")

    for name in scenario_names:
        lines += ["", f"## Scenario: {name}", "",
                  f"![trajectories](trajectories_{name}.png)", "",
                  "| Metric | " + " | ".join(agents) + " |",
                  "|---|" + "---|" * len(agents)]
        for key, label in TABLE_ROWS:
            row = [_fmt(results[a]["by_scenario"][name].get(key))
                   for a in agents]
            lines.append(f"| {label} | " + " | ".join(row) + " |")

    lines += [
        "",
        "## Notes",
        "",
        "- All metrics are computed from ground-truth engine state in the "
        "episode logs, never from agent self-reports.",
        "- Per-step decision records (including avoidance candidate tables "
        "and RL action distributions) are in `episodes/*.jsonl`; replay any "
        "of them with `python main.py replay <file>`.",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------- plots

def _plot_outcomes(results: Dict[str, Any], path: Path) -> None:
    agents = list(results)
    outcomes = ["goal", "collision", "grounding", "out_of_bounds", "timeout"]
    counts = {a: {o: 0 for o in outcomes} for a in agents}
    for a in agents:
        for ep in results[a]["episodes"]:
            counts[a][ep["outcome"]] = counts[a].get(ep["outcome"], 0) + 1

    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.8 / len(agents)
    xs = np.arange(len(outcomes))
    for i, a in enumerate(agents):
        ax.bar(xs + i * width, [counts[a][o] for o in outcomes],
               width, label=a)
    ax.set_xticks(xs + width * (len(agents) - 1) / 2)
    ax.set_xticklabels(outcomes)
    ax.set_ylabel("episodes")
    ax.set_title("Episode outcomes by agent")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_trajectories(results: Dict[str, Any], scenario: str,
                       base_seed: int, log_dir: Path, path: Path) -> None:
    agents = list(results)
    fig, axes = plt.subplots(1, len(agents),
                             figsize=(5.2 * len(agents), 5.2),
                             squeeze=False)
    for ax, agent in zip(axes[0], agents):
        log = log_dir / f"{agent}_{scenario}_{base_seed}.jsonl"
        if not log.exists():
            continue
        ep = read_episode(log)
        header, steps = ep["header"], ep["steps"]
        world = header["config"]["world"]
        sc = header["scenario"]

        from src.visualization.viewer import _rebuild_grid
        grid = _rebuild_grid(sc, world)
        ax.imshow(grid, origin="lower", cmap="YlOrBr", vmin=0, vmax=2,
                  extent=(0, world["width"], 0, world["height"]))

        xs = [s["vessel"]["x"] for s in steps]
        ys = [s["vessel"]["y"] for s in steps]
        modes = [s["decision"].get("mode", "") for s in steps]
        colors = ["crimson" if m == "avoidance" else "tab:blue"
                  for m in modes]
        ax.scatter(xs, ys, c=colors, s=2)

        traffic: Dict[int, List] = {}
        for s in steps:
            for ob in s["obstacles"]:
                traffic.setdefault(ob["id"], []).append((ob["x"], ob["y"]))
        for tr in traffic.values():
            t = np.array(tr)
            ax.plot(t[:, 0], t[:, 1], color="darkorange", lw=1, alpha=0.8)

        route = header["agent"].get("waypoints")
        if route:
            r = np.array(route)
            ax.plot(r[:, 0], r[:, 1], "--", color="gray", lw=1)
        ax.plot(*sc["start"], "g^", markersize=8)
        ax.plot(*sc["goal"], "g*", markersize=12)
        outcome = (ep["summary"] or {}).get("outcome", "?")
        ax.set_title(f"{agent} - {outcome}")
        ax.set_xlim(0, world["width"])
        ax.set_ylim(0, world["height"])
        ax.set_aspect("equal")
    fig.suptitle(f"{scenario} (seed {base_seed}; blue=track, red=avoiding, "
                 f"orange=traffic)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
