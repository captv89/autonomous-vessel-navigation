"""
VesselNav-Bench: a frozen, seeded benchmark any navigation policy can run
against — the leaderboard layer on top of the evaluation runner.

Submitting a model requires only the Agent contract (src/agents/base.py):
`reset(obs)` + `decide(obs) -> Decision`, constructed with a Config. Agents
are referenced by spec string:

    classical                       built-in classical pipeline
    classical-legacy                built-in with the legacy avoider
    rl:models/ppo_vessel            built-in PPO wrapper + model path
    my_package.my_module:MyAgent    any importable class

Fairness and reproducibility guarantees:
- every agent runs the identical (scenario, seed, condition) triples;
- disturbances are scenario-seeded, so realizations match across agents;
- the SHA-256 of the resolved physics config is stored in the results —
  numbers are only comparable when the hash matches;
- all metrics (incl. COLREGs compliance) are computed from ground-truth
  episode logs, never from agent self-reports;
- rates carry Wilson 95% CIs, means carry seeded bootstrap 95% CIs.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import importlib
import json
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import yaml

from src.config import Config
from src.evaluation.runner import run_episode
from src.evaluation.colregs import (score_episode, aggregate_compliance,
                                     score_tss, aggregate_tss)
from src.evaluation.stats import (bootstrap_mean, fmt_mean, fmt_rate,
                                  wilson_interval)
from src.sim.recorder import read_episode
from src.sim.scenarios import build_scenario

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- loading

def load_suite(path: str | Path) -> Dict[str, Any]:
    with open(path) as f:
        suite = yaml.safe_load(f)
    required = {"name", "version", "scenarios", "episodes_per_scenario",
                "base_seed", "conditions"}
    missing = required - set(suite)
    if missing:
        raise ValueError(f"benchmark suite {path} missing keys: {missing}")
    return suite


def config_hash(config: Config) -> str:
    """Hash of everything that defines the exam's physics and scoring."""
    data = config.to_dict()
    data.pop("training", None)        # training setup doesn't affect the exam
    blob = json.dumps(data, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def make_agent_factory(spec: str, config: Config) -> Callable:
    """Resolve an agent spec string to a zero-arg factory."""
    if spec == "classical":
        from src.agents.classical import ClassicalAgent
        return lambda: ClassicalAgent(config)
    if spec == "classical-legacy":
        from src.agents.classical import ClassicalAgent
        legacy_cfg = copy.deepcopy(config)
        legacy_cfg.avoidance.implementation = "legacy"

        def factory():
            agent = ClassicalAgent(legacy_cfg)
            agent.name = "classical-legacy"
            return agent
        return factory
    if spec.startswith("rl:"):
        from src.agents.rl_agent import RLAgent
        model_path = spec.split(":", 1)[1]
        return lambda: RLAgent(config, model_path)
    if spec == "mpc":
        from src.agents.mpc import MPCAgent
        return lambda: MPCAgent(config)
    if spec.startswith("rl-shielded:"):
        from src.agents.shielded import ShieldedRLAgent
        model_path = spec.split(":", 1)[1]
        return lambda: ShieldedRLAgent(config, model_path)
    if ":" in spec:
        module_name, class_name = spec.rsplit(":", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        return lambda: cls(config)
    raise ValueError(
        f"Unknown agent spec '{spec}'. Use 'classical', 'classical-legacy', "
        f"'mpc', 'rl:<model.zip>', 'rl-shielded:<model.zip>', or "
        f"'<module>:<AgentClass>'.")


def apply_condition(base: Config, override: Dict[str, Any]) -> Config:
    merged = base.to_dict()
    for section, values in (override or {}).items():
        merged.setdefault(section, {}).update(values)
    return Config.from_dict(merged)


# ----------------------------------------------------------------- running

def _display_keys(agent_specs: List[str],
                  base_names: Dict[str, str]) -> Dict[str, str]:
    """Map each spec to a unique results key.

    Agents name themselves by class (e.g. every `rl:<path>` reports
    `rl_ppo`), so running two models behind one class would collide and the
    later result would silently overwrite the earlier. When a base name is
    used by more than one spec, disambiguate it with the model file name:
    `rl_ppo[ppo_10m]` vs `rl_ppo[ppo_vessel]`. Single-model runs are
    unchanged, so existing reports and the paper pipeline keep their names.
    """
    counts = Counter(base_names.values())
    keys: Dict[str, str] = {}
    for spec in agent_specs:
        name = base_names[spec]
        if counts[name] > 1 and ":" in spec:
            keys[spec] = f"{name}[{Path(spec.split(':', 1)[1]).name}]"
        else:
            keys[spec] = name
    return keys


def _run_agent_condition(
        spec: str, cond_name: str, override: Dict[str, Any], *,
        display_key: str, base_cfg_dict: Dict[str, Any],
        scenarios: List[str], episodes_per_scenario: int, base_seed: int,
        log_root: str, keep_logs: bool) -> Tuple[str, str, Dict[str, Any]]:
    """Run one agent over one condition (all scenarios x seeds).

    The unit of parallel work. A fresh agent is built per episode (identical,
    stateless starts, exactly as the sequential path did) but the heavy PPO
    model load is cached per process, so each worker reads a model at most
    once. Self-contained and picklable so it runs in a worker process.
    """
    base_config = Config.from_dict(base_cfg_dict)
    config = apply_condition(base_config, override)
    factory = make_agent_factory(spec, config)

    episodes: List[Dict[str, Any]] = []
    compliance: List[Any] = []
    tss_scores: List[Dict[str, float]] = []
    for scenario_name in scenarios:
        for i in range(episodes_per_scenario):
            seed = base_seed + i
            scenario = build_scenario(scenario_name, seed=seed)
            log_path = (Path(log_root) / display_key / cond_name /
                        f"{scenario_name}_{seed}.jsonl"
                        if keep_logs else None)
            stats = run_episode(config, scenario, factory(),
                                log_path=log_path)
            record = {"scenario": scenario_name, "seed": seed,
                      **stats.metrics}
            if log_path is not None:
                steps = read_episode(log_path)["steps"]
                scores = score_episode(
                    steps,
                    safe_distance=config.avoidance.detector_safe_distance,
                    collision_radius=config.simulation.collision_radius,
                    domain=config.ship_domain)
                compliance.append(scores)
                record["colregs"] = [s.to_dict() for s in scores]
                if scenario.tss is not None:
                    rule10 = score_tss(steps, scenario.tss)
                    record["rule10"] = rule10
                    tss_scores.append(rule10)
            episodes.append(record)
        logger.info("%s | %s | %s done", display_key, cond_name,
                    scenario_name)

    agg = _aggregate(episodes, compliance)
    if tss_scores:
        agg["rule10"] = aggregate_tss(tss_scores)
    agg["episodes"] = episodes
    return display_key, cond_name, agg


def run_benchmark(base_config: Config, suite: Dict[str, Any],
                  agent_specs: List[str], out_dir: str | Path,
                  keep_logs: bool = True, workers: int = 1) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_root = out / "episodes"

    # Suite-wide config override (e.g. a v2 ship domain), applied under every
    # condition. Folded into the config hash like any other physics/scoring.
    base_config = apply_condition(base_config, suite.get("base"))

    results: Dict[str, Any] = {
        "suite": {k: suite[k] for k in
                  ("name", "version", "description", "scenarios",
                   "episodes_per_scenario", "base_seed", "conditions")},
        "generated": datetime.now().isoformat(timespec="seconds"),
        "config_hash": {},
        "agents": {},
    }

    # Probe each spec once for metadata + self-reported name (the PPO load
    # this triggers is cached, so workers reuse it rather than re-reading).
    probes = {spec: make_agent_factory(spec, base_config)().metadata()
              for spec in agent_specs}
    base_names = {spec: probes[spec].get("name", "") for spec in agent_specs}
    keys = _display_keys(agent_specs, base_names)

    # Pre-create agent shells in spec order so the output stays ordered
    # regardless of which task finishes first under parallelism.
    for spec in agent_specs:
        results["agents"][keys[spec]] = {
            "spec": spec, "name": keys[spec], "conditions": {},
            "about": {
                "family": probes[spec].get("family", ""),
                "author": probes[spec].get("author", ""),
                "summary": probes[spec].get("summary", ""),
            },
        }

    # Config hash is a property of the condition's physics, not the agent.
    for cond_name, override in suite["conditions"].items():
        results["config_hash"][cond_name] = config_hash(
            apply_condition(base_config, override))

    tasks = [(spec, cond_name, override)
             for spec in agent_specs
             for cond_name, override in suite["conditions"].items()]
    work = functools.partial(
        _run_agent_condition,
        base_cfg_dict=base_config.to_dict(), scenarios=suite["scenarios"],
        episodes_per_scenario=suite["episodes_per_scenario"],
        base_seed=suite["base_seed"], log_root=str(log_root),
        keep_logs=keep_logs)

    def store(display_key: str, cond_name: str, agg: Dict[str, Any]) -> None:
        results["agents"][display_key]["conditions"][cond_name] = agg

    if workers and workers > 1:
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(work, spec, cond, ov, display_key=keys[spec])
                       for spec, cond, ov in tasks]
            for fut in as_completed(futures):
                display_key, cond_name, agg = fut.result()
                store(display_key, cond_name, agg)
                done += 1
                logger.info("[%d/%d] %s | %s complete",
                            done, len(tasks), display_key, cond_name)
    else:
        for spec, cond, ov in tasks:
            store(*work(spec, cond, ov, display_key=keys[spec]))

    (out / "results.json").write_text(json.dumps(results, indent=1))
    report = _leaderboard_markdown(results)
    (out / "leaderboard.md").write_text(report)
    from src.evaluation.html_report import write_html
    write_html(out / "results.json", out / "index.html")
    logger.info("Benchmark written to %s", out)
    return out / "leaderboard.md"


def _aggregate(episodes: List[Dict[str, Any]],
               compliance: List[List]) -> Dict[str, Any]:
    n = len(episodes)
    succ = [e for e in episodes if e.get("success")]
    agg: Dict[str, Any] = {
        "n_episodes": n,
        "success": wilson_interval(len(succ), n),
        "collision": wilson_interval(
            sum(1 for e in episodes if e["outcome"] == "collision"), n),
        "grounding": wilson_interval(
            sum(1 for e in episodes if e["outcome"] == "grounding"), n),
        "duration_s": bootstrap_mean([e["duration_s"] for e in succ]),
        "path_efficiency": bootstrap_mean(
            [e["path_efficiency"] for e in succ]),
        "min_separation": bootstrap_mean(
            [e.get("min_separation") for e in episodes]),
        "colregs": aggregate_compliance(compliance),
    }
    # Headline score: documented, deliberately simple, recomputable from the
    # published components.
    colregs_mean = agg["colregs"].get("mean_score")
    eff = agg["path_efficiency"]["mean"] if agg["path_efficiency"] else 0.0
    agg["benchmark_score"] = round(100 * (
        0.6 * agg["success"]["rate"]
        + 0.25 * (colregs_mean if colregs_mean is not None else 0.0)
        + 0.15 * min(eff, 1.0)), 1)
    return agg


# --------------------------------------------------------------- reporting

def _leaderboard_markdown(results: Dict[str, Any]) -> str:
    suite = results["suite"]
    agents = results["agents"]
    conditions = list(suite["conditions"])

    lines = [
        f"# {suite['name']} v{suite['version']} - Leaderboard",
        "",
        f"Generated: {results['generated']}  ",
        f"Scenarios: {', '.join(suite['scenarios'])} x "
        f"{suite['episodes_per_scenario']} episodes "
        f"(seeds {suite['base_seed']}..)  ",
        f"Config hash: " + ", ".join(
            f"{c}: `{h}`" for c, h in results["config_hash"].items()),
        "",
        "**Benchmark score** = 100 x (0.6 x success rate + 0.25 x COLREGs "
        "compliance + 0.15 x path efficiency). All components below; "
        "rates show Wilson 95% CIs, means show bootstrap 95% CIs.",
        "",
    ]

    for cond in conditions:
        ranked = sorted(
            agents.values(),
            key=lambda a: a["conditions"][cond]["benchmark_score"],
            reverse=True)
        lines += [f"## Condition: {cond}", "",
                  "| Rank | Agent | Score | Success | Collision | Grounding "
                  "| COLREGs | Time to goal (s) | Path eff. | Min sep. |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for rank, agent in enumerate(ranked, 1):
            c = agent["conditions"][cond]
            colregs = c["colregs"]
            colregs_str = (f"{colregs['mean_score']:.2f} "
                           f"(n={colregs['encounters']})"
                           if colregs.get("encounters") else "-")
            lines.append(
                f"| {rank} | **{agent['name']}** "
                f"| **{c['benchmark_score']}** "
                f"| {fmt_rate(c['success'])} "
                f"| {fmt_rate(c['collision'])} "
                f"| {fmt_rate(c['grounding'])} "
                f"| {colregs_str} "
                f"| {fmt_mean(c['duration_s'], 1)} "
                f"| {fmt_mean(c['path_efficiency'])} "
                f"| {fmt_mean(c['min_separation'])} |")
        # Per-encounter compliance breakdown
        lines += ["", "Per-encounter COLREGs compliance:", ""]
        encounter_types = sorted({
            enc for a in ranked
            for enc in a["conditions"][cond]["colregs"]
            .get("by_encounter", {})})
        if encounter_types:
            lines.append("| Agent | " + " | ".join(encounter_types) + " |")
            lines.append("|---|" + "---|" * len(encounter_types))
            for agent in ranked:
                by_enc = (agent["conditions"][cond]["colregs"]
                          .get("by_encounter", {}))
                row = [f"{by_enc[e]['mean_score']:.2f} (n={by_enc[e]['n']})"
                       if e in by_enc else "-" for e in encounter_types]
                lines.append(f"| {agent['name']} | " + " | ".join(row) + " |")
        lines.append("")

    lines += ["", "## The agents", ""]
    for agent in agents.values():
        about = agent.get("about", {})
        lines.append(f"- **{agent['name']}** — {about.get('family', '')}; "
                     f"by {about.get('author', 'unknown')}. "
                     f"{about.get('summary', '')}")
    lines += [
        "",
        "## Reproducing / submitting",
        "",
        "```bash",
        "python main.py benchmark --suite benchmarks/v1.yaml \\",
        "    --agent classical --agent rl:models/ppo_vessel \\",
        "    --agent your_package.your_module:YourAgent",
        "```",
        "",
        "A submission implements the `Agent` contract "
        "(`src/agents/base.py`): constructed with a `Config`, then "
        "`reset(obs)` and `decide(obs) -> Decision` per step. Every episode "
        "is recorded to `episodes/` and can be replayed with "
        "`python main.py replay <log>`.",
        "",
        "Scores are comparable only between runs with identical config "
        "hashes.",
        "",
    ]
    return "\n".join(lines)
