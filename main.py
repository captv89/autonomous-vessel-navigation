#!/usr/bin/env python3
"""
Autonomous vessel navigation simulator - command line interface.

    python main.py scenarios                       list scenarios
    python main.py simulate  --agent classical --scenario head_on --render
    python main.py simulate  --agent rl --model models/ppo_vessel
    python main.py train     --timesteps 600000
    python main.py evaluate  --model models/ppo_vessel --episodes 10
    python main.py replay    logs/classical_head_on_0.jsonl

All command parameters that shape physics/behavior come from the YAML
config (--config, default values in src/config.py).
"""

from __future__ import annotations

import argparse
import logging
import sys

from src.config import Config


def cmd_scenarios(args) -> None:
    from src.sim.scenarios import list_scenarios
    for name, desc in list_scenarios():
        print(f"{name:20s} {desc}")


def _make_agent(config: Config, kind: str, model: str | None):
    if kind == "classical":
        from src.agents.classical import ClassicalAgent
        return ClassicalAgent(config)
    if kind == "rl":
        if not model:
            sys.exit("--model is required for the RL agent")
        from src.agents.rl_agent import RLAgent
        return RLAgent(config, model)
    sys.exit(f"unknown agent '{kind}'")


def cmd_simulate(args) -> None:
    from src.sim.scenarios import build_scenario
    config = Config.load(args.config)
    scenario = build_scenario(args.scenario, seed=args.seed)
    agent = _make_agent(config, args.agent, args.model)
    log_path = args.log or f"logs/{args.agent}_{args.scenario}_{args.seed or 0}.jsonl"

    if args.render:
        from src.visualization.viewer import live
        outcome = live(config, scenario, agent, log_path=log_path)
    else:
        from src.evaluation.runner import run_episode
        stats = run_episode(config, scenario, agent, log_path=log_path)
        outcome = stats.outcome
        print("metrics:")
        for k, v in stats.metrics.items():
            print(f"  {k}: {v}")
    print(f"outcome: {outcome}")
    print(f"episode log: {log_path}  (replay with: python main.py replay {log_path})")


def cmd_train(args) -> None:
    from src.rl.train import train
    config = Config.load(args.config)
    train(config,
          total_timesteps=args.timesteps or config.training.total_timesteps,
          scenario=args.scenario or config.training.scenario,
          out_path=args.out, n_envs=args.n_envs)


def cmd_evaluate(args) -> None:
    from src.evaluation.report import generate_report
    config = Config.load(args.config)
    scenario_names = (args.scenarios.split(",") if args.scenarios else
                      ["open_water", "head_on", "crossing_starboard",
                       "crossing_port", "overtaking", "coastal", "random"])

    factories = {}
    for kind in args.agents.split(","):
        kind = kind.strip()
        factories[kind if kind != "rl" else "rl_ppo"] = (
            lambda k=kind: _make_agent(config, k, args.model))

    report = generate_report(config, factories, scenario_names,
                             episodes_per_scenario=args.episodes,
                             base_seed=args.seed, out_dir=args.out)
    print(f"report: {report}")


def cmd_benchmark(args) -> None:
    from src.evaluation.benchmark import load_suite, run_benchmark
    config = Config.load(args.config)
    suite = load_suite(args.suite)
    report = run_benchmark(config, suite, args.agent, out_dir=args.out,
                           keep_logs=not args.no_logs)
    print(f"leaderboard: {report}")


def cmd_replay(args) -> None:
    from src.visualization.viewer import replay
    replay(args.log)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="main.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("scenarios", help="list available scenarios")

    p = sub.add_parser("simulate", help="run one episode")
    p.add_argument("--agent", default="classical", choices=["classical", "rl"])
    p.add_argument("--scenario", default="coastal")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--model", default=None, help="RL model path (.zip)")
    p.add_argument("--config", default=None, help="YAML config")
    p.add_argument("--log", default=None, help="episode log output path")
    p.add_argument("--render", action="store_true", help="pygame live view")

    p = sub.add_parser("train", help="train the PPO agent")
    p.add_argument("--config", default=None)
    p.add_argument("--timesteps", type=int, default=None)
    p.add_argument("--scenario", default=None)
    p.add_argument("--out", default="models/ppo_vessel")
    p.add_argument("--n-envs", type=int, default=None)

    p = sub.add_parser("evaluate", help="run comparison suite + report")
    p.add_argument("--agents", default="classical,rl")
    p.add_argument("--model", default="models/ppo_vessel",
                   help="RL model path")
    p.add_argument("--scenarios", default=None,
                   help="comma-separated scenario names (default: all)")
    p.add_argument("--episodes", type=int, default=5,
                   help="episodes per scenario")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", default=None)
    p.add_argument("--out", default="reports/latest")

    p = sub.add_parser("benchmark",
                       help="run the frozen benchmark suite (leaderboard)")
    p.add_argument("--suite", default="benchmarks/v1.yaml")
    p.add_argument("--agent", action="append", required=True,
                   help="agent spec: classical | classical-legacy | "
                        "rl:<model.zip> | <module>:<AgentClass> "
                        "(repeatable)")
    p.add_argument("--config", default=None)
    p.add_argument("--out", default="reports/benchmark")
    p.add_argument("--no-logs", action="store_true",
                   help="skip per-episode logs (faster, disables COLREGs "
                        "scoring and replays)")

    p = sub.add_parser("replay", help="replay a recorded episode")
    p.add_argument("log", help="JSONL episode log")

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s")

    {"scenarios": cmd_scenarios, "simulate": cmd_simulate,
     "train": cmd_train, "evaluate": cmd_evaluate,
     "benchmark": cmd_benchmark, "replay": cmd_replay}[args.command](args)


if __name__ == "__main__":
    main()
