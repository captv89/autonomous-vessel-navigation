"""
Unified episode runner.

Runs any Agent against any Scenario on the shared SimulationEngine,
recording every step. Both approaches are evaluated through this exact loop
— identical physics, identical scoring — which is what makes the
classical-vs-RL comparison fair.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.config import Config
from src.agents.base import Agent
from src.sim.engine import SimulationEngine
from src.sim.perception import Perception
from src.sim.scenarios import Scenario, build_scenario
from src.sim.recorder import EpisodeRecorder
from src.evaluation.metrics import compute_metrics, aggregate

logger = logging.getLogger(__name__)


@dataclass
class EpisodeStats:
    scenario: str
    agent: str
    outcome: str
    metrics: Dict[str, Any]
    log_path: Optional[str] = None


def run_episode(config: Config, scenario: Scenario, agent: Agent,
                log_path: Optional[str | Path] = None,
                step_callback: Optional[Callable] = None) -> EpisodeStats:
    """Run one full episode; returns its stats.

    step_callback(engine, obs, decision, result) is invoked after every step
    (used by the live viewer).
    """
    engine = SimulationEngine(config, scenario)
    obs = engine.reset()
    # The agent decides on perceived traffic (degraded under the matching
    # condition); the recorder below always logs the engine's ground truth, so
    # scoring stays physically correct regardless of perception.
    perception = Perception(config, scenario.seed)
    percept = perception.perceive(obs)
    agent.reset(percept)

    with EpisodeRecorder(log_path) as rec:
        rec.header(scenario=scenario.to_dict(), agent=agent.metadata(),
                   config=config.to_dict())
        outcome = "timeout"
        while True:
            decision = agent.decide(percept)
            result = engine.step(decision.desired_heading,
                                 decision.desired_speed)
            rec.step(t=result.obs.t, vessel=result.obs.vessel,
                     obstacles=result.obs.obstacles,
                     decision=decision.to_record(), events=result.events)
            if step_callback is not None:
                if step_callback(engine, result.obs, decision, result) is False:
                    outcome = result.outcome or "aborted"
                    break
            obs = result.obs
            if result.done:
                outcome = result.outcome
                break
            percept = perception.perceive(obs)

        steps = [r for r in rec.records if r["type"] == "step"]
        metrics = compute_metrics(
            steps, outcome, engine.goal,
            near_miss_distance=config.avoidance.detector_safe_distance)
        rec.summary(outcome=outcome, metrics=metrics)

    return EpisodeStats(scenario=scenario.name, agent=agent.name,
                        outcome=outcome, metrics=metrics,
                        log_path=str(log_path) if log_path else None)


def run_suite(config: Config, agent_factory: Callable[[], Agent],
              scenario_names: List[str], episodes_per_scenario: int = 5,
              base_seed: int = 0,
              log_dir: Optional[str | Path] = None) -> Dict[str, Any]:
    """Run an agent over a scenario suite with deterministic seeds.

    Same (scenario, seed) pairs are produced for every agent evaluated with
    the same arguments, so agents face identical situations.
    """
    results: List[EpisodeStats] = []
    for name in scenario_names:
        for i in range(episodes_per_scenario):
            seed = base_seed + i
            scenario = build_scenario(name, seed=seed)
            agent = agent_factory()
            log_path = None
            if log_dir is not None:
                log_path = Path(log_dir) / f"{agent.name}_{name}_{seed}.jsonl"
            stats = run_episode(config, scenario, agent, log_path=log_path)
            logger.info("%s | %s seed=%d -> %s",
                        agent.name, name, seed, stats.outcome)
            results.append(stats)

    by_scenario: Dict[str, Any] = {}
    for name in scenario_names:
        eps = [r.metrics for r in results if r.scenario == name]
        by_scenario[name] = aggregate(eps)
    return {
        "episodes": [r.__dict__ for r in results],
        "by_scenario": by_scenario,
        "overall": aggregate([r.metrics for r in results]),
    }
