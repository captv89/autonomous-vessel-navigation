import numpy as np

from src.config import Config
from src.agents.mpc import MPCAgent
from src.evaluation.runner import run_episode
from src.sim.scenarios import build_scenario


def test_open_water_reaches_goal():
    cfg = Config()
    stats = run_episode(cfg, build_scenario("open_water"), MPCAgent(cfg))
    assert stats.outcome == "goal"


def test_head_on_resolved_safely_and_starboard():
    cfg = Config()
    stats = run_episode(cfg, build_scenario("head_on"), MPCAgent(cfg),
                        log_path="/tmp/_mpc_head_on.jsonl")
    assert stats.outcome == "goal"
    assert stats.metrics["min_separation"] > cfg.simulation.collision_radius
    # COLREGs prior: maneuvers may include hold-course/slow-down plans,
    # but a head-on alteration must never be to port.
    from src.sim.recorder import read_episode
    steps = read_episode("/tmp/_mpc_head_on.jsonl")["steps"]
    maneuvers = [s["decision"]["plan"] for s in steps
                 if s["decision"].get("mode") == "mpc"]
    assert maneuvers
    # The INITIAL course alteration is what Rule 14 governs; later port
    # plans (e.g. swinging back to the route once clear) are legitimate.
    first_turn = next((m for m in maneuvers if not m.startswith("hold")),
                      None)
    assert first_turn is not None and first_turn.startswith("stbd")


def test_decisions_carry_cost_decomposition():
    cfg = Config()
    run_episode(cfg, build_scenario("crossing_starboard"), MPCAgent(cfg),
                log_path="/tmp/_mpc_cross.jsonl")
    from src.sim.recorder import read_episode
    steps = read_episode("/tmp/_mpc_cross.jsonl")["steps"]
    candidate_tables = [s["decision"]["candidates"] for s in steps
                        if s["decision"].get("candidates")]
    assert candidate_tables
    entry = candidate_tables[0][0]
    assert {"candidate", "cost", "feasible", "costs"} <= set(entry)
