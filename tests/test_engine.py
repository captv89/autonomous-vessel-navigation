import numpy as np

from src.config import Config
from src.sim.engine import SimulationEngine
from src.sim.scenarios import build_scenario


def run_constant(engine, heading, speed, steps):
    states = []
    for _ in range(steps):
        result = engine.step(heading, speed)
        states.append((result.obs.vessel["x"], result.obs.vessel["y"]))
        if result.done:
            break
    return states, result


def test_deterministic():
    cfg = Config()
    sc = build_scenario("coastal", seed=7)
    a, _ = run_constant(SimulationEngine(cfg, sc), 0.5, 2.0, 200)
    b, _ = run_constant(SimulationEngine(cfg, sc), 0.5, 2.0, 200)
    assert a == b


def test_goal_termination():
    cfg = Config()
    sc = build_scenario("open_water")
    engine = SimulationEngine(cfg, sc)
    heading = np.arctan2(sc.goal[1] - sc.start[1], sc.goal[0] - sc.start[0])
    _, result = run_constant(engine, heading, cfg.vessel.cruise_speed, 5000)
    assert result.done and result.outcome == "goal"


def test_grounding_termination():
    cfg = Config()
    sc = build_scenario("coastal")
    engine = SimulationEngine(cfg, sc)
    # Sail due east from (8, 12) straight into the rectangle at x>=25.
    _, result = run_constant(engine, 0.0, cfg.vessel.cruise_speed, 5000)
    assert result.done and result.outcome == "grounding"


def test_out_of_bounds_termination():
    cfg = Config()
    sc = build_scenario("open_water")
    engine = SimulationEngine(cfg, sc)
    _, result = run_constant(engine, np.pi, cfg.vessel.cruise_speed, 5000)
    assert result.done and result.outcome == "out_of_bounds"


def test_timeout():
    cfg = Config()
    cfg.simulation.max_duration = 5.0
    sc = build_scenario("open_water")
    engine = SimulationEngine(cfg, sc)
    _, result = run_constant(engine, 0.0, 0.0, 5000)
    assert result.done and result.outcome == "timeout"
