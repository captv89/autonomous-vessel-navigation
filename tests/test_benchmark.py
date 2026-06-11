import json

import numpy as np
import pytest

from src.config import Config
from src.evaluation.benchmark import (apply_condition, config_hash,
                                      load_suite, make_agent_factory,
                                      run_benchmark)
from src.evaluation.colregs import score_episode
from src.evaluation.stats import bootstrap_mean, wilson_interval


def test_wilson_interval_bounds():
    ci = wilson_interval(0, 20)
    assert ci["rate"] == 0.0 and ci["low"] == 0.0 and ci["high"] > 0.0
    ci = wilson_interval(20, 20)
    assert ci["rate"] == 1.0 and ci["high"] == 1.0 and ci["low"] < 1.0


def test_bootstrap_mean_seeded_reproducible():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert bootstrap_mean(values) == bootstrap_mean(values)
    ci = bootstrap_mean(values)
    assert ci["low"] <= ci["mean"] <= ci["high"]


def test_config_hash_changes_with_physics_not_training():
    a, b = Config(), Config()
    b.training.total_timesteps = 1
    assert config_hash(a) == config_hash(b)
    b.vessel.nomoto_K = 0.9
    assert config_hash(a) != config_hash(b)


def test_suite_loads():
    suite = load_suite("benchmarks/v1.yaml")
    assert suite["version"] == 1
    assert "calm" in suite["conditions"]


def test_condition_override():
    cfg = apply_condition(Config(), {"environment": {"randomize": True}})
    assert cfg.environment.randomize is True
    assert Config().environment.randomize is False


def test_agent_spec_resolution():
    cfg = Config()
    assert make_agent_factory("classical", cfg)().name == "classical"
    assert make_agent_factory("classical-legacy", cfg)().name == \
        "classical-legacy"
    # external import path uses the same mechanism as third-party agents
    factory = make_agent_factory("src.agents.classical:ClassicalAgent", cfg)
    assert factory().name == "classical"
    with pytest.raises(ValueError):
        make_agent_factory("nonsense", cfg)


def _make_steps(own_xy, obstacle_xy, own_heading=0.0, ob_heading=np.pi,
                dt=0.5):
    steps = []
    for i, ((ox, oy), (bx, by)) in enumerate(zip(own_xy, obstacle_xy)):
        steps.append({
            "t": i * dt,
            "vessel": {"x": ox, "y": oy, "heading": own_heading,
                       "speed": 2.0, "sway": 0.0, "turn_rate": 0.0,
                       "rudder_angle": 0.0},
            "obstacles": [{"id": 1, "x": bx, "y": by,
                           "heading": ob_heading, "speed": 2.0}],
            "decision": {}, "events": []})
    return steps


def test_colregs_head_on_no_turn_scores_low():
    # Both vessels hold course straight at each other and pass at 1 cell.
    own = [(i * 1.0, 0.0) for i in range(60)]
    other = [(60 - i * 1.0, 1.0) for i in range(60)]
    scores = score_episode(_make_steps(own, other), safe_distance=8.0,
                           collision_radius=0.5)
    assert len(scores) == 1
    s = scores[0]
    assert s.encounter == "head-on"
    assert s.initial_turn is None
    assert s.score < 0.2


def test_colregs_starboard_turn_scores_high():
    # Own ship alters 30 deg to starboard early and passes wide.
    steps = []
    for i in range(80):
        t = i * 0.5
        heading = 0.0 if t < 2.0 else -np.radians(35)
        x = 1.0 * i
        y = 0.0 if t < 2.0 else -(t - 2.0) * 1.2
        steps.append({
            "t": t,
            "vessel": {"x": x, "y": y, "heading": heading, "speed": 2.0,
                       "sway": 0.0, "turn_rate": 0.0, "rudder_angle": 0.0},
            "obstacles": [{"id": 1, "x": 70 - i * 1.0, "y": 0.0,
                           "heading": np.pi, "speed": 2.0}],
            "decision": {}, "events": []})
    scores = score_episode(steps, safe_distance=8.0, collision_radius=0.5)
    assert len(scores) == 1
    assert scores[0].initial_turn == "starboard"
    assert scores[0].score > 0.7


def test_benchmark_end_to_end_smoke(tmp_path):
    suite = {
        "name": "smoke", "version": 0, "description": "test",
        "scenarios": ["open_water", "head_on"],
        "episodes_per_scenario": 1, "base_seed": 1,
        "conditions": {"calm": {}},
    }
    report = run_benchmark(Config(), suite, ["classical"], tmp_path)
    assert report.exists()
    results = json.loads((tmp_path / "results.json").read_text())
    agent = results["agents"]["classical"]["conditions"]["calm"]
    assert agent["n_episodes"] == 2
    assert agent["success"]["rate"] == 1.0
    assert "benchmark_score" in agent
    text = report.read_text()
    assert "Leaderboard" in text and "classical" in text
    html = (tmp_path / "index.html").read_text()
    assert "<html" in html and "classical" in html
    assert str(agent["benchmark_score"]) in html
