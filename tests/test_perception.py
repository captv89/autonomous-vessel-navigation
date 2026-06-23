"""Degraded-perception model: determinism, identity when off, and the three
effects (noise, staleness, dropout)."""

from src.config import Config
from src.evaluation.benchmark import load_suite
from src.sim.engine import Observation
from src.sim.perception import Perception


def _obs(t, x, y, heading=0.0, speed=1.0):
    return Observation(
        t=t,
        vessel={"x": 0.0, "y": 0.0, "heading": 0.0, "speed": 0.5,
                "sway": 0.0, "turn_rate": 0.0, "rudder_angle": 0.0,
                "rudder_command": 0.0},
        obstacles=[{"id": 1, "x": x, "y": y, "heading": heading, "speed": speed}],
        goal=(50.0, 50.0),
        world=None)


def _degraded(**over):
    cfg = Config()
    cfg.perception.enabled = True
    for k, v in over.items():
        setattr(cfg.perception, k, v)
    return cfg


def test_disabled_is_identity():
    p = Perception(Config(), seed=1)
    obs = _obs(0.0, 10.0, 20.0)
    out = p.perceive(obs)
    assert out is obs                       # untouched, same object


def test_seeded_realization_matches_across_agents():
    cfg = _degraded(position_noise=2.0, dropout_prob=0.1)
    seq = [_obs(i * 0.5, 10.0 + i, 20.0) for i in range(20)]
    a = [o.obstacles for o in (Perception(cfg, 7).perceive(o) for o in seq)]
    b = [o.obstacles for o in (Perception(cfg, 7).perceive(o) for o in seq)]
    assert a == b
    # A different seed gives a different realization.
    c = [o.obstacles for o in (Perception(cfg, 8).perceive(o) for o in seq)]
    assert a != c


def test_position_noise_perturbs_targets():
    cfg = _degraded(position_noise=2.0)
    out = Perception(cfg, 3).perceive(_obs(0.0, 10.0, 20.0))
    assert out.obstacles[0]["x"] != 10.0 or out.obstacles[0]["y"] != 20.0


def test_staleness_holds_last_fix():
    # No noise: with a 10 s update interval the t=1 report must echo the t=0
    # fix, not the target's true (moved) position.
    cfg = _degraded(update_interval=10.0)
    p = Perception(cfg, 0)
    p.perceive(_obs(0.0, 10.0, 20.0))
    out = p.perceive(_obs(1.0, 15.0, 20.0))
    assert out.obstacles[0]["x"] == 10.0


def test_dropout_loses_target():
    cfg = _degraded(dropout_prob=1.0)
    out = Perception(cfg, 0).perceive(_obs(0.0, 10.0, 20.0))
    assert out.obstacles == []


def test_v2_suite_has_degraded_condition():
    suite = load_suite("benchmarks/v2.yaml")
    assert suite["version"] == 2
    assert set(suite["conditions"]) == {
        "calm", "disturbed", "degraded", "shoaling",
        "hull_randomized", "restricted_visibility"}
    assert suite["conditions"]["degraded"]["perception"]["enabled"] is True
    assert suite["conditions"]["shoaling"]["world"]["depth_model"] == "shoaling"
    assert (suite["conditions"]["hull_randomized"]["vessel"]["randomize_hull"]
            is True)
    assert (suite["conditions"]["restricted_visibility"]["perception"]
            ["restricted_visibility"] is True)
