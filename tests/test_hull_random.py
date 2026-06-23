"""Own-ship hull parameter randomization (G11)."""

import dataclasses

import numpy as np

from src.config import Config
from src.sim.engine import SimulationEngine
from src.sim.scenarios import build_scenario
from src.vessel.dynamics import HULL_RANDOM_PARAMS, sample_hull


def _engine(cfg, seed):
    return SimulationEngine(cfg, build_scenario("open_water", seed=seed))


def test_disabled_by_default():
    # Off by default: physics unchanged, no parameters sampled.
    cfg = Config()
    assert cfg.vessel.randomize_hull is False
    assert _engine(cfg, 1).hull_params == {}


def test_two_seeds_differ_but_reproducible():
    cfg = Config()
    cfg.vessel.randomize_hull = True

    a1 = _engine(cfg, 1).hull_params
    a2 = _engine(cfg, 1).hull_params
    b = _engine(cfg, 2).hull_params

    assert set(a1) == set(HULL_RANDOM_PARAMS)
    assert a1 == a2                                  # same seed -> reproducible
    assert a1 != b                                   # different seed -> different


def test_jitter_within_configured_range():
    cfg = Config()
    cfg.vessel.randomize_hull = True
    cfg.vessel.hull_jitter = 0.25
    params = _engine(cfg, 5).hull_params
    for name in HULL_RANDOM_PARAMS:
        nominal = getattr(cfg.vessel, name)
        assert 0.75 * nominal <= params[name] <= 1.25 * nominal


def test_sample_hull_returns_jittered_config():
    vc = Config().vessel
    rng = np.random.default_rng(0)
    jittered, sampled = sample_hull(vc, rng)
    assert dataclasses.is_dataclass(jittered)
    assert jittered.nomoto_K == sampled["nomoto_K"]
    # Non-hull fields are carried through untouched.
    assert jittered.max_speed == vc.max_speed
