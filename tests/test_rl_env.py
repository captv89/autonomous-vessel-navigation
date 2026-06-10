import numpy as np
import pytest

from src.config import Config
from src.rl.env import VesselNavEnv
from src.rl.observation import ObservationBuilder
from src.sim.engine import SimulationEngine
from src.sim.scenarios import build_scenario


def test_gymnasium_api():
    from gymnasium.utils.env_checker import check_env
    check_env(VesselNavEnv(Config(), scenario_name="random", seed=0),
              skip_render_check=True)


def test_observation_labels_match_size():
    cfg = Config()
    builder = ObservationBuilder(cfg)
    assert len(builder.labels()) == builder.size

    engine = SimulationEngine(cfg, build_scenario("coastal"))
    vec = builder.build(engine.reset())
    assert vec.shape == (builder.size,)
    assert np.all(vec >= -1.0) and np.all(vec <= 1.0)


def test_reward_components_sum():
    env = VesselNavEnv(Config(), scenario_name="head_on", seed=3)
    env.reset(seed=3)
    _, reward, _, _, info = env.step(3)  # steady
    assert reward == pytest.approx(sum(info["reward_components"].values()))


def test_progress_reward_sign():
    cfg = Config()
    env = VesselNavEnv(cfg, scenario_name="open_water", seed=0)
    env.reset(seed=0)
    # open_water starts heading straight at the goal: steady = progress.
    steady = list(cfg.rl.heading_actions_deg).index(0.0)
    _, _, _, _, info = env.step(steady)
    assert info["reward_components"]["progress"] > 0


def test_episode_seeding_reproducible():
    a = VesselNavEnv(Config(), scenario_name="random", seed=5)
    b = VesselNavEnv(Config(), scenario_name="random", seed=5)
    obs_a, info_a = a.reset(seed=5)
    obs_b, info_b = b.reset(seed=5)
    assert info_a["scenario_seed"] == info_b["scenario_seed"]
    assert np.array_equal(obs_a, obs_b)
