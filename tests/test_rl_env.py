import numpy as np
import pytest

from src.config import Config
from src.rl.env import VesselNavEnv
from src.rl.observation import ObservationBuilder
from src.sim.engine import SimulationEngine
from src.sim.scenarios import build_scenario


def _steady_full(cfg):
    """Action for 'hold heading at full speed' in the configured action mode."""
    heading_idx = list(cfg.rl.heading_actions_deg).index(0.0)
    if cfg.rl.action_mode == "steer_speed":
        speed_idx = list(cfg.rl.speed_factors).index(1.0)
        return [heading_idx, speed_idx]
    return heading_idx


def test_gymnasium_api():
    from gymnasium.utils.env_checker import check_env
    check_env(VesselNavEnv(Config(), scenario_name="random", seed=0),
              skip_render_check=True)


def test_gymnasium_api_steer_only():
    from gymnasium.utils.env_checker import check_env
    cfg = Config()
    cfg.rl.action_mode = "steer_only"
    check_env(VesselNavEnv(cfg, scenario_name="random", seed=0),
              skip_render_check=True)


def test_action_space_by_mode():
    from gymnasium import spaces
    cfg = Config()
    assert cfg.rl.action_mode == "steer_speed"  # G5 default
    env = VesselNavEnv(cfg, scenario_name="random", seed=0)
    assert isinstance(env.action_space, spaces.MultiDiscrete)
    assert list(env.action_space.nvec) == [len(cfg.rl.heading_actions_deg),
                                            len(cfg.rl.speed_factors)]

    cfg.rl.action_mode = "steer_only"
    env = VesselNavEnv(cfg, scenario_name="random", seed=0)
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == len(cfg.rl.heading_actions_deg)


def test_speed_action_modulates_progress():
    """A slower speed factor must reach the goal less quickly: same heading,
    lower engine order => less progress over the held decision."""
    cfg = Config()
    heading_idx = list(cfg.rl.heading_actions_deg).index(0.0)
    full_idx = list(cfg.rl.speed_factors).index(1.0)
    slow_idx = int(np.argmin(cfg.rl.speed_factors))
    assert slow_idx != full_idx

    fast = VesselNavEnv(cfg, scenario_name="open_water", seed=0)
    fast.reset(seed=0)
    _, _, _, _, info_fast = fast.step([heading_idx, full_idx])

    slow = VesselNavEnv(cfg, scenario_name="open_water", seed=0)
    slow.reset(seed=0)
    _, _, _, _, info_slow = slow.step([heading_idx, slow_idx])

    assert (info_slow["reward_components"]["progress"]
            < info_fast["reward_components"]["progress"])


def test_reward_components_unchanged_keys():
    """G5 acceptance: speed actions add no new reward terms."""
    env = VesselNavEnv(Config(), scenario_name="head_on", seed=3)
    env.reset(seed=3)
    _, _, _, _, info = env.step(_steady_full(env.config))
    assert "speed" not in info["reward_components"]
    assert set(info["reward_components"]) <= {
        "heading_change", "time", "progress", "near_miss", "land_proximity",
        "goal", "collision", "grounding", "out_of_bounds"}


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
    _, reward, _, _, info = env.step(_steady_full(env.config))
    assert reward == pytest.approx(sum(info["reward_components"].values()))


def test_progress_reward_sign():
    cfg = Config()
    env = VesselNavEnv(cfg, scenario_name="open_water", seed=0)
    env.reset(seed=0)
    # open_water starts heading straight at the goal: steady = progress.
    _, _, _, _, info = env.step(_steady_full(cfg))
    assert info["reward_components"]["progress"] > 0


def test_episode_seeding_reproducible():
    a = VesselNavEnv(Config(), scenario_name="random", seed=5)
    b = VesselNavEnv(Config(), scenario_name="random", seed=5)
    obs_a, info_a = a.reset(seed=5)
    obs_b, info_b = b.reset(seed=5)
    assert info_a["scenario_seed"] == info_b["scenario_seed"]
    assert np.array_equal(obs_a, obs_b)
