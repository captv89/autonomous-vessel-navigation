from pathlib import Path

import pytest

from src.config import Config
from src.sim.scenarios import (Scenario, build_scenario, coastal,
                               load_world_file)

WORLDS = Path("worlds")
WORLD_FILES = ["static_only.yaml", "dynamic_only.yaml", "mixed.yaml"]


def test_scenario_dict_roundtrip_preserves_traffic():
    original = coastal(seed=7)
    restored = Scenario.from_dict(original.to_dict())
    assert restored.to_dict() == original.to_dict()
    # Traffic behaviors and waypoints survive the round trip.
    wp = [t for t in restored.traffic if t.behavior == "waypoint"]
    assert wp and wp[0].waypoints == [(50, 50), (45, 65), (30, 55), (45, 35)]


@pytest.mark.parametrize("fname", WORLD_FILES)
def test_world_file_builds(fname):
    config = Config()
    scenario, _ = load_world_file(str(WORLDS / fname))
    world = scenario.make_world(config)
    traffic = scenario.make_traffic()
    assert world is not None and traffic is not None
    for (x, y) in (scenario.start, scenario.goal):
        assert 0 <= x <= config.world.width
        assert 0 <= y <= config.world.height


def test_static_only_has_obstacles_no_traffic():
    scenario = build_scenario(str(WORLDS / "static_only.yaml"))
    assert scenario.traffic == []
    assert scenario.rect_obstacles or scenario.circle_obstacles


def test_dynamic_only_has_traffic_no_obstacles():
    scenario = build_scenario(str(WORLDS / "dynamic_only.yaml"))
    assert scenario.traffic
    assert not scenario.rect_obstacles and not scenario.circle_obstacles


def test_build_scenario_seed_overrides_file():
    scenario = build_scenario(str(WORLDS / "static_only.yaml"), seed=42)
    assert scenario.seed == 42


def test_world_file_environment_override_applies():
    # mixed.yaml declares an environment current; the simulate merge step
    # (replicated here) should surface it on the config.
    _, overrides = load_world_file(str(WORLDS / "mixed.yaml"))
    assert "environment" in overrides
    base = Config().to_dict()
    for section, vals in overrides.items():
        base.setdefault(section, {}).update(vals)
    config = Config.from_dict(base)
    assert config.environment.current_speed > 0.0


def test_registry_scenarios_still_resolve():
    # Path-awareness must not disturb plain registry lookups.
    assert build_scenario("head_on").name == "head_on"
    with pytest.raises(KeyError):
        build_scenario("does_not_exist")
