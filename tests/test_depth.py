"""Depth / under-keel-clearance chart (gap G7).

Binary land/water is the special case (depth_model="none"); "shoaling"
synthesizes a depth field that shoals toward land and folds sub-clearance
water into the no-go grid so grounding, planning, and lidar all respect it.
"""

import numpy as np
import pytest

from src.config import Config
from src.environment.grid_world import GridWorld
from src.sim.engine import SimulationEngine
from src.sim.scenarios import build_scenario


def _shoaling_config():
    cfg = Config()
    cfg.world.depth_model = "shoaling"
    return cfg


def test_binary_default_has_no_depth_field():
    cfg = Config()
    assert cfg.world.depth_model == "none"
    world = build_scenario("coastal").make_world(cfg)
    # No depth model => binary land/water, exactly the frozen v1 behavior.
    assert world.depth is None


def test_shoaling_open_water_is_uniformly_deep():
    cfg = _shoaling_config()
    world = build_scenario("open_water").make_world(cfg)
    assert world.depth is not None
    assert np.allclose(world.depth, cfg.world.deep_depth)
    # No land => nothing is folded into the no-go grid; grounding unchanged.
    assert np.all(world.grid < 0.5)


def test_shoaling_folds_subclearance_water_into_nogo():
    # Land strip along the top row; for a column, the EDT distance of the
    # water cell at row y is exactly y cells, so depth[y] = slope*y*cell_size.
    world = GridWorld(width=20, height=20, cell_size=10.0)
    world.add_obstacle(0, 0, 20, 1)            # grid[0, :] = land
    world.apply_shoaling_depth(deep_depth=20.0, shoal_slope=0.1,
                               cell_size=10.0, grounding_threshold=2.5)

    # Land cell: depth 0, no-go.
    assert world.depth[0, 5] == 0.0
    assert world.grid[0, 5] > 0.5
    # depth[y] = min(20, 0.1 * y * 10) = min(20, y) metres.
    assert world.depth[1, 5] == pytest.approx(1.0)
    assert world.depth[3, 5] == pytest.approx(3.0)
    # Apron: depth < 2.5 m folds into no-go (rows 1-2); row 3 (3.0 m) is clear.
    assert world.grid[1, 5] > 0.5
    assert world.grid[2, 5] > 0.5
    assert world.grid[3, 5] < 0.5


def test_grounding_threshold_scales_with_draft():
    """Configurable draft/UKC: a deeper keel widens the no-go apron."""
    def nogo_count(threshold):
        w = GridWorld(width=30, height=30, cell_size=10.0)
        w.add_circular_obstacle(15, 15, 4)
        before = int((w.grid > 0.5).sum())
        w.apply_shoaling_depth(20.0, 0.1, 10.0, threshold)
        return int((w.grid > 0.5).sum()) - before

    assert nogo_count(4.5) > nogo_count(2.5) > 0


@pytest.mark.parametrize("name", ["coastal", "narrow_channel"])
def test_shoaling_keeps_start_navigable(name):
    """The apron narrows fairways but must not strand the start in a shoal."""
    cfg = _shoaling_config()
    scenario = build_scenario(name)
    world = scenario.make_world(cfg)
    sx, sy = scenario.start
    assert world.grid[int(sy), int(sx)] < 0.5


def test_shoaling_engine_runs():
    """End-to-end smoke: an episode under the shoaling chart steps cleanly and
    does not start grounded."""
    cfg = _shoaling_config()
    engine = SimulationEngine(cfg, build_scenario("coastal", seed=7))
    engine.reset()
    result = engine.step(np.radians(75), cfg.vessel.cruise_speed)
    assert result.outcome != "grounding"
