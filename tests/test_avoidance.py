import numpy as np
import pytest

from src.config import Config, ShipDomain
from src.agents.avoidance import PredictiveAvoider
from src.environment.grid_world import GridWorld


def make_avoider():
    cfg = Config()
    world = GridWorld(100, 100, 10.0)
    return cfg, PredictiveAvoider(cfg, world)


def own_state(x=10.0, y=50.0, heading=0.0, speed=2.5):
    return {"x": x, "y": y, "heading": heading, "speed": speed,
            "turn_rate": 0.0, "rudder_angle": 0.0}


def test_no_traffic_inactive():
    _, avoider = make_avoider()
    decision = avoider.update(0.0, own_state(), 0.0, [])
    assert not decision.active


def test_distant_traffic_inactive():
    _, avoider = make_avoider()
    traffic = [{"x": 50.0, "y": 90.0, "heading": 0.0, "speed": 2.0}]
    decision = avoider.update(0.0, own_state(), 0.0, traffic)
    assert not decision.active


def test_head_on_turns_starboard():
    """COLREGs Rule 14: head-on traffic -> starboard (negative offset)."""
    _, avoider = make_avoider()
    traffic = [{"x": 60.0, "y": 50.0, "heading": np.pi, "speed": 2.0}]
    decision = avoider.update(0.0, own_state(), 0.0, traffic)
    assert decision.active
    assert decision.offset_deg < 0
    # Full candidate table is exposed for auditability.
    assert any(c.label == "track" for c in decision.candidates)


def test_commitment_holds_between_replans():
    _, avoider = make_avoider()
    traffic = [{"x": 60.0, "y": 50.0, "heading": np.pi, "speed": 2.0}]
    first = avoider.update(0.0, own_state(), 0.0, traffic)
    held = avoider.update(0.3, own_state(x=10.7), 0.0, traffic)
    assert held.active and held.heading == first.heading


def test_rollout_circular_domain_matches_legacy():
    # With the default circular domain, the rollout's domain intrusion is
    # exactly min_separation / safe_distance (the legacy safety test).
    _, avoider = make_avoider()
    obst = [{"x": 30.0, "y": 62.0, "heading": 0.0, "speed": 0.0}]
    r = avoider._rollout(own_state(speed=0.5), obst, target_heading=0.0)
    assert r.min_domain == pytest.approx(r.min_separation / avoider.safe_distance)
    assert r.safe                       # passes ~12 cells off, safe @ 10


def test_rollout_uses_ship_domain_for_safety():
    # Same passing geometry, but a target to port with a doubled port domain
    # now reads as an intrusion the circular test missed.
    cfg = Config()
    cfg.ship_domain = ShipDomain(port=2.0)
    world = GridWorld(100, 100, 10.0)
    avoider = PredictiveAvoider(cfg, world)
    obst = [{"x": 30.0, "y": 62.0, "heading": 0.0, "speed": 0.0}]
    r = avoider._rollout(own_state(speed=0.5), obst, target_heading=0.0)
    assert r.min_domain < r.min_separation / avoider.safe_distance
    assert not r.safe


def test_resumes_when_clear():
    _, avoider = make_avoider()
    traffic = [{"x": 60.0, "y": 50.0, "heading": np.pi, "speed": 2.0}]
    assert avoider.update(0.0, own_state(), 0.0, traffic).active
    # Traffic now well astern and opening.
    behind = [{"x": -40.0, "y": 50.0, "heading": np.pi, "speed": 2.0}]
    decision = avoider.update(10.0, own_state(x=30.0), 0.0, behind)
    assert not decision.active
