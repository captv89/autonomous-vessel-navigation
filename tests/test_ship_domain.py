"""Asymmetric ship domain: circular special case, fore/aft/port/stbd bias,
and its effect on the COLREGs passing-distance score."""

import numpy as np

from src.config import Config, ShipDomain
from src.evaluation.benchmark import load_suite, run_benchmark
from src.evaluation.colregs import score_episode
from src.sim.ship_domain import domain_distance


def _dd(own_heading, tx, ty, domain, safe=10.0):
    return domain_distance(0.0, 0.0, own_heading, tx, ty, safe, domain)


def test_circular_domain_reduces_to_range_over_safe():
    d = ShipDomain()  # all 1.0
    for tx, ty in [(10.0, 0.0), (0.0, 7.0), (-5.0, -5.0)]:
        assert _dd(0.0, tx, ty, d) == np.hypot(tx, ty) / 10.0


def test_more_clearance_ahead_than_astern():
    # Heading +x. A target the same range ahead intrudes more (smaller
    # normalized distance) than one astern, because fore > aft.
    d = ShipDomain(fore=1.5, aft=0.5, starboard=1.0, port=1.0)
    ahead = _dd(0.0, 10.0, 0.0, d)
    astern = _dd(0.0, -10.0, 0.0, d)
    assert ahead < astern


def test_starboard_bias():
    # Heading +x: starboard is -y (clockwise). stbd domain larger than port,
    # so a starboard target reads as the deeper intrusion at equal range.
    d = ShipDomain(fore=1.0, aft=1.0, starboard=1.2, port=0.8)
    stbd = _dd(0.0, 0.0, -10.0, d)
    port = _dd(0.0, 0.0, 10.0, d)
    assert stbd < port


def _head_on_steps(min_y):
    # Own ship runs +x along y=0; target runs -x passing at lateral offset
    # min_y. Straight reciprocal -> head-on encounter, no turn.
    own = [(i * 1.0, 0.0) for i in range(60)]
    other = [(60 - i * 1.0, min_y) for i in range(60)]
    steps = []
    for i, ((ox, oy), (bx, by)) in enumerate(zip(own, other)):
        steps.append({
            "t": i * 0.5,
            "vessel": {"x": ox, "y": oy, "heading": 0.0, "speed": 2.0,
                       "sway": 0.0, "turn_rate": 0.0, "rudder_angle": 0.0},
            "obstacles": [{"id": 1, "x": bx, "y": by,
                           "heading": np.pi, "speed": 2.0}],
            "decision": {}, "events": []})
    return steps


def test_scorer_circular_default_unchanged():
    # No domain arg == circular == legacy behavior.
    steps = _head_on_steps(6.0)
    s = score_episode(steps, safe_distance=8.0, collision_radius=0.5)[0]
    assert s.components["safe_distance"] == min(1.0, 6.0 / 8.0)


def test_scorer_applies_domain_at_passing_side():
    # Head-on pass with the target abeam to port (heading +x, target at +y).
    # Enlarging the port domain deepens the intrusion -> lower passing score
    # than the circular default; this is how the domain flows into scoring.
    steps = _head_on_steps(6.0)
    base = score_episode(steps, 8.0, 0.5)[0].components["safe_distance"]
    port_biased = score_episode(
        steps, 8.0, 0.5, domain=ShipDomain(port=2.0))[0]
    assert port_biased.components["safe_distance"] < base


def test_suite_base_override_applies_and_hashes(tmp_path):
    suite = load_suite("benchmarks/v2.yaml")
    suite = {**suite, "scenarios": ["head_on"], "episodes_per_scenario": 1,
             "conditions": {"calm": {}}}
    run_benchmark(Config(), suite, ["classical"], tmp_path)
    import json
    res = json.loads((tmp_path / "results.json").read_text())
    # v2 calm hash must differ from a plain circular calm config.
    from src.evaluation.benchmark import config_hash
    assert res["config_hash"]["calm"] != config_hash(Config())


def test_v2_suite_declares_ship_domain():
    suite = load_suite("benchmarks/v2.yaml")
    assert suite["base"]["ship_domain"]["fore"] == 1.5
