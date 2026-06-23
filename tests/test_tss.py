"""Traffic separation schemes and Rule 10 scoring (gap G9)."""

import numpy as np
import pytest

from src.sim.scenarios import (build_scenario, Scenario, TSScheme, SCENARIOS)
from src.evaluation.colregs import score_tss, aggregate_tss


def _track(x0, y0, heading_deg, speed=0.5, n=160):
    """A straight constant-speed track as recorded episode steps."""
    h = np.radians(heading_deg)
    return [{"t": float(i),
             "vessel": {"x": x0 + speed * np.cos(h) * i,
                        "y": y0 + speed * np.sin(h) * i,
                        "heading": h, "speed": speed},
             "obstacles": []}
            for i in range(n)]


# --------------------------------------------------------------- scenarios

@pytest.mark.parametrize("name", ["tss_transit", "tss_crossing"])
def test_tss_scenarios_registered_with_scheme(name):
    assert name in SCENARIOS
    sc = build_scenario(name, seed=1000)
    assert sc.geometry == "traffic separation scheme"
    assert isinstance(sc.tss, TSScheme)
    assert len(sc.tss.lanes) == 2           # one each way
    assert sc.tss.zone[0] < sc.tss.zone[1]


def test_scheme_round_trips_through_dict():
    sc = build_scenario("tss_crossing", seed=1000)
    rebuilt = Scenario.from_dict(sc.to_dict())
    assert rebuilt.tss.axis_deg == sc.tss.axis_deg
    assert rebuilt.tss.lanes == sc.tss.lanes
    assert rebuilt.tss.zone == sc.tss.zone


def test_non_tss_scenario_has_no_scheme():
    assert build_scenario("head_on").tss is None


# ------------------------------------------------------------------ scorer

def test_score_tss_empty_inputs():
    assert score_tss([], build_scenario("tss_transit").tss) == {}
    assert score_tss(_track(70, 10, 90), None) == {}


def test_ideal_transit_scores_perfectly():
    tss = build_scenario("tss_transit").tss   # east lane (60,80) flows north 90
    s = score_tss(_track(70, 10, 90), tss)    # north up the east lane, out of zone
    assert s["with_flow"] == pytest.approx(1.0)
    assert s["zone_clear"] == pytest.approx(1.0)
    assert s["score"] == pytest.approx(1.0)


def test_wrong_way_transit_is_penalized():
    tss = build_scenario("tss_transit").tss
    s = score_tss(_track(70, 90, 270), tss)   # south down the northbound lane
    assert s["with_flow"] == pytest.approx(0.0)
    assert s["score"] < 0.5


def test_perpendicular_crossing_scores_high():
    tss = build_scenario("tss_crossing").tss  # N-S lanes, axis 90
    s = score_tss(_track(10, 50, 0), tss)     # due east = right angles
    assert s["crossing_angle"] == pytest.approx(1.0)
    assert s["zone_clear"] > 0.95             # only sampling-level zone excess
    assert s["score"] > 0.95


def test_oblique_crossing_loses_angle_score():
    tss = build_scenario("tss_crossing").tss
    s = score_tss(_track(10, 20, 30), tss)    # 30 deg from the lanes' normal
    # course 30 deg vs N-S axis => 60 deg crossing angle => 60/90.
    assert s["crossing_angle"] == pytest.approx(60.0 / 90.0, abs=1e-2)
    assert s["score"] < 1.0


def test_aggregate_tss_means_scores():
    rows = [{"score": 1.0}, {"score": 0.5}, {}]
    agg = aggregate_tss(rows)
    assert agg["episodes"] == 2
    assert agg["mean_score"] == pytest.approx(0.75)
