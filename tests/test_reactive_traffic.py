"""Reactive / COLREGs-compliant traffic (gap G1): targets that give way.

Guards the acceptance criteria: a stand-on scenario where the target gives
way while the own ship holds; per-vessel compliance selection; determinism;
and back-compatibility (default traffic is non-reactive, so frozen scenarios
are unchanged)."""

import numpy as np
import pytest

from src.config import Config
from src.sim.engine import SimulationEngine
from src.sim.scenarios import Scenario, TrafficSpec, build_scenario


def _crossing(compliance):
    # Own ship eastbound (10,50)->(90,50); target crossing from the own
    # ship's PORT, so the own ship is stand-on (Rule 17) and the target is
    # the give-way vessel (own ship on its starboard).
    return Scenario(name="stand_on", start=(10, 50), start_heading_deg=0,
                    goal=(90, 50), seed=1,
                    traffic=[TrafficSpec(x=55, y=90, heading_deg=-90,
                                         speed=0.5, compliance=compliance)])


def _run_holding(scenario):
    """Run with the own ship holding a straight course to the goal (a pure
    stand-on policy). Returns (outcome, min_separation, own-heading spread,
    most-starboard target alteration in deg, final own xy)."""
    cfg = Config()
    eng = SimulationEngine(cfg, scenario)
    obs = eng.reset()
    nominal = np.radians(scenario.traffic[0].heading_deg)
    min_sep = float("inf")
    cmd_headings, target_alter = [], []
    while True:
        cmd = np.arctan2(obs.goal[1] - obs.vessel["y"],
                         obs.goal[0] - obs.vessel["x"])
        cmd_headings.append(cmd)
        res = eng.step(cmd, cfg.vessel.cruise_speed)
        obs = res.obs
        for ob in obs.obstacles:
            min_sep = min(min_sep, float(np.hypot(ob["x"] - obs.vessel["x"],
                                                  ob["y"] - obs.vessel["y"])))
            d = np.arctan2(np.sin(ob["heading"] - nominal),
                           np.cos(ob["heading"] - nominal))
            target_alter.append(np.degrees(d))   # negative = starboard
        if res.done:
            break
    spread = np.degrees(np.ptp(cmd_headings))
    return (res.outcome, min_sep, spread, min(target_alter),
            (obs.vessel["x"], obs.vessel["y"]))


def test_traffic_is_non_reactive_by_default():
    assert TrafficSpec(0, 0, 0, 0).compliance == "none"
    # A built-in scenario's targets hold their nominal course unchanged.
    sc = build_scenario("crossing_starboard")
    eng = SimulationEngine(Config(), sc)
    eng.reset()
    nominal = [mv.obstacle.heading for mv in eng.traffic.obstacles]
    for _ in range(200):
        if eng.step(0.0, 0.5).done:
            break
    assert [mv.obstacle.heading for mv in eng.traffic.obstacles] == nominal


def test_compliant_target_gives_way_while_own_ship_holds():
    outcome, sep, own_spread, alter, _ = _run_holding(_crossing("compliant"))
    # Target took a clear starboard alteration...
    assert alter < -20.0
    # ...the own ship genuinely held course (commanded heading barely moved)...
    assert own_spread < 5.0
    # ...and they passed clear.
    assert outcome == "goal"
    assert sep >= Config().simulation.collision_radius


def test_giving_way_increases_separation_over_non_reactive():
    _, sep_none, _, alter_none, _ = _run_holding(_crossing("none"))
    _, sep_comp, _, _, _ = _run_holding(_crossing("compliant"))
    assert alter_none == pytest.approx(0.0, abs=1e-6)   # 'none' never turns
    assert sep_comp > sep_none


def test_reactive_episode_is_deterministic():
    a = _run_holding(_crossing("compliant"))
    b = _run_holding(_crossing("compliant"))
    assert a[1] == b[1] and a[4] == b[4]


def test_unknown_compliance_is_rejected():
    sc = Scenario(name="bad", start=(10, 50), start_heading_deg=0,
                  goal=(90, 50),
                  traffic=[TrafficSpec(50, 50, 0, 0.4, compliance="complaint")])
    with pytest.raises(ValueError):
        sc.make_traffic()


def test_compliance_survives_dict_roundtrip():
    original = _crossing("partial")
    restored = Scenario.from_dict(original.to_dict())
    assert restored.traffic[0].compliance == "partial"
    assert restored.to_dict() == original.to_dict()
