"""Imazu problem (G2): the 22 canonical encounters reproduced from Sawada
et al. 2021, Table 4. These tests guard the transcription, the NM->cell
mapping, determinism, and runnability through the standard runner."""

import numpy as np
import pytest

from src.config import Config
from src.evaluation.benchmark import load_suite
from src.evaluation.runner import run_episode
from src.agents.classical import ClassicalAgent
from src.sim.scenarios import IMAZU_CASES, SCENARIOS, build_scenario

IMAZU_NAMES = [f"imazu_{c:02d}" for c in range(1, 23)]


def test_all_22_cases_registered_and_selectable():
    assert len(IMAZU_CASES) == 22
    for name in IMAZU_NAMES:
        assert name in SCENARIOS
        assert build_scenario(name).name == name


def test_case_group_sizes_match_paper():
    # Sawada Table 4: cases 1-4 have 1 target, 5-12 two, 13-22 three.
    sizes = {c: len(IMAZU_CASES[c]) for c in range(1, 23)}
    assert [c for c, n in sizes.items() if n == 1] == [1, 2, 3, 4]
    assert [c for c, n in sizes.items() if n == 2] == list(range(5, 13))
    assert [c for c, n in sizes.items() if n == 3] == list(range(13, 23))


def test_own_ship_track_matches_head_on_geometry():
    s = build_scenario("imazu_01")
    assert s.start == (10.0, 50.0)
    assert s.goal == (90.0, 50.0)
    assert s.start_heading_deg == 0.0
    # Case 1 is the pure head-on: one reciprocal target dead ahead.
    assert len(s.traffic) == 1
    t = s.traffic[0]
    assert (round(t.x, 1), round(t.y, 1), t.heading_deg) == (90.0, 50.0, 180.0)


@pytest.mark.parametrize("name", IMAZU_NAMES)
def test_every_target_is_on_a_collision_course_to_the_centre(name):
    # Faithful reproduction: each target heads exactly at the encounter point
    # (50, 50) and its speed is range-scaled so all ships arrive together.
    s = build_scenario(name)
    for t in s.traffic:
        assert 0.0 <= t.x <= 100.0 and 0.0 <= t.y <= 100.0
        bearing = np.degrees(np.arctan2(50.0 - t.y, 50.0 - t.x))
        diff = (bearing - t.heading_deg + 180.0) % 360.0 - 180.0
        assert abs(diff) < 0.5, (name, diff)
        # range 6.0 NM -> speed 0.5; range 4.2 NM -> speed 0.35
        assert t.speed in (0.5, 0.35)


@pytest.mark.parametrize("name", ["imazu_05", "imazu_13"])
def test_scenarios_are_deterministic(name):
    assert build_scenario(name, seed=7).to_dict() == \
        build_scenario(name, seed=7).to_dict()


def test_case_runs_through_standard_runner():
    cfg = Config()
    stats = run_episode(cfg, build_scenario("imazu_13", seed=1000),
                        ClassicalAgent(cfg))
    assert stats.outcome in ("goal", "collision", "grounding", "timeout")


def test_imazu_suite_loads_with_22_scenarios():
    suite = load_suite("benchmarks/imazu.yaml")
    assert len(suite["scenarios"]) == 22
    assert set(suite["scenarios"]) == set(IMAZU_NAMES)
    assert "disturbed" in suite["conditions"]
