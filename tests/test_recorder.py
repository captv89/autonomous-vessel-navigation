import numpy as np

from src.sim.recorder import EpisodeRecorder, read_episode
from src.evaluation.metrics import compute_metrics, aggregate


def test_roundtrip(tmp_path):
    path = tmp_path / "ep.jsonl"
    with EpisodeRecorder(path) as rec:
        rec.header(scenario={"name": "t"}, agent={"name": "a"}, config={})
        rec.step(t=0.1, vessel={"x": np.float64(1.0), "y": 2.0,
                                "heading": 0.0, "speed": 2.0,
                                "turn_rate": 0.0, "rudder_angle": 0.0},
                 obstacles=[], decision={"mode": "path_following"},
                 events=["goal_reached"])
        rec.summary(outcome="goal", metrics={"success": True})

    ep = read_episode(path)
    assert ep["header"]["scenario"]["name"] == "t"
    assert len(ep["steps"]) == 1
    assert ep["steps"][0]["vessel"]["x"] == 1.0  # numpy converted
    assert ep["summary"]["outcome"] == "goal"


def make_step(t, x, y, events=()):
    return {"t": t, "vessel": {"x": x, "y": y, "heading": 0.0, "speed": 2.0,
                               "turn_rate": 0.0, "rudder_angle": 0.1},
            "obstacles": [], "decision": {"mode": "path_following",
                                          "cross_track_error": 0.5},
            "events": list(events)}


def test_metrics_basics():
    steps = [make_step(0.1 * i, i * 0.2, 0.0) for i in range(100)]
    m = compute_metrics(steps, "goal", goal=(19.8, 0.0),
                        near_miss_distance=8.0)
    assert m["success"]
    assert m["path_efficiency"] > 0.95
    assert m["collisions"] == 0


def test_aggregate():
    eps = [{"success": True, "outcome": "goal", "duration_s": 10.0},
           {"success": False, "outcome": "collision", "duration_s": 20.0}]
    agg = aggregate(eps)
    assert agg["success_rate"] == 0.5
    assert agg["collision_rate"] == 0.5
    # duration is a route-quality metric: aggregated over successes only
    assert agg["mean_duration_s"] == 10.0
