from src.config import Config
from src.agents.classical import ClassicalAgent
from src.evaluation.runner import run_episode
from src.sim.scenarios import build_scenario


def test_open_water_reaches_goal():
    cfg = Config()
    stats = run_episode(cfg, build_scenario("open_water"), ClassicalAgent(cfg))
    assert stats.outcome == "goal"


def test_head_on_resolved_safely():
    cfg = Config()
    stats = run_episode(cfg, build_scenario("head_on"), ClassicalAgent(cfg))
    assert stats.outcome == "goal"
    assert stats.metrics["collisions"] == 0
    assert stats.metrics["min_separation"] > cfg.simulation.collision_radius


def test_decisions_are_explained():
    cfg = Config()
    sc = build_scenario("head_on")
    stats = run_episode(cfg, sc, ClassicalAgent(cfg),
                        log_path="/tmp/_test_episode.jsonl")
    from src.sim.recorder import read_episode
    ep = read_episode("/tmp/_test_episode.jsonl")
    assert ep["header"]["agent"]["name"] == "classical"
    avoidance_steps = [s for s in ep["steps"]
                       if s["decision"]["mode"] == "avoidance"]
    assert avoidance_steps, "head-on should trigger avoidance"
    a = avoidance_steps[0]["decision"]["avoidance"]
    assert "reason" in a and "candidates" in a
    # Every step logs the risk picture.
    assert all("risks" in s["decision"] for s in ep["steps"])
