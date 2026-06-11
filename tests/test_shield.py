import numpy as np

from src.config import Config
from src.agents.shielded import ShieldedRLAgent
from src.agents.base import Agent, Decision
from src.agents.avoidance import PredictiveAvoider
from src.environment.grid_world import GridWorld
from src.sim.engine import SimulationEngine
from src.sim.scenarios import build_scenario
from src.evaluation.runner import run_episode


class SuicidalAgent(Agent):
    """Test stand-in for a bad policy: always steers at the nearest hazard."""
    name = "suicidal"

    def __init__(self, config):
        self.config = config

    def reset(self, obs):
        pass

    def decide(self, obs):
        if obs.obstacles:
            ob = obs.obstacles[0]
            heading = np.arctan2(ob["y"] - obs.vessel["y"],
                                 ob["x"] - obs.vessel["x"])
        else:
            heading = np.pi  # straight at the western boundary
        return Decision(float(heading), self.config.vessel.cruise_speed, {})


class ShieldedSuicidal(SuicidalAgent):
    """Suicidal policy wrapped in the same shield the RL agent uses."""
    name = "suicidal_shielded"

    def reset(self, obs):
        import copy
        cfg = copy.deepcopy(self.config)
        cfg.avoidance.safe_distance = self.config.avoidance.shield_safe_distance
        self.shield = PredictiveAvoider(cfg, obs.world)
        self.shield.engage_without_traffic = True
        self.shield.always_check_track = True

    def decide(self, obs):
        proposal = super().decide(obs)
        self.shield.current = obs.current
        verdict = self.shield.update(obs.t, obs.vessel,
                                     proposal.desired_heading, obs.obstacles)
        if not verdict.active:
            return proposal
        return Decision(float(verdict.heading),
                        proposal.desired_speed * verdict.speed_factor,
                        {"shield": {"intervened": True}})


def test_shield_prevents_collision_with_traffic():
    cfg = Config()
    scenario = build_scenario("head_on", seed=1)
    unshielded = run_episode(cfg, scenario, SuicidalAgent(cfg))
    shielded = run_episode(cfg, scenario, ShieldedSuicidal(cfg))
    assert unshielded.outcome == "collision"
    assert shielded.outcome != "collision"
    assert shielded.metrics["min_separation"] > cfg.simulation.collision_radius


def test_shield_prevents_boundary_exit():
    cfg = Config()
    cfg.simulation.max_duration = 120.0
    scenario = build_scenario("open_water", seed=1)
    unshielded = run_episode(cfg, scenario, SuicidalAgent(cfg))
    shielded = run_episode(cfg, scenario, ShieldedSuicidal(cfg))
    assert unshielded.outcome == "out_of_bounds"
    assert shielded.outcome not in ("out_of_bounds", "grounding")


def test_shield_passes_safe_decisions_through():
    """A sensible policy heading for open water is never overridden."""
    cfg = Config()
    engine = SimulationEngine(cfg, build_scenario("open_water", seed=1))
    obs = engine.reset()
    shield = PredictiveAvoider(cfg, obs.world)
    shield.engage_without_traffic = True
    goal_heading = np.arctan2(obs.goal[1] - obs.vessel["y"],
                              obs.goal[0] - obs.vessel["x"])
    verdict = shield.update(0.0, obs.vessel, goal_heading, obs.obstacles)
    assert not verdict.active
