# Submitting a model to VesselNav-Bench

Any navigation policy can be scored on the benchmark by implementing one
small contract — no forking of the harness required.

## 1. Implement the Agent contract

```python
# your_package/your_agent.py
from src.agents.base import Agent, Decision
from src.config import Config
from src.sim.engine import Observation


class YourAgent(Agent):
    name = "your-agent-name"          # appears on the leaderboard

    def __init__(self, config: Config):
        self.config = config          # physics/limits your agent may read

    def reset(self, obs: Observation) -> None:
        """Called once per episode with the initial observation.
        Plan a route, reset internal state, etc."""

    def decide(self, obs: Observation) -> Decision:
        """Called every simulation step (dt = config.simulation.dt).
        Return the helm order for this step."""
        return Decision(
            desired_heading=...,      # radians, world frame (0 = east, CCW+)
            desired_speed=...,        # cells/s, clipped to vessel.max_speed
            explanation={...},        # OPTIONAL but encouraged: anything
        )                             # JSON-serializable describing *why*
```

What your agent observes each step (`Observation`):

| Field | Content |
|---|---|
| `obs.t` | simulation time (s) |
| `obs.vessel` | `x, y, heading, speed (surge), sway, turn_rate, rudder_angle, rudder_command` |
| `obs.obstacles` | per traffic vessel: `id, x, y, heading, speed` (ground truth) |
| `obs.goal` | goal position |
| `obs.world` | static grid (`world.grid[y, x] > 0.5` = land), `width`, `height` |
| `obs.current` | water-current vector (drift estimate) |

The engine converts your desired heading into rudder movement through a PD
autopilot with IMO rate limits — you command *intent*, the physics decides
what the hull does. RL policies typically wrap their network here (see
`src/agents/rl_agent.py` for the built-in example, including how to expose
action probabilities in the explanation).

## 2. Run the benchmark

```bash
uv run python main.py benchmark --suite benchmarks/v1.yaml \
    --agent classical \
    --agent your_package.your_agent:YourAgent \
    --out reports/my-submission
```

Built-in baseline specs: `classical`, `classical-legacy`,
`rl:<model.zip>`, and `rl-shielded:<model.zip>` (the RL policy wrapped in
the predictive runtime safety filter; every intervention is logged in the
decision record and surfaced as `shield_intervention_fraction`).

This produces `leaderboard.md` (ranked table with 95% confidence
intervals), `results.json` (every number, recomputable), and
`episodes/` (a replayable JSONL log of every episode:
`python main.py replay <file>`).

## 3. Rules

- Do not modify the suite file, the physics config, or the engine. Results
  are stamped with a config hash; only matching hashes are comparable.
- Agents receive ground-truth observations; adding sensor noise models is a
  benchmark-version change, not an agent choice.
- Training may use the simulator freely (e.g. via `VesselNavEnv`), but the
  benchmark seeds (`base_seed` onward) are held out by convention: do not
  train on them.
- Report the full leaderboard row including confidence intervals, and keep
  the episode logs — they are the evidence behind your numbers.
