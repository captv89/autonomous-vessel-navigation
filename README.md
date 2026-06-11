# Autonomous Vessel Navigation Simulator + VesselNav-Bench

A 2D ship navigation simulator and a **frozen, seeded benchmark
(VesselNav-Bench)** for comparing navigation policies — classical
(physics/rule-based), reinforcement learning, or your own — under identical
physics, scenarios, and ground-truth scoring, with complete visibility into
every decision any approach makes.

```bash
# Score any agent on the benchmark and get a ranked leaderboard
uv run python main.py benchmark --suite benchmarks/v1.yaml \
    --agent classical --agent rl:models/ppo_vessel \
    --agent your_package.your_module:YourAgent
```

Submissions implement one small contract (`reset`/`decide`) — see
[docs/SUBMITTING.md](docs/SUBMITTING.md). The leaderboard reports success /
collision / grounding rates with Wilson 95% CIs, COLREGs-compliance scores
per encounter type, and efficiency metrics with bootstrap CIs; every number
is backed by a replayable episode log.

Current baselines on
[VesselNav-Bench v1](reports/benchmark-v1/leaderboard.md)
(280 seeded episodes per agent per condition, calm condition shown):

| Agent | Score | Success | Collision | COLREGs | Time to goal |
|---|---|---|---|---|---|
| classical (A* + ILOS + predictive avoidance) | 97.4 | 100% | 0% | 0.94 | 56.4 s |
| classical-legacy (original avoider, ablation) | 85.1 | 84.3% | 0% | 0.80 | 49.2 s |
| rl_ppo (PPO, 1M steps, mixed curriculum) | 79.4 | 80.0% | 0.7% | 0.67 | 41.2 s |

## How the comparison stays fair and transparent

- **One physics engine.** All agents drive the same headless
  `SimulationEngine`: a 3-DOF surge-sway-yaw maneuvering model (Nomoto yaw
  channel, sideslip, turn speed loss, first-order surge), IMO rudder-rate
  limits, course-over-ground PD autopilot, water current + seeded wind
  gusts, dynamic traffic, grid-based land. The RL gymnasium env is a thin
  wrapper over it — there is no second physics implementation to drift.
- **One decision contract.** Every step, an agent returns a `Decision`:
  a helm order *plus a structured explanation*.
  - Classical: active mode (path following / avoidance), per-vessel CPA/TCPA
    and COLREGs encounter classification, cross-track error, and the full
    candidate-maneuver table with predicted miss distances and rejection
    reasons.
  - RL: the labeled observation vector the policy saw, the full action
    probability distribution, and the value estimate. The network weights are
    a black box; what it saw, what it considered, and how confident it was are
    not.
- **One log format.** Every episode is recorded step-by-step to JSONL and can
  be replayed in the viewer; all metrics are computed from ground-truth engine
  state in those logs, never from agent self-reports.
- **Identical exams.** Evaluation runs both agents on identical
  (scenario, seed) pairs.

## Quick start

```bash
uv sync                                   # install dependencies

uv run python main.py scenarios           # list scenarios
uv run python main.py simulate --agent classical --scenario head_on --render
uv run python main.py replay logs/classical_head_on_0.jsonl

# RL: train, then watch / evaluate
uv run python main.py train --timesteps 500000
uv run python main.py simulate --agent rl --model models/ppo_vessel --scenario crossing_starboard --render

# Full comparison report (markdown + figures + per-episode logs)
uv run python main.py evaluate --model models/ppo_vessel --episodes 10
```

`--render` opens the pygame viewer (live); `replay` opens any recorded
episode with pause/step/speed controls and a decision-inspector panel.

## Scenarios

| Name | Description |
|---|---|
| `open_water` | No obstacles; pure path following |
| `head_on` | COLREGs Rule 14 reciprocal encounter |
| `crossing_starboard` | Rule 15, own ship gives way |
| `crossing_port` | Stand-on, but must act if needed |
| `overtaking` | Rule 13, slow vessel ahead |
| `coastal` | Landmasses + mixed traffic |
| `random` | Seeded generator (islands + traffic) used for training/eval suites |

## Architecture

```
main.py                 CLI: scenarios / simulate / train / evaluate /
                        benchmark / replay
configs/default.yaml    every tunable parameter (YAML mirror of src/config.py)
benchmarks/v1.yaml      frozen benchmark suite (scenarios, seeds, conditions)
docs/SUBMITTING.md      how to score your own model on the benchmark
src/
  config.py             typed config dataclasses
  sim/                  engine.py (headless physics loop), scenarios.py,
                        recorder.py (JSONL episode logs)
  agents/               base.py (Agent + Decision contract)
                        classical.py (A* + ILOS + avoidance)
                        avoidance.py (predictive COLREGs avoider, v2)
                        rl_agent.py (SB3 PPO policy wrapper)
  rl/                   observation.py (labeled feature vector, shared
                        train/eval), env.py (gymnasium), train.py (PPO)
  evaluation/           runner.py (seeded suites), metrics.py, report.py,
                        benchmark.py (leaderboard), colregs.py (rule
                        compliance scoring), stats.py (CIs)
  visualization/        viewer.py (pygame live + replay)
  environment/          grid world, traffic vessels, CPA/TCPA detection
  pathfinding/          A* + string pulling
  vessel/               Nomoto vessel model, ILOS/pure-pursuit followers,
                        legacy avoider (kept; select via config)
tests/                  pytest suite
examples/               original phase-1 matplotlib demos (still runnable)
```

### Classical pipeline
A* plans on a safety-inflated grid (waypoints string-pulled and re-split);
ILOS guidance tracks the route with progress-based waypoint switching; a
predictive avoider rolls out candidate course offsets (starboard first, per
COLREGs) using the *real* vessel model + autopilot against constant-velocity
traffic predictions, commits with hysteresis, and resumes the route when it
is predicted safe.

### RL pipeline
PPO (Stable-Baselines3) over a 34-feature observation (goal vector, own
dynamics, 16-ray land lidar, 3 nearest traffic vessels with closing speed).
Discrete course-change actions held for `rl.action_repeat` engine steps —
the same helm-order abstraction the classical agent uses. Reward is decomposed
(progress / time / near-miss / terminal events) and the per-component
breakdown is logged every step during training and evaluation.

## Vessel dynamics

`vessel.model` selects the dynamics (both share the factory used by the
engine *and* by the classical agent's prediction rollouts):

- `fossen3` (default): 3-DOF linear maneuvering model — Nomoto yaw channel
  (K, T keep their meaning), first-order sideslip toward `-gain*u*r`,
  turn-induced speed loss, first-order surge response, water current in the
  kinematics, seeded wind-gust forces.
- `nomoto`: the original first-order yaw model, kept for ablations.

Environmental disturbances live in the `environment` config section and can
be fixed or scenario-seeded (`randomize: true`), so every agent experiences
the identical realization per episode seed.

## Units & scale

World coordinates are **grid cells** (default 100x100, `cell_size` = 10 m per
cell, for reporting). Headings are radians internally, 0 = east,
counter-clockwise positive; config files use degrees where suffixed `_deg`.
Vessel dynamics (Nomoto K/T, IMO rudder rate) are tuned so the maneuvering
timescale fits the arena — see `configs/default.yaml` to change any of it.

## Tests

```bash
uv run pytest
```

Covers dynamics (rudder-rate, Nomoto steady turn), engine termination and
determinism, follower switching (incl. the missed-waypoint regression),
avoidance COLREGs preference and hysteresis, classical agent end-to-end,
gymnasium API/reward decomposition, and recorder/metrics round-trips.
