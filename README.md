<div align="center">

# ⚓ VesselNav-Bench

### A transparent benchmark for autonomous ship navigation

**Classical pipelines · Model Predictive Control · Reinforcement Learning —
one physics engine, one exam, one leaderboard.**

[![CI](https://github.com/captv89/autonomous-vessel-navigation/actions/workflows/ci.yml/badge.svg)](https://github.com/captv89/autonomous-vessel-navigation/actions/workflows/ci.yml)
[![Leaderboard](https://img.shields.io/badge/leaderboard-live-2ea44f)](https://captv89.github.io/autonomous-vessel-navigation/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](pyproject.toml)

</div>

A 2D ship navigation simulator and a **frozen, seeded benchmark** for
comparing navigation policies — rule-based, optimization-based (MPC),
learned (RL), shielded hybrids, or **your own** — under identical physics,
identical scenarios, and identical ground-truth scoring, with complete
visibility into every decision any approach makes.

**▶ Live leaderboard: https://captv89.github.io/autonomous-vessel-navigation/**

```bash
# Score any agent on the benchmark and get a ranked leaderboard
uv run python main.py benchmark --suite benchmarks/v1.yaml \
    --agent classical --agent mpc \
    --agent rl:models/ppo_vessel --agent rl-shielded:models/ppo_vessel \
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

| Agent | Score | Success | Collision | Grounding | COLREGs | Time to goal |
|---|---|---|---|---|---|---|
| classical (A* + ILOS + predictive avoidance) | 97.8 | 100% | 0% | 0% | 0.94 | 56.9 s |
| **rl_ppo_shielded** (PPO + runtime safety filter) | 85.6 | 85.7% | **0%** | **0%** | 0.79 | 45.7 s |
| classical-legacy (original avoider, ablation) | 85.1 | 84.3% | 0% | 15.7% | 0.80 | 49.2 s |
| rl_ppo (PPO, 1M steps, unshielded) | 79.4 | 80.0% | 0.7% | 18.6% | 0.67 | 41.2 s |

The shield ablation is the benchmark's headline experiment: wrapping the
learned policy in a classical predictive safety filter eliminates all of
its collisions and groundings while keeping most of its speed advantage —
and every shield intervention is logged, so the division of labor between
policy and shield is measurable per episode.

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

## The five built-in baselines

| Spec | Family | One-line description |
|---|---|---|
| `classical` | Rule-based | A* route + ILOS guidance + predictive COLREGs avoidance override |
| `classical-legacy` | Rule-based (ablation) | Same pipeline with the original avoidance module |
| `mpc` | Optimization | A* route + sampling MPC: one cost function unifies tracking, traffic separation, effort, and a COLREGs prior |
| `rl:<model>` | Learning | PPO over a labeled 35-feature observation, helm-order actions |
| `rl-shielded:<model>` | Hybrid | The PPO policy inside a predictive runtime safety filter; every intervention logged |

```mermaid
flowchart LR
    subgraph Agents
        C[classical<br/>A* + ILOS + avoider]
        M[mpc<br/>A* + sampling MPC]
        R[rl_ppo<br/>PPO policy]
        S[rl_ppo_shielded<br/>PPO + safety filter]
    end
    E[SimulationEngine<br/>3-DOF dynamics · autopilot<br/>traffic · current · gusts]
    L[(JSONL episode logs<br/>per-step decisions)]
    B[VesselNav-Bench<br/>seeded suite + CIs<br/>COLREGs scoring]
    V[Viewer<br/>live + replay]
    H[HTML leaderboard<br/>GitHub Pages]

    Agents -- "Decision (helm order + explanation)" --> E
    E -- "Observation (ground truth)" --> Agents
    E --> L
    L --> B
    L --> V
    B --> H
```

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
