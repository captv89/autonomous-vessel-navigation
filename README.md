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

Current leaderboard on
[VesselNav-Bench v1](reports/benchmark-v1/leaderboard.md)
(9 scenarios x 20 seeded episodes per condition = 360 episodes per agent;
calm condition shown):

| # | Agent | Author | Score | Success | Coll. | Ground. | COLREGs | Time |
|---|---|---|---|---|---|---|---|---|
| 1 | classical | VesselNav-Bench | 97.9 | 100% | 0% | 0% | 0.95 | 63.2 s |
| 2 | mpc | VesselNav-Bench | 97.0 | 99.4% | 0% | 0% | 0.93 | **46.6 s** |
| 3 | velocity-obstacles | Fiorini & Shiller (1998) | 93.9 | 99.4% | 0% | 0% | 0.82 | 55.5 s |
| 4 | classical-legacy | VesselNav-Bench | 88.1 | 87.8% | 0% | 12.2% | 0.84 | 49.4 s |
| 5 | rl_ppo_shielded | VesselNav-Bench | 86.5 | 87.8% | **0%** | **0%** | 0.79 | 48.2 s |
| 6 | rl_ppo (PPO, 4M steps) | VesselNav-Bench | 82.4 | 87.2% | 0% | 12.8% | 0.62 | 39.5 s |
| 7 | potential-fields | Khatib (1986) | 48.4 | 44.4% | 55.6% | 0% | 0.33 | 71.9 s |

Three headline observations fall out of the table:

- **Three families plus a published external method, one exam**:
  rule-based, optimization-based, learned, and the classic Velocity
  Obstacles method (submitted exactly as any third party would — see
  [docs/SUBMITTING.md](docs/SUBMITTING.md)) are scored on identical
  seeded episodes with identical ground-truth metrics.
- **Safety shielding works**: wrapping the learned policy in a classical
  predictive filter removes all groundings (16.4% → 0) at a small cost in
  speed, with every intervention logged per decision.
- **Learning closes in but doesn't win (yet)**: 4M training steps lifted
  the pure policy to 83.6% success and zero collisions in calm water, but
  unfamiliar landmass shapes still ground it — the benchmark localizes
  exactly where the generalization gap lives.

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

## Scenarios and the COLREGs rules they test

| Name | COLREGs rule | Situation |
|---|---|---|
| `open_water` | — | No obstacles; pure route following |
| `head_on` | Rule 14 | Reciprocal courses: both vessels alter to starboard, pass port-to-port |
| `crossing_starboard` | Rules 15/16 (give-way) | Traffic from starboard: take early, substantial action to keep clear |
| `crossing_port` | Rule 17 (stand-on) | Traffic from port: hold course and speed while it remains safe |
| `overtaking` | Rule 13 | Slow vessel ahead, same course: overtaker keeps clear, either side |
| `coastal` | Mixed | Landmasses forming a channel + two traffic vessels |
| `multi_vessel` | Rules 14+15 | Two simultaneous conflicts timed to overlap + a passer-by |
| `narrow_channel` | Rule 14 / Rule 9 flavor | Oncoming vessel inside a narrow fairway, limited sea room |
| `random` | Mixed | Seeded random islands and wandering traffic |
| `random_encounter` | 13/14/15 (training only) | Randomized guaranteed collision-course geometry; held out of the exam |

Benchmark conditions: **calm** (no disturbances) and **disturbed**
(scenario-seeded random current up to 0.3 cells/s + wind gusts) — every
agent faces the identical seeded episodes.

## The agents, in plain language

Every agent answers the same question each step — *"given what I can see,
what course and speed do I order?"* — but arrives at the answer
differently:

| Spec | Family | Author | How it decides |
|---|---|---|---|
| `classical` | Rule-based | VesselNav-Bench baseline | A* plans a route, ILOS guidance steers along it, and a predictive COLREGs avoider overrides the helm when a rollout predicts a conflict — explicit rules, layered like a ship's bridge team |
| `classical-legacy` | Rule-based (ablation) | VesselNav-Bench baseline | The same pipeline with the project's original avoidance module — kept to measure how much the avoider matters |
| `mpc` | Optimization | VesselNav-Bench baseline | Solves a small optimal-control problem every second: simulate candidate maneuver plans with the real physics, pick the cheapest by one cost (progress + separation + effort + COLREGs prior) |
| `rl:<model>` | Learning | VesselNav-Bench baseline | A PPO neural policy trained in the simulator; no hand-written rules — behavior emerges from reward. Logs its action probabilities and value estimate every decision |
| `rl-shielded:<model>` | Hybrid | VesselNav-Bench baseline | The PPO policy proposes; a classical predictive filter vets every proposal and substitutes the nearest safe course when needed |
| `submissions...:VOAgent` | Reactive | Fiorini & Shiller (1998) | **Example external submission**: Velocity Obstacles — steer the velocity closest to the preferred course that lies outside every obstacle's collision cone |
| `submissions...:APFAgent` | Reactive | Khatib (1986) | **Example external submission**: Artificial Potential Fields — follow the resultant of attractive (route) and repulsive (traffic, land) forces; historically foundational, known weaknesses included on purpose |

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
