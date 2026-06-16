# vesselnav-worlds v1 - Leaderboard

Generated: 2026-06-15T20:54:53  
Scenarios: worlds/static_only.yaml, worlds/dynamic_only.yaml, worlds/mixed.yaml x 10 episodes (seeds 1000..)  
Config hash: calm: `1baf1e352220ed61`, disturbed: `9571b64e338c5753`

**Benchmark score** = 100 x (0.6 x success rate + 0.25 x COLREGs compliance + 0.15 x path efficiency). All components below; rates show Wilson 95% CIs, means show bootstrap 95% CIs.

## Condition: calm

| Rank | Agent | Score | Success | Collision | Grounding | COLREGs | Time to goal (s) | Path eff. | Min sep. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **mpc** | **72.8** | 100.0% [88.6%, 100.0%] | 0.0% [0.0%, 11.3%] | 0.0% [0.0%, 11.3%] | - | 57.3 [51.7, 62.5] | 0.85 [0.81, 0.89] | 14.21 [12.56, 15.86] |
| 2 | **classical** | **71.8** | 100.0% [88.6%, 100.0%] | 0.0% [0.0%, 11.3%] | 0.0% [0.0%, 11.3%] | - | 76.0 [67.9, 83.9] | 0.79 [0.75, 0.82] | 13.71 [12.92, 14.50] |

Per-encounter COLREGs compliance:


## Condition: disturbed

| Rank | Agent | Score | Success | Collision | Grounding | COLREGs | Time to goal (s) | Path eff. | Min sep. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **classical** | **71.5** | 100.0% [88.6%, 100.0%] | 0.0% [0.0%, 11.3%] | 0.0% [0.0%, 11.3%] | - | 80.2 [72.4, 88.0] | 0.77 [0.74, 0.80] | 14.23 [13.21, 15.21] |
| 2 | **mpc** | **66.7** | 90.0% [74.4%, 96.5%] | 0.0% [0.0%, 11.3%] | 6.7% [1.8%, 21.3%] | - | 58.7 [52.1, 65.0] | 0.85 [0.80, 0.89] | 13.32 [11.76, 14.84] |

Per-encounter COLREGs compliance:



## The agents

- **classical** — rule-based (layered); by V. Ravendranathan (VesselNav-Bench baseline). A* plans the route, ILOS guidance tracks it, and a predictive COLREGs-inspired avoider overrides the helm when rollouts with the real vessel dynamics predict a conflict on the route.
- **mpc** — optimization (model predictive control); by V. Ravendranathan (VesselNav-Bench baseline). Solves a local optimal-control problem each second: candidate maneuver plans are rolled out with the real dynamics and scored by one cost combining progress, separation, effort, and a COLREGs prior.

## Reproducing / submitting

```bash
python main.py benchmark --suite benchmarks/v1.yaml \
    --agent classical --agent rl:models/ppo_vessel \
    --agent your_package.your_module:YourAgent
```

A submission implements the `Agent` contract (`src/agents/base.py`): constructed with a `Config`, then `reset(obs)` and `decide(obs) -> Decision` per step. Every episode is recorded to `episodes/` and can be replayed with `python main.py replay <log>`.

Scores are comparable only between runs with identical config hashes.
