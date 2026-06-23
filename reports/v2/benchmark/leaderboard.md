# vesselnav-bench v2 - Leaderboard

Generated: 2026-06-23T08:22:10  
Scenarios: open_water, head_on, crossing_starboard, crossing_port, overtaking, coastal, multi_vessel, narrow_channel, tss_transit, tss_crossing, random x 20 episodes (seeds 1000..)  
Config hash: calm: `16e87b9cbe01a104`, disturbed: `8eb3fc5d0e399f8b`, degraded: `73438db0aa38ecac`, shoaling: `9f29bc10527d6257`, hull_randomized: `284c51f7d6bc18f4`, restricted_visibility: `a9551fd639e0949d`

**Benchmark score** = 100 x (0.6 x success rate + 0.25 x COLREGs compliance + 0.15 x path efficiency). All components below; rates show Wilson 95% CIs, means show bootstrap 95% CIs.

## Condition: calm

| Rank | Agent | Score | Success | Collision | Grounding | COLREGs | Time to goal (s) | Path eff. | Min sep. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **classical** | **96.8** | 99.6% [97.5%, 99.9%] | 0.0% [0.0%, 1.7%] | 0.4% [0.1%, 2.5%] | 0.92 (n=212) | 210.2 [204.7, 215.7] | 0.94 [0.92, 0.95] | 10.84 [9.97, 11.85] |
| 2 | **mpc** | **95.7** | 100.0% [98.3%, 100.0%] | 0.0% [0.0%, 1.7%] | 0.0% [0.0%, 1.7%] | 0.87 (n=230) | 209.9 [204.2, 216.0] | 0.94 [0.92, 0.95] | 11.17 [10.30, 12.19] |
| 3 | **rl_ppo_shielded** | **85.4** | 90.9% [86.4%, 94.0%] | 0.0% [0.0%, 1.7%] | 0.0% [0.0%, 1.7%] | 0.70 (n=223) | 256.6 [246.4, 266.9] | 0.89 [0.87, 0.92] | 9.92 [8.75, 11.21] |
| 4 | **rl_ppo** | **60.5** | 61.4% [54.8%, 67.5%] | 27.3% [21.8%, 33.5%] | 11.4% [7.8%, 16.2%] | 0.35 (n=243) | 203.7 [198.7, 209.0] | 1.05 [1.05, 1.05] | 7.15 [5.85, 8.54] |

Per-encounter COLREGs compliance:

| Agent | crossing-port | crossing-starboard | head-on |
|---|---|---|---|
| classical | 0.76 (n=68) | 1.00 (n=84) | 1.00 (n=60) |
| mpc | 0.72 (n=106) | 0.98 (n=64) | 1.00 (n=60) |
| rl_ppo_shielded | 0.65 (n=81) | 0.58 (n=82) | 0.93 (n=60) |
| rl_ppo | 0.48 (n=81) | 0.40 (n=102) | 0.07 (n=60) |

## Condition: disturbed

| Rank | Agent | Score | Success | Collision | Grounding | COLREGs | Time to goal (s) | Path eff. | Min sep. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **rl_ppo_shielded** | **88.1** | 91.8% [87.4%, 94.8%] | 0.9% [0.2%, 3.2%] | 2.7% [1.3%, 5.8%] | 0.76 (n=202) | 269.7 [251.6, 290.1] | 0.94 [0.91, 0.96] | 12.01 [10.74, 13.45] |
| 2 | **mpc** | **86.3** | 86.4% [81.2%, 90.3%] | 1.8% [0.7%, 4.6%] | 0.0% [0.0%, 1.7%] | 0.81 (n=225) | 208.8 [200.4, 218.6] | 0.94 [0.93, 0.96] | 10.73 [9.67, 11.91] |
| 3 | **classical** | **84.0** | 80.5% [74.7%, 85.2%] | 1.8% [0.7%, 4.6%] | 0.9% [0.2%, 3.2%] | 0.88 (n=227) | 247.5 [234.3, 262.0] | 0.92 [0.90, 0.93] | 12.01 [10.96, 13.18] |
| 4 | **rl_ppo** | **71.3** | 74.1% [67.9%, 79.4%] | 13.6% [9.7%, 18.8%] | 10.4% [7.1%, 15.2%] | 0.47 (n=211) | 216.3 [204.6, 231.3] | 1.03 [1.03, 1.04] | 10.94 [9.37, 12.54] |

Per-encounter COLREGs compliance:

| Agent | crossing-port | crossing-starboard | head-on |
|---|---|---|---|
| rl_ppo_shielded | 0.68 (n=75) | 0.73 (n=66) | 0.88 (n=61) |
| mpc | 0.73 (n=80) | 0.85 (n=85) | 0.87 (n=60) |
| classical | 0.74 (n=79) | 0.96 (n=88) | 0.95 (n=60) |
| rl_ppo | 0.54 (n=76) | 0.56 (n=74) | 0.28 (n=61) |

## Condition: degraded

| Rank | Agent | Score | Success | Collision | Grounding | COLREGs | Time to goal (s) | Path eff. | Min sep. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **mpc** | **95.6** | 100.0% [98.3%, 100.0%] | 0.0% [0.0%, 1.7%] | 0.0% [0.0%, 1.7%] | 0.85 (n=237) | 203.6 [198.8, 208.6] | 0.96 [0.95, 0.97] | 9.83 [8.98, 10.88] |
| 2 | **classical** | **92.7** | 93.6% [89.6%, 96.2%] | 2.3% [1.0%, 5.2%] | 0.9% [0.2%, 3.2%] | 0.91 (n=230) | 263.7 [255.5, 272.6] | 0.91 [0.89, 0.92] | 11.35 [10.48, 12.37] |
| 3 | **rl_ppo_shielded** | **89.5** | 95.5% [91.8%, 97.5%] | 2.3% [1.0%, 5.2%] | 0.4% [0.1%, 2.5%] | 0.75 (n=224) | 278.8 [261.3, 299.0] | 0.89 [0.86, 0.92] | 9.71 [8.52, 11.04] |
| 4 | **rl_ppo** | **59.6** | 60.0% [53.4%, 66.2%] | 28.6% [23.1%, 34.9%] | 11.4% [7.8%, 16.2%] | 0.34 (n=242) | 201.5 [197.0, 206.0] | 1.05 [1.05, 1.05] | 6.59 [5.30, 7.98] |

Per-encounter COLREGs compliance:

| Agent | crossing-port | crossing-starboard | head-on |
|---|---|---|---|
| mpc | 0.76 (n=92) | 0.90 (n=85) | 0.92 (n=60) |
| classical | 0.73 (n=72) | 0.99 (n=98) | 1.00 (n=60) |
| rl_ppo_shielded | 0.64 (n=85) | 0.72 (n=79) | 0.96 (n=60) |
| rl_ppo | 0.44 (n=86) | 0.42 (n=96) | 0.07 (n=60) |

## Condition: shoaling

| Rank | Agent | Score | Success | Collision | Grounding | COLREGs | Time to goal (s) | Path eff. | Min sep. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **classical** | **96.8** | 99.6% [97.5%, 99.9%] | 0.0% [0.0%, 1.7%] | 0.4% [0.1%, 2.5%] | 0.92 (n=212) | 210.2 [204.7, 215.7] | 0.94 [0.92, 0.95] | 10.84 [9.97, 11.85] |
| 2 | **mpc** | **95.7** | 100.0% [98.3%, 100.0%] | 0.0% [0.0%, 1.7%] | 0.0% [0.0%, 1.7%] | 0.87 (n=230) | 209.9 [204.2, 216.0] | 0.94 [0.92, 0.95] | 11.17 [10.30, 12.19] |
| 3 | **rl_ppo_shielded** | **85.4** | 90.9% [86.4%, 94.0%] | 0.0% [0.0%, 1.7%] | 0.0% [0.0%, 1.7%] | 0.70 (n=223) | 256.6 [246.4, 266.9] | 0.89 [0.87, 0.92] | 9.92 [8.75, 11.21] |
| 4 | **rl_ppo** | **60.5** | 61.4% [54.8%, 67.5%] | 27.3% [21.8%, 33.5%] | 11.4% [7.8%, 16.2%] | 0.35 (n=243) | 203.7 [198.7, 209.0] | 1.05 [1.05, 1.05] | 7.15 [5.85, 8.54] |

Per-encounter COLREGs compliance:

| Agent | crossing-port | crossing-starboard | head-on |
|---|---|---|---|
| classical | 0.76 (n=68) | 1.00 (n=84) | 1.00 (n=60) |
| mpc | 0.72 (n=106) | 0.98 (n=64) | 1.00 (n=60) |
| rl_ppo_shielded | 0.65 (n=81) | 0.58 (n=82) | 0.93 (n=60) |
| rl_ppo | 0.48 (n=81) | 0.40 (n=102) | 0.07 (n=60) |

## Condition: hull_randomized

| Rank | Agent | Score | Success | Collision | Grounding | COLREGs | Time to goal (s) | Path eff. | Min sep. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **classical** | **96.5** | 99.1% [96.8%, 99.8%] | 0.0% [0.0%, 1.7%] | 0.9% [0.2%, 3.2%] | 0.92 (n=212) | 220.5 [214.4, 227.0] | 0.93 [0.92, 0.95] | 10.83 [9.97, 11.83] |
| 2 | **mpc** | **95.7** | 100.0% [98.3%, 100.0%] | 0.0% [0.0%, 1.7%] | 0.0% [0.0%, 1.7%] | 0.87 (n=230) | 210.6 [204.7, 216.8] | 0.93 [0.92, 0.95] | 11.03 [10.18, 12.04] |
| 3 | **rl_ppo_shielded** | **89.4** | 97.7% [94.8%, 99.0%] | 0.0% [0.0%, 1.7%] | 0.4% [0.1%, 2.5%] | 0.70 (n=223) | 282.2 [263.9, 301.6] | 0.89 [0.86, 0.92] | 9.86 [8.67, 11.16] |
| 4 | **rl_ppo** | **60.5** | 61.4% [54.8%, 67.5%] | 27.3% [21.8%, 33.5%] | 11.4% [7.8%, 16.2%] | 0.35 (n=243) | 203.8 [198.7, 209.0] | 1.05 [1.05, 1.05] | 7.17 [5.88, 8.56] |

Per-encounter COLREGs compliance:

| Agent | crossing-port | crossing-starboard | head-on |
|---|---|---|---|
| classical | 0.76 (n=68) | 1.00 (n=84) | 1.00 (n=60) |
| mpc | 0.73 (n=106) | 0.97 (n=64) | 1.00 (n=60) |
| rl_ppo_shielded | 0.66 (n=81) | 0.58 (n=82) | 0.90 (n=60) |
| rl_ppo | 0.49 (n=81) | 0.40 (n=102) | 0.07 (n=60) |

## Condition: restricted_visibility

| Rank | Agent | Score | Success | Collision | Grounding | COLREGs | Time to goal (s) | Path eff. | Min sep. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **mpc** | **94.1** | 97.7% [94.8%, 99.0%] | 2.3% [1.0%, 5.2%] | 0.0% [0.0%, 1.7%] | 0.84 (n=241) | 202.7 [197.7, 207.9] | 0.97 [0.96, 0.98] | 8.96 [8.03, 10.02] |
| 2 | **classical** | **94.0** | 94.5% [90.7%, 96.9%] | 2.7% [1.3%, 5.8%] | 0.4% [0.1%, 2.5%] | 0.94 (n=232) | 285.5 [275.5, 296.1] | 0.92 [0.90, 0.93] | 11.58 [10.62, 12.63] |
| 3 | **rl_ppo_shielded** | **88.5** | 94.1% [90.1%, 96.5%] | 5.0% [2.8%, 8.7%] | 0.0% [0.0%, 1.7%] | 0.74 (n=225) | 275.3 [256.1, 297.9] | 0.91 [0.88, 0.93] | 9.29 [8.06, 10.67] |
| 4 | **rl_ppo** | **57.4** | 55.5% [48.9%, 61.9%] | 33.2% [27.3%, 39.6%] | 11.4% [7.8%, 16.2%] | 0.37 (n=228) | 197.0 [192.8, 201.4] | 1.05 [1.05, 1.05] | 7.48 [6.02, 8.99] |

Per-encounter COLREGs compliance:

| Agent | crossing-port | crossing-starboard | head-on |
|---|---|---|---|
| mpc | 0.79 (n=93) | 0.86 (n=88) | 0.88 (n=60) |
| classical | 0.89 (n=77) | 0.96 (n=95) | 0.98 (n=60) |
| rl_ppo_shielded | 0.72 (n=83) | 0.75 (n=82) | 0.74 (n=60) |
| rl_ppo | 0.38 (n=84) | 0.54 (n=84) | 0.10 (n=60) |


## The agents

- **classical** — rule-based (layered); by V. Ravendranathan (VesselNav-Bench baseline). A* plans the route, ILOS guidance tracks it, and a predictive COLREGs-inspired avoider overrides the helm when rollouts with the real vessel dynamics predict a conflict on the route.
- **mpc** — optimization (model predictive control); by V. Ravendranathan (VesselNav-Bench baseline). Solves a local optimal-control problem each second: candidate maneuver plans are rolled out with the real dynamics and scored by one cost combining progress, separation, effort, and a COLREGs prior.
- **rl_ppo** — learning (deep reinforcement learning); by V. Ravendranathan (VesselNav-Bench baseline). PPO policy over a labeled 35-feature observation (goal vector, own dynamics, land lidar, nearest traffic); each decision logs the full action probability distribution and value estimate.
- **rl_ppo_shielded** — hybrid (learning + runtime safety filter); by V. Ravendranathan (VesselNav-Bench baseline). The PPO policy proposes; a classical predictive filter vets every proposal with real-dynamics rollouts and substitutes the nearest safe course when needed. Every intervention is logged.

## Reproducing / submitting

```bash
python main.py benchmark --suite benchmarks/v1.yaml \
    --agent classical --agent rl:models/ppo_vessel \
    --agent your_package.your_module:YourAgent
```

A submission implements the `Agent` contract (`src/agents/base.py`): constructed with a `Config`, then `reset(obs)` and `decide(obs) -> Decision` per step. Every episode is recorded to `episodes/` and can be replayed with `python main.py replay <log>`.

Scores are comparable only between runs with identical config hashes.
