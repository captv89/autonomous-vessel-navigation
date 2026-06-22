# vesselnav-bench v1 - Leaderboard

Generated: 2026-06-14T12:51:09  
Scenarios: open_water, head_on, crossing_starboard, crossing_port, overtaking, coastal, multi_vessel, narrow_channel, random x 20 episodes (seeds 1000..)  
Config hash: calm: `1baf1e352220ed61`, disturbed: `9571b64e338c5753`

**Benchmark score** = 100 x (0.6 x success rate + 0.25 x COLREGs compliance + 0.15 x path efficiency). All components below; rates show Wilson 95% CIs, means show bootstrap 95% CIs.

## Condition: calm

| Rank | Agent | Score | Success | Collision | Grounding | COLREGs | Time to goal (s) | Path eff. | Min sep. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **classical** | **97.9** | 100.0% [97.9%, 100.0%] | 0.0% [0.0%, 2.1%] | 0.0% [0.0%, 2.1%] | 0.95 (n=148) | 63.2 [59.9, 66.7] | 0.94 [0.92, 0.95] | 13.34 [12.27, 14.56] |
| 2 | **mpc** | **97.0** | 99.4% [96.9%, 99.9%] | 0.0% [0.0%, 2.1%] | 0.0% [0.0%, 2.1%] | 0.93 (n=193) | 46.6 [44.6, 48.7] | 0.93 [0.91, 0.95] | 13.39 [12.24, 14.75] |
| 3 | **velocity-obstacles** | **93.9** | 99.4% [96.9%, 99.9%] | 0.0% [0.0%, 2.1%] | 0.0% [0.0%, 2.1%] | 0.82 (n=188) | 55.5 [52.9, 58.1] | 0.91 [0.89, 0.93] | 13.32 [12.22, 14.55] |
| 4 | **classical-legacy** | **88.1** | 87.8% [82.2%, 91.8%] | 0.0% [0.0%, 2.1%] | 12.2% [8.2%, 17.8%] | 0.84 (n=167) | 49.4 [48.1, 51.0] | 0.96 [0.95, 0.97] | 13.09 [11.90, 14.40] |
| 5 | **rl_ppo_shielded** | **86.3** | 88.3% [82.8%, 92.2%] | 0.0% [0.0%, 2.1%] | 0.0% [0.0%, 2.1%] | 0.85 (n=192) | 65.8 [60.0, 72.4] | 0.81 [0.77, 0.85] | 12.37 [11.30, 13.59] |
| 6 | **rl_ppo** | **83.4** | 87.8% [82.2%, 91.8%] | 0.0% [0.0%, 2.1%] | 11.7% [7.8%, 17.2%] | 0.63 (n=190) | 38.7 [38.0, 39.4] | 1.00 [0.99, 1.00] | 10.26 [9.01, 11.64] |
| 7 | **potential-fields** | **48.4** | 44.4% [37.4%, 51.7%] | 55.6% [48.3%, 62.6%] | 0.0% [0.0%, 2.1%] | 0.33 (n=188) | 71.9 [61.5, 82.9] | 0.90 [0.86, 0.95] | 5.28 [3.90, 6.74] |

Per-encounter COLREGs compliance:

| Agent | crossing-port | crossing-starboard | head-on |
|---|---|---|---|
| classical | 0.76 (n=25) | 0.99 (n=63) | 1.00 (n=60) |
| mpc | 0.76 (n=50) | 0.99 (n=83) | 1.00 (n=60) |
| velocity-obstacles | 0.76 (n=45) | 0.84 (n=83) | 0.83 (n=60) |
| classical-legacy | 0.76 (n=45) | 0.75 (n=62) | 1.00 (n=60) |
| rl_ppo_shielded | 0.76 (n=66) | 0.95 (n=66) | 0.83 (n=60) |
| rl_ppo | 0.76 (n=64) | 0.71 (n=66) | 0.41 (n=60) |
| potential-fields | 0.36 (n=65) | 0.30 (n=63) | 0.31 (n=60) |

## Condition: disturbed

| Rank | Agent | Score | Success | Collision | Grounding | COLREGs | Time to goal (s) | Path eff. | Min sep. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **mpc** | **96.3** | 98.3% [95.2%, 99.4%] | 0.0% [0.0%, 2.1%] | 0.0% [0.0%, 2.1%] | 0.92 (n=191) | 45.4 [43.3, 47.7] | 0.95 [0.93, 0.96] | 13.22 [11.96, 14.60] |
| 2 | **classical** | **94.7** | 97.2% [93.7%, 98.8%] | 1.7% [0.6%, 4.8%] | 1.1% [0.3%, 4.0%] | 0.90 (n=146) | 66.1 [62.1, 70.3] | 0.92 [0.91, 0.94] | 13.32 [12.19, 14.62] |
| 3 | **velocity-obstacles** | **93.6** | 98.3% [95.2%, 99.4%] | 0.6% [0.1%, 3.1%] | 0.0% [0.0%, 2.1%] | 0.83 (n=186) | 55.0 [52.2, 57.7] | 0.92 [0.90, 0.94] | 13.09 [11.94, 14.35] |
| 4 | **rl_ppo_shielded** | **87.9** | 92.2% [87.4%, 95.3%] | 1.1% [0.3%, 4.0%] | 0.0% [0.0%, 2.1%] | 0.81 (n=192) | 63.0 [56.9, 69.5] | 0.82 [0.78, 0.85] | 12.64 [11.56, 13.88] |
| 5 | **classical-legacy** | **84.9** | 83.9% [77.8%, 88.5%] | 4.4% [2.3%, 8.5%] | 11.1% [7.3%, 16.5%] | 0.81 (n=169) | 49.3 [47.4, 51.5] | 0.95 [0.93, 0.96] | 12.58 [11.28, 13.95] |
| 6 | **rl_ppo** | **83.2** | 87.2% [81.6%, 91.3%] | 0.0% [0.0%, 2.1%] | 12.8% [8.7%, 18.4%] | 0.64 (n=189) | 38.3 [37.7, 39.0] | 1.00 [0.99, 1.01] | 10.08 [8.88, 11.39] |
| 7 | **potential-fields** | **49.5** | 46.1% [39.0%, 53.4%] | 53.3% [46.1%, 60.5%] | 0.0% [0.0%, 2.1%] | 0.31 (n=189) | 63.3 [54.2, 72.9] | 0.94 [0.90, 0.97] | 5.56 [4.18, 7.10] |

Per-encounter COLREGs compliance:

| Agent | crossing-port | crossing-starboard | head-on |
|---|---|---|---|
| mpc | 0.75 (n=50) | 0.98 (n=81) | 0.99 (n=60) |
| classical | 0.76 (n=25) | 0.99 (n=61) | 0.87 (n=60) |
| velocity-obstacles | 0.74 (n=43) | 0.87 (n=83) | 0.85 (n=60) |
| rl_ppo_shielded | 0.76 (n=66) | 0.91 (n=66) | 0.76 (n=60) |
| classical-legacy | 0.66 (n=45) | 0.76 (n=64) | 1.00 (n=60) |
| rl_ppo | 0.76 (n=65) | 0.71 (n=64) | 0.42 (n=60) |
| potential-fields | 0.44 (n=55) | 0.27 (n=74) | 0.24 (n=60) |


## The agents

- **classical** — rule-based (layered); by V. Ravendranathan (VesselNav-Bench baseline). A* plans the route, ILOS guidance tracks it, and a predictive COLREGs-inspired avoider overrides the helm when rollouts with the real vessel dynamics predict a conflict on the route.
- **classical-legacy** — rule-based (layered, original avoider); by V. Ravendranathan (VesselNav-Bench baseline). Ablation of the classical pipeline: same A* route and ILOS guidance, but collision avoidance uses the project's original commitment-based avoider, which lacks route-aware rollouts and reduced-speed escape maneuvers. Kept to quantify how much the avoidance layer contributes.
- **mpc** — optimization (model predictive control); by V. Ravendranathan (VesselNav-Bench baseline). Solves a local optimal-control problem each second: candidate maneuver plans are rolled out with the real dynamics and scored by one cost combining progress, separation, effort, and a COLREGs prior.
- **rl_ppo** — learning (deep reinforcement learning); by V. Ravendranathan (VesselNav-Bench baseline). PPO policy over a labeled 35-feature observation (goal vector, own dynamics, land lidar, nearest traffic); each decision logs the full action probability distribution and value estimate.
- **rl_ppo_shielded** — hybrid (learning + runtime safety filter); by V. Ravendranathan (VesselNav-Bench baseline). The PPO policy proposes; a classical predictive filter vets every proposal with real-dynamics rollouts and substitutes the nearest safe course when needed. Every intervention is logged.
- **velocity-obstacles** — reactive (velocity space); by Fiorini & Shiller (1998); reference implementation. Picks the velocity closest to the preferred course that lies outside every obstacle's collision cone (starboard-preferring, A* route reference).
- **potential-fields** — reactive (force field); by Khatib (1986); reference implementation. Steers along the resultant of an attractive force toward the route and repulsive forces from traffic and land. Historically foundational; known to suffer local minima and oscillation.

## Reproducing / submitting

```bash
python main.py benchmark --suite benchmarks/v1.yaml \
    --agent classical --agent rl:models/ppo_vessel \
    --agent your_package.your_module:YourAgent
```

A submission implements the `Agent` contract (`src/agents/base.py`): constructed with a `Config`, then `reset(obs)` and `decide(obs) -> Decision` per step. Every episode is recorded to `episodes/` and can be replayed with `python main.py replay <log>`.

Scores are comparable only between runs with identical config hashes.
