# vesselnav-bench v1 - Leaderboard

Generated: 2026-06-12T07:18:32  
Scenarios: open_water, head_on, crossing_starboard, crossing_port, overtaking, coastal, random x 20 episodes (seeds 1000..)  
Config hash: calm: `1baf1e352220ed61`, disturbed: `9571b64e338c5753`

**Benchmark score** = 100 x (0.6 x success rate + 0.25 x COLREGs compliance + 0.15 x path efficiency). All components below; rates show Wilson 95% CIs, means show bootstrap 95% CIs.

## Condition: calm

| Rank | Agent | Score | Success | Collision | Grounding | COLREGs | Time to goal (s) | Path eff. | Min sep. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **classical** | **97.8** | 100.0% [97.3%, 100.0%] | 0.0% [0.0%, 2.7%] | 0.0% [0.0%, 2.7%] | 0.94 (n=108) | 56.9 [53.8, 60.4] | 0.95 [0.93, 0.97] | 14.24 [12.91, 15.92] |
| 2 | **mpc** | **96.5** | 99.3% [96.1%, 99.9%] | 0.0% [0.0%, 2.7%] | 0.0% [0.0%, 2.7%] | 0.93 (n=113) | 49.2 [46.8, 51.8] | 0.91 [0.89, 0.93] | 14.58 [13.08, 16.33] |
| 3 | **velocity-obstacles** | **95.6** | 99.3% [96.1%, 99.9%] | 0.0% [0.0%, 2.7%] | 0.0% [0.0%, 2.7%] | 0.88 (n=108) | 55.1 [52.4, 58.2] | 0.93 [0.92, 0.95] | 14.28 [12.88, 15.98] |
| 4 | **classical-legacy** | **85.1** | 84.3% [77.3%, 89.4%] | 0.0% [0.0%, 2.7%] | 15.7% [10.6%, 22.7%] | 0.80 (n=107) | 49.2 [47.5, 51.1] | 0.97 [0.95, 0.98] | 13.04 [11.39, 14.92] |
| 5 | **rl_ppo_shielded** | **84.4** | 84.3% [77.3%, 89.4%] | 0.0% [0.0%, 2.7%] | 0.0% [0.0%, 2.7%] | 0.78 (n=111) | 48.3 [41.1, 59.8] | 0.95 [0.92, 0.98] | 13.98 [12.55, 15.72] |
| 6 | **rl_ppo** | **81.9** | 83.6% [76.5%, 88.8%] | 0.0% [0.0%, 2.7%] | 16.4% [11.2%, 23.4%] | 0.69 (n=108) | 40.9 [40.2, 41.5] | 0.97 [0.96, 0.97] | 12.80 [11.20, 14.64] |

Per-encounter COLREGs compliance:

| Agent | crossing-port | crossing-starboard | head-on |
|---|---|---|---|
| classical | 0.76 (n=25) | 0.99 (n=63) | 1.00 (n=20) |
| mpc | 0.76 (n=30) | 0.99 (n=63) | 1.00 (n=20) |
| velocity-obstacles | 0.76 (n=25) | 0.89 (n=63) | 1.00 (n=20) |
| classical-legacy | 0.76 (n=25) | 0.75 (n=62) | 1.00 (n=20) |
| rl_ppo_shielded | 0.76 (n=47) | 0.94 (n=44) | 0.50 (n=20) |
| rl_ppo | 0.76 (n=45) | 0.70 (n=43) | 0.50 (n=20) |

## Condition: disturbed

| Rank | Agent | Score | Success | Collision | Grounding | COLREGs | Time to goal (s) | Path eff. | Min sep. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **classical** | **97.6** | 100.0% [97.3%, 100.0%] | 0.0% [0.0%, 2.7%] | 0.0% [0.0%, 2.7%] | 0.94 (n=105) | 61.1 [57.1, 65.6] | 0.94 [0.92, 0.96] | 14.25 [12.86, 15.95] |
| 2 | **mpc** | **95.6** | 97.9% [93.9%, 99.3%] | 0.0% [0.0%, 2.7%] | 0.0% [0.0%, 2.7%] | 0.92 (n=111) | 47.6 [45.1, 50.6] | 0.93 [0.91, 0.95] | 14.23 [12.55, 16.03] |
| 3 | **velocity-obstacles** | **94.4** | 97.9% [93.9%, 99.3%] | 0.7% [0.1%, 3.9%] | 0.0% [0.0%, 2.7%] | 0.87 (n=106) | 55.1 [52.1, 58.4] | 0.93 [0.92, 0.95] | 14.00 [12.60, 15.70] |
| 4 | **rl_ppo_shielded** | **85.6** | 90.7% [84.8%, 94.5%] | 0.0% [0.0%, 2.7%] | 2.1% [0.7%, 6.1%] | 0.76 (n=107) | 62.2 [55.2, 70.2] | 0.82 [0.77, 0.86] | 14.64 [13.19, 16.44] |
| 5 | **classical-legacy** | **80.8** | 79.3% [71.8%, 85.2%] | 5.7% [2.9%, 10.9%] | 14.3% [9.4%, 21.0%] | 0.76 (n=109) | 49.1 [46.9, 51.9] | 0.95 [0.93, 0.97] | 12.32 [10.54, 14.30] |
| 6 | **rl_ppo** | **80.2** | 81.4% [74.2%, 87.0%] | 2.1% [0.7%, 6.1%] | 15.0% [10.0%, 21.8%] | 0.68 (n=107) | 40.9 [40.2, 41.7] | 0.96 [0.95, 0.97] | 12.95 [11.32, 14.77] |

Per-encounter COLREGs compliance:

| Agent | crossing-port | crossing-starboard | head-on |
|---|---|---|---|
| classical | 0.76 (n=24) | 0.99 (n=61) | 1.00 (n=20) |
| mpc | 0.74 (n=30) | 0.97 (n=61) | 0.99 (n=20) |
| velocity-obstacles | 0.76 (n=23) | 0.90 (n=63) | 0.91 (n=20) |
| rl_ppo_shielded | 0.76 (n=46) | 0.89 (n=41) | 0.50 (n=20) |
| classical-legacy | 0.57 (n=25) | 0.76 (n=64) | 1.00 (n=20) |
| rl_ppo | 0.76 (n=43) | 0.67 (n=44) | 0.50 (n=20) |


## The agents

- **classical** — rule-based (layered); by V. Ravendranathan (VesselNav-Bench baseline). A* plans the route, ILOS guidance tracks it, and a predictive COLREGs-inspired avoider overrides the helm when traffic conflicts are predicted.
- **classical-legacy** — rule-based (layered); by V. Ravendranathan (VesselNav-Bench baseline). A* plans the route, ILOS guidance tracks it, and a predictive COLREGs-inspired avoider overrides the helm when traffic conflicts are predicted.
- **mpc** — optimization (model predictive control); by V. Ravendranathan (VesselNav-Bench baseline). Solves a local optimal-control problem each second: candidate maneuver plans are rolled out with the real dynamics and scored by one cost combining progress, separation, effort, and a COLREGs prior.
- **rl_ppo** — learning (deep reinforcement learning); by V. Ravendranathan (VesselNav-Bench baseline). PPO policy over a labeled 35-feature observation (goal vector, own dynamics, land lidar, nearest traffic); each decision logs the full action probability distribution and value estimate.
- **rl_ppo_shielded** — hybrid (learning + runtime safety filter); by V. Ravendranathan (VesselNav-Bench baseline). The PPO policy proposes; a classical predictive filter vets every proposal with real-dynamics rollouts and substitutes the nearest safe course when needed. Every intervention is logged.
- **velocity-obstacles** — reactive (velocity space); by Fiorini & Shiller (1998); reference implementation. Picks the velocity closest to the preferred course that lies outside every obstacle's collision cone (starboard-preferring, A* route reference).

## Reproducing / submitting

```bash
python main.py benchmark --suite benchmarks/v1.yaml \
    --agent classical --agent rl:models/ppo_vessel \
    --agent your_package.your_module:YourAgent
```

A submission implements the `Agent` contract (`src/agents/base.py`): constructed with a `Config`, then `reset(obs)` and `decide(obs) -> Decision` per step. Every episode is recorded to `episodes/` and can be replayed with `python main.py replay <log>`.

Scores are comparable only between runs with identical config hashes.
