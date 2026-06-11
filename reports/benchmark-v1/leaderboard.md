# vesselnav-bench v1 - Leaderboard

Generated: 2026-06-11T17:57:59  
Scenarios: open_water, head_on, crossing_starboard, crossing_port, overtaking, coastal, random x 20 episodes (seeds 1000..)  
Config hash: calm: `1baf1e352220ed61`, disturbed: `9571b64e338c5753`

**Benchmark score** = 100 x (0.6 x success rate + 0.25 x COLREGs compliance + 0.15 x path efficiency). All components below; rates show Wilson 95% CIs, means show bootstrap 95% CIs.

## Condition: calm

| Rank | Agent | Score | Success | Collision | Grounding | COLREGs | Time to goal (s) | Path eff. | Min sep. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **classical** | **97.8** | 100.0% [97.3%, 100.0%] | 0.0% [0.0%, 2.7%] | 0.0% [0.0%, 2.7%] | 0.94 (n=108) | 56.9 [53.8, 60.4] | 0.95 [0.93, 0.97] | 14.24 [12.91, 15.92] |
| 2 | **mpc** | **90.4** | 100.0% [97.3%, 100.0%] | 0.0% [0.0%, 2.7%] | 0.0% [0.0%, 2.7%] | 0.89 (n=90) | 168.4 [142.0, 198.9] | 0.54 [0.49, 0.59] | 18.07 [16.46, 19.80] |
| 3 | **rl_ppo_shielded** | **85.6** | 85.7% [79.0%, 90.6%] | 0.0% [0.0%, 2.7%] | 0.0% [0.0%, 2.7%] | 0.79 (n=111) | 45.7 [42.3, 50.5] | 0.95 [0.93, 0.97] | 13.50 [11.86, 15.39] |
| 4 | **classical-legacy** | **85.1** | 84.3% [77.3%, 89.4%] | 0.0% [0.0%, 2.7%] | 15.7% [10.6%, 22.7%] | 0.80 (n=107) | 49.2 [47.5, 51.1] | 0.97 [0.95, 0.98] | 13.04 [11.39, 14.92] |
| 5 | **rl_ppo** | **79.4** | 80.0% [72.6%, 85.8%] | 0.7% [0.1%, 3.9%] | 18.6% [13.0%, 25.8%] | 0.67 (n=107) | 41.2 [39.8, 42.6] | 0.98 [0.97, 0.98] | 12.53 [10.75, 14.52] |

Per-encounter COLREGs compliance:

| Agent | crossing-port | crossing-starboard | head-on |
|---|---|---|---|
| classical | 0.76 (n=25) | 0.99 (n=63) | 1.00 (n=20) |
| mpc | 0.76 (n=26) | 0.92 (n=44) | 1.00 (n=20) |
| rl_ppo_shielded | 0.78 (n=48) | 0.95 (n=43) | 0.50 (n=20) |
| classical-legacy | 0.76 (n=25) | 0.75 (n=62) | 1.00 (n=20) |
| rl_ppo | 0.76 (n=42) | 0.66 (n=45) | 0.50 (n=20) |

## Condition: disturbed

| Rank | Agent | Score | Success | Collision | Grounding | COLREGs | Time to goal (s) | Path eff. | Min sep. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **classical** | **97.6** | 100.0% [97.3%, 100.0%] | 0.0% [0.0%, 2.7%] | 0.0% [0.0%, 2.7%] | 0.94 (n=105) | 61.1 [57.1, 65.6] | 0.94 [0.92, 0.96] | 14.25 [12.86, 15.95] |
| 2 | **rl_ppo_shielded** | **86.3** | 90.7% [84.8%, 94.5%] | 0.7% [0.1%, 3.9%] | 0.7% [0.1%, 3.9%] | 0.74 (n=111) | 55.0 [49.2, 61.7] | 0.89 [0.85, 0.92] | 13.74 [12.08, 15.57] |
| 3 | **mpc** | **84.3** | 90.7% [84.8%, 94.5%] | 0.0% [0.0%, 2.7%] | 1.4% [0.4%, 5.1%] | 0.91 (n=93) | 136.6 [122.0, 153.4] | 0.48 [0.45, 0.52] | 19.17 [17.40, 21.18] |
| 4 | **classical-legacy** | **80.8** | 79.3% [71.8%, 85.2%] | 5.7% [2.9%, 10.9%] | 14.3% [9.4%, 21.0%] | 0.76 (n=109) | 49.1 [46.9, 51.9] | 0.95 [0.93, 0.97] | 12.32 [10.54, 14.30] |
| 5 | **rl_ppo** | **74.9** | 73.6% [65.7%, 80.2%] | 2.9% [1.1%, 7.1%] | 17.1% [11.8%, 24.2%] | 0.65 (n=107) | 40.3 [38.9, 41.9] | 0.98 [0.96, 0.98] | 12.35 [10.55, 14.36] |

Per-encounter COLREGs compliance:

| Agent | crossing-port | crossing-starboard | head-on |
|---|---|---|---|
| classical | 0.76 (n=24) | 0.99 (n=61) | 1.00 (n=20) |
| rl_ppo_shielded | 0.76 (n=45) | 0.83 (n=46) | 0.50 (n=20) |
| mpc | 0.77 (n=27) | 0.95 (n=46) | 1.00 (n=20) |
| classical-legacy | 0.57 (n=25) | 0.76 (n=64) | 1.00 (n=20) |
| rl_ppo | 0.76 (n=42) | 0.61 (n=45) | 0.50 (n=20) |

## Reproducing / submitting

```bash
python main.py benchmark --suite benchmarks/v1.yaml \
    --agent classical --agent rl:models/ppo_vessel \
    --agent your_package.your_module:YourAgent
```

A submission implements the `Agent` contract (`src/agents/base.py`): constructed with a `Config`, then `reset(obs)` and `decide(obs) -> Decision` per step. Every episode is recorded to `episodes/` and can be replayed with `python main.py replay <log>`.

Scores are comparable only between runs with identical config hashes.
