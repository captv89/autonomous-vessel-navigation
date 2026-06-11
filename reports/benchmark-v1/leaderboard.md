# vesselnav-bench v1 - Leaderboard

Generated: 2026-06-11T12:35:15  
Scenarios: open_water, head_on, crossing_starboard, crossing_port, overtaking, coastal, random x 20 episodes (seeds 1000..)  
Config hash: calm: `b051e3e0543e5d14`, disturbed: `0f61f870b15c8da8`

**Benchmark score** = 100 x (0.6 x success rate + 0.25 x COLREGs compliance + 0.15 x path efficiency). All components below; rates show Wilson 95% CIs, means show bootstrap 95% CIs.

## Condition: calm

| Rank | Agent | Score | Success | Collision | Grounding | COLREGs | Time to goal (s) | Path eff. | Min sep. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **classical** | **97.4** | 100.0% [97.3%, 100.0%] | 0.0% [0.0%, 2.7%] | 0.0% [0.0%, 2.7%] | 0.94 (n=109) | 56.4 [53.4, 59.7] | 0.93 [0.91, 0.95] | 14.02 [12.67, 15.72] |
| 2 | **classical-legacy** | **85.1** | 84.3% [77.3%, 89.4%] | 0.0% [0.0%, 2.7%] | 15.7% [10.6%, 22.7%] | 0.80 (n=107) | 49.2 [47.5, 51.1] | 0.97 [0.95, 0.98] | 13.04 [11.39, 14.92] |
| 3 | **rl_ppo** | **79.4** | 80.0% [72.6%, 85.8%] | 0.7% [0.1%, 3.9%] | 18.6% [13.0%, 25.8%] | 0.67 (n=107) | 41.2 [39.8, 42.6] | 0.98 [0.97, 0.98] | 12.53 [10.75, 14.52] |

Per-encounter COLREGs compliance:

| Agent | crossing-port | crossing-starboard | head-on |
|---|---|---|---|
| classical | 0.76 (n=26) | 0.99 (n=63) | 1.00 (n=20) |
| classical-legacy | 0.76 (n=25) | 0.75 (n=62) | 1.00 (n=20) |
| rl_ppo | 0.76 (n=42) | 0.66 (n=45) | 0.50 (n=20) |

## Condition: disturbed

| Rank | Agent | Score | Success | Collision | Grounding | COLREGs | Time to goal (s) | Path eff. | Min sep. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **classical** | **97.4** | 100.0% [97.3%, 100.0%] | 0.0% [0.0%, 2.7%] | 0.0% [0.0%, 2.7%] | 0.94 (n=106) | 57.5 [54.1, 61.3] | 0.93 [0.91, 0.94] | 14.08 [12.70, 15.80] |
| 2 | **classical-legacy** | **80.8** | 79.3% [71.8%, 85.2%] | 5.7% [2.9%, 10.9%] | 14.3% [9.4%, 21.0%] | 0.76 (n=109) | 49.1 [46.9, 51.9] | 0.95 [0.93, 0.97] | 12.32 [10.54, 14.30] |
| 3 | **rl_ppo** | **74.9** | 73.6% [65.7%, 80.2%] | 2.9% [1.1%, 7.1%] | 17.1% [11.8%, 24.2%] | 0.65 (n=107) | 40.3 [38.9, 41.9] | 0.98 [0.96, 0.98] | 12.35 [10.55, 14.36] |

Per-encounter COLREGs compliance:

| Agent | crossing-port | crossing-starboard | head-on |
|---|---|---|---|
| classical | 0.77 (n=25) | 0.99 (n=61) | 1.00 (n=20) |
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
