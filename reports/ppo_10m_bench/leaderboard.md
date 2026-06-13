# vesselnav-bench v1 - Leaderboard

Generated: 2026-06-13T18:06:30  
Scenarios: open_water, head_on, crossing_starboard, crossing_port, overtaking, coastal, multi_vessel, narrow_channel, random x 20 episodes (seeds 1000..)  
Config hash: calm: `1baf1e352220ed61`, disturbed: `9571b64e338c5753`

**Benchmark score** = 100 x (0.6 x success rate + 0.25 x COLREGs compliance + 0.15 x path efficiency). All components below; rates show Wilson 95% CIs, means show bootstrap 95% CIs.

## Condition: calm

| Rank | Agent | Score | Success | Collision | Grounding | COLREGs | Time to goal (s) | Path eff. | Min sep. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **rl_ppo** | **81.3** | 85.0% [79.1%, 89.5%] | 0.0% [0.0%, 2.1%] | 15.0% [10.5%, 20.9%] | 0.63 (n=189) | 39.8 [39.1, 40.5] | 0.97 [0.96, 0.98] | 10.82 [9.49, 12.30] |

Per-encounter COLREGs compliance:

| Agent | crossing-port | crossing-starboard | head-on |
|---|---|---|---|
| rl_ppo | 0.76 (n=65) | 0.72 (n=64) | 0.39 (n=60) |

## Condition: disturbed

| Rank | Agent | Score | Success | Collision | Grounding | COLREGs | Time to goal (s) | Path eff. | Min sep. |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **rl_ppo** | **81.3** | 86.1% [80.3%, 90.4%] | 2.8% [1.2%, 6.3%] | 11.1% [7.3%, 16.5%] | 0.60 (n=187) | 39.9 [39.1, 40.7] | 0.97 [0.96, 0.98] | 10.77 [9.39, 12.29] |

Per-encounter COLREGs compliance:

| Agent | crossing-port | crossing-starboard | head-on |
|---|---|---|---|
| rl_ppo | 0.76 (n=63) | 0.65 (n=64) | 0.39 (n=60) |


## The agents

- **rl_ppo** — learning (deep reinforcement learning); by V. Ravendranathan (VesselNav-Bench baseline). PPO policy over a labeled 35-feature observation (goal vector, own dynamics, land lidar, nearest traffic); each decision logs the full action probability distribution and value estimate.

## Reproducing / submitting

```bash
python main.py benchmark --suite benchmarks/v1.yaml \
    --agent classical --agent rl:models/ppo_vessel \
    --agent your_package.your_module:YourAgent
```

A submission implements the `Agent` contract (`src/agents/base.py`): constructed with a `Config`, then `reset(obs)` and `decide(obs) -> Decision` per step. Every episode is recorded to `episodes/` and can be replayed with `python main.py replay <log>`.

Scores are comparable only between runs with identical config hashes.
