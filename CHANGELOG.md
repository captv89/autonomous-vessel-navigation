# Changelog

All notable changes to VesselNav-Bench are documented here. The benchmark
suite (`benchmarks/v1.yaml`) is frozen at each major version; results are
only comparable between runs with matching config hashes.

## [Unreleased]

Benchmark v2 work (the frozen v1 exam is untouched; new scenarios and
conditions land in additive suites so v1 numbers stay comparable).

### Added
- **Degraded-perception condition (gap G3).** Seeded perception model
  (`src/sim/perception.py`) degrades *perceived traffic* — Gaussian
  position/course/speed noise, finite update interval (stale fixes on moving
  targets), and occasional target loss — at the shared observation funnel, so
  every agent (classical, MPC, RL) sees the same noise while ground truth
  still drives scoring. Shipped as the `degraded` condition in
  `benchmarks/v2.yaml`; the leaderboard robustness chart now renders any
  number of conditions. Off by default and in v1.
- **Imazu problem set (gap G2)** — 22 canonical 1–4-ship encounter
  geometries as scenarios + a diagnostic suite (`benchmarks/imazu.yaml`).
- **Reactive / COLREGs-compliant traffic (gap G1)** — per-vessel
  `compliance` level; give-way targets take their own starboard action
  (`src/sim/reactive_traffic.py`). Off in frozen v1.

## [1.0.0] - 2026-06-14

Archived on Zenodo: [10.5281/zenodo.20689528](https://doi.org/10.5281/zenodo.20689528)

First public release: a 2D autonomous-vessel navigation simulator and a
frozen, seeded benchmark (**VesselNav-Bench**) for comparing classical,
optimization-based, and learning-based navigation policies under identical
physics, scenarios, and ground-truth scoring.

### Simulator
- Headless `SimulationEngine` shared by every agent and the RL training env
  (one physics path, no drift).
- 3-DOF surge–sway–yaw vessel dynamics (Nomoto yaw channel, sideslip,
  turn-induced speed loss, first-order surge), IMO rudder actuator (35°
  limit, 70°-in-11-s slew), course-over-ground PD autopilot.
- Environmental disturbances: scenario-seeded water current and wind gusts
  (calm vs disturbed benchmark conditions).
- Typed YAML configuration; deterministic, seeded episodes; JSONL per-step
  decision logs with a pygame live + replay viewer.

### Agents (baselines)
- `classical` — A* route + ILOS guidance + predictive COLREGs avoider.
- `classical-legacy` — ablation with the original avoidance module.
- `mpc` — A* route + sampling-based model-predictive control.
- `rl_ppo` — PPO policy (Stable-Baselines3), trained 10M steps with a
  scenario-mixing curriculum.
- `rl_ppo_shielded` — the PPO policy inside a predictive runtime safety
  filter (every intervention logged).

### Benchmark & reporting
- Frozen 9-scenario suite × 2 conditions; Wilson CIs for rates, seeded
  bootstrap CIs for means; quantitative per-rule COLREGs compliance scoring.
- SHA-256 config-hash stamping so only like-for-like runs are compared.
- Trivial external submission via the `Agent` contract; two worked example
  submissions of published methods: Velocity Obstacles (Fiorini & Shiller,
  1998) and Artificial Potential Fields (Khatib, 1986).
- Markdown + self-contained HTML leaderboard (outcome bars, safety-vs-speed
  scatter, per-scenario radar, calm-vs-disturbed robustness chart),
  auto-deployed to GitHub Pages.

### Project
- MIT licensed; `CITATION.cff`; CI (pytest + benchmark smoke); 53 tests.
- Documentation: `README.md`, `docs/SUBMITTING.md`, `docs/SIMULATOR.md`
  (model spec sheet + literature gap analysis).

### Known gaps / roadmap
Simulator-fidelity gaps for a future v2 are tracked in GitHub issues
(#1–#11, roadmap #12) and described in `docs/SIMULATOR.md` §2: reactive
COLREGs-compliant traffic, the Imazu problem set, sensor-noise condition,
ship-domain scoring, RL speed actions, depth/draft, AIS-replay scenarios,
traffic separation schemes, restricted visibility, hull randomization, and
sea state.

[1.0.0]: https://github.com/captv89/autonomous-vessel-navigation/releases/tag/v1.0.0
