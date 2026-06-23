# Changelog

All notable changes to VesselNav-Bench are documented here. The benchmark
suite (`benchmarks/v1.yaml`) is frozen at each major version; results are
only comparable between runs with matching config hashes.

## [Unreleased]

Benchmark v2 work (the frozen v1 exam is untouched; new scenarios and
conditions land in additive suites so v1 numbers stay comparable).

### Added
- **Traffic separation schemes / Rule 10 (gap G9).** A `TSScheme` chart
  overlay (`src/sim/scenarios.py`) — two opposed lanes around a central
  separation zone — plus `tss_transit` and `tss_crossing` scenarios, added to
  `benchmarks/v2.yaml`. The COLREGs scorer gains Rule 10 terms (`score_tss`):
  crossing angle near 90° (Rule 10(c)), with-the-flow lane use, and zone
  keep-clear, surfaced per episode as `rule10` and aggregated as a mean Rule 10
  score. The terms activate only when a scenario declares a scheme, so all
  other scenarios are unaffected; the separation zone is a soft scored region,
  not a physical wall (a crossing vessel may transit it, but a perpendicular
  crossing's unavoidable zone time is not penalized — only lingering /
  wrong-way use is). The headline benchmark score is unchanged.
- **Depth / under-keel clearance (gap G7).** Opt-in depth chart
  (`world.depth_model: shoaling`) replaces binary land/water: a synthesized
  depth field shoals toward land (`src/environment/grid_world.py:
  apply_shoaling_depth`, depth = min(deep_depth, shoal_slope × metres from
  land)), grounding becomes *charted depth < draft + UKC margin*, and
  sub-clearance water folds into the no-go grid so grounding, A* planning, and
  the lidar perception all respect the shoal apron — no agent retrain needed
  (shoals read like land). Binary land/water is the default special case
  (depth 0 vs deep), so frozen v1 is unchanged; shipped as the `shoaling`
  condition in `benchmarks/v2.yaml` (draft 2.0 m, UKC 0.5 m, ~10 m apron).
  Coastal scenarios gain a realistic shoal apron that narrows fairways; open
  water stays uniformly deep. Viewer depth contours and a depth observation
  channel are deferred follow-ups.
- **Speed actions in the RL action space (gap G5).** `rl.action_mode`
  selects `steer_speed` (default) — a compound `MultiDiscrete([heading,
  speed])` action where the policy modulates engine orders (`rl.speed_factors`
  × `cruise_speed`, default `[1.0, 0.75, 0.5]`) like the classical/MPC
  baselines — or `steer_only` for the original heading-only ablation. The env
  decodes both spaces; `RLAgent` follows the *loaded model's* action space
  (so an older steer-only `Discrete` model keeps loading) and logs separate
  heading and speed probability tables per decision. The reward decomposition
  is unchanged — slowing trades against progress/time, no new reward term.
  A `steer_speed` model still needs training + benchmarking for the final
  paper numbers.
- **Asymmetric ship domain (gap G4).** Four-quadrant Goodwin domain
  (`src/sim/ship_domain.py`) shared by the COLREGs passing-distance scorer
  *and* the predictive avoider's separation check, so "passed too close" and
  "keep clear" agree on the keep-clear zone mariners actually hold (more room
  ahead than astern, biased to starboard) rather than a bare circle. Circular
  is the default special case, so frozen v1 is unchanged; `benchmarks/v2.yaml`
  enables it via a new suite-level `base` override (fore 1.5× / aft 0.5× /
  stbd 1.2× / port 0.8× safe distance). Under v2 the classical agent and RL
  shield keep more clearance to starboard and less to port, as expected.
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
