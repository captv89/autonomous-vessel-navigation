# The Simulator: What It Models, What It Doesn't, and Why

This document is the honest spec sheet of the VesselNav-Bench simulator —
written for two audiences at once: researchers deciding whether the
benchmark is a fair exam, and mariners who can judge whether the world it
simulates feels like the sea. Every modeled feature lists its config knob;
every gap lists its impact and cost so contributions can be prioritized.

---

## 1. What the simulator models today (in plain terms)

**The ship.** One own ship, seen from above (2D). It does not move like a
car: when you order a turn, the rudder takes time to swing (IMO standard:
full 70° sweep in 11 s), the bow answers slowly (first-order Nomoto yaw
response, gain `vessel.nomoto_K`, lag `vessel.nomoto_T`), the hull crabs
sideways in a turn (sideslip, `vessel.sideslip_gain`), it loses speed
while turning (`vessel.turn_speed_loss`), and engine orders take time to
bite (`vessel.surge_time_constant`). An autopilot steers the
course-over-ground you order, automatically compensating drift — agents
give helm *intent* (course + speed), physics decides what actually
happens.

**The water and the wind.** A uniform water current (set or randomized per
episode, `environment.*`) that carries the own ship *and* all traffic, plus
random wind-gust forces. No waves, no shallow-water effects, no tide.

**Other ships.** Traffic vessels follow fixed behaviors — straight line,
waypoint circuit, or circle — and drift with the same current. By default
they do **not** react to you: they are stand-on vessels that never give way.
A target can optionally be made *reactive* (`compliance: compliant | partial`),
in which case it takes its own starboard give-way action when it is the
give-way vessel — a benchmark-v2 capability (gap G1) kept off in the frozen
v1 exam.

**The chart.** A grid: each cell is water or land. Land is land — there is
no depth, no draft, no under-keel clearance (gap G7). Leaving the mapped
area counts as a navigational failure.

**What an agent perceives.** Ground truth: exact own-ship state, exact
position/course/speed of every traffic vessel (as if from perfect AIS),
the full chart, and the current. There is no sensor noise, no detection
delay, no target loss (gap G3).

**What an episode is.** Start pose → reach the goal circle. Ends on
arrival, collision, grounding, leaving the area, or timeout. Every step of
every episode is logged (ship state, every agent decision with its
reasoning, all traffic) and is replayable.

**What is scored.** Success/collision/grounding rates (Wilson CIs), time,
path efficiency, minimum passing distance, rudder effort, and a
COLREGs-compliance score per encounter (initial turn direction, earliness,
achieved passing distance) — all computed from logged ground truth, never
from agent self-reports.

---

## 2. Gap analysis vs. the maritime DRL literature

Features that published DRL environments (e.g. gym-auv and the broader
2020-2025 literature) consider, compared with ours. Impact = effect on the
benchmark's scientific credibility; Effort = implementation cost in this
codebase.

| # | Feature | Common in literature | Ours today | Impact | Effort |
|---|---------|---------------------|------------|--------|--------|
| G1 | **Reactive / rule-compliant traffic** (target ships that give way, or behave imperfectly) | Mixed; strong papers test against both compliant and rogue targets | ✅ Addressed for v2: per-vessel `compliance` (none/rogue, compliant, partial) — give-way targets take starboard action (`src/sim/reactive_traffic.py`); off in frozen v1 | **High** — stand-on scenarios (Rule 17) are only half-tested when the other ship never acts; multi-ship realism limited | Medium — a traffic behavior layer can reuse the classical avoider |
| G2 | **Imazu problem set** (22 canonical 1-4-ship encounter geometries used across Japanese/Korean COLREGs-DRL papers) | Frequent as a standard exam | Our own 9 scenarios | **High** for comparability — citing Imazu results lets readers cross-reference decades of literature | Low — pure scenario definitions |
| G3 | **Sensor realism** (position/course noise, AIS update intervals, radar dropouts, partial observability) | Common in sim2real-oriented work; Gaussian-noise digital twins | ✅ Addressed for v2: seeded perception model (`src/sim/perception.py`) degrades *perceived traffic* — Gaussian position/course/speed noise, update-interval staleness, target dropout — at the shared observation funnel, so all agents see the same noise; ground truth still drives scoring. Shipped as the `degraded` condition in `benchmarks/v2.yaml`; off in frozen v1 | **High** for any robustness/sim2real claim; Medium for pure algorithm comparison | Medium — noise + latency model in one place, benchmark as a third condition |
| G4 | **Ship domain** (asymmetric safety zone — more clearance ahead than astern, e.g. Fujii/Goodwin) | Standard in compliance evaluation papers | ✅ Addressed for v2: four-quadrant Goodwin domain (`src/sim/ship_domain.py`) shared by the COLREGs passing-distance scorer *and* the predictive avoider's separation check; circular is the default special case, so frozen v1 is unchanged. Enabled in `benchmarks/v2.yaml` (fore 1.5× / aft 0.5× / stbd 1.2× / port 0.8× safe distance) | Medium — affects who "passed too close"; mariners think in domains, not circles | Low-Medium — swap the distance test in scoring + avoidance |
| G5 | **Speed actions for RL** (engine orders as part of the action space) | Usual | ✅ Addressed: `rl.action_mode: steer_speed` (default) gives the policy a compound `MultiDiscrete([heading, speed])` action — `speed_factors` × `cruise_speed` ([1.0, 0.75, 0.5]) — so it modulates engine orders like the classical/MPC baselines; `steer_only` keeps the heading-only space for ablation. The reward decomposition is unchanged (no new term: slowing trades against progress/time as it does for the other agents). A new `steer_speed` model still needs training + benchmarking to land the final numbers | Medium — slight structural handicap for the learned baseline | Low — config flag + retrain |
| G6 | **Waves / sea state** (1st-order motions, added resistance) | Sometimes (3-DOF+ environments) | Not modeled | Low for decision-level benchmarking; matters for control-level studies | High |
| G7 | **Depth / draft** (charted depths, under-keel clearance instead of binary land) | Rare in DRL papers, real for mariners | Binary land/water | Medium for coastal realism | Medium — grid becomes a depth field |
| G8 | **AIS-replay scenarios** (encounters mined from real traffic data) | Emerging best practice for test generation | Synthetic scenarios only | Medium-High for external validity | High — data sourcing + map alignment |
| G9 | **Traffic separation schemes / Rule 10** (lanes, crossing them at right angles) | Rare | Not modeled | Medium — distinctive scenario type | Low-Medium once G7-style charts exist |
| G10 | **Restricted visibility / Rule 19** | Rare | Not modeled | Low-Medium; requires G3 first | Low after G3 |
| G11 | **Hull parameter randomization** (own-ship variety) | Common as domain randomization | Config-selectable but fixed per run | Low-Medium | Low — randomize K/T per episode like the current |

**What we already match or exceed:** Gymnasium API, lidar-style
rangefinders, stochastic seeded environments, current disturbance, full
maneuvering-theory dynamics with actuator limits (gym-auv-class), plus
things most environments *lack*: frozen seeded exams with config hashes,
identical-physics classical/MPC baselines, per-decision audit logs, and
quantitative per-rule COLREGs scoring.

**Recommended order for benchmark v2:** G2 (cheap credibility) → G1 (the
single biggest realism gap) → G3 as a third benchmark condition
("degraded perception") → G4 → G5. G6-G11 are roadmap items.

---

## 3. Where a mariner's eye is needed (open review questions)

You don't need to write code to improve this simulator — judgment calls
below are worth more than features:

1. **Maneuvering sanity.** At cruise speed the own ship needs roughly
   10 cells (~100 m at the nominal 10 m/cell) to complete a 90° turn and
   about 11 s for a full rudder sweep. Does that feel like the vessel
   class you'd expect for a ~25 kn craft? What K/T would you pick for a
   harbor tug vs a feeder vs a yacht? (`configs/default.yaml: vessel`)
2. **Passing distances.** The benchmark calls < 8 cells a near miss and
   ~10 cells a comfortable CPA in open water. Reasonable? Should the
   comfort distance shrink inside the narrow channel scenario (it
   currently doesn't)?
3. **The compliance scorer's idea of good seamanship.** It rewards: an
   early, visible (≥10°) starboard alteration for give-way; holding
   course for stand-on while safe; any-side passing for overtaking. What
   would a watchkeeper add — e.g. penalize a string of small nibbling
   alterations vs one bold one (Rule 8's "readily apparent")?
4. **Scenario realism.** Do the encounter geometries and speeds look like
   situations you've stood watch through? What's missing — fishing
   clusters, a TSS crossing, pilot boarding areas, anchored vessels
   swinging, a stand-on vessel that fails to act until in extremis?
5. **The ship domain shape (G4).** How much more clearance ahead than
   abeam/astern would you score against? *v2 currently uses a Goodwin-style
   domain — fore 1.5× / aft 0.5× / starboard 1.2× / port 0.8× the safe
   distance (`benchmarks/v2.yaml`, `ship_domain`). Are those ratios right
   for this vessel class, and should they shrink in the narrow channel?*

Open an issue (or annotate this file) with anything the sea says and the
simulation doesn't.
