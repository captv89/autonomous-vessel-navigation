# The Imazu Problem in VesselNav-Bench

The **Imazu problem** is a set of 22 canonical encounter geometries (1–3
target ships) used across the COLREGs / deep-RL collision-avoidance
literature as a standard exam. Encoding it here lets readers cross-reference
VesselNav results against decades of prior work.

This is a **diagnostic suite**, not the frozen v1 exam. It is part of the
benchmark-v2 roadmap (gap **G2** in [`SIMULATOR.md`](SIMULATOR.md) §2).

## Source

Geometries are transcribed verbatim from **Table 4** of:

> R. Sawada, K. Sato, T. Majima (2021). *Automatic ship collision avoidance
> using deep reinforcement learning with LSTM in continuous action spaces.*
> Journal of Marine Science and Technology 26, 509–524.
> <https://doi.org/10.1007/s00773-020-00755-0>

The cases originate with H. Imazu (1987), *Research on collision avoidance
manoeuvre* (PhD thesis, University of Tokyo).

## Convention and the VesselNav mapping

In the paper, the own ship (OS) starts at `(X, Y) = (-6.0, 0.0)` NM heading
`ψ = 0°` toward a waypoint at `(6.0, 0.0)`. Each target ship (TS) holds a
straight collision course to the **origin**, with its speed set so that, if
nobody manoeuvres, every ship reaches the origin at the same time
(TCPA = 30 min).

Sawada's coordinate frame matches ours in handedness and heading convention
(`0° = +x`, counter-clockwise positive), so the only transform needed is a
translate + scale into our `100×100`-cell world:

```
world_x = 50 + (40/6) · X_NM          # (-6, 0) NM → (10, 50);  (6, 0) NM → (90, 50)
world_y = 50 + (40/6) · Y_NM
heading = ψ                            # unchanged
speed   = 0.5 · range_NM / 6.0         # range-scaled so all ships still meet at the centre
```

So the own ship runs `(10, 50) → (90, 50)` cells — the same track as the
built-in `head_on` scenario — and every target keeps the paper's
simultaneous-arrival timing. (The paper additionally jitters the OS heading
±5° per episode for generalisation; we keep it deterministic so the suite is
reproducible — disturbances instead come from the benchmark's `disturbed`
condition.)

## Case mapping

`our case name` ↔ Sawada case number is the identity: `imazu_07` is Sawada
case 7. Cases **1–4** have one target ship (a two-ship encounter), **5–12**
two, and **13–22** three.

| Our name | Sawada case | #TS | Sawada targets `(X_NM, Y_NM, ψ°)` | Our world `(x, y, heading°, speed)` |
|---|---|---|---|---|
| `imazu_01` | 1 | 1 | (6.000, 0.000, 180°) | (90.0, 50.0, 180°, 0.5) |
| `imazu_02` | 2 | 1 | (0.000, 6.000, -90°) | (50.0, 90.0, -90°, 0.5) |
| `imazu_03` | 3 | 1 | (-4.200, 0.000, 0°) | (22.0, 50.0, 0°, 0.35) |
| `imazu_04` | 4 | 1 | (-4.243, -4.243, 45°) | (21.7, 21.7, 45°, 0.5) |
| `imazu_05` | 5 | 2 | (6.000, 0.000, 180°); (0.000, 6.000, -90°) | (90.0, 50.0, 180°, 0.5); (50.0, 90.0, -90°, 0.5) |
| `imazu_06` | 6 | 2 | (-5.909, 1.042, -10°); (-4.243, 4.243, -45°) | (10.6, 56.9, -10°, 0.5); (21.7, 78.3, -45°, 0.5) |
| `imazu_07` | 7 | 2 | (-4.200, 0.000, 0°); (-4.243, 4.243, -45°) | (22.0, 50.0, 0°, 0.35); (21.7, 78.3, -45°, 0.5) |
| `imazu_08` | 8 | 2 | (6.000, 0.000, 180°); (0.000, 6.000, -90°) | (90.0, 50.0, 180°, 0.5); (50.0, 90.0, -90°, 0.5) |
| `imazu_09` | 9 | 2 | (-5.196, 3.000, -30°); (0.000, 6.000, -90°) | (15.4, 70.0, -30°, 0.5); (50.0, 90.0, -90°, 0.5) |
| `imazu_10` | 10 | 2 | (0.000, 6.000, -90°); (-5.796, -1.553, 15°) | (50.0, 90.0, -90°, 0.5); (11.4, 39.6, 15°, 0.5) |
| `imazu_11` | 11 | 2 | (0.000, -6.000, 90°); (-5.196, 3.000, -30°) | (50.0, 10.0, 90°, 0.5); (15.4, 70.0, -30°, 0.5) |
| `imazu_12` | 12 | 2 | (-4.243, 4.243, -45°); (-5.909, 1.042, -10°) | (21.7, 78.3, -45°, 0.5); (10.6, 56.9, -10°, 0.5) |
| `imazu_13` | 13 | 3 | (6.000, 0.000, 180°); (-5.909, -1.042, 10°); (-4.243, -4.243, 45°) | (90.0, 50.0, 180°, 0.5); (10.6, 43.1, 10°, 0.5); (21.7, 21.7, 45°, 0.5) |
| `imazu_14` | 14 | 3 | (-5.909, 1.042, -10°); (-4.243, 4.243, -45°); (0.000, 6.000, -90°) | (10.6, 56.9, -10°, 0.5); (21.7, 78.3, -45°, 0.5); (50.0, 90.0, -90°, 0.5) |
| `imazu_15` | 15 | 3 | (-4.200, 0.000, 0°); (-4.243, 4.243, -45°); (0.000, 6.000, -90°) | (22.0, 50.0, 0°, 0.35); (21.7, 78.3, -45°, 0.5); (50.0, 90.0, -90°, 0.5) |
| `imazu_16` | 16 | 3 | (-2.970, -2.970, 45°); (0.000, -6.000, 90°); (0.000, 6.000, -90°) | (30.2, 30.2, 45°, 0.35); (50.0, 10.0, 90°, 0.5); (50.0, 90.0, -90°, 0.5) |
| `imazu_17` | 17 | 3 | (-4.200, 0.000, 0°); (-5.909, -1.042, 10°); (-4.243, 4.243, -45°) | (22.0, 50.0, 0°, 0.35); (10.6, 43.1, 10°, 0.5); (21.7, 78.3, -45°, 0.5) |
| `imazu_18` | 18 | 3 | (4.243, 4.243, -135°); (-5.796, 1.553, -15°); (-5.196, 3.000, -30°) | (78.3, 78.3, -135°, 0.5); (11.4, 60.4, -15°, 0.5); (15.4, 70.0, -30°, 0.5) |
| `imazu_19` | 19 | 3 | (-5.796, -1.553, 15°); (-5.796, 1.553, -15°); (4.243, 4.243, -135°) | (11.4, 39.6, 15°, 0.5); (11.4, 60.4, -15°, 0.5); (78.3, 78.3, -135°, 0.5) |
| `imazu_20` | 20 | 3 | (-4.200, 0.000, 0°); (-5.796, 1.553, -15°); (0.000, 6.000, -90°) | (22.0, 50.0, 0°, 0.35); (11.4, 60.4, -15°, 0.5); (50.0, 90.0, -90°, 0.5) |
| `imazu_21` | 21 | 3 | (-5.796, 1.553, -15°); (-5.796, -1.553, 15°); (0.000, 6.000, -90°) | (11.4, 60.4, -15°, 0.5); (11.4, 39.6, 15°, 0.5); (50.0, 90.0, -90°, 0.5) |
| `imazu_22` | 22 | 3 | (-4.200, 0.000, 0°); (-4.243, 4.243, -45°); (0.000, 6.000, -90°) | (22.0, 50.0, 0°, 0.35); (21.7, 78.3, -45°, 0.5); (50.0, 90.0, -90°, 0.5) |

## Running

Any single case runs through the standard runner by name:

```bash
uv run python main.py simulate --scenario imazu_13 --agent classical
```

The whole set runs as a diagnostic benchmark suite:

```bash
uv run python main.py benchmark --suite benchmarks/imazu.yaml \
    --agent classical --agent mpc --agent rl:models/ppo_vessel
```
