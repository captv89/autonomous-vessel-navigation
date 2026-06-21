"""
Scenario library.

A Scenario is a declarative, JSON-serializable description of one navigation
problem: world size, static obstacles, own-ship start/goal, and traffic
vessels. Named builders cover the canonical COLREGs encounters plus a seeded
random generator for training and large evaluation suites.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import Config
from src.environment.grid_world import GridWorld
from src.environment.dynamic_obstacles import DynamicObstacleManager


@dataclass
class TrafficSpec:
    x: float
    y: float
    heading_deg: float
    speed: float
    behavior: str = "straight"                # 'straight' | 'waypoint' | 'circular'
    waypoints: Optional[List[Tuple[float, float]]] = None
    # COLREGs reactivity (gap G1): 'none'/'rogue' never give way (default,
    # the historical behaviour); 'compliant'/'partial' take starboard
    # avoiding action when give-way. See src/sim/reactive_traffic.py.
    compliance: str = "none"


@dataclass
class TSScheme:
    """A vertical-band traffic separation scheme (Rule 10, gap G9).

    Lanes are vertical bands (separated along x) running in the general
    direction `axis_deg` (90 = north-south); each lane carries traffic one way
    (`flow_deg`). A central separation `zone` divides them. This is a chart
    overlay, not a physical obstacle — the zone is passable (a crossing vessel
    may transit it). Present only on TSS scenarios; when set it enables the
    Rule 10 terms in the COLREGs scorer (`score_tss`).
    """
    axis_deg: float                                   # general lane direction
    lanes: List[Tuple[float, float, float]]           # (x0, x1, flow_deg), cells
    zone: Tuple[float, float]                          # (x0, x1) zone, cells

    def to_dict(self) -> Dict[str, Any]:
        return {"axis_deg": self.axis_deg,
                "lanes": [list(l) for l in self.lanes],
                "zone": list(self.zone)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TSScheme":
        return cls(axis_deg=float(d["axis_deg"]),
                   lanes=[tuple(float(v) for v in l) for l in d["lanes"]],
                   zone=tuple(float(v) for v in d["zone"]))


@dataclass
class Scenario:
    name: str
    start: Tuple[float, float]
    start_heading_deg: float
    goal: Tuple[float, float]
    rect_obstacles: List[Tuple[int, int, int, int]] = field(default_factory=list)   # x, y, w, h
    circle_obstacles: List[Tuple[int, int, int]] = field(default_factory=list)      # cx, cy, r
    traffic: List[TrafficSpec] = field(default_factory=list)
    seed: Optional[int] = None
    description: str = ""
    # Traffic separation scheme (Rule 10, gap G9); None on non-TSS scenarios.
    tss: Optional[TSScheme] = None
    # Landmass-geometry class, derived from the static obstacles actually
    # present. Used to group benchmark results by terrain type (e.g. grounding
    # rate per geometry). "open water" = no static land.
    geometry: str = "open water"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scenario":
        """Rebuild a Scenario from its serialized form (inverse of to_dict).

        Only Scenario fields are read; any other keys (e.g. world/environment
        config overrides in a world file) are ignored here.
        """
        def _tuples(items):
            return [tuple(item) for item in (items or [])]

        traffic = [TrafficSpec(
            x=float(t["x"]), y=float(t["y"]),
            heading_deg=float(t["heading_deg"]), speed=float(t["speed"]),
            behavior=t.get("behavior", "straight"),
            waypoints=_tuples(t.get("waypoints")) or None,
            compliance=t.get("compliance", "none"))
            for t in data.get("traffic", [])]
        return cls(
            name=data.get("name", "world"),
            start=tuple(data["start"]),
            start_heading_deg=float(data["start_heading_deg"]),
            goal=tuple(data["goal"]),
            rect_obstacles=_tuples(data.get("rect_obstacles")),
            circle_obstacles=_tuples(data.get("circle_obstacles")),
            traffic=traffic,
            seed=data.get("seed"),
            description=data.get("description", ""),
            geometry=data.get("geometry", "open water"),
            tss=TSScheme.from_dict(data["tss"]) if data.get("tss") else None)

    def make_world(self, config: Config) -> GridWorld:
        world = GridWorld(config.world.width, config.world.height,
                          config.world.cell_size)
        for x, y, w, h in self.rect_obstacles:
            world.add_obstacle(x, y, w, h)
        for cx, cy, r in self.circle_obstacles:
            world.add_circular_obstacle(cx, cy, r)
        if config.world.depth_model == "shoaling":
            world.apply_shoaling_depth(
                deep_depth=config.world.deep_depth,
                shoal_slope=config.world.shoal_slope,
                cell_size=config.world.cell_size,
                grounding_threshold=(config.vessel.draft
                                     + config.world.ukc_margin))
        return world

    def make_traffic(self) -> DynamicObstacleManager:
        from src.sim.reactive_traffic import VALID_COMPLIANCE
        manager = DynamicObstacleManager()
        for spec in self.traffic:
            if spec.compliance not in VALID_COMPLIANCE:
                raise ValueError(
                    f"unknown traffic compliance {spec.compliance!r}; "
                    f"expected one of {sorted(VALID_COMPLIANCE)}")
            vessel = manager.add_obstacle(
                spec.x, spec.y, np.radians(spec.heading_deg), spec.speed,
                behavior=spec.behavior, compliance=spec.compliance)
            if spec.behavior == "waypoint" and spec.waypoints:
                vessel.set_waypoints(list(spec.waypoints))
        return manager


# --------------------------------------------------------------------------
# Named scenarios (100x100 cell world assumed; positions in cells)
# --------------------------------------------------------------------------

def open_water(seed: Optional[int] = None) -> Scenario:
    return Scenario(
        name="open_water",
        description="No obstacles; pure path following from SW to NE.",
        start=(10, 10), start_heading_deg=45, goal=(90, 90), seed=seed)


def head_on(seed: Optional[int] = None) -> Scenario:
    return Scenario(
        name="head_on",
        description="COLREGs Rule 14: reciprocal-course vessel dead ahead. "
                    "Both should pass port-to-port (own ship turns starboard).",
        start=(10, 50), start_heading_deg=0, goal=(90, 50),
        traffic=[TrafficSpec(x=90, y=50, heading_deg=180, speed=0.45)],
        seed=seed)


def crossing_starboard(seed: Optional[int] = None) -> Scenario:
    return Scenario(
        name="crossing_starboard",
        description="COLREGs Rule 15: traffic crossing from starboard; "
                    "own ship is give-way and should pass astern.",
        start=(10, 50), start_heading_deg=0, goal=(90, 50),
        traffic=[TrafficSpec(x=55, y=10, heading_deg=90, speed=0.45)],
        seed=seed)


def crossing_port(seed: Optional[int] = None) -> Scenario:
    return Scenario(
        name="crossing_port",
        description="Traffic crossing from port; own ship is stand-on but "
                    "must still act if the other vessel does not.",
        start=(10, 50), start_heading_deg=0, goal=(90, 50),
        traffic=[TrafficSpec(x=55, y=90, heading_deg=-90, speed=0.45)],
        seed=seed)


def overtaking(seed: Optional[int] = None) -> Scenario:
    return Scenario(
        name="overtaking",
        description="COLREGs Rule 13: slow vessel ahead on the same course.",
        start=(10, 50), start_heading_deg=0, goal=(90, 50),
        traffic=[TrafficSpec(x=30, y=50, heading_deg=0, speed=0.25)],
        seed=seed)


def coastal(seed: Optional[int] = None) -> Scenario:
    return Scenario(
        name="coastal",
        description="Static landmasses forming a channel plus two traffic "
                    "vessels; requires planning and avoidance together.",
        start=(8, 12), start_heading_deg=75, goal=(92, 88),
        rect_obstacles=[(25, 0, 14, 45), (55, 55, 16, 45), (40, 75, 10, 10)],
        circle_obstacles=[(75, 30, 8), (20, 70, 6)],
        geometry="channel + islands",
        traffic=[
            TrafficSpec(x=85, y=60, heading_deg=-130, speed=0.4),
            TrafficSpec(x=45, y=35, heading_deg=60, speed=0.3,
                        behavior="waypoint",
                        waypoints=[(50, 50), (45, 65), (30, 55), (45, 35)]),
        ],
        seed=seed)


def coastal_random(seed: Optional[int] = None) -> Scenario:
    """Seeded procedural coastal world: random landmasses and islets forming a
    cluttered channel, with traffic.

    Deliberately distinct from the fixed `coastal` benchmark geometry: this is
    a *training* generator for coastal planning + avoidance skill, so a policy
    can learn to handle obstacle-strewn channels without ever seeing the exam
    map (training on `coastal` itself would be test-set leakage). Obstacles are
    kept clear of the straight start->goal corridor so every episode stays
    solvable while still forcing detours around the flanking landmasses.
    """
    rng = np.random.default_rng(seed)

    corners = [((10, 10), 45, (90, 90)), ((10, 90), -45, (90, 10)),
               ((90, 10), 135, (10, 90)), ((90, 90), -135, (10, 10))]
    start, heading, goal = corners[rng.integers(len(corners))]
    margin = 11.0                       # keep the direct corridor navigable

    def _corridor_pts():
        for t in np.linspace(0.0, 1.0, 25):
            yield (start[0] + t * (goal[0] - start[0]),
                   start[1] + t * (goal[1] - start[1]))

    def rect_clears_corridor(x, y, w, h):
        return all(not (x - margin <= px <= x + w + margin and
                        y - margin <= py <= y + h + margin)
                   for px, py in _corridor_pts())

    def circle_clears_corridor(cx, cy, r):
        return all(np.hypot(cx - px, cy - py) >= r + margin
                   for px, py in _corridor_pts())

    rects: List[Tuple[int, int, int, int]] = []
    for _ in range(rng.integers(2, 5)):
        for _attempt in range(25):
            w, h = int(rng.integers(8, 22)), int(rng.integers(8, 22))
            x, y = int(rng.integers(8, 92 - w)), int(rng.integers(8, 92 - h))
            if rect_clears_corridor(x, y, w, h):
                rects.append((x, y, w, h))
                break

    circles: List[Tuple[int, int, int]] = []
    for _ in range(rng.integers(1, 4)):
        for _attempt in range(25):
            cx, cy, r = (int(rng.integers(15, 85)), int(rng.integers(15, 85)),
                         int(rng.integers(4, 9)))
            if circle_clears_corridor(cx, cy, r):
                circles.append((cx, cy, r))
                break

    def blocked(px, py):
        if any(x <= px <= x + w and y <= py <= y + h for x, y, w, h in rects):
            return True
        return any(np.hypot(cx - px, cy - py) < r for cx, cy, r in circles)

    traffic = []
    for _ in range(rng.integers(1, 3)):
        for _attempt in range(25):
            tx, ty = rng.uniform(30, 70, size=2)
            if not blocked(tx, ty):
                traffic.append(TrafficSpec(
                    x=float(tx), y=float(ty),
                    heading_deg=float(rng.uniform(-180, 180)),
                    speed=float(rng.uniform(0.2, 0.45))))
                break

    return Scenario(
        name="coastal_random",
        description=f"Procedural coastal world (seed={seed}).",
        start=start, start_heading_deg=heading, goal=goal,
        rect_obstacles=rects, circle_obstacles=circles, traffic=traffic,
        geometry="channel + islands", seed=seed)


def random_scenario(seed: Optional[int] = None) -> Scenario:
    """Seeded random scenario: a few islands and traffic vessels.

    Start/goal are kept in opposite corners with clearance from obstacles so
    every generated episode is solvable.
    """
    rng = np.random.default_rng(seed)
    width = height = 100

    corners = [((15, 15), 45, (85, 85)), ((15, 85), -45, (85, 15)),
               ((85, 15), 135, (15, 85)), ((85, 85), -135, (15, 15))]
    start, heading, goal = corners[rng.integers(len(corners))]

    def clear_of_endpoints(cx, cy, r):
        for px, py in (start, goal):
            if np.hypot(cx - px, cy - py) < r + 15:
                return False
        return True

    circles = []
    for _ in range(rng.integers(2, 5)):
        for _attempt in range(20):
            cx, cy = rng.integers(20, 80, size=2)
            r = int(rng.integers(4, 9))
            if clear_of_endpoints(cx, cy, r):
                circles.append((int(cx), int(cy), r))
                break

    traffic = []
    for _ in range(rng.integers(1, 4)):
        x, y = rng.uniform(25, 75, size=2)
        traffic.append(TrafficSpec(
            x=float(x), y=float(y),
            heading_deg=float(rng.uniform(-180, 180)),
            speed=float(rng.uniform(0.2, 0.45))))

    return Scenario(
        name="random",
        description=f"Random scenario (seed={seed}).",
        start=start, start_heading_deg=heading, goal=goal,
        geometry="scattered islands",
        circle_obstacles=circles, traffic=traffic, seed=seed)


def multi_vessel(seed: Optional[int] = None) -> Scenario:
    return Scenario(
        name="multi_vessel",
        description="Two simultaneous conflicts: a reciprocal-course vessel "
                    "ahead and a give-way crossing from starboard, timed to "
                    "overlap. Tests rule interplay under multi-ship load.",
        start=(10, 50), start_heading_deg=0, goal=(90, 50),
        traffic=[
            TrafficSpec(x=90, y=50, heading_deg=180, speed=0.45),
            TrafficSpec(x=52, y=14, heading_deg=90, speed=0.45),
            TrafficSpec(x=70, y=78, heading_deg=-115, speed=0.35),
        ],
        seed=seed)


def narrow_channel(seed: Optional[int] = None) -> Scenario:
    return Scenario(
        name="narrow_channel",
        description="A narrow fairway between two coasts with an oncoming "
                    "vessel inside the channel: limited sea room to both "
                    "sides while resolving a Rule 14 encounter (Rule 9 "
                    "flavor).",
        start=(8, 50), start_heading_deg=0, goal=(92, 50),
        rect_obstacles=[(0, 0, 100, 36), (0, 64, 100, 36)],
        geometry="narrow channel",
        traffic=[TrafficSpec(x=90, y=53, heading_deg=180, speed=0.4)],
        seed=seed)


def _tss_scheme() -> TSScheme:
    """The canonical north-south scheme shared by the TSS scenarios (G9).

    West lane is southbound, east lane northbound, so a northbound vessel keeps
    the separation zone on its port hand (IMO convention). Headings: 90 = north
    (+y), 270 = south (-y); lanes and zone are vertical x-bands.
    """
    return TSScheme(axis_deg=90.0,
                    lanes=[(20.0, 40.0, 270.0), (60.0, 80.0, 90.0)],
                    zone=(40.0, 60.0))


def tss_transit(seed: Optional[int] = None) -> Scenario:
    return Scenario(
        name="tss_transit",
        description="Transit a traffic separation scheme (Rule 10): proceed "
                    "north in the east lane, overtaking a slower vessel ahead "
                    "while opposing traffic runs south in the west lane.",
        start=(70, 8), start_heading_deg=90, goal=(70, 92),
        geometry="traffic separation scheme",
        tss=_tss_scheme(),
        traffic=[
            TrafficSpec(x=70, y=40, heading_deg=90, speed=0.3),   # slow, same lane
            TrafficSpec(x=30, y=85, heading_deg=270, speed=0.4),  # opposing lane
        ],
        seed=seed)


def tss_crossing(seed: Optional[int] = None) -> Scenario:
    return Scenario(
        name="tss_crossing",
        description="Cross a traffic separation scheme as near to right angles "
                    "as practicable (Rule 10(c)): head east across the west "
                    "(southbound) and east (northbound) lanes.",
        start=(8, 50), start_heading_deg=0, goal=(92, 50),
        geometry="traffic separation scheme",
        tss=_tss_scheme(),
        traffic=[
            TrafficSpec(x=30, y=72, heading_deg=270, speed=0.5),  # west lane crosser
            TrafficSpec(x=70, y=8, heading_deg=90, speed=0.4),    # east lane crosser
        ],
        seed=seed)


def random_encounter(seed: Optional[int] = None) -> Scenario:
    """Seeded collision-course encounter with randomized geometry.

    One traffic vessel is placed so that, holding course and speed, it
    reaches the own-ship's straight track exactly when the own ship does —
    a guaranteed close-quarters situation with a randomly drawn encounter
    angle. Used for training agents on COLREGs-relevant traffic without
    exposing them to the benchmark's fixed exam geometries.
    """
    rng = np.random.default_rng(seed)
    start, goal = (12.0, 50.0), (88.0, 50.0)
    own_speed = 0.5                     # nominal cruise (config default)

    # Meeting point ahead on the own-ship track
    meet_x = rng.uniform(40.0, 65.0)
    t_meet = (meet_x - start[0]) / own_speed
    # Encounter angle relative to own course (0 = same direction):
    # head-on, starboard/port crossings, or overtaking-ish geometry.
    angle = rng.choice([180.0, -90.0, 90.0,
                        rng.uniform(120, 180), rng.uniform(-150, -60),
                        rng.uniform(60, 150), 15.0])
    speed = (0.25 if abs(angle) < 30 else float(rng.uniform(0.35, 0.55)))
    # Traffic course such that the drawn angle is the relative encounter
    # angle (180 = reciprocal/head-on, -90 = crossing from starboard).
    heading = np.radians(angle + 180.0)
    # Place the vessel so it arrives at the meeting point at t_meet
    tx = meet_x - speed * np.cos(heading) * t_meet
    ty = 50.0 - speed * np.sin(heading) * t_meet

    traffic = [TrafficSpec(x=float(tx), y=float(ty),
                           heading_deg=float(np.degrees(heading)),
                           speed=float(speed))]
    # Occasionally a second, non-conflicting wanderer
    if rng.random() < 0.4:
        wx, wy = rng.uniform(25, 75), rng.choice([20.0, 80.0])
        traffic.append(TrafficSpec(
            x=float(wx), y=float(wy),
            heading_deg=float(rng.uniform(-180, 180)),
            speed=float(rng.uniform(0.2, 0.4))))

    return Scenario(
        name="random_encounter",
        description=f"Randomized collision-course encounter (seed={seed}).",
        start=start, start_heading_deg=0, goal=goal,
        traffic=traffic, seed=seed)


# --------------------------------------------------------------------------
# Imazu problem (Sawada et al. 2021, "Automatic ship collision avoidance
# using deep reinforcement learning with LSTM in continuous action spaces",
# J. Mar. Sci. Technol., Table 4; geometries originally due to Imazu 1987).
#
# The canonical 22-case exam used across the COLREGs / DRL collision-
# avoidance literature. In the paper the own ship starts at (X, Y) =
# (-6.0, 0.0) NM heading psi = 0 deg toward a waypoint at (6.0, 0.0); every
# target ship holds a straight collision course to the origin, with its
# speed set so all ships reach the origin together (TCPA 30 min).
#
# Each entry below is a list of (X_NM, Y_NM, psi_deg) target ships,
# transcribed verbatim from Table 4. Cases 1-4 have one target, 5-12 two,
# 13-22 three. See docs/IMAZU.md for the case-by-case mapping.
IMAZU_CASES: Dict[int, List[Tuple[float, float, float]]] = {
    1:  [(6.000, 0.000, 180.0)],
    2:  [(0.000, 6.000, -90.0)],
    3:  [(-4.200, 0.000, 0.0)],
    4:  [(-4.243, -4.243, 45.0)],
    5:  [(6.000, 0.000, 180.0), (0.000, 6.000, -90.0)],
    6:  [(-5.909, 1.042, -10.0), (-4.243, 4.243, -45.0)],
    7:  [(-4.200, 0.000, 0.0), (-4.243, 4.243, -45.0)],
    8:  [(6.000, 0.000, 180.0), (0.000, 6.000, -90.0)],
    9:  [(-5.196, 3.000, -30.0), (0.000, 6.000, -90.0)],
    10: [(0.000, 6.000, -90.0), (-5.796, -1.553, 15.0)],
    11: [(0.000, -6.000, 90.0), (-5.196, 3.000, -30.0)],
    12: [(-4.243, 4.243, -45.0), (-5.909, 1.042, -10.0)],
    13: [(6.000, 0.000, 180.0), (-5.909, -1.042, 10.0), (-4.243, -4.243, 45.0)],
    14: [(-5.909, 1.042, -10.0), (-4.243, 4.243, -45.0), (0.000, 6.000, -90.0)],
    15: [(-4.200, 0.000, 0.0), (-4.243, 4.243, -45.0), (0.000, 6.000, -90.0)],
    16: [(-2.970, -2.970, 45.0), (0.000, -6.000, 90.0), (0.000, 6.000, -90.0)],
    17: [(-4.200, 0.000, 0.0), (-5.909, -1.042, 10.0), (-4.243, 4.243, -45.0)],
    18: [(4.243, 4.243, -135.0), (-5.796, 1.553, -15.0), (-5.196, 3.000, -30.0)],
    19: [(-5.796, -1.553, 15.0), (-5.796, 1.553, -15.0), (4.243, 4.243, -135.0)],
    20: [(-4.200, 0.000, 0.0), (-5.796, 1.553, -15.0), (0.000, 6.000, -90.0)],
    21: [(-5.796, 1.553, -15.0), (-5.796, -1.553, 15.0), (0.000, 6.000, -90.0)],
    22: [(-4.200, 0.000, 0.0), (-4.243, 4.243, -45.0), (0.000, 6.000, -90.0)],
}

# Affine map from Sawada's NM frame to our 100x100 cell world: the frame is
# identical in handedness and heading convention (0 deg = +x, CCW), so only
# a translate+scale is needed. (-6, 0) NM -> (10, 50) cells and (6, 0) NM ->
# (90, 50) cells, matching the existing head_on geometry.
_IMAZU_RANGE_NM = 6.0
_IMAZU_SCALE = 40.0 / _IMAZU_RANGE_NM            # cells per NM
_IMAZU_CENTER = 50.0                             # encounter point, cells
_IMAZU_REF_SPEED = 0.5                           # own-ship cruise (cells/s)


def imazu(case: int, seed: Optional[int] = None) -> Scenario:
    """Build Imazu problem case `case` (1..22) on the VesselNav scale.

    Own ship runs across the world (10, 50) -> (90, 50); each target ship is
    placed on a straight collision course to the centre, its speed scaled by
    range so all ships would meet there together (faithful to the paper's
    simultaneous-arrival construction).
    """
    if case not in IMAZU_CASES:
        raise KeyError(f"Imazu case {case} out of range 1..22")
    targets = IMAZU_CASES[case]
    traffic = [TrafficSpec(
        x=_IMAZU_CENTER + _IMAZU_SCALE * x_nm,
        y=_IMAZU_CENTER + _IMAZU_SCALE * y_nm,
        heading_deg=psi_deg,
        speed=round(_IMAZU_REF_SPEED * float(np.hypot(x_nm, y_nm))
                    / _IMAZU_RANGE_NM, 3))
        for (x_nm, y_nm, psi_deg) in targets]
    n = len(targets)
    return Scenario(
        name=f"imazu_{case:02d}",
        description=f"Imazu problem case {case} (Sawada et al. 2021): "
                    f"{n} target ship{'s' if n != 1 else ''} on a collision "
                    f"course to the encounter point.",
        start=(10.0, 50.0), start_heading_deg=0.0, goal=(90.0, 50.0),
        traffic=traffic, seed=seed)


SCENARIOS = {
    "open_water": open_water,
    "head_on": head_on,
    "crossing_starboard": crossing_starboard,
    "crossing_port": crossing_port,
    "overtaking": overtaking,
    "coastal": coastal,
    "coastal_random": coastal_random,
    "multi_vessel": multi_vessel,
    "narrow_channel": narrow_channel,
    "tss_transit": tss_transit,
    "tss_crossing": tss_crossing,
    "random": random_scenario,
    "random_encounter": random_encounter,
}

# Imazu problem cases register as imazu_01 .. imazu_22.
for _case in range(1, 23):
    SCENARIOS[f"imazu_{_case:02d}"] = partial(imazu, _case)


# Config sections a world file may override (mirrors Config's sections).
_CONFIG_SECTIONS = ("simulation", "world", "vessel", "environment", "control",
                    "planner", "follower", "avoidance", "rl", "training")


def _is_world_file(name: str) -> bool:
    return (name.endswith((".yaml", ".yml")) or os.sep in name
            or os.path.exists(name))


def load_world_file(path: str) -> Tuple[Scenario, Dict[str, Any]]:
    """Load a self-contained world file.

    Returns the Scenario (geometry + traffic) and a dict of config section
    overrides (e.g. {"environment": {...}}) present in the file. The caller
    decides whether/how to apply the overrides.
    """
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    overrides = {k: data[k] for k in _CONFIG_SECTIONS if k in data}
    return Scenario.from_dict(data), overrides


def build_scenario(name: str, seed: Optional[int] = None) -> Scenario:
    """Resolve a scenario name (registry) or a world-file path to a Scenario.

    A passed `seed` overrides whatever the world file declares; world-file
    config overrides are dropped here (use load_world_file to apply them).
    """
    if _is_world_file(name):
        scenario, _ = load_world_file(name)
        if seed is not None:
            scenario.seed = seed
        return scenario
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{name}'. "
                       f"Available: {', '.join(sorted(SCENARIOS))}")
    return SCENARIOS[name](seed=seed)


def list_scenarios() -> List[Tuple[str, str]]:
    return [(name, fn().description) for name, fn in SCENARIOS.items()]


def scenario_geometry(name: str) -> str:
    """Landmass-geometry class for a scenario name or world-file path.

    Derived from the scenario's static obstacles (see Scenario.geometry);
    returns "open water" for unknown names.
    """
    try:
        return build_scenario(name, seed=0).geometry
    except KeyError:
        return "open water"
