"""
Scenario library.

A Scenario is a declarative, JSON-serializable description of one navigation
problem: world size, static obstacles, own-ship start/goal, and traffic
vessels. Named builders cover the canonical COLREGs encounters plus a seeded
random generator for training and large evaluation suites.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def make_world(self, config: Config) -> GridWorld:
        world = GridWorld(config.world.width, config.world.height,
                          config.world.cell_size)
        for x, y, w, h in self.rect_obstacles:
            world.add_obstacle(x, y, w, h)
        for cx, cy, r in self.circle_obstacles:
            world.add_circular_obstacle(cx, cy, r)
        return world

    def make_traffic(self) -> DynamicObstacleManager:
        manager = DynamicObstacleManager()
        for spec in self.traffic:
            vessel = manager.add_obstacle(
                spec.x, spec.y, np.radians(spec.heading_deg), spec.speed,
                behavior=spec.behavior)
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
        traffic=[TrafficSpec(x=90, y=50, heading_deg=180, speed=2.0)],
        seed=seed)


def crossing_starboard(seed: Optional[int] = None) -> Scenario:
    return Scenario(
        name="crossing_starboard",
        description="COLREGs Rule 15: traffic crossing from starboard; "
                    "own ship is give-way and should pass astern.",
        start=(10, 50), start_heading_deg=0, goal=(90, 50),
        traffic=[TrafficSpec(x=55, y=10, heading_deg=90, speed=2.0)],
        seed=seed)


def crossing_port(seed: Optional[int] = None) -> Scenario:
    return Scenario(
        name="crossing_port",
        description="Traffic crossing from port; own ship is stand-on but "
                    "must still act if the other vessel does not.",
        start=(10, 50), start_heading_deg=0, goal=(90, 50),
        traffic=[TrafficSpec(x=55, y=90, heading_deg=-90, speed=2.0)],
        seed=seed)


def overtaking(seed: Optional[int] = None) -> Scenario:
    return Scenario(
        name="overtaking",
        description="COLREGs Rule 13: slow vessel ahead on the same course.",
        start=(10, 50), start_heading_deg=0, goal=(90, 50),
        traffic=[TrafficSpec(x=30, y=50, heading_deg=0, speed=1.0)],
        seed=seed)


def coastal(seed: Optional[int] = None) -> Scenario:
    return Scenario(
        name="coastal",
        description="Static landmasses forming a channel plus two traffic "
                    "vessels; requires planning and avoidance together.",
        start=(8, 12), start_heading_deg=75, goal=(92, 88),
        rect_obstacles=[(25, 0, 14, 45), (55, 55, 16, 45), (40, 75, 10, 10)],
        circle_obstacles=[(75, 30, 8), (20, 70, 6)],
        traffic=[
            TrafficSpec(x=85, y=60, heading_deg=-130, speed=1.5),
            TrafficSpec(x=45, y=35, heading_deg=60, speed=1.2,
                        behavior="waypoint",
                        waypoints=[(50, 50), (45, 65), (30, 55), (45, 35)]),
        ],
        seed=seed)


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
            speed=float(rng.uniform(1.0, 2.2))))

    return Scenario(
        name="random",
        description=f"Random scenario (seed={seed}).",
        start=start, start_heading_deg=heading, goal=goal,
        circle_obstacles=circles, traffic=traffic, seed=seed)


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
    own_speed = 2.5                     # nominal cruise (config default)

    # Meeting point ahead on the own-ship track
    meet_x = rng.uniform(40.0, 65.0)
    t_meet = (meet_x - start[0]) / own_speed
    # Encounter angle relative to own course (0 = same direction):
    # head-on, starboard/port crossings, or overtaking-ish geometry.
    angle = rng.choice([180.0, -90.0, 90.0,
                        rng.uniform(120, 180), rng.uniform(-150, -60),
                        rng.uniform(60, 150), 15.0])
    speed = (1.0 if abs(angle) < 30 else float(rng.uniform(1.5, 2.5)))
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
            speed=float(rng.uniform(1.0, 2.0))))

    return Scenario(
        name="random_encounter",
        description=f"Randomized collision-course encounter (seed={seed}).",
        start=start, start_heading_deg=0, goal=goal,
        traffic=traffic, seed=seed)


SCENARIOS = {
    "open_water": open_water,
    "head_on": head_on,
    "crossing_starboard": crossing_starboard,
    "crossing_port": crossing_port,
    "overtaking": overtaking,
    "coastal": coastal,
    "random": random_scenario,
    "random_encounter": random_encounter,
}


def build_scenario(name: str, seed: Optional[int] = None) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario '{name}'. "
                       f"Available: {', '.join(sorted(SCENARIOS))}")
    return SCENARIOS[name](seed=seed)


def list_scenarios() -> List[Tuple[str, str]]:
    return [(name, fn().description) for name, fn in SCENARIOS.items()]
