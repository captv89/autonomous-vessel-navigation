"""
Pygame viewer: live simulation and episode replay.

Both modes render from the same per-step record structure the JSONL logs
use, so anything you can watch live you can replay later, with the decision
inspector showing exactly why the agent acted (classical: mode, avoidance
reason, per-vessel CPA/TCPA; RL: action probabilities and value estimate).

Controls:
    SPACE       pause / resume
    LEFT/RIGHT  step one frame back / forward (while paused; replay only)
    UP/DOWN     playback speed x2 / x0.5
    G           toggle grid
    ESC / Q     quit
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np

# Colors
WATER = (16, 36, 56)
LAND = (196, 178, 128)
GRID_LINE = (28, 52, 76)
ROUTE = (90, 110, 140)
OWN_SHIP = (120, 220, 130)
OWN_TRAIL = (70, 140, 85)
TRAFFIC = (240, 150, 80)
TRAFFIC_TRAIL = (150, 95, 55)
GOAL = (120, 220, 130)
PANEL_BG = (24, 26, 30)
TEXT = (220, 222, 226)
DIM = (140, 144, 150)
WARN = (240, 120, 100)
BAR = (90, 160, 220)

PANEL_W = 380


class Viewer:
    def __init__(self, world_width: int, world_height: int,
                 title: str = "Vessel Navigation Simulator",
                 window_px: int = 760):
        import pygame
        self.pygame = pygame
        pygame.init()
        self.scale = window_px / max(world_width, world_height)
        self.world_w, self.world_h = world_width, world_height
        self.map_px_w = int(world_width * self.scale)
        self.map_px_h = int(world_height * self.scale)
        self.screen = pygame.display.set_mode(
            (self.map_px_w + PANEL_W, self.map_px_h))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("monospace", 13)
        self.font_big = pygame.font.SysFont("monospace", 16, bold=True)
        self.clock = pygame.time.Clock()
        self.show_grid = False
        self.paused = False
        self.speed = 1.0
        self._land_surface = None

    # -------------------------------------------------------------- helpers

    def to_px(self, x: float, y: float):
        # World y points up; screen y points down.
        return (int(x * self.scale), int(self.map_px_h - y * self.scale))

    def _render_land(self, grid: np.ndarray):
        pg = self.pygame
        surf = pg.Surface((self.map_px_w, self.map_px_h))
        surf.fill(WATER)
        cell = max(int(np.ceil(self.scale)), 1)
        ys, xs = np.nonzero(grid > 0.5)
        for x, y in zip(xs, ys):
            px, py = self.to_px(x, y + 1)
            pg.draw.rect(surf, LAND, (px, py, cell, cell))
        if self.show_grid:
            for i in range(0, self.world_w + 1, 10):
                pg.draw.line(surf, GRID_LINE, self.to_px(i, 0),
                             self.to_px(i, self.world_h))
                pg.draw.line(surf, GRID_LINE, self.to_px(0, i),
                             self.to_px(self.world_w, i))
        return surf

    def _ship_polygon(self, x: float, y: float, heading: float,
                      size: float = 8.0):
        cx, cy = self.to_px(x, y)
        pts = [(size, 0), (-0.6 * size, 0.45 * size),
               (-0.35 * size, 0), (-0.6 * size, -0.45 * size)]
        c, s = np.cos(-heading), np.sin(-heading)  # screen y is flipped
        return [(cx + px * c - py * s, cy + px * s + py * c)
                for px, py in pts]

    def _text(self, surface, lines: List, x: int, y: int,
              line_h: int = 16) -> int:
        for item in lines:
            if isinstance(item, tuple):
                text, color = item
            else:
                text, color = item, TEXT
            surface.blit(self.font.render(str(text)[:52], True, color), (x, y))
            y += line_h
        return y

    # ------------------------------------------------------------ rendering

    def draw_frame(self, grid: np.ndarray, goal, goal_tolerance: float,
                   step: Dict[str, Any], trail: List, traffic_trails: Dict,
                   route: Optional[List] = None,
                   status: str = "") -> None:
        pg = self.pygame
        if self._land_surface is None:
            self._land_surface = self._render_land(grid)
        self.screen.blit(self._land_surface, (0, 0))

        # Planned route
        if route and len(route) > 1:
            pts = [self.to_px(x, y) for x, y in route]
            pg.draw.lines(self.screen, ROUTE, False, pts, 1)
            for p in pts:
                pg.draw.circle(self.screen, ROUTE, p, 2)

        # Goal
        gp = self.to_px(*goal)
        pg.draw.circle(self.screen, GOAL, gp,
                       max(int(goal_tolerance * self.scale), 4), 1)
        pg.draw.circle(self.screen, GOAL, gp, 3)

        # Trails
        if len(trail) > 1:
            pg.draw.lines(self.screen, OWN_TRAIL, False,
                          [self.to_px(x, y) for x, y in trail], 2)
        for tid, tr in traffic_trails.items():
            if len(tr) > 1:
                pg.draw.lines(self.screen, TRAFFIC_TRAIL, False,
                              [self.to_px(x, y) for x, y in tr], 1)

        # Traffic vessels
        v = step["vessel"]
        for ob in step["obstacles"]:
            pg.draw.polygon(self.screen, TRAFFIC,
                            self._ship_polygon(ob["x"], ob["y"],
                                               ob["heading"], 7))

        # Own ship + commanded heading
        d = step.get("decision", {})
        if "desired_heading" in d:
            hx = v["x"] + 8 * np.cos(d["desired_heading"])
            hy = v["y"] + 8 * np.sin(d["desired_heading"])
            pg.draw.line(self.screen, DIM, self.to_px(v["x"], v["y"]),
                         self.to_px(hx, hy), 1)
        pg.draw.polygon(self.screen, OWN_SHIP,
                        self._ship_polygon(v["x"], v["y"], v["heading"], 9))

        self._draw_panel(step, status)
        pg.display.flip()

    def _draw_panel(self, step: Dict[str, Any], status: str) -> None:
        pg = self.pygame
        panel = pg.Surface((PANEL_W, self.map_px_h))
        panel.fill(PANEL_BG)
        v, d = step["vessel"], step.get("decision", {})
        x0, y = 12, 10

        panel.blit(self.font_big.render(status, True, TEXT), (x0, y)); y += 26
        y = self._text(panel, [
            f"t = {step['t']:7.1f} s    speed x{self.speed:g}"
            + ("  [PAUSED]" if self.paused else ""),
            f"pos  ({v['x']:6.1f}, {v['y']:6.1f})",
            f"hdg  {np.degrees(v['heading']):6.1f} deg   "
            f"spd {v['speed']:.2f}",
            f"rot  {np.degrees(v['turn_rate']):6.2f} deg/s "
            f"rudder {np.degrees(v['rudder_angle']):5.1f} deg",
        ], x0, y)
        y += 8

        mode = d.get("mode", "?")
        mode_color = WARN if mode == "avoidance" else TEXT
        y = self._text(panel, [(f"mode: {mode}", mode_color)], x0, y)

        # Classical: avoidance + risks
        av = d.get("avoidance")
        if av:
            y = self._text(panel, [(av.get("reason", "")[:50], WARN)], x0, y)
        risks = d.get("risks")
        if risks:
            y += 4
            y = self._text(panel, ["traffic risk:"], x0, y)
            for r in risks[:4]:
                y = self._text(panel, [
                    (f" #{r['vessel_id']} {r['encounter'][:18]:18s} "
                     f"cpa {r['cpa']:5.1f} tcpa {r['tcpa']:5.0f}",
                     WARN if r["risk_level"] >= 2 else DIM)], x0, y)

        # RL: action distribution + value
        probs = d.get("action_probs")
        if probs:
            y += 6
            y = self._text(panel, [
                f"value estimate: {d.get('value_estimate', 0):.1f}"], x0, y)
            y = self._text(panel, ["action probabilities:"], x0, y)
            chosen = d.get("action")
            for label, p in probs.items():
                bar_w = int(p * (PANEL_W - 150))
                color = OWN_SHIP if label == chosen else BAR
                pg.draw.rect(panel, color, (110, y + 3, bar_w, 9))
                self._text(panel, [(f"{label:>9s} {p:4.2f}",
                                    TEXT if label == chosen else DIM)], x0, y)
                y += 15

        events = step.get("events") or []
        if events:
            y += 6
            y = self._text(panel, [(f"events: {', '.join(events)}", WARN)],
                           x0, y)

        y = self.map_px_h - 70
        self._text(panel, [
            ("SPACE pause  LEFT/RIGHT step", DIM),
            ("UP/DOWN speed  G grid  Q quit", DIM),
        ], x0, y)
        self.screen.blit(panel, (self.map_px_w, 0))

    # --------------------------------------------------------------- events

    def handle_events(self) -> Dict[str, Any]:
        """Returns {'quit': bool, 'step_delta': int}."""
        pg = self.pygame
        out = {"quit": False, "step_delta": 0}
        for event in pg.event.get():
            if event.type == pg.QUIT:
                out["quit"] = True
            elif event.type == pg.KEYDOWN:
                if event.key in (pg.K_ESCAPE, pg.K_q):
                    out["quit"] = True
                elif event.key == pg.K_SPACE:
                    self.paused = not self.paused
                elif event.key == pg.K_UP:
                    self.speed = min(self.speed * 2, 64)
                elif event.key == pg.K_DOWN:
                    self.speed = max(self.speed / 2, 0.25)
                elif event.key == pg.K_g:
                    self.show_grid = not self.show_grid
                    self._land_surface = None
                elif event.key == pg.K_RIGHT:
                    out["step_delta"] = 1
                elif event.key == pg.K_LEFT:
                    out["step_delta"] = -1
        return out

    def close(self) -> None:
        self.pygame.quit()


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------

def replay(log_path: str) -> None:
    """Replay a recorded JSONL episode."""
    from src.sim.recorder import read_episode

    episode = read_episode(log_path)
    header, steps = episode["header"], episode["steps"]
    if not steps:
        raise ValueError(f"{log_path}: no steps recorded")

    world_cfg = header["config"]["world"]
    sim_cfg = header["config"]["simulation"]
    scenario = header["scenario"]
    grid = _rebuild_grid(scenario, world_cfg)
    route = header["agent"].get("waypoints")
    agent_name = header["agent"].get("name", "?")
    outcome = (episode.get("summary") or {}).get("outcome", "?")
    dt = sim_cfg["dt"]

    viewer = Viewer(world_cfg["width"], world_cfg["height"],
                    title=f"Replay: {scenario['name']} / {agent_name}")
    idx = 0
    trail: List = []
    traffic_trails: Dict[int, List] = {}
    while True:
        ev = viewer.handle_events()
        if ev["quit"]:
            break
        if viewer.paused and ev["step_delta"]:
            idx = int(np.clip(idx + ev["step_delta"], 0, len(steps) - 1))
        elif not viewer.paused:
            idx = min(idx + max(int(viewer.speed), 1), len(steps) - 1)

        trail = [(s["vessel"]["x"], s["vessel"]["y"]) for s in steps[:idx + 1]]
        traffic_trails = {}
        for s in steps[:idx + 1]:
            for ob in s["obstacles"]:
                traffic_trails.setdefault(ob["id"], []).append(
                    (ob["x"], ob["y"]))

        status = (f"{scenario['name']} | {agent_name} | "
                  f"{idx + 1}/{len(steps)} | {outcome}")
        viewer.draw_frame(grid, tuple(scenario["goal"]),
                          sim_cfg["goal_tolerance"], steps[idx], trail,
                          traffic_trails, route=route, status=status)
        viewer.clock.tick(max(viewer.speed / dt, 1) if not viewer.paused
                          else 30)
    viewer.close()


def live(config, scenario, agent, log_path: Optional[str] = None) -> str:
    """Run an episode live with rendering; returns the outcome."""
    from src.evaluation.runner import run_episode

    viewer = Viewer(config.world.width, config.world.height,
                    title=f"Live: {scenario.name} / {agent.name}")
    grid_holder = {}
    trail: List = []
    traffic_trails: Dict[int, List] = {}

    def callback(engine, obs, decision, result) -> Optional[bool]:
        if "grid" not in grid_holder:
            grid_holder["grid"] = engine.world.grid
            grid_holder["route"] = agent.metadata().get("waypoints")
        ev = viewer.handle_events()
        if ev["quit"]:
            return False
        while viewer.paused:
            ev = viewer.handle_events()
            if ev["quit"]:
                return False
            viewer.clock.tick(30)

        trail.append((obs.vessel["x"], obs.vessel["y"]))
        for ob in obs.obstacles:
            traffic_trails.setdefault(ob["id"], []).append((ob["x"], ob["y"]))
        step = {"t": obs.t, "vessel": obs.vessel, "obstacles": obs.obstacles,
                "decision": decision.to_record(), "events": result.events}
        viewer.draw_frame(grid_holder["grid"], engine.goal,
                          config.simulation.goal_tolerance, step, trail,
                          traffic_trails, route=grid_holder.get("route"),
                          status=f"{scenario.name} | {agent.name}")
        viewer.clock.tick(viewer.speed / config.simulation.dt)
        return None

    stats = run_episode(config, scenario, agent, log_path=log_path,
                        step_callback=callback)
    viewer.close()
    return stats.outcome


def _rebuild_grid(scenario: Dict[str, Any], world_cfg: Dict[str, Any]):
    """Reconstruct the static grid from the scenario description."""
    from src.environment.grid_world import GridWorld
    world = GridWorld(world_cfg["width"], world_cfg["height"],
                      world_cfg["cell_size"])
    for x, y, w, h in scenario.get("rect_obstacles", []):
        world.add_obstacle(x, y, w, h)
    for cx, cy, r in scenario.get("circle_obstacles", []):
        world.add_circular_obstacle(cx, cy, r)
    return world.grid
