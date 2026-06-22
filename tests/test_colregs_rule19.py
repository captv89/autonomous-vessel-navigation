"""Restricted-visibility Rule 19 scoring (G10)."""

import numpy as np

from src.evaluation.colregs import score_episode


def _head_on_steps(turn_dir, dt=1.0, n=25):
    """Own ship east, target west on a collision course; own alters course
    to `turn_dir` ('starboard' | 'port' | None) after the encounter opens."""
    ox, oy, oh = 0.0, 0.0, 0.0
    tx, ty, th, speed = 60.0, 0.0, np.pi, 2.0
    steps = []
    for k in range(n):
        if k >= 6 and turn_dir == "starboard":
            oh = np.radians(-25)
        elif k >= 6 and turn_dir == "port":
            oh = np.radians(25)
        steps.append({
            "t": k * dt,
            "vessel": {"x": ox, "y": oy, "heading": oh, "speed": speed},
            "obstacles": [{"id": 1, "x": tx, "y": ty,
                           "heading": th, "speed": speed}],
        })
        ox += speed * np.cos(oh) * dt
        oy += speed * np.sin(oh) * dt
        tx += speed * np.cos(th) * dt
        ty += speed * np.sin(th) * dt
    return steps


def _score(steps, restricted):
    return score_episode(steps, safe_distance=8.0, collision_radius=2.0,
                         restricted_visibility=restricted)


def test_rule19_replaces_roles():
    # Default (good visibility) head-on is a Rule 14 give-way encounter...
    normal = _score(_head_on_steps("starboard"), restricted=False)
    assert normal and normal[0].role == "give_way"
    assert "starboard_turn" in normal[0].components

    # ...but under restricted visibility it is scored under Rule 19 instead.
    r19 = _score(_head_on_steps("starboard"), restricted=True)
    assert r19 and r19[0].role == "any"
    assert "early_action" in r19[0].components
    assert "starboard_turn" not in r19[0].components


def test_rule19_penalizes_port_turn_for_forward_target():
    stbd = _score(_head_on_steps("starboard"), restricted=True)[0]
    port = _score(_head_on_steps("port"), restricted=True)[0]
    # Target is forward of the beam, so the avoid-port component applies.
    assert stbd.components["avoid_port_turn"] == 1.0
    assert port.components["avoid_port_turn"] == 0.0
    assert stbd.score > port.score
