"""
Asymmetric ship domain (Fujii / Goodwin): the keep-clear zone around own ship,
larger ahead than astern and biased to starboard. Shared by the COLREGs scorer
(and, later, the avoider) so "passed too close" is judged against the domain a
mariner actually keeps rather than a bare circle.

`domain_distance` returns the target's range normalized by the domain radius in
its direction: 1.0 sits on the boundary, < 1.0 is an intrusion, > 1.0 is clear.
With all multipliers 1.0 the domain is a circle of radius `safe_distance` and
the result reduces to `range / safe_distance` — exactly the legacy test, so
frozen v1 scoring is unchanged.

Heading convention matches the rest of the codebase: a starboard turn decreases
heading (clockwise), so the starboard side is heading rotated by -90 degrees.
"""

from __future__ import annotations

import numpy as np


def domain_distance(own_x: float, own_y: float, own_heading: float,
                    tgt_x: float, tgt_y: float, safe_distance: float,
                    domain) -> float:
    """Range to the target normalized by the ship-domain radius toward it.

    `domain` is any object with `fore`, `aft`, `starboard`, `port` multipliers
    (e.g. `config.ship_domain`).
    """
    dx, dy = tgt_x - own_x, tgt_y - own_y
    ch, sh = np.cos(own_heading), np.sin(own_heading)
    fwd = dx * ch + dy * sh                      # along heading (+ = ahead)
    stbd = dx * sh - dy * ch                     # abeam (+ = starboard)

    a = safe_distance * (domain.fore if fwd >= 0.0 else domain.aft)
    b = safe_distance * (domain.starboard if stbd >= 0.0 else domain.port)
    return float(np.hypot(fwd / a, stbd / b))
