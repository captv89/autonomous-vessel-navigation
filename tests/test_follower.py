import numpy as np

from src.vessel.path_follower import LOSController


def test_progress_based_switching():
    """Regression: a vessel that misses the acceptance circle must still
    advance to the next segment once it passes the waypoint."""
    ctl = LOSController(lookahead_distance=10, path_tolerance=2.0)
    waypoints = [(0, 0), (50, 0), (50, 50)]
    # Vessel slightly past waypoint 1, but 5 units off it (outside radius 2).
    ctl.compute_desired_heading((10, 5), waypoints)
    assert ctl.current_wp_idx == 1
    ctl.compute_desired_heading((52, 5), waypoints)
    assert ctl.current_wp_idx == 2


def test_converges_to_straight_path():
    ctl = LOSController(lookahead_distance=10, path_tolerance=3.0)
    waypoints = [(0, 0), (200, 0)]
    x, y, heading, speed, dt = 0.0, 15.0, 0.0, 2.0, 0.5
    for _ in range(400):
        desired = ctl.compute_desired_heading((x, y), waypoints, dt=dt)
        if desired is None:
            break
        # Idealized vessel that instantly follows the commanded heading.
        heading = desired
        x += speed * np.cos(heading) * dt
        y += speed * np.sin(heading) * dt
    assert abs(y) < 1.5


def test_end_of_path_returns_none():
    ctl = LOSController(lookahead_distance=10, path_tolerance=3.0)
    waypoints = [(0, 0), (10, 0)]
    assert ctl.compute_desired_heading((9.9, 0.1), waypoints) is None
