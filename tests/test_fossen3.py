import numpy as np
import pytest

from src.config import Config
from src.vessel.dynamics import Fossen3DOFVessel, make_vessel
from src.vessel.vessel_model import NomotoVessel


def make_ship(**kwargs):
    defaults = dict(max_speed=4.0, K=0.4, T=4.0,
                    max_rudder=np.radians(35), rudder_rate=None)
    defaults.update(kwargs)
    return Fossen3DOFVessel(0.0, 0.0, 0.0, 2.5, **defaults)


def run(ship, seconds, rudder=0.0, speed=None, dt=0.1):
    for _ in range(int(seconds / dt)):
        ship.update(dt, rudder_command=rudder, desired_speed=speed)


def test_yaw_channel_matches_nomoto_gain():
    ship = make_ship()
    run(ship, 60, rudder=np.radians(20))
    assert np.isclose(ship.get_turn_rate(), 0.4 * np.radians(20), rtol=0.05)


def test_sideslip_develops_in_turn():
    ship = make_ship()
    run(ship, 30, rudder=np.radians(20))
    # Steady sideslip should oppose the turn direction and be significant.
    assert ship.get_sway() == pytest.approx(
        -1.2 * ship.get_speed() * ship.get_turn_rate(), rel=0.1)
    assert abs(ship.get_sway()) > 0.05


def test_speed_loss_in_turn():
    straight = make_ship()
    run(straight, 40, rudder=0.0, speed=2.5)
    turning = make_ship()
    run(turning, 40, rudder=np.radians(30), speed=2.5)
    assert turning.get_speed() < straight.get_speed() - 0.1


def test_surge_response_is_not_instant():
    ship = make_ship()
    ship.update(0.1, desired_speed=4.0)
    assert ship.get_speed() < 2.6  # far from the new command after 0.1 s


def test_current_drifts_vessel():
    ship = make_ship(current=(0.5, 0.0))
    ship._u_cmd = 0.0
    ship.u = 0.0
    run(ship, 20, speed=0.0)
    x, y = ship.get_position()
    assert x == pytest.approx(0.5 * 20, rel=0.05)
    assert abs(y) < 0.5


def test_gusts_are_seeded_deterministic():
    a = make_ship(wind_gust_accel=0.05, rng=np.random.default_rng(7))
    b = make_ship(wind_gust_accel=0.05, rng=np.random.default_rng(7))
    run(a, 20, rudder=np.radians(10))
    run(b, 20, rudder=np.radians(10))
    assert a.get_position() == b.get_position()


def test_factory_selects_model():
    cfg = Config()
    cfg.vessel.model = "nomoto"
    assert isinstance(make_vessel(cfg, x=0, y=0, heading=0, speed=2.5),
                      NomotoVessel)
    cfg.vessel.model = "fossen3"
    assert isinstance(make_vessel(cfg, x=0, y=0, heading=0, speed=2.5),
                      Fossen3DOFVessel)
    cfg.vessel.model = "bogus"
    with pytest.raises(ValueError):
        make_vessel(cfg, x=0, y=0, heading=0, speed=2.5)
