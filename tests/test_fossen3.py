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


def test_waves_off_by_default_match_calm():
    # Hs = 0 must leave the dynamics identical to the no-wave model (frozen v1).
    plain = make_ship()
    waves_off = make_ship(significant_wave_height=0.0)
    run(plain, 30, rudder=np.radians(10), speed=2.5)
    run(waves_off, 30, rudder=np.radians(10), speed=2.5)
    assert plain.get_position() == waves_off.get_position()
    assert plain.get_speed() == waves_off.get_speed()


def test_added_resistance_slows_in_head_sea():
    # Head sea: waves travel toward -x (pi) while the ship heads +x (0) -> the
    # added-resistance speed loss should hold the ship below its calm speed.
    calm = make_ship()
    head = make_ship(significant_wave_height=2.0, wave_direction=np.pi,
                     rng=np.random.default_rng(1))
    run(calm, 60, speed=2.5)
    run(head, 60, speed=2.5)
    assert head.get_speed() < calm.get_speed() - 0.05


def test_head_sea_slows_more_than_following():
    head = make_ship(significant_wave_height=2.0, wave_direction=np.pi,
                     rng=np.random.default_rng(2))
    following = make_ship(significant_wave_height=2.0, wave_direction=0.0,
                          rng=np.random.default_rng(2))
    run(head, 60, speed=2.5)
    run(following, 60, speed=2.5)
    # Following sea has ~no added resistance, so it stays faster.
    assert following.get_speed() > head.get_speed() + 0.05


def test_first_order_yaw_oscillates_in_beam_sea():
    # Beam sea (waves abeam) with no rudder: heading should wander but stay
    # bounded (zero-mean oscillation), not run away.
    ship = make_ship(significant_wave_height=2.0, wave_direction=np.pi / 2,
                     rng=np.random.default_rng(3))
    headings = []
    for _ in range(600):
        ship.update(0.1, rudder_command=0.0, desired_speed=2.5)
        headings.append(ship.get_heading())
    headings = np.array(headings)
    assert headings.std() > 1e-3            # waves do move the heading
    assert np.abs(headings).max() < np.radians(30)  # but it stays bounded


def test_waves_seeded_deterministic():
    a = make_ship(significant_wave_height=2.0, wave_direction=np.pi / 3,
                  rng=np.random.default_rng(11))
    b = make_ship(significant_wave_height=2.0, wave_direction=np.pi / 3,
                  rng=np.random.default_rng(11))
    run(a, 30, rudder=np.radians(10), speed=2.5)
    run(b, 30, rudder=np.radians(10), speed=2.5)
    assert a.get_position() == b.get_position()
    assert a.get_heading() == b.get_heading()


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
