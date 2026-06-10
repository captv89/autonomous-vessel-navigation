import numpy as np

from src.vessel.vessel_model import NomotoVessel


def test_rudder_rate_limit():
    vessel = NomotoVessel(rudder_rate=0)  # 0 -> IMO standard rate
    vessel.update(0.1, rudder_command=np.radians(35))
    # After one step the rudder can have moved at most rate * dt.
    assert vessel.get_rudder_angle() <= vessel.IMO_RUDDER_RATE * 0.1 + 1e-9


def test_full_rudder_slew_time():
    vessel = NomotoVessel(rudder_rate=0)
    dt, t = 0.1, 0.0
    while vessel.get_rudder_angle() < np.radians(35) - 1e-6 and t < 30:
        vessel.update(dt, rudder_command=np.radians(35))
        t += dt
    # IMO: 35 deg from neutral in ~5.5 s
    assert 5.0 < t < 6.5


def test_steady_turn_rate_matches_nomoto_gain():
    K, T = 0.4, 4.0
    vessel = NomotoVessel(K=K, T=T, rudder_rate=None)
    delta = np.radians(20)
    for _ in range(int(20 * T / 0.1)):
        vessel.update(0.1, rudder_command=delta)
    assert np.isclose(vessel.get_turn_rate(), K * delta, rtol=0.05)


def test_speed_clipped_to_max():
    vessel = NomotoVessel(max_speed=4.0)
    vessel.update(0.1, desired_speed=99.0)
    assert vessel.get_speed() == 4.0
