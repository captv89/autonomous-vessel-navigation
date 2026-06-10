import numpy as np

from src.config import Config


def test_defaults_load():
    cfg = Config()
    assert cfg.simulation.dt > 0
    assert cfg.vessel.max_rudder == np.radians(cfg.vessel.max_rudder_deg)


def test_yaml_roundtrip(tmp_path):
    cfg = Config()
    cfg.vessel.cruise_speed = 1.23
    cfg.rl.reward.collision = -42.0
    path = tmp_path / "cfg.yaml"
    cfg.save(path)

    loaded = Config.load(path)
    assert loaded.vessel.cruise_speed == 1.23
    assert loaded.rl.reward.collision == -42.0
    assert loaded.to_dict() == cfg.to_dict()


def test_partial_yaml_uses_defaults(tmp_path):
    path = tmp_path / "partial.yaml"
    path.write_text("vessel:\n  cruise_speed: 9.0\n")
    cfg = Config.load(path)
    assert cfg.vessel.cruise_speed == 9.0
    assert cfg.world.width == Config().world.width
