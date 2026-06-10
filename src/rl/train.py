"""
PPO training on the shared simulation engine.

Usage:
    uv run python -m src.rl.train [--config configs/default.yaml]
                                  [--timesteps N] [--scenario random]
                                  [--out models/ppo_vessel]

Training is seeded and the resolved config is saved next to the model so
every run is reproducible. Metrics stream to `runs/` (TensorBoard format if
tensorboard is installed, CSV always).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.config import Config

logger = logging.getLogger(__name__)


def make_env_fn(config: Config, scenario: str, rank: int, base_seed: int):
    def _init():
        from src.rl.env import VesselNavEnv
        from stable_baselines3.common.monitor import Monitor
        env = VesselNavEnv(config, scenario_name=scenario,
                           seed=base_seed + rank)
        return Monitor(env, info_keywords=("outcome",))
    return _init


def train(config: Config, total_timesteps: int, scenario: str,
          out_path: str, n_envs: int | None = None) -> str:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                                  VecNormalize)

    tr = config.training
    n_envs = n_envs or tr.n_envs
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(tr.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    env_fns = [make_env_fn(config, scenario, i, tr.seed) for i in range(n_envs)]
    vec_env = (SubprocVecEnv(env_fns) if n_envs > 1 else DummyVecEnv(env_fns))
    # Return normalization stabilizes PPO's value/advantage scales; the
    # policy's observation interface is untouched (norm_obs=False), so the
    # saved model needs no normalization stats at inference time.
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True,
                           gamma=tr.gamma)

    try:
        import tensorboard  # noqa: F401
        tb_log = str(log_dir)
    except ImportError:
        tb_log = None

    model = PPO(
        "MlpPolicy", vec_env,
        learning_rate=tr.learning_rate,
        n_steps=max(tr.n_steps // n_envs, 64),
        batch_size=tr.batch_size,
        gamma=tr.gamma,
        ent_coef=tr.ent_coef,
        seed=tr.seed,
        verbose=1,
        device="cpu",            # MLP policy: CPU is faster than GPU here
        tensorboard_log=tb_log)

    logger.info("Training PPO for %d timesteps on scenario '%s' (%d envs)",
                total_timesteps, scenario, n_envs)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    model.save(out)
    config.save(out.with_suffix(".config.yaml"))
    vec_env.close()
    logger.info("Saved model to %s.zip", out)
    return str(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--out", default="models/ppo_vessel")
    parser.add_argument("--n-envs", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    config = Config.load(args.config)
    train(config,
          total_timesteps=args.timesteps or config.training.total_timesteps,
          scenario=args.scenario or config.training.scenario,
          out_path=args.out,
          n_envs=args.n_envs)


if __name__ == "__main__":
    main()
