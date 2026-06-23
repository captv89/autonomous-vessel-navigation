"""
PPO training on the shared simulation engine.

Usage:
    uv run python -m src.rl.train [--config configs/default.yaml]
                                  [--timesteps N] [--scenario random]
                                  [--out models/ppo_vessel]
                                  [--finetune models/ppo_10m]

--scenario accepts a comma-separated list for curriculum mixing, e.g.
    --scenario random,coastal,coastal,coastal
Each episode independently samples from the pool (uniform), so repetition
biases the draw without hard phasing.

--finetune loads an existing model's weights and continues training on the
new scenario mix — the recommended way to do curriculum continuation.

Training is seeded and the resolved config is saved next to the model so
every run is reproducible. Metrics stream to `runs/` (TensorBoard format if
tensorboard is installed, CSV always).
"""

from __future__ import annotations

import argparse
import logging
from collections import deque
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


def _make_outcome_callback(window: int = 200):
    """Callback logging success/grounding/collision rates over recent episodes.

    PPO already logs `rollout/ep_rew_mean`; this adds the outcome rates the
    blog's training curves need (success and grounding vs timesteps), computed
    over a rolling window of the last `window` finished episodes. Recorded on
    every rollout end so they land in every configured logger output (TB + CSV).
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class OutcomeRateCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.outcomes: deque = deque(maxlen=window)

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                # Monitor adds the "episode" key only on episode termination.
                if "episode" in info and "outcome" in info:
                    self.outcomes.append(info["outcome"])
            return True

        def _on_rollout_end(self) -> None:
            if not self.outcomes:
                return
            n = len(self.outcomes)
            for name, key in (("success_rate", "goal"),
                              ("grounding_rate", "grounding"),
                              ("collision_rate", "collision"),
                              ("timeout_rate", "timeout")):
                rate = sum(1 for o in self.outcomes if o == key) / n
                self.logger.record(f"rollout/{name}", rate)

    return OutcomeRateCallback()


def train(config: Config, total_timesteps: int, scenario: str,
          out_path: str, n_envs: int | None = None,
          finetune: str | None = None) -> str:
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

    try:
        import tensorboard  # noqa: F401
        tb_log = str(log_dir)
    except ImportError:
        tb_log = None

    if finetune:
        vecnorm_path = Path(finetune).with_suffix(".vecnorm")
        if vecnorm_path.exists():
            # Restore the reward-normalizer running stats so the value function
            # calibration from the base model remains valid on the new scenario mix.
            logger.info("Restoring VecNormalize stats from %s", vecnorm_path)
            vec_env = VecNormalize.load(str(vecnorm_path), vec_env)
            vec_env.training = True
        else:
            logger.warning(
                "No VecNormalize stats found at %s — starting normalizer from "
                "scratch; value-function scale may be mismatched for ~1M steps",
                vecnorm_path)
            vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True,
                                   gamma=tr.gamma)
        logger.info("Loading weights from %s for curriculum continuation", finetune)
        model = PPO.load(finetune, env=vec_env, device="cpu",
                         tensorboard_log=tb_log)
    else:
        # Return normalization stabilizes PPO's value/advantage scales; the
        # policy's observation interface is untouched (norm_obs=False), so the
        # saved model needs no normalization stats at inference time.
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True,
                               gamma=tr.gamma)
        model = PPO(
            "MlpPolicy", vec_env,
            learning_rate=tr.learning_rate,
            n_steps=max(tr.n_steps // n_envs, 64),
            batch_size=tr.batch_size,
            gamma=tr.gamma,
            ent_coef=tr.ent_coef,
            seed=tr.seed,
            verbose=1,
            device="cpu",
            tensorboard_log=tb_log)

    # Stream metrics to a timestamped run dir in both TensorBoard and CSV, so
    # the training curves can be reconstructed from `progress.csv` even without
    # TensorBoard installed (parsed by scripts/make_training_curves.py).
    from datetime import datetime
    from stable_baselines3.common.logger import configure
    run_dir = log_dir / f"ppo_{datetime.now():%Y%m%d_%H%M%S}"
    formats = ["stdout", "csv"] + (["tensorboard"] if tb_log else [])
    model.set_logger(configure(str(run_dir), formats))
    logger.info("Logging training metrics to %s (%s)", run_dir,
                ", ".join(formats))

    logger.info("Training PPO for %d timesteps on scenario '%s' (%d envs)%s",
                total_timesteps, scenario, n_envs,
                f" (finetuning from {finetune})" if finetune else "")
    model.learn(total_timesteps=total_timesteps, progress_bar=False,
                callback=_make_outcome_callback())

    model.save(out)
    vec_env.save(str(out.with_suffix(".vecnorm")))
    config.save(out.with_suffix(".config.yaml"))
    vec_env.close()
    logger.info("Saved model to %s.zip", out)
    return str(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="YAML config path")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--scenario", default=None,
                        help="Scenario name or comma-separated pool, e.g. random,coastal,coastal")
    parser.add_argument("--out", default="models/ppo_vessel")
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--finetune", default=None, metavar="MODEL_PATH",
                        help="Load weights from this model and continue training (curriculum)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(message)s")
    config = Config.load(args.config)
    train(config,
          total_timesteps=args.timesteps or config.training.total_timesteps,
          scenario=args.scenario or config.training.scenario,
          out_path=args.out,
          n_envs=args.n_envs,
          finetune=args.finetune)


if __name__ == "__main__":
    main()
