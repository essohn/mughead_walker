"""PPO baseline trainer for MugheadWalker-v0.

Uses SB3 PPO with default hyperparameters, SubprocVecEnv for parallelism,
and a custom callback that logs payload survival + distance to TensorBoard.

Usage:
    python examples/train_ppo.py --timesteps 1000000 --n-envs 8 --tag v1

Results land in `runs/<tag>_<timestamp>/`:
    - tb/            TensorBoard logs
    - model.zip      final SB3 model
    - vecnorm.pkl    (reserved, not used by default)
    - train.log      stdout/stderr mirror (when invoked with tee)
"""
import argparse
import datetime as _dt
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

import mughead_walker  # noqa: F401  (registers MugheadWalker-v0)


class EpisodeMetricsCallback(BaseCallback):
    """Log payload survival + distance at episode end, aggregated over a rolling window."""

    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window = window
        self._payloads: list[int] = []
        self._distances: list[float] = []
        self._falls: list[int] = []  # 1 if hull fell (reward==-100 at end), else 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if not done:
                continue
            # Monitor wraps raw info; terminal-obs info is in info["terminal_observation"]-land.
            # But our custom keys are in every-step info, which Monitor forwards.
            if "payloads_remaining" in info:
                self._payloads.append(int(info["payloads_remaining"]))
            if "distance" in info:
                self._distances.append(float(info["distance"]))
            # Success = episode ended without the fall-penalty reward.
            # Monitor adds info["episode"]["r"]; if final reward==-100, treat as fall.
            ep = info.get("episode")
            if ep is not None:
                self._falls.append(1 if ep.get("r", 0) <= -80 else 0)
            # Trim windows
            for buf in (self._payloads, self._distances, self._falls):
                if len(buf) > self.window:
                    del buf[: len(buf) - self.window]
        if self._payloads:
            self.logger.record("rollout/payloads_remaining_mean", float(np.mean(self._payloads)))
        if self._distances:
            self.logger.record("rollout/distance_mean", float(np.mean(self._distances)))
        if self._falls:
            self.logger.record("rollout/fall_rate", float(np.mean(self._falls)))
        return True


def make_env(rank: int, seed: int):
    def _init():
        env = gym.make("MugheadWalker-v0")
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="ppo_baseline")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--dummy-vec", action="store_true",
                        help="Use DummyVecEnv instead of SubprocVecEnv (for debugging).")
    args = parser.parse_args()

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"{args.tag}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    env_cls = DummyVecEnv if args.dummy_vec else SubprocVecEnv
    vec = env_cls([make_env(i, args.seed) for i in range(args.n_envs)])

    model = PPO(
        "MlpPolicy",
        vec,
        verbose=1,
        seed=args.seed,
        device=args.device,
        tensorboard_log=str(run_dir / "tb"),
    )

    callbacks = [
        EpisodeMetricsCallback(window=100),
        CheckpointCallback(
            save_freq=max(1, args.checkpoint_every // args.n_envs),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
        ),
    ]

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=False)
    model.save(run_dir / "model.zip")
    vec.close()
    print(f"\n✅ Training complete. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
