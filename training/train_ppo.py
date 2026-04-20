"""Baseline trainer for MugheadWalker-v0.

Supports PPO, SAC, TD3, A2C via --algo.
Supports fine-tuning from a checkpoint via --load-checkpoint.
Use --list-models to browse available checkpoints (newest first).

Usage:
    # Fresh training
    python examples/train_ppo.py --algo ppo --timesteps 1000000 --n-envs 8 --tag v1

    # List saved models
    python examples/train_ppo.py --list-models

    # Fine-tune from checkpoint
    python examples/train_ppo.py \\
        --load-checkpoint runs/ppo_waist_20260418_022338/model.zip \\
        --timesteps 1000000 --terrain-difficulty 1 --tag hardcore_ft

Results land in runs/<tag>_<timestamp>/:
    tb/            TensorBoard logs
    model.zip      final model
    best_model.zip best eval checkpoint
    config.json    full config (algo, timesteps, source checkpoint, …)
"""
import argparse
import datetime as _dt
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

import mughead_walker  # noqa: F401

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3, "a2c": A2C}


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class EpisodeMetricsCallback(BaseCallback):
    """Log payload survival + distance at episode end over a rolling window."""

    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window = window
        self._payloads: list[int] = []
        self._distances: list[float] = []
        self._falls: list[int] = []

    def _on_step(self) -> bool:
        for info, done in zip(self.locals.get("infos", []), self.locals.get("dones", [])):
            if not done:
                continue
            if "payloads_remaining" in info:
                self._payloads.append(int(info["payloads_remaining"]))
            if "distance" in info:
                self._distances.append(float(info["distance"]))
            ep = info.get("episode")
            if ep is not None:
                self._falls.append(1 if ep.get("r", 0) <= -80 else 0)
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


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def make_env(rank: int, seed: int, terrain_difficulty: int = 0):
    def _init():
        import mughead_walker as _mw  # noqa: F811
        env = gym.make("MugheadWalker-v0", terrain_difficulty=terrain_difficulty)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env
    return _init


# ---------------------------------------------------------------------------
# Model listing
# ---------------------------------------------------------------------------

def _find_models(runs_dir: Path = Path("runs")) -> list[dict]:
    """Return dicts {path, tag, algo, mtime} sorted newest-first."""
    if not runs_dir.exists():
        return []
    models = []
    for zip_path in runs_dir.rglob("*.zip"):
        # Skip tiny optimizer-state files (< 50 KB are unlikely to be full models)
        if zip_path.stat().st_size < 50_000:
            continue
        run_dir = zip_path.parent
        # For checkpoints/ subdirs, go up one level for config
        config_path = run_dir / "config.json"
        if not config_path.exists():
            config_path = run_dir.parent / "config.json"
        algo = "?"
        if config_path.exists():
            try:
                algo = json.loads(config_path.read_text()).get("algo", "?")
            except Exception:
                pass
        models.append({
            "path": zip_path,
            "tag": zip_path.stem,
            "run": run_dir.name if run_dir.name != "checkpoints" else run_dir.parent.name,
            "algo": algo,
            "mtime": zip_path.stat().st_mtime,
        })
    return sorted(models, key=lambda m: m["mtime"], reverse=True)


def list_models():
    models = _find_models()
    if not models:
        print("No saved models found in runs/")
        return
    print(f"\n{'#':<4} {'Modified':<20} {'Algo':<6} {'File'}")
    print("-" * 80)
    for i, m in enumerate(models):
        ts = _dt.datetime.fromtimestamp(m["mtime"]).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{i:<4} {ts:<20} {m['algo']:<6} {m['path']}")
    print()


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _detect_algo(checkpoint_path: Path, fallback: str) -> str:
    """Infer algo name from config.json near the checkpoint."""
    for config_path in [
        checkpoint_path.parent / "config.json",
        checkpoint_path.parent.parent / "config.json",
    ]:
        if config_path.exists():
            try:
                return json.loads(config_path.read_text()).get("algo", fallback)
            except Exception:
                pass
    return fallback


def load_checkpoint(checkpoint_path: Path, algo_cls, vec, device: str):
    """Load a saved SB3 model and attach a new vec env for fine-tuning."""
    print(f"  Loading checkpoint: {checkpoint_path}")
    model = algo_cls.load(checkpoint_path, env=vec, device=device)
    model.set_env(vec)
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--algo", type=str, default="ppo", choices=list(ALGOS.keys()))
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--dummy-vec", action="store_true")
    parser.add_argument("--terrain-difficulty", type=int, default=0,
                        help="0=flat, 1=hardcore (stumps/pits/stairs)")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        metavar="PATH",
                        help="Fine-tune from this .zip checkpoint.")
    parser.add_argument("--list-models", action="store_true",
                        help="List saved models (newest first) and exit.")
    args = parser.parse_args()

    if args.list_models:
        list_models()
        return

    algo_name = args.algo.lower()

    # Auto-detect algo from checkpoint's config.json
    if args.load_checkpoint:
        detected = _detect_algo(Path(args.load_checkpoint), algo_name)
        if detected != algo_name:
            print(f"  Auto-detected algo '{detected}' from checkpoint config.")
            algo_name = detected
    algo_cls = ALGOS[algo_name]

    # SAC/TD3 require single env
    off_policy = algo_name in ("sac", "td3")
    if off_policy and args.n_envs > 1:
        print(f"  {algo_name.upper()} works best with n_envs=1. Forcing n_envs=1.")
        args.n_envs = 1

    tag = args.tag or f"{algo_name}_{'ft' if args.load_checkpoint else 'run'}"
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"{tag}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["algo"] = algo_name
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    VecEnvCls = DummyVecEnv if (args.dummy_vec or off_policy) else SubprocVecEnv
    vec = VecEnvCls([make_env(i, args.seed, args.terrain_difficulty) for i in range(args.n_envs)])
    eval_vec = DummyVecEnv([make_env(0, args.seed + 1000, args.terrain_difficulty)])

    if args.load_checkpoint:
        model = load_checkpoint(Path(args.load_checkpoint), algo_cls, vec, args.device)
        model.tensorboard_log = str(run_dir / "tb")
        reset_num_timesteps = False
    else:
        model = algo_cls(
            "MlpPolicy",
            vec,
            verbose=1,
            seed=args.seed,
            device=args.device,
            tensorboard_log=str(run_dir / "tb"),
            policy_kwargs=dict(
                net_arch=[256, 256],
                activation_fn=nn.Tanh,
            ),
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
        )
        reset_num_timesteps = True

    callbacks = [
        EpisodeMetricsCallback(window=100),
        CheckpointCallback(
            save_freq=max(1, args.checkpoint_every // args.n_envs),
            save_path=str(run_dir / "checkpoints"),
            name_prefix=algo_name,
        ),
        EvalCallback(
            eval_vec,
            best_model_save_path=str(run_dir),
            log_path=str(run_dir / "eval_logs"),
            eval_freq=max(1, args.eval_every // args.n_envs),
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
        ),
    ]

    source = f" (fine-tuned from {args.load_checkpoint})" if args.load_checkpoint else ""
    print(f"\n  algo={algo_name.upper()}  timesteps={args.timesteps:,}  "
          f"n_envs={args.n_envs}  terrain={args.terrain_difficulty}{source}")

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_num_timesteps,
        progress_bar=True,
    )
    model.save(run_dir / "model.zip")
    eval_vec.close()
    vec.close()
    print(f"\n  Training complete. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
