"""Evaluate a trained MugheadWalker-v0 policy over N episodes.

Usage:
    python examples/evaluate.py --model runs/ppo_baseline_20260418_120000/model.zip
    python examples/evaluate.py --model <path> --episodes 10 --render

Reports (per spec §4.2):
    - mean episode reward
    - mean payloads surviving
    - mean distance travelled
    - success rate (did not fall)
"""
import argparse
import json
import statistics
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, SAC, TD3, A2C

ALGOS = {"ppo": PPO, "sac": SAC, "td3": TD3, "a2c": A2C}

import mughead_walker  # noqa: F401


def evaluate(model_path: str, episodes: int, render: bool, seed: int, out_path: str | None, terrain_difficulty: int = 0, algo: str = "ppo"):
    render_mode = "human" if render else None
    env = gym.make("MugheadWalker-v0", render_mode=render_mode, terrain_difficulty=terrain_difficulty)
    algo_cls = ALGOS[algo.lower()]
    model = algo_cls.load(model_path, device="cpu")

    rewards: list[float] = []
    payloads: list[int] = []
    distances: list[float] = []
    successes: list[int] = []

    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        ep_reward = 0.0
        last_info = info
        terminated = truncated = False
        fell = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, last_info = env.step(action)
            ep_reward += float(reward)
            if terminated and reward <= -80:
                fell = True
        rewards.append(ep_reward)
        payloads.append(int(last_info.get("payloads_remaining", 0)))
        distances.append(float(last_info.get("distance", 0.0)))
        successes.append(0 if fell else 1)
        print(f"ep {ep:2d}: reward={ep_reward:7.2f}  payloads={payloads[-1]}  "
              f"distance={distances[-1]:6.2f}  success={bool(successes[-1])}")

    env.close()

    summary = {
        "model": model_path,
        "episodes": episodes,
        "mean_reward": statistics.mean(rewards),
        "std_reward": statistics.pstdev(rewards),
        "mean_payloads": statistics.mean(payloads),
        "mean_distance": statistics.mean(distances),
        "success_rate": statistics.mean(successes),
    }
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if out_path:
        Path(out_path).write_text(json.dumps(summary, indent=2))
        print(f"\nWrote {out_path}")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional JSON output path for summary metrics.")
    parser.add_argument("--terrain-difficulty", type=int, default=0,
                        help="0=flat, 1=hardcore (stumps/pits/stairs)")
    parser.add_argument("--algo", type=str, default="ppo",
                        choices=list(ALGOS.keys()),
                        help="Algorithm used to train the model")
    args = parser.parse_args()
    evaluate(args.model, args.episodes, args.render, args.seed, args.out, args.terrain_difficulty, args.algo)


if __name__ == "__main__":
    main()
