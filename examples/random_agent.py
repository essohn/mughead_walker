"""Random policy rollout for visual inspection of MugheadWalker-v0."""
import argparse

import gymnasium as gym
import numpy as np

import mughead_walker  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--render", action="store_true", help="show pygame window")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = gym.make(
        "MugheadWalker-v0",
        render_mode="human" if args.render else None,
    )

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total_r = 0.0
        steps = 0
        while True:
            action = env.action_space.sample()
            obs, r, terminated, truncated, _ = env.step(action)
            total_r += r
            steps += 1
            if terminated or truncated:
                break
        remaining = int(round(obs[39] * 3))
        print(
            f"episode {ep}: steps={steps} "
            f"reward={total_r:.1f} "
            f"surviving_payloads={remaining}/3 "
            f"hull_x={env.unwrapped.hull.position[0]:.1f}m"
        )

    env.close()


if __name__ == "__main__":
    main()
