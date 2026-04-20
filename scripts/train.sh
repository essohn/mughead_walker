#!/usr/bin/env bash
# Train a PPO agent on MugheadWalker-v0.
# Usage: ./scripts/train.sh <tag> [extra train_ppo.py args...]
# Defaults (timesteps, n-envs, etc.) come from training/train_ppo.py's argparse.
set -euo pipefail
TAG="${1:?usage: ./scripts/train.sh <tag> [extra args...]}"
python training/train_ppo.py --tag "$TAG" "${@:2}"
