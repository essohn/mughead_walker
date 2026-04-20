#!/usr/bin/env bash
# Watch a trained MugheadWalker agent in a rendered window.
# Usage: ./scripts/play.sh <run_dir> [extra evaluate.py args...]
# Example: ./scripts/play.sh runs/ppo_smoke_test_20260420
set -euo pipefail
RUN_DIR="${1:?usage: ./scripts/play.sh <run_dir> [extra args...]}"
MODEL="$RUN_DIR/model.zip"
if [[ ! -f "$MODEL" ]]; then
    echo "error: $MODEL not found" >&2
    exit 1
fi
python training/evaluate.py --model "$MODEL" --render --episodes 3 "${@:2}"
