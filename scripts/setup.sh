#!/usr/bin/env bash
# Idempotent setup for MugheadWalker on a fresh macOS machine.
# Safe to re-run: each step checks current state before acting.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

log()  { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# 1. Homebrew
if have brew; then
  log "Homebrew already installed ($(brew --version | head -1))"
else
  log "Installing Homebrew..."
  NONINTERACTIVE=1 /bin/bash -c \
    "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [[ -x /usr/local/bin/brew ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
fi

# 2. System deps (Python + swig for box2d build)
for pkg in python@3.11 swig; do
  if brew list --versions "$pkg" >/dev/null 2>&1; then
    log "brew: $pkg already installed"
  else
    log "brew install $pkg"
    brew install "$pkg"
  fi
done

# 3. Pick python
if ! have "$PYTHON_BIN"; then
  if have python3.11; then PYTHON_BIN=python3.11
  elif have python3.10; then PYTHON_BIN=python3.10
  else
    warn "No python3.10/3.11 found; falling back to python3"
    PYTHON_BIN=python3
  fi
fi
log "Using $(${PYTHON_BIN} --version) at $(command -v ${PYTHON_BIN})"

# 4. venv
if [[ -d "$VENV_DIR" ]]; then
  log "venv already exists at $VENV_DIR"
else
  log "Creating venv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# 5. Install project
log "Upgrading pip"
pip install --upgrade pip >/dev/null

if python -c "import mughead_walker, stable_baselines3, pytest, tqdm, rich, matplotlib" 2>/dev/null; then
  log "Project + dev + rl extras already installed"
else
  log "Installing project with [dev,rl] extras (this may take a few minutes)"
  pip install -e '.[dev,rl]'
fi

log "Done. Activate with: source ${VENV_DIR}/bin/activate"
