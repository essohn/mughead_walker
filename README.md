# MugheadWalker-v0

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/essohn/mughead_walker/blob/main/notebooks/mughead_walker_colab.ipynb)

Gymnasium environment where a BipedalWalker carries a mug full of payloads.
Forked from `gymnasium.envs.box2d.bipedal_walker`. Built for the Yonsei
"Understanding and Applying AI" RL competition.

Quickest way to try it: click the **Open In Colab** badge above — install, train a small PPO, watch the agent, all in your browser.

## Install

```bash
pip install -e '.[dev]'
```

Requires Python 3.10+, `gymnasium[box2d]`, `pygame`, `pytest`.

## Use

```python
import gymnasium as gym
import mughead_walker  # registers MugheadWalker-v0

env = gym.make("MugheadWalker-v0", render_mode="human")
obs, info = env.reset(seed=0)
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

### Quick visual sanity check

```bash
python examples/random_agent.py --episodes 3 --render
```

### Run tests

```bash
pytest
```

### Train a PPO baseline

```bash
pip install -e '.[rl]'
./scripts/train.sh ppo_baseline                   # defaults: 1M timesteps, 8 envs
./scripts/play.sh runs/<run_dir>                  # watches 3 episodes rendered
python tools/plot_curves.py runs/<run_dir>        # or use interactive: python tools/manager.py
```

Baseline results and difficulty assessment: [`docs/baseline/ppo_baseline_report.md`](docs/baseline/ppo_baseline_report.md).

## Bodies

Instead of a single hull, the walker has a **chassis** (narrow pelvis where the legs mount) and a **mug** (U-shaped cup that carries payloads). They are connected by a motorized revolute **waist joint** with ±π/4 limits, so the agent can actively tilt the cup to keep cargo in. Either body touching the ground terminates the episode.

## Observation & action space

- **Observation:** 42-dim float32 vector
  - 0–3: chassis angle, angular velocity, linear (vx, vy)
  - 4–13: 4 joint angles/speeds + 2 leg ground-contact flags
  - 14–23: 10 LIDAR measurements
  - 24–25: waist joint angle and speed
  - 26–31: 3 × payload mug-local position (x, y) / (50/SCALE)
  - 32–37: 3 × payload mug-local velocity (vx, vy) / (10/SCALE)
  - 38–40: 3 × `in_cup` flag (0/1)
  - 41: surviving payload count / 3
- **Action:** 5-dim continuous `[-1, 1]` — hip 1, knee 1, hip 2, knee 2, waist.

## Reward

Original BipedalWalker shaping (forward distance, angle penalty, motor torque penalty, −100 on hull fall) plus:
- `+0.05 × in_cup_count` every step.
- `−20` per payload lost.
- Episode continues when all payloads are lost.

## Configurable parameters

```python
gym.make(
    "MugheadWalker-v0",
    num_payloads=3,            # 0–3 supported
    payload_mass_ratio=0.06,   # fraction of chassis mass per payload
    payload_bounciness=0.15,   # restitution
    mug_inner_width=50.0,      # cup interior width (10–120); wider = easier
    terrain_difficulty=0,      # only 0 supported now (1–3 in a later spec)
    obstacles=False,           # only False supported now
    external_force=0.0,        # only 0 supported now
)
```

## License

MIT. Derived from `gymnasium.envs.box2d.bipedal_walker`. See `mughead_walker/mughead_walker.py` header for original credits.
