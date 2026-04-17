# MugheadWalker-v0

Gymnasium environment where a BipedalWalker carries a mug full of payloads.
Forked from `gymnasium.envs.box2d.bipedal_walker`. Built for the Yonsei
"Understanding and Applying AI" RL competition.

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

## Observation & action space

- **Observation:** 40-dim float32 vector
  - 0–23: original BipedalWalker state (hull pose/velocity, 4 joint angles/speeds, 2 leg ground-contact flags, 10 LIDAR measurements)
  - 24–29: 3 × payload hull-local position (x, y) / 50
  - 30–35: 3 × payload hull-local velocity (vx, vy) / 10
  - 36–38: 3 × `in_cup` flag (0/1)
  - 39: surviving payload count / 3
- **Action:** 4-dim continuous `[-1, 1]` (hip/knee torque for both legs).

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
    payload_mass_ratio=0.06,   # fraction of hull mass per payload
    payload_bounciness=0.15,   # restitution
    terrain_difficulty=0,      # only 0 supported now (1–3 in a later spec)
    obstacles=False,           # only False supported now
    external_force=0.0,        # only 0 supported now
)
```

## License

MIT. Derived from `gymnasium.envs.box2d.bipedal_walker`. See `mughead_walker/mughead_walker.py` header for original credits.
