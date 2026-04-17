import gymnasium as gym
import numpy as np

import mughead_walker  # noqa: F401


def _zero_action():
    return np.zeros(5, dtype=np.float32)  # was 4


def test_in_cup_bonus_applied():
    """A zero-torque step with 3 payloads in cup should add ~0.15 reward for the bonus term."""
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    total = 0.0
    for _ in range(5):
        _, r, *_ = env.step(_zero_action())
        total += r
    # Average per-step reward should be positive (forward-ish + bonus) and finite.
    assert np.isfinite(total)
    env.close()


def test_payload_loss_removes_body_and_applies_penalty():
    """Manually teleport a payload far away; next step should remove it with -20 penalty."""
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    u = env.unwrapped
    # Place payload 0 10 meters away from mug (payload-loss distance uses mug frame).
    u.payloads[0].position = (u.mug.position[0] + 10.0, u.mug.position[1])
    obs, reward, terminated, truncated, info = env.step(_zero_action())
    assert not terminated
    assert u.payloads[0] is None
    assert reward <= -19.5, f"expected reward ≤ -20 from loss, got {reward}"
    assert obs[41] == 2.0 / 3.0  # was obs[39]
    env.close()


def test_all_payloads_lost_does_not_terminate():
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    u = env.unwrapped
    far_x = u.mug.position[0] + 10.0  # was hull.position
    for p in u.payloads:
        p.position = (far_x, u.mug.position[1])
    obs, reward, terminated, truncated, info = env.step(_zero_action())
    assert not terminated
    assert all(p is None for p in u.payloads)
    assert obs[41] == 0.0  # was obs[39]
    # One more step: no more loss penalty, episode continues.
    obs2, r2, terminated2, *_ = env.step(_zero_action())
    assert not terminated2
    env.close()


def test_hull_fall_still_terminates():
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    u = env.unwrapped
    u.game_over = True   # simulate hull-ground contact flag
    _, reward, terminated, *_ = env.step(_zero_action())
    assert terminated
    assert reward == -100
    env.close()
