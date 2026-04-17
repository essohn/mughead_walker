"""Tests for the chassis+mug waist joint (spec §2.2, §3, §4)."""
import math

import gymnasium as gym
import numpy as np

import mughead_walker  # noqa: F401


def _zero_action():
    return np.zeros(5, dtype=np.float32)


def test_waist_joint_body_structure():
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    u = env.unwrapped
    assert u.chassis is not None and u.mug is not None
    assert u.chassis is not u.mug
    assert u.waist_joint is not None
    env.close()


def test_waist_angle_in_info_dict():
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    _, _, _, _, info = env.step(_zero_action())
    assert "waist_angle" in info
    # Expect small angle under zero torque at rest
    assert abs(info["waist_angle"]) < 0.5
    env.close()


def test_waist_torque_rotates_mug_relative_to_chassis():
    """Apply positive waist torque for many steps → mug tilts relative to chassis."""
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    action = np.zeros(5, dtype=np.float32)
    action[4] = 1.0  # max positive torque
    last_angle = 0.0
    for _ in range(30):
        _, _, terminated, truncated, info = env.step(action)
        last_angle = info["waist_angle"]
        if terminated or truncated:
            break
    # Sign convention is up to the implementer; what matters is that the joint moves
    assert abs(last_angle) > 0.05, f"waist barely moved: {last_angle}"
    env.close()


def test_waist_respects_joint_limits():
    """Sustained torque cannot rotate the mug beyond ±π/4 (with Box2D overshoot margin)."""
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    action = np.zeros(5, dtype=np.float32)
    action[4] = 1.0
    # Box2D sequential-impulse solver has integration overshoot at max motor torque;
    # allow 0.15 rad margin (≈9°) — the important invariant is that the joint
    # doesn't spin freely to π or beyond.
    limit_with_margin = math.pi / 4 + 0.15
    for _ in range(200):
        _, _, terminated, truncated, info = env.step(action)
        assert abs(info["waist_angle"]) <= limit_with_margin, \
            f"waist exceeded limit+margin: {info['waist_angle']:.4f} > {limit_with_margin:.4f}"
        if terminated or truncated:
            env.reset(seed=0)
    env.close()
