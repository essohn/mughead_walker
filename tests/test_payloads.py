import math

import Box2D
import gymnasium as gym
import numpy as np

import mughead_walker  # noqa: F401


def test_three_payloads_created():
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    payloads = env.unwrapped.payloads
    assert len(payloads) == 3
    for p in payloads:
        assert p is not None
        assert len(p.fixtures) == 1
        assert isinstance(p.fixtures[0].shape, Box2D.b2CircleShape)
    env.close()


def test_payload_initial_positions_inside_mug():
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    hull = env.unwrapped.hull
    for p in env.unwrapped.payloads:
        local = hull.GetLocalPoint(p.position)
        # Payload center must be between walls and above the slab top.
        assert -35.0 / 30.0 < local[0] < 35.0 / 30.0
        assert -9.5 / 30.0 < local[1] < 22.0 / 30.0
    env.close()


def test_payload_mass_ratio():
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    hull_mass = env.unwrapped.hull.mass
    for p in env.unwrapped.payloads:
        ratio = p.mass / hull_mass
        # Default payload_mass_ratio = 0.06 — allow ±50% slack for float precision.
        assert 0.03 < ratio < 0.09, f"payload mass ratio {ratio:.3f} out of range"
    env.close()


def test_payload_reset_cleans_up():
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    first_ids = [id(p) for p in env.unwrapped.payloads]
    env.reset(seed=0)
    second_ids = [id(p) for p in env.unwrapped.payloads]
    assert first_ids != second_ids, "payloads should be fresh bodies after reset"
    env.close()
