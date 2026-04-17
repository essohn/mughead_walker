import gymnasium as gym
import numpy as np

import mughead_walker  # noqa: F401


def test_mug_has_three_fixtures():
    """Mug body should have 3 fixtures (slab + 2 walls), now separate from chassis."""
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    mug = env.unwrapped.mug
    assert len(mug.fixtures) == 3, (
        f"mug should have 3 fixtures (slab + 2 walls), got {len(mug.fixtures)}"
    )
    env.close()


def test_chassis_exists_and_is_distinct_from_mug():
    """After reset, both chassis and mug bodies should exist and be different objects."""
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    u = env.unwrapped
    assert u.chassis is not None, "chassis body should exist"
    assert u.mug is not None, "mug body should exist"
    assert u.chassis is not u.mug, "chassis and mug must be distinct bodies"
    env.close()


def test_mug_inner_width_default_50():
    """The default mug_inner_width should be 50.0 (spec §2.1)."""
    env = gym.make("MugheadWalker-v0")
    assert env.unwrapped.mug_inner_width == 50.0
    env.close()


def test_mug_mass_close_to_original():
    """Total mug mass should be roughly similar to original BipedalWalker hull."""
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    mass = env.unwrapped.mug.mass
    # Original hull: ~5.0 density * ~1.09 m^2 area ≈ 5.4. Allow generous tolerance
    # since mug geometry changed (narrower).
    assert 2.0 < mass < 10.0, f"mug mass {mass:.2f} outside expected range"
    env.close()


def test_rollout_100_steps_no_crash():
    """Zero-torque rollout for 100 steps — mug and chassis should remain intact."""
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(np.zeros(5, dtype=np.float32))
        assert np.all(np.isfinite(obs))
    env.close()
