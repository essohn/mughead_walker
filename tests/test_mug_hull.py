import gymnasium as gym
import numpy as np

import mughead_walker  # noqa: F401


def test_hull_has_three_fixtures():
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    hull = env.unwrapped.hull
    assert len(hull.fixtures) == 3, (
        f"mug hull should have 3 fixtures (slab + 2 walls), got {len(hull.fixtures)}"
    )
    env.close()


def test_hull_mass_close_to_original():
    """Total mug mass should be roughly similar to original BipedalWalker hull."""
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    mass = env.unwrapped.hull.mass
    # Original hull: ~5.0 density * ~1.09 m^2 area ≈ 5.4. Allow ±30% tolerance.
    assert 3.5 < mass < 7.5, f"hull mass {mass:.2f} outside expected range"
    env.close()


def test_rollout_100_steps_no_crash():
    """Zero-torque rollout for 100 steps — hull should remain intact."""
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        assert np.all(np.isfinite(obs))
    env.close()
