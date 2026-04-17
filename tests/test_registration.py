import gymnasium as gym
import numpy as np

import mughead_walker  # noqa: F401 - triggers registration


def test_make_env():
    env = gym.make("MugheadWalker-v0")
    assert env.spec.id == "MugheadWalker-v0"
    env.close()


def test_reset_returns_obs_tuple():
    env = gym.make("MugheadWalker-v0")
    obs, info = env.reset(seed=0)
    assert obs.dtype == np.float32
    assert obs.shape == (40,)
    assert isinstance(info, dict)
    env.close()


def test_step_five_tuple():
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
    assert obs.shape == (40,)
    assert isinstance(reward, (int, float, np.floating))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    env.close()
