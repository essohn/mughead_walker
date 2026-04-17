"""Canonical smoke tests per spec §11."""
import gymnasium as gym
import numpy as np
import pytest

import mughead_walker  # noqa: F401


def _zero_action():
    return np.zeros(4, dtype=np.float32)


def test_registration():
    env = gym.make("MugheadWalker-v0")
    assert env.spec.id == "MugheadWalker-v0"
    env.close()


def test_spaces():
    env = gym.make("MugheadWalker-v0")
    assert env.observation_space.shape == (40,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.shape == (4,)
    assert env.action_space.dtype == np.float32
    assert np.all(env.action_space.low == -1)
    assert np.all(env.action_space.high == 1)
    env.close()


def test_reset_step_contract():
    env = gym.make("MugheadWalker-v0")
    obs, info = env.reset(seed=0)
    assert obs.shape == (40,) and obs.dtype == np.float32
    assert isinstance(info, dict)
    obs2, reward, terminated, truncated, info2 = env.step(_zero_action())
    assert obs2.shape == (40,) and obs2.dtype == np.float32
    assert isinstance(reward, (int, float, np.floating)) and np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info2, dict)
    env.close()


def test_obs_no_nan_long_rollout():
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    for _ in range(500):
        obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs))
        for flag in obs[36:39]:
            assert flag in (0.0, 1.0)
        assert 0.0 <= obs[39] <= 1.0
        if terminated or truncated:
            env.reset(seed=0)
    env.close()


def test_seed_reproducibility():
    env1 = gym.make("MugheadWalker-v0")
    env2 = gym.make("MugheadWalker-v0")
    o1, _ = env1.reset(seed=123)
    o2, _ = env2.reset(seed=123)
    np.testing.assert_array_equal(o1, o2)
    for _ in range(50):
        a = _zero_action()
        r1 = env1.step(a); r2 = env2.step(a)
        np.testing.assert_array_equal(r1[0], r2[0])
        assert r1[1] == r2[1]
    env1.close(); env2.close()


def test_payload_stays_in_cup_at_rest():
    """Physics sanity: 3 payloads at rest stay in the cup for 100 zero-torque steps."""
    env = gym.make("MugheadWalker-v0")
    obs, _ = env.reset(seed=0)
    for _ in range(100):
        obs, _, terminated, _, _ = env.step(_zero_action())
        if terminated:
            pytest.fail("walker fell over during rest test — hull/leg geometry may be wrong")
    np.testing.assert_array_equal(obs[36:39], [1.0, 1.0, 1.0])
    assert obs[39] == 1.0


def test_hull_fall_terminates():
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    u = env.unwrapped
    u.game_over = True
    _, reward, terminated, *_ = env.step(_zero_action())
    assert terminated
    assert reward == -100


def test_all_payloads_lost_no_terminate():
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    u = env.unwrapped
    for p in u.payloads:
        p.position = (u.hull.position[0] + 10.0, u.hull.position[1])
    obs, _, terminated, *_ = env.step(_zero_action())
    assert not terminated
    assert obs[39] == 0.0
    obs2, _, terminated2, *_ = env.step(_zero_action())
    assert not terminated2
    env.close()


def test_configurable_num_payloads_zero():
    env = gym.make("MugheadWalker-v0", num_payloads=0)
    obs, _ = env.reset(seed=0)
    np.testing.assert_array_equal(obs[24:40], np.zeros(16, dtype=np.float32))
    obs2, reward, *_ = env.step(_zero_action())
    assert reward != -20
    env.close()


def test_reward_is_python_float():
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    _, reward, *_ = env.step(_zero_action())
    assert type(reward) is float, f"reward type must be float, got {type(reward).__name__}"
    env.close()
