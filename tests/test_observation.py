import gymnasium as gym
import numpy as np

import mughead_walker  # noqa: F401


def test_obs_shape_42():
    env = gym.make("MugheadWalker-v0")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (42,)  # was 40
    assert obs.dtype == np.float32
    env.close()


def test_waist_slots_exist_and_finite():
    """Waist angle at index 24 and waist speed at index 25 (new in spec §3)."""
    env = gym.make("MugheadWalker-v0")
    obs, _ = env.reset(seed=0)
    assert np.isfinite(obs[24]), f"waist angle obs[24] not finite: {obs[24]}"
    assert np.isfinite(obs[25]), f"waist speed obs[25] not finite: {obs[25]}"
    env.close()


def test_remaining_count_default_one():
    """With num_payloads=3, remaining_count/3 starts at 1.0 (index 41)."""
    env = gym.make("MugheadWalker-v0")
    obs, _ = env.reset(seed=0)
    assert obs[41] == 1.0  # was obs[39]
    env.close()


def test_in_cup_flags_initial():
    """At reset, all 3 payloads should be in the cup (in_cup=1 at indices 38-40)."""
    env = gym.make("MugheadWalker-v0")
    obs, _ = env.reset(seed=0)
    np.testing.assert_array_equal(obs[38:41], [1.0, 1.0, 1.0])  # was obs[36:39]
    env.close()


def test_obs_no_nan_over_rollout():
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    for _ in range(100):
        obs, _, _, _, _ = env.step(np.zeros(5, dtype=np.float32))  # was 4
        assert np.all(np.isfinite(obs))
        # in_cup flags always {0, 1}
        for f in obs[38:41]:  # was obs[36:39]
            assert f in (0.0, 1.0)
        # remaining_count/3 ∈ [0, 1]
        assert 0.0 <= obs[41] <= 1.0  # was obs[39]
    env.close()


def test_seed_reproducible():
    env1 = gym.make("MugheadWalker-v0")
    env2 = gym.make("MugheadWalker-v0")
    o1, _ = env1.reset(seed=42)
    o2, _ = env2.reset(seed=42)
    np.testing.assert_array_equal(o1, o2)
    for _ in range(30):
        a = np.zeros(5, dtype=np.float32)  # was 4
        s1, *_ = env1.step(a)
        s2, *_ = env2.step(a)
        np.testing.assert_array_equal(s1, s2)
    env1.close(); env2.close()
