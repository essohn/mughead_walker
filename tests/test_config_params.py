import gymnasium as gym
import numpy as np
import pytest

import mughead_walker  # noqa: F401


def test_num_payloads_zero():
    env = gym.make("MugheadWalker-v0", num_payloads=0)
    obs, _ = env.reset(seed=0)
    assert obs[41] == 0.0  # was obs[39]
    # Payload slots 26-41 all zero (shifted by +2 for waist slots at 24-25).
    np.testing.assert_array_equal(obs[26:42], np.zeros(16, dtype=np.float32))
    assert len(env.unwrapped.payloads) == 0
    env.close()


def test_num_payloads_one():
    env = gym.make("MugheadWalker-v0", num_payloads=1)
    obs, _ = env.reset(seed=0)
    assert obs[41] == pytest.approx(1.0 / 3.0)  # was obs[39]
    # in_cup slots: slot 0 populated, slots 1 and 2 zero.
    assert obs[38] == 1.0   # was obs[36]
    assert obs[39] == 0.0   # was obs[37]
    assert obs[40] == 0.0   # was obs[38]
    env.close()


def test_num_payloads_out_of_range_rejected():
    with pytest.raises((ValueError, AssertionError)):
        gym.make("MugheadWalker-v0", num_payloads=4)


def test_num_payloads_float_rejected():
    with pytest.raises(TypeError):
        gym.make("MugheadWalker-v0", num_payloads=3.0)


def test_num_payloads_bool_rejected():
    # bool is a subclass of int in Python; explicitly reject it
    with pytest.raises(TypeError):
        gym.make("MugheadWalker-v0", num_payloads=True)


def test_terrain_difficulty_nonzero_not_implemented():
    with pytest.raises(NotImplementedError):
        gym.make("MugheadWalker-v0", terrain_difficulty=1)


def test_obstacles_true_not_implemented():
    with pytest.raises(NotImplementedError):
        gym.make("MugheadWalker-v0", obstacles=True)


def test_external_force_nonzero_not_implemented():
    with pytest.raises(NotImplementedError):
        gym.make("MugheadWalker-v0", external_force=1.0)


def test_payload_bounciness_override():
    env = gym.make("MugheadWalker-v0", payload_bounciness=0.5)
    env.reset(seed=0)
    p = env.unwrapped.payloads[0]
    assert p.fixtures[0].restitution == pytest.approx(0.5)
    env.close()


def test_payload_mass_ratio_override():
    env_a = gym.make("MugheadWalker-v0", payload_mass_ratio=0.03)
    env_b = gym.make("MugheadWalker-v0", payload_mass_ratio=0.09)
    env_a.reset(seed=0); env_b.reset(seed=0)
    m_a = env_a.unwrapped.payloads[0].mass
    m_b = env_b.unwrapped.payloads[0].mass
    assert m_b > m_a * 2, f"higher ratio should produce heavier payload: {m_a=} {m_b=}"
    env_a.close(); env_b.close()


def test_payload_bounciness_out_of_range_rejected():
    with pytest.raises(ValueError):
        gym.make("MugheadWalker-v0", payload_bounciness=1.5)
    with pytest.raises(ValueError):
        gym.make("MugheadWalker-v0", payload_bounciness=-0.1)


def test_payload_mass_ratio_nonpositive_rejected():
    with pytest.raises(ValueError):
        gym.make("MugheadWalker-v0", payload_mass_ratio=0.0)
    with pytest.raises(ValueError):
        gym.make("MugheadWalker-v0", payload_mass_ratio=-0.05)


# --- New tests for mug_inner_width (Step 7.5) ---

def test_mug_inner_width_narrow():
    env = gym.make("MugheadWalker-v0", mug_inner_width=30.0)
    env.reset(seed=0)
    env.close()  # should construct without error


def test_mug_inner_width_wide():
    env = gym.make("MugheadWalker-v0", mug_inner_width=100.0)
    env.reset(seed=0)
    env.close()


def test_mug_inner_width_out_of_range_rejected():
    with pytest.raises(ValueError):
        gym.make("MugheadWalker-v0", mug_inner_width=5.0)
    with pytest.raises(ValueError):
        gym.make("MugheadWalker-v0", mug_inner_width=200.0)
