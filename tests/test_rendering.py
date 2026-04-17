import gymnasium as gym
import numpy as np

import mughead_walker  # noqa: F401


def test_rgb_array_shape_and_dtype():
    env = gym.make("MugheadWalker-v0", render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.dtype == np.uint8
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    env.close()


def test_rgb_array_contains_payload_colors():
    """After reset, we should see at least some red, green, and blue pixels from the 3 payloads."""
    env = gym.make("MugheadWalker-v0", render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    # Check presence of each payload color (exact match rare; use rough 'dominant channel' count).
    red_like = np.sum((frame[..., 0] > 180) & (frame[..., 1] < 120) & (frame[..., 2] < 120))
    green_like = np.sum((frame[..., 0] < 120) & (frame[..., 1] > 160) & (frame[..., 2] < 120))
    blue_like = np.sum((frame[..., 0] < 120) & (frame[..., 1] < 140) & (frame[..., 2] > 180))
    assert red_like > 10, f"expected red payload pixels, got {red_like}"
    assert green_like > 10, f"expected green payload pixels, got {green_like}"
    assert blue_like > 10, f"expected blue payload pixels, got {blue_like}"
    env.close()


def test_flash_activated_on_loss():
    env = gym.make("MugheadWalker-v0", render_mode="rgb_array")
    env.reset(seed=0)
    u = env.unwrapped
    # Force a loss.
    u.payloads[0].position = (u.hull.position[0] + 10.0, u.hull.position[1])
    env.step(np.zeros(4, dtype=np.float32))
    assert u._flash_frames > 0
    env.close()
