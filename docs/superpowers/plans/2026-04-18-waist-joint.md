# Waist Joint + Narrower Mug — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the Hull into chassis + mug bodies connected by a motorized waist joint, and narrow the mug interior from 70 → 50, to make payload management a real strategic axis (target: PPO 1M baseline payload survival ≈ 2/3, down from 3/3).

**Spec:** [`docs/superpowers/specs/2026-04-18-waist-joint-design.md`](../specs/2026-04-18-waist-joint-design.md)

**Architecture:** Single env file (`mughead_walker/mughead_walker.py`) refactored. Tests updated in-place. Rendering updated to draw two polygons + hinge. Validation via re-trained PPO 1M baseline.

**Tech Stack:** Python 3.10+, Gymnasium 1.0+, Box2D via `gymnasium[box2d]`, SB3 2.8, pytest, matplotlib/tensorboard for curves.

---

## Preconditions

- On branch `feat/waist-joint` (already created).
- Working tree clean after spec commit.
- All 47 tests currently passing on the branch tip.

---

### Task 1: Core env refactor (body split + obs 42 + action 5 + mug-frame payloads + tests)

**Files:**
- Modify: `mughead_walker/mughead_walker.py`
- Modify: `tests/test_env.py`
- Modify: `tests/test_observation.py`
- Modify: `tests/test_payloads.py`
- Modify: `tests/test_mug_hull.py`
- Modify: `tests/test_reward_and_loss.py`
- Create: `tests/test_waist_joint.py`

This is the single coherent refactor of the env. It's large but interconnected — body split, obs layout, action dim, and collision all must change together or tests won't pass.

- [ ] **Step 1: Write failing tests for new shape**

In `tests/test_env.py`, update:

```python
def _zero_action():
    return np.zeros(5, dtype=np.float32)  # was 4


def test_spaces():
    env = gym.make("MugheadWalker-v0")
    assert env.observation_space.shape == (42,)  # was 40
    assert env.observation_space.dtype == np.float32
    assert env.action_space.shape == (5,)  # was 4
    assert env.action_space.dtype == np.float32
    assert np.all(env.action_space.low == -1)
    assert np.all(env.action_space.high == 1)
    env.close()


def test_reset_step_contract():
    env = gym.make("MugheadWalker-v0")
    obs, info = env.reset(seed=0)
    assert obs.shape == (42,) and obs.dtype == np.float32  # was 40
    assert isinstance(info, dict)
    obs2, reward, terminated, truncated, info2 = env.step(_zero_action())
    assert obs2.shape == (42,) and obs2.dtype == np.float32  # was 40
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
        for flag in obs[38:41]:  # was obs[36:39]
            assert flag in (0.0, 1.0)
        assert 0.0 <= obs[41] <= 1.0  # was obs[39]
        if terminated or truncated:
            env.reset(seed=0)
    env.close()


def test_payload_stays_in_cup_at_rest():
    env = gym.make("MugheadWalker-v0")
    obs, _ = env.reset(seed=0)
    for _ in range(100):
        obs, _, terminated, _, _ = env.step(_zero_action())
        if terminated:
            pytest.fail("walker fell over during rest test")
    np.testing.assert_array_equal(obs[38:41], [1.0, 1.0, 1.0])  # was obs[36:39]
    assert obs[41] == 1.0  # was obs[39]


def test_all_payloads_lost_no_terminate():
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    u = env.unwrapped
    for p in u.payloads:
        p.position = (u.chassis.position[0] + 10.0, u.chassis.position[1])  # was u.hull.position
    obs, _, terminated, *_ = env.step(_zero_action())
    assert not terminated
    assert obs[41] == 0.0  # was obs[39]
    obs2, _, terminated2, *_ = env.step(_zero_action())
    assert not terminated2
    env.close()


def test_configurable_num_payloads_zero():
    env = gym.make("MugheadWalker-v0", num_payloads=0)
    obs, _ = env.reset(seed=0)
    np.testing.assert_array_equal(obs[26:42], np.zeros(16, dtype=np.float32))  # was obs[24:40]
    obs2, reward, *_ = env.step(_zero_action())
    assert reward != -20
    env.close()


def test_info_dict_has_metrics():
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    _, _, _, _, info = env.step(_zero_action())
    assert "payloads_remaining" in info
    assert "distance" in info
    assert "waist_angle" in info  # NEW
    assert isinstance(info["payloads_remaining"], int)
    assert info["payloads_remaining"] == 3
    assert isinstance(info["distance"], float)
    assert isinstance(info["waist_angle"], float)
    env.close()


def test_hull_fall_terminates():
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    u = env.unwrapped
    u.game_over = True
    _, reward, terminated, *_ = env.step(_zero_action())
    assert terminated
    assert reward == -100
```

`test_reward_is_python_float` stays unchanged.

- [ ] **Step 2: Update test_observation.py index constants**

In `tests/test_observation.py`, any references to payload-obs indices (`obs[24:40]`, `obs[36:39]`, `obs[39]`) must shift by +2 to account for the two waist slots at indices 24–25. Add a new test asserting `obs[24]` and `obs[25]` exist and are finite (waist angle + speed). Search-and-replace offsets carefully; confirm each one lines up with spec §3.

- [ ] **Step 3: Update test_payloads.py**

Any assertions on hull-local payload positions need to switch to mug-local. Direct attribute access `u.hull` → `u.chassis` where it's being used for fall/locomotion reasoning; `u.mug` where it's being used for payload frame. Spatial assertions about "payload near hull" become "payload near mug".

- [ ] **Step 4: Update test_mug_hull.py**

The expected mug inner width is now 50 (was 70). Outer width is 60. Add an assertion that both `u.chassis` and `u.mug` exist and are distinct Box2D bodies.

- [ ] **Step 5: Update test_reward_and_loss.py**

Any references to `u.hull.position` become `u.chassis.position` for distance/reward shaping, and `u.mug.position` for payload-loss geometry. Action zeroing uses `np.zeros(5)`.

- [ ] **Step 6: Create tests/test_waist_joint.py**

```python
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
    """Apply positive waist torque for many steps → mug tilts clockwise relative to chassis."""
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
    """Sustained torque cannot rotate the mug beyond ±π/4."""
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    action = np.zeros(5, dtype=np.float32)
    action[4] = 1.0
    for _ in range(200):
        _, _, terminated, truncated, info = env.step(action)
        assert abs(info["waist_angle"]) <= math.pi / 4 + 1e-3, \
            f"waist exceeded limit: {info['waist_angle']}"
        if terminated or truncated:
            env.reset(seed=0)
    env.close()
```

- [ ] **Step 7: Run tests → confirm they fail as expected**

```
pytest -q
```

Expected: many failures in `test_env.py`, `test_observation.py`, etc, plus the 4 new failures in `test_waist_joint.py`. All failures should be about missing attributes (`u.chassis`, `u.mug`, `u.waist_joint`), wrong shapes (`(40,)` vs `(42,)`), or out-of-range indexes.

- [ ] **Step 7.5: Add `mug_inner_width` configurable parameter**

In `MugheadWalkerEnv.__init__`, add after existing kwargs:

```python
mug_inner_width: float = 50.0,
```

After existing validation block:

```python
if not (10.0 <= mug_inner_width <= 120.0):
    raise ValueError(f"mug_inner_width must be in [10, 120], got {mug_inner_width}")
self.mug_inner_width = mug_inner_width
```

Update `_is_in_cup` to no longer be `@staticmethod` (it needs `self`):

```python
def _is_in_cup(self, local_pos) -> bool:
    lx, ly = local_pos[0] * SCALE, local_pos[1] * SCALE
    half_w = self.mug_inner_width / 2
    return (-half_w <= lx <= half_w) and (-9.5 <= ly <= 22.0)
```

Update `_build_mug_fixtures()` to use `self.mug_inner_width` instead of the constant.

Update the call site `self._is_in_cup(...)` (no longer `MugheadWalkerEnv._is_in_cup(...)`).

Add a test to `tests/test_config_params.py`:

```python
def test_mug_inner_width_narrow():
    env = gym.make("MugheadWalker-v0", mug_inner_width=30.0)
    env.reset(seed=0)
    env.close()  # should construct without error


def test_mug_inner_width_out_of_range_rejected():
    import pytest
    with pytest.raises(ValueError):
        gym.make("MugheadWalker-v0", mug_inner_width=5.0)
    with pytest.raises(ValueError):
        gym.make("MugheadWalker-v0", mug_inner_width=200.0)
```

- [ ] **Step 8: Update constants in mughead_walker.py**

At the top of `mughead_walker/mughead_walker.py`:

```python
# --- Mug interior (spec §2.1) ---
# MUG_INNER_WIDTH is now per-instance (self.mug_inner_width); default 50 set in __init__.
MUG_OUTER_WIDTH = 60  # was 80
MUG_WALL_T = 5         # wall thickness (unchanged)
# MUG_HEIGHT etc unchanged

# --- Chassis (new body, spec §2.1) ---
CHASSIS_W = 20        # width in physics units
CHASSIS_H = 6         # height
CHASSIS_DENSITY = 5.0  # match old hull density

# --- Waist joint (new, spec §2.1) ---
WAIST_TORQUE = 80     # = MOTORS_TORQUE (hip torque)
WAIST_SPEED = 4.0     # = SPEED_HIP
WAIST_LIMIT = math.pi / 4  # ±45 degrees

# --- Collision categories (spec §2.3) ---
CAT_TERRAIN = 0x01
CAT_LEG = 0x20
CAT_MUG = 0x40
CAT_CHASSIS = 0x80  # NEW
CAT_PAYLOAD = 0x100  # shift to free up 0x80
```

Adjust existing `CAT_PAYLOAD` users accordingly.

- [ ] **Step 9: Replace body creation — `_create_chassis` and `_create_mug`**

Delete the existing monolithic `self.hull = self.world.CreateDynamicBody(...)` block. Replace with two methods called from `reset()`:

```python
def _create_chassis(self):
    """Narrow pelvis body; legs attach here."""
    init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
    init_y = TERRAIN_HEIGHT + 2 * LEG_H
    poly = [
        (-CHASSIS_W / 2, -CHASSIS_H / 2),
        ( CHASSIS_W / 2, -CHASSIS_H / 2),
        ( CHASSIS_W / 2,  CHASSIS_H / 2),
        (-CHASSIS_W / 2,  CHASSIS_H / 2),
    ]
    fix = fixtureDef(
        shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in poly]),
        density=CHASSIS_DENSITY,
        friction=0.1,
        restitution=0.0,
        categoryBits=CAT_CHASSIS,
        maskBits=CAT_TERRAIN,
    )
    self.chassis = self.world.CreateDynamicBody(
        position=(init_x, init_y),
        fixtures=fix,
    )
    self.chassis.color1 = (127, 51, 229)
    self.chassis.color2 = (76, 30, 137)

def _create_mug(self):
    """U-shaped open-top cup above the chassis; carries payloads."""
    init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
    init_y = TERRAIN_HEIGHT + 2 * LEG_H + (CHASSIS_H + MUG_HEIGHT) / SCALE / 2
    fixtures = self._build_mug_fixtures()  # already returns a list of fixtureDef
    self.mug = self.world.CreateDynamicBody(
        position=(init_x, init_y),
        fixtures=fixtures,
    )
    self.mug.color1 = (127, 51, 229)
    self.mug.color2 = (76, 30, 137)
```

(Colors are illustrative — copy the spec's pastel if preferred. Keep chassis + mug visually distinct.)

Make sure `_build_mug_fixtures` uses the new narrower dimensions and correct `categoryBits=CAT_MUG`, `maskBits=CAT_TERRAIN | CAT_PAYLOAD`. The left/right/bottom walls should honor `MUG_INNER_WIDTH=50`.

- [ ] **Step 10: Create the waist joint**

```python
def _create_waist_joint(self):
    """Motorized revolute joint between chassis top and mug bottom."""
    self.waist_joint = self.world.CreateRevoluteJoint(
        bodyA=self.chassis,
        bodyB=self.mug,
        anchor=(self.chassis.position[0],
                self.chassis.position[1] + CHASSIS_H / 2 / SCALE),
        enableMotor=True,
        enableLimit=True,
        lowerAngle=-WAIST_LIMIT,
        upperAngle=+WAIST_LIMIT,
        motorSpeed=0.0,
        maxMotorTorque=WAIST_TORQUE,
        collideConnected=False,
    )
```

Call this from `reset()` after both bodies exist.

- [ ] **Step 11: Update `_create_payloads`**

Payloads are initialized in the **mug's** local frame (was hull-local). X-range must fit inside the narrower interior. Given `MUG_INNER_WIDTH=50`, keep payloads within x ∈ [-20, +20] in mug-local. Code:

```python
def _create_payloads(self):
    self.payloads: list = []
    if self.num_payloads <= 0:
        return
    payload_radius = (MUG_INNER_WIDTH / 8) / SCALE
    mass_per = self.chassis.mass * self.payload_mass_ratio  # was hull.mass
    density = mass_per / (math.pi * payload_radius ** 2)
    # Initial positions: vertically spaced, slightly jittered x for stability
    base_x = self.mug.position[0]
    base_y = self.mug.position[1]
    for i in range(min(self.num_payloads, 3)):
        x = base_x
        y = base_y - (MUG_HEIGHT / 3) / SCALE + i * (payload_radius * 2.5)
        fix = fixtureDef(
            shape=circleShape(radius=payload_radius),
            density=density,
            friction=0.4,
            restitution=self.payload_bounciness,
            categoryBits=CAT_PAYLOAD,
            maskBits=CAT_TERRAIN | CAT_MUG | CAT_LEG | CAT_PAYLOAD,
        )
        body = self.world.CreateDynamicBody(position=(x, y), fixtures=fix)
        body.color1 = PAYLOAD_COLORS[i % len(PAYLOAD_COLORS)][0]
        body.color2 = PAYLOAD_COLORS[i % len(PAYLOAD_COLORS)][1]
        self.payloads.append(body)
```

(Adjust `PAYLOAD_COLORS` references to match existing code.)

- [ ] **Step 12: Update `_is_in_cup` and `_check_payload_losses`**

```python
@staticmethod
def _is_in_cup(local_pos) -> bool:
    """Mug-local rectangle (inner interior) check."""
    lx, ly = local_pos[0] * SCALE, local_pos[1] * SCALE
    return (-25.0 <= lx <= 25.0) and (-9.5 <= ly <= 22.0)
    # x range tightened from ±35 to ±25 (MUG_INNER_WIDTH=50)

def _check_payload_losses(self) -> int:
    num_lost = 0
    for i, p in enumerate(self.payloads):
        if p is None:
            continue
        dx = p.position[0] - self.mug.position[0]  # was hull
        dy = p.position[1] - self.mug.position[1]  # was hull
        dist = (dx * dx + dy * dy) ** 0.5
        below_ground = p.position[1] < (TERRAIN_HEIGHT - LOSS_Y_BELOW_GROUND)
        if dist > LOSS_DISTANCE or below_ground:
            self.world.DestroyBody(p)
            self.payloads[i] = None
            num_lost += 1
    return num_lost
```

- [ ] **Step 13: Update `_payload_obs` for mug-local frame**

Inside `_payload_obs`, replace every `self.hull.GetLocalPoint(...)` / `GetLocalVector(...)` / `GetLinearVelocityFromWorldPoint(...)` with `self.mug.<same method>`.

- [ ] **Step 14: Update `observation_space` and `action_space`**

```python
low = np.array([-np.inf] * 42, dtype=np.float32)  # was 40
high = np.array([np.inf] * 42, dtype=np.float32)
self.observation_space = spaces.Box(low, high, dtype=np.float32)
self.action_space = spaces.Box(-1, 1, shape=(5,), dtype=np.float32)  # was (4,)
```

- [ ] **Step 15: Rewrite `step()` state assembly**

Replace every `self.hull.<x>` with `self.chassis.<x>`, plus insert waist slots, and apply waist torque from `action[4]`:

```python
# existing joints 0-3 control unchanged ...

# NEW: waist control from action[4]
self.waist_joint.motorSpeed = float(WAIST_SPEED * np.sign(action[4]))
self.waist_joint.maxMotorTorque = float(
    WAIST_TORQUE * np.clip(np.abs(action[4]), 0, 1)
)

self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

pos = self.chassis.position  # was self.hull.position
vel = self.chassis.linearVelocity  # was self.hull.linearVelocity

# LIDAR origin is chassis.position (unchanged conceptually)
for i in range(10):
    ...  # existing lidar loop, using pos

state = [
    self.chassis.angle,           # 0  was hull.angle
    2.0 * self.chassis.angularVelocity / FPS,  # 1
    0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,   # 2
    0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,   # 3
    self.joints[0].angle, self.joints[0].speed / SPEED_HIP,  # 4, 5
    self.joints[1].angle + 1.0, self.joints[1].speed / SPEED_KNEE,  # 6, 7
    1.0 if self.legs[1].ground_contact else 0.0,  # 8
    self.joints[2].angle, self.joints[2].speed / SPEED_HIP,  # 9, 10
    self.joints[3].angle + 1.0, self.joints[3].speed / SPEED_KNEE,  # 11, 12
    1.0 if self.legs[3].ground_contact else 0.0,  # 13
]
state += [l.fraction for l in self.lidar]  # 14-23
assert len(state) == 24
# NEW waist slots (spec §3)
state += [
    self.waist_joint.angle,                          # 24
    self.waist_joint.speed / SPEED_HIP,              # 25
]
assert len(state) == 26
state += self._payload_obs()  # 26-41 (16 elems, mug-local)
assert len(state) == 42
```

Shaping uses chassis:

```python
shaping = 130 * pos[0] / SCALE
shaping -= 5.0 * abs(state[0])  # chassis angle
...
```

Torque penalty now iterates over all 5 action dims (already handled by the existing `for a in action` loop — just confirm it iterates `len(action)`, not hardcoded 4).

`in_cup_count = int(sum(state[38:41]))` (was `state[36:39]`).

Info dict:

```python
info = {
    "payloads_remaining": sum(1 for p in self.payloads if p is not None),
    "distance": float(pos.x),
    "waist_angle": float(self.waist_joint.angle),
}
```

- [ ] **Step 16: Update `_destroy`**

Destroy terrain, legs, chassis, mug, and the waist joint. The waist joint must be destroyed *before* its anchor bodies. Payload destruction path unchanged (uses `self.world.DestroyBody(p)` inline).

- [ ] **Step 17: Update `ContactDetector`**

Change:

```python
if self.env.hull == contact.fixtureA.body or ...
```

to:

```python
if contact.fixtureA.body in (self.env.chassis, self.env.mug) or \
   contact.fixtureB.body in (self.env.chassis, self.env.mug):
    # Ignore expected payload contacts with mug
    other = (contact.fixtureB.body if contact.fixtureA.body in (self.env.chassis, self.env.mug)
             else contact.fixtureA.body)
    if other in [p for p in self.env.payloads if p is not None]:
        return
    self.env.game_over = True
```

Leg ground-contact detection is unchanged.

- [ ] **Step 18: Run tests**

```
pytest -q
```

Expected: all existing tests green with updated indices; new `test_waist_joint.py` tests green.

- [ ] **Step 19: Commit**

```
git add -A
git commit -m "feat: split hull into chassis + mug with motorized waist joint

- MUG_INNER_WIDTH 70→50 (spec §2.1)
- Two bodies connected by a revolute joint with ±π/4 limits
- Obs 40→42 (waist angle+speed at 24,25; payload block shifts to 26-41)
- Action 4→5 (waist torque at action[4])
- Payload observations + loss check now in mug-local frame
- ContactDetector treats both chassis and mug as fall-triggering
- info dict gains waist_angle

Closes spec §2-5. Rendering updated in Task 2."
```

---

### Task 2: Rendering updates

**Files:**
- Modify: `mughead_walker/mughead_walker.py` (render section only)

- [ ] **Step 1: Update the `drawlist`**

The existing `render()` builds `drawlist` dynamically. After the Task 1 refactor, it should include both `self.chassis` and `self.mug`:

```python
drawlist = self.terrain + self.legs + [self.chassis, self.mug] \
           + [p for p in self.payloads if p is not None]
```

Each polygon/circle fixture in `body.fixtures` draws as before — pygame doesn't care that chassis and mug are separate bodies. Verify visually.

- [ ] **Step 2: Draw the waist hinge marker**

After the drawlist loop and after HUD text, overlay a small filled circle at the waist joint anchor, rendered in world → screen coordinates:

```python
anchor_world = self.waist_joint.anchorA  # Box2D reports joint anchor
sx = anchor_world[0] * SCALE - self.scroll * SCALE
sy = VIEWPORT_H - anchor_world[1] * SCALE
pygame.draw.circle(self.surf, (40, 40, 40), (sx, sy), 3)
```

(If `anchorA` isn't directly exposed by pybox2d, derive it from `chassis.GetWorldPoint(waist_joint.localAnchorA)`.)

- [ ] **Step 3: Manual verification**

```
python examples/random_agent.py --episodes 1 --render
```

Expect:
- Walker has two visible bodies stacked (chassis below, mug above).
- A small dark circle visible between them (hinge).
- Mug sometimes tilts relative to chassis when it's perturbed.

- [ ] **Step 4: Run tests**

```
pytest -q tests/test_rendering.py
```

Expected: all pass (rgb_array shape unchanged).

- [ ] **Step 5: Commit**

```
git add mughead_walker/mughead_walker.py
git commit -m "feat: render chassis + mug separately with waist hinge marker"
```

---

### Task 3: Visual smoke check + random agent verification

**Files:** none (verification only, unless bugs found)

- [ ] **Step 1: Run random agent with rendering**

```
python examples/random_agent.py --episodes 3 --render
```

Manually verify:
- [ ] Walker visible with two bodies stacked, connected at the waist
- [ ] Hinge marker shows
- [ ] Payloads spawn inside the narrower mug and initially settle in it
- [ ] With random actions, the mug visibly tilts relative to chassis (waist joint working)
- [ ] Payloads can fall out of the narrower mug during motion
- [ ] Walker falls over at some point → game ends cleanly

- [ ] **Step 2: Full test sweep**

```
pytest -q
```

Expected: all tests pass. If anything red, fix in this task.

- [ ] **Step 3: If any fix was needed, commit**

```
git add -A
git commit -m "fix: <describe>"
```

(If no fix, no commit needed.)

---

### Task 4: PPO 1M re-training

**Files:**
- Modify: `examples/plot_curves.py` (maybe — verify TB tags unchanged)

- [ ] **Step 1: Launch training in background**

```
python examples/train_ppo.py --timesteps 1000000 --n-envs 8 --tag ppo_waist 2>&1 | tee runs/train_waist.log
```

Wait for completion notification (~5 min expected, possibly longer due to 5-dim action + 42-dim obs).

- [ ] **Step 2: Evaluate**

```
python examples/evaluate.py --model runs/ppo_waist_<ts>/model.zip \
    --episodes 10 \
    --out runs/ppo_waist_<ts>/eval.json
```

- [ ] **Step 3: Generate curves**

```
python examples/plot_curves.py runs/ppo_waist_<ts>
```

- [ ] **Step 4: Copy artifacts into docs**

```
cp runs/ppo_waist_<ts>/curves.png docs/baseline/ppo_waist_curves.png
cp runs/ppo_waist_<ts>/eval.json docs/baseline/ppo_waist_eval.json
```

---

### Task 5: Update report, README, and commit

**Files:**
- Modify: `docs/baseline/ppo_baseline_report.md`
- Modify: `README.md`
- Create: the new curves PNG + eval.json are already in `docs/baseline/`

- [ ] **Step 1: Update the baseline report**

Prepend (or add a top section to) `docs/baseline/ppo_baseline_report.md` with a **Before vs After** comparison:

```markdown
## 2026-04-18 Update: Waist joint + narrower mug (v0)

After adding a motorized waist joint and narrowing the mug interior from 70 → 50
(spec: [waist-joint-design](../superpowers/specs/2026-04-18-waist-joint-design.md)),
the re-trained baseline shows:

| Metric | Before (single hull) | After (waist + narrow mug) | Δ |
| --- | --- | --- | --- |
| Mean reward (10 ep) | 261.9 | <NEW> | |
| Mean payload survival | 3.0 / 3 | <NEW> | |
| Mean distance | 57.0 m | <NEW> | |
| Success rate | 40 % | <NEW> | |

![curves](./ppo_waist_curves.png)

Verdict: <PLACEHOLDER: too easy / too hard / calibrated>.
```

Keep the original report content below, so the history is preserved.

- [ ] **Step 2: Update README.md**

Observation table: change `40-dim` → `42-dim`, insert waist angle + speed rows at 24–25, shift payload index ranges. Action section: 4-dim → 5-dim, add waist mention.

- [ ] **Step 3: Run full test suite one last time**

```
pytest -q
```

- [ ] **Step 4: Commit**

```
git add docs/ README.md
git commit -m "docs: update baseline report + README for waist joint v0

Post-refactor 1M PPO baseline: <metrics summary>.
Payload survival <changed from 3.0 to X>."
```

---

### Task 6: Merge + push

- [ ] **Step 1: Verify on feature branch**

```
git status
git log --oneline main..HEAD
pytest -q
```

- [ ] **Step 2: Merge to main**

```
git checkout main
git merge feat/waist-joint
pytest -q
```

- [ ] **Step 3: Push**

```
git push origin main
```

- [ ] **Step 4: Delete feature branch**

```
git branch -d feat/waist-joint
```

---

## Self-review checklist

- Every index shift (obs[36:39] → obs[38:41], obs[39] → obs[41]) appears consistently across Task 1 Steps 1–5 and Step 15.
- Action-space dim bump (4 → 5) is reflected in both the test `_zero_action` helpers and the env's Box bounds.
- `self.hull` references are fully replaced: physics/LIDAR/shaping/distance → `self.chassis`; payload local frame / in-cup / loss → `self.mug`.
- `ContactDetector` treats terrain contact with either chassis or mug as game_over.
- Waist joint: `enableMotor=True`, `enableLimit=True`, `lowerAngle/upperAngle = ±WAIST_LIMIT`, `collideConnected=False`.
- `info["waist_angle"]` present in every step.
- Rendering draws both bodies and a hinge marker.
- Baseline report gets an update section (not a replacement) so the old numbers remain.
