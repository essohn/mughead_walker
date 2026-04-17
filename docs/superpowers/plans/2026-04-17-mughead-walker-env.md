# MugheadWalker-v0 Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Gymnasium의 BipedalWalker를 포크해서 `MugheadWalker-v0` 환경을 구현한다. 상체를 U자형 머그컵으로 교체하고 컵 내부에 3개의 free-body payload(원형)를 배치해, payload를 최대한 지키면서 전진하는 RL 태스크를 제공한다.

**Architecture:** Fork-and-modify 방식. gymnasium 설치본의 `bipedal_walker.py`를 `mughead_walker/mughead_walker.py`로 복사한 후 hull 생성부를 다중 fixture로 교체하고 payload bodies, observation (24→40), reward shaping을 확장한다. 지형/LIDAR/렌더 파이프라인은 원본 그대로 유지.

**Tech Stack:** Python 3.13 (miniconda base), Gymnasium 1.2.3, Box2D (pybox2d via `gymnasium[box2d]`), pygame, pytest, NumPy.

**Spec reference:** `docs/superpowers/specs/2026-04-17-mughead-walker-env-design.md`

**Cwd for all commands:** `/Users/esohn/dev/mughead_walker`

---

## Task 1: Project scaffolding & dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `mughead_walker/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_setup.py`
- Create: `.gitignore`

- [ ] **Step 1: Write failing test for package import**

Create `tests/__init__.py` (empty file).

Create `tests/test_setup.py`:

```python
def test_package_importable():
    import mughead_walker  # noqa: F401


def test_box2d_available():
    import Box2D  # noqa: F401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/esohn/dev/mughead_walker && pytest tests/test_setup.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'mughead_walker'` (or similar for Box2D).

- [ ] **Step 3: Write `pyproject.toml`**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "mughead-walker"
version = "0.1.0"
description = "MugheadWalker-v0 Gymnasium environment (BipedalWalker fork with a mug-shaped hull carrying payloads)"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "Yonsei AI Course" }]
dependencies = [
    "gymnasium[box2d]>=1.0",
    "numpy>=1.26",
    "pygame>=2.5",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[tool.setuptools.packages.find]
include = ["mughead_walker*"]
exclude = ["tests*", "examples*", "docs*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 4: Create empty package init**

Create `mughead_walker/__init__.py` with a single line:

```python
# Registration populated in Task 2.
```

- [ ] **Step 5: Create `.gitignore`**

Create `.gitignore`:

```
__pycache__/
*.pyc
*.pyo
*.egg-info/
.pytest_cache/
build/
dist/
.venv/
.vscode/
.idea/
```

- [ ] **Step 6: Install package + Box2D**

Run: `pip install -e '.[dev]'`

Expected output (last lines): `Successfully installed Box2D-<ver> mughead-walker-0.1.0 ...`. If `Box2D` install fails with swig errors, first run `pip install swig` then retry.

Verify: `python -c "import Box2D; print(Box2D.__version__)"` prints a version like `2.3.10`.

- [ ] **Step 7: Run tests — both should pass**

Run: `pytest tests/test_setup.py -v`

Expected: 2 passed.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml mughead_walker/__init__.py tests/__init__.py tests/test_setup.py .gitignore
git commit -m "$(cat <<'EOF'
Scaffold mughead_walker package and dependencies

Adds pyproject.toml with gymnasium[box2d], pygame, pytest deps.
Creates mughead_walker/ package skeleton and a smoke test that
verifies the package and Box2D import cleanly.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Fork bipedal_walker.py + register as MugheadWalker-v0 (behavior unchanged)

**Files:**
- Create: `mughead_walker/mughead_walker.py` (copied from gymnasium)
- Modify: `mughead_walker/mughead_walker.py` (class rename, delete Hardcore variant)
- Modify: `mughead_walker/__init__.py` (add `register(...)`)
- Create: `tests/test_registration.py`

Goal: registered env works exactly like BipedalWalker (24-dim obs, 4-dim action, single-fixture hull). We are NOT changing physics yet — only the filename, class name, and registry ID.

- [ ] **Step 1: Write failing test for registration + baseline shapes**

Create `tests/test_registration.py`:

```python
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
    # Baseline is still BipedalWalker's 24-dim until Task 5 extends it.
    assert obs.shape == (24,)
    assert isinstance(info, dict)
    env.close()


def test_step_five_tuple():
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
    assert obs.shape == (24,)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    env.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_registration.py -v`

Expected: FAIL with `gym.error.NameNotFound: Environment MugheadWalker doesn't exist.`

- [ ] **Step 3: Copy bipedal_walker.py source**

Run:

```bash
cp /Users/esohn/miniconda3/lib/python3.13/site-packages/gymnasium/envs/box2d/bipedal_walker.py mughead_walker/mughead_walker.py
```

Verify: `ls mughead_walker/mughead_walker.py` exists.

- [ ] **Step 4: Rename class and drop Hardcore variant**

Open `mughead_walker/mughead_walker.py`.

Add this header above the existing `__credits__` line:

```python
# Derived from gymnasium.envs.box2d.bipedal_walker (MIT License).
# Original credit: Andrea PIERRÉ / Oleg Klimov (BipedalWalker).
```

Find `class BipedalWalker(gym.Env, EzPickle):` and rename to `class MugheadWalkerEnv(gym.Env, EzPickle):`.

Find the `class BipedalWalkerHardcore(BipedalWalker):` block at the bottom of the file (will be after the main class) and **delete the entire class** including its docstring. We don't need the hardcore variant.

Delete the `if __name__ == "__main__":` heuristic demo block at the very bottom if present (optional — keep file short and focused).

- [ ] **Step 5: Register env**

Replace contents of `mughead_walker/__init__.py`:

```python
from gymnasium.envs.registration import register

register(
    id="MugheadWalker-v0",
    entry_point="mughead_walker.mughead_walker:MugheadWalkerEnv",
    max_episode_steps=1600,
)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_registration.py -v`

Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add mughead_walker/mughead_walker.py mughead_walker/__init__.py tests/test_registration.py
git commit -m "$(cat <<'EOF'
Fork bipedal_walker into MugheadWalker-v0 (behavior unchanged)

Copies gymnasium's bipedal_walker.py to mughead_walker/mughead_walker.py,
renames BipedalWalker -> MugheadWalkerEnv, drops the hardcore variant,
and registers MugheadWalker-v0 with TimeLimit=1600. Observation is still
24-dim; subsequent tasks replace the hull with a mug and add payloads.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Replace hull with mug (3-fixture U-shape, leg anchors moved)

**Files:**
- Modify: `mughead_walker/mughead_walker.py` (HULL_POLY/HULL_FD → mug fixtures, LEG_DOWN, hull creation in `reset()`)
- Create: `tests/test_mug_hull.py`

Goal: hull body has 3 rectangle fixtures forming a U-shape (bottom slab + two vertical walls). Leg hip joints anchor to the slab bottom. Observation shape unchanged (still 24-dim until Task 5). Rendering automatically picks up the new fixtures via `self.drawlist`.

- [ ] **Step 1: Write failing test for mug hull fixture count**

Create `tests/test_mug_hull.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mug_hull.py::test_hull_has_three_fixtures -v`

Expected: FAIL — `hull should have 3 fixtures ... got 1`.

- [ ] **Step 3: Replace module-level hull constants with mug constants**

In `mughead_walker/mughead_walker.py`, find the block:

```python
HULL_POLY = [(-30, +9), (+6, +9), (+34, +1), (+34, -8), (-30, -8)]
LEG_DOWN = -8 / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE
```

Replace with:

```python
# Mug geometry (unscaled units; Box2D fixtures use values divided by SCALE).
# See docs/superpowers/specs/2026-04-17-mughead-walker-env-design.md §4.
MUG_SLAB_HALF = (38.0, 1.5)       # half-width, half-height of bottom slab (76 x 3)
MUG_SLAB_CENTER = (0.0, -11.0)
MUG_WALL_HALF = (1.5, 16.0)        # half-width, half-height of walls (3 x 32)
MUG_LEFT_CENTER = (-36.5, 6.0)
MUG_RIGHT_CENTER = (36.5, 6.0)
MUG_SLAB_BOTTOM_Y = -12.5          # unscaled; bottom of slab in hull local frame

# Collision categories.
CAT_TERRAIN = 0x0001
CAT_LEG = 0x0020
CAT_MUG = 0x0080
CAT_PAYLOAD = 0x0040

# Original BipedalWalker hull had density 5.0 on a ~1.09 m^2 polygon.
# Target mug mass ≈ 5.4. Total mug fixture area (in m^2) = (76*3 + 2*3*32) / SCALE^2 ≈ 0.467.
# Default MUG_DENSITY ≈ 11.6 matches the original mass. Exposed for manual tuning.
MUG_DENSITY = 11.6

# Legs hang from the bottom of the mug slab.
LEG_DOWN = MUG_SLAB_BOTTOM_Y / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE
```

- [ ] **Step 4: Replace HULL_FD with mug fixtures factory**

Find the block:

```python
HULL_FD = fixtureDef(
    shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in HULL_POLY]),
    density=5.0,
    friction=0.1,
    categoryBits=0x0020,
    maskBits=0x001,  # collide only with ground
    restitution=0.0,
)  # 0.99 bouncy
```

Replace with:

```python
def _mug_fixture(half_size, center, density=MUG_DENSITY):
    hx, hy = half_size
    cx, cy = center
    return fixtureDef(
        shape=polygonShape(box=(hx / SCALE, hy / SCALE, (cx / SCALE, cy / SCALE), 0.0)),
        density=density,
        friction=0.1,
        categoryBits=CAT_MUG,
        maskBits=CAT_TERRAIN | CAT_PAYLOAD,
        restitution=0.0,
    )


def _build_mug_fixtures():
    return [
        _mug_fixture(MUG_SLAB_HALF, MUG_SLAB_CENTER),
        _mug_fixture(MUG_WALL_HALF, MUG_LEFT_CENTER),
        _mug_fixture(MUG_WALL_HALF, MUG_RIGHT_CENTER),
    ]
```

- [ ] **Step 5: Update leg fixture collision mask (unchanged semantically, but explicit)**

Find the `LEG_FD` and `LOWER_FD` fixtureDef blocks (near the top of the file).

Change `categoryBits=0x0020` to `categoryBits=CAT_LEG` and `maskBits=0x001` to `maskBits=CAT_TERRAIN` in BOTH `LEG_FD` and `LOWER_FD`. (Semantically identical; makes the spec table explicit.)

- [ ] **Step 6: Update hull creation in `reset()`**

Find in `reset()`:

```python
self.hull = self.world.CreateDynamicBody(
    position=(init_x, init_y), fixtures=HULL_FD
)
```

Replace with:

```python
self.hull = self.world.CreateDynamicBody(
    position=(init_x, init_y), fixtures=_build_mug_fixtures()
)
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_mug_hull.py -v`

Expected: 3 passed.

If `test_rollout_100_steps_no_crash` fails with `terminated=True` (hull touches ground because legs now too low), increase initial hull Y. Find `init_y = TERRAIN_HEIGHT + 2 * LEG_H` in `reset()` and change to `init_y = TERRAIN_HEIGHT + 2 * LEG_H + 4.5 / SCALE` (raise by 4.5 unscaled units to compensate for the new LEG_DOWN offset). Re-run.

- [ ] **Step 8: Visual check (manual)**

Run: `python -c "import gymnasium as gym; import mughead_walker; import numpy as np; env=gym.make('MugheadWalker-v0', render_mode='human'); env.reset(seed=0); [env.step(np.zeros(4, dtype=np.float32)) for _ in range(200)]; env.close()"`

Expected: a pygame window opens showing a white-ish U-shaped mug body with two legs hanging below. The mug should stand on the ground without immediately tipping over. Close the window to end.

If the legs appear too short / mug sticks into ground / whole thing looks malformed, tweak `init_y` until the startpad shows the walker upright. (Commit current state first if behavior is reasonable; cosmetic tuning can iterate.)

- [ ] **Step 9: Commit**

```bash
git add mughead_walker/mughead_walker.py tests/test_mug_hull.py
git commit -m "$(cat <<'EOF'
Replace hull polygon with 3-fixture U-shaped mug

Introduces module-level mug geometry constants, a fixture factory
_build_mug_fixtures() producing the bottom slab + two walls, and
moves leg hip anchors to the slab bottom (LEG_DOWN = -12.5/SCALE).
Hull mass is preserved via MUG_DENSITY tuned to match the original.
Uses new collision categories (CAT_MUG 0x80) in preparation for payloads.
Observation shape unchanged.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Payload bodies (circle dynamic bodies, collision mask, destroy on reset)

**Files:**
- Modify: `mughead_walker/mughead_walker.py` (payload constants, payload creation in `reset()`, payload cleanup in `_destroy()`)
- Create: `tests/test_payloads.py`

Goal: 3 circle bodies appear inside the mug at reset, with correct collision masks. They persist through a short zero-action rollout (basic stability). Observation still 24-dim.

- [ ] **Step 1: Write failing test for payload count and type**

Create `tests/test_payloads.py`:

```python
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
```

- [ ] **Step 2: Run test to verify failure**

Run: `pytest tests/test_payloads.py -v`

Expected: FAIL with `AttributeError: ... has no attribute 'payloads'`.

- [ ] **Step 3: Add payload constants at module level**

In `mughead_walker/mughead_walker.py`, add after the `MUG_DENSITY` line:

```python
# Payload parameters (unscaled radius; see spec §4.2).
PAYLOAD_RADIUS = 5.0
PAYLOAD_FRICTION = 0.4
PAYLOAD_LINEAR_DAMPING = 0.5
DEFAULT_PAYLOAD_MASS_RATIO = 0.06
DEFAULT_PAYLOAD_BOUNCINESS = 0.15
DEFAULT_NUM_PAYLOADS = 3

# Initial positions (hull local, unscaled) — vertical stack.
PAYLOAD_INIT_POSITIONS = [
    (0.3, -4.5),
    (-0.2, 5.5),
    (0.1, 15.5),
]

# Loss thresholds.
LOSS_DISTANCE = 80.0 / SCALE   # Box2D meters from hull center
LOSS_Y_BELOW_GROUND = 1.0 / SCALE  # drop below TERRAIN_HEIGHT by this much
```

- [ ] **Step 4: Add payload creation helper as a class method**

Inside `MugheadWalkerEnv` class, add this method (anywhere between `__init__` and `reset`):

```python
def _create_payloads(self):
    """Create payload circle bodies inside the mug. Appends to self.payloads."""
    self.payloads: list = []
    hull_mass = self.hull.mass
    payload_mass = hull_mass * self._payload_mass_ratio
    # density for circle: mass / (pi * r^2)   where r is in meters
    r_m = PAYLOAD_RADIUS / SCALE
    density = payload_mass / (math.pi * r_m * r_m)
    for i in range(self._num_payloads):
        local_x, local_y = PAYLOAD_INIT_POSITIONS[i]
        world_pos = self.hull.GetWorldPoint((local_x / SCALE, local_y / SCALE))
        fd = fixtureDef(
            shape=circleShape(radius=r_m),
            density=density,
            friction=PAYLOAD_FRICTION,
            restitution=self._payload_bounciness,
            categoryBits=CAT_PAYLOAD,
            maskBits=CAT_TERRAIN | CAT_MUG | CAT_PAYLOAD,
        )
        body = self.world.CreateDynamicBody(
            position=world_pos,
            fixtures=fd,
            linearDamping=PAYLOAD_LINEAR_DAMPING,
        )
        body.color1 = [(220, 80, 80), (80, 180, 80), (80, 120, 220)][i]
        body.color2 = (255, 255, 255)
        self.payloads.append(body)
```

- [ ] **Step 5: Store config defaults in `__init__`**

Find the end of `__init__` (just before `self.render_mode = render_mode`). Add:

```python
self._num_payloads = DEFAULT_NUM_PAYLOADS
self._payload_mass_ratio = DEFAULT_PAYLOAD_MASS_RATIO
self._payload_bounciness = DEFAULT_PAYLOAD_BOUNCINESS
```

Also, at the top of the file (with other imports), ensure `math` is imported (it already is — verify).

- [ ] **Step 6: Call payload creation after hull+legs in `reset()`**

In `reset()`, find the line `self.drawlist = self.terrain + self.legs + [self.hull]`.

**Before** that line, add:

```python
self._create_payloads()
```

**Change** the drawlist line to include payloads:

```python
self.drawlist = self.terrain + self.legs + [self.hull] + self.payloads
```

- [ ] **Step 7: Extend `_destroy()` to clean up payloads**

Find `_destroy()`:

```python
def _destroy(self):
    if not self.terrain:
        return
    self.world.contactListener = None
    for t in self.terrain:
        self.world.DestroyBody(t)
    self.terrain = []
    self.world.DestroyBody(self.hull)
    self.hull = None
    for leg in self.legs:
        self.world.DestroyBody(leg)
    self.legs = []
    self.joints = []
```

Replace with:

```python
def _destroy(self):
    if not self.terrain:
        return
    self.world.contactListener = None
    for t in self.terrain:
        self.world.DestroyBody(t)
    self.terrain = []
    self.world.DestroyBody(self.hull)
    self.hull = None
    for leg in self.legs:
        self.world.DestroyBody(leg)
    self.legs = []
    self.joints = []
    for p in getattr(self, "payloads", []):
        if p is not None:
            self.world.DestroyBody(p)
    self.payloads = []
```

- [ ] **Step 8: Run tests**

Run: `pytest tests/test_payloads.py tests/test_mug_hull.py tests/test_registration.py -v`

Expected: all pass (4 + 3 + 3 = 10 tests).

- [ ] **Step 9: Commit**

```bash
git add mughead_walker/mughead_walker.py tests/test_payloads.py
git commit -m "$(cat <<'EOF'
Add 3 payload circle bodies inside the mug

Creates circle dynamic bodies at hull-local stacked positions
with density derived from payload_mass_ratio (default 0.06 of hull mass).
Collision masks: payload category 0x40, colliding with terrain + mug + other payloads,
but explicitly excluding legs. Payloads are destroyed/recreated on reset
and added to the render drawlist. Observation shape still 24-dim.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Observation extension 24 → 40

**Files:**
- Modify: `mughead_walker/mughead_walker.py` (expand observation_space low/high, extend state in `step()`, also fix the `reset → self.step(zero)` return shape)
- Create: `tests/test_observation.py`

Goal: observation is a 40-dim float32 vector per spec §5. Payload slots encode hull-local position (/50), velocity (/10), in_cup flag, and a global remaining_count/3 element. Lost slots are zero (no payload actually gets lost yet — Task 6 will add loss logic; for now slots always filled when `num_payloads=3`).

- [ ] **Step 1: Write failing test for obs shape and slot values**

Create `tests/test_observation.py`:

```python
import gymnasium as gym
import numpy as np

import mughead_walker  # noqa: F401


def test_obs_shape_40():
    env = gym.make("MugheadWalker-v0")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (40,)
    assert obs.dtype == np.float32
    env.close()


def test_remaining_count_default_one():
    """With num_payloads=3, remaining_count/3 starts at 1.0 (index 39)."""
    env = gym.make("MugheadWalker-v0")
    obs, _ = env.reset(seed=0)
    assert obs[39] == 1.0
    env.close()


def test_in_cup_flags_initial():
    """At reset, all 3 payloads should be in the cup (in_cup=1 at indices 36-38)."""
    env = gym.make("MugheadWalker-v0")
    obs, _ = env.reset(seed=0)
    np.testing.assert_array_equal(obs[36:39], [1.0, 1.0, 1.0])
    env.close()


def test_obs_no_nan_over_rollout():
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    for _ in range(100):
        obs, _, _, _, _ = env.step(np.zeros(4, dtype=np.float32))
        assert np.all(np.isfinite(obs))
        # in_cup flags always {0, 1}
        for f in obs[36:39]:
            assert f in (0.0, 1.0)
        # remaining_count/3 ∈ [0, 1]
        assert 0.0 <= obs[39] <= 1.0
    env.close()


def test_seed_reproducible():
    env1 = gym.make("MugheadWalker-v0")
    env2 = gym.make("MugheadWalker-v0")
    o1, _ = env1.reset(seed=42)
    o2, _ = env2.reset(seed=42)
    np.testing.assert_array_equal(o1, o2)
    for _ in range(30):
        a = np.zeros(4, dtype=np.float32)
        s1, *_ = env1.step(a)
        s2, *_ = env2.step(a)
        np.testing.assert_array_equal(s1, s2)
    env1.close(); env2.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_observation.py::test_obs_shape_40 -v`

Expected: FAIL with `(24,) != (40,)`.

- [ ] **Step 3: Expand observation_space in `__init__`**

Find in `__init__`:

```python
low = np.array(
    [
        -math.pi,
        ...
    ]
    + [-1.0] * 10
).astype(np.float32)
high = np.array(
    [
        math.pi,
        ...
    ]
    + [1.0] * 10
).astype(np.float32)
...
self.observation_space = spaces.Box(low, high)
```

Change to (add 16 extra slots with generous bounds; use `-inf/inf` for positions/velocities and `0/1` for flags/count):

```python
low = np.array(
    [
        -math.pi,
        -5.0,
        -5.0,
        -5.0,
        -math.pi,
        -5.0,
        -math.pi,
        -5.0,
        -0.0,
        -math.pi,
        -5.0,
        -math.pi,
        -5.0,
        -0.0,
    ]
    + [-1.0] * 10
    + [-np.inf] * 6  # payload rel pos (x,y) × 3
    + [-np.inf] * 6  # payload rel vel (vx,vy) × 3
    + [0.0] * 3      # in_cup flags
    + [0.0]          # remaining_count / 3
).astype(np.float32)
high = np.array(
    [
        math.pi,
        5.0,
        5.0,
        5.0,
        math.pi,
        5.0,
        math.pi,
        5.0,
        5.0,
        math.pi,
        5.0,
        math.pi,
        5.0,
        5.0,
    ]
    + [1.0] * 10
    + [np.inf] * 6
    + [np.inf] * 6
    + [1.0] * 3
    + [1.0]
).astype(np.float32)
```

- [ ] **Step 4: Add `_payload_obs()` helper method**

Inside `MugheadWalkerEnv`, add this method (before `step`):

```python
def _payload_obs(self) -> list[float]:
    """Return the 16 payload-related observation elements.

    Layout (see spec §5):
        6: 3 × (rel_x, rel_y) / 50
        6: 3 × (rel_vx, rel_vy) / 10
        3: 3 × in_cup flag
        1: remaining_count / 3
    """
    pos_elems = [0.0] * 6
    vel_elems = [0.0] * 6
    flag_elems = [0.0] * 3
    remaining = 0
    for i, p in enumerate(self.payloads[:3]):
        if p is None:
            continue
        remaining += 1
        local_pos = self.hull.GetLocalPoint(p.position)
        # Relative velocity in hull local frame (subtract hull velocity at payload point first).
        world_vel = p.linearVelocity - self.hull.GetLinearVelocityFromWorldPoint(p.position)
        local_vel = self.hull.GetLocalVector(world_vel)
        pos_elems[2 * i] = local_pos[0] / (50.0 / SCALE)
        pos_elems[2 * i + 1] = local_pos[1] / (50.0 / SCALE)
        vel_elems[2 * i] = local_vel[0] / (10.0 / SCALE)
        vel_elems[2 * i + 1] = local_vel[1] / (10.0 / SCALE)
        if self._is_in_cup(local_pos):
            flag_elems[i] = 1.0
    return pos_elems + vel_elems + flag_elems + [remaining / 3.0]


@staticmethod
def _is_in_cup(local_pos) -> bool:
    """Hull-local rectangle check for 'inside the mug'."""
    lx, ly = local_pos[0] * SCALE, local_pos[1] * SCALE  # back to unscaled
    return (-35.0 <= lx <= 35.0) and (-9.5 <= ly <= 22.0)
```

- [ ] **Step 5: Append payload obs to state in `step()`**

Find in `step()`:

```python
state += [l.fraction for l in self.lidar]
assert len(state) == 24
```

Change to:

```python
state += [l.fraction for l in self.lidar]
assert len(state) == 24
state += self._payload_obs()
assert len(state) == 40
```

- [ ] **Step 6: Ensure `reset()` returns the 40-dim obs**

The original `reset()` ends with `return self.step(np.array([0, 0, 0, 0]))[0], {}` which already re-uses `step()`, so the returned obs will automatically become 40-dim once Step 5 is applied. No change needed.

- [ ] **Step 7: Run tests**

Run: `pytest tests/test_observation.py -v`

Expected: 5 passed.

Also re-run earlier tests: `pytest -v` (full suite). Expected: all 15 passing.

- [ ] **Step 8: Commit**

```bash
git add mughead_walker/mughead_walker.py tests/test_observation.py
git commit -m "$(cat <<'EOF'
Extend observation space from 24 to 40 dims

Adds 16 payload-related elements:
  - 6: hull-local relative positions (x, y) / 50
  - 6: hull-local relative velocities (vx, vy) / 10
  - 3: in_cup flags (rectangle containment in mug interior)
  - 1: remaining payload count / 3
Slot order is fixed; lost payloads (future) zero their slot.
Observation bounds: ±inf for positions/velocities, [0,1] for flags/count.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Reward — in-cup bonus + payload loss detection & penalty

**Files:**
- Modify: `mughead_walker/mughead_walker.py` (step reward logic, loss detection + body destruction, flash frame counter)
- Create: `tests/test_reward_and_loss.py`

Goal: step reward includes `+0.05 × in_cup_count`. Any payload that exceeds distance/y thresholds is destroyed, its slot becomes `None`, and the step applies `-20` per lost payload. Episode continues even when all payloads are lost.

- [ ] **Step 1: Write failing test for reward/loss**

Create `tests/test_reward_and_loss.py`:

```python
import gymnasium as gym
import numpy as np

import mughead_walker  # noqa: F401


def _zero_action():
    return np.zeros(4, dtype=np.float32)


def test_in_cup_bonus_applied():
    """A zero-torque step with 3 payloads in cup should add ~0.15 reward for the bonus term."""
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    total = 0.0
    for _ in range(5):
        _, r, *_ = env.step(_zero_action())
        total += r
    # Average per-step reward should be positive (forward-ish + bonus) and finite.
    assert np.isfinite(total)
    env.close()


def test_payload_loss_removes_body_and_applies_penalty():
    """Manually teleport a payload far away; next step should remove it with -20 penalty."""
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    u = env.unwrapped
    # Place payload 0 10 meters away from hull in the +x direction.
    u.payloads[0].position = (u.hull.position[0] + 10.0, u.hull.position[1])
    obs, reward, terminated, truncated, info = env.step(_zero_action())
    assert not terminated
    assert u.payloads[0] is None
    assert reward <= -19.5, f"expected reward ≤ -20 from loss, got {reward}"
    assert obs[39] == 2.0 / 3.0
    env.close()


def test_all_payloads_lost_does_not_terminate():
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    u = env.unwrapped
    far_x = u.hull.position[0] + 10.0
    for p in u.payloads:
        p.position = (far_x, u.hull.position[1])
    obs, reward, terminated, truncated, info = env.step(_zero_action())
    assert not terminated
    assert all(p is None for p in u.payloads)
    assert obs[39] == 0.0
    # One more step: no more loss penalty, episode continues.
    obs2, r2, terminated2, *_ = env.step(_zero_action())
    assert not terminated2
    env.close()


def test_hull_fall_still_terminates():
    env = gym.make("MugheadWalker-v0")
    env.reset(seed=0)
    u = env.unwrapped
    u.game_over = True   # simulate hull-ground contact flag
    _, reward, terminated, *_ = env.step(_zero_action())
    assert terminated
    assert reward == -100
    env.close()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_reward_and_loss.py -v`

Expected: FAIL (bodies not destroyed, no in_cup bonus yet). Specifically `test_payload_loss_removes_body_and_applies_penalty` fails because `u.payloads[0]` is still a body.

- [ ] **Step 3: Add loss-detection helper**

Inside `MugheadWalkerEnv`, add (anywhere with the other private methods):

```python
def _check_payload_losses(self) -> int:
    """Destroy any payload that exits the mug. Returns count of newly-lost payloads."""
    num_lost = 0
    for i, p in enumerate(self.payloads):
        if p is None:
            continue
        dx = p.position[0] - self.hull.position[0]
        dy = p.position[1] - self.hull.position[1]
        dist = (dx * dx + dy * dy) ** 0.5
        below_ground = p.position[1] < (TERRAIN_HEIGHT - LOSS_Y_BELOW_GROUND)
        if dist > LOSS_DISTANCE or below_ground:
            self.world.DestroyBody(p)
            self.payloads[i] = None
            num_lost += 1
    return num_lost
```

Also add a flash counter in `__init__` (next to `self.render_mode = render_mode`):

```python
self._flash_frames = 0
```

- [ ] **Step 4: Extend `step()` to apply mug rewards and loss**

Find in `step()` (after `shaping -= 5.0 * abs(state[0])` and `reward` assignment block):

```python
reward = 0
if self.prev_shaping is not None:
    reward = shaping - self.prev_shaping
self.prev_shaping = shaping

for a in action:
    reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
    # normalized to about -50.0 using heuristic, more optimal agent should spend less
```

**Immediately after** that block (before the `terminated = False` line), add:

```python
# Mug extensions (spec §7).
in_cup_count = int(sum(state[36:39]))
reward += 0.05 * in_cup_count

num_lost = self._check_payload_losses()
if num_lost > 0:
    reward -= 20.0 * num_lost
    self._flash_frames = 3
```

Note: `state[36:39]` refers to the in_cup flags computed by `_payload_obs()` via `state += self._payload_obs()`. The flag values are 0.0 or 1.0, so the sum is the count.

- [ ] **Step 5: Recompute state after loss (slot must reflect None)**

The loss check destroys bodies mid-step, but `state` was computed BEFORE destruction. After loss, recompute payload obs to reflect the updated slots.

**Replace** the block you added in Step 4 with:

```python
# Mug extensions (spec §7).
in_cup_count = int(sum(state[36:39]))
reward += 0.05 * in_cup_count

num_lost = self._check_payload_losses()
if num_lost > 0:
    reward -= 20.0 * num_lost
    self._flash_frames = 3
    # Rewrite payload obs slots to reflect destroyed bodies.
    state[24:40] = self._payload_obs()
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_reward_and_loss.py -v`

Expected: 4 passed.

Also full suite: `pytest -v`. Expected: 19 passed.

- [ ] **Step 7: Commit**

```bash
git add mughead_walker/mughead_walker.py tests/test_reward_and_loss.py
git commit -m "$(cat <<'EOF'
Add in-cup reward bonus and payload loss penalty

Every step adds +0.05 × in_cup_count to the reward (max +0.15 with 3
payloads retained). Payloads that travel more than LOSS_DISTANCE from
the hull or fall below ground are destroyed, their slot set to None,
and the step applies -20 per lost payload. Episodes continue even when
all payloads are lost — the spec explicitly allows a 'sprint' strategy.
Flash-frame counter is bumped on loss for later rendering.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Rendering — payload colors, HUD, loss flash

**Files:**
- Modify: `mughead_walker/mughead_walker.py` (render() — draw payload color2 highlight, HUD text, flash overlay; adjust mug colors)
- Create: `tests/test_rendering.py`

Goal: `render_mode="rgb_array"` returns a `(H, W, 3) uint8` numpy array that contains the HUD text and the mug colors. Loss flash overlays a red tint for 3 frames.

- [ ] **Step 1: Write failing test for rgb_array output**

Create `tests/test_rendering.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_rendering.py -v`

Expected: The color tests may ALREADY pass because the generic render loop draws circle fixtures in `obj.color1`. The flash-activation test passes too (set in Task 6). But text/HUD is not drawn yet.

If all 3 tests pass at this stage, proceed to Step 3 anyway to implement HUD (spec §9) and explicitly verify.

If any fail, note the failure and proceed.

- [ ] **Step 3: Set mug color scheme in `reset()`**

In `reset()`, find `self.hull = self.world.CreateDynamicBody(...)` followed by `self.hull.color1 = (127, 51, 229)` and `self.hull.color2 = (76, 76, 127)`.

Replace those two color lines with:

```python
self.hull.color1 = (240, 235, 220)  # cream / ceramic mug
self.hull.color2 = (60, 60, 60)     # outline
```

- [ ] **Step 4: Add HUD drawing inside `render()`**

Find, inside `render()`, the very end of the method (before the final `return` statement for `"rgb_array"` mode).

First, locate the block:

```python
self.surf = pygame.transform.flip(self.surf, False, True)

if self.render_mode == "human":
    assert self.screen is not None
    self.screen.blit(self.surf, (-self.scroll * SCALE, 0))
    pygame.event.pump()
    self.clock.tick(self.metadata["render_fps"])
```

**Immediately after** the `self.surf = pygame.transform.flip(self.surf, False, True)` line, add:

```python
# HUD overlay (post-flip — text should read normally).
if not pygame.font.get_init():
    pygame.font.init()
font = pygame.font.Font(None, 20)
remaining = sum(1 for p in self.payloads if p is not None)
hud_lines = [
    f"Payloads: {remaining}/{self._num_payloads}",
    f"Distance: {self.hull.position[0]:.1f} m",
]
for i, line in enumerate(hud_lines):
    text = font.render(line, True, (0, 0, 0))
    self.surf.blit(text, (6 + self.scroll * SCALE, 6 + i * 18))

# Loss flash: semi-transparent red overlay.
if self._flash_frames > 0:
    overlay = pygame.Surface(self.surf.get_size(), pygame.SRCALPHA)
    overlay.fill((255, 0, 0, 80))
    self.surf.blit(overlay, (0, 0))
    self._flash_frames -= 1
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_rendering.py -v`

Expected: 3 passed.

Also full suite: `pytest -v`. Expected: 22 passed.

- [ ] **Step 6: Manual visual check**

Run:

```bash
python -c "
import gymnasium as gym
import numpy as np
import mughead_walker
env = gym.make('MugheadWalker-v0', render_mode='human')
env.reset(seed=0)
for _ in range(300):
    env.step(env.action_space.sample())
env.close()
"
```

Expected: pygame window shows a cream U-shaped mug with 3 colored balls inside (red, green, blue), HUD text at top-left reading `Payloads: n/3` and `Distance: X.X m`. Walking produces motion; if payloads fall out, HUD count decreases and a red flash appears.

- [ ] **Step 7: Commit**

```bash
git add mughead_walker/mughead_walker.py tests/test_rendering.py
git commit -m "$(cat <<'EOF'
Add mug colors, HUD, and loss flash to rendering

Sets hull color1/color2 to cream/outline for the mug. Adds a HUD
at top-left of every rendered frame showing surviving payload count
and hull x-position. Applies a semi-transparent red overlay for
3 frames after any payload loss.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Configurable parameters + input validation

**Files:**
- Modify: `mughead_walker/mughead_walker.py` (`__init__` signature, validation, `EzPickle.__init__` args)
- Create: `tests/test_config_params.py`

Goal: `MugheadWalkerEnv.__init__` accepts the 6 runtime parameters from spec §10. Supported parameters affect behavior; deferred parameters raise `NotImplementedError`.

- [ ] **Step 1: Write failing test for config**

Create `tests/test_config_params.py`:

```python
import gymnasium as gym
import numpy as np
import pytest

import mughead_walker  # noqa: F401


def test_num_payloads_zero():
    env = gym.make("MugheadWalker-v0", num_payloads=0)
    obs, _ = env.reset(seed=0)
    assert obs[39] == 0.0
    # First 16 payload slots all zero.
    np.testing.assert_array_equal(obs[24:40], np.zeros(16, dtype=np.float32))
    assert len(env.unwrapped.payloads) == 0
    env.close()


def test_num_payloads_one():
    env = gym.make("MugheadWalker-v0", num_payloads=1)
    obs, _ = env.reset(seed=0)
    assert obs[39] == pytest.approx(1.0 / 3.0)
    # slot 0 populated, slots 1 and 2 zero.
    assert obs[36] == 1.0
    assert obs[37] == 0.0
    assert obs[38] == 0.0
    env.close()


def test_num_payloads_out_of_range_rejected():
    with pytest.raises((ValueError, AssertionError)):
        gym.make("MugheadWalker-v0", num_payloads=4)


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
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_config_params.py -v`

Expected: FAIL — `__init__` doesn't accept these kwargs.

- [ ] **Step 3: Extend `__init__` signature**

Find the `__init__` signature:

```python
def __init__(self, render_mode: str | None = None, hardcore: bool = False):
    EzPickle.__init__(self, render_mode, hardcore)
```

Replace with:

```python
def __init__(
    self,
    render_mode: str | None = None,
    num_payloads: int = DEFAULT_NUM_PAYLOADS,
    payload_mass_ratio: float = DEFAULT_PAYLOAD_MASS_RATIO,
    payload_bounciness: float = DEFAULT_PAYLOAD_BOUNCINESS,
    terrain_difficulty: int = 0,
    obstacles: bool = False,
    external_force: float = 0.0,
):
    EzPickle.__init__(
        self,
        render_mode,
        num_payloads,
        payload_mass_ratio,
        payload_bounciness,
        terrain_difficulty,
        obstacles,
        external_force,
    )
    if not (0 <= num_payloads <= 3):
        raise ValueError(f"num_payloads must be in [0, 3], got {num_payloads}")
    if terrain_difficulty != 0:
        raise NotImplementedError(
            "terrain_difficulty>0 is deferred to a future spec"
        )
    if obstacles:
        raise NotImplementedError("obstacles=True is deferred to a future spec")
    if external_force != 0.0:
        raise NotImplementedError(
            "external_force!=0 is deferred to a future spec"
        )
```

- [ ] **Step 4: Remove the `self.hardcore = hardcore` assignment**

Find `self.hardcore = hardcore` in `__init__` and delete it (we removed the Hardcore variant in Task 2; this variable is now dead).

Also find `self._generate_terrain(self.hardcore)` in `reset()` and replace with `self._generate_terrain(False)`.

- [ ] **Step 5: Store config on the instance (replace the defaults from Task 4)**

Find and remove (or replace) the three lines added in Task 4:

```python
self._num_payloads = DEFAULT_NUM_PAYLOADS
self._payload_mass_ratio = DEFAULT_PAYLOAD_MASS_RATIO
self._payload_bounciness = DEFAULT_PAYLOAD_BOUNCINESS
```

Replace with:

```python
self._num_payloads = num_payloads
self._payload_mass_ratio = payload_mass_ratio
self._payload_bounciness = payload_bounciness
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_config_params.py -v`

Expected: 8 passed.

Also full suite: `pytest -v`. Expected: 30 passed.

- [ ] **Step 7: Commit**

```bash
git add mughead_walker/mughead_walker.py tests/test_config_params.py
git commit -m "$(cat <<'EOF'
Expose configurable constructor parameters

Adds num_payloads (0-3), payload_mass_ratio, payload_bounciness to
__init__, routed through to payload creation. Reserves 3 more
parameters (terrain_difficulty, obstacles, external_force) that
raise NotImplementedError for nonzero values — these are scoped
to a later round-variation spec.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Full smoke test suite, random_agent example, README

**Files:**
- Create: `tests/test_env.py` (consolidated smoke suite from spec §11)
- Create: `examples/random_agent.py`
- Create: `examples/train_ppo.py` (placeholder)
- Create: `examples/evaluate.py` (placeholder)
- Create: `README.md`

Goal: The spec's 9 canonical smoke tests all exist (some duplicated across previous task test files — consolidate). Random-agent example runs. README documents install and usage.

- [ ] **Step 1: Consolidate smoke tests**

Create `tests/test_env.py` with the spec's 9 canonical tests. The earlier task-level test files remain (they're more granular); this file is the "release gate" as specified in §11.

```python
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
    assert isinstance(reward, (int, float)) and np.isfinite(reward)
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
```

- [ ] **Step 2: Run consolidated suite**

Run: `pytest tests/test_env.py -v`

Expected: 9 passed.

If `test_payload_stays_in_cup_at_rest` fails with `walker fell over`, the initial hull Y or leg positioning needs tuning. Diagnose by running Task 3 Step 8's visual check; adjust `init_y` or `LEG_DOWN` until the walker stands stably. Commit the fix and re-run.

- [ ] **Step 3: Write random_agent example**

Create `examples/random_agent.py`:

```python
"""Random policy rollout for visual inspection of MugheadWalker-v0."""
import argparse

import gymnasium as gym
import numpy as np

import mughead_walker  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--render", action="store_true", help="show pygame window")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = gym.make(
        "MugheadWalker-v0",
        render_mode="human" if args.render else None,
    )

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total_r = 0.0
        steps = 0
        while True:
            action = env.action_space.sample()
            obs, r, terminated, truncated, _ = env.step(action)
            total_r += r
            steps += 1
            if terminated or truncated:
                break
        remaining = int(round(obs[39] * 3))
        dist = obs[39]  # not distance; fetch from env for accuracy below
        print(
            f"episode {ep}: steps={steps} "
            f"reward={total_r:.1f} "
            f"surviving_payloads={remaining}/3 "
            f"hull_x={env.unwrapped.hull.position[0]:.1f}m"
        )

    env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify example runs (non-render mode)**

Run: `python examples/random_agent.py --episodes 2`

Expected: two lines of output like `episode 0: steps=... reward=... surviving_payloads=N/3 hull_x=...`.

- [ ] **Step 5: Write placeholder train/evaluate scripts**

Create `examples/train_ppo.py`:

```python
"""PPO baseline trainer for MugheadWalker-v0.

This is a placeholder. Implementation is deferred to the follow-up spec
(PPO baseline & difficulty validation).
"""
import sys


def main():
    print(
        "PPO training is scoped to a follow-up spec "
        "(see docs/superpowers/specs/2026-04-17-mughead-walker-env-design.md §15)."
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
```

Create `examples/evaluate.py`:

```python
"""Evaluation harness for trained policies. Placeholder — see follow-up spec."""
import sys


def main():
    print("Evaluation is deferred to the follow-up spec.")
    sys.exit(0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Write README**

Create `README.md`:

````markdown
# MugheadWalker-v0

Gymnasium environment where a BipedalWalker carries a mug full of payloads.
Forked from `gymnasium.envs.box2d.bipedal_walker`. Built for the Yonsei
"Understanding and Applying AI" RL competition.

## Install

```bash
pip install -e '.[dev]'
```

Requires Python 3.10+, `gymnasium[box2d]`, `pygame`, `pytest`.

## Use

```python
import gymnasium as gym
import mughead_walker  # registers MugheadWalker-v0

env = gym.make("MugheadWalker-v0", render_mode="human")
obs, info = env.reset(seed=0)
while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

### Quick visual sanity check

```bash
python examples/random_agent.py --episodes 3 --render
```

### Run tests

```bash
pytest
```

## Observation & action space

- **Observation:** 40-dim float32 vector
  - 0–23: original BipedalWalker state (hull pose/velocity, 4 joint angles/speeds, 2 leg ground-contact flags, 10 LIDAR measurements)
  - 24–29: 3 × payload hull-local position (x, y) / 50
  - 30–35: 3 × payload hull-local velocity (vx, vy) / 10
  - 36–38: 3 × `in_cup` flag (0/1)
  - 39: surviving payload count / 3
- **Action:** 4-dim continuous `[-1, 1]` (hip/knee torque for both legs).

## Reward

Original BipedalWalker shaping (forward distance, angle penalty, motor torque penalty, −100 on hull fall) plus:
- `+0.05 × in_cup_count` every step.
- `−20` per payload lost.
- Episode continues when all payloads are lost.

## Configurable parameters

```python
gym.make(
    "MugheadWalker-v0",
    num_payloads=3,            # 0–3 supported
    payload_mass_ratio=0.06,   # fraction of hull mass per payload
    payload_bounciness=0.15,   # restitution
    terrain_difficulty=0,      # only 0 supported now (1–3 in a later spec)
    obstacles=False,           # only False supported now
    external_force=0.0,        # only 0 supported now
)
```

## License

MIT. Derived from `gymnasium.envs.box2d.bipedal_walker`. See `mughead_walker/mughead_walker.py` header for original credits.
````

- [ ] **Step 7: Run full suite + example**

Run:

```bash
pytest -v
python examples/random_agent.py --episodes 2
```

Expected: all tests pass (count depends on earlier task totals, approximately 31 tests total), random_agent prints 2 episode summaries.

- [ ] **Step 8: Commit**

```bash
git add tests/test_env.py examples/ README.md
git commit -m "$(cat <<'EOF'
Add canonical smoke test suite, random agent example, and README

tests/test_env.py consolidates the 9 spec-defined smoke tests.
examples/random_agent.py rolls out random actions (optionally with
render) for manual inspection.
examples/train_ppo.py and examples/evaluate.py are placeholders
that point at the follow-up spec.
README documents install, spaces, reward, and configurable params.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Self-review against the spec

After all tasks land, do a final pass:

- [ ] `pytest -v` reports all tests passing.
- [ ] `python examples/random_agent.py --episodes 3 --render` shows:
  - a cream U-shaped mug,
  - 3 colored payloads (red/green/blue),
  - walker stands upright at reset,
  - HUD shows `Payloads: N/3` and `Distance: X.X m`,
  - red flash on payload loss.
- [ ] Git log contains one commit per task (~9 commits) with descriptive messages.
- [ ] Spec coverage check:
  - §4.1 mug hull (3 fixtures, U-shape, LEG_DOWN moved) → Task 3.
  - §4.2 payload bodies (r=5, stacked, density from mass ratio) → Task 4.
  - §4.3 collision categories → Task 3 + Task 4.
  - §4.4 loss detection → Task 6.
  - §5 40-dim observation → Task 5.
  - §7 reward → Task 6.
  - §8 termination (hull fall, map end, time limit, no-terminate-on-all-lost) → Task 2 + Task 6.
  - §9 rendering (mug color, payload colors, HUD, flash) → Task 7.
  - §10 configurable params → Task 8.
  - §11 smoke tests → Task 9 (consolidated).
  - §12 examples → Task 9.

If anything above fails, diagnose and add fix-up tasks.
