# Waist Joint + Narrower Mug — Design

**Date:** 2026-04-18
**Status:** Approved, pending implementation
**Predecessor:** [2026-04-17-mughead-walker-env-design.md](2026-04-17-mughead-walker-env-design.md)
**Motivation:** PPO baseline (1M steps) showed mean payload survival **3.0/3** — payload preservation is too easy. The agent has no incentive to manage cargo actively because a wide, low-tilt cup holds everything. This spec makes cargo management a real strategic axis.

---

## 1. Summary of changes

Split the single `hull` body into two bodies connected by a motorized revolute joint:

- `chassis` — the lower "pelvis" that the legs attach to; carries LIDAR, handles distance/fall detection.
- `mug`   — the upper U-shaped cup that holds payloads; rotates relative to the chassis via the waist joint.

Also narrow the mug interior width from 70 → 50 units so it no longer trivially contains payloads.

Target after change: PPO 1M baseline yields mean payload survival ≈ **2 / 3** (previously 3.0/3). This confirms payload management is a meaningful trade-off.

## 2. Geometry & physics

### 2.1 Constants

| Constant | Current | New | Rationale |
| --- | --- | --- | --- |
| `MUG_INNER_WIDTH` | 70 | **50** | Tighter fit; lateral tilt now matters |
| `MUG_OUTER_WIDTH` | 80 | 60 | Keeps 5-unit wall thickness |
| `WAIST_TORQUE` | — | 80 | Same as hip (`MOTORS_TORQUE`) |
| `WAIST_SPEED` | — | 4.0 | Same as hip (`SPEED_HIP`) |
| `WAIST_LIMIT` | — | π / 4 | ±45 ° relative to chassis |
| `CHASSIS_POLY` | — | 20 × 6 rectangle | Narrow pelvis just wide enough for two hips |

### 2.2 Body layout (world-up = +y)

```
                 ┌──────────────┐
                 │   mug (U)    │   ← mug, contains payloads
                 │  🟡 🟡 🟡    │
                 └──┬────────┬──┘
                    │ waist  │   ← revolute joint, motorized
                 ┌──┴────────┴──┐
                 │   chassis    │   ← narrow rectangle, legs attach
                 └───┬──────┬───┘
                     │      │
                   leg1   leg2
```

- Chassis spawned at the old hull's position (`INIT_X`, `INIT_Y`).
- Mug spawned just above the chassis (chassis top + small offset) with 0 relative rotation.
- Waist joint anchors at the midpoint between chassis top and mug bottom, centered horizontally.
- Payloads initialized inside the mug as before, but x-range clamped to the new narrower interior.

### 2.3 Collision filtering

| Body | `categoryBits` | `maskBits` (collides with) |
| --- | --- | --- |
| chassis | `CAT_CHASSIS` (new) | terrain, other walkers |
| mug | `CAT_MUG` (existing) | terrain, payloads |
| legs | `CAT_LEG` | terrain, payloads (unchanged) |
| payloads | `CAT_PAYLOAD` | terrain, mug, legs, other payloads |

`collideConnected=False` on the waist joint (chassis and mug must not self-collide even if their polygons overlap).

### 2.4 Ground-contact → game_over

`ContactDetector.BeginContact` triggers `game_over=True` if terrain contacts **either** the chassis **or** the mug. Legs and payloads do not trigger game_over (unchanged).

## 3. Observation space (40 → 42)

| Index | Value | Notes |
| --- | --- | --- |
| 0 | chassis angle | was hull angle |
| 1 | chassis angular velocity / FPS scaled | was hull |
| 2, 3 | chassis linear velocity (x, y) scaled | was hull |
| 4 – 13 | 4 leg joints + 2 contact flags | unchanged |
| 14 – 23 | 10 LIDAR fractions | originates at chassis position |
| **24** | **waist joint angle** | new |
| **25** | **waist joint speed** | new, scaled by `SPEED_HIP` |
| 26 – 31 | 3 × payload (rel_x, rel_y) in **mug-local frame** / (50 / SCALE) | was hull-local |
| 32 – 37 | 3 × payload (rel_vx, rel_vy) in mug-local frame / (10 / SCALE) | was hull-local |
| 38 – 40 | 3 × in-cup flags | shifted +2 from old layout |
| 41 | surviving payload count / 3 | shifted +2 |

Total: **42 float32 values.**

## 4. Action space (4 → 5)

| Index | Joint |
| --- | --- |
| 0 | hip 1 |
| 1 | knee 1 |
| 2 | hip 2 |
| 3 | knee 2 |
| **4** | **waist** (new) |

Action bounds remain `[-1, 1]`. Applied as torque-control (`maxMotorTorque = WAIST_TORQUE * |a|`), matching the existing hip/knee convention.

## 5. Reward

Baseline shaping uses **chassis** position & angle (was hull):

```
shaping = 130 * chassis.x / SCALE
shaping -= 5.0 * abs(chassis.angle)
```

All other terms unchanged:

- `-0.00035 * MOTORS_TORQUE * |a|` per action dimension (now 5 dims → slightly larger torque penalty)
- `+0.05 * in_cup_count` per step
- `-20` per payload lost
- `-100` and terminate on game_over

**No explicit mug-angle penalty.** The agent must discover that tilting the mug via waist torque keeps payloads in.

## 6. Rendering

- Draw chassis polygon and mug polygon separately (both follow their body's transform).
- Colors: chassis same hull-brown as before; mug walls distinct (off-white / pastel).
- At the waist anchor, draw a small filled circle (radius ≈ 3 px) to indicate the hinge.
- Payload colors, HUD, loss-flash overlay all unchanged.

## 7. Configurable parameters

Add one new `__init__` kwarg:

- `mug_inner_width: float = 50.0` — widens/narrows the cup; useful for future round tuning (wider → easier).

Existing kwargs unchanged. `__init__` validates `10.0 <= mug_inner_width <= 120.0`.

## 8. File map

### Modified

- `mughead_walker/mughead_walker.py`
  - Constants: `MUG_INNER_WIDTH=50`, add `WAIST_*`, `CHASSIS_POLY`, `CAT_CHASSIS`
  - `_build_mug_fixtures()`: update width
  - Body creation: split into `_create_chassis()` + `_create_mug()` (both called from `reset()`); `self.hull` is retired in favor of `self.chassis` + `self.mug` (no backward alias — v0 is pre-competition)
  - `_create_waist_joint()`: new, motorized revolute
  - `_create_payloads()`: mug-frame initial positions, narrower x-range
  - `_is_in_cup(local_pos)`: mug-local rect, new bounds `(-25 … 25) × (-9.5 … 22)`
  - `_check_payload_losses()`: measure distance from mug (not chassis)
  - `_payload_obs()`: use `self.mug.GetLocalPoint`/`GetLocalVector`
  - `step()`: expand action to 5, apply waist torque, use chassis for shaping, emit `info["waist_angle"]` for logging
  - `render()`: draw chassis + mug + hinge; `drawlist` now includes both
  - `_destroy()`: destroy both bodies and the waist joint
  - `close()`: unchanged
  - `ContactDetector.BeginContact` / `EndContact`: replace `self.env.hull` check with `self.env.chassis` OR `self.env.mug` (either triggers game_over); keep the existing "ignore hull-payload contacts" logic but now for mug-payload contacts (expected contacts, not game_over)

- `mughead_walker/__init__.py`: unchanged (same id, same registration)

- `README.md`: update observation table (40 → 42 dims, insert waist rows, shift payload indices)

- Tests that hard-code observation indices:
  - `tests/test_env.py` (`assert env.observation_space.shape == (42,)`; update `obs[36:39]` → `obs[38:41]`, `obs[39]` → `obs[41]`; `assert env.action_space.shape == (5,)`; `_zero_action` → `np.zeros(5)`)
  - `tests/test_observation.py`: indices shifted, add assertions on waist slots
  - `tests/test_payloads.py`: update for mug-local frame
  - `tests/test_mug_hull.py`: update expected mug width, add chassis existence assertion

### New

- `tests/test_waist_joint.py`
  - Waist joint exists on the env (after `reset`)
  - Action space is 5-dim
  - Positive `action[4]` rotates mug clockwise relative to chassis
  - Joint respects ±π/4 limits over a long torque rollout
  - Info dict includes `waist_angle`

## 9. Validation plan

1. Unit tests pass (all previously-passing tests updated + new waist tests).
2. `examples/random_agent.py --render --episodes 3` shows chassis and mug rendered as two bodies with a visible hinge; random policy still behaves physically (walker falls, payloads mostly stay put early, etc).
3. PPO 1M re-run with n_envs=8, defaults → update `docs/baseline/ppo_baseline_report.md` with before/after table:

   | Metric | Old (single hull) | New (waist + narrow mug) | Target |
   | --- | --- | --- | --- |
   | Mean reward | 261.9 | ? | > 50 |
   | Payload survival | 3.0 / 3 | ? | **≈ 2 / 3** |
   | Success rate | 40 % | ? | — |

   If post-change payload survival ∈ [1.5, 2.5] → difficulty is calibrated per §1 goal.
   If < 1.0 → env is too hard; relax (waist torque ↑, cup width +5, bounciness ↓).
   If > 2.5 → still too easy; tighten (cup width -5, mass ratio 0.06 → 0.08).

## 10. Non-goals

- Retraining the old baseline is unnecessary — it is superseded.
- `terrain_difficulty`, `obstacles`, `external_force` still deferred to their own spec.
- No Monitor/VecNorm changes; the existing training harness works unchanged after action/obs dim bump (SB3 reads from `env.observation_space` / `env.action_space`).
