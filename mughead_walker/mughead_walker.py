# Derived from gymnasium.envs.box2d.bipedal_walker (MIT License).
# Original credit: Andrea PIERRÉ / Oleg Klimov (BipedalWalker).

__credits__ = ["Andrea PIERRÉ"]

import math
from typing import TYPE_CHECKING

import numpy as np

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle


try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e


if TYPE_CHECKING:
    import pygame

FPS = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MOTORS_TORQUE = 80
SPEED_HIP = 4
SPEED_KNEE = 6
LIDAR_RANGE = 160 / SCALE

INITIAL_RANDOM = 5

# --- Mug geometry (unscaled units; Box2D fixtures use values divided by SCALE) ---
# MUG_INNER_WIDTH is now per-instance (self.mug_inner_width); default 50 set in __init__.
# Spec §2.1: MUG_OUTER_WIDTH 80→60, wall thickness unchanged.
MUG_OUTER_WIDTH = 60        # was 80
MUG_WALL_T = 5              # wall thickness (unchanged)
MUG_HEIGHT = 35             # total height of mug body (unscaled)
MUG_SLAB_HALF_H = 1.5       # half-height of bottom slab
MUG_WALL_HALF_H = 16.0      # half-height of walls
MUG_SLAB_CENTER_Y = -11.0   # slab center in mug-local frame (unscaled)
MUG_LEFT_CENTER_X = -27.5   # default for 50-wide interior; recomputed per instance
MUG_RIGHT_CENTER_X = 27.5
MUG_WALL_CENTER_Y = 6.0     # wall center-y in mug-local frame (unscaled)

# --- Chassis (new body, spec §2.1) ---
CHASSIS_W = 20        # width in physics units (unscaled)
CHASSIS_H = 6         # height in physics units (unscaled)
CHASSIS_DENSITY = 5.0  # match old hull density

# --- Waist joint (new, spec §2.1) ---
WAIST_TORQUE = 80     # = MOTORS_TORQUE (hip torque)
WAIST_SPEED = 4.0     # = SPEED_HIP
WAIST_LIMIT = math.pi / 4  # ±45 degrees

# Original BipedalWalker hull had density 5.0 on a ~1.09 m^2 polygon.
# Target mug mass ≈ 5.4. Adjusted density for narrower mug.
MUG_DENSITY = 11.6

# --- Collision categories (spec §2.3) ---
CAT_TERRAIN = 0x0001
CAT_LEG = 0x0020
CAT_MUG = 0x0040
CAT_CHASSIS = 0x0080   # NEW
CAT_PAYLOAD = 0x0100   # shifted from 0x0040 to free up bits

# Payload parameters (unscaled radius; see spec §4.2).
PAYLOAD_RADIUS = 5.0
PAYLOAD_FRICTION = 0.4
PAYLOAD_LINEAR_DAMPING = 0.5
DEFAULT_PAYLOAD_MASS_RATIO = 0.06
DEFAULT_PAYLOAD_BOUNCINESS = 0.15
DEFAULT_NUM_PAYLOADS = 3

# Loss thresholds.
LOSS_DISTANCE = 80.0 / SCALE   # Box2D meters from mug center
LOSS_Y_BELOW_GROUND = 1.0 / SCALE  # drop below TERRAIN_HEIGHT by this much

# Legs hang from the bottom of the chassis.
# LEG_DOWN is the local anchor on chassis in world units (m).
LEG_DOWN = -(CHASSIS_H / 2) / SCALE  # bottom of chassis
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200  # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10  # how long are grass spots, in steps
TERRAIN_STARTPAD = 20  # in steps
FRICTION = 2.5


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


LEG_FD = fixtureDef(
    shape=polygonShape(box=(LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=CAT_LEG,
    maskBits=CAT_TERRAIN,
)

LOWER_FD = fixtureDef(
    shape=polygonShape(box=(0.8 * LEG_W / 2, LEG_H / 2)),
    density=1.0,
    restitution=0.0,
    categoryBits=CAT_LEG,
    maskBits=CAT_TERRAIN,
)

PAYLOAD_COLORS = [
    ((220, 80, 80), (255, 255, 255)),
    ((80, 180, 80), (255, 255, 255)),
    ((80, 120, 220), (255, 255, 255)),
]


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        bodies = {contact.fixtureA.body, contact.fixtureB.body}
        # Check if chassis or mug contacts terrain (either triggers game_over).
        if self.env.chassis in bodies or self.env.mug in bodies:
            # Mug–payload contacts are expected (payload sitting in cup); ignore them.
            other_bodies = bodies - {self.env.chassis, self.env.mug}
            for other in other_bodies:
                if other not in [p for p in self.env.payloads if p is not None]:
                    self.env.game_over = True
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = True

    def EndContact(self, contact):
        for leg in [self.env.legs[1], self.env.legs[3]]:
            if leg in [contact.fixtureA.body, contact.fixtureB.body]:
                leg.ground_contact = False


class MugheadWalkerEnv(gym.Env, EzPickle):
    """
    ## Description

    MugheadWalker is a BipedalWalker variant built for the Yonsei
    "Understanding and Applying AI" RL competition. The walker's hull
    is split into a chassis (pelvis, legs attach here) and a mug
    (U-shaped open-top cup that holds payloads), connected by a
    motorized revolute waist joint with ±π/4 limits.

    Goal: walk forward as far as possible while keeping the payloads
    in the mug.

    Forked from ``gymnasium.envs.box2d.bipedal_walker``. MIT licensed.

    ## Action Space

    5-dim continuous ``[-1, 1]``: hip/knee torque × 2 legs, plus waist torque.

    ## Observation Space

    42-dim ``float32`` vector:

    - ``obs[0:24]``: chassis pose/velocity, 4 joint angles/speeds, 2 contact
      flags, 10 LIDAR measurements.
    - ``obs[24]``: waist joint angle.
    - ``obs[25]``: waist joint speed / SPEED_HIP.
    - ``obs[26:32]``: 3 × payload mug-local position ``(x, y) / 50``.
    - ``obs[32:38]``: 3 × payload mug-local velocity ``(vx, vy) / 10``.
    - ``obs[38:41]``: 3 × ``in_cup`` flag (0.0 or 1.0).
    - ``obs[41]``: surviving payload count / 3.

    ## Rewards

    Original BipedalWalker shaping (forward progress, angle penalty,
    motor torque penalty, −100 on fall) plus:

    - ``+0.05 × in_cup_count`` every step.
    - ``−20`` per payload lost.

    ## Arguments

    ::

        gym.make(
            "MugheadWalker-v0",
            num_payloads=3,            # 0-3 supported
            payload_mass_ratio=0.06,   # chassis mass fraction per payload
            payload_bounciness=0.15,   # restitution (0-1)
            mug_inner_width=50.0,      # interior cup width (10-120)
            terrain_difficulty=0,      # only 0 supported now
            obstacles=False,           # only False supported now
            external_force=0.0,        # only 0 supported now
        )
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        num_payloads: int = DEFAULT_NUM_PAYLOADS,
        payload_mass_ratio: float = DEFAULT_PAYLOAD_MASS_RATIO,
        payload_bounciness: float = DEFAULT_PAYLOAD_BOUNCINESS,
        mug_inner_width: float = 50.0,
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
            mug_inner_width,
            terrain_difficulty,
            obstacles,
            external_force,
        )
        if not isinstance(num_payloads, int) or isinstance(num_payloads, bool):
            raise TypeError(f"num_payloads must be an int, got {type(num_payloads).__name__}")
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
        if not (0.0 <= payload_bounciness <= 1.0):
            raise ValueError(
                f"payload_bounciness must be in [0, 1], got {payload_bounciness}"
            )
        if payload_mass_ratio <= 0.0:
            raise ValueError(
                f"payload_mass_ratio must be positive, got {payload_mass_ratio}"
            )
        if not (10.0 <= mug_inner_width <= 120.0):
            raise ValueError(f"mug_inner_width must be in [10, 120], got {mug_inner_width}")

        self.mug_inner_width = mug_inner_width
        self.isopen = True

        self.world = Box2D.b2World()
        self.terrain: list[Box2D.b2Body] = []
        self.chassis: Box2D.b2Body | None = None
        self.mug: Box2D.b2Body | None = None
        self.waist_joint = None

        self.prev_shaping = None

        self.fd_polygon = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]),
            friction=FRICTION,
        )

        self.fd_edge = fixtureDef(
            shape=edgeShape(vertices=[(0, 0), (1, 1)]),
            friction=FRICTION,
            categoryBits=0x0001,
        )

        low = np.array(
            [
                -math.pi,  # 0: chassis angle
                -5.0,      # 1: chassis angular velocity
                -5.0,      # 2: chassis vel x
                -5.0,      # 3: chassis vel y
                -math.pi,  # 4: hip1 angle
                -5.0,      # 5: hip1 speed
                -math.pi,  # 6: knee1 angle (+1)
                -5.0,      # 7: knee1 speed
                -0.0,      # 8: leg1 contact
                -math.pi,  # 9: hip2 angle
                -5.0,      # 10: hip2 speed
                -math.pi,  # 11: knee2 angle (+1)
                -5.0,      # 12: knee2 speed
                -0.0,      # 13: leg2 contact
            ]
            + [-1.0] * 10   # 14-23: LIDAR
            + [-math.pi]     # 24: waist angle
            + [-5.0]         # 25: waist speed
            + [-np.inf] * 6  # 26-31: payload rel pos (x,y) × 3
            + [-np.inf] * 6  # 32-37: payload rel vel (vx,vy) × 3
            + [0.0] * 3      # 38-40: in_cup flags
            + [0.0]          # 41: remaining_count / 3
        ).astype(np.float32)
        high = np.array(
            [
                math.pi,   # 0
                5.0,       # 1
                5.0,       # 2
                5.0,       # 3
                math.pi,   # 4
                5.0,       # 5
                math.pi,   # 6
                5.0,       # 7
                5.0,       # 8
                math.pi,   # 9
                5.0,       # 10
                math.pi,   # 11
                5.0,       # 12
                5.0,       # 13
            ]
            + [1.0] * 10    # 14-23: LIDAR
            + [math.pi]      # 24: waist angle
            + [5.0]          # 25: waist speed
            + [np.inf] * 6   # 26-31: payload rel pos
            + [np.inf] * 6   # 32-37: payload rel vel
            + [1.0] * 3      # 38-40: in_cup flags
            + [1.0]          # 41: remaining_count
        ).astype(np.float32)

        self.action_space = spaces.Box(
            np.array([-1, -1, -1, -1, -1]).astype(np.float32),
            np.array([1, 1, 1, 1, 1]).astype(np.float32),
        )
        self.observation_space = spaces.Box(low, high)

        self._num_payloads = num_payloads
        self._payload_mass_ratio = payload_mass_ratio
        self._payload_bounciness = payload_bounciness

        self.render_mode = render_mode
        self._flash_frames = 0
        self.screen: pygame.Surface | None = None
        self.clock = None

    def _build_mug_fixtures(self):
        """Build mug fixtures using per-instance mug_inner_width.

        Wall thickness is fixed at MUG_WALL_T=5 unscaled, regardless of inner width.
        The slab spans the full outer width (inner + 2 walls).
        """
        half_inner = self.mug_inner_width / 2
        wall_t = MUG_WALL_T  # 5 units
        half_outer = half_inner + wall_t  # outer = inner + 2 walls
        wall_half_w = wall_t / 2
        # Wall center x in mug-local (unscaled): ± (half_inner + half_wall_t)
        wall_center_x = half_inner + wall_half_w

        slab_half = (half_outer, MUG_SLAB_HALF_H)
        slab_center = (0.0, MUG_SLAB_CENTER_Y)
        left_half = (wall_half_w, MUG_WALL_HALF_H)
        left_center = (-wall_center_x, MUG_WALL_CENTER_Y)
        right_half = (wall_half_w, MUG_WALL_HALF_H)
        right_center = (wall_center_x, MUG_WALL_CENTER_Y)

        return [
            _mug_fixture(slab_half, slab_center),
            _mug_fixture(left_half, left_center),
            _mug_fixture(right_half, right_center),
        ]

    def _create_chassis(self):
        """Narrow pelvis body; legs attach here."""
        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
        init_y = TERRAIN_HEIGHT + 2 * LEG_H
        half_w = CHASSIS_W / 2
        half_h = CHASSIS_H / 2
        poly = [
            (-half_w / SCALE, -half_h / SCALE),
            ( half_w / SCALE, -half_h / SCALE),
            ( half_w / SCALE,  half_h / SCALE),
            (-half_w / SCALE,  half_h / SCALE),
        ]
        fix = fixtureDef(
            shape=polygonShape(vertices=poly),
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
        self.chassis.color1 = (180, 130, 70)   # warm brown (like original hull)
        self.chassis.color2 = (60, 40, 20)

    def _create_mug(self):
        """U-shaped open-top cup above the chassis; carries payloads."""
        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
        # Mug center sits above chassis top: chassis_top_y + half of mug height offset
        chassis_top_y = self.chassis.position[1] + (CHASSIS_H / 2) / SCALE
        # The mug slab center is at MUG_SLAB_CENTER_Y (unscaled) = -11.0 relative to mug center.
        # We want the mug slab bottom (≈ MUG_SLAB_CENTER_Y - MUG_SLAB_HALF_H = -12.5) to sit
        # just above chassis_top_y. So mug_center_y = chassis_top_y + 12.5/SCALE + small gap.
        gap = 1.0 / SCALE  # small gap between chassis top and mug slab bottom
        mug_slab_bottom_offset = abs(MUG_SLAB_CENTER_Y + MUG_SLAB_HALF_H) / SCALE  # 12.5/SCALE
        init_y = chassis_top_y + mug_slab_bottom_offset + gap

        fixtures = self._build_mug_fixtures()
        self.mug = self.world.CreateDynamicBody(
            position=(init_x, init_y),
            fixtures=fixtures,
        )
        self.mug.color1 = (240, 235, 220)   # cream / ceramic mug
        self.mug.color2 = (60, 60, 60)       # outline

    def _create_waist_joint(self):
        """Motorized revolute joint between chassis top and mug bottom."""
        # Anchor at chassis top center (in world coords).
        anchor_x = self.chassis.position[0]
        anchor_y = self.chassis.position[1] + (CHASSIS_H / 2) / SCALE
        self.waist_joint = self.world.CreateRevoluteJoint(
            bodyA=self.chassis,
            bodyB=self.mug,
            anchor=(anchor_x, anchor_y),
            enableMotor=True,
            enableLimit=True,
            lowerAngle=-WAIST_LIMIT,
            upperAngle=+WAIST_LIMIT,
            motorSpeed=0.0,
            maxMotorTorque=WAIST_TORQUE,
            collideConnected=False,
        )

    def _create_payloads(self):
        """Create payload circle bodies inside the mug. Appends to self.payloads."""
        self.payloads: list = []
        if self._num_payloads <= 0:
            return
        # Fixed payload radius (unscaled), same as original BipedalWalker derivation.
        r_unscaled = PAYLOAD_RADIUS  # 5.0
        r_m = r_unscaled / SCALE
        chassis_mass = self.chassis.mass
        payload_mass = chassis_mass * self._payload_mass_ratio
        # density for circle: mass / (pi * r^2)
        density = payload_mass / (math.pi * r_m * r_m)

        # Initial positions in mug-local frame (unscaled).
        # Stack vertically just above the slab top (-9.5 unscaled).
        # Spacing = 2.2 * radius to avoid overlap.
        slab_top_unscaled = MUG_SLAB_CENTER_Y + MUG_SLAB_HALF_H  # -9.5
        base_y_unscaled = slab_top_unscaled + r_unscaled  # first payload center
        step_y_unscaled = r_unscaled * 2.2  # vertical spacing

        # Horizontal jitter (unscaled) — same tiny offsets as original PAYLOAD_INIT_POSITIONS.
        x_jitter = [0.3, -0.2, 0.1]

        for i in range(min(self._num_payloads, 3)):
            local_x = x_jitter[i] / SCALE
            local_y = (base_y_unscaled + i * step_y_unscaled) / SCALE
            world_pos = self.mug.GetWorldPoint((local_x, local_y))
            fd = fixtureDef(
                shape=circleShape(radius=r_m),
                density=density,
                friction=PAYLOAD_FRICTION,
                restitution=self._payload_bounciness,
                categoryBits=CAT_PAYLOAD,
                maskBits=CAT_TERRAIN | CAT_MUG | CAT_LEG | CAT_PAYLOAD,
            )
            body = self.world.CreateDynamicBody(
                position=world_pos,
                fixtures=fd,
                linearDamping=PAYLOAD_LINEAR_DAMPING,
            )
            body.color1 = PAYLOAD_COLORS[i % len(PAYLOAD_COLORS)][0]
            body.color2 = PAYLOAD_COLORS[i % len(PAYLOAD_COLORS)][1]
            self.payloads.append(body)

    def _payload_obs(self) -> list[float]:
        """Return the 16 payload-related observation elements.

        Layout (spec §3):
            6: 3 × (rel_x, rel_y) in mug-local frame / (50 / SCALE)
            6: 3 × (rel_vx, rel_vy) in mug-local frame / (10 / SCALE)
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
            local_pos = self.mug.GetLocalPoint(p.position)
            # Relative velocity in mug-local frame.
            world_vel = p.linearVelocity - self.mug.GetLinearVelocityFromWorldPoint(p.position)
            local_vel = self.mug.GetLocalVector(world_vel)
            pos_elems[2 * i] = local_pos[0] / (50.0 / SCALE)
            pos_elems[2 * i + 1] = local_pos[1] / (50.0 / SCALE)
            vel_elems[2 * i] = local_vel[0] / (10.0 / SCALE)
            vel_elems[2 * i + 1] = local_vel[1] / (10.0 / SCALE)
            if self._is_in_cup(local_pos):
                flag_elems[i] = 1.0
        return pos_elems + vel_elems + flag_elems + [remaining / 3.0]

    def _is_in_cup(self, local_pos) -> bool:
        """Mug-local rectangle check for 'inside the mug'.

        Uses self.mug_inner_width so narrow/wide cup instances work correctly.
        """
        lx, ly = local_pos[0] * SCALE, local_pos[1] * SCALE  # back to unscaled
        half_w = self.mug_inner_width / 2
        return (-half_w <= lx <= half_w) and (-9.5 <= ly <= 22.0)

    def _check_payload_losses(self) -> int:
        """Destroy any payload that exits the mug. Returns count of newly-lost payloads."""
        num_lost = 0
        for i, p in enumerate(self.payloads):
            if p is None:
                continue
            dx = p.position[0] - self.mug.position[0]
            dy = p.position[1] - self.mug.position[1]
            dist = (dx * dx + dy * dy) ** 0.5
            below_ground = p.position[1] < (TERRAIN_HEIGHT - LOSS_Y_BELOW_GROUND)
            if dist > LOSS_DISTANCE or below_ground:
                self.world.DestroyBody(p)
                self.payloads[i] = None
                num_lost += 1
        return num_lost

    def _destroy(self):
        if not self.terrain:
            return
        self.world.contactListener = None
        for t in self.terrain:
            self.world.DestroyBody(t)
        self.terrain = []
        # Destroy waist joint before its bodies.
        if self.waist_joint is not None:
            self.world.DestroyJoint(self.waist_joint)
            self.waist_joint = None
        if self.chassis is not None:
            self.world.DestroyBody(self.chassis)
            self.chassis = None
        if self.mug is not None:
            self.world.DestroyBody(self.mug)
            self.mug = None
        for leg in getattr(self, "legs", []):
            self.world.DestroyBody(leg)
        self.legs = []
        self.joints = []
        for p in getattr(self, "payloads", []):
            if p is not None:
                self.world.DestroyBody(p)
        self.payloads = []

    def _generate_terrain(self, hardcore):
        GRASS, STUMP, STAIRS, PIT, _STATES_ = range(5)
        state = GRASS
        velocity = 0.0
        y = TERRAIN_HEIGHT
        counter = TERRAIN_STARTPAD
        oneshot = False
        self.terrain = []
        self.terrain_x = []
        self.terrain_y = []

        stair_steps, stair_width, stair_height = 0, 0, 0
        original_y = 0
        for i in range(TERRAIN_LENGTH):
            x = i * TERRAIN_STEP
            self.terrain_x.append(x)

            if state == GRASS and not oneshot:
                velocity = 0.8 * velocity + 0.01 * np.sign(TERRAIN_HEIGHT - y)
                if i > TERRAIN_STARTPAD:
                    velocity += self.np_random.uniform(-1, 1) / SCALE  # 1
                y += velocity

            elif state == PIT and oneshot:
                counter = self.np_random.integers(3, 5)
                poly = [
                    (x, y),
                    (x + TERRAIN_STEP, y),
                    (x + TERRAIN_STEP, y - 4 * TERRAIN_STEP),
                    (x, y - 4 * TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)

                self.fd_polygon.shape.vertices = [
                    (p[0] + TERRAIN_STEP * counter, p[1]) for p in poly
                ]
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)
                counter += 2
                original_y = y

            elif state == PIT and not oneshot:
                y = original_y
                if counter > 1:
                    y -= 4 * TERRAIN_STEP

            elif state == STUMP and oneshot:
                counter = self.np_random.integers(1, 3)
                poly = [
                    (x, y),
                    (x + counter * TERRAIN_STEP, y),
                    (x + counter * TERRAIN_STEP, y + counter * TERRAIN_STEP),
                    (x, y + counter * TERRAIN_STEP),
                ]
                self.fd_polygon.shape.vertices = poly
                t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                self.terrain.append(t)

            elif state == STAIRS and oneshot:
                stair_height = +1 if self.np_random.random() > 0.5 else -1
                stair_width = self.np_random.integers(4, 5)
                stair_steps = self.np_random.integers(3, 5)
                original_y = y
                for s in range(stair_steps):
                    poly = [
                        (
                            x + (s * stair_width) * TERRAIN_STEP,
                            y + (s * stair_height) * TERRAIN_STEP,
                        ),
                        (
                            x + ((1 + s) * stair_width) * TERRAIN_STEP,
                            y + (s * stair_height) * TERRAIN_STEP,
                        ),
                        (
                            x + ((1 + s) * stair_width) * TERRAIN_STEP,
                            y + (-1 + s * stair_height) * TERRAIN_STEP,
                        ),
                        (
                            x + (s * stair_width) * TERRAIN_STEP,
                            y + (-1 + s * stair_height) * TERRAIN_STEP,
                        ),
                    ]
                    self.fd_polygon.shape.vertices = poly
                    t = self.world.CreateStaticBody(fixtures=self.fd_polygon)
                    t.color1, t.color2 = (255, 255, 255), (153, 153, 153)
                    self.terrain.append(t)
                counter = stair_steps * stair_width

            elif state == STAIRS and not oneshot:
                s = stair_steps * stair_width - counter - stair_height
                n = s / stair_width
                y = original_y + (n * stair_height) * TERRAIN_STEP

            oneshot = False
            self.terrain_y.append(y)
            counter -= 1
            if counter == 0:
                counter = self.np_random.integers(TERRAIN_GRASS / 2, TERRAIN_GRASS)
                if state == GRASS and hardcore:
                    state = self.np_random.integers(1, _STATES_)
                    oneshot = True
                else:
                    state = GRASS
                    oneshot = True

        self.terrain_poly = []
        for i in range(TERRAIN_LENGTH - 1):
            poly = [
                (self.terrain_x[i], self.terrain_y[i]),
                (self.terrain_x[i + 1], self.terrain_y[i + 1]),
            ]
            self.fd_edge.shape.vertices = poly
            t = self.world.CreateStaticBody(fixtures=self.fd_edge)
            color = (76, 255 if i % 2 == 0 else 204, 76)
            t.color1 = color
            t.color2 = color
            self.terrain.append(t)
            color = (102, 153, 76)
            poly += [(poly[1][0], 0), (poly[0][0], 0)]
            self.terrain_poly.append((poly, color))
        self.terrain.reverse()

    def _generate_clouds(self):
        # Sorry for the clouds, couldn't resist
        self.cloud_poly = []
        for i in range(TERRAIN_LENGTH // 20):
            x = self.np_random.uniform(0, TERRAIN_LENGTH) * TERRAIN_STEP
            y = VIEWPORT_H / SCALE * 3 / 4
            poly = [
                (
                    x
                    + 15 * TERRAIN_STEP * math.sin(3.14 * 2 * a / 5)
                    + self.np_random.uniform(0, 5 * TERRAIN_STEP),
                    y
                    + 5 * TERRAIN_STEP * math.cos(3.14 * 2 * a / 5)
                    + self.np_random.uniform(0, 5 * TERRAIN_STEP),
                )
                for a in range(5)
            ]
            x1 = min(p[0] for p in poly)
            x2 = max(p[0] for p in poly)
            self.cloud_poly.append((poly, x1, x2))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.game_over = False
        self.prev_shaping = None
        self.scroll = 0.0
        self.lidar_render = 0

        self._generate_terrain(False)
        self._generate_clouds()

        # Create chassis first (legs attach to it), then mug above it, then waist joint.
        self._create_chassis()
        self._create_mug()
        self._create_waist_joint()

        # Apply small random force to chassis for variation.
        self.chassis.ApplyForceToCenter(
            (self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True
        )

        init_x = self.chassis.position[0]
        init_y = self.chassis.position[1]

        self.legs: list[Box2D.b2Body] = []
        self.joints: list[Box2D.b2RevoluteJoint] = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LEG_FD,
            )
            leg.color1 = (153 - i * 25, 76 - i * 25, 127 - i * 25)
            leg.color2 = (102 - i * 25, 51 - i * 25, 76 - i * 25)
            rjd = revoluteJointDef(
                bodyA=self.chassis,
                bodyB=leg,
                localAnchorA=(0, LEG_DOWN),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=i,
                lowerAngle=-0.8,
                upperAngle=1.1,
            )
            self.legs.append(leg)
            self.joints.append(self.world.CreateJoint(rjd))

            lower = self.world.CreateDynamicBody(
                position=(init_x, init_y - LEG_H * 3 / 2 - LEG_DOWN),
                angle=(i * 0.05),
                fixtures=LOWER_FD,
            )
            lower.color1 = (153 - i * 25, 76 - i * 25, 127 - i * 25)
            lower.color2 = (102 - i * 25, 51 - i * 25, 76 - i * 25)
            rjd = revoluteJointDef(
                bodyA=leg,
                bodyB=lower,
                localAnchorA=(0, -LEG_H / 2),
                localAnchorB=(0, LEG_H / 2),
                enableMotor=True,
                enableLimit=True,
                maxMotorTorque=MOTORS_TORQUE,
                motorSpeed=1,
                lowerAngle=-1.6,
                upperAngle=-0.1,
            )
            lower.ground_contact = False
            self.legs.append(lower)
            self.joints.append(self.world.CreateJoint(rjd))

        self._create_payloads()

        class LidarCallback(Box2D.b2.rayCastCallback):
            def ReportFixture(self, fixture, point, normal, fraction):
                if (fixture.filterData.categoryBits & 1) == 0:
                    return -1
                self.p2 = point
                self.fraction = fraction
                return fraction

        self.lidar = [LidarCallback() for _ in range(10)]
        if self.render_mode == "human":
            self.render()
        return self.step(np.array([0, 0, 0, 0, 0]))[0], {}

    def step(self, action: np.ndarray):
        assert self.chassis is not None
        assert self.mug is not None

        control_speed = False  # Should be easier as well
        if control_speed:
            self.joints[0].motorSpeed = float(SPEED_HIP * np.clip(action[0], -1, 1))
            self.joints[1].motorSpeed = float(SPEED_KNEE * np.clip(action[1], -1, 1))
            self.joints[2].motorSpeed = float(SPEED_HIP * np.clip(action[2], -1, 1))
            self.joints[3].motorSpeed = float(SPEED_KNEE * np.clip(action[3], -1, 1))
        else:
            self.joints[0].motorSpeed = float(SPEED_HIP * np.sign(action[0]))
            self.joints[0].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[0]), 0, 1)
            )
            self.joints[1].motorSpeed = float(SPEED_KNEE * np.sign(action[1]))
            self.joints[1].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[1]), 0, 1)
            )
            self.joints[2].motorSpeed = float(SPEED_HIP * np.sign(action[2]))
            self.joints[2].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[2]), 0, 1)
            )
            self.joints[3].motorSpeed = float(SPEED_KNEE * np.sign(action[3]))
            self.joints[3].maxMotorTorque = float(
                MOTORS_TORQUE * np.clip(np.abs(action[3]), 0, 1)
            )

        # Waist control from action[4] (spec §4).
        self.waist_joint.motorSpeed = float(WAIST_SPEED * np.sign(action[4]))
        self.waist_joint.maxMotorTorque = float(
            WAIST_TORQUE * np.clip(np.abs(action[4]), 0, 1)
        )

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.chassis.position
        vel = self.chassis.linearVelocity

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
                pos[1] - math.cos(1.5 * i / 10.0) * LIDAR_RANGE,
            )
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        state = [
            self.chassis.angle,                              # 0
            2.0 * self.chassis.angularVelocity / FPS,        # 1
            0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,        # 2
            0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,        # 3
            self.joints[0].angle,                            # 4
            self.joints[0].speed / SPEED_HIP,               # 5
            self.joints[1].angle + 1.0,                      # 6
            self.joints[1].speed / SPEED_KNEE,               # 7
            1.0 if self.legs[1].ground_contact else 0.0,     # 8
            self.joints[2].angle,                            # 9
            self.joints[2].speed / SPEED_HIP,               # 10
            self.joints[3].angle + 1.0,                      # 11
            self.joints[3].speed / SPEED_KNEE,               # 12
            1.0 if self.legs[3].ground_contact else 0.0,     # 13
        ]
        state += [l.fraction for l in self.lidar]  # 14-23
        assert len(state) == 24
        # New waist slots (spec §3).
        state += [
            self.waist_joint.angle,                  # 24
            self.waist_joint.speed / SPEED_HIP,      # 25
        ]
        assert len(state) == 26
        state += self._payload_obs()  # 26-41
        assert len(state) == 42

        self.scroll = pos.x - VIEWPORT_W / SCALE / 5

        shaping = 130 * pos[0] / SCALE
        shaping -= 5.0 * abs(state[0])  # chassis angle

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)

        # Payload bonus (spec §7).
        in_cup_count = int(sum(state[38:41]))
        reward += 0.05 * in_cup_count

        num_lost = self._check_payload_losses()
        if num_lost > 0:
            reward -= 20.0 * num_lost
            self._flash_frames = 3
            # Rewrite payload obs slots to reflect destroyed bodies.
            state[26:42] = self._payload_obs()

        terminated = False
        if self.game_over or pos[0] < 0:
            reward = -100
            terminated = True
        if pos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            terminated = True

        if self.render_mode == "human":
            self.render()
        info = {
            "payloads_remaining": sum(1 for p in self.payloads if p is not None),
            "distance": float(pos.x),
            "waist_angle": float(self.waist_joint.angle),
        }
        return np.array(state, dtype=np.float32), float(reward), terminated, False, info

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[box2d]"`'
            ) from e

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface(
            (VIEWPORT_W + max(0.0, self.scroll) * SCALE, VIEWPORT_H)
        )

        pygame.transform.scale(self.surf, (SCALE, SCALE))

        pygame.draw.polygon(
            self.surf,
            color=(215, 215, 255),
            points=[
                (self.scroll * SCALE, 0),
                (self.scroll * SCALE + VIEWPORT_W, 0),
                (self.scroll * SCALE + VIEWPORT_W, VIEWPORT_H),
                (self.scroll * SCALE, VIEWPORT_H),
            ],
        )

        for poly, x1, x2 in self.cloud_poly:
            if x2 < self.scroll / 2:
                continue
            if x1 > self.scroll / 2 + VIEWPORT_W / SCALE:
                continue
            pygame.draw.polygon(
                self.surf,
                color=(255, 255, 255),
                points=[
                    (p[0] * SCALE + self.scroll * SCALE / 2, p[1] * SCALE) for p in poly
                ],
            )
            gfxdraw.aapolygon(
                self.surf,
                [(p[0] * SCALE + self.scroll * SCALE / 2, p[1] * SCALE) for p in poly],
                (255, 255, 255),
            )
        for poly, color in self.terrain_poly:
            if poly[1][0] < self.scroll:
                continue
            if poly[0][0] > self.scroll + VIEWPORT_W / SCALE:
                continue
            scaled_poly = []
            for coord in poly:
                scaled_poly.append([coord[0] * SCALE, coord[1] * SCALE])
            pygame.draw.polygon(self.surf, color=color, points=scaled_poly)
            gfxdraw.aapolygon(self.surf, scaled_poly, color)

        self.lidar_render = (self.lidar_render + 1) % 100
        i = self.lidar_render
        if i < 2 * len(self.lidar):
            single_lidar = (
                self.lidar[i]
                if i < len(self.lidar)
                else self.lidar[len(self.lidar) - i - 1]
            )
            if hasattr(single_lidar, "p1") and hasattr(single_lidar, "p2"):
                pygame.draw.line(
                    self.surf,
                    color=(255, 0, 0),
                    start_pos=(single_lidar.p1[0] * SCALE, single_lidar.p1[1] * SCALE),
                    end_pos=(single_lidar.p2[0] * SCALE, single_lidar.p2[1] * SCALE),
                    width=1,
                )

        drawlist = self.terrain + self.legs + [self.chassis, self.mug] \
                   + [p for p in self.payloads if p is not None]

        for obj in drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color2,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                        width=1,
                    )
                else:
                    path = [trans * v * SCALE for v in f.shape.vertices]
                    if len(path) > 2:
                        pygame.draw.polygon(self.surf, color=obj.color1, points=path)
                        gfxdraw.aapolygon(self.surf, path, obj.color1)
                        path.append(path[0])
                        pygame.draw.polygon(
                            self.surf, color=obj.color2, points=path, width=1
                        )
                        gfxdraw.aapolygon(self.surf, path, obj.color2)
                    else:
                        pygame.draw.aaline(
                            self.surf,
                            start_pos=path[0],
                            end_pos=path[1],
                            color=obj.color1,
                        )

        flagy1 = TERRAIN_HEIGHT * SCALE
        flagy2 = flagy1 + 50
        x = TERRAIN_STEP * 3 * SCALE
        pygame.draw.aaline(
            self.surf, color=(0, 0, 0), start_pos=(x, flagy1), end_pos=(x, flagy2)
        )
        f = [
            (x, flagy2),
            (x, flagy2 - 10),
            (x + 25, flagy2 - 5),
        ]
        pygame.draw.polygon(self.surf, color=(230, 51, 0), points=f)
        pygame.draw.lines(
            self.surf, color=(0, 0, 0), points=f + [f[0]], width=1, closed=False
        )

        self.surf = pygame.transform.flip(self.surf, False, True)

        # Waist hinge marker (dark circle at joint anchor, post-flip).
        anchor_world = self.waist_joint.anchorA
        sx = anchor_world[0] * SCALE - self.scroll * SCALE
        sy = VIEWPORT_H - anchor_world[1] * SCALE
        pygame.draw.circle(self.surf, (40, 40, 40), (int(sx), int(sy)), 3)

        # HUD overlay (post-flip — text should read normally).
        if not pygame.font.get_init():
            pygame.font.init()
        font = pygame.font.Font(None, 20)
        remaining = sum(1 for p in self.payloads if p is not None)
        hud_lines = [
            f"Payloads: {remaining}/{self._num_payloads}",
            f"Distance: {self.chassis.position[0]:.1f} m",
            f"Waist: {math.degrees(self.waist_joint.angle):.1f}°",
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

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (-self.scroll * SCALE, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )[:, -VIEWPORT_W:]

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
            self.screen = None
            self.clock = None
