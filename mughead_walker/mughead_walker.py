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

# Legs hang from the bottom of the mug slab.
LEG_DOWN = MUG_SLAB_BOTTOM_Y / SCALE
LEG_W, LEG_H = 8 / SCALE, 34 / SCALE

VIEWPORT_W = 600
VIEWPORT_H = 400

TERRAIN_STEP = 14 / SCALE
TERRAIN_LENGTH = 200  # in steps
TERRAIN_HEIGHT = VIEWPORT_H / SCALE / 4
TERRAIN_GRASS = 10  # low long are grass spots, in steps
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


def _build_mug_fixtures():
    return [
        _mug_fixture(MUG_SLAB_HALF, MUG_SLAB_CENTER),
        _mug_fixture(MUG_WALL_HALF, MUG_LEFT_CENTER),
        _mug_fixture(MUG_WALL_HALF, MUG_RIGHT_CENTER),
    ]

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


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        bodies = {contact.fixtureA.body, contact.fixtureB.body}
        if self.env.hull in bodies:
            # Payload-hull contact is normal (payload sitting in the cup); ignore it.
            other = (bodies - {self.env.hull}).pop()
            if other not in self.env.payloads:
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
    This is a simple 4-joint walker robot environment.
    There are two versions:
    - Normal, with slightly uneven terrain.
    - Hardcore, with ladders, stumps, pitfalls.

    To solve the normal version, you need to get 300 points in 1600 time steps.
    To solve the hardcore version, you need 300 points in 2000 time steps.

    A heuristic is provided for testing. It's also useful to get demonstrations
    to learn from. To run the heuristic:
    ```
    python gymnasium/envs/box2d/bipedal_walker.py
    ```

    ## Action Space
    Actions are motor speed values in the [-1, 1] range for each of the
    4 joints at both hips and knees.

    ## Observation Space
    State consists of hull angle speed, angular velocity, horizontal speed,
    vertical speed, position of joints and joints angular speed, legs contact
    with ground, and 10 lidar rangefinder measurements. There are no coordinates
    in the state vector.

    ## Rewards
    Reward is given for moving forward, totaling 300+ points up to the far end.
    If the robot falls, it gets -100. Applying motor torque costs a small
    amount of points. A more optimal agent will get a better score.

    ## Starting State
    The walker starts standing at the left end of the terrain with the hull
    horizontal, and both legs in the same position with a slight knee angle.

    ## Episode Termination
    The episode will terminate if the hull gets in contact with the ground or
    if the walker exceeds the right end of the terrain length.

    ## Arguments

    To use the _hardcore_ environment, you need to specify the `hardcore=True`:

    ```python
    >>> import gymnasium as gym
    >>> env = gym.make("BipedalWalker-v3", hardcore=True, render_mode="rgb_array")
    >>> env
    <TimeLimit<OrderEnforcing<PassiveEnvChecker<BipedalWalker<BipedalWalker-v3>>>>>

    ```

    ## Version History
    - v3: Returns the closest lidar trace instead of furthest;
        faster video recording
    - v2: Count energy spent
    - v1: Legs now report contact with ground; motors have higher torque and
        speed; ground has higher friction; lidar rendered less nervously.
    - v0: Initial version


    <!-- ## References -->

    ## Credits
    Created by Oleg Klimov

    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(self, render_mode: str | None = None, hardcore: bool = False):
        EzPickle.__init__(self, render_mode, hardcore)
        self.isopen = True

        self.world = Box2D.b2World()
        self.terrain: list[Box2D.b2Body] = []
        self.hull: Box2D.b2Body | None = None

        self.prev_shaping = None

        self.hardcore = hardcore

        self.fd_polygon = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]),
            friction=FRICTION,
        )

        self.fd_edge = fixtureDef(
            shape=edgeShape(vertices=[(0, 0), (1, 1)]),
            friction=FRICTION,
            categoryBits=0x0001,
        )

        # we use 5.0 to represent the joints moving at maximum
        # 5 x the rated speed due to impulses from ground contact etc.
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
        self.action_space = spaces.Box(
            np.array([-1, -1, -1, -1]).astype(np.float32),
            np.array([1, 1, 1, 1]).astype(np.float32),
        )
        self.observation_space = spaces.Box(low, high)

        # state = [
        #     self.hull.angle,  # Normal angles up to 0.5 here, but sure more is possible.
        #     2.0 * self.hull.angularVelocity / FPS,
        #     0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Normalized to get -1..1 range
        #     0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
        #     self.joints[
        #         0
        #     ].angle,  # This will give 1.1 on high up, but it's still OK (and there should be spikes on hitting the ground, that's normal too)
        #     self.joints[0].speed / SPEED_HIP,
        #     self.joints[1].angle + 1.0,
        #     self.joints[1].speed / SPEED_KNEE,
        #     1.0 if self.legs[1].ground_contact else 0.0,
        #     self.joints[2].angle,
        #     self.joints[2].speed / SPEED_HIP,
        #     self.joints[3].angle + 1.0,
        #     self.joints[3].speed / SPEED_KNEE,
        #     1.0 if self.legs[3].ground_contact else 0.0,
        # ]
        # state += [l.fraction for l in self.lidar]

        self._num_payloads = DEFAULT_NUM_PAYLOADS
        self._payload_mass_ratio = DEFAULT_PAYLOAD_MASS_RATIO
        self._payload_bounciness = DEFAULT_PAYLOAD_BOUNCINESS

        self.render_mode = render_mode
        self._flash_frames = 0
        self.screen: pygame.Surface | None = None
        self.clock = None

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

        self._generate_terrain(self.hardcore)
        self._generate_clouds()

        init_x = TERRAIN_STEP * TERRAIN_STARTPAD / 2
        init_y = TERRAIN_HEIGHT + 2 * LEG_H
        self.hull = self.world.CreateDynamicBody(
            position=(init_x, init_y), fixtures=_build_mug_fixtures()
        )
        self.hull.color1 = (240, 235, 220)  # cream / ceramic mug
        self.hull.color2 = (60, 60, 60)     # outline
        self.hull.ApplyForceToCenter(
            (self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM), 0), True
        )

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
                bodyA=self.hull,
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
        self.drawlist = self.terrain + self.legs + [self.hull] + self.payloads

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
        return self.step(np.array([0, 0, 0, 0]))[0], {}

    def step(self, action: np.ndarray):
        assert self.hull is not None

        # self.hull.ApplyForceToCenter((0, 20), True) -- Uncomment this to receive a bit of stability help
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

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        pos = self.hull.position
        vel = self.hull.linearVelocity

        for i in range(10):
            self.lidar[i].fraction = 1.0
            self.lidar[i].p1 = pos
            self.lidar[i].p2 = (
                pos[0] + math.sin(1.5 * i / 10.0) * LIDAR_RANGE,
                pos[1] - math.cos(1.5 * i / 10.0) * LIDAR_RANGE,
            )
            self.world.RayCast(self.lidar[i], self.lidar[i].p1, self.lidar[i].p2)

        state = [
            self.hull.angle,  # Normal angles up to 0.5 here, but sure more is possible.
            2.0 * self.hull.angularVelocity / FPS,
            0.3 * vel.x * (VIEWPORT_W / SCALE) / FPS,  # Normalized to get -1..1 range
            0.3 * vel.y * (VIEWPORT_H / SCALE) / FPS,
            self.joints[0].angle,
            # This will give 1.1 on high up, but it's still OK (and there should be spikes on hitting the ground, that's normal too)
            self.joints[0].speed / SPEED_HIP,
            self.joints[1].angle + 1.0,
            self.joints[1].speed / SPEED_KNEE,
            1.0 if self.legs[1].ground_contact else 0.0,
            self.joints[2].angle,
            self.joints[2].speed / SPEED_HIP,
            self.joints[3].angle + 1.0,
            self.joints[3].speed / SPEED_KNEE,
            1.0 if self.legs[3].ground_contact else 0.0,
        ]
        state += [l.fraction for l in self.lidar]
        assert len(state) == 24
        state += self._payload_obs()
        assert len(state) == 40

        self.scroll = pos.x - VIEWPORT_W / SCALE / 5

        shaping = (
            130 * pos[0] / SCALE
        )  # moving forward is a way to receive reward (normalized to get 300 on completion)
        shaping -= 5.0 * abs(
            state[0]
        )  # keep head straight, other than that and falling, any behavior is unpunished

        reward = 0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping

        for a in action:
            reward -= 0.00035 * MOTORS_TORQUE * np.clip(np.abs(a), 0, 1)
            # normalized to about -50.0 using heuristic, more optimal agent should spend less

        # Mug extensions (spec §7).
        in_cup_count = int(sum(state[36:39]))
        reward += 0.05 * in_cup_count

        num_lost = self._check_payload_losses()
        if num_lost > 0:
            reward -= 20.0 * num_lost
            self._flash_frames = 3
            # Rewrite payload obs slots to reflect destroyed bodies.
            state[24:40] = self._payload_obs()

        terminated = False
        if self.game_over or pos[0] < 0:
            reward = -100
            terminated = True
        if pos[0] > (TERRAIN_LENGTH - TERRAIN_GRASS) * TERRAIN_STEP:
            terminated = True

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return np.array(state, dtype=np.float32), reward, terminated, False, {}

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

        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if type(f.shape) is circleShape:
                    pygame.draw.circle(
                        self.surf,
                        color=obj.color1,
                        center=trans * f.shape.pos * SCALE,
                        radius=f.shape.radius * SCALE,
                    )
                    # Draw color2 as a 1-pixel border outline (preserves color1 fill).
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
