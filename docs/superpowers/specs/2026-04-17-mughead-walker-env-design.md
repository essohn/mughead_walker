# MugheadWalker-v0 Environment — Design Spec

**Date:** 2026-04-17
**Scope:** `claude.md` 작업 단계 1–6 + 기본 동작 검증 (random agent 스크립트 포함). PPO 1M-step 베이스라인 학습과 난이도 검증 보고서는 후속 spec에서 다룬다.
**Out of scope (follow-up spec):** PPO 베이스라인 학습 (step 7), 난이도 검증 보고서, 라운드별 파라미터 세팅 확정, terrain_difficulty/obstacles/external_force 실제 구현.

## 1. Goal

BipedalWalker(Gymnasium Box2D)를 포크해서 `MugheadWalker-v0` 강화학습 환경을 만든다. 상체를 오픈탑 머그컵으로 교체하고, 컵 안에 3개의 free-body payload를 배치해 "payload를 지키면서 최대한 멀리 전진"하는 태스크를 제공한다. 연세대학교 교양 수업 "인공지능의 이해와 활용"의 강화학습 경쟁 프로젝트 기반 환경으로 사용된다.

## 2. High-level approach

**포크 전략 — `bipedal_walker.py`를 복사해서 수정**. gymnasium 원본 `bipedal_walker.py`를 `mughead_walker/mughead_walker.py`로 복사한 후 직접 수정한다. 서브클래싱은 원본의 `__init__`이 body 생성까지 한 번에 처리하는 구조라 override가 지저분해지고, 처음부터 새로 쓰는 것은 지형 생성/LIDAR/렌더링까지 재구현해야 해서 범위 초과. claude.md의 "포크" 지시와도 일치.

## 3. Package layout

Repo root = `/Users/esohn/dev/mughead_walker/`.

```
mughead_walker/                 # Python package (repo와 동명)
├── __init__.py                 # gym.register("MugheadWalker-v0", ...)
└── mughead_walker.py           # MugheadWalkerEnv 클래스
examples/
├── random_agent.py             # 이번 spec에 포함
├── train_ppo.py                # placeholder (다음 spec)
└── evaluate.py                 # placeholder (다음 spec)
tests/
└── test_env.py                 # pytest 스모크 테스트
docs/superpowers/specs/
└── 2026-04-17-mughead-walker-env-design.md   # 이 문서
pyproject.toml
README.md
claude.md                       # 기존
```

**Python 환경**: 기존 miniconda base (Python 3.13, gymnasium 1.2.3 + SB3 2.8.0 이미 설치됨). 추가로 `gymnasium[box2d]`, `pygame`, `pytest`를 pip install. 모든 의존성은 `pyproject.toml`에 선언.

**Registration** (`mughead_walker/__init__.py`):
```python
from gymnasium.envs.registration import register
register(
    id="MugheadWalker-v0",
    entry_point="mughead_walker.mughead_walker:MugheadWalkerEnv",
    max_episode_steps=1600,
)
```

## 4. 물리 바디 구조

모든 좌표는 원본 BipedalWalker 컨벤션을 따라 `SCALE = 30`의 언스케일 단위.

### 4.1 Hull (머그컵)

단일 Box2D dynamic body에 3개 rectangle fixture를 부착한 U자 형태.

| Fixture | 중심 (hull local) | 크기 (w × h) |
|---|---|---|
| 바닥 slab | (0, −11) | 76 × 3 |
| 좌벽 | (−36.5, +6) | 3 × 32 |
| 우벽 | (+36.5, +6) | 3 × 32 |

- 바닥 slab: y ∈ [−12.5, −9.5].
- 벽: y ∈ [−10, +22] (벽이 slab 옆면에 0.5 겹쳐서 이음새 틈 없음).
- 내부 공간: x ∈ [−35, +35] (너비 70, 원본 hull 대비 ~1.2×), y ∈ [−9.5, +22] (높이 31.5, 원본 hull 높이 17의 ~1.85×). 원본 "1.5×" 목표보다 약간 더 큰데, 이는 아래 §4.2의 payload 3개 수직 적층이 기하적으로 가능하려면 필요.
- 상단(y > +22) 개방.
- 다리 관절 anchor는 바닥 slab 밑면(y ≈ −12.5)으로 이동. hip joint의 local anchor는 원본에서 해당 offset만큼 내린다.
- 각 fixture의 density는 원본 hull 질량과 총 질량이 비슷하도록 자동 계산 (원본 hull의 `area × density=5.0`를 구해서 `total_mass / total_fixture_area`를 쓴다). `MUG_DENSITY` 상수로 노출해서 수동 조정 가능.

### 4.2 Payload (최대 3개, circle dynamic body)

- 반지름 = 5. claude.md는 "hull 너비의 약 1/8"(≈8.75)을 제시하지만, 반지름 9로는 내부 높이에 3개를 수직 적층할 수 없어(3 × 지름 = 54 > 내부 31.5), 반지름을 줄이고 내부 높이도 늘려 적층을 가능하게 했다. "약 1/8" 가이드라인에서 1/14로 벗어나지만, 물리 안정성을 우선.
- 초기 위치 (hull local, 수직 적층): slot 0 = `(+0.3, −4.5)` (바닥에 접촉), slot 1 = `(−0.2, +5.5)`, slot 2 = `(+0.1, +15.5)`. 지름 10 간격으로 서로 접촉만 하게 배치, 약간의 x 지터(±0.5)로 완전 대칭 회피.
- 지름 = 10, 3개 적층 시 총 높이 30 < 내부 높이 31.5 → 맨 위 ball 꼭대기(y=20.5)는 벽 상단(y=22) 아래에 위치.
- density는 `payload_mass_ratio` 파라미터(기본 0.06)로 계산: `density = (hull_mass × mass_ratio) / (π × r²)`.
- `friction = 0.4`, `restitution = payload_bounciness` (기본 0.15), `linearDamping = 0.5`.

### 4.3 Collision 규칙

- payload ↔ 머그컵 fixture: **충돌**
- payload ↔ payload: **충돌**
- payload ↔ 지면/지형: **충돌**
- payload ↔ 다리: **제외**. 떨어진 payload가 다리에 걸려 물리가 이상해지는 현상을 방지.

구현은 Box2D `categoryBits` / `maskBits` 사용 (원본 BipedalWalker 컨벤션 확장):

| 바디 | category | mask (충돌 대상) |
|---|---|---|
| 지면/지형 | `0x0001` | `0xFFFF` (전체) |
| 다리 (원본 유지) | `0x0020` | `0x0001` (지면만) |
| 머그컵 fixture | `0x0080` (신규) | `0x0001` \| `0x0040` (지면 + payload) |
| payload | `0x0040` (신규) | `0x0001` \| `0x0080` \| `0x0040` (지면 + 머그컵 + 다른 payload) |

다리는 payload category(`0x0040`)가 mask에 없으므로 충돌 제외 자동 성립.

### 4.4 Loss 판정 (매 step)

각 살아있는 payload에 대해:

1. `|payload.world_pos − hull.world_pos| > LOSS_DISTANCE` (Box2D 미터 단위, 기본 `80 / SCALE ≈ 2.67 m` — 머그컵 외곽 대각선의 ~2배), 또는
2. `payload.world_y < TERRAIN_STARTPAD_HEIGHT − 1/SCALE` (지면 아래로 약 3cm 이상 낙하)

둘 중 하나 충족 시 "loss"로 판정. 해당 step에 `−20 × num_lost_this_step` 페널티. Loss된 payload는 `world.DestroyBody()`로 제거하고 slot에 `None`을 저장. Obs의 해당 slot은 이후 모두 0으로 패딩.

좌표 단위 주의: 본 문서의 `§4.1`, `§4.2` 치수는 모두 "unscaled" 단위(원본 BipedalWalker 컨벤션, `SCALE = 30`). Box2D에 fixture를 생성할 때는 `SCALE`로 나눈다. Loss/in_cup 검사는 Box2D 미터 단위(= unscaled / SCALE)에서 수행.

## 5. Observation space (40차원, float32)

| Index | 차원 수 | 내용 | 정규화 |
|---|---|---|---|
| 0–23 | 24 | 원본 BipedalWalker 관측 (hull pose/vel, 관절 × 4, 접지 × 2, LIDAR × 10) | 원본 규칙 그대로 |
| 24–29 | 6 | payload 3 slot × hull-local 상대 위치 (x, y) | `/ 50` |
| 30–35 | 6 | payload 3 slot × hull-local 상대 속도 (vx, vy) | `/ 10` |
| 36–38 | 3 | payload 3 slot × `in_cup` 플래그 | {0.0, 1.0} |
| 39 | 1 | `remaining_count / 3` | [0, 1] |

**Hull-local 변환**: `hull.GetLocalPoint(payload.world_pos)`, 속도는 `hull.GetLocalVector(payload.linear_velocity - hull.GetLinearVelocityFromWorldPoint(payload.world_pos))`.

**`in_cup` 판정** (hull local 기하 검사): `local_x ∈ [−35, 35]` AND `local_y ∈ [−9.5, 22]` (컵 벽 안쪽과 바닥~벽 상단 사이). 여기서 벗어나면 컵 밖 (loss 판정은 별도로 §4.4 거리 기준으로 수행).

**Slot 고정**: payload 순서는 초기 배치 순서(slot 0/1/2)에 고정. loss 시 해당 slot이 0으로 유지되고 뒤의 slot은 당겨오지 않는다. `remaining_count`로 실제 남은 개수 전달.

`observation_space = Box(low=-np.inf, high=np.inf, shape=(40,), dtype=np.float32)`. 원본 BipedalWalker가 이미 `-inf~inf`를 쓰는 컨벤션을 따른다.

## 6. Action space (원본 유지)

`action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)`. 각 원소는 hip1/knee1/hip2/knee2 관절 토크.

## 7. Reward

```python
# 원본 BipedalWalker shaping 그대로
reward  = 130 * (hull.x - prev_hull_x) / SCALE
reward -= 0.00035 * MOTORS_TORQUE * sum(|clipped_actions|)
reward -= 5.0 * |hull_angle|

# Mug 확장
reward += 0.05 * in_cup_count          # 0~0.15 per step
if num_lost_this_step > 0:
    reward -= 20.0 * num_lost_this_step

# 종료 페널티
if hull_ground_contact:
    reward = -100.0
    terminated = True
```

- `in_cup_count` = slot별 `in_cup` 플래그 합 (살아있고 컵 안에 있는 payload 수).
- 전진 보너스와 in-cup 보너스가 독립 가산 → 좋은 주행 시 step당 reward 대략 `0.05 × forward_speed + 0.15`.

## 8. Episode termination

- **`terminated = True`**: hull이 지면 접촉 (reward = −100), 또는 hull.x가 맵 끝(`(TERRAIN_LENGTH − TERRAIN_GRASS) × TERRAIN_STEP`) 넘음.
- **`truncated = True`**: 1600 step 도달 (gymnasium TimeLimit wrapper, `max_episode_steps`에서 관리).
- Payload 전부 loss여도 episode 계속 (`num_payloads=0`과 동일한 관측 상태로 진행).

## 9. Rendering

원본 `bipedal_walker.py`의 pygame 파이프라인 유지. 변경 포인트:

**머그컵**: hull body의 3개 fixture를 각각 polygon으로 그림. 채움 `(240, 235, 220)` 크림/화이트, 외곽선 `(60, 60, 60)` 2px. 손잡이는 생략.

**Payload**: 색상 있는 원 + 흰색 1px 하이라이트.
- slot 0: `(220, 80, 80)` 빨강
- slot 1: `(80, 180, 80)` 초록
- slot 2: `(80, 120, 220)` 파랑

Lost payload는 world에서 제거되므로 자연스럽게 사라짐.

**HUD** (좌상단, pygame font size 20):
```
Payloads: 2/3
Distance: 37.4 m
```
Distance = `hull.position.x / SCALE`.

**Loss flash**: loss 감지 step에서 `self._flash_frames = 3`. `_flash_frames > 0`이면 화면 전체에 반투명 빨강 `(255, 0, 0, 80)` 오버레이 + 감소. 다수 loss는 한 번의 flash로 통합.

**render_mode**:
- `"human"`: pygame window, FPS=50 (원본과 동일).
- `"rgb_array"`: `pygame.surfarray.array3d(surface).transpose(1, 0, 2)` → `(H, W, 3) uint8` 반환.
- `None`: 렌더링 skip.
- `close()`에서 pygame 정리.

LIDAR rays, 지형, 구름 등 원본 데코레이션은 유지.

## 10. Configurable parameters

`MugheadWalkerEnv.__init__(render_mode=None, num_payloads=3, payload_mass_ratio=0.06, payload_bounciness=0.15, terrain_difficulty=0, obstacles=False, external_force=0.0)`.

| 파라미터 | 기본값 | 이번 spec 범위 |
|---|---|---|
| `num_payloads` | 3 | 정수 [0, 3] 지원. obs는 3-slot 고정, 없는 slot은 0으로 패딩. |
| `payload_mass_ratio` | 0.06 | 양수 지원 (payload density 자동 계산). |
| `payload_bounciness` | 0.15 | [0, 1] 지원 (restitution). |
| `terrain_difficulty` | 0 | 0만 지원. 1/2/3은 `NotImplementedError("terrain_difficulty>0 is deferred to a future spec")`. |
| `obstacles` | False | False만 지원. True는 `NotImplementedError`. |
| `external_force` | 0.0 | 0.0만 지원. 나머지는 `NotImplementedError`. |
| `render_mode` | None | "human", "rgb_array", None. |

terrain/obstacles/external_force는 라운드 1–6 준비용으로 별도 spec에서 구현.

## 11. Pytest 스모크 테스트 (`tests/test_env.py`)

1. **`test_registration`**: `gym.make("MugheadWalker-v0")` 성공.
2. **`test_spaces`**: `env.observation_space.shape == (40,)`, dtype float32; `env.action_space.shape == (4,)`, dtype float32, bounds `[-1, 1]`.
3. **`test_reset_step_contract`**: `reset(seed=0)`은 `(obs, info)` 튜플, obs는 `(40,) float32`; `step(action)`은 `(obs, reward, terminated, truncated, info)` 5-튜플; 모든 수치 필드 NaN/Inf 없음.
4. **`test_obs_no_nan_long_rollout`**: 500 step 랜덤 액션 롤아웃 동안 obs에 NaN/Inf 없음. `in_cup` 플래그(idx 36–38)는 항상 {0.0, 1.0}. `remaining_count/3`(idx 39)은 [0, 1].
5. **`test_seed_reproducibility`**: 두 개 env, 같은 seed, 같은 action 시퀀스 → 모든 obs가 완전히 동일 (bitwise).
6. **`test_payload_stays_in_cup_at_rest`**: 0-토크 action으로 100 step 실행 시 3개 payload의 `in_cup=1`이 계속 유지. 물리 튜닝이 잘못되면 여기서 실패 — 핵심 sanity test.
7. **`test_hull_fall_terminates`**: hull을 강제로 땅에 닿게 (hull angle ~π/2 강제) → `terminated == True`, `reward ≤ −100`.
8. **`test_all_payloads_lost_no_terminate`**: 모든 payload의 world 위치를 강제로 loss 조건 밖으로 이동 → 다음 step에서 모두 제거되고 `terminated == False`, `remaining_count == 0`, 그 이후 step도 정상 동작.
9. **`test_configurable_num_payloads_zero`**: `num_payloads=0`으로 reset → obs의 payload slot 모두 0, loss 페널티 발생 안 함, 에피소드 기본 동작.

## 12. Examples

- **`examples/random_agent.py`**: `gym.make("MugheadWalker-v0", render_mode="human")` → 랜덤 액션으로 5 에피소드 실행, 각 에피소드의 reward/distance/surviving payload 출력. 시각 확인용.
- **`examples/train_ppo.py`**: 다음 spec에서 구현할 placeholder. `raise NotImplementedError` + 주석으로 다음 spec 참조.
- **`examples/evaluate.py`**: 위와 동일 placeholder.

## 13. 작업 단계 (plan에서 세분화 예정)

1. Setup: `pyproject.toml`, 패키지 skeleton, `gymnasium[box2d] + pygame + pytest` 설치.
2. Fork: gymnasium 원본 `bipedal_walker.py`를 `mughead_walker/mughead_walker.py`로 복사, 클래스/상수/파일 이름 정리, 원본과 동일 동작 확인.
3. Mug hull: hull polygon 교체 → 3-fixture U자, 다리 anchor 재조정, 렌더링에서 fixture 순회.
4. Payload 바디: circle 3개 생성 (density/friction/restitution 반영), 충돌 카테고리.
5. Observation 확장: 24 → 40, hull-local 변환, `in_cup` 판정, lost slot 패딩.
6. Reward & loss: per-step payload 보너스, loss 검사 + 바디 제거 + −20 페널티.
7. Rendering: payload 원, HUD, loss flash.
8. Configurable params: `__init__` 인자 7개 + 미지원 값 `NotImplementedError`.
9. Registration: `__init__.py`의 `gym.register`.
10. Tests: pytest 스모크 9개.
11. Examples: `random_agent.py` 작동.
12. README: 설치, 사용, 파라미터 docstring 요약.

## 14. 검증 기준 (이번 spec)

- `pytest tests/` 9개 전부 통과.
- `python examples/random_agent.py` 에피소드 실행하면서 시각적으로:
  - 머그컵이 U자 형태로 자연스럽게 렌더됨.
  - Payload 3개가 초기에는 컵 안에 안정적으로 놓여 있음.
  - 걷는 동안 payload가 흔들리며 적절히 튀거나 떨어짐.
  - Payload loss 시 빨간 플래시 + HUD 카운터 감소.
- 원본 BipedalWalker와 유사한 수준의 FPS (50Hz human render 가능).

## 15. Follow-up specs (이 spec 범위 밖)

- **PPO 베이스라인 학습 & 검증**: 1M timesteps, 학습 곡선 플롯, 성능 지표(평균 reward, payload 생존 수, 이동 거리, 성공률). 필요 시 물리/reward 파라미터 재튜닝 제안.
- **Terrain / 장애물 / 외란**: `terrain_difficulty` 1–3, `obstacles=True`, `external_force > 0` 실제 구현. 라운드 1–6 파라미터 세팅 초안.
- **Round pack & 리더보드 인프라**: 학생 대회 운영용.
