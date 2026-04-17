# MugheadWalker 환경 개발 프로젝트

BipedalWalker (Gymnasium Box2D)를 포크해서 `MugheadWalker-v0`이라는 커스텀 강화학습 환경을 만들어줘. 이 환경은 연세대학교 교양 수업 "인공지능의 이해와 활용"의 강화학습 경쟁 프로젝트에 사용될 예정이고, 학생들이 6라운드에 걸쳐 성능을 겨루게 돼.

## 1. 핵심 컨셉

- 기존 BipedalWalker의 2관절 × 2다리 하체 구조는 유지
- 상체를 **머그컵(mug) 형태의 오픈탑 컨테이너**로 변경
- 컵 안에 **free body payload 3개**를 배치 (Box2D dynamic bodies)
- 목표: payload를 최대한 지키면서 최대한 멀리 전진

## 2. 환경 스펙

### 2.1 물리 바디 구조

**Hull (상체 - 머그컵)**
- 형태: 오픈탑 U자 형태. Box2D `polygonShape` 또는 여러 개의 `edgeShape`로 컵 벽 구성
- 내부 공간: payload 3개가 들어갈 만한 크기 (기존 hull의 약 1.2배 너비, 1.5배 높이)
- 입구는 위쪽으로 열림 (payload가 기울기/가속도에 따라 빠져나갈 수 있음)
- 질량은 기존 BipedalWalker hull과 유사하게 유지 (payload 추가로 인한 학습 부담 최소화)

**다리 (Legs)**
- 기존 BipedalWalker와 동일: hip joint + knee joint × 2다리
- Action space 4차원 유지 (각 관절 토크)

**Payload (3개)**
- 형태: 작은 원형 Box2D `circleShape` (반지름 hull 너비의 약 1/8)
- 질량: 각 payload는 hull 질량의 약 5~8% (3개 합쳐도 hull의 20% 이내)
- 초기 위치: 컵 내부에 세로로 약간씩 간격을 두고 배치
- 마찰 계수: 중간 (0.3~0.5), 탄성계수: 낮음 (0.1~0.2, 너무 튀지 않게)
- 서로 충돌 가능, 컵 벽과 충돌 가능, 지면과 충돌 가능

### 2.2 Observation Space

기존 BipedalWalker observation(24차원)을 확장:

**기본 관측 (기존 유지, 24차원)**
- Hull 각도, 각속도
- Hull 수평/수직 속도
- 각 관절 각도 및 각속도 (4개 관절)
- 각 다리의 접지 여부 (2개)
- 10개 LIDAR 측정값

**Payload 관측 (+16차원)**
- 각 payload의 hull 기준 상대 위치 (x, y) × 3 = 6차원
- 각 payload의 hull 기준 상대 속도 (vx, vy) × 3 = 6차원
- 각 payload의 "컵 내부 여부" 플래그 (0 또는 1) × 3 = 3차원
- 남은 payload 총 개수 (0~3 정수를 float으로) × 1 = 1차원

**총 observation: 40차원**

모든 값은 적절한 스케일로 정규화 (기존 BipedalWalker 컨벤션 따라 대략 -1~1 범위)

### 2.3 Action Space

기존 BipedalWalker와 동일: 4차원 연속값 (-1, 1), 각 관절 토크

### 2.4 Reward 구조

**Step reward (매 스텝):**
- 기존 BipedalWalker의 shaping reward 대부분 유지 (전진 속도, 관절 토크 페널티, 각도 페널티)
- 단, `reward += 0.05 × payload_in_cup_count` 추가 (컵 안에 payload가 많을수록 매 스텝 소폭 보너스)

**Payload 손실 시:**
- payload 하나가 컵에서 빠져나가 "loss"로 판정되면 즉시 `-20` 페널티
- Loss 판정 기준: payload의 y좌표가 hull의 y좌표보다 아래로 내려가거나, payload가 hull 중심에서 일정 거리 이상 벗어남
- 한번 loss된 payload는 시뮬레이션에서 제거 (또는 collision disable)

**종료 시:**
- 기존 종료 조건 유지 (hull이 지면 접촉 → -100 페널티, 목표 지점 도달 시 종료)
- 추가: 모든 payload를 잃어도 에피소드는 계속됨 (학생이 "그냥 빠르게 가기" 전략도 선택 가능)

### 2.5 Episode Termination

- Hull이 지면에 닿음 (넘어짐): -100 reward, episode 종료
- 최대 스텝 도달 (기본 1600 steps): 종료
- 지도 끝 도달: 종료

## 3. 구현 요구사항

### 3.1 파일 구조

```
mughead_walker/
├── __init__.py
├── mughead_walker.py      # 메인 환경 코드
├── README.md              # 사용법, 설치 방법
├── examples/
│   ├── random_agent.py    # 랜덤 정책 테스트
│   ├── train_ppo.py       # SB3 PPO 베이스라인 학습 스크립트
│   └── evaluate.py        # 학습된 모델 평가 스크립트
└── tests/
    └── test_env.py        # 환경 기본 동작 테스트
```

### 3.2 기술 스택

- Python 3.10+
- Gymnasium (최신 안정 버전)
- Box2D (gymnasium[box2d])
- Stable-Baselines3 (베이스라인 학습용)
- NumPy, Pygame (렌더링)

### 3.3 Gymnasium 호환성

- `gym.register()`로 `MugheadWalker-v0` 등록
- 표준 `reset(seed=None, options=None)`, `step(action)` 인터페이스
- `render_mode` 지원: "human", "rgb_array"
- seed 고정 시 재현 가능성 보장

### 3.4 렌더링

- 기본 BipedalWalker의 Pygame 렌더링을 확장
- 머그컵 외형은 단순한 U자 형태로 그리기 (흰색 또는 파스텔 톤)
- Payload는 색깔 있는 원으로 그리기 (3개 다른 색깔로 구분)
- 화면 상단에 `Payloads: 3/3`, `Distance: XX.X` 같은 HUD 표시
- Payload loss 순간 짧은 시각 효과 (예: 빨간 플래시 프레임)

## 4. 베이스라인 검증 요구사항

환경 구현 후, 다음 베이스라인 실험을 수행해서 난이도가 적절한지 검증해줘:

### 4.1 PPO 베이스라인

- Stable-Baselines3 PPO 기본 하이퍼파라미터
- 1M timesteps 학습
- 학습 곡선 (episode reward, payload survival rate, distance) 플롯 저장

### 4.2 성능 지표

- 최종 정책의 평균 reward (10 에피소드 평가)
- 평균 payload 생존 개수
- 평균 이동 거리
- Episode 성공률 (넘어지지 않고 끝까지 간 비율)

### 4.3 검증 기준

- PPO 1M steps로 "의미 있는" 성능(average reward > 50, payload 평균 1개 이상 생존)이 나오면 적절
- 너무 쉬우면 (payload 3개 평균 생존 + 최대 거리 쉽게 도달) payload 물리 파라미터 조정 제안
- 너무 어려우면 (학습이 아예 수렴 안 됨) reward shaping 파라미터 조정 제안

## 5. 라운드별 변주를 위한 설정 가능 파라미터

환경 생성 시 다음 파라미터로 난이도 조절 가능하도록 설계:

```python
env = gym.make(
    "MugheadWalker-v0",
    num_payloads=3,           # payload 개수
    payload_mass_ratio=0.06,  # hull 대비 payload 질량 비율
    terrain_difficulty=0,     # 0: 평지, 1: 경사, 2: 계단, 3: 랜덤
    obstacles=False,          # 장애물 유무
    external_force=0.0,       # 바람/외란 세기
    payload_bounciness=0.15,  # payload 탄성계수
)
```

이 파라미터들은 6라운드 구성 시 사용될 예정.

## 6. 작업 순서

1. BipedalWalker 소스 코드를 Gymnasium 레포에서 확인하고 포크 (`bipedal_walker.py`)
2. Hull 형태를 머그컵으로 변경 (가장 먼저 렌더링으로 시각 확인)
3. Payload 3개 추가 및 물리 파라미터 튜닝 (정적 상태에서 안정적으로 컵 안에 있는지 확인)
4. Observation space 확장 및 정규화 확인
5. Reward 구조 구현 및 payload loss 판정 로직 구현
6. Gymnasium 등록 및 random agent로 기본 동작 테스트
7. PPO 베이스라인 학습 및 난이도 검증
8. README 작성 (설치, 사용법, 파라미터 설명)

## 7. 주의사항

- BipedalWalker 원본 코드의 라이선스(MIT)를 확인하고 적절히 attribution
- Gymnasium API는 예전 OpenAI Gym과 다르니 최신 문서 참고 (특히 `reset`, `step` 반환 튜플)
- Box2D 물리 파라미터는 처음부터 완벽할 수 없으니, 시각적 확인 → 파라미터 조정 반복 필요
- payload 3개가 정지 상태에서 컵 바닥에 안정적으로 놓이는지부터 확인 (이게 안 되면 학습은 불가능)
- 코드 작성 중간중간 `render_mode="human"`으로 시각 확인하면서 진행

## 8. 최종 산출물

- 동작하는 `MugheadWalker-v0` 환경 (pip install 가능한 패키지 형태)
- PPO 학습 결과와 학습 곡선
- 난이도 검증 보고서 (markdown): 현재 난이도가 적절한지, 조정이 필요하다면 어떤 파라미터를 바꿀지
- 각 라운드별 권장 파라미터 세팅 초안

작업을 시작하기 전에 질문이 있으면 먼저 물어보고, 없으면 1번 작업부터 순서대로 진행해줘. 중간에 의사결정이 필요한 지점(예: 물리 파라미터의 구체적 값)에서는 몇 가지 옵션과 추천을 함께 제시해줘.