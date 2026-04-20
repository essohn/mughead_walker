# Directory Restructure for Student Distribution

**Date:** 2026-04-20
**Status:** Design

## 배경

`MugheadWalker-v0`는 연세대 "인공지능의 이해와 활용" 수업의 강화학습 경쟁 프로젝트 환경이다. 학생 배포 전 현재 디렉토리 구조를 정리한다.

**현재 문제점:**
- `manager.py`(16KB 대화형 CLI 도구)가 `examples/` 폴더 안에 있어 "교육용 예시"와 "실행 도구"가 섞여 있다.
- 학생이 외워야 할 `python ...` 명령이 길어서 타이핑 오류가 나기 쉽다.
- 학생이 수정할 파일(학습 코드, 리워드)과 실행만 할 파일(매니저, 플롯)이 시각적으로 구분되지 않는다.

## 학생의 수정 범위

학생은 다음 3가지만 수정한다 (환경 자체는 최소한으로만 건드림):

1. **NN 구조 수정** — `train_ppo.py`의 `policy_kwargs`
2. **하이퍼파라미터 튜닝** — CLI 인자 또는 `train_ppo.py` 상수
3. **리워드 일부 수정** — `mughead_walker/mughead_walker.py`의 reward 계산 영역

따라서 위 세 파일은 **찾기 쉽고 읽기 쉬워야** 하고, 그 외 파일은 "실행만 하는 것"으로 명확히 분리되어야 한다.

## Before → After

**Before:**
```
examples/
├── manager.py          # 16KB 대화형 도구 (예시 아님)
├── train_ppo.py
├── evaluate.py
├── plot_curves.py
└── random_agent.py
scripts/
└── setup.sh
```

**After:**
```
training/               # 학생이 복사/수정하는 곳
├── train_ppo.py
└── evaluate.py
tools/                  # 학생이 실행만 하는 유틸
├── manager.py
└── plot_curves.py
examples/               # 학생이 읽는 API 데모
└── random_agent.py
scripts/                # 쉘 진입점
├── setup.sh            # 기존 유지
├── train.sh            # 신규
└── play.sh             # 신규
```

변경하지 않는 것: `mughead_walker/` 패키지 폴더, `tests/`, `runs/`, `docs/`, `notebooks/`, 패키지 이름, 의존성.

## 변경 범위 상세

### 1. 파일 이동 (`git mv`로 히스토리 보존)

| From | To |
|---|---|
| `examples/train_ppo.py` | `training/train_ppo.py` |
| `examples/evaluate.py` | `training/evaluate.py` |
| `examples/manager.py` | `tools/manager.py` |
| `examples/plot_curves.py` | `tools/plot_curves.py` |
| `examples/random_agent.py` | 그대로 |

### 2. 신규 쉘 스크립트

**`scripts/train.sh`**
```bash
#!/usr/bin/env bash
set -euo pipefail
TAG="${1:?usage: ./scripts/train.sh <tag> [extra args...]}"
python training/train_ppo.py --tag "$TAG" "${@:2}"
```
사용: `./scripts/train.sh my_run_v1`
→ `train_ppo.py`의 기본값(`--timesteps 1_000_000 --n-envs 8` 등)을 그대로 사용.
학생이 빠른 실험을 원하면: `./scripts/train.sh smoke --timesteps 10000`.
학생이 `cat scripts/train.sh`로 CLI 구조를 자연스럽게 학습.

**`scripts/play.sh`**
```bash
#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="${1:?usage: ./scripts/play.sh <run_dir>}"
python training/evaluate.py --model "$RUN_DIR/model.zip" --render --episodes 3 "${@:2}"
```
사용: `./scripts/play.sh runs/my_run_v1`

둘 다 `chmod +x`로 실행 권한 부여.

### 3. `tools/manager.py` 경로 상수 변경

현재 `EXAMPLES_DIR = Path("examples")` 하나로 모든 스크립트를 가리키고 있음. 분리:

```python
TRAINING_DIR = Path("training")   # train_ppo.py, evaluate.py
TOOLS_DIR = Path("tools")         # plot_curves.py
EXAMPLES_DIR = Path("examples")   # random_agent.py
```

각 `action_*` 함수의 경로 호출을 해당 상수로 교체.

### 4. `pyproject.toml`

`exclude`에 신규 최상위 폴더 추가:
```toml
exclude = ["tests*", "examples*", "training*", "tools*", "scripts*", "docs*"]
```

### 5. `README.md` 업데이트

`python examples/...` 예시를 다음으로 교체:
- `python examples/random_agent.py --episodes 3 --render` (유지)
- `python examples/train_ppo.py ...` → `./scripts/train.sh <tag>`
- `python examples/evaluate.py ...` → `./scripts/play.sh <run_dir>`
- 상세 CLI 플래그는 `python training/train_ppo.py --help` 안내 문구로.

`manager.py` 실행법: `python tools/manager.py`.

### 6. `notebooks/mughead_walker_colab.ipynb`

`examples/` 경로 참조를 새 구조에 맞게 갱신 (학생이 Colab에서 badge 클릭 시 바로 작동해야 함).

## 변경하지 않는 것 (범위 밖)

다음은 별도 스펙으로 분리한다:

- `mughead_walker.py` 내부 리워드 영역 리팩터링 (학생이 쉽게 찾도록)
- `train_ppo.py` 내부의 NN 구조 영역 주석/마킹
- 6라운드 난이도 프리셋
- 학생 배포 패키징 (zip, 리포지토리 분리 등)

## 검증 기준

- [ ] `pip install -e '.[dev,rl]'` 재설치 후 `import mughead_walker` 정상
- [ ] `pytest`가 이동 전과 동일하게 통과
- [ ] `python tools/manager.py` 실행 시 메뉴 정상, 모든 메뉴 항목이 올바른 경로의 스크립트를 호출
- [ ] `./scripts/train.sh smoke --timesteps 10000`이 짧게 끝까지 실행됨
- [ ] `./scripts/play.sh <기존 runs/ 항목>`이 모델 로드 후 렌더링
- [ ] Colab 노트북 첫 셀부터 끝까지 에러 없이 실행 (로컬에서 `jupyter nbconvert --execute`로 확인)
- [ ] `grep -r "examples/train_ppo\|examples/evaluate\|examples/manager\|examples/plot_curves"` 결과가 0건

## Open Questions

없음. 구현 시작 가능.
