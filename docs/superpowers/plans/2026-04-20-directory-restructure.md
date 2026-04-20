# Directory Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `examples/`에 섞여 있는 도구/학습 코드를 역할별 디렉토리(`training/`, `tools/`)로 분리하고, 학생이 외울 명령어를 `./scripts/*.sh`로 단순화한다.

**Architecture:** 파일 이동은 `git mv`로 히스토리를 보존한다. `tools/manager.py`는 경로 상수(`TRAINING_DIR`, `TOOLS_DIR`, `EXAMPLES_DIR`)를 나눠 어디서 어떤 스크립트를 호출하는지 자체 문서화된다. `pyproject.toml`의 `exclude`에 새 최상위 폴더를 추가해 패키지 빌드에 포함되지 않게 한다.

**Tech Stack:** Python 3.10+, setuptools, gymnasium, stable-baselines3, bash.

**Spec reference:** `docs/superpowers/specs/2026-04-20-directory-restructure-design.md`

---

### Task 0: Baseline — 현재 상태 기록

**Files:** (읽기만)

- [ ] **Step 1: 현재 테스트 결과 기록**

Run: `pytest -q 2>&1 | tail -5`
Expected: `N passed in Xs` 라인. **숫자 N을 기억해 둔다** — 이후 작업에서 동일한 N이 유지되어야 한다.

- [ ] **Step 2: 현재 manager.py 동작 확인**

Run: `echo 'q' | python examples/manager.py 2>&1 | head -20`
Expected: 메뉴 출력이 보이고 에러 없이 종료. (q 입력은 unknown menu option이지만 파싱은 성공.)

- [ ] **Step 3: 현재 구조 스냅샷**

Run:
```bash
ls examples/ scripts/
git status
```
Expected: `examples/`에 `manager.py evaluate.py plot_curves.py random_agent.py train_ppo.py`가 있고 working tree는 clean.

이 태스크는 커밋 없음 — 읽기 전용 확인.

---

### Task 1: 빈 target 디렉토리 생성 + pyproject.toml 업데이트

**Files:**
- Create: `training/` (빈 디렉토리)
- Create: `tools/` (빈 디렉토리)
- Modify: `pyproject.toml:25`

- [ ] **Step 1: 디렉토리 생성**

Run:
```bash
mkdir -p training tools
```

`scripts/` 는 이미 존재한다 (setup.sh 있음).

- [ ] **Step 2: pyproject.toml의 exclude 업데이트**

Edit `pyproject.toml` line 25:

**Old:**
```toml
exclude = ["tests*", "examples*", "docs*"]
```

**New:**
```toml
exclude = ["tests*", "examples*", "training*", "tools*", "scripts*", "docs*", "notebooks*", "runs*"]
```

(`notebooks`와 `runs`는 원래 `__init__.py`가 없어 패키지로 인식되진 않지만, 방어적으로 명시한다.)

- [ ] **Step 3: 재설치**

Run: `pip install -e '.[dev,rl]' 2>&1 | tail -5`
Expected: `Successfully installed mughead-walker-0.1.0` 또는 `already installed` 없이 에러 없이 종료.

- [ ] **Step 4: import sanity check**

Run: `python -c "import mughead_walker; import gymnasium as gym; env = gym.make('MugheadWalker-v0'); env.reset(seed=0); print('ok:', env.observation_space.shape)"`
Expected: `ok: (42,)`

- [ ] **Step 5: pytest 회귀 확인**

Run: `pytest -q 2>&1 | tail -3`
Expected: Task 0 Step 1과 동일한 N passed.

- [ ] **Step 6: 커밋**

빈 디렉토리는 git이 추적하지 않으므로, 이 커밋은 `pyproject.toml` 변경만 담는다. 디렉토리는 Task 2와 Task 3에서 파일이 들어갈 때 자동으로 추적된다.

```bash
git add pyproject.toml
git commit -m "chore: exclude new top-level dirs from setuptools package discovery"
```

---

### Task 2: training/ 로 이동 (train_ppo.py, evaluate.py)

**Files:**
- Move: `examples/train_ppo.py` → `training/train_ppo.py`
- Move: `examples/evaluate.py` → `training/evaluate.py`

- [ ] **Step 1: git mv**

Run:
```bash
git mv examples/train_ppo.py training/train_ppo.py
git mv examples/evaluate.py training/evaluate.py
```

- [ ] **Step 2: 스크립트 실행 가능 확인**

Run: `python training/train_ppo.py --help 2>&1 | head -20`
Expected: argparse help 출력에 `--timesteps`, `--n-envs`, `--tag` 등이 보임.

Run: `python training/evaluate.py --help 2>&1 | head -10`
Expected: argparse help 출력에 `--model`, `--episodes` 등이 보임.

- [ ] **Step 3: pytest 회귀 확인**

Run: `pytest -q 2>&1 | tail -3`
Expected: Task 0 Step 1과 동일한 N passed.

- [ ] **Step 4: 커밋**

```bash
git add -A
git commit -m "refactor: move train_ppo.py and evaluate.py to training/"
```

---

### Task 3: tools/ 로 이동 (manager.py, plot_curves.py) + 경로 상수 분리

**Files:**
- Move: `examples/manager.py` → `tools/manager.py`
- Move: `examples/plot_curves.py` → `tools/plot_curves.py`
- Modify: `tools/manager.py` (docstring line 8, path constants line 22, 4 call sites)

- [ ] **Step 1: git mv**

Run:
```bash
git mv examples/manager.py tools/manager.py
git mv examples/plot_curves.py tools/plot_curves.py
```

- [ ] **Step 2: manager.py docstring 업데이트**

Edit `tools/manager.py` line 8:

**Old:**
```python
Usage:
    python examples/manager.py
```

**New:**
```python
Usage:
    python tools/manager.py
```

- [ ] **Step 3: manager.py 경로 상수 분리**

Edit `tools/manager.py` line 22:

**Old:**
```python
RUNS_DIR = Path("runs")
EXAMPLES_DIR = Path("examples")
```

**New:**
```python
RUNS_DIR = Path("runs")
TRAINING_DIR = Path("training")   # train_ppo.py, evaluate.py
TOOLS_DIR = Path("tools")         # plot_curves.py
EXAMPLES_DIR = Path("examples")   # random_agent.py
```

- [ ] **Step 4: `train_ppo.py` 호출 두 곳을 `TRAINING_DIR`로 교체**

Edit `tools/manager.py` — 두 번 나오는 다음 줄을 `replace_all`로 교체:

**Old:**
```python
        sys.executable, str(EXAMPLES_DIR / "train_ppo.py"),
```

**New:**
```python
        sys.executable, str(TRAINING_DIR / "train_ppo.py"),
```

(정확히 2개 매치되어야 한다. 0개나 3개면 중단하고 파일 상태를 검토한다.)

- [ ] **Step 5: `evaluate.py` 호출을 `TRAINING_DIR`로 교체**

Edit `tools/manager.py`:

**Old:**
```python
        sys.executable, str(EXAMPLES_DIR / "evaluate.py"),
```

**New:**
```python
        sys.executable, str(TRAINING_DIR / "evaluate.py"),
```

- [ ] **Step 6: `plot_curves.py` 호출을 `TOOLS_DIR`로 교체**

Edit `tools/manager.py`:

**Old:**
```python
    cmd = [sys.executable, str(EXAMPLES_DIR / "plot_curves.py"), str(run["path"])]
```

**New:**
```python
    cmd = [sys.executable, str(TOOLS_DIR / "plot_curves.py"), str(run["path"])]
```

`random_agent.py` 호출(약 line 443)은 그대로 `EXAMPLES_DIR / "random_agent.py"`로 둔다.

- [ ] **Step 7: 남은 `EXAMPLES_DIR` 사용처가 `random_agent.py` 하나뿐인지 확인**

Run: `grep -n "EXAMPLES_DIR /" tools/manager.py`
Expected: 정확히 한 줄, `random_agent.py`를 참조.

- [ ] **Step 8: manager.py 구문 체크**

Run: `python -c "import ast; ast.parse(open('tools/manager.py').read()); print('ok')"`
Expected: `ok`.

- [ ] **Step 9: manager.py 실제 메뉴 로딩 확인**

Run: `echo '' | python tools/manager.py 2>&1 | head -30`
Expected: 메뉴 출력이 에러 없이 보임 (빈 입력은 메인 루프를 한 번 돌리고 다시 prompt로 돌아감, 또는 루프 종료). 트레이스백이 없어야 함.

빈 입력에 대한 처리가 없으면 출력이 짧을 수 있으니, 필요하면 `python tools/manager.py` 를 수동으로 한번 띄워서 메뉴 1번(학습 시작) 선택 → 즉시 Ctrl-C로 종료해 정상 동작을 확인해도 된다.

- [ ] **Step 10: pytest 회귀 확인**

Run: `pytest -q 2>&1 | tail -3`
Expected: Task 0 Step 1과 동일한 N passed.

- [ ] **Step 11: 커밋**

```bash
git add -A
git commit -m "refactor: move manager.py and plot_curves.py to tools/, split path constants"
```

---

### Task 4: scripts/train.sh 생성

**Files:**
- Create: `scripts/train.sh`

- [ ] **Step 1: 파일 작성**

Create `scripts/train.sh`:

```bash
#!/usr/bin/env bash
# Train a PPO agent on MugheadWalker-v0.
# Usage: ./scripts/train.sh <tag> [extra train_ppo.py args...]
# Defaults (timesteps, n-envs, etc.) come from training/train_ppo.py's argparse.
set -euo pipefail
TAG="${1:?usage: ./scripts/train.sh <tag> [extra args...]}"
python training/train_ppo.py --tag "$TAG" "${@:2}"
```

- [ ] **Step 2: 실행 권한 부여**

Run: `chmod +x scripts/train.sh`

- [ ] **Step 3: 빈 인자로 usage 메시지 확인**

Run: `./scripts/train.sh 2>&1 | head -3 || true`
Expected: `usage: ./scripts/train.sh <tag> [extra args...]` 형태의 에러 메시지, non-zero exit. (`|| true`는 set -e 회피.)

- [ ] **Step 4: 짧은 스모크 학습**

Run: `./scripts/train.sh smoke_test --timesteps 20000 --device cpu 2>&1 | tail -15`
Expected: 에러 없이 학습이 끝까지 돌고 `runs/ppo_smoke_test_*`(또는 `runs/smoke_test_*`) 디렉토리가 생성됨. 약 30초~2분.

- [ ] **Step 5: run 디렉토리 확인**

Run: `ls runs/ | grep smoke_test | tail -1`
Expected: 새 run 디렉토리 이름 한 줄. 내부에 `model.zip` 이 있어야 함:

```bash
RUN=$(ls -dt runs/*smoke_test* | head -1)
ls "$RUN"
```

Expected 출력에 `model.zip` 포함.

- [ ] **Step 6: 커밋**

```bash
git add scripts/train.sh
git commit -m "feat: add scripts/train.sh shell wrapper for training"
```

---

### Task 5: scripts/play.sh 생성

**Files:**
- Create: `scripts/play.sh`

- [ ] **Step 1: 파일 작성**

Create `scripts/play.sh`:

```bash
#!/usr/bin/env bash
# Watch a trained MugheadWalker agent in a rendered window.
# Usage: ./scripts/play.sh <run_dir> [extra evaluate.py args...]
# Example: ./scripts/play.sh runs/ppo_smoke_test_20260420
set -euo pipefail
RUN_DIR="${1:?usage: ./scripts/play.sh <run_dir> [extra args...]}"
MODEL="$RUN_DIR/model.zip"
if [[ ! -f "$MODEL" ]]; then
    echo "error: $MODEL not found" >&2
    exit 1
fi
python training/evaluate.py --model "$MODEL" --render --episodes 3 "${@:2}"
```

- [ ] **Step 2: 실행 권한 부여**

Run: `chmod +x scripts/play.sh`

- [ ] **Step 3: usage 메시지 확인**

Run: `./scripts/play.sh 2>&1 | head -3 || true`
Expected: `usage: ./scripts/play.sh <run_dir> [extra args...]`.

- [ ] **Step 4: 존재하지 않는 run에 대한 에러 메시지 확인**

Run: `./scripts/play.sh runs/does_not_exist 2>&1 | head -3 || true`
Expected: `error: runs/does_not_exist/model.zip not found`.

- [ ] **Step 5: 기존 run으로 실제 시각 확인 (렌더 없이 짧게)**

Run:
```bash
RUN=$(ls -dt runs/*smoke_test* runs/ppo_run_* 2>/dev/null | head -1)
echo "Using run: $RUN"
./scripts/play.sh "$RUN" --episodes 1 2>&1 | tail -5
```

Expected: 렌더 창이 잠깐 뜨고 한 에피소드 종료 후 스크립트 정상 종료. (헤드리스 환경이면 pygame이 DISPLAY 없다고 실패할 수 있으니 맥 로컬에서 실행.)

- [ ] **Step 6: 커밋**

```bash
git add scripts/play.sh
git commit -m "feat: add scripts/play.sh to render trained agents"
```

---

### Task 6: README.md 업데이트

**Files:**
- Modify: `README.md:38, 51-53`

- [ ] **Step 1: 현재 해당 섹션 확인**

Run: `sed -n '35,55p' README.md`

그대로 Step 2에서 수정한다.

- [ ] **Step 2: 랜덤 에이전트 경로는 유지, 나머지 갱신**

Edit `README.md`.

**Old (lines 51-54):**
```bash
pip install -e '.[rl]'
python examples/train_ppo.py --timesteps 1000000 --n-envs 8 --tag ppo_baseline
python examples/evaluate.py --model runs/<run_dir>/model.zip --episodes 10
python examples/plot_curves.py runs/<run_dir>
```

**New:**
```bash
pip install -e '.[rl]'
./scripts/train.sh ppo_baseline                   # defaults: 1M timesteps, 8 envs
./scripts/play.sh runs/<run_dir>                  # watches 3 episodes rendered
python tools/plot_curves.py runs/<run_dir>        # or use interactive: python tools/manager.py
```

Line 38 (`python examples/random_agent.py ...`)은 그대로 둔다 — `random_agent.py`는 `examples/`에 남아 있다.

- [ ] **Step 3: 변경 미리보기**

Run: `git diff README.md`
Expected: 3줄만 교체, 나머지는 그대로.

- [ ] **Step 4: 커밋**

```bash
git add README.md
git commit -m "docs: update README to reference new scripts/ wrappers and tools/ paths"
```

---

### Task 7: Colab 노트북 경로 업데이트

**Files:**
- Modify: `notebooks/mughead_walker_colab.ipynb:277`

- [ ] **Step 1: 해당 라인 확인**

Run: `grep -n "examples" notebooks/mughead_walker_colab.ipynb`
Expected: 정확히 1 매치 (line 277).

- [ ] **Step 2: 문자열 교체**

Edit `notebooks/mughead_walker_colab.ipynb`:

**Old:**
```
"- **경쟁 제출**: 저장한 `.zip` 모델을 로컬로 다운로드해서 `examples/evaluate.py`로 벤치마크.\n",
```

**New:**
```
"- **경쟁 제출**: 저장한 `.zip` 모델을 로컬로 다운로드해서 `training/evaluate.py`(또는 `./scripts/play.sh`)로 벤치마크.\n",
```

- [ ] **Step 3: JSON 유효성 확인**

Run: `python -c "import json; json.load(open('notebooks/mughead_walker_colab.ipynb')); print('ok')"`
Expected: `ok`.

- [ ] **Step 4: 남은 `examples/` 레퍼런스 확인**

Run: `grep -n "examples/" notebooks/mughead_walker_colab.ipynb || echo 'none'`
Expected: `none` 또는 `random_agent.py` 언급만 남음. 그 외 `examples/train_ppo|examples/evaluate|examples/manager|examples/plot_curves`가 남아 있으면 수동으로 검토해서 교체한다.

- [ ] **Step 5: 커밋**

```bash
git add notebooks/mughead_walker_colab.ipynb
git commit -m "docs: update colab notebook to reference training/ path"
```

---

### Task 8: 최종 검증 및 스윕

**Files:** (읽기만)

- [ ] **Step 1: 전체 pytest**

Run: `pytest -v 2>&1 | tail -10`
Expected: Task 0 Step 1과 동일한 N passed, 0 failed.

- [ ] **Step 2: 옛 경로 레퍼런스가 남아 있는지 확인**

Run:
```bash
grep -rn --include='*.py' --include='*.md' --include='*.ipynb' --include='*.sh' --include='*.toml' \
    -e 'examples/train_ppo' -e 'examples/evaluate' -e 'examples/manager' -e 'examples/plot_curves' \
    . \
    | grep -v '.venv' | grep -v '.git' | grep -v 'docs/superpowers' || echo 'clean'
```

Expected: `clean`. `docs/superpowers` 아래는 스펙/플랜 문서라 "before" 예시가 남아 있는 게 정상이라 제외한다.

- [ ] **Step 3: `examples/` 내용 확인**

Run: `ls examples/`
Expected: 정확히 `random_agent.py` 하나만.

- [ ] **Step 4: 새 구조 확인**

Run:
```bash
ls training/ tools/ scripts/ examples/
```

Expected:
- `training/`: `evaluate.py train_ppo.py`
- `tools/`: `manager.py plot_curves.py`
- `scripts/`: `play.sh setup.sh train.sh`
- `examples/`: `random_agent.py`

- [ ] **Step 5: 패키지 import 최종 확인**

Run: `python -c "import mughead_walker; import gymnasium as gym; env = gym.make('MugheadWalker-v0'); obs, _ = env.reset(seed=0); print('obs_shape:', obs.shape)"`
Expected: `obs_shape: (42,)`.

- [ ] **Step 6: 스모크 run 정리 (선택)**

Task 4에서 생성한 `runs/ppo_smoke_test_*` 또는 `runs/smoke_test_*` 디렉토리는 학생 배포본에 포함시킬 필요가 없다. 필요 시 삭제:

```bash
rm -rf runs/*smoke_test*
```

(기존 `ppo_run_*` 러닝은 건드리지 않는다.)

- [ ] **Step 7: 최종 커밋 (필요 시)**

Step 6에서 러닝을 지웠다면 이미 `runs/`가 `.gitignore`에 올라가 있으므로 커밋 대상이 아닐 수 있다. 확인:

```bash
git status
```

Untracked한 것만 남고 staged 변경이 없으면 이 태스크는 커밋 없이 종료.

---

## 검증 요약 체크리스트

스펙의 "검증 기준"을 Task별로 매핑:

- ✅ `pip install -e '.[dev,rl]'` 후 import — Task 1 Step 4, Task 8 Step 5
- ✅ `pytest` 전후 동일 — Task 0/2/3/8
- ✅ `python tools/manager.py` 메뉴 동작 — Task 3 Step 9
- ✅ `./scripts/train.sh smoke --timesteps 20000` 완주 — Task 4 Step 4
- ✅ `./scripts/play.sh <run>` 렌더 — Task 5 Step 5
- ✅ Colab 노트북 경로 갱신 — Task 7 (실행 검증은 수동, 아래 "추가 권장" 참고)
- ✅ `grep` 결과 0건 — Task 8 Step 2

## 추가 권장 (필수 아님)

Task 7 이후, 실제로 Colab 노트북을 끝까지 실행해 보고 싶으면:

```bash
jupyter nbconvert --to notebook --execute notebooks/mughead_walker_colab.ipynb --output /tmp/colab_test.ipynb
```

오래 걸리고(학습 셀 포함) 로컬 환경에서 Colab 전용 셀이 실패할 수 있으므로 필수 검증은 아니다. GitHub의 Colab badge 링크를 클릭해서 새 구조로 실행되는지 수동 확인이 더 안전하다.
