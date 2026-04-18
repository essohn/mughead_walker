#!/usr/bin/env python3
"""MugheadWalker interactive manager.

Menu-based interface for training, evaluation, and plotting.
Scans `runs/` for saved models and lets you pick interactively.

Usage:
    python examples/manager.py
"""
import json
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RUNS_DIR = Path("runs")
EXAMPLES_DIR = Path("examples")


def pause():
    input("\n[Enter] 계속...")


def prompt_int(msg: str, default: int) -> int:
    raw = input(f"{msg} (기본값: {default}): ").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"  잘못된 입력, 기본값 {default} 사용")
        return default


def prompt_str(msg: str, default: str) -> str:
    raw = input(f"{msg} (기본값: {default}): ").strip()
    return raw if raw else default


def prompt_bool(msg: str, default: bool = False) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = input(f"{msg} ({hint}): ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def discover_runs() -> list[dict]:
    """Return list of run metadata dicts, sorted newest-first."""
    if not RUNS_DIR.exists():
        return []
    runs = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        config_path = d / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        final_model = d / "model.zip"
        checkpoints = sorted(
            (d / "checkpoints").glob("*.zip")
        ) if (d / "checkpoints").exists() else []
        has_tb = (d / "tb").exists() and any((d / "tb").iterdir())
        has_curves = (d / "curves.png").exists()
        runs.append({
            "name": d.name,
            "path": d,
            "config": config,
            "final_model": final_model if final_model.exists() else None,
            "checkpoints": checkpoints,
            "has_tb": has_tb,
            "has_curves": has_curves,
        })
    return runs


def pick_run(runs: list[dict], prompt_msg: str = "실행 선택") -> dict | None:
    """Display runs and let user pick one."""
    if not runs:
        print("  저장된 실행이 없습니다.")
        return None
    print(f"\n{'#':>3}  {'실행 이름':<45} {'Timesteps':>10}  {'모델':>5}  {'체크':>4}  {'TB':>3}")
    print("-" * 85)
    for i, r in enumerate(runs, 1):
        ts = r["config"].get("timesteps", "?")
        model_mark = "O" if r["final_model"] else "-"
        ckpt_count = len(r["checkpoints"])
        tb_mark = "O" if r["has_tb"] else "-"
        print(f"{i:>3}  {r['name']:<45} {ts:>10}  {model_mark:>5}  {ckpt_count:>4}  {tb_mark:>3}")
    print(f"  0  <- 뒤로가기")
    while True:
        raw = input(f"\n{prompt_msg} [0-{len(runs)}]: ").strip()
        if not raw:
            continue
        try:
            idx = int(raw)
        except ValueError:
            continue
        if idx == 0:
            return None
        if 1 <= idx <= len(runs):
            return runs[idx - 1]


def pick_model(run: dict) -> Path | None:
    """Let user pick a specific model (final or checkpoint) from a run."""
    options: list[tuple[str, Path]] = []
    if run["final_model"]:
        options.append(("최종 모델 (model.zip)", run["final_model"]))
    for ckpt in run["checkpoints"]:
        match = re.search(r"(\d+)_steps", ckpt.name)
        step_label = f"{int(match.group(1)):,} steps" if match else ckpt.name
        options.append((f"체크포인트: {step_label}", ckpt))

    if not options:
        print("  이 실행에 사용 가능한 모델이 없습니다.")
        return None

    print(f"\n  모델 목록 ({run['name']}):")
    for i, (label, _) in enumerate(options, 1):
        print(f"    {i:>3}. {label}")
    print(f"      0. <- 뒤로가기")

    while True:
        raw = input(f"\n  모델 선택 [0-{len(options)}]: ").strip()
        if not raw:
            continue
        try:
            idx = int(raw)
        except ValueError:
            continue
        if idx == 0:
            return None
        if 1 <= idx <= len(options):
            return options[idx - 1][1]


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def action_train():
    """Interactive training setup."""
    print("\n" + "=" * 60)
    print("  모델 학습")
    print("=" * 60)

    algo = prompt_str("  알고리즘 (ppo/sac/td3/a2c)", "ppo").lower()
    if algo not in ("ppo", "sac", "td3", "a2c"):
        print(f"  지원하지 않는 알고리즘: {algo}")
        return
    timesteps = prompt_int("  총 타임스텝", 1_000_000)
    n_envs = prompt_int("  병렬 환경 수", 8)
    seed = prompt_int("  시드", 0)
    tag = prompt_str("  태그 (실행 이름 접두사)", f"{algo}_run")
    device = prompt_str("  디바이스 (cpu/cuda/mps)", "cpu")
    checkpoint_every = prompt_int("  체크포인트 간격 (steps)", 100_000)
    terrain = prompt_int("  지형 난이도 (0=평지, 1=하드코어)", 0)
    dummy = prompt_bool("  DummyVecEnv 사용 (디버그용)?", False)

    cmd = [
        sys.executable, str(EXAMPLES_DIR / "train_ppo.py"),
        "--algo", algo,
        "--timesteps", str(timesteps),
        "--n-envs", str(n_envs),
        "--seed", str(seed),
        "--tag", tag,
        "--device", device,
        "--checkpoint-every", str(checkpoint_every),
        "--terrain-difficulty", str(terrain),
    ]
    if dummy:
        cmd.append("--dummy-vec")

    print(f"\n  실행 명령어:")
    print(f"  {' '.join(cmd)}\n")

    if not prompt_bool("  학습을 시작할까요?", True):
        return

    subprocess.run(cmd)
    pause()


def action_evaluate():
    """Interactive evaluation."""
    print("\n" + "=" * 60)
    print("  모델 평가")
    print("=" * 60)

    runs = discover_runs()
    run = pick_run(runs, "평가할 실행 선택")
    if run is None:
        return

    model_path = pick_model(run)
    if model_path is None:
        return

    # Auto-detect algo from config.json, fall back to ppo
    saved_algo = run["config"].get("algo", "ppo")
    print(f"\n  선택된 모델: {model_path}")
    print(f"  감지된 알고리즘: {saved_algo}")
    algo = prompt_str("  알고리즘 (ppo/sac/td3/a2c)", saved_algo).lower()
    episodes = prompt_int("  평가 에피소드 수", 10)
    seed = prompt_int("  시드", 1000)
    terrain = prompt_int("  지형 난이도 (0=평지, 1=하드코어)", 0)
    render = prompt_bool("  렌더링 (human 모드)?", False)
    save_json = prompt_bool("  결과를 JSON으로 저장?", True)

    out_path = None
    if save_json:
        default_out = str(run["path"] / "eval_result.json")
        out_path = prompt_str("  JSON 저장 경로", default_out)

    cmd = [
        sys.executable, str(EXAMPLES_DIR / "evaluate.py"),
        "--model", str(model_path),
        "--algo", algo,
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--terrain-difficulty", str(terrain),
    ]
    if render:
        cmd.append("--render")
    if out_path:
        cmd.extend(["--out", out_path])

    print(f"\n  실행 명령어:")
    print(f"  {' '.join(cmd)}\n")

    subprocess.run(cmd)
    pause()


def action_plot():
    """Interactive plot generation."""
    print("\n" + "=" * 60)
    print("  학습 곡선 플롯")
    print("=" * 60)

    runs = discover_runs()
    tb_runs = [r for r in runs if r["has_tb"]]
    if not tb_runs:
        print("  TensorBoard 로그가 있는 실행이 없습니다.")
        pause()
        return

    run = pick_run(tb_runs, "플롯할 실행 선택")
    if run is None:
        return

    cmd = [sys.executable, str(EXAMPLES_DIR / "plot_curves.py"), str(run["path"])]

    print(f"\n  실행 명령어:")
    print(f"  {' '.join(cmd)}\n")

    subprocess.run(cmd)

    curves_path = run["path"] / "curves.png"
    if curves_path.exists():
        print(f"\n  그래프 저장됨: {curves_path}")
        if prompt_bool("  이미지를 열어볼까요?", True):
            if sys.platform == "darwin":
                subprocess.run(["open", str(curves_path)])
            elif sys.platform == "linux":
                subprocess.run(["xdg-open", str(curves_path)])
            else:
                print(f"  직접 열어주세요: {curves_path}")
    pause()


def action_browse():
    """Browse runs and show details."""
    print("\n" + "=" * 60)
    print("  실행 목록 상세 보기")
    print("=" * 60)

    runs = discover_runs()
    run = pick_run(runs, "상세 보기할 실행 선택")
    if run is None:
        return

    print(f"\n  실행: {run['name']}")
    print(f"  경로: {run['path']}")
    print(f"  최종 모델: {run['final_model'] or '없음'}")
    print(f"  체크포인트: {len(run['checkpoints'])}개")
    print(f"  TensorBoard: {'있음' if run['has_tb'] else '없음'}")
    print(f"  곡선 그래프: {'있음' if run['has_curves'] else '없음'}")

    if run["config"]:
        print(f"\n  학습 설정:")
        for k, v in run["config"].items():
            print(f"    {k}: {v}")

    eval_path = run["path"] / "eval_result.json"
    if eval_path.exists():
        result = json.loads(eval_path.read_text())
        print(f"\n  평가 결과:")
        for k, v in result.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    pause()


def action_tensorboard():
    """Launch TensorBoard for a selected run."""
    print("\n" + "=" * 60)
    print("  TensorBoard 실행")
    print("=" * 60)

    runs = discover_runs()
    tb_runs = [r for r in runs if r["has_tb"]]
    if not tb_runs:
        print("  TensorBoard 로그가 있는 실행이 없습니다.")
        pause()
        return

    print("\n  여러 실행을 비교할 수 있습니다.")
    all_mode = prompt_bool("  전체 실행을 한번에 볼까요?", False)

    if all_mode:
        logdir = str(RUNS_DIR)
    else:
        run = pick_run(tb_runs, "TensorBoard로 볼 실행 선택")
        if run is None:
            return
        logdir = str(run["path"] / "tb")

    port = prompt_int("  포트", 6006)

    cmd = ["tensorboard", "--logdir", logdir, "--port", str(port)]
    print(f"\n  실행 명령어:")
    print(f"  {' '.join(cmd)}")
    print(f"\n  브라우저에서 http://localhost:{port} 으로 접속하세요.")
    print("  종료하려면 Ctrl+C를 누르세요.\n")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n  TensorBoard 종료.")
    pause()


def action_random_agent():
    """Run random agent for visual check."""
    print("\n" + "=" * 60)
    print("  랜덤 에이전트 실행 (시각 확인)")
    print("=" * 60)

    terrain = prompt_int("  지형 난이도 (0=평지, 1=하드코어)", 0)
    render = prompt_bool("  렌더링 (human 모드)?", True)

    cmd = [sys.executable, str(EXAMPLES_DIR / "random_agent.py"),
           "--terrain-difficulty", str(terrain)]
    if render:
        cmd.append("--render")
    print(f"\n  실행 명령어: {' '.join(cmd)}\n")
    subprocess.run(cmd)
    pause()


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

MENU = [
    ("1", "모델 학습 (PPO/SAC/TD3/A2C)", action_train),
    ("2", "모델 평가", action_evaluate),
    ("3", "학습 곡선 플롯", action_plot),
    ("4", "실행 목록/상세 보기", action_browse),
    ("5", "TensorBoard 실행", action_tensorboard),
    ("6", "랜덤 에이전트 (시각 확인)", action_random_agent),
    ("q", "종료", None),
]


def main_menu():
    while True:
        print("\n" + "=" * 60)
        print("  MugheadWalker Manager")
        print("=" * 60)

        runs = discover_runs()
        print(f"\n  저장된 실행: {len(runs)}개")
        total_models = sum(
            (1 if r["final_model"] else 0) + len(r["checkpoints"]) for r in runs
        )
        print(f"  사용 가능한 모델: {total_models}개\n")

        for key, label, _ in MENU:
            print(f"  [{key}] {label}")

        choice = input("\n  선택: ").strip().lower()

        if choice == "q":
            print("\n  종료합니다.\n")
            break

        for key, _, action in MENU:
            if choice == key and action is not None:
                action()
                break
        else:
            if choice != "q":
                print("  잘못된 선택입니다.")
                pause()


if __name__ == "__main__":
    main_menu()
