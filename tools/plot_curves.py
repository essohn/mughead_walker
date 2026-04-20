"""Extract scalars from TensorBoard logs and save learning curve plots as PNG.

Usage:
    python tools/plot_curves.py runs/ppo_baseline_20260418_013739
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


METRICS = [
    ("rollout/ep_rew_mean", "Episode Reward (rolling 100)"),
    ("rollout/payloads_remaining_mean", "Payloads Surviving (rolling 100)"),
    ("rollout/distance_mean", "Distance Travelled [m] (rolling 100)"),
    ("rollout/fall_rate", "Fall Rate (rolling 100)"),
]


def main():
    if len(sys.argv) < 2:
        print("usage: plot_curves.py <run_dir>")
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    tb_dirs = list((run_dir / "tb").glob("PPO_*"))
    if not tb_dirs:
        print(f"no TB logs under {run_dir / 'tb'}")
        sys.exit(1)
    ea = event_accumulator.EventAccumulator(
        str(tb_dirs[0]), size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (tag, title) in zip(axes.flat, METRICS):
        if tag not in ea.Tags()["scalars"]:
            ax.text(0.5, 0.5, f"no data for {tag}", ha="center")
            continue
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        vals = [e.value for e in events]
        ax.plot(steps, vals)
        ax.set_title(title)
        ax.set_xlabel("timesteps")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"PPO baseline: {run_dir.name}")
    fig.tight_layout()
    out = run_dir / "curves.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
