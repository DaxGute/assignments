"""Compare global RMS trajectories from representative past interventions."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import wandb

from utils import WANDB_ENTITY, WANDB_PROJECT

PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
OUT_PATH = Path(__file__).resolve().parent / "plots" / "a1-p6b-interventions.png"
STATISTICS = ("parameter", "activation", "gradient")
INTERVENTIONS = (
    ("baseline", "a1-p1-1d-v1", {}),
    ("LR = .0003", "a1-p1-1d-v1", {"learning_rate": 3e-4}),
    ("LR = .03", "a1-p1-1d-v1", {"learning_rate": 0.03}),
    ("batch = 256", "a1-p1-1d-v1", {"batch_size": 256}),
    ("weight decay = 1", "a1-p1-1d-v1", {"weight_decay": 1.0}),
)
DEFAULTS = {
    "model_name": "d8",
    "learning_rate": 3e-3,
    "batch_size": 64,
    "lr_schedule": "linear",
    "optimizer_name": "adamw",
    "weight_decay": 0.1,
}


def same(left, right) -> bool:
    if isinstance(right, float):
        return abs(float(left) - right) <= 1e-12 * max(1.0, abs(right))
    return left == right


def matching_run(runs, tag, changes):
    expected = {**DEFAULTS, **changes}
    candidates = []
    for run in runs:
        if tag not in set(run.tags or ()):
            continue
        config = run.config
        if all(
            key in config and same(config[key], value)
            for key, value in expected.items()
        ):
            candidates.append(run)
    if not candidates:
        raise RuntimeError(f"No finished run for {tag}: {changes}")
    return candidates[0]


def history(run, statistic):
    metric = f"logging/rms/global/{statistic}"
    points = []
    for row in run.scan_history(keys=["progress", metric]):
        if row.get("progress") is not None and row.get(metric) is not None:
            points.append((float(row["progress"]), float(row[metric])))
    if not points:
        raise RuntimeError(f"{metric!r} missing from {run.name!r}.")
    return zip(*points, strict=True)


def main() -> None:
    tags = [tag for _, tag, _ in INTERVENTIONS]
    runs = list(
        wandb.Api().runs(
            PROJECT_PATH,
            filters={"tags": {"$in": tags}, "state": "finished"},
            order="-created_at",
        )
    )
    selected = [
        (label, matching_run(runs, tag, changes))
        for label, tag, changes in INTERVENTIONS
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, statistic in zip(axes, STATISTICS, strict=True):
        for label, run in selected:
            x, y = history(run, statistic)
            ax.plot(x, y, label=label)
        ax.set(
            title=f"Global {statistic} RMS",
            xlabel="Training progress",
            ylabel="RMS",
            yscale="log",
        )
        ax.grid(True, which="both", linestyle=":", alpha=0.35)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("How past interventions change training statistics")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    print(OUT_PATH.resolve())


if __name__ == "__main__":
    main()
