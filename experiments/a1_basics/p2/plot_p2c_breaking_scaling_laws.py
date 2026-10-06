"""Show deviations from power-law fits for the interventions in part 2c."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb

from utils import WANDB_ENTITY, WANDB_PROJECT

EXPERIMENT_KEY = "a1-p2c-breaking-v1"
PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
OUT_PATH = Path(__file__).resolve().parent / "plots" / f"{EXPERIMENT_KEY}.png"
SERIES = ("high-lr-no-clip", "sgd", "dropout-constant")


def depth(run) -> int:
    config = run.config.get("model_config") or {}
    return int(config.get("num_hidden_layers") or run.config["model_name"][1:])


def intervention(run) -> str | None:
    tags = set(run.tags or ())
    return next((name for name in SERIES if name in tags), None)


def plot_series(ax, points, label, color) -> None:
    points = sorted(points)
    x = np.array([point[0] for point in points], dtype=float)
    y = np.array([point[1] for point in points], dtype=float)
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    fitted = np.exp(intercept) * x**slope
    max_deviation = np.max(np.abs(np.log(y) - np.log(fitted)))
    ax.plot(
        x,
        y,
        marker="o",
        color=color,
        label=f"{label} (max log residual {max_deviation:.3f})",
    )
    ax.plot(x, fitted, linestyle="--", color=color, alpha=0.65)


def main() -> None:
    runs = wandb.Api().runs(
        PROJECT_PATH,
        filters={"tags": {"$in": [EXPERIMENT_KEY]}, "state": "finished"},
    )
    points = {
        "model-scaling": {name: [] for name in SERIES},
        "data-scaling": {name: [] for name in SERIES},
    }
    for run in runs:
        name = intervention(run)
        loss = run.summary.get("val_loss")
        tags = set(run.tags or ())
        if name is None or loss is None:
            continue
        if "model-scaling" in tags:
            points["model-scaling"][name].append((depth(run), float(loss)))
        if "data-scaling" in tags:
            tokens = int(run.config["num_train_sequences"]) * int(
                run.config["train_dataset"]["context_length"]
            )
            points["data-scaling"][name].append((tokens, float(loss)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    colors = {"high-lr-no-clip": "#e74c3c", "sgd": "#7030A0", "dropout-constant": "#168aad"}
    labels = {
        "high-lr-no-clip": "LR=.1, no clipping",
        "sgd": "SGD, LR=.03",
        "dropout-constant": "dropout=.5, constant LR",
    }
    for ax, axis, title, xlabel in (
        (axes[0], "model-scaling", "Model scaling", "Model depth"),
        (axes[1], "data-scaling", "Data scaling", "Training tokens"),
    ):
        plotted = False
        for name, series_points in points[axis].items():
            if len(series_points) < 2:
                continue
            plot_series(ax, series_points, labels[name], colors[name])
            plotted = True
        if not plotted:
            raise RuntimeError(f"Need at least two finished runs for {axis}.")
        ax.set(
            title=title,
            xlabel=xlabel,
            ylabel="Final validation loss",
            xscale="log",
            yscale="log",
        )
        ax.grid(True, which="both", linestyle=":", alpha=0.35)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Breaking scaling laws (dashed lines are power-law fits)")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    print(OUT_PATH.resolve())


if __name__ == "__main__":
    main()
