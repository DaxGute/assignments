"""Compare run-to-run variability across the seven conditions in part 3c."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb

from utils import WANDB_ENTITY, WANDB_PROJECT

EXPERIMENT_KEY = "a1-p3c-hyperparameters-v1"
PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
OUT_PATH = Path(__file__).resolve().parent / "plots" / f"{EXPERIMENT_KEY}.png"
CONDITIONS = (
    "baseline",
    "high-lr",
    "dropout",
    "constant-lr",
    "no-clipping",
    "larger-d10",
    "longer-2epochs",
)


def main() -> None:
    runs = wandb.Api().runs(
        PROJECT_PATH,
        filters={"tags": {"$in": [EXPERIMENT_KEY]}, "state": "finished"},
    )
    losses = {condition: [] for condition in CONDITIONS}
    for run in runs:
        tags = set(run.tags or ())
        condition = next((value for value in CONDITIONS if value in tags), None)
        loss = run.summary.get("val_loss")
        if condition is not None and loss is not None:
            losses[condition].append(float(loss))
    missing = [condition for condition, values in losses.items() if len(values) < 2]
    if missing:
        raise RuntimeError(f"Need two finished runs for: {', '.join(missing)}")

    means = np.array([np.mean(losses[value]) for value in CONDITIONS])
    stds = np.array([np.std(losses[value], ddof=1) for value in CONDITIONS])
    x = np.arange(len(CONDITIONS))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    for index, condition in enumerate(CONDITIONS):
        axes[0].scatter([index] * 2, losses[condition], color="#111111", zorder=3)
    axes[0].errorbar(
        x,
        means,
        yerr=stds,
        fmt="none",
        ecolor="#7030A0",
        capsize=5,
        linewidth=2,
    )
    axes[0].set(title="Terminal losses", ylabel="Final validation loss")
    axes[0].set_xticks(x, CONDITIONS, rotation=30, ha="right")

    axes[1].scatter(means, stds, color="#7030A0", s=60)
    for condition, mean, std in zip(CONDITIONS, means, stds, strict=True):
        axes[1].annotate(condition, (mean, std), xytext=(4, 4), textcoords="offset points")
    axes[1].set(
        title="Variation versus capability",
        xlabel="Mean final validation loss (lower is better)",
        ylabel="Run-to-run standard deviation",
    )
    axes[1].grid(True, linestyle=":", alpha=0.35)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    print(OUT_PATH.resolve())


if __name__ == "__main__":
    main()
