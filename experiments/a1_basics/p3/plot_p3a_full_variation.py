"""Plot terminal-loss and training-curve variation for part 3a."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb

from utils import WANDB_ENTITY, WANDB_PROJECT

EXPERIMENT_KEY = "a1-p3a-full-variation-v1"
PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
OUT_PATH = Path(__file__).resolve().parent / "plots" / f"{EXPERIMENT_KEY}.png"


def main() -> None:
    runs = list(
        wandb.Api().runs(
            PROJECT_PATH,
            filters={"tags": {"$in": [EXPERIMENT_KEY]}, "state": "finished"},
        )
    )
    if len(runs) < 3:
        raise RuntimeError(f"Expected 3 finished runs, found {len(runs)}.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    losses = np.array([float(run.summary["val_loss"]) for run in runs])
    axes[0].hist(losses, bins="auto", color="#7030A0", alpha=0.75)
    axes[0].scatter(losses, np.zeros_like(losses), color="#111111", zorder=3)
    axes[0].set(
        title=f"Terminal loss (std={losses.std(ddof=1):.2e})",
        xlabel="Final validation loss",
        ylabel="Count",
    )

    for run in runs:
        history = list(run.scan_history(keys=["optimizer_step", "train_loss"]))
        x = [row["optimizer_step"] for row in history if "train_loss" in row]
        y = [row["train_loss"] for row in history if "train_loss" in row]
        gpu = next((tag.upper() for tag in run.tags if tag in {"h100", "a100"}), "GPU")
        axes[1].plot(x, y, alpha=0.8, label=f"{gpu}, seed={run.config['model_seed']}")
    axes[1].set(
        title="Training-loss curves",
        xlabel="Optimizer step",
        ylabel="Training loss",
    )
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(True, linestyle=":", alpha=0.35)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    print(OUT_PATH.resolve())


if __name__ == "__main__":
    main()
