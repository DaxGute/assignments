from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb

from utils import WANDB_ENTITY, WANDB_PROJECT

EXPERIMENT_KEY = "a1-p1-covarying-v1"
PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
OUT_PATH = Path(__file__).resolve().parent / "plots" / f"{EXPERIMENT_KEY}.png"

PAIRS = (
    {
        "tag": "learning-rate-batch-size",
        "field": "batch_size",
        "title": "Learning rate × batch size",
        "learning_rates": (1e-3, 3e-3, 1e-2, 3e-2),
        "values": (32, 128),
    },
    {
        "tag": "learning-rate-warmup",
        "field": "warmup_percent",
        "title": "Learning rate × warmup",
        "learning_rates": (1e-3, 3e-3, 1e-2),
        "values": (0.003, 0.03),
    },
    {
        "tag": "learning-rate-weight-decay",
        "field": "weight_decay",
        "title": "Learning rate × weight decay",
        "learning_rates": (1e-3, 3e-3, 1e-2),
        "values": (0.03, 0.3),
    },
)


def main():
    runs = list(
        wandb.Api().runs(
            PROJECT_PATH,
            filters={"tags": {"$in": [EXPERIMENT_KEY]}, "state": "finished"},
        )
    )

    matrices = []
    for pair in PAIRS:
        matrix = np.full((len(pair["values"]), len(pair["learning_rates"])), np.nan)
        for run in runs:
            if pair["tag"] not in run.tags:
                continue
            row = pair["values"].index(run.config[pair["field"]])
            column = pair["learning_rates"].index(run.config["learning_rate"])
            matrix[row, column] = run.summary["val_loss"]
        matrices.append(matrix)

    all_losses = np.concatenate([matrix.ravel() for matrix in matrices])
    vmin = np.nanmin(all_losses)
    vmax = np.nanmax(all_losses)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for ax, pair, matrix in zip(axes, PAIRS, matrices, strict=True):
        image = ax.imshow(matrix, cmap="viridis_r", vmin=vmin, vmax=vmax)
        ax.set_title(pair["title"])
        ax.set_xlabel("Learning rate")
        ax.set_ylabel(pair["field"].replace("_", " ").title())
        ax.set_xticks(
            range(len(pair["learning_rates"])),
            [f"{value:g}" for value in pair["learning_rates"]],
        )
        ax.set_yticks(
            range(len(pair["values"])),
            [f"{value:g}" for value in pair["values"]],
        )

        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]:.3f}",
                    ha="center",
                    va="center",
                    color="white",
                )

    fig.colorbar(image, ax=axes, label="Final validation loss")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    print(OUT_PATH.resolve())


if __name__ == "__main__":
    main()
