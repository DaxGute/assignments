from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb

from utils import WANDB_ENTITY, WANDB_PROJECT

EXPERIMENT_KEY = "a1-p1-schedulers-v2"
PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
OUT_PATH = Path(__file__).resolve().parent / "plots" / f"{EXPERIMENT_KEY}.png"

SCHEDULES = ("linear", "cos", "constant", "wsd0.1", "wsd0.2")
LEARNING_RATES = (3e-3, 1e-2)


def main():
    runs = wandb.Api().runs(
        PROJECT_PATH,
        filters={"tags": {"$in": [EXPERIMENT_KEY]}, "state": "finished"},
    )

    losses = {}
    for run in runs:
        key = (
            run.config["lr_schedule"],
            run.config["learning_rate"],
            run.config["batch_size"],
            run.config["warmup_percent"],
            run.config["weight_decay"],
            run.config["optimizer_name"],
            run.config["beta1"],
            run.config["beta2"],
            run.config["grad_norm"],
        )
        losses[key] = run.summary["val_loss"]

    def loss(**changes):
        config = {
            "lr_schedule": "linear",
            "learning_rate": 3e-3,
            "batch_size": 64,
            "warmup_percent": 0.01,
            "weight_decay": 0.1,
            "optimizer_name": "adamw",
            "beta1": 0.9,
            "beta2": 0.95,
            "grad_norm": 1.0,
            **changes,
        }
        return losses[
            (
                config["lr_schedule"],
                config["learning_rate"],
                config["batch_size"],
                config["warmup_percent"],
                config["weight_decay"],
                config["optimizer_name"],
                config["beta1"],
                config["beta2"],
                config["grad_norm"],
            )
        ]

    lr_matrix = np.array(
        [
            [
                loss(lr_schedule=schedule, learning_rate=learning_rate)
                for learning_rate in LEARNING_RATES
            ]
            for schedule in SCHEDULES
        ]
    )
    interaction_labels = ("default", "batch=128", "warmup=0.03", "wd=0.3")
    interaction_losses = (
        loss(lr_schedule="cos"),
        loss(lr_schedule="cos", batch_size=128),
        loss(lr_schedule="cos", warmup_percent=0.03),
        loss(lr_schedule="cos", weight_decay=0.3),
    )
    optimizer_labels = (
        "AdamW default",
        "SGD lr=.003",
        "SGD lr=.03",
        "β₁=.8",
        "β₁=.95",
        "β₂=.9",
        "β₂=.99",
        "no clipping",
    )
    optimizer_losses = (
        loss(lr_schedule="cos"),
        loss(lr_schedule="cos", optimizer_name="sgd", learning_rate=3e-3),
        loss(lr_schedule="cos", optimizer_name="sgd", learning_rate=3e-2),
        loss(lr_schedule="cos", beta1=0.8),
        loss(lr_schedule="cos", beta1=0.95),
        loss(lr_schedule="cos", beta2=0.9),
        loss(lr_schedule="cos", beta2=0.99),
        loss(lr_schedule="cos", grad_norm=None),
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    image = axes[0].imshow(lr_matrix, cmap="viridis_r")
    axes[0].set_title("Scheduler × learning rate")
    axes[0].set_xlabel("Learning rate")
    axes[0].set_ylabel("Scheduler")
    axes[0].set_xticks(
        range(len(LEARNING_RATES)), [f"{value:g}" for value in LEARNING_RATES]
    )
    axes[0].set_yticks(range(len(SCHEDULES)), SCHEDULES)
    for row in range(lr_matrix.shape[0]):
        for column in range(lr_matrix.shape[1]):
            axes[0].text(
                column,
                row,
                f"{lr_matrix[row, column]:.3f}",
                ha="center",
                va="center",
                color="white",
            )
    fig.colorbar(image, ax=axes[0], label="Final validation loss")

    axes[1].bar(interaction_labels, interaction_losses)
    axes[1].set_title("Cosine with Part A/B settings")
    axes[1].set_ylabel("Final validation loss")
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].barh(optimizer_labels, optimizer_losses)
    axes[2].invert_yaxis()
    axes[2].set_title("Other optimizer hyperparameters")
    axes[2].set_xlabel("Final validation loss")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    print(OUT_PATH.resolve())


if __name__ == "__main__":
    main()
