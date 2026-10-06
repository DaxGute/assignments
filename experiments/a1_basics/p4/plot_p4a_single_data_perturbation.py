"""Compare terminal loss after deterministic pre-training data perturbations."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb

from utils import WANDB_ENTITY, WANDB_PROJECT

EXPERIMENT_KEY = "a1-p4a-perturbation-v1"
PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
OUT_PATH = Path(__file__).resolve().parent / "plots" / f"{EXPERIMENT_KEY}.png"
SERIES = ("baseline", "1-token", "1024-tokens")


def main() -> None:
    runs = wandb.Api().runs(
        PROJECT_PATH,
        filters={"tags": {"$in": [EXPERIMENT_KEY]}, "state": "finished"},
        order="-created_at",
    )
    losses = {}
    for run in runs:
        tags = set(run.tags or ())
        label = next((name for name in SERIES if name in tags), None)
        loss = run.summary.get("val_loss")
        if label is not None and loss is not None:
            losses.setdefault(label, float(loss))

    missing = [label for label in SERIES if label not in losses]
    if missing:
        raise RuntimeError(f"Missing finished runs: {missing}")

    values = np.array([losses[label] for label in SERIES])
    deltas = np.abs(values - values[0])
    labels = ("unchanged", "1 token", "1 sequence")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    axes[0].bar(labels, values, color=("#777777", "#7030A0", "#168aad"))
    axes[0].set(title="Terminal validation loss", ylabel="Validation loss")
    axes[1].bar(labels[1:], deltas[1:], color=("#7030A0", "#168aad"))
    axes[1].set(
        title="Variation from deterministic baseline",
        ylabel="Absolute terminal loss difference",
        yscale="log",
    )
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    print(OUT_PATH.resolve())


if __name__ == "__main__":
    main()
