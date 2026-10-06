"""Compare isolated stochastic sources with the full variation from part 3a."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb

from utils import WANDB_ENTITY, WANDB_PROJECT

EXPERIMENT_KEY = "a1-p3b-isolated-v1"
FULL_VARIATION_KEY = "a1-p3a-full-variation-v1"
PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
OUT_PATH = Path(__file__).resolve().parent / "plots" / f"{EXPERIMENT_KEY}.png"
SOURCES = ("reference", "model-seed", "data-seed", "hardware")


def tagged_runs(api, key):
    return list(
        api.runs(
            PROJECT_PATH,
            filters={"tags": {"$in": [key]}, "state": "finished"},
        )
    )


def main() -> None:
    api = wandb.Api()
    runs = tagged_runs(api, EXPERIMENT_KEY)
    full_runs = tagged_runs(api, FULL_VARIATION_KEY)
    if len(runs) < 8 or len(full_runs) < 2:
        raise RuntimeError("Part 3a and all 8 part 3b runs must finish before plotting.")

    losses = {source: [] for source in SOURCES}
    for run in runs:
        source = next((value for value in SOURCES if value in set(run.tags or ())), None)
        if source is not None:
            losses[source].append(float(run.summary["val_loss"]))
    full_losses = np.array([float(run.summary["val_loss"]) for run in full_runs])
    full_std = full_losses.std(ddof=1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for index, source in enumerate(SOURCES):
        values = losses[source]
        axes[0].scatter([index] * len(values), values, s=55)
        if values:
            axes[0].hlines(np.mean(values), index - 0.25, index + 0.25, color="#111111")
    axes[0].set(
        title="Terminal losses by isolated source",
        ylabel="Final validation loss",
    )
    axes[0].set_xticks(range(len(SOURCES)), SOURCES, rotation=20)

    fractions = [
        np.std(losses[source], ddof=1) / full_std if len(losses[source]) > 1 else np.nan
        for source in SOURCES
    ]
    axes[1].bar(SOURCES, fractions, color="#7030A0")
    axes[1].axhline(1.0, color="#e74c3c", linestyle="--", label="full variation")
    axes[1].set(
        title="Fraction of full run-to-run variation",
        ylabel="Isolated std / full std",
    )
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(frameon=False)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    print(OUT_PATH.resolve())


if __name__ == "__main__":
    main()
