"""Plot the model- and data-scaling interventions from part 2b."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb

from utils import WANDB_ENTITY, WANDB_PROJECT

EXPERIMENT_KEY = "a1-p2b-slopes-v1"
PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
OUT_PATH = Path(__file__).resolve().parent / "plots" / f"{EXPERIMENT_KEY}.png"


def depth(run) -> int:
    config = run.config.get("model_config") or {}
    return int(config.get("num_hidden_layers") or run.config["model_name"][1:])


def fit_and_plot(ax, x, y, title, xlabel) -> None:
    order = np.argsort(x)
    x, y = np.asarray(x)[order], np.asarray(y)[order]
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    curve = np.geomspace(x.min(), x.max(), 200)
    ax.scatter(x, y, color="#111111", zorder=3)
    ax.plot(curve, np.exp(intercept) * curve**slope, color="#7030A0")
    ax.set(
        title=f"{title} (slope {slope:.3f})",
        xlabel=xlabel,
        ylabel="Final validation loss",
        xscale="log",
        yscale="log",
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.35)


def main() -> None:
    runs = wandb.Api().runs(
        PROJECT_PATH,
        filters={"tags": {"$in": [EXPERIMENT_KEY]}, "state": "finished"},
    )
    points = {"model-scaling": [], "data-scaling": []}
    for run in runs:
        loss = run.summary.get("val_loss")
        if loss is None:
            continue
        tags = set(run.tags or ())
        if "model-scaling" in tags:
            points["model-scaling"].append((depth(run), float(loss)))
        if "data-scaling" in tags:
            tokens = int(run.config["num_train_sequences"]) * int(
                run.config["train_dataset"]["context_length"]
            )
            points["data-scaling"].append((tokens, float(loss)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, key, title, xlabel in (
        (axes[0], "model-scaling", "Model scaling", "Model depth"),
        (axes[1], "data-scaling", "Data scaling", "Training tokens"),
    ):
        if len(points[key]) < 2:
            raise RuntimeError(f"Need at least two finished {key} runs.")
        x, y = zip(*points[key], strict=True)
        fit_and_plot(ax, x, y, title, xlabel)

    fig.suptitle("Slope-preserving intervention: batch=128, warmup=0.03")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    print(OUT_PATH.resolve())


if __name__ == "__main__":
    main()
