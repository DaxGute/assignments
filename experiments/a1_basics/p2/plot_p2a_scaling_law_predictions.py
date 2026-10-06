"""Preregister d8/d9/d20 predictions, then plot the held-out results."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb

from utils import WANDB_ENTITY, WANDB_PROJECT

EXPERIMENT_KEY = "a1-p2a-scaling-v1"
PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
PREREG_PATH = Path(__file__).resolve().parent / f"{EXPERIMENT_KEY}_preregistered.json"
OUT_PATH = Path(__file__).resolve().parent / "plots" / f"{EXPERIMENT_KEY}.png"
RECIPES = {
    "baseline": "Baseline",
    "constant-lr": "Constant LR",
    "dropout-0.2": "Dropout = 0.2",
    "lr-0.03": "Learning rate = 0.03",
}
FIT_DEPTHS = (4, 5, 6, 7)
TEST_DEPTHS = (8, 9)
PREDICT_DEPTHS = (8, 9, 20)


def depth(run) -> int:
    config = run.config.get("model_config") or {}
    return int(config.get("num_hidden_layers") or run.config["model_name"][1:])


def collect() -> dict[str, dict[int, float]]:
    runs = wandb.Api().runs(
        PROJECT_PATH,
        filters={"tags": {"$in": [EXPERIMENT_KEY]}, "state": "finished"},
        order="-created_at",
    )
    losses = {recipe: {} for recipe in RECIPES}
    for run in runs:
        tags = set(run.tags or ())
        recipe = next((name for name in RECIPES if name in tags), None)
        loss = run.summary.get("val_loss")
        if recipe is not None and loss is not None:
            losses[recipe].setdefault(depth(run), float(loss))
    return losses


def fit(points: dict[int, float]) -> dict:
    missing = [value for value in FIT_DEPTHS if value not in points]
    if missing:
        raise RuntimeError(f"Missing finished fit runs at depths {missing}.")
    x = np.array(FIT_DEPTHS, dtype=float)
    y = np.array([points[value] for value in FIT_DEPTHS])
    slope, intercept = np.polyfit(np.log(x), np.log(y), 1)
    coefficient = float(np.exp(intercept))
    return {
        "slope": float(slope),
        "coefficient": coefficient,
        "fit_losses": y.tolist(),
        "predictions": {
            str(value): coefficient * value**slope for value in PREDICT_DEPTHS
        },
    }


def preregister(losses) -> dict:
    if PREREG_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite {PREREG_PATH}.")
    record = {
        "experiment_key": EXPERIMENT_KEY,
        "model": "validation_loss = coefficient * depth ** slope",
        "recipes": {name: fit(losses[name]) for name in RECIPES},
    }
    PREREG_PATH.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Preregistered predictions: {PREREG_PATH.resolve()}")
    return record


def plot(losses, record) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    curve_x = np.linspace(4, 20, 300)
    for ax, (recipe, title) in zip(axes.flat, RECIPES.items(), strict=True):
        model = record["recipes"][recipe]
        prediction = model["predictions"]
        curve_y = model["coefficient"] * curve_x ** model["slope"]
        ax.plot(
            curve_x,
            curve_y,
            color="#7030A0",
            label=f"preregistered fit (slope {model['slope']:.3f})",
        )
        ax.scatter(
            FIT_DEPTHS,
            [losses[recipe][value] for value in FIT_DEPTHS],
            color="#111111",
            label="fit: d4-d7",
        )
        completed = [value for value in TEST_DEPTHS if value in losses[recipe]]
        if completed:
            ax.scatter(
                completed,
                [losses[recipe][value] for value in completed],
                marker="s",
                color="#e74c3c",
                label="held out: d8-d9",
            )
        ax.scatter(
            PREDICT_DEPTHS,
            [prediction[str(value)] for value in PREDICT_DEPTHS],
            marker="x",
            color="#168aad",
            label="predictions",
        )
        ax.set(
            title=title,
            xlabel="Model depth",
            ylabel="Final validation loss",
            xscale="log",
            yscale="log",
        )
        ax.set_xticks((4, 5, 6, 7, 8, 9, 20))
        ax.set_xticklabels(("d4", "d5", "d6", "d7", "d8", "d9", "d20"))
        ax.grid(True, which="both", linestyle=":", alpha=0.35)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Scaling-law predictions fitted only on d4-d7")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    print(OUT_PATH.resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregister", action="store_true")
    args = parser.parse_args()
    losses = collect()
    if args.preregister:
        record = preregister(losses)
    elif PREREG_PATH.exists():
        record = json.loads(PREREG_PATH.read_text())
    else:
        raise FileNotFoundError("Run with --preregister before launching d8-d9.")
    plot(losses, record)
