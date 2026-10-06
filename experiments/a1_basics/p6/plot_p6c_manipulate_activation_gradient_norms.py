"""Compare five norm interventions against a standard d8 baseline."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb

from utils import WANDB_ENTITY, WANDB_PROJECT

EXPERIMENT_KEY = "a1-p6c-norm-controls-v1"
BASELINE_TAG = "a1-p1-1d-v1"
PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
OUT_PATH = Path(__file__).resolve().parent / "plots" / f"{EXPERIMENT_KEY}.png"
STATISTICS = ("parameter", "activation", "gradient")
INTERVENTIONS = (
    "low-lr-large-batch",
    "high-lr-small-batch",
    "no-qk-norm",
    "heavy-dropout",
    "sgd-no-clip",
)


def baseline_run(runs):
    for run in runs:
        config = run.config
        if (
            BASELINE_TAG in set(run.tags or ())
            and config.get("model_name") == "d8"
            and float(config.get("learning_rate")) == 3e-3
            and int(config.get("batch_size")) == 64
            and config.get("lr_schedule") == "linear"
            and config.get("optimizer_name") == "adamw"
        ):
            return run
    raise RuntimeError("No finished standard d8 baseline found.")


def intervention_runs(runs):
    selected = {}
    for name in INTERVENTIONS:
        selected[name] = next(
            (
                run
                for run in runs
                if EXPERIMENT_KEY in set(run.tags or ())
                and name in set(run.tags or ())
            ),
            None,
        )
        if selected[name] is None:
            raise RuntimeError(f"No finished run found for {name!r}.")
    return selected


def history(run, statistic):
    metric = f"logging/rms/global/{statistic}"
    points = [
        (float(row["progress"]), float(row[metric]))
        for row in run.scan_history(keys=["progress", metric])
        if row.get("progress") is not None and row.get(metric) is not None
    ]
    if not points:
        raise RuntimeError(f"{metric!r} missing from {run.name!r}.")
    x, y = zip(*points, strict=True)
    return np.asarray(x), np.asarray(y)


def main() -> None:
    runs = list(
        wandb.Api().runs(
            PROJECT_PATH,
            filters={
                "tags": {"$in": [EXPERIMENT_KEY, BASELINE_TAG]},
                "state": "finished",
            },
            order="-created_at",
        )
    )
    baseline = baseline_run(runs)
    interventions = intervention_runs(runs)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for column, statistic in enumerate(STATISTICS):
        baseline_x, baseline_y = history(baseline, statistic)
        axes[0, column].plot(
            baseline_x, baseline_y, color="#111111", linewidth=2, label="baseline"
        )
        axes[1, column].axhline(1.0, color="#111111", linewidth=1)
        for name, run in interventions.items():
            x, y = history(run, statistic)
            axes[0, column].plot(x, y, label=name)
            baseline_at_x = np.interp(x, baseline_x, baseline_y)
            axes[1, column].plot(x, y / baseline_at_x, label=name)

        axes[0, column].set(
            title=f"Global {statistic} RMS",
            ylabel="RMS",
            yscale="log",
        )
        axes[1, column].set(
            xlabel="Training progress",
            ylabel="Ratio to baseline",
            yscale="log",
        )
        for row in range(2):
            axes[row, column].grid(True, which="both", linestyle=":", alpha=0.35)

    axes[0, -1].legend(frameon=False, fontsize=7)
    fig.suptitle("Controlling parameter, activation, and gradient RMS")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    print(OUT_PATH.resolve())


if __name__ == "__main__":
    main()
