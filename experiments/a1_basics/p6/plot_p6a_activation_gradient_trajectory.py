"""Plot start/middle/end RMS statistics for a standard d8 training run."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb

from utils import WANDB_ENTITY, WANDB_PROJECT

SOURCE_TAG = "a1-p1-1d-v1"
PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
OUT_PATH = Path(__file__).resolve().parent / "plots" / "a1-p6a-trajectory.png"
LAYERS = (0, 3, 7)
COMPONENTS = (
    ("attn q", "self_attn.q_proj"),
    ("attn out", "self_attn.o_proj"),
    ("MLP gate", "mlp.gate_proj"),
    ("MLP out", "mlp.down_proj"),
)
STATISTICS = ("parameter", "activation", "gradient")
TARGET_PROGRESS = (0.0, 0.5, 1.0)


def standard_d8_run():
    runs = wandb.Api().runs(
        PROJECT_PATH,
        filters={"tags": {"$in": [SOURCE_TAG]}, "state": "finished"},
        order="-created_at",
    )
    for run in runs:
        config = run.config
        if (
            config.get("model_name") == "d8"
            and float(config.get("learning_rate", config.get("optim_lr"))) == 3e-3
            and config.get("lr_schedule") == "linear"
            and float(config.get("dropout", 0.0)) == 0.0
            and int(config.get("batch_size")) == 64
            and float(config.get("weight_decay")) == 0.1
            and float(config.get("warmup_percent")) == 0.01
        ):
            return run
    raise RuntimeError(f"No finished standard d8 run found with tag {SOURCE_TAG!r}.")


def values_at_progress(run, metrics) -> dict[str, list[float]]:
    points = {metric: [] for metric in metrics}
    for row in run.scan_history(keys=["progress", *metrics]):
        progress = row.get("progress")
        if progress is None:
            continue
        for metric in metrics:
            value = row.get(metric)
            if value is not None:
                points[metric].append((float(progress), float(value)))

    values = {}
    for metric, metric_points in points.items():
        if not metric_points:
            raise RuntimeError(f"{metric!r} was not logged by {run.name!r}.")
        values[metric] = [
            min(metric_points, key=lambda point: abs(point[0] - target))[1]
            for target in TARGET_PROGRESS
        ]
    return values


def main() -> None:
    run = standard_d8_run()
    row_labels = [
        f"L{layer} {label}" for layer in LAYERS for label, _ in COMPONENTS
    ]
    metrics = [
        f"logging/rms/model.layers.{layer}.{component}/{statistic}"
        for statistic in STATISTICS
        for layer in LAYERS
        for _, component in COMPONENTS
    ]
    values = values_at_progress(run, metrics)
    matrices = {}
    for statistic in STATISTICS:
        rows = []
        for layer in LAYERS:
            for _, component in COMPONENTS:
                metric = f"logging/rms/model.layers.{layer}.{component}/{statistic}"
                rows.append(values[metric])
        matrices[statistic] = np.log10(np.asarray(rows))

    color_values = np.concatenate([matrix.ravel() for matrix in matrices.values()])
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 6), constrained_layout=True)
    for ax, statistic in zip(axes, STATISTICS, strict=True):
        image = ax.imshow(
            matrices[statistic],
            aspect="auto",
            cmap="viridis",
            vmin=color_values.min(),
            vmax=color_values.max(),
        )
        ax.set_title(f"{statistic.title()} RMS")
        ax.set_xticks(range(3), ("start", "middle", "end"))
        ax.set_yticks(range(len(row_labels)), row_labels if ax is axes[0] else ())

    fig.colorbar(image, ax=axes, label="log10 RMS")
    fig.suptitle(f"Activation and gradient trajectory: {run.name}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    print(OUT_PATH.resolve())


if __name__ == "__main__":
    main()
