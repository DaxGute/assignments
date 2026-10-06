"""Measure microscopic loss noise in existing part-1 runs."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

from experiments.a1_basics.p5.loss_curve_utils import (
    config_value,
    load_curves,
    smoothness,
)

OUT_PATH = Path(__file__).resolve().parent / "plots" / "a1-p5b-smoothness.png"
P1_TAGS = ("a1-p1-1d-v1", "a1-p1-schedulers-v2")


def is_default(run, ignored=()) -> bool:
    expected = {
        "learning_rate": 0.003,
        "batch_size": 64,
        "warmup_percent": 0.01,
        "weight_decay": 0.1,
    }
    return all(
        key in ignored or config_value(run, key) == value
        for key, value in expected.items()
    )


def scatter_numeric(ax, points, title, xlabel) -> None:
    if not points:
        raise RuntimeError(f"No matching prior runs found for {title}.")
    points = sorted(points)
    ax.scatter(*zip(*points, strict=True), color="#7030A0", s=55)
    ax.plot(*zip(*points, strict=True), color="#999999", alpha=0.5)
    ax.set(title=title, xlabel=xlabel, ylabel="Micro-noise (log-loss MAD)")
    ax.set_xscale("log")
    ax.grid(True, linestyle=":", alpha=0.3)


def scatter_categories(ax, points, title, xlabel) -> None:
    if not points:
        raise RuntimeError(f"No matching prior runs found for {title}.")
    labels = sorted({label for label, _ in points})
    for label, value in points:
        ax.scatter(labels.index(label), value, color="#7030A0", s=55)
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set(title=title, xlabel=xlabel, ylabel="Micro-noise (log-loss MAD)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.3)


def main() -> None:
    curves = load_curves({"tags": {"$in": list(P1_TAGS)}})
    batch, learning_rate, beta1, schedule = [], [], [], []
    for curve in curves:
        run = curve.run
        tags = set(run.tags or ())
        noise = smoothness(curve)
        if "a1-p1-1d-v1" in tags:
            if is_default(run, ignored=("batch_size",)):
                batch.append((int(config_value(run, "batch_size")), noise))
            if is_default(run, ignored=("learning_rate",)):
                learning_rate.append(
                    (float(config_value(run, "learning_rate")), noise)
                )
        if "a1-p1-schedulers-v2" in tags:
            if (
                config_value(run, "lr_schedule") == "cos"
                and config_value(run, "learning_rate") == 0.003
                and config_value(run, "optimizer_name") == "adamw"
            ):
                beta1.append((f"{float(config_value(run, 'beta1')):g}", noise))
            if (
                config_value(run, "learning_rate") == 0.003
                and config_value(run, "batch_size") == 64
                and config_value(run, "beta1") == 0.9
            ):
                schedule.append((str(config_value(run, "lr_schedule")), noise))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    scatter_numeric(axes[0, 0], batch, "Batch size", "Batch size")
    scatter_numeric(axes[0, 1], learning_rate, "Learning rate", "Learning rate")
    scatter_categories(axes[1, 0], beta1, "Momentum", "beta1")
    scatter_categories(axes[1, 1], schedule, "LR schedule", "Schedule")
    fig.suptitle("Which hyperparameters smooth the training loss?")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    print(OUT_PATH.resolve())


if __name__ == "__main__":
    main()
