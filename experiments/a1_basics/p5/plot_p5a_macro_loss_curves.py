"""Compare macroscopic loss-curve shapes across five optimizer settings."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

from experiments.a1_basics.p5.loss_curve_utils import (
    PROJECT_PATH,
    config_value,
    load_curves,
    rolling_mean,
)

EXPERIMENT_KEY = "a1-p5a-macro-v1"
OUT_PATH = Path(__file__).resolve().parent / "plots" / f"{EXPERIMENT_KEY}.png"


def label(curve) -> str:
    run = curve.run
    lr = float(config_value(run, "learning_rate", 0.003))
    beta1 = float(config_value(run, "beta1", 0.9))
    batch = int(config_value(run, "batch_size", 64))
    schedule = config_value(run, "lr_schedule", "linear")
    if lr != 0.003:
        return f"learning rate = {lr:g}"
    if beta1 != 0.9:
        return f"beta1 = {beta1:g}"
    if batch != 64:
        return f"batch size = {batch}"
    if schedule != "linear":
        return f"schedule = {schedule}"
    return "baseline"


def main() -> None:
    curves = load_curves({"tags": {"$in": [EXPERIMENT_KEY]}})
    latest = {}
    for curve in curves:
        latest.setdefault(label(curve), curve)
    if len(latest) != 5:
        raise RuntimeError(
            f"Expected five finished {EXPERIMENT_KEY} curves in {PROJECT_PATH}; "
            f"found {sorted(latest)}."
        )

    order = (
        "baseline",
        "learning rate = 0.01",
        "beta1 = 0.8",
        "batch size = 128",
        "schedule = constant",
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, name in zip(axes.flat, order, strict=False):
        curve = latest[name]
        progress = (curve.steps - curve.steps.min()) / (
            curve.steps.max() - curve.steps.min()
        )
        ax.plot(progress, curve.losses, color="#8CD9FF", alpha=0.25, linewidth=0.6)
        ax.plot(progress, rolling_mean(curve.losses), color="#7030A0", linewidth=1.5)
        ax.set(title=name, xlabel="Training progress", ylabel="Training loss")
        ax.grid(True, linestyle=":", alpha=0.3)
    axes.flat[-1].remove()

    fig.suptitle("Macro loss curves: raw loss and rolling mean")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    print(OUT_PATH.resolve())


if __name__ == "__main__":
    main()
