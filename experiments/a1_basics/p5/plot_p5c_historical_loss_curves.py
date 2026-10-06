"""Find unusually shaped macro and micro loss curves across prior W&B runs."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt

from experiments.a1_basics.p5.loss_curve_utils import (
    load_curves,
    macro_deviation,
    rolling_mean,
    smoothness,
)

OUT_PATH = Path(__file__).resolve().parent / "plots" / "a1-p5c-history-audit.png"
MAX_RUNS = 100
TOP_K = 5


def short_name(curve) -> str:
    name = curve.run.name or curve.run.id
    return name if len(name) <= 42 else f"{name[:39]}..."


def plot_ranked(ax, ranked, metric, title, raw) -> None:
    for score, curve in ranked:
        progress = (curve.steps - curve.steps.min()) / (
            curve.steps.max() - curve.steps.min()
        )
        values = curve.losses if raw else rolling_mean(curve.losses)
        ax.plot(progress, values, linewidth=0.9, label=f"{score:.3f}  {short_name(curve)}")
    ax.set(title=title, xlabel="Training progress", ylabel="Training loss")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.legend(title=metric, frameon=False, fontsize=7)


def main() -> None:
    curves = load_curves(limit=MAX_RUNS)
    if len(curves) < TOP_K:
        raise RuntimeError(f"Need at least {TOP_K} finished historical runs.")

    macro = sorted(
        ((macro_deviation(curve), curve) for curve in curves),
        key=lambda item: item[0],
        reverse=True,
    )[:TOP_K]
    micro = sorted(
        ((smoothness(curve), curve) for curve in curves),
        key=lambda item: item[0],
        reverse=True,
    )[:TOP_K]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    plot_ranked(
        axes[0],
        macro,
        "quadratic-fit RMSE",
        "Largest macro-shape deviations",
        raw=False,
    )
    plot_ranked(
        axes[1],
        micro,
        "log-loss MAD",
        "Noisiest microscopic curves",
        raw=True,
    )
    fig.suptitle(f"Historical loss-curve audit ({len(curves)} recent finished runs)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    print(OUT_PATH.resolve())


if __name__ == "__main__":
    main()
