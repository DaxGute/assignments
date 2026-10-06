"""Plot the four one-at-a-time sweeps side by side."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import wandb
from matplotlib.lines import Line2D

from utils import WANDB_ENTITY, WANDB_PROJECT


EXPERIMENT_KEY = "a1-p1-1d-v1"
SWEEPS = (
    {
        "field": "warmup_percent",
        "label": "Warmup fraction",
        "values": (0.001, 0.003, 0.01, 0.03, 0.1),
        "default": 0.01,
    },
    {
        "field": "learning_rate",
        "label": "Learning rate",
        "values": (3e-4, 1e-3, 3e-3, 1e-2, 3e-2),
        "default": 3e-3,
    },
    {
        "field": "batch_size",
        "label": "Batch size",
        "values": (16, 32, 64, 128, 256),
        "default": 64,
    },
    {
        "field": "weight_decay",
        "label": "Weight decay",
        "values": (0.01, 0.03, 0.1, 0.3, 1.0),
        "default": 0.1,
    },
)

PLOT_DIR = Path(__file__).resolve().parent / "plots"
OUT_PATH = PLOT_DIR / f"{EXPERIMENT_KEY}_loss_summary.png"
PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"

PURPLE = "#7030A0"
LIGHT_BLUE = "#8CD9FF"
RED = "#e74c3c"
DARK = "#111111"
GRAY = "#777777"


@dataclass(frozen=True)
class SweepRun:
    field: str
    value: float
    val_loss: float
    name: str
    url: str


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["P052", "Palatino", "Palatino Linotype", "serif"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "P052",
            "mathtext.it": "P052:italic",
            "mathtext.bf": "P052:bold",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.18,
        }
    )


def finite_float(value) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def same(left, right) -> bool:
    return abs(float(left) - float(right)) <= 1e-12 * max(1.0, abs(float(right)))


def run_created_at(run) -> str:
    return str(getattr(run, "created_at", None) or getattr(run, "createdAt", "") or "")


def created_at_sort_key(created_at: str) -> float:
    if not created_at:
        return 0.0
    try:
        return datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def is_exact_experiment_run(run) -> bool:
    return (run.name or "").endswith(f"-{EXPERIMENT_KEY}")


def config_value(run, field: str) -> float | None:
    keys = (field, "optim_lr") if field == "learning_rate" else (field,)
    for key in keys:
        value = finite_float(run.config.get(key))
        if value is not None:
            return value
    return None


def final_val_loss(run) -> float | None:
    value = finite_float(run.summary.get("val_loss"))
    if value is not None:
        return value

    latest_step = -1
    latest_value = None
    for row in run.scan_history(keys=["optimizer_step", "val_loss"]):
        value = finite_float(row.get("val_loss"))
        if value is None:
            continue
        step = int(row.get("optimizer_step", latest_step + 1))
        if step >= latest_step:
            latest_step = step
            latest_value = value
    return latest_value


def expected_config(sweep, value) -> dict[str, float]:
    expected = {item["field"]: item["default"] for item in SWEEPS}
    expected[sweep["field"]] = value
    return expected


def matches_config(run, expected: dict[str, float]) -> bool:
    for field, value in expected.items():
        actual = config_value(run, field)
        if actual is None or not same(actual, value):
            return False
    return True


def format_value(field: str, value: float) -> str:
    if field == "batch_size":
        return str(int(value))
    if field == "learning_rate":
        mantissa, exponent = f"{value:.0e}".split("e")
        return f"{mantissa}e{int(exponent)}"
    return f"{value:g}"


def matching_runs() -> tuple[dict[str, list[SweepRun]], list[str]]:
    api = wandb.Api()
    runs = api.runs(PROJECT_PATH, filters={"tags": {"$in": [EXPERIMENT_KEY]}})

    finished = []
    pending = []
    for run in runs:
        if EXPERIMENT_KEY not in set(run.tags or []) or not is_exact_experiment_run(run):
            continue
        if run.state != "finished":
            pending.append(f"{run.state}: {run.name}")
            continue
        finished.append(run)

    collected: dict[str, list[SweepRun]] = {}
    missing = []
    for sweep in SWEEPS:
        field = sweep["field"]
        points = []
        for value in sweep["values"]:
            expected = expected_config(sweep, value)
            candidates = [run for run in finished if matches_config(run, expected)]
            if not candidates:
                missing.append(f"{field}={format_value(field, value)}")
                continue
            run = max(candidates, key=lambda item: created_at_sort_key(run_created_at(item)))
            val_loss = final_val_loss(run)
            if val_loss is None:
                missing.append(f"{field}={format_value(field, value)} (no val_loss)")
                continue
            points.append(
                SweepRun(
                    field=field,
                    value=float(value),
                    val_loss=val_loss,
                    name=run.name,
                    url=run.url,
                )
            )
        collected[field] = points
    missing.extend(pending)
    return collected, missing


def plot_sweeps(collected: dict[str, list[SweepRun]]) -> Path:
    if not any(collected.values()):
        raise RuntimeError(
            f"No finished W&B runs found for tag {EXPERIMENT_KEY!r} in {PROJECT_PATH}."
        )

    set_style()
    fig, axes = plt.subplots(1, len(SWEEPS), figsize=(14.2, 3.7), sharey=True)

    legend_handles = [
        Line2D([0], [0], color=GRAY, linestyle="--", linewidth=1.0, label="default"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor=RED,
            markeredgewidth=1.6,
            markersize=9,
            label="best",
        ),
    ]

    for ax, sweep in zip(axes, SWEEPS, strict=True):
        field = sweep["field"]
        points = collected[field]
        title = sweep["label"]
        ax.set_title(title)
        ax.set_xlabel(title)
        ax.set_xscale("log")
        ax.axvline(sweep["default"], color=GRAY, linestyle="--", linewidth=1.0, zorder=2)
        ax.grid(True, which="major", linestyle=":", alpha=0.35)
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())

        if not points:
            ax.text(
                0.5,
                0.5,
                "no finished runs",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=GRAY,
            )
            continue

        values = np.array([point.value for point in points])
        losses = np.array([point.val_loss for point in points])
        best_index = int(np.argmin(losses))
        norm = mcolors.LogNorm(vmin=min(sweep["values"]), vmax=max(sweep["values"]))
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f"{field}_purple_light_blue",
            [PURPLE, LIGHT_BLUE],
        )

        ax.plot(values, losses, color=GRAY, linewidth=1.0, alpha=0.45, zorder=3)
        ax.scatter(
            values,
            losses,
            s=70,
            c=values,
            cmap=cmap,
            norm=norm,
            edgecolor=DARK,
            linewidth=1.0,
            zorder=5,
        )
        ax.scatter(
            [values[best_index]],
            [losses[best_index]],
            s=140,
            facecolors="none",
            edgecolor=RED,
            linewidth=1.8,
            zorder=6,
        )
        ax.set_xticks(list(sweep["values"]))
        ax.set_xticklabels(
            [format_value(field, value) for value in sweep["values"]],
            rotation=25,
            ha="right",
        )
        ax.margins(y=0.18)

        for value, loss in zip(values, losses, strict=True):
            ax.annotate(
                f"{loss:.3f}",
                xy=(value, loss),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=GRAY,
                clip_on=False,
            )

    axes[0].set_ylabel("Final validation loss")
    axes[0].legend(handles=legend_handles, frameon=False, loc="best", fontsize=9)
    fig.suptitle("d8 loss along each hyperparameter", fontsize=14, y=1.04)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH)
    plt.close(fig)
    return OUT_PATH


def main() -> None:
    collected, missing = matching_runs()
    for sweep in SWEEPS:
        for point in collected[sweep["field"]]:
            print(
                f"{point.field}={format_value(point.field, point.value)} "
                f"val_loss={point.val_loss:.6f} name={point.name}"
            )
    if missing:
        print("Missing or unfinished:")
        for item in missing:
            print(f"  {item}")
    print(plot_sweeps(collected).resolve())


if __name__ == "__main__":
    main()
