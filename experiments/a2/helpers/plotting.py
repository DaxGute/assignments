"""Figure styling and save helpers. Problem-specific analysis stays in plots/."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from experiments.a2.helpers.fitting import (
    bivariate_loss,
    log_quadratic_loss,
    power_law,
    predict_power_law,
    quadratic_log_optimum,
    r_squared,
)
from experiments.a2.helpers.metadata import display_name
from experiments.a2.helpers.results import LOSS_ATOL


class MissingData(RuntimeError):
    """A figure cannot be drawn because required completed results are absent."""


@dataclass
class PlotReport:
    name: str
    written: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add_written(self, paths) -> None:
        self.written.extend(str(path) for path in paths)

    def add_missing(self, message: str) -> None:
        self.missing.append(message)

    def add_note(self, message: str) -> None:
        self.notes.append(message)


SERIES_COLORS = ("#1f4e79", "#c0392b", "#0e7c66", "#b86e00", "#6c3483", "#117a65")
SERIES_MARKERS = ("o", "s", "D", "^", "v", "P")
PARAMETERIZATION_STYLE = {
    "baseline": {"color": "#222222", "marker": "o", "linestyle": "-"},
    "kaiming": {"color": "#c0392b", "marker": "s", "linestyle": "--"},
    "mup": {"color": "#1f4e79", "marker": "D", "linestyle": "-"},
    "depth_mup": {"color": "#0e7c66", "marker": "^", "linestyle": "-."},
    "completep": {"color": "#b86e00", "marker": "v", "linestyle": ":"},
    "adamw": {"color": "#1f4e79", "marker": "o", "linestyle": "-"},
    "adamh": {"color": "#6c3483", "marker": "P", "linestyle": "--"},
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["P052", "Palatino", "Palatino Linotype", "DejaVu Serif", "serif"],
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "legend.frameon": False,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
            "axes.axisbelow": True,
        }
    )


def format_tokens(tokens: int) -> str:
    tokens = int(tokens)
    if tokens >= 1_000_000_000:
        text = f"{tokens / 1_000_000_000:.4f}".rstrip("0").rstrip(".")
        return f"{text}B"
    if tokens >= 1_000_000:
        text = f"{tokens / 1_000_000:.4f}".rstrip("0").rstrip(".")
        return f"{text}M"
    return f"{tokens:g}"


def style_for(name: str, index: int = 0) -> dict:
    canonical = str(name)
    if canonical in PARAMETERIZATION_STYLE:
        style = dict(PARAMETERIZATION_STYLE[canonical])
        style["label"] = display_name(canonical)
        return style
    style = {
        "color": SERIES_COLORS[index % len(SERIES_COLORS)],
        "marker": SERIES_MARKERS[index % len(SERIES_MARKERS)],
        "linestyle": "-",
        "label": str(name),
    }
    return style


def place_legend(axis, ncol=2, extra=()) -> None:
    handles, labels = axis.get_legend_handles_labels()
    handles.extend(extra)
    labels.extend(handle.get_label() for handle in extra)
    axis.legend(
        handles,
        labels,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=ncol,
        frameon=False,
    )


def prepare_axis(ax, xlabel, ylabel, *, xlog=False, ylog=False) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.grid(True, which="major", color="#dddddd", linewidth=0.6)


def unique_points(records, x_key, y_key="final_val_loss"):
    grouped = {}
    identities = {}
    for record in records:
        x_value = float(record[x_key])
        y_value = float(record[y_key])
        grouped.setdefault(x_value, []).append(y_value)
        identities.setdefault(x_value, []).append(str(record.get("run_id") or record.get("name") or ""))
    points = []
    for x_value, losses in sorted(grouped.items()):
        if max(losses) - min(losses) > LOSS_ATOL:
            rendered = ", ".join(
                f"{run_id or '(no id)'}={loss:.6g}"
                for run_id, loss in zip(identities[x_value], losses)
            )
            raise MissingData(
                f"Completed runs disagree at {x_key}={x_value:g} by more than {LOSS_ATOL}: {rendered}"
            )
        points.append({"x": x_value, "y": min(losses), "run_id": identities[x_value][0]})
    return points


def fit_log_quadratic(points):
    if len(points) < 3:
        raise MissingData("a quadratic fit needs at least three measured points")
    return quadratic_log_optimum([point["x"] for point in points], [point["y"] for point in points])


def fit_power(points):
    if len(points) < 2:
        raise MissingData("a power law needs at least two points")
    return power_law([point["x"] for point in points], [point["y"] for point in points])


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, (np.floating, np.integer)):
        return _json_ready(value.item())
    return value


def save_bundle(path: Path, figure, rows, provenance) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep dots inside names such as 153.6M; do not treat them as file suffixes.
    png = path.parent / f"{path.name}.png"
    pdf = path.parent / f"{path.name}.pdf"
    csv_path = path.parent / f"{path.name}.csv"
    json_path = path.parent / f"{path.name}.json"
    figure.savefig(png)
    figure.savefig(pdf)
    plt.close(figure)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    json_path.write_text(json.dumps(_json_ready(provenance), indent=2) + "\n")
    return [png, pdf, csv_path, json_path]


def loss_curve_figure(series, *, xlabel, ylabel, path: Path, title=None):
    """Measured points, the existing log-quadratic fit, and its optimum."""
    apply_style()
    figure, axis = plt.subplots(figsize=(6.4, 4.3))
    rows = []
    fits = []
    for index, item in enumerate(series):
        points = unique_points(item["records"], item.get("x_key", "peak_lr"), item.get("y_key", "final_val_loss"))
        if len(points) < 2:
            raise MissingData(f"{item['label']} has fewer than two measured points")
        style = style_for(item.get("style") or item["label"], index)
        axis.plot(
            [point["x"] for point in points],
            [point["y"] for point in points],
            linestyle="none",
            marker=style["marker"],
            color=style["color"],
            markersize=6,
            label=item["label"],
            zorder=3,
        )
        for point in points:
            rows.append(
                {
                    "series": item["label"],
                    "kind": "measured",
                    "x": point["x"],
                    "y": point["y"],
                    "run_id": point["run_id"],
                }
            )
        if len(points) >= 3:
            fit = fit_log_quadratic(points)
            fits.append({"series": item["label"], **fit})
            grid = np.geomspace(min(point["x"] for point in points), max(point["x"] for point in points), 200)
            curve = [log_quadratic_loss(fit, value) for value in grid]
            axis.plot(grid, curve, color=style["color"], linewidth=1.3, zorder=2)
            for x_value, y_value in zip(grid[::10], curve[::10]):
                rows.append({"series": item["label"], "kind": "fit", "x": float(x_value), "y": float(y_value)})
            optimum = float(fit["optimum"])
            optimum_loss = (
                float(fit["fitted_loss_at_optimum"])
                if fit["method"] == "log_quadratic_vertex"
                else float(fit["best_sampled_loss"])
            )
            axis.scatter(
                [optimum],
                [optimum_loss],
                marker="X",
                s=46,
                color=style["color"],
                linewidths=0.6,
                edgecolors="white",
                zorder=4,
                label="fitted optimum" if index == 0 else "_nolegend_",
            )
            rows.append(
                {
                    "series": item["label"],
                    "kind": "optimum",
                    "x": optimum,
                    "y": optimum_loss,
                    "method": fit["method"],
                }
            )
    prepare_axis(axis, xlabel, ylabel, xlog=True, ylog=False)
    if title:
        axis.set_title(title, fontsize=11, pad=8)
    place_legend(axis, ncol=2)
    provenance = {"fit_type": "log_quadratic", "fits": fits}
    return save_bundle(path, figure, rows, provenance), fits


def scaling_figure(series, *, xlabel, ylabel, path: Path, predictions=None, overlay_laws=None, xlog=True, ylog=True):
    """Measured optima and a power-law fit from ``power_law``."""
    apply_style()
    figure, axis = plt.subplots(figsize=(6.4, 4.3))
    rows = []
    fits = []
    predictions = predictions or []
    overlay_laws = overlay_laws or []
    for index, item in enumerate(series):
        points = sorted(item["points"], key=lambda point: float(point["x"]))
        style = style_for(item.get("style") or item["label"], index)
        axis.plot(
            [point["x"] for point in points],
            [point["y"] for point in points],
            linestyle="none",
            marker=style["marker"],
            color=style["color"],
            markersize=7,
            label=item["label"],
            zorder=3,
        )
        for point in points:
            rows.append({"series": item["label"], "kind": point.get("kind", "measured"), "x": point["x"], "y": point["y"], **{k: point.get(k, "") for k in ("run_id", "note") if k in point or True}})
        if item.get("fit", True) and len(points) >= 2 and all(point["x"] > 0 and point["y"] > 0 for point in points):
            law = fit_power(points)
            fits.append({"series": item["label"], **law})
            grid = np.geomspace(min(point["x"] for point in points), max(point["x"] for point in points) * (1.8 if predictions else 1), 100)
            curve = [predict_power_law(law, value) for value in grid]
            score = r_squared(law)
            axis.plot(
                grid,
                curve,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.3,
                label=f"{item['label']} fit, $R^2$={score:.3f}",
                zorder=2,
            )
            for x_value, y_value in zip(grid[::8], curve[::8]):
                rows.append({"series": item["label"], "kind": "fit", "x": float(x_value), "y": float(y_value), "r_squared": score})
    for law_item in overlay_laws:
        law = law_item["law"]
        grid = np.geomspace(law_item["xmin"], law_item["xmax"], 100)
        curve = [predict_power_law(law, value) for value in grid]
        score = r_squared(law)
        axis.plot(
            grid,
            curve,
            color=law_item.get("color", "#444444"),
            linestyle=law_item.get("linestyle", "--"),
            linewidth=1.3,
            label=f"{law_item['label']}, $R^2$={score:.3f}",
        )
        for x_value, y_value in zip(grid[::8], curve[::8]):
            rows.append({"series": law_item["label"], "kind": "fit", "x": float(x_value), "y": float(y_value), "r_squared": score})
    for prediction in predictions:
        axis.scatter(
            [prediction["x"]],
            [prediction["y"]],
            marker="*",
            s=90,
            color="#111111",
            zorder=5,
            label=prediction["label"],
        )
        rows.append({"series": prediction["label"], "kind": "prediction", "x": prediction["x"], "y": prediction["y"]})
    prepare_axis(axis, xlabel, ylabel, xlog=xlog, ylog=ylog)
    place_legend(axis, ncol=2)
    return save_bundle(path, figure, rows, {"fit_type": "power_law", "fits": fits, "predictions": predictions})


def mark_reference(axis, value, label, *, orientation="vertical"):
    if orientation == "vertical":
        axis.axvline(value, color="#444444", linewidth=0.9, linestyle=":", label=label, zorder=1)
    else:
        axis.axhline(value, color="#444444", linewidth=0.9, linestyle=":", label=label, zorder=1)


def legend_proxy(label, **style):
    return Line2D([0], [0], label=label, **style)


def contour_figure(rows_xy, fit, *, path: Path, product=None, xlabel="Peak learning rate", ylabel="Weight decay"):
    apply_style()
    figure, axis = plt.subplots(figsize=(5.6, 4.8))
    learning_rates = np.asarray([row["learning_rate"] for row in rows_xy], dtype=float)
    weight_decays = np.asarray([row["weight_decay"] for row in rows_xy], dtype=float)
    log_lr = np.linspace(np.log(learning_rates.min()), np.log(learning_rates.max()), 80)
    log_wd = np.linspace(np.log(weight_decays.min()), np.log(weight_decays.max()), 80)
    grid_lr, grid_wd = np.meshgrid(log_lr, log_wd)
    surface = np.vectorize(lambda x, y: bivariate_loss(fit, math.exp(x), math.exp(y)))(grid_lr, grid_wd)
    image = axis.contourf(np.exp(grid_lr), np.exp(grid_wd), surface, levels=12, cmap="viridis")
    figure.colorbar(image, ax=axis, label="Fitted validation loss")
    axis.scatter(
        learning_rates,
        weight_decays,
        s=28,
        facecolors="white",
        edgecolors="black",
        linewidths=0.7,
        zorder=3,
        label="measured",
    )
    axis.scatter(
        [fit["optimum_lr"]],
        [fit["optimum_wd"]],
        marker="*",
        s=110,
        color="#111111",
        zorder=4,
        label="fitted optimum",
    )
    table = [
        {
            "kind": "measured",
            "learning_rate": float(row["learning_rate"]),
            "weight_decay": float(row["weight_decay"]),
            "final_val_loss": float(row["final_val_loss"]),
            "run_id": row.get("run_id", ""),
        }
        for row in rows_xy
    ]
    table.append(
        {
            "kind": "optimum",
            "learning_rate": float(fit["optimum_lr"]),
            "weight_decay": float(fit["optimum_wd"]),
            "final_val_loss": "",
            "run_id": fit.get("best_sampled_run_id", ""),
        }
    )
    proxies = []
    if product is not None and product > 0:
        line_lr = np.geomspace(learning_rates.min(), learning_rates.max(), 100)
        line_wd = product / line_lr
        mask = (line_wd >= weight_decays.min()) & (line_wd <= weight_decays.max())
        axis.plot(line_lr[mask], line_wd[mask], color="white", linewidth=1.2, linestyle="--", label="_nolegend_")
        proxies.append(legend_proxy("constant product", color="#222222", linewidth=1.2, linestyle="--"))
        for x_value, y_value in zip(line_lr[::12], line_wd[::12]):
            table.append({"kind": "product_line", "learning_rate": float(x_value), "weight_decay": float(y_value), "final_val_loss": "", "run_id": ""})
    prepare_axis(axis, xlabel, ylabel, xlog=True, ylog=True)
    place_legend(axis, ncol=2, extra=proxies)
    provenance = {
        "fit_type": "bivariate_log_quadratic",
        "coefficients": fit.get("coefficients"),
        "method": fit.get("method"),
        "optimum_lr": fit.get("optimum_lr"),
        "optimum_wd": fit.get("optimum_wd"),
        "optimum_product": fit.get("optimum_product"),
        "source_runs": [row.get("run_id") for row in rows_xy],
    }
    return save_bundle(path, figure, table, provenance)
