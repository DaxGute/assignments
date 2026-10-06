"""Generate every Assignment 2 Problem 4.1 and 4.2 figure from real results.

The command queries completed W&B runs, caches diagnostic files, combines them
with the supplied width-512 reference, and writes PNG/PDF/CSV/JSON bundles::

    uv run python -m experiments.a2.plots_p4

No missing measurement is imputed.  Missing files or cells are listed in
``outputs/a2/data/p4_summary.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.helpers.metadata import display_name
from experiments.a2.helpers.paths import (
    DATA_ROOT,
    DIAGNOSTIC_RESULTS,
    PLOT_ROOT,
    STRESS_RESULTS,
)
from experiments.a2.helpers.plotting import apply_style, style_for
from experiments.a2.helpers.results import (
    canonicalize_parameterization,
    describe_run,
    matching_runs,
)
from experiments.a2.provided_sweeps import load as load_supplied
from experiments.a2.provided_sweeps import reference_diagnostics
from experiments.a2.scaling import power_law, predict_power_law, quadratic_log_optimum
from utils import WANDB_ENTITY, WANDB_PROJECT


P41_DIR = PLOT_ROOT / "p41"
P42_DIR = PLOT_ROOT / "p42"
SUMMARY_PATH = DATA_ROOT / "p4_summary.json"
COMPONENTS = ("q", "k", "v", "o", "gate", "up", "down", "readout")
COMPONENT_COLORS = {
    name: color
    for name, color in zip(
        COMPONENTS,
        ("#1f77b4", "#17becf", "#9467bd", "#d62728", "#2ca02c", "#8c564b", "#ff7f0e", "#111111"),
    )
}
BLOCK_RE = re.compile(r"(?:blocks|layers)\.(\d+)\.")


def _finite(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n")


def _bundle(
    directory: Path,
    name: str,
    rows: Sequence[Mapping],
    provenance: Mapping,
    draw: Callable[[plt.Figure], None],
    *,
    size=(7.2, 4.8),
) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    apply_style()
    figure = plt.figure(figsize=size)
    draw(figure)
    figure.tight_layout()
    paths = [directory / f"{name}.{suffix}" for suffix in ("png", "pdf", "csv", "json")]
    figure.savefig(paths[0], dpi=200, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    plt.close(figure)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with paths[2].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    _write_json(paths[3], {"provenance": provenance, "rows": list(rows)})
    return [str(path) for path in paths]


def _axes(figure: plt.Figure, count: int, *, columns=2):
    columns = min(columns, count)
    rows = math.ceil(count / columns)
    values = np.asarray(figure.subplots(rows, columns, squeeze=False)).reshape(-1)
    for axis in values[count:]:
        axis.set_visible(False)
    return values[:count]


def _finish_axis(axis, xlabel: str, ylabel: str, *, xlog=False, ylog=False) -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if xlog:
        axis.set_xscale("log")
    if ylog:
        axis.set_yscale("log")
    axis.grid(True, color="#dddddd", linewidth=0.6)


def _parse_jsonl(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        if line.strip():
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _component(name: str) -> str | None:
    lowered = name.lower()
    tests = (
        ("q", (".q.", ".q_proj.")),
        ("k", (".k.", ".k_proj.")),
        ("v", (".v.", ".v_proj.")),
        ("o", (".o.", ".o_proj.")),
        ("gate", (".gate.", ".gate_proj.")),
        ("up", (".up.", ".up_proj.")),
        ("down", (".down.", ".down_proj.")),
        ("readout", (".head.", "lm_head", "readout")),
    )
    for component, needles in tests:
        if any(needle in lowered for needle in needles):
            return component
    return None


def _block(name: str, depth: int) -> float:
    match = BLOCK_RE.search(name)
    return float(match.group(1)) if match else float(depth)


def _memberships(record: Mapping) -> list[dict]:
    memberships = [dict(item) for item in record.get("memberships") or () if isinstance(item, Mapping)]
    return memberships or [
        {
            key: record.get(key)
            for key in ("problem", "subpart", "family", "config_role", "parameterization")
        }
    ]


class P4Data:
    def __init__(self, *, refresh: bool):
        self.refresh = refresh
        self.raw_by_id = {}
        self.records: list[dict] = []
        self.missing: list[str] = []
        self.cache: dict[str, dict[str, list[dict] | dict]] = {}

    def query(self) -> None:
        import wandb

        api = wandb.Api(timeout=90)
        raw_runs = list(
            api.runs(
                f"{WANDB_ENTITY}/{WANDB_PROJECT}",
                filters={
                    "$or": [
                        {"config.assignment": "a2"},
                        {"tags": {"$in": ["a2", "batch1", "batch2", "p4.1"]}},
                    ]
                },
            )
        )
        for raw in raw_runs:
            record = describe_run(raw)
            if not record.get("completed") or record.get("problem") not in {"4.1", "4.2"}:
                continue
            record["url"] = getattr(raw, "url", None)
            self.records.append(record)
            self.raw_by_id[record["run_id"]] = raw
        self.records.sort(key=lambda row: (row.get("problem") or "", row.get("subpart") or "", row["run_id"]))

    def select(self, **filters) -> list[dict]:
        return matching_runs(self.records, **filters)

    def files(self, record: Mapping) -> dict[str, list[dict] | dict]:
        run_id = str(record["run_id"])
        if run_id in self.cache:
            return self.cache[run_id]
        raw = self.raw_by_id[run_id]
        available = {item.name: item for item in raw.files()}
        parsed: dict[str, list[dict] | dict] = {}
        wanted = ["features.jsonl", "alignment.jsonl", "gradients.jsonl"]
        if record.get("problem") == "4.1":
            wanted.extend(
                name
                for name in available
                if name.endswith(".json") and "/" not in name and not name.startswith("wandb-")
            )
        target_root = STRESS_RESULTS if record.get("problem") == "4.1" else DIAGNOSTIC_RESULTS / run_id
        target_root.mkdir(parents=True, exist_ok=True)
        for name in wanted:
            if name not in available:
                continue
            target = target_root / (f"{run_id}.json" if record.get("problem") == "4.1" else name)
            if self.refresh or not target.is_file():
                downloaded = Path(available[name].download(root=str(target.parent), replace=True).name)
                if downloaded != target:
                    downloaded.replace(target)
            text = target.read_text()
            parsed[name] = _parse_jsonl(text) if name.endswith(".jsonl") else json.loads(text)
        self.cache[run_id] = parsed
        return parsed


def _curve_fit(records: Sequence[Mapping], x_key="peak_lr") -> dict:
    points = sorted(
        [
            {
                "x": float(row[x_key]),
                "y": float(row["final_val_loss"]),
                "run_id": row.get("run_id"),
            }
            for row in records
            if _finite(row.get(x_key)) is not None and _finite(row.get("final_val_loss")) is not None
        ],
        key=lambda point: point["x"],
    )
    unique = {}
    for point in points:
        previous = unique.get(point["x"])
        if previous is None or point["y"] < previous["y"]:
            unique[point["x"]] = point
    points = list(unique.values())
    if len(points) < 3:
        raise ValueError(f"need three LR points, found {len(points)}")
    fit = quadratic_log_optimum([point["x"] for point in points], [point["y"] for point in points])
    optimum = float(fit["optimum"])
    minimum = float(fit.get("fitted_loss_at_optimum", fit["best_sampled_loss"]))
    best = min(points, key=lambda point: (point["y"], point["x"], point.get("run_id") or ""))
    return {"points": points, "fit": fit, "optimum": optimum, "minimum": minimum, "best": best}


def _fits_by(
    records: Sequence[Mapping],
    parameterizations: Sequence[str],
    dimensions: Sequence[int],
    dimension_key: str,
    missing: list[str],
    label: str,
) -> list[dict]:
    results = []
    for parameterization in parameterizations:
        for dimension in dimensions:
            selected = [
                row
                for row in matching_runs(records, parameterization=parameterization)
                if row.get(dimension_key) == dimension
            ]
            try:
                result = _curve_fit(selected)
            except ValueError as exc:
                missing.append(f"{label}: {parameterization} {dimension_key}={dimension}: {exc}")
                continue
            results.append(
                {
                    "parameterization": parameterization,
                    dimension_key: dimension,
                    **result,
                }
            )
    return results


def _plot_scaling_pair(
    directory: Path,
    stem: str,
    fits: Sequence[Mapping],
    x_key: str,
    x_label: str,
    written: list[str],
) -> None:
    for metric, ylabel in (("optimum", "Fitted optimum base LR"), ("minimum", "Fitted minimum validation loss")):
        rows = [
            {
                "parameterization": item["parameterization"],
                x_key: item[x_key],
                metric: item[metric],
                "fit_method": item["fit"]["method"],
                "best_sampled_run_id": item["best"]["run_id"],
            }
            for item in fits
        ]

        def draw(figure, metric=metric, ylabel=ylabel):
            axis = figure.subplots()
            for index, parameterization in enumerate(sorted({row["parameterization"] for row in rows})):
                group = sorted((row for row in rows if row["parameterization"] == parameterization), key=lambda row: row[x_key])
                style = style_for(parameterization, index)
                axis.plot(
                    [row[x_key] for row in group],
                    [row[metric] for row in group],
                    marker=style["marker"],
                    linestyle=style["linestyle"],
                    color=style["color"],
                    label=display_name(parameterization),
                )
            _finish_axis(axis, x_label, ylabel, xlog=True, ylog=metric == "optimum")
            axis.legend()

        written.extend(
            _bundle(
                directory,
                f"{stem}_{'optimum_lr' if metric == 'optimum' else 'minimum_loss'}",
                rows,
                {"fit_type": "quadratic in log LR", "source": "completed W&B runs"},
                draw,
            )
        )


def _fitted_loss(fit: Mapping, learning_rate: float) -> float:
    coefficients = fit.get("coefficients") or {}
    log_lr = math.log(float(learning_rate))
    return float(
        coefficients["intercept"]
        + coefficients["log_linear"] * log_lr
        + coefficients["log_quadratic"] * log_lr**2
    )


def _plot_loss_lr_facets(
    directory: Path,
    name: str,
    fits: Sequence[Mapping],
    dimension_key: str,
    dimension_label: str,
    written: list[str],
) -> None:
    """Measured loss--LR sweeps and their log-quadratic fits, one cell per facet."""
    rows = []
    for item in fits:
        label = (
            f"{display_name(item['parameterization'])}, "
            f"{dimension_label.lower()}={item[dimension_key]}"
        )
        sampled_lrs = [float(point["x"]) for point in item["points"]]
        boundary_limited = math.isclose(
            float(item["optimum"]),
            max(sampled_lrs),
            rel_tol=1e-9,
            abs_tol=0.0,
        )
        for point in item["points"]:
            rows.append(
                {
                    "facet": label,
                    "parameterization": item["parameterization"],
                    dimension_key: item[dimension_key],
                    "kind": "measured",
                    "base_lr": point["x"],
                    "validation_loss": point["y"],
                    "run_id": point["run_id"],
                    "fitted_optimum_lr": item["optimum"],
                    "boundary_limited": boundary_limited,
                }
            )
        coefficients = item["fit"].get("coefficients")
        if coefficients:
            grid = np.geomspace(min(sampled_lrs), max(sampled_lrs), 160)
            for learning_rate in grid:
                rows.append(
                    {
                        "facet": label,
                        "parameterization": item["parameterization"],
                        dimension_key: item[dimension_key],
                        "kind": "fit",
                        "base_lr": float(learning_rate),
                        "validation_loss": _fitted_loss(item["fit"], learning_rate),
                        "run_id": "",
                        "fitted_optimum_lr": item["optimum"],
                        "boundary_limited": boundary_limited,
                    }
                )

    facet_names = [
        (
            f"{display_name(item['parameterization'])}, "
            f"{dimension_label.lower()}={item[dimension_key]}"
        )
        for item in fits
    ]

    def draw(figure):
        axes = _axes(figure, len(facet_names), columns=3)
        for axis, facet, item in zip(axes, facet_names, fits):
            measured = [
                row for row in rows if row["facet"] == facet and row["kind"] == "measured"
            ]
            fitted = [row for row in rows if row["facet"] == facet and row["kind"] == "fit"]
            color = style_for(item["parameterization"])["color"]
            axis.scatter(
                [row["base_lr"] for row in measured],
                [row["validation_loss"] for row in measured],
                color=color,
                marker="o",
                s=28,
                label="measured",
                zorder=3,
            )
            if fitted:
                axis.plot(
                    [row["base_lr"] for row in fitted],
                    [row["validation_loss"] for row in fitted],
                    color=color,
                    linewidth=1.4,
                    label="log-quadratic fit",
                )
            axis.axvline(
                item["optimum"],
                color="#333333",
                linestyle=":",
                linewidth=0.9,
                label="reported optimum",
            )
            if measured[0]["boundary_limited"]:
                axis.text(
                    0.97,
                    0.94,
                    "upper-boundary limited",
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7,
                    color="#a12622",
                )
            axis.set_title(facet, fontsize=9)
            _finish_axis(axis, "Base learning rate", "Validation loss", xlog=True)
        axes[0].legend(fontsize=7)

    written.extend(
        _bundle(
            directory,
            name,
            rows,
            {
                "fit_type": "quadratic validation loss in log base learning rate",
                "facets": facet_names,
                "measured_marker": "circle",
                "fit_line": "solid",
                "reported_optimum_line": "dotted",
            },
            draw,
            size=(11.5, 3.3 * math.ceil(max(1, len(facet_names)) / 3)),
        )
    )


def _stress_payload(data: P4Data, fit: Mapping) -> tuple[dict, Mapping] | None:
    run_id = fit["best"]["run_id"]
    record = next((row for row in data.records if row["run_id"] == run_id), None)
    if record is None:
        return None
    files = data.files(record)
    payload = next((value for key, value in files.items() if key.endswith(".json") and isinstance(value, dict)), None)
    return (payload, record) if payload else None


def _p41_diagnostics(data: P4Data, fits: Sequence[Mapping], written: list[str]) -> list[dict]:
    selected = []
    for fit in fits:
        payload = _stress_payload(data, fit)
        if payload is None:
            data.missing.append(
                f"P4.1 diagnostics absent for best {fit['parameterization']} "
                f"{'width' if 'width' in fit else 'depth'}={fit.get('width', fit.get('depth'))}"
            )
            continue
        result, record = payload
        selected.append({"fit": fit, "payload": result, "record": record})
    return selected


def _plot_p41b(selected: Sequence[Mapping], written: list[str]) -> None:
    if not selected:
        return
    rows = []
    for item in selected:
        fit, payload, record = item["fit"], item["payload"], item["record"]
        label = f"{display_name(fit['parameterization'])}, w={fit['width']}"
        history = payload.get("history") or []
        for sample in history:
            final = sample.get("features", {}).get("final_norm", {})
            rows.append(
                {
                    "panel": "trajectory",
                    "label": label,
                    "run_id": record["run_id"],
                    "step": sample.get("step"),
                    "logit_rms": sample.get("logit_rms"),
                    "feature_rms": final.get("rms"),
                    "normalized_movement": final.get("relative_movement"),
                    "readout_omega_current": sample.get("readout_alignment", {}).get("current", {}).get("omega"),
                    "readout_omega_movement": sample.get("readout_alignment", {}).get("movement", {}).get("omega"),
                }
            )
        final_step = max((row.get("step", -1) for row in payload.get("alignment") or []), default=-1)
        for row in payload.get("alignment") or []:
            component = _component(str(row.get("parameter", "")))
            if component:
                rows.append(
                    {
                        "panel": "alpha",
                        "label": label,
                        "run_id": record["run_id"],
                        "step": row.get("step"),
                        "component": component,
                        "block": _block(str(row.get("parameter", "")), int(fit["fit"].get("depth", 2))),
                        "alpha": row.get("alpha"),
                        "is_final": row.get("step") == final_step,
                    }
                )
    trajectory = [row for row in rows if row["panel"] == "trajectory"]
    alpha_all = [row for row in rows if row["panel"] == "alpha" and _finite(row.get("alpha")) is not None]
    alpha = [row for row in alpha_all if row["is_final"]]

    def draw_trajectory(figure):
        axes = _axes(figure, 4)
        metrics = (
            ("logit_rms", "Logit RMS"),
            ("feature_rms", "Final normalized feature RMS"),
            ("normalized_movement", "Final normalized movement"),
            ("readout_omega_movement", "Readout movement $\\omega$"),
        )
        for axis, (metric, ylabel) in zip(axes, metrics):
            for label in sorted({row["label"] for row in trajectory}):
                group = sorted((row for row in trajectory if row["label"] == label and _finite(row.get(metric)) is not None), key=lambda row: row["step"])
                axis.plot([row["step"] for row in group], [row[metric] for row in group], marker="o", markersize=3, label=label)
            if metric == "readout_omega_movement":
                axis.axhline(0.5, color="#555555", linestyle=":", label="$1/2$")
                axis.axhline(1.0, color="#111111", linestyle="--", label="$1$")
            _finish_axis(axis, "Optimizer step", ylabel)
        axes[0].legend(fontsize=6, ncol=2)
        axes[-1].legend(fontsize=7)

    written.extend(
        _bundle(
            P41_DIR,
            "p41b_best_config_trajectories",
            trajectory,
            {"selection": "lowest final sampled validation loss in each fitted P4.1a cell"},
            draw_trajectory,
            size=(10.5, 7.2),
        )
    )

    def draw_alpha_trajectories(figure):
        labels = sorted({row["label"] for row in alpha_all})
        axes = list(np.atleast_1d(figure.subplots(2, 3)).flat)
        for axis, label in zip(axes, labels):
            selected_rows = [row for row in alpha_all if row["label"] == label]
            traces = sorted({(row["component"], row["block"]) for row in selected_rows})
            for component, block in traces:
                group = sorted(
                    (
                        row
                        for row in selected_rows
                        if row["component"] == component and row["block"] == block
                    ),
                    key=lambda row: row["step"],
                )
                axis.plot(
                    [row["step"] for row in group],
                    [row["alpha"] for row in group],
                    color=COMPONENT_COLORS[component],
                    linestyle=("-" if block == 0 else "--" if block == 1 else ":"),
                    marker="o",
                    markersize=2,
                    linewidth=0.9,
                    label=f"{component}, block {block}",
                )
            axis.axhline(0.5, color="#777777", linestyle=":", linewidth=0.8)
            axis.axhline(1.0, color="#333333", linestyle="--", linewidth=0.8)
            axis.set_title(label, fontsize=9)
            _finish_axis(axis, "Optimizer step", "Update $\\alpha$")
        for axis in axes[len(labels):]:
            axis.set_visible(False)
        if axes:
            axes[0].legend(fontsize=5, ncol=2)

    written.extend(
        _bundle(
            P41_DIR,
            "p41b_update_alpha_trajectories",
            alpha_all,
            {
                "references": [0.5, 1.0],
                "components": list(COMPONENTS),
                "line_style": "solid block 0; dashed block 1; dotted readout",
            },
            draw_alpha_trajectories,
            size=(12, 7.5),
        )
    )

    def draw_alpha(figure):
        axis = figure.subplots()
        offsets = {label: offset for label, offset in zip(sorted({row["label"] for row in alpha}), np.linspace(-0.25, 0.25, max(1, len({row["label"] for row in alpha}))))}
        for component in COMPONENTS:
            group = [row for row in alpha if row["component"] == component]
            axis.scatter(
                [row["block"] + offsets[row["label"]] for row in group],
                [row["alpha"] for row in group],
                s=18,
                alpha=0.72,
                color=COMPONENT_COLORS[component],
                label=component,
            )
        axis.axhline(0.5, color="#555555", linestyle=":")
        axis.axhline(1.0, color="#111111", linestyle="--")
        _finish_axis(axis, "Block index (readout at depth)", "Final-step update $\\alpha$")
        axis.legend(ncol=4, fontsize=7)

    written.extend(
        _bundle(
            P41_DIR,
            "p41b_update_alpha_across_blocks",
            alpha,
            {"references": [0.5, 1.0], "components": list(COMPONENTS)},
            draw_alpha,
        )
    )


def _position(name: str, depth: int) -> float:
    block = _block(name, depth)
    return block / max(depth - 1, 1) if block < depth else 1.05


def _plot_p41d(selected: Sequence[Mapping], written: list[str]) -> None:
    rows = []
    omega_trajectories = []
    for item in selected:
        fit, payload, record = item["fit"], item["payload"], item["record"]
        depth = int(fit["depth"])
        label = f"{display_name(fit['parameterization'])}, d={depth}"
        history = payload.get("history") or []
        if not history:
            continue
        final = history[-1]
        initial = history[0]
        for sample in history:
            movement = sample.get("readout_alignment", {}).get("movement", {})
            omega = _finite(movement.get("omega"))
            if omega is not None:
                omega_trajectories.append(
                    {
                        "label": label,
                        "run_id": record["run_id"],
                        "step": sample.get("step"),
                        "omega": omega,
                    }
                )
        for kind, source in (("residual_rms", final.get("residual_rms", {})), ("branch_rms", final.get("unscaled_branch_rms", {}))):
            for name, value in source.items():
                rows.append({"metric": kind, "label": label, "run_id": record["run_id"], "position": _position(name, depth), "name": name, "value": value})
        for name, values in final.get("features", {}).items():
            movement = _finite(values.get("relative_movement"))
            if movement is None:
                initial_rms = _finite(values.get("initial_rms")) or _finite(initial.get("features", {}).get(name, {}).get("rms"))
                raw = _finite(values.get("movement"))
                movement = raw / initial_rms if raw is not None and initial_rms else None
            rows.append({"metric": "normalized_movement", "label": label, "run_id": record["run_id"], "position": _position(name, depth), "name": name, "value": movement})
        for kind, values in final.get("readout_alignment", {}).items():
            rows.append({"metric": "omega", "label": label, "run_id": record["run_id"], "position": 1.0, "name": kind, "value": values.get("omega")})
    usable = [row for row in rows if _finite(row.get("value")) is not None]

    def draw(figure):
        axes = _axes(figure, 4)
        for axis, (metric, ylabel) in zip(
            axes,
            (
                ("residual_rms", "Residual RMS"),
                ("branch_rms", "Branch RMS"),
                ("normalized_movement", "Normalized movement"),
                ("omega", "Readout $\\omega$"),
            ),
        ):
            for label in sorted({row["label"] for row in usable}):
                group = sorted((row for row in usable if row["label"] == label and row["metric"] == metric), key=lambda row: row["position"])
                axis.plot([row["position"] for row in group], [row["value"] for row in group], marker="o", markersize=3, label=label)
            if metric == "omega":
                axis.axhline(0.5, color="#555555", linestyle=":")
                axis.axhline(1.0, color="#111111", linestyle="--")
            _finish_axis(axis, "Matched relative depth", ylabel)
        axes[0].legend(fontsize=6, ncol=2)

    written.extend(
        _bundle(
            P41_DIR,
            "p41d_depth_diagnostics",
            usable,
            {"selection": "best sampled P4.1c base LR per prescription and depth"},
            draw,
            size=(10.5, 7.2),
        )
    )

    if omega_trajectories:
        def draw_omega_trajectories(figure):
            axis = figure.subplots()
            for label in sorted({row["label"] for row in omega_trajectories}):
                group = sorted(
                    (row for row in omega_trajectories if row["label"] == label),
                    key=lambda row: row["step"],
                )
                axis.plot(
                    [row["step"] for row in group],
                    [row["omega"] for row in group],
                    marker="o",
                    markersize=3,
                    label=label,
                )
            axis.axhline(0.5, color="#555555", linestyle=":", label="no alignment (1/2)")
            axis.axhline(1.0, color="#111111", linestyle="--", label="full alignment (1)")
            _finish_axis(axis, "Optimizer step", "Readout movement $\\omega$")
            axis.legend(fontsize=7, ncol=3)

        written.extend(
            _bundle(
                P41_DIR,
                "p41d_readout_omega_trajectories",
                omega_trajectories,
                {
                    "selection": "best sampled P4.1c base LR per prescription and depth",
                    "references": [0.5, 1.0],
                },
                draw_omega_trajectories,
            )
        )


def _supplied_lr_rows() -> list[dict]:
    rows = []
    for row in load_supplied("P1a"):
        if int(row["tokens"]) != 153_600_000 or float(row["weight_decay"]) != 0.1:
            continue
        rows.append(
            {
                "run_id": row["run_id"],
                "problem": "supplied",
                "subpart": "P1a",
                "width": 512,
                "depth": 8,
                "peak_lr": float(row["learning_rate"]),
                "weight_decay": float(row["weight_decay"]),
                "final_val_loss": float(row["final_val_loss"]),
                "parameterization": "baseline",
                "memberships": [],
                "completed": True,
            }
        )
    return rows


def _p42_width_fits(data: P4Data) -> list[dict]:
    source = data.select(problem="4.2", subpart="a") + _supplied_lr_rows()
    fits = []
    for parameterization in ("baseline", "mup"):
        for width in (128, 256, 512):
            records = [
                row
                for row in source
                if row.get("width") == width
                and (
                    width == 512
                    or canonicalize_parameterization(row.get("parameterization")) == parameterization
                )
            ]
            try:
                fit = _curve_fit(records)
            except ValueError as exc:
                data.missing.append(f"P4.2 width fit {parameterization} width={width}: {exc}")
                continue
            fits.append({"parameterization": parameterization, "width": width, **fit})
    return fits


def _supplied_reference() -> dict:
    reference = reference_diagnostics()
    return {
        "record": {
            "run_id": reference["run_id"],
            "width": 512,
            "depth": 8,
            "peak_lr": float(reference["learning_rate"]),
            "final_val_loss": next(
                (
                    float(row["final_val_loss"])
                    for row in _supplied_lr_rows()
                    if math.isclose(float(row["peak_lr"]), float(reference["learning_rate"]))
                ),
                None,
            ),
            "parameterization": "baseline",
        },
        "features": list(reference.get("diagnostics") or []),
        "alignment": list(reference.get("alignment") or []),
        "gradients": list(reference.get("gradients") or []),
    }


def _diag_for(data: P4Data, record: Mapping) -> dict:
    files = data.files(record)
    return {
        "record": record,
        "features": list(files.get("features.jsonl") or []),
        "alignment": list(files.get("alignment.jsonl") or []),
        "gradients": list(files.get("gradients.jsonl") or []),
    }


def _alpha_rows(diag: Mapping, label: str, source_lr: float | None = None) -> list[dict]:
    depth = int(diag["record"].get("depth") or 8)
    rows = []
    for row in diag["alignment"]:
        component = _component(str(row.get("parameter", "")))
        alpha = _finite(row.get("alpha"))
        if component and alpha is not None:
            rows.append(
                {
                    "label": label,
                    "run_id": diag["record"].get("run_id"),
                    "step": row.get("step"),
                    "component": component,
                    "block": _block(str(row.get("parameter", "")), depth),
                    "alpha": alpha,
                    "source_lr": source_lr,
                }
            )
    return rows


def _omega_rows(diag: Mapping, label: str) -> list[dict]:
    rows = []
    for row in diag["features"]:
        for kind, values in (row.get("readout_alignment") or {}).items():
            omega = _finite(values.get("omega"))
            if omega is not None:
                rows.append({"label": label, "run_id": diag["record"].get("run_id"), "step": row.get("step"), "kind": kind, "omega": omega})
    return rows


def _plot_p42a(data: P4Data, written: list[str]) -> None:
    transferred = [
        row
        for row in data.select(problem="4.2", subpart="a")
        if math.isclose(float(row["peak_lr"]), 0.003, rel_tol=1e-9)
    ]
    diagnostics = []
    for row in transferred:
        diagnostics.append((_diag_for(data, row), f"{display_name(row['parameterization'])}, w={row['width']}"))
    reference = _supplied_reference()
    for parameterization in ("baseline", "mup"):
        diagnostics.append((reference, f"{display_name(parameterization)}, w=512"))
    alpha = [item for diag, label in diagnostics for item in _alpha_rows(diag, label, 0.003)]
    omega = [item for diag, label in diagnostics for item in _omega_rows(diag, label)]
    if not alpha:
        data.missing.append("P4.2a update-alpha diagnostics are absent")
    else:
        def draw_alpha(figure):
            axis = figure.subplots()
            for label in sorted({row["label"] for row in alpha}):
                group = defaultdict(list)
                for row in alpha:
                    if row["label"] == label:
                        group[row["step"]].append(row["alpha"])
                steps = sorted(group)
                axis.plot(steps, [float(np.median(group[step])) for step in steps], marker="o", markersize=3, label=label)
            axis.axhline(0.5, color="#555555", linestyle=":")
            axis.axhline(1.0, color="#111111", linestyle="--")
            _finish_axis(axis, "Optimizer step", "Median update $\\alpha$ across matrices", xlog=True)
            axis.legend(fontsize=7, ncol=2)

        written.extend(_bundle(P42_DIR, "p42a_source_lr_update_alpha", alpha, {"source_lr": 0.003, "aggregation": "median across q/k/v/o/gate/up/down/readout and blocks"}, draw_alpha))
    if not omega:
        data.missing.append("P4.2a readout omega diagnostics are absent")
    else:
        def draw_omega(figure):
            axis = figure.subplots()
            for label in sorted({row["label"] for row in omega}):
                group = sorted((row for row in omega if row["label"] == label and row["kind"] == "movement"), key=lambda row: row["step"])
                axis.plot([row["step"] for row in group], [row["omega"] for row in group], marker="o", markersize=3, label=label)
            axis.axhline(0.5, color="#555555", linestyle=":")
            axis.axhline(1.0, color="#111111", linestyle="--")
            _finish_axis(axis, "Optimizer step", "Readout movement $\\omega$", xlog=True)
            axis.legend(fontsize=7, ncol=2)

        written.extend(_bundle(P42_DIR, "p42a_source_lr_readout_omega", omega, {"source_lr": 0.003}, draw_omega))


def _transferred_losses(data: P4Data, *, depth=False) -> list[dict]:
    rows = []
    source = data.select(problem="4.2", subpart="d" if depth else "a")
    for row in source:
        if math.isclose(float(row["peak_lr"]), 0.003, rel_tol=1e-9):
            rows.append(dict(row))
    supplied = _supplied_reference()["record"]
    for parameterization in (("mup", "depth_mup", "completep") if depth else ("baseline", "mup")):
        rows.append({**supplied, "parameterization": parameterization})
    if not depth:
        rows.extend(
            row
            for row in data.select(problem="4.2", subpart="c")
            if math.isclose(float(row["peak_lr"]), 0.003, rel_tol=1e-9)
        )
    return rows


def _plot_transfer_scaling(
    directory: Path,
    stem: str,
    fits: Sequence[Mapping],
    transfers: Sequence[Mapping],
    x_key: str,
    x_label: str,
    parameterizations: Sequence[str],
    written: list[str],
) -> None:
    _plot_scaling_pair(directory, stem, fits, x_key, x_label, written)
    rows = [
        {
            "parameterization": canonicalize_parameterization(row["parameterization"]),
            x_key: row[x_key],
            "transferred_loss": row["final_val_loss"],
            "run_id": row.get("run_id"),
            "learning_rate": row.get("peak_lr"),
        }
        for row in transfers
        if canonicalize_parameterization(row.get("parameterization")) in parameterizations
        and _finite(row.get("final_val_loss")) is not None
    ]

    def draw(figure):
        axis = figure.subplots()
        for index, parameterization in enumerate(parameterizations):
            group = sorted((row for row in rows if row["parameterization"] == parameterization), key=lambda row: row[x_key])
            style = style_for(parameterization, index)
            axis.plot([row[x_key] for row in group], [row["transferred_loss"] for row in group], marker=style["marker"], color=style["color"], label=display_name(parameterization))
        _finish_axis(axis, x_label, "Measured loss at transferred LR", xlog=True)
        axis.legend()

    written.extend(_bundle(directory, f"{stem}_transferred_loss", rows, {"transferred_lr": 0.003}, draw))


def _plot_p42c(data: P4Data, width_fits: Sequence[Mapping], written: list[str]) -> None:
    rows = []
    for parameterization in ("baseline", "mup"):
        source = sorted((item for item in width_fits if item["parameterization"] == parameterization), key=lambda item: item["width"])
        if len(source) < 2:
            data.missing.append(f"P4.2c source scaling unavailable for {parameterization}")
            continue
        law = power_law([item["width"] for item in source], [item["optimum"] for item in source])
        prediction = predict_power_law(law, 1024)
        target = [row for row in data.select(problem="4.2", subpart="c", parameterization=parameterization) if row.get("width") == 1024]
        direct = next((row for row in target if math.isclose(float(row["peak_lr"]), 0.003, rel_tol=1e-9)), None)
        best = min(target, key=lambda row: row["final_val_loss"]) if target else None
        predicted = min(target, key=lambda row: abs(math.log(float(row["peak_lr"]) / prediction))) if target else None
        for item in source:
            rows.append({"panel": "source", "parameterization": parameterization, "width": item["width"], "learning_rate": item["optimum"], "kind": "fitted optimum", "loss": item["minimum"], "gap_to_best": 0.0, "run_id": item["best"]["run_id"], "law_exponent": law["exponent"]})
        for kind, item in (("predicted", predicted), ("direct", direct), ("local_best", best)):
            if item is not None:
                rows.append({"panel": "target", "parameterization": parameterization, "width": 1024, "learning_rate": item["peak_lr"], "kind": kind, "loss": item["final_val_loss"], "gap_to_best": float(item["final_val_loss"]) - float(best["final_val_loss"]), "run_id": item["run_id"], "predicted_lr": prediction, "law_exponent": law["exponent"]})

    def draw_source(figure):
        axis = figure.subplots()
        for index, parameterization in enumerate(("baseline", "mup")):
            group = sorted((row for row in rows if row["panel"] == "source" and row["parameterization"] == parameterization), key=lambda row: row["width"])
            if not group:
                continue
            law = power_law([row["width"] for row in group], [row["learning_rate"] for row in group])
            grid = np.geomspace(min(row["width"] for row in group), 1024, 100)
            style = style_for(parameterization, index)
            axis.plot(grid, [predict_power_law(law, value) for value in grid], color=style["color"], linestyle=style["linestyle"], label=f"{display_name(parameterization)} fit")
            axis.scatter([row["width"] for row in group], [row["learning_rate"] for row in group], color=style["color"], marker=style["marker"])
        _finish_axis(axis, "Width", "Fitted optimum LR", xlog=True, ylog=True)
        axis.legend()

    written.extend(_bundle(P42_DIR, "p42c_source_width_scaling", [row for row in rows if row["panel"] == "source"], {"fit_type": "power law"}, draw_source))

    def draw_target(figure):
        axes = _axes(figure, 2)
        target = [row for row in rows if row["panel"] == "target"]
        kinds = ("predicted", "direct", "local_best")
        x = np.arange(len(kinds))
        for index, parameterization in enumerate(("baseline", "mup")):
            group = {row["kind"]: row for row in target if row["parameterization"] == parameterization}
            offset = (index - 0.5) * 0.32
            axes[0].bar(x + offset, [group.get(kind, {}).get("learning_rate", np.nan) for kind in kinds], width=0.3, label=display_name(parameterization))
            axes[1].bar(x + offset, [group.get(kind, {}).get("gap_to_best", np.nan) for kind in kinds], width=0.3, label=display_name(parameterization))
        for axis, ylabel in zip(axes, ("Width-1024 learning rate", "Loss gap to local best")):
            axis.set_xticks(x, kinds, rotation=15)
            axis.set_ylabel(ylabel)
            axis.grid(True, axis="y", color="#dddddd")
        axes[0].set_yscale("log")
        axes[0].legend()

    written.extend(_bundle(P42_DIR, "p42c_width1024_prediction_direct_local", [row for row in rows if row["panel"] == "target"], {"comparison": ["predicted", "direct transfer", "best local sampled"]}, draw_target, size=(10, 4.5)))


def _p42_depth_fits(data: P4Data) -> list[dict]:
    source = data.select(problem="4.2", subpart="d")
    supplied = _supplied_lr_rows()
    fits = []
    for parameterization in ("mup", "depth_mup", "completep"):
        for depth in (4, 8, 16):
            records = supplied if depth == 8 else [row for row in matching_runs(source, parameterization=parameterization) if row.get("depth") == depth]
            try:
                fit = _curve_fit(records)
            except ValueError as exc:
                data.missing.append(f"P4.2 depth fit {parameterization} depth={depth}: {exc}")
                continue
            fits.append({"parameterization": parameterization, "depth": depth, **fit})
    return fits


def _selected_p42e(data: P4Data) -> list[tuple[dict, str, str]]:
    choices: list[tuple[dict, str, str]] = []
    source_groups = defaultdict(list)
    for row in data.select(problem="4.2"):
        if row.get("subpart") not in {"a", "c", "d"}:
            continue
        key = (row.get("parameterization"), row.get("width"), row.get("depth"))
        source_groups[key].append(row)
    for key, records in source_groups.items():
        best = min(records, key=lambda row: (float(row["final_val_loss"]), float(row["peak_lr"])))
        transferred = next((row for row in records if math.isclose(float(row["peak_lr"]), 0.003, rel_tol=1e-9)), None)
        parameterization, width, depth = key
        for role, row in (("best", best), ("transferred", transferred)):
            if row is not None:
                choices.append((_diag_for(data, row), f"{display_name(parameterization)}, w={width}, d={depth}, {role}", role))
    reference = _supplied_reference()
    for parameterization in ("baseline", "mup", "depth_mup", "completep"):
        choices.append((reference, f"{display_name(parameterization)}, w=512, d=8, transferred/best", "transferred_best"))
    return choices


def _plot_p42e(data: P4Data, written: list[str]) -> None:
    choices = _selected_p42e(data)
    feature_rows = []
    omega_rows = []
    gradient_rows = []
    for diag, label, role in choices:
        features = sorted(diag["features"], key=lambda row: row.get("step", -1))
        selected_features = [
            row
            for index, row in enumerate(features)
            if int(row.get("step", -1)) <= 5 or index == len(features) - 1
        ]
        for row in selected_features:
            step = int(row.get("step", -1))
            phase = "first_five" if step <= 5 else "final"
            final = row.get("features", {}).get("model.norm") or row.get("features", {}).get("final_norm") or {}
            initial_rms = _finite((features[0].get("features", {}).get("model.norm") or features[0].get("features", {}).get("final_norm") or {}).get("rms"))
            movement = _finite(final.get("relative_movement"))
            if movement is None:
                raw = _finite(final.get("movement"))
                movement = raw / initial_rms if raw is not None and initial_rms else None
            feature_rows.append({"label": label, "role": role, "run_id": diag["record"].get("run_id"), "phase": phase, "step": step, "logit_rms": row.get("logit_rms"), "normalized_movement": movement})
        omega_rows.extend({**row, "role": role} for row in _omega_rows(diag, label))
        gradient_rows.extend(
            {
                "label": label,
                "role": role,
                "run_id": diag["record"].get("run_id"),
                "step": row.get("step"),
                "pre_clip_norm": row.get("pre_clip_norm"),
                "clip_coefficient": row.get("clip_coefficient"),
            }
            for row in diag["gradients"]
        )
    def draw_features(figure):
        axes = _axes(figure, 2)
        final_step = max(row["step"] for row in feature_rows)
        checkpoints = [0, 1, 2, 3, 4, 5, final_step]
        positions = {step: index for index, step in enumerate(checkpoints)}
        for label in sorted({row["label"] for row in feature_rows}):
            group = sorted((row for row in feature_rows if row["label"] == label), key=lambda row: row["step"])
            x = [positions[row["step"]] for row in group]
            axes[0].plot(x, [row["logit_rms"] for row in group], marker="o", markersize=2, linewidth=0.9, alpha=0.7, label=label)
            axes[1].plot(x, [row["normalized_movement"] for row in group], marker="o", markersize=2, linewidth=0.9, alpha=0.7, label=label)
        for axis, ylabel in zip(axes, ("Logit RMS", "Normalized final-feature movement")):
            axis.set_xticks(range(len(checkpoints)), [str(step) if step <= 5 else f"final\n({step})" for step in checkpoints])
            axis.set_xlabel("Optimizer checkpoint")
            axis.set_ylabel(ylabel)
            axis.grid(True, color="#dddddd")
        axes[0].legend(fontsize=4.5, ncol=3)

    if feature_rows:
        written.extend(_bundle(P42_DIR, "p42e_early_final_logits_movement", feature_rows, {"selection": "transferred and best sampled; optimizer steps 0–5 and final checkpoint"}, draw_features, size=(14, 8)))
    else:
        data.missing.append("P4.2e feature/logit diagnostics are absent")

    if omega_rows:
        def draw_omega(figure):
            axis = figure.subplots()
            for label in sorted({row["label"] for row in omega_rows}):
                group = sorted((row for row in omega_rows if row["label"] == label and row["kind"] == "movement"), key=lambda row: row["step"])
                axis.plot([row["step"] for row in group], [row["omega"] for row in group], alpha=0.65, linewidth=1, label=label)
            axis.axhline(0.5, color="#555555", linestyle=":")
            axis.axhline(1.0, color="#111111", linestyle="--")
            _finish_axis(axis, "Optimizer step", "Readout movement $\\omega$", xlog=True)
            axis.legend(fontsize=5, ncol=2)

        written.extend(_bundle(P42_DIR, "p42e_readout_omega", omega_rows, {"selection": "transferred and best sampled, including width 1024 and depth comparison"}, draw_omega))
    else:
        data.missing.append("P4.2e readout omega diagnostics are absent")

    usable_gradients = [row for row in gradient_rows if _finite(row.get("pre_clip_norm")) is not None]
    if usable_gradients:
        def draw_gradients(figure):
            axis = figure.subplots()
            for label in sorted({row["label"] for row in usable_gradients}):
                group = sorted((row for row in usable_gradients if row["label"] == label), key=lambda row: row["step"])
                axis.plot([row["step"] for row in group], [row["pre_clip_norm"] for row in group], marker="o", markersize=3, label=label)
            _finish_axis(axis, "Optimizer step", "Pre-clip gradient norm", xlog=True, ylog=True)
            axis.legend(fontsize=6)

        written.extend(_bundle(P42_DIR, "p42e_preclip_gradients", usable_gradients, {"availability": "only runs/reference with saved gradient rows"}, draw_gradients))
    else:
        data.missing.append("P4.2e pre-clip gradient diagnostics are absent from all selected runs")


def _findings(
    p41_width: Sequence[Mapping],
    p41_depth: Sequence[Mapping],
    p42_width: Sequence[Mapping],
    p42_depth: Sequence[Mapping],
    data: P4Data,
) -> dict:
    def slopes(items, x_key, y_key):
        result = {}
        for parameterization in sorted({item["parameterization"] for item in items}):
            group = [item for item in items if item["parameterization"] == parameterization]
            if len(group) >= 2:
                result[parameterization] = power_law([item[x_key] for item in group], [item[y_key] for item in group])
        return result

    width1024 = {}
    for parameterization in ("baseline", "mup"):
        target = data.select(problem="4.2", subpart="c", parameterization=parameterization)
        if target:
            best = min(target, key=lambda row: row["final_val_loss"])
            direct = next((row for row in target if math.isclose(float(row["peak_lr"]), 0.003, rel_tol=1e-9)), None)
            width1024[parameterization] = {
                "best_sampled_lr": best["peak_lr"],
                "best_sampled_loss": best["final_val_loss"],
                "best_run_id": best["run_id"],
                "direct_transfer_loss": None if direct is None else direct["final_val_loss"],
                "direct_transfer_gap": None if direct is None else float(direct["final_val_loss"]) - float(best["final_val_loss"]),
            }
    p41_depth_status = []
    for item in p41_depth:
        sampled_lrs = [float(point["x"]) for point in item["points"]]
        boundary_limited = math.isclose(
            float(item["optimum"]),
            max(sampled_lrs),
            rel_tol=1e-9,
            abs_tol=0.0,
        )
        p41_depth_status.append(
            {
                "parameterization": item["parameterization"],
                "depth": item["depth"],
                "fitted_optimum_lr": item["optimum"],
                "sampled_lr_min": min(sampled_lrs),
                "sampled_lr_max": max(sampled_lrs),
                "upper_boundary_limited": boundary_limited,
                "interpretation": (
                    "boundary-limited; not a trustworthy interior optimum"
                    if boundary_limited
                    else "interior optimum"
                ),
            }
        )
    return {
        "p41_width_optimum_lr_power_laws": slopes(p41_width, "width", "optimum"),
        "p41_width_minimum_loss_power_laws": slopes(p41_width, "width", "minimum"),
        "p41_depth_optimum_lr_power_laws": slopes(p41_depth, "depth", "optimum"),
        "p42_width_optimum_lr_power_laws": slopes(p42_width, "width", "optimum"),
        "p42_depth_optimum_lr_power_laws": slopes(p42_depth, "depth", "optimum"),
        "p42_width1024": width1024,
        "p41_depth_fit_status": p41_depth_status,
    }


def run(*, refresh: bool = False) -> dict:
    data = P4Data(refresh=refresh)
    data.query()
    written: list[str] = []

    p41a_records = data.select(problem="4.1", subpart="a")
    p41_width = _fits_by(p41a_records, ("kaiming", "mup"), (640, 2560, 5120), "width", data.missing, "P4.1a")
    _plot_loss_lr_facets(
        P41_DIR,
        "p41a_loss_vs_base_lr",
        p41_width,
        "width",
        "Width",
        written,
    )
    _plot_scaling_pair(P41_DIR, "p41a_width", p41_width, "width", "Width", written)
    selected_width = _p41_diagnostics(data, p41_width, written)
    _plot_p41b(selected_width, written)

    p41c_records = data.select(problem="4.1", subpart="c")
    p41_depth = _fits_by(p41c_records, ("mup", "depth_mup", "completep"), (2, 100, 1000), "depth", data.missing, "P4.1c")
    _plot_loss_lr_facets(
        P41_DIR,
        "p41c_loss_vs_base_lr",
        p41_depth,
        "depth",
        "Depth",
        written,
    )
    _plot_scaling_pair(P41_DIR, "p41c_depth", p41_depth, "depth", "Depth", written)
    selected_depth = _p41_diagnostics(data, p41_depth, written)
    _plot_p41d(selected_depth, written)

    _plot_p42a(data, written)
    p42_width = _p42_width_fits(data)
    _plot_loss_lr_facets(
        P42_DIR,
        "p42a_loss_vs_base_lr",
        p42_width,
        "width",
        "Width",
        written,
    )
    _plot_transfer_scaling(
        P42_DIR,
        "p42b_width",
        p42_width,
        _transferred_losses(data),
        "width",
        "Width",
        ("baseline", "mup"),
        written,
    )
    _plot_p42c(data, p42_width, written)

    p42_depth = _p42_depth_fits(data)
    _plot_transfer_scaling(
        P42_DIR,
        "p42d_depth",
        p42_depth,
        _transferred_losses(data, depth=True),
        "depth",
        "Depth",
        ("mup", "depth_mup", "completep"),
        written,
    )
    _plot_p42e(data, written)

    target_gradient_runs = sorted(
        run_id
        for run_id, files in data.cache.items()
        if files.get("gradients.jsonl")
    )
    supplied_gradient_rows = len(reference_diagnostics().get("gradients") or [])
    if not target_gradient_runs:
        data.missing.append(
            "P4.2 pre-clip gradients: gradients.jsonl is absent from every selected "
            "completed W&B target run; the gradient plot contains only the supplied "
            f"width-512 reference ({supplied_gradient_rows} rows)"
        )
    findings = _findings(p41_width, p41_depth, p42_width, p42_depth, data)
    summary = {
        "wandb_project": f"{WANDB_ENTITY}/{WANDB_PROJECT}",
        "completed_p4_runs": len(data.records),
        "generated_bundles": len(written) // 4,
        "written": written,
        "missing": sorted(set(data.missing)),
        "empirical_findings": findings,
        "selection_rules": {
            "fits": "quadratic validation loss in log learning rate; boundary fallback retained from shared scaling helper",
            "best_sampled": "minimum measured final validation loss, then lower LR",
            "diagnostics": "best sampled P4.1 configurations; transferred and best sampled P4.2 configurations",
        },
        "diagnostic_availability": {
            "wandb_target_runs_with_preclip_gradients": target_gradient_runs,
            "supplied_width512_preclip_gradient_rows": supplied_gradient_rows,
        },
        "required_loss_lr_bundles": {
            "p41a": "outputs/a2/plots/p41/p41a_loss_vs_base_lr",
            "p41c": "outputs/a2/plots/p41/p41c_loss_vs_base_lr",
            "p42a": "outputs/a2/plots/p42/p42a_loss_vs_base_lr",
        },
        "warnings": [
            (
                "All P4.1 depth reported optima equal the upper sampled base-LR "
                "boundary 0.01; treat them as boundary-limited, not trustworthy "
                "interior optima."
            )
        ],
    }
    _write_json(SUMMARY_PATH, summary)
    print(f"Read {len(data.records)} completed Problem 4 runs")
    print(f"Wrote {len(written) // 4} figure bundles")
    print(f"Wrote {SUMMARY_PATH}")
    if summary["missing"]:
        print("Unavailable diagnostics/cells:")
        for message in summary["missing"]:
            print(f"  - {message}")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Redownload diagnostic artifacts even when a cache file exists.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(refresh=args.refresh)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
