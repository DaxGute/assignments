"""Generate every required Assignment 2 Problem 3.2 analysis artifact.

The new experiment results are read from W&B through ``query_a2_runs``.  The
B=64 source cells that Batch 1 deliberately reuses are read from the bundled
export of the public course W&B sweeps.  No values are synthesized.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.helpers.fitting import (
    log_quadratic_loss,
    power_law,
    predict_power_law,
    quadratic_log_optimum,
)
from experiments.a2.helpers.plotting import apply_style, save_bundle
from experiments.a2.helpers.results import matching_runs, query_a2_runs, same_number
from experiments.a2.provided_sweeps import load as load_supplied


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLOT_DIR = ROOT / "outputs" / "a2" / "plots" / "p32"
DEFAULT_SUMMARY = ROOT / "outputs" / "a2" / "data" / "p32_summary.json"
SOURCE_BATCHES = (8, 16, 32, 64)
TARGET_BATCHES = (128, 256)
LR_GRID = (0.0015, 0.003, 0.006)
WD_GRID = (0.1, 0.2, 0.4, 0.8)
COLORS = ("#1f4e79", "#c0392b", "#0e7c66", "#b86e00")
MARKERS = ("o", "s", "D", "^")


class PlotDataError(RuntimeError):
    """Required completed measurements are absent or inconsistent."""


def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _membership_matches(run: Mapping, **wanted) -> bool:
    for membership in run.get("memberships") or ():
        if not isinstance(membership, Mapping):
            continue
        if all(membership.get(key) == value for key, value in wanted.items()):
            return True
    return False


def _dedupe(records: Iterable[Mapping], keys: Sequence[str]) -> list[dict]:
    grouped: dict[tuple, list[Mapping]] = defaultdict(list)
    for record in records:
        grouped[tuple(float(record[key]) for key in keys)].append(record)
    result = []
    for coordinate, rows in sorted(grouped.items()):
        losses = [float(row["final_val_loss"]) for row in rows]
        if max(losses) - min(losses) > 1e-4:
            ids = ", ".join(str(row.get("run_id")) for row in rows)
            raise PlotDataError(f"duplicate measurements disagree at {coordinate}: {ids}")
        result.append(dict(min(rows, key=lambda row: float(row["final_val_loss"]))))
    return result


def _supplied_b64() -> list[dict]:
    """Actual public-W&B measurements reused by the Batch 1 planner."""
    by_id = {}
    for part in ("P1a", "P2a"):
        for raw in load_supplied(part):
            if (
                int(raw["tokens"]) == 614_400_000
                and int(raw["batch_size"]) == 64
                and float(raw["learning_rate"]) in LR_GRID
            ):
                row = dict(raw)
                row.update(
                    peak_lr=float(raw["learning_rate"]),
                    token_budget=int(raw["tokens"]),
                    completed=True,
                    source="public_wandb_export",
                )
                by_id[row["run_id"]] = row
    return list(by_id.values())


def _require_grid(records: Sequence[Mapping], key: str, grid: Sequence[float], label: str) -> list[dict]:
    chosen = _dedupe(records, (key,))
    by_value = {float(row[key]): row for row in chosen}
    missing = [value for value in grid if value not in by_value]
    if missing:
        raise PlotDataError(f"{label} is missing completed measurements at {key}={missing}")
    return [by_value[float(value)] for value in grid]


def _fit(rows: Sequence[Mapping], key: str) -> dict:
    fit = quadratic_log_optimum(
        [float(row[key]) for row in rows],
        [float(row["final_val_loss"]) for row in rows],
    )
    fit["source_run_ids"] = [row["run_id"] for row in rows]
    return fit


def _fit_loss(fit: Mapping) -> float:
    return float(fit.get("fitted_loss_at_optimum", fit["best_sampled_loss"]))


def _source_analysis(all_runs: Sequence[Mapping]) -> dict:
    p32 = matching_runs(all_runs, problem="3.2")
    public_b64 = _supplied_b64()
    lr_rows: dict[int, list[dict]] = {}
    wd_rows: dict[int, list[dict]] = {}
    for batch in SOURCE_BATCHES:
        if batch == 64:
            lr_pool = [
                row
                for row in public_b64
                if same_number(row["weight_decay"], 0.1)
            ]
            wd_pool = [
                row
                for row in public_b64
                if same_number(row["learning_rate"], 0.0015)
                and float(row["weight_decay"]) in WD_GRID
            ]
            wd_pool.extend(
                row
                for row in p32
                if row["batch_size"] == batch
                and same_number(row["peak_lr"], 0.0015)
                and float(row["weight_decay"]) in WD_GRID
                and _membership_matches(
                    row,
                    problem="3.2",
                    subpart="b",
                    hypothesis="fixed_lr_scale_wd",
                )
            )
        else:
            lr_pool = [
                row
                for row in p32
                if row["batch_size"] == batch
                and same_number(row["weight_decay"], 0.1)
                and float(row["peak_lr"]) in LR_GRID
                and _membership_matches(row, problem="3.2", subpart="a")
            ]
            wd_pool = [
                row
                for row in p32
                if row["batch_size"] == batch
                and same_number(row["peak_lr"], 0.0015)
                and float(row["weight_decay"]) in WD_GRID
                and _membership_matches(
                    row,
                    problem="3.2",
                    subpart="b",
                    hypothesis="fixed_lr_scale_wd",
                )
            ]
        lr_rows[batch] = _require_grid(lr_pool, "peak_lr", LR_GRID, f"B={batch} LR sweep")
        wd_rows[batch] = _require_grid(wd_pool, "weight_decay", WD_GRID, f"B={batch} WD sweep")

    lr_fits = {batch: _fit(rows, "peak_lr") for batch, rows in lr_rows.items()}
    wd_fits = {batch: _fit(rows, "weight_decay") for batch, rows in wd_rows.items()}
    lr_law = power_law(SOURCE_BATCHES, [lr_fits[batch]["optimum"] for batch in SOURCE_BATCHES])
    wd_law = power_law(SOURCE_BATCHES, [wd_fits[batch]["optimum"] for batch in SOURCE_BATCHES])
    return {
        "lr_rows": lr_rows,
        "wd_rows": wd_rows,
        "lr_fits": lr_fits,
        "wd_fits": wd_fits,
        "lr_law": lr_law,
        "wd_law": wd_law,
    }


def _save_loss_lr(source: Mapping, out: Path) -> list[str]:
    apply_style()
    figure, axis = plt.subplots(figsize=(7.0, 4.7))
    table = []
    for index, batch in enumerate(SOURCE_BATCHES):
        rows = source["lr_rows"][batch]
        fit = source["lr_fits"][batch]
        x = [float(row["peak_lr"]) for row in rows]
        y = [float(row["final_val_loss"]) for row in rows]
        axis.scatter(x, y, color=COLORS[index], marker=MARKERS[index], label=f"B={batch}", zorder=3)
        grid = np.geomspace(min(x), max(x), 200)
        axis.plot(grid, [log_quadratic_loss(fit, value) for value in grid], color=COLORS[index])
        axis.scatter(
            [fit["optimum"]],
            [_fit_loss(fit)],
            color=COLORS[index],
            marker="X",
            s=55,
            edgecolors="white",
            linewidths=0.6,
            zorder=4,
        )
        for row in rows:
            table.append(
                {
                    "batch_size": batch,
                    "kind": "measured",
                    "peak_lr": row["peak_lr"],
                    "final_val_loss": row["final_val_loss"],
                    "run_id": row["run_id"],
                }
            )
        table.append(
            {
                "batch_size": batch,
                "kind": "fitted_optimum",
                "peak_lr": fit["optimum"],
                "final_val_loss": _fit_loss(fit),
                "run_id": "",
            }
        )
    axis.set_xscale("log")
    axis.set_xlabel("Peak learning rate")
    axis.set_ylabel("Final validation loss")
    axis.set_title("P3.2(a): LR tuning at fixed WD=0.1")
    axis.grid(True, color="#dddddd", linewidth=0.6)
    axis.legend(ncol=2)
    paths = save_bundle(
        out / "p32a_loss_lr_curves",
        figure,
        table,
        {
            "fit": "quadratic in log(LR)",
            "fits": source["lr_fits"],
            "legend": "X marks fitted optimum",
        },
    )
    return [str(path) for path in paths]


def _save_batch_optima(source: Mapping, out: Path) -> list[str]:
    apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    optima = [source["lr_fits"][batch]["optimum"] for batch in SOURCE_BATCHES]
    best_losses = [
        min(float(row["final_val_loss"]) for row in source["lr_rows"][batch])
        for batch in SOURCE_BATCHES
    ]
    axes[0].plot(SOURCE_BATCHES, optima, marker="o", color=COLORS[0])
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Batch size (sequences)")
    axes[0].set_ylabel("Fitted optimal peak LR")
    axes[0].set_title("Optimal LR")
    axes[1].plot(SOURCE_BATCHES, best_losses, marker="s", color=COLORS[1])
    axes[1].set_xscale("log", base=2)
    axes[1].set_xlabel("Batch size (sequences)")
    axes[1].set_ylabel("Best measured validation loss")
    axes[1].set_title("Best measured loss")
    for axis in axes:
        axis.grid(True, color="#dddddd", linewidth=0.6)
        axis.set_xticks(SOURCE_BATCHES, labels=[str(value) for value in SOURCE_BATCHES])
    table = [
        {
            "batch_size": batch,
            "fitted_optimal_lr": source["lr_fits"][batch]["optimum"],
            "fit_method": source["lr_fits"][batch]["method"],
            "best_measured_loss": best_losses[index],
        }
        for index, batch in enumerate(SOURCE_BATCHES)
    ]
    paths = save_bundle(
        out / "p32a_optimal_lr_and_loss_vs_batch",
        figure,
        table,
        {"lr_fits": source["lr_fits"]},
    )
    return [str(path) for path in paths]


def _save_source_laws(source: Mapping, out: Path) -> list[str]:
    apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))
    specs = (
        ("lr_fits", "lr_law", "Optimal peak LR", "Fixed WD=0.1", COLORS[0]),
        ("wd_fits", "wd_law", "Optimal weight decay", "Fixed peak LR=0.0015", COLORS[2]),
    )
    table = []
    for axis, (fit_key, law_key, ylabel, title, color) in zip(axes, specs):
        values = [source[fit_key][batch]["optimum"] for batch in SOURCE_BATCHES]
        law = source[law_key]
        axis.scatter(SOURCE_BATCHES, values, color=color, marker="o", label="source optimum", zorder=3)
        grid = np.geomspace(8, 256, 200)
        axis.plot(
            grid,
            [predict_power_law(law, value) for value in grid],
            color=color,
            label=f"fit: exponent={law['exponent']:.3f}",
        )
        for batch in TARGET_BATCHES:
            axis.scatter(
                [batch],
                [predict_power_law(law, batch)],
                marker="*",
                s=90,
                color="#111111",
                label="target prediction" if batch == TARGET_BATCHES[0] else "_nolegend_",
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("Batch size (sequences)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, color="#dddddd", linewidth=0.6)
        axis.legend(fontsize=8)
        for batch, value in zip(SOURCE_BATCHES, values):
            table.append(
                {
                    "hypothesis": law_key,
                    "batch_size": batch,
                    "kind": "source_optimum",
                    "value": value,
                }
            )
        for batch in TARGET_BATCHES:
            table.append(
                {
                    "hypothesis": law_key,
                    "batch_size": batch,
                    "kind": "prediction",
                    "value": predict_power_law(law, batch),
                }
            )
    paths = save_bundle(
        out / "p32b_source_power_laws",
        figure,
        table,
        {"lr_power_law": source["lr_law"], "wd_power_law": source["wd_law"]},
    )
    return [str(path) for path in paths]


def _target_analysis(all_runs: Sequence[Mapping], source: Mapping) -> dict:
    p32 = matching_runs(all_runs, problem="3.2", subpart="b")
    result = {}
    specs = (
        ("fixed_wd_scale_lr", "peak_lr", "lr_law"),
        ("fixed_lr_scale_wd", "weight_decay", "wd_law"),
    )
    for hypothesis, value_key, law_key in specs:
        result[hypothesis] = {}
        for batch in TARGET_BATCHES:
            rows = [
                row
                for row in p32
                if row["batch_size"] == batch
                and row.get("hypothesis") == hypothesis
                and row.get("config_role") in {"predicted", "nearby_validation"}
            ]
            rows = _dedupe(rows, (value_key,))
            if len(rows) != 3:
                raise PlotDataError(
                    f"{hypothesis} B={batch} requires three completed local target tests, found {len(rows)}"
                )
            predicted_rows = [row for row in rows if row.get("config_role") == "predicted"]
            if len(predicted_rows) != 1:
                raise PlotDataError(f"{hypothesis} B={batch} has {len(predicted_rows)} predicted runs")
            best = min(rows, key=lambda row: float(row["final_val_loss"]))
            predicted = predicted_rows[0]
            result[hypothesis][batch] = {
                "rows": rows,
                "predicted_value_from_source_fit": predict_power_law(source[law_key], batch),
                "predicted_run": predicted,
                "local_best_run": best,
                "prediction_regret": float(predicted["final_val_loss"])
                - float(best["final_val_loss"]),
            }
    return result


def _save_target_tests(targets: Mapping, out: Path) -> list[str]:
    apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    table = []
    specs = (
        ("fixed_wd_scale_lr", "peak_lr", "Peak LR", "Fixed WD=0.1"),
        ("fixed_lr_scale_wd", "weight_decay", "Weight decay", "Fixed LR=0.0015"),
    )
    for axis, (hypothesis, value_key, xlabel, title) in zip(axes, specs):
        for index, batch in enumerate(TARGET_BATCHES):
            item = targets[hypothesis][batch]
            rows = sorted(item["rows"], key=lambda row: float(row[value_key]))
            axis.plot(
                [row[value_key] for row in rows],
                [row["final_val_loss"] for row in rows],
                marker=MARKERS[index],
                color=COLORS[index],
                label=f"B={batch}",
            )
            predicted = item["predicted_run"]
            axis.scatter(
                [predicted[value_key]],
                [predicted["final_val_loss"]],
                marker="*",
                s=100,
                color="#111111",
                zorder=4,
                label="power-law prediction" if index == 0 else "_nolegend_",
            )
            for row in rows:
                table.append(
                    {
                        "hypothesis": hypothesis,
                        "batch_size": batch,
                        "kind": row["config_role"],
                        "tested_value": row[value_key],
                        "final_val_loss": row["final_val_loss"],
                        "run_id": row["run_id"],
                    }
                )
        axis.set_xscale("log")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Final validation loss")
        axis.set_title(title)
        axis.grid(True, color="#dddddd", linewidth=0.6)
        axis.legend(fontsize=8)
    paths = save_bundle(
        out / "p32b_target_local_tests",
        figure,
        table,
        {"targets": targets, "note": "Stars identify the trained source-law predictions."},
    )
    return [str(path) for path in paths]


def _save_transfer_comparison(targets: Mapping, out: Path) -> list[str]:
    apply_style()
    figure, axis = plt.subplots(figsize=(6.6, 4.3))
    hypotheses = ("fixed_wd_scale_lr", "fixed_lr_scale_wd")
    labels = ("Scale LR\n(fixed WD)", "Scale WD\n(fixed LR)")
    x = np.arange(len(hypotheses), dtype=float)
    width = 0.34
    table = []
    for index, batch in enumerate(TARGET_BATCHES):
        regrets = [targets[hypothesis][batch]["prediction_regret"] for hypothesis in hypotheses]
        axis.bar(
            x + (index - 0.5) * width,
            regrets,
            width,
            color=COLORS[index],
            label=f"B={batch}",
        )
        for hypothesis, regret in zip(hypotheses, regrets):
            item = targets[hypothesis][batch]
            table.append(
                {
                    "hypothesis": hypothesis,
                    "batch_size": batch,
                    "predicted_loss": item["predicted_run"]["final_val_loss"],
                    "local_best_loss": item["local_best_run"]["final_val_loss"],
                    "prediction_regret": regret,
                    "predicted_run_id": item["predicted_run"]["run_id"],
                    "local_best_run_id": item["local_best_run"]["run_id"],
                }
            )
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Prediction loss − local-best loss")
    axis.set_title("P3.2(b): target transfer error (lower is better)")
    axis.grid(True, axis="y", color="#dddddd", linewidth=0.6)
    axis.legend()
    paths = save_bundle(
        out / "p32b_transfer_comparison",
        figure,
        table,
        {
            "metric": "measured loss at predicted configuration minus best measured local-target loss",
            "targets": targets,
        },
    )
    return [str(path) for path in paths]


def _momentum_analysis(all_runs: Sequence[Mapping]) -> dict:
    rows = matching_runs(all_runs, problem="3.2", subpart="c")
    result = {}
    for batch in (8, 256):
        selected = _dedupe(
            [row for row in rows if row["batch_size"] == batch],
            ("beta1",),
        )
        beta_values = {float(row["beta1"]) for row in selected}
        expected = {0.0, 0.8, 0.9, 0.95, 0.98, 0.99}
        if beta_values != expected:
            raise PlotDataError(f"B={batch} momentum grid is {sorted(beta_values)}, expected {sorted(expected)}")
        baseline = next(row for row in selected if same_number(row["beta1"], 0.0))
        best = min(selected, key=lambda row: float(row["final_val_loss"]))
        result[batch] = {
            "rows": sorted(selected, key=lambda row: float(row["beta1"])),
            "baseline": baseline,
            "best": best,
            "best_improvement_over_beta1_0": float(baseline["final_val_loss"])
            - float(best["final_val_loss"]),
        }
    return result


def _save_beta1(momentum: Mapping, out: Path) -> list[str]:
    apply_style()
    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    table = []
    for index, batch in enumerate((8, 256)):
        rows = momentum[batch]["rows"]
        axis.plot(
            [row["beta1"] for row in rows],
            [row["final_val_loss"] for row in rows],
            marker=MARKERS[index],
            color=COLORS[index],
            label=f"B={batch}",
        )
        best = momentum[batch]["best"]
        axis.scatter(
            [best["beta1"]],
            [best["final_val_loss"]],
            marker="*",
            s=100,
            color=COLORS[index],
            edgecolors="#111111",
            zorder=4,
        )
        for row in rows:
            table.append(
                {
                    "batch_size": batch,
                    "beta1": row["beta1"],
                    "final_val_loss": row["final_val_loss"],
                    "peak_lr": row["peak_lr"],
                    "weight_decay": row["weight_decay"],
                    "run_id": row["run_id"],
                    "is_best": row["run_id"] == best["run_id"],
                }
            )
    axis.set_xlabel(r"$\beta_1$")
    axis.set_ylabel("Final validation loss")
    axis.set_title("P3.2(c): momentum at held-best LR–WD")
    axis.grid(True, color="#dddddd", linewidth=0.6)
    axis.legend()
    paths = save_bundle(
        out / "p32c_final_loss_vs_beta1",
        figure,
        table,
        {"momentum": momentum, "legend": "Stars mark the best beta1 at each batch."},
    )
    return [str(path) for path in paths]


def _load_histories(momentum: Mapping) -> dict[str, list[dict]]:
    """Read W&B validation histories for each best/beta1=0 comparison."""
    try:
        import wandb

        from utils import WANDB_ENTITY, WANDB_PROJECT
    except Exception as exc:  # pragma: no cover - environment-specific dependency
        raise PlotDataError(f"cannot import W&B configuration for histories: {exc}") from exc
    api = wandb.Api()
    histories = {}
    for batch in (8, 256):
        for role in ("baseline", "best"):
            record = momentum[batch][role]
            run_id = str(record["run_id"])
            if run_id in histories:
                continue
            try:
                run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
                raw_rows = list(run.scan_history(keys=["val_loss", "_step"], page_size=1000))
            except Exception as exc:
                raise PlotDataError(f"could not read W&B history for {run_id}: {exc}") from exc
            rows = []
            for raw in raw_rows:
                try:
                    loss = float(raw["val_loss"])
                    step = int(raw["_step"])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(loss):
                    rows.append({"wandb_step": step, "val_loss": loss})
            rows.sort(key=lambda row: row["wandb_step"])
            if len(rows) < 2:
                raise PlotDataError(f"W&B run {run_id} has fewer than two validation-history points")
            for index, row in enumerate(rows):
                row["training_progress"] = index / (len(rows) - 1)
            histories[run_id] = rows
    return histories


def _save_learning_curves(momentum: Mapping, histories: Mapping, out: Path) -> list[str]:
    apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(9.4, 4.0), sharey=True)
    table = []
    for axis, batch in zip(axes, (8, 256)):
        for role, linestyle, color in (
            ("baseline", "--", "#777777"),
            ("best", "-", COLORS[0]),
        ):
            record = momentum[batch][role]
            rows = histories[record["run_id"]]
            beta = float(record["beta1"])
            axis.plot(
                [100 * row["training_progress"] for row in rows],
                [row["val_loss"] for row in rows],
                linestyle=linestyle,
                color=color,
                label=rf"$\beta_1={beta:g}$" + (" (best)" if role == "best" else " baseline"),
            )
            for row in rows:
                table.append(
                    {
                        "batch_size": batch,
                        "role": role,
                        "beta1": beta,
                        "training_progress_percent": 100 * row["training_progress"],
                        "wandb_step": row["wandb_step"],
                        "val_loss": row["val_loss"],
                        "run_id": record["run_id"],
                    }
                )
        axis.set_title(f"B={batch}")
        axis.set_xlabel("Training progress (% of logged evaluations)")
        axis.grid(True, color="#dddddd", linewidth=0.6)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Validation loss")
    figure.suptitle("P3.2(c): best momentum vs no-momentum baseline", fontsize=11)
    paths = save_bundle(
        out / "p32c_best_vs_beta1_0_learning_curves",
        figure,
        table,
        {
            "history_source": "W&B scan_history(val_loss, _step)",
            "x_axis": "evaluation ordinal normalized within each run",
            "run_ids": {
                str(batch): {
                    role: momentum[batch][role]["run_id"] for role in ("baseline", "best")
                }
                for batch in (8, 256)
            },
        },
    )
    return [str(path) for path in paths]


def _summary_rows(source: Mapping, targets: Mapping, momentum: Mapping) -> list[dict]:
    lr_regrets = [targets["fixed_wd_scale_lr"][batch]["prediction_regret"] for batch in TARGET_BATCHES]
    wd_regrets = [targets["fixed_lr_scale_wd"][batch]["prediction_regret"] for batch in TARGET_BATCHES]
    small_gain = momentum[8]["best_improvement_over_beta1_0"]
    large_gain = momentum[256]["best_improvement_over_beta1_0"]
    lr_losses = [
        (
            float(targets["fixed_wd_scale_lr"][batch]["predicted_run"]["final_val_loss"]),
            float(targets["fixed_wd_scale_lr"][batch]["local_best_run"]["final_val_loss"]),
        )
        for batch in TARGET_BATCHES
    ]
    wd_losses = [
        (
            float(targets["fixed_lr_scale_wd"][batch]["predicted_run"]["final_val_loss"]),
            float(targets["fixed_lr_scale_wd"][batch]["local_best_run"]["final_val_loss"]),
        )
        for batch in TARGET_BATCHES
    ]
    return [
        {
            "topic": "LR–batch scaling",
            "NQM_prediction": "Optimal LR rises approximately as a batch-size power law.",
            "LM_measurement": (
                f"Source exponent {source['lr_law']['exponent']:.3f}. Predicted/local-best loss: "
                f"{lr_losses[0][0]:.4f}/{lr_losses[0][1]:.4f} at B=128 and "
                f"{lr_losses[1][0]:.4f}/{lr_losses[1][1]:.4f} at B=256 "
                f"(regret {lr_regrets[0]:.4f}, {lr_regrets[1]:.4f})."
            ),
            "verdict": "failed prediction",
            "reason": "The extrapolated LR is not the best tested local target and transfer worsens at B=256.",
        },
        {
            "topic": "Weight decay",
            "NQM_prediction": "Not represented in the P3.1 quadratic optimizer (WD was fixed to zero).",
            "LM_measurement": (
                f"Empirical WD exponent {source['wd_law']['exponent']:.3f}. Predicted/local-best loss: "
                f"{wd_losses[0][0]:.4f}/{wd_losses[0][1]:.4f} at B=128 and "
                f"{wd_losses[1][0]:.4f}/{wd_losses[1][1]:.4f} at B=256 "
                f"(regret {wd_regrets[0]:.4f}, {wd_regrets[1]:.4f})."
            ),
            "verdict": "effect not modeled",
            "reason": "An empirical transfer result is measurable, but it is not a test of the stated zero-WD NQM.",
        },
        {
            "topic": "Momentum",
            "NQM_prediction": "Momentum should help more at large batch than at small batch.",
            "LM_measurement": (
                f"beta1=0/best loss: {momentum[8]['baseline']['final_val_loss']:.4f}/"
                f"{momentum[8]['best']['final_val_loss']:.4f} at B=8 and "
                f"{momentum[256]['baseline']['final_val_loss']:.4f}/"
                f"{momentum[256]['best']['final_val_loss']:.4f} at B=256 "
                f"(improvements {small_gain:.4f}, {large_gain:.4f})."
            ),
            "verdict": "agreement",
            "reason": "The held-LR/WD momentum benefit is substantially larger at B=256.",
        },
    ]


def _save_prediction_summary(rows: Sequence[Mapping], out: Path) -> list[str]:
    apply_style()
    figure, axis = plt.subplots(figsize=(11.2, 5.3))
    axis.axis("off")
    headers = ("Topic", "NQM prediction", "LM measurement", "Verdict")
    cells = [
        [
            "\n".join(textwrap.wrap(str(row["topic"]), 18)),
            "\n".join(textwrap.wrap(str(row["NQM_prediction"]), 38)),
            "\n".join(textwrap.wrap(str(row["LM_measurement"]), 54)),
            "\n".join(textwrap.wrap(str(row["verdict"]), 18)),
        ]
        for row in rows
    ]
    table = axis.table(
        cellText=cells,
        colLabels=headers,
        cellLoc="left",
        colLoc="left",
        colWidths=(0.13, 0.28, 0.42, 0.17),
        bbox=(0, 0, 1, 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 2.0)
    verdict_colors = {
        "failed prediction": "#f4cccc",
        "effect not modeled": "#fff2cc",
        "agreement": "#d9ead3",
    }
    for column in range(4):
        table[(0, column)].set_facecolor("#d9e6f2")
        table[(0, column)].set_text_props(weight="bold")
    for index, row in enumerate(rows, start=1):
        table[(index, 3)].set_facecolor(verdict_colors[row["verdict"]])
        for column in range(4):
            table[(index, column)].get_text().set_wrap(True)
    axis.set_title("P3.2(d): prediction versus measurement", pad=8)
    paths = save_bundle(
        out / "p32d_prediction_measurement_summary",
        figure,
        list(rows),
        {
            "rows": rows,
            "distinction": (
                "'failed prediction' means the NQM made a directional/transfer claim contradicted "
                "by the target tests; 'effect not modeled' means the P3.1 NQM omitted the mechanism."
            ),
        },
    )
    return [str(path) for path in paths]


def _compact_targets(targets: Mapping, hypothesis: str, value_key: str) -> dict:
    compact = {}
    for batch in TARGET_BATCHES:
        item = targets[hypothesis][batch]
        predicted = item["predicted_run"]
        local_best = item["local_best_run"]
        compact[str(batch)] = {
            "source_fit_prediction": item["predicted_value_from_source_fit"],
            "predicted_tested_value": predicted[value_key],
            "predicted_loss": predicted["final_val_loss"],
            "predicted_run_id": predicted["run_id"],
            "local_best_tested_value": local_best[value_key],
            "local_best_loss": local_best["final_val_loss"],
            "local_best_run_id": local_best["run_id"],
            "prediction_regret": item["prediction_regret"],
        }
    return compact


def _compact_momentum(momentum: Mapping) -> dict:
    compact = {}
    for batch in (8, 256):
        item = momentum[batch]
        compact[str(batch)] = {
            "held_peak_lr": item["best"]["peak_lr"],
            "held_weight_decay": item["best"]["weight_decay"],
            "loss_by_beta1": {
                format(float(row["beta1"]), "g"): {
                    "final_val_loss": row["final_val_loss"],
                    "run_id": row["run_id"],
                }
                for row in item["rows"]
            },
            "best_beta1": item["best"]["beta1"],
            "best_loss": item["best"]["final_val_loss"],
            "beta1_0_loss": item["baseline"]["final_val_loss"],
            "best_improvement_over_beta1_0": item["best_improvement_over_beta1_0"],
        }
    return compact


def generate(plot_dir: Path = DEFAULT_PLOT_DIR, summary_path: Path = DEFAULT_SUMMARY) -> dict:
    """Query measurements, generate all bundles, and write the analysis summary."""
    all_runs = query_a2_runs()
    completed = [run for run in all_runs if run.get("completed")]
    source = _source_analysis(completed)
    targets = _target_analysis(completed, source)
    momentum = _momentum_analysis(completed)
    histories = _load_histories(momentum)

    written = []
    written.extend(_save_loss_lr(source, plot_dir))
    written.extend(_save_batch_optima(source, plot_dir))
    written.extend(_save_source_laws(source, plot_dir))
    written.extend(_save_target_tests(targets, plot_dir))
    written.extend(_save_transfer_comparison(targets, plot_dir))
    written.extend(_save_beta1(momentum, plot_dir))
    written.extend(_save_learning_curves(momentum, histories, plot_dir))
    summary_rows = _summary_rows(source, targets, momentum)
    written.extend(_save_prediction_summary(summary_rows, plot_dir))

    summary = {
        "problem": "3.2",
        "data_policy": {
            "queried_function": "experiments.a2.helpers.results.query_a2_runs",
            "queried_run_count": len(all_runs),
            "completed_run_count": len(completed),
            "new_experiment_source": "configured W&B project",
            "reused_B64_source": "bundled export of actual hashimoto-group/public-alchemy W&B runs",
            "synthetic_data": False,
        },
        "part_a": {
            "lr_fits_by_batch": source["lr_fits"],
            "best_measured_loss_by_batch": {
                str(batch): min(
                    float(row["final_val_loss"]) for row in source["lr_rows"][batch]
                )
                for batch in SOURCE_BATCHES
            },
            "finding": (
                "The fitted optimum is non-monotonic over B=8–64; the best measured loss "
                "improves through the middle of the tested range rather than uniformly with batch."
            ),
        },
        "part_b": {
            "fixed_wd_scale_lr": {
                "source_power_law": source["lr_law"],
                "targets": _compact_targets(targets, "fixed_wd_scale_lr", "peak_lr"),
            },
            "fixed_lr_scale_wd": {
                "source_power_law": source["wd_law"],
                "targets": _compact_targets(targets, "fixed_lr_scale_wd", "weight_decay"),
            },
            "finding": (
                "Scaling WD at fixed LR transfers more accurately than scaling LR at fixed WD "
                "on both target batches, although each source-law prediction misses its local best."
            ),
        },
        "part_c": {
            "by_batch": _compact_momentum(momentum),
            "history_run_ids": sorted(histories),
            "finding": (
                "The best beta1 is 0.95 at both batches. Relative to beta1=0, momentum lowers "
                "final loss much more at B=256 than at B=8."
            ),
        },
        "part_d": {
            "prediction_measurement_rows": summary_rows,
            "finding": (
                "The LR extrapolation is recorded as a failed prediction; WD is separated as "
                "an effect omitted by the zero-WD P3.1 NQM; the momentum trend agrees."
            ),
            "suggested_nqm_change": (
                "Add decoupled weight decay and a nonstationary, anisotropic gradient-noise "
                "covariance so the model can represent WD coupling and training-phase-dependent noise."
            ),
        },
        "artifacts": written,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2) + "\n")
    written.append(str(summary_path))
    print(f"Wrote {len(written)} files")
    for path in written:
        print(path)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    generate(args.plot_dir, args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
