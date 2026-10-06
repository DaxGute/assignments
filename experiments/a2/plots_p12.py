"""Generate every required Assignment 2 Problem 1 and Problem 2 figure.

The supplied sweeps are read offline.  Measurements for the new P1c, P1d, and
P2c target experiments are read from completed W&B runs through
``query_a2_runs``.  Missing, ambiguous, or inconsistent data aborts the whole
analysis before any figure is written.

Run from the repository root with::

    uv run python -m experiments.a2.plots_p12
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.a2.helpers.fitting import (
    HYPERBALL_SOURCE_TOKENS,
    HYPERBALL_TARGET_TOKENS,
    P1_LARGE_TOKENS,
    P1_TARGET_TOKENS,
    P1_TOKENS,
    PRODUCT_SOURCE_TOKENS,
    PRODUCT_TARGET_LR,
    PRODUCT_TARGET_TOKENS,
    bivariate_loss,
    bivariate_optimum,
    fit_hyperball,
    fit_lr_wd_product,
    fit_token_lr,
    log_quadratic_loss,
    power_law,
    predict_power_law,
    quadratic_log_optimum,
)
from experiments.a2.helpers.paths import DATA_ROOT, plot_directory
from experiments.a2.helpers.plotting import (
    MissingData,
    apply_style,
    contour_figure,
    format_tokens,
    loss_curve_figure,
    place_legend,
    prepare_axis,
    save_bundle,
    scaling_figure,
)
from experiments.a2.helpers.results import LOSS_ATOL, matching_runs, query_a2_runs
from experiments.a2.provided_sweeps import load


P1_SMALL_TOKENS = P1_TOKENS[:3]
P1_GRID = (0.0015, 0.003, 0.006)
P2C_GRID_WDS = (0.05, 0.1, 0.2)
SUMMARY_PATH = DATA_ROOT / "p12_summary.json"


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-7, abs_tol=1e-12)


def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise MissingData(f"refusing to serialize non-finite value {value}")
    return value


def _supplied(part: str) -> list[dict]:
    return [dict(row) for row in load(part)]


def _queried_row(run: Mapping) -> dict:
    return {
        "run_id": str(run.get("run_id") or ""),
        "run_url": "",
        "tokens": int(run["token_budget"]),
        "learning_rate": float(run["peak_lr"]),
        "weight_decay": float(run["weight_decay"]),
        "final_val_loss": float(run["final_val_loss"]),
        "batch_size": int(run["batch_size"]),
        "optimizer": str(run["optimizer"]),
        "config_role": run.get("config_role"),
        "source": "wandb",
    }


def _dedupe(rows: Sequence[Mapping], keys: Sequence[str]) -> list[dict]:
    grouped: dict[tuple, list[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    result = []
    for identity, group in grouped.items():
        losses = [float(row["final_val_loss"]) for row in group]
        if max(losses) - min(losses) > LOSS_ATOL:
            ids = ", ".join(
                f"{row.get('run_id', '(no id)')}={float(row['final_val_loss']):.8g}"
                for row in group
            )
            raise MissingData(f"conflicting completed measurements at {identity}: {ids}")
        result.append(dict(sorted(group, key=lambda row: str(row.get("run_id") or ""))[0]))
    return sorted(result, key=lambda row: tuple(float(row[key]) if isinstance(row[key], (int, float)) else str(row[key]) for key in keys))


def _require_grid(
    rows: Sequence[Mapping],
    *,
    label: str,
    tokens: Sequence[int],
    learning_rates: Sequence[float] | None = None,
    weight_decays: Mapping[int, Sequence[float]] | None = None,
) -> None:
    missing = []
    for budget in tokens:
        budget_rows = [row for row in rows if int(row["tokens"]) == int(budget)]
        if learning_rates is not None:
            for lr in learning_rates:
                if not any(_close(row["learning_rate"], lr) for row in budget_rows):
                    missing.append(f"{format_tokens(budget)} LR={lr:g}")
        if weight_decays is not None:
            for lr in learning_rates or ():
                for wd in weight_decays[budget]:
                    if not any(
                        _close(row["learning_rate"], lr)
                        and _close(row["weight_decay"], wd)
                        for row in budget_rows
                    ):
                        missing.append(f"{format_tokens(budget)} LR={lr:g}, WD={wd:g}")
    if missing:
        raise MissingData(f"{label} is missing required measurements: " + "; ".join(missing))


def _one(rows: Sequence[Mapping], *, label: str, **criteria) -> dict:
    hits = []
    for row in rows:
        if all(
            _close(row[key], expected)
            if isinstance(expected, (int, float)) and row.get(key) is not None
            else row.get(key) == expected
            for key, expected in criteria.items()
        ):
            hits.append(row)
    if not hits:
        rendered = ", ".join(f"{key}={value}" for key, value in criteria.items())
        raise MissingData(f"{label}: missing completed measurement ({rendered})")
    deduped = _dedupe(hits, tuple(criteria))
    if len(deduped) != 1:
        raise MissingData(f"{label}: ambiguous completed measurement for {criteria}")
    return deduped[0]


def collect_data() -> dict:
    """Load and fully validate all data before creating output directories."""
    p1a = _dedupe(_supplied("P1a"), ("tokens", "learning_rate", "weight_decay"))
    p1b = _dedupe(_supplied("P1b"), ("tokens", "learning_rate", "weight_decay"))
    p1d_source = _dedupe(_supplied("P1d"), ("tokens", "learning_rate", "weight_decay"))
    p2a = _dedupe(_supplied("P2a"), ("tokens", "learning_rate", "weight_decay"))
    _require_grid(p1a, label="P1a", tokens=P1_SMALL_TOKENS, learning_rates=P1_GRID)
    _require_grid(p1b, label="P1b", tokens=P1_LARGE_TOKENS, learning_rates=P1_GRID)
    _require_grid(
        p1d_source,
        label="P1d source",
        tokens=HYPERBALL_SOURCE_TOKENS,
        learning_rates=(0.0003, 0.001, 0.0015, 0.003, 0.006, 0.01, 0.015, 0.03),
    )
    p2_wds = {
        153_600_000: (0.4, 1.6, 6.4),
        307_200_000: (0.1, 0.4, 0.8),
        614_400_000: (0.1, 0.2, 0.4),
        1_228_800_000: (0.1, 0.2, 0.4),
    }
    p2_lrs = {
        153_600_000: (0.00075, 0.0015, 0.003),
        307_200_000: P1_GRID,
        614_400_000: P1_GRID,
        1_228_800_000: P1_GRID,
    }
    for budget in PRODUCT_SOURCE_TOKENS:
        _require_grid(
            p2a,
            label="P2a",
            tokens=(budget,),
            learning_rates=p2_lrs[budget],
            weight_decays={budget: p2_wds[budget]},
        )

    all_runs = query_a2_runs()
    p1c_runs = [_queried_row(run) for run in matching_runs(all_runs, problem="1", subpart="c")]
    p1d_runs = [_queried_row(run) for run in matching_runs(all_runs, problem="1", subpart="d")]
    p2c_runs = [_queried_row(run) for run in matching_runs(all_runs, problem="2", subpart="c")]

    source_p1 = p1a + p1b
    all6 = fit_token_lr(source_p1, P1_TOKENS)
    large3 = fit_token_lr(source_p1, P1_LARGE_TOKENS)
    for lr in (*P1_GRID, all6["predicted_lr"], large3["predicted_lr"]):
        _one(
            p1c_runs,
            label="P1c target",
            tokens=P1_TARGET_TOKENS,
            learning_rate=lr,
            weight_decay=0.1,
            optimizer="adamw",
        )

    hyperball_fit = fit_hyperball(p1d_source)
    expected_hyperball = (
        hyperball_fit["predicted_lr"],
        hyperball_fit["predicted_lr"] * 0.5,
        hyperball_fit["predicted_lr"] * 2**-0.5,
        hyperball_fit["predicted_lr"] * 2**0.5,
    )
    for lr in expected_hyperball:
        _one(
            p1d_runs,
            label="P1d target",
            tokens=HYPERBALL_TARGET_TOKENS,
            learning_rate=lr,
            weight_decay=0.0,
            optimizer="adamh",
        )

    product_fit = fit_lr_wd_product(p2a)
    p2c_combined = list(p2c_runs)
    supplied_target = [
        row
        for row in p1b
        if row["tokens"] == PRODUCT_TARGET_TOKENS
        and _close(row["learning_rate"], PRODUCT_TARGET_LR)
        and _close(row["weight_decay"], 0.1)
    ]
    p2c_combined.extend(supplied_target)
    for wd in (*P2C_GRID_WDS, product_fit["predicted_wd"]):
        _one(
            p2c_combined,
            label="P2c fixed-LR target",
            tokens=PRODUCT_TARGET_TOKENS,
            learning_rate=PRODUCT_TARGET_LR,
            weight_decay=wd,
            optimizer="adamw",
        )
    best_small = min(
        (row for row in p2a if row["tokens"] == PRODUCT_SOURCE_TOKENS[0]),
        key=lambda row: row["final_val_loss"],
    )
    _one(
        p2c_combined,
        label="P2c transferred source-best target",
        tokens=PRODUCT_TARGET_TOKENS,
        learning_rate=best_small["learning_rate"],
        weight_decay=best_small["weight_decay"],
        optimizer="adamw",
    )
    return {
        "p1a": p1a,
        "p1b": p1b,
        "p1d_source": p1d_source,
        "p2a": p2a,
        "p1c_target": _dedupe(p1c_runs, ("tokens", "learning_rate", "weight_decay", "optimizer")),
        "p1d_target": _dedupe(p1d_runs, ("tokens", "learning_rate", "weight_decay", "optimizer")),
        "p2c_target": _dedupe(p2c_combined, ("tokens", "learning_rate", "weight_decay", "optimizer")),
        "fits": {"p1_all6": all6, "p1_large3": large3, "hyperball": hyperball_fit, "product": product_fit},
    }


def _series_by_budget(rows: Sequence[Mapping], budgets: Sequence[int]) -> list[dict]:
    return [
        {
            "label": format_tokens(tokens),
            "records": [row for row in rows if int(row["tokens"]) == int(tokens)],
            "x_key": "learning_rate",
        }
        for tokens in budgets
    ]


def _optimum_points(fit: Mapping) -> list[dict]:
    return [
        {"x": int(row["tokens"]), "y": float(row["optimum"]), "note": row["method"]}
        for row in fit["optima"]
    ]


def _record(report: dict, name: str, paths: Sequence[Path]) -> None:
    report["figures"][name] = [str(path) for path in paths]


def plot_p1(data: Mapping, report: dict) -> dict:
    output = plot_directory("p1")
    p1a_fit = fit_token_lr(data["p1a"], P1_SMALL_TOKENS)
    p1b_fit = fit_token_lr(data["p1b"], P1_LARGE_TOKENS)
    all6 = data["fits"]["p1_all6"]
    large3 = data["fits"]["p1_large3"]

    paths, curve_fits = loss_curve_figure(
        _series_by_budget(data["p1a"], P1_SMALL_TOKENS),
        xlabel="Peak learning rate",
        ylabel="Final validation loss",
        path=output / "p1a_loss_vs_lr",
        title="P1a: source loss–learning-rate fits",
    )
    _record(report, "p1a_loss_vs_lr", paths)
    paths = scaling_figure(
        [{"label": "P1a fitted optima", "points": _optimum_points(p1a_fit), "style": "adamw"}],
        xlabel="Training tokens",
        ylabel="Fitted optimal peak learning rate",
        path=output / "p1a_optimal_lr_scaling",
    )
    _record(report, "p1a_optimal_lr_scaling", paths)

    paths, p1b_curves = loss_curve_figure(
        _series_by_budget(data["p1b"], P1_LARGE_TOKENS),
        xlabel="Peak learning rate",
        ylabel="Final validation loss",
        path=output / "p1b_loss_vs_lr",
        title="P1b: held-out loss–learning-rate fits",
    )
    _record(report, "p1b_loss_vs_lr", paths)
    predictions = [
        {
            "x": tokens,
            "y": predict_power_law(p1a_fit["power_law"], tokens),
            "label": "P1a rule predictions" if index == 0 else "_nolegend_",
        }
        for index, tokens in enumerate(P1_LARGE_TOKENS)
    ]
    paths = scaling_figure(
        [
            {
                "label": "P1a fitted optima",
                "points": _optimum_points(p1a_fit),
                "style": "adamw",
                "fit": False,
            },
            {
                "label": "P1b fitted optima",
                "points": _optimum_points(p1b_fit),
                "style": "adamh",
                "fit": False,
            },
        ],
        xlabel="Training tokens",
        ylabel="Optimal / predicted peak learning rate",
        path=output / "p1b_predictions_vs_fitted_optima",
        predictions=predictions,
        overlay_laws=[
            {
                "label": "P1a source rule",
                "law": p1a_fit["power_law"],
                "xmin": P1_SMALL_TOKENS[0],
                "xmax": P1_LARGE_TOKENS[-1],
            }
        ],
    )
    _record(report, "p1b_predictions_vs_fitted_optima", paths)

    paths = scaling_figure(
        [
            {
                "label": "all six source optima",
                "points": _optimum_points(all6),
                "style": "adamw",
                "fit": False,
            }
        ],
        xlabel="Training tokens",
        ylabel="Fitted optimal peak learning rate",
        path=output / "p1c_two_source_power_laws",
        predictions=[
            {"x": P1_TARGET_TOKENS, "y": all6["predicted_lr"], "label": "all-six prediction"},
            {"x": P1_TARGET_TOKENS, "y": large3["predicted_lr"], "label": "large-three prediction"},
        ],
        overlay_laws=[
            {
                "label": "all-six law",
                "law": all6["power_law"],
                "xmin": P1_TOKENS[0],
                "xmax": P1_TARGET_TOKENS,
                "color": "#1f4e79",
            },
            {
                "label": "large-three law",
                "law": large3["power_law"],
                "xmin": P1_LARGE_TOKENS[0],
                "xmax": P1_TARGET_TOKENS,
                "color": "#c0392b",
                "linestyle": "-.",
            },
        ],
    )
    _record(report, "p1c_two_source_power_laws", paths)

    target_grid = [
        row for row in data["p1c_target"] if any(_close(row["learning_rate"], lr) for lr in P1_GRID)
    ]
    paths, target_fits = loss_curve_figure(
        [{"label": format_tokens(P1_TARGET_TOKENS), "records": target_grid, "x_key": "learning_rate"}],
        xlabel="Peak learning rate",
        ylabel="Final validation loss",
        path=output / "p1c_target_curve",
        title="P1c: target-grid fit at 4.9152B tokens",
    )
    _record(report, "p1c_target_curve", paths)
    p1c_best = min(target_grid, key=lambda row: row["final_val_loss"])
    p1c_comparison = []
    for label, lr in (
        ("all-six prediction", all6["predicted_lr"]),
        ("large-three prediction", large3["predicted_lr"]),
    ):
        row = _one(data["p1c_target"], label=label, learning_rate=lr)
        p1c_comparison.append(
            {
                "label": label,
                "learning_rate": lr,
                "loss": row["final_val_loss"],
                "gap_to_best_grid": row["final_val_loss"] - p1c_best["final_val_loss"],
                "run_id": row["run_id"],
            }
        )
    _comparison_bars(
        p1c_comparison,
        output / "p1c_predictions_and_measured_gaps",
        title="P1c: prediction losses and gaps to best target-grid run",
        report=report,
        name="p1c_predictions_and_measured_gaps",
    )

    hyperball = data["fits"]["hyperball"]
    adam_small = fit_token_lr(data["p1a"], P1_SMALL_TOKENS)
    paths = scaling_figure(
        [
            {"label": "Hyperball source optima", "points": _optimum_points(hyperball), "style": "adamh"},
            {"label": "AdamW source optima", "points": _optimum_points(adam_small), "style": "adamw"},
        ],
        xlabel="Training tokens",
        ylabel="Fitted optimal peak learning rate",
        path=output / "p1d_source_scaling_hyperball_vs_adamw",
        predictions=[
            {
                "x": HYPERBALL_TARGET_TOKENS,
                "y": hyperball["predicted_lr"],
                "label": "Hyperball target prediction",
            },
            {
                "x": HYPERBALL_TARGET_TOKENS,
                "y": predict_power_law(adam_small["power_law"], HYPERBALL_TARGET_TOKENS),
                "label": "AdamW source-law prediction",
            },
        ],
    )
    _record(report, "p1d_source_scaling_hyperball_vs_adamw", paths)
    adam_target = [row for row in data["p1b"] if row["tokens"] == HYPERBALL_TARGET_TOKENS]
    apply_style()
    figure, axis = plt.subplots(figsize=(6.6, 4.5))
    rows = []
    for label, records, color, marker in (
        ("Hyperball target", data["p1d_target"], "#6c3483", "P"),
        ("AdamW target", adam_target, "#1f4e79", "o"),
    ):
        ordered = sorted(records, key=lambda row: row["learning_rate"])
        axis.plot(
            [row["learning_rate"] for row in ordered],
            [row["final_val_loss"] for row in ordered],
            marker=marker,
            color=color,
            label=label,
        )
        rows.extend(
            {
                "optimizer": label,
                "learning_rate": row["learning_rate"],
                "final_val_loss": row["final_val_loss"],
                "run_id": row["run_id"],
            }
            for row in ordered
        )
    prepare_axis(axis, "Peak learning rate", "Final validation loss", xlog=True)
    axis.set_title("P1d: target comparison at 1.2288B tokens")
    place_legend(axis)
    paths = save_bundle(
        output / "p1d_target_hyperball_vs_adamw",
        figure,
        rows,
        {
            "hyperball_prediction": hyperball["predicted_lr"],
            "hyperball_best_measured": min(data["p1d_target"], key=lambda row: row["final_val_loss"]),
            "adamw_best_measured": min(adam_target, key=lambda row: row["final_val_loss"]),
        },
    )
    _record(report, "p1d_target_hyperball_vs_adamw", paths)
    return {
        "p1a": {"curve_fits": curve_fits, "scaling": p1a_fit},
        "p1b": {
            "curve_fits": p1b_curves,
            "scaling": p1b_fit,
            "source_rule_predictions": {
                str(tokens): predict_power_law(p1a_fit["power_law"], tokens)
                for tokens in P1_LARGE_TOKENS
            },
        },
        "p1c": {
            "all_six": all6,
            "large_three": large3,
            "target_fit": target_fits[0],
            "best_target_grid": p1c_best,
            "predictions": p1c_comparison,
        },
        "p1d": {
            "hyperball": hyperball,
            "adamw_source": adam_small,
            "hyperball_best_target": min(data["p1d_target"], key=lambda row: row["final_val_loss"]),
            "adamw_best_target": min(adam_target, key=lambda row: row["final_val_loss"]),
        },
    }


def _comparison_bars(items, path: Path, *, title: str, report: dict, name: str) -> None:
    apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    labels = [item["label"] for item in items]
    losses = [item["loss"] for item in items]
    gaps = [item["gap_to_best_grid"] for item in items]
    positions = np.arange(len(items))
    axes[0].bar(positions, losses, color="#1f4e79")
    axes[1].bar(positions, gaps, color="#c0392b")
    for axis, ylabel in zip(axes, ("Final validation loss", "Loss gap")):
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", color="#dddddd", linewidth=0.6)
    axes[0].set_title(title)
    rows = [
        {
            "label": item["label"],
            "learning_rate": item.get("learning_rate", ""),
            "weight_decay": item.get("weight_decay", ""),
            "final_val_loss": item["loss"],
            "gap_to_best_grid": item["gap_to_best_grid"],
            "run_id": item.get("run_id", ""),
        }
        for item in items
    ]
    paths = save_bundle(path, figure, rows, {"comparisons": items})
    _record(report, name, paths)


def _p2_fits(rows: Sequence[Mapping]) -> list[dict]:
    fits = []
    for tokens in PRODUCT_SOURCE_TOKENS:
        budget_rows = [row for row in rows if row["tokens"] == tokens]
        fits.append({"tokens": tokens, **bivariate_optimum(budget_rows)})
    return fits


def _p2_trend_figure(fits, key: str, ylabel: str, path: Path, report: dict, name: str) -> dict:
    points = [{"x": fit["tokens"], "y": fit[key], "note": fit["method"]} for fit in fits]
    paths = scaling_figure(
        [{"label": ylabel, "points": points, "style": "adamw"}],
        xlabel="Training tokens",
        ylabel=ylabel,
        path=path,
    )
    _record(report, name, paths)
    return power_law([point["x"] for point in points], [point["y"] for point in points])


def plot_p2(data: Mapping, report: dict) -> dict:
    output = plot_directory("p2")
    fits = _p2_fits(data["p2a"])
    p1 = data["p1a"] + data["p1b"]
    p1_by_budget = defaultdict(list)
    for row in p1:
        p1_by_budget[row["tokens"]].append(row)

    for fit in fits:
        tokens = fit["tokens"]
        rows = [
            {
                "learning_rate": row["learning_rate"],
                "weight_decay": row["weight_decay"],
                "final_val_loss": row["final_val_loss"],
                "run_id": row["run_id"],
            }
            for row in data["p2a"]
            if row["tokens"] == tokens
        ]
        paths = contour_figure(
            rows,
            fit,
            path=output / f"p2a_contour_{format_tokens(tokens)}",
        )
        _record(report, f"p2a_contour_{format_tokens(tokens)}", paths)

    lr_law = _p2_trend_figure(
        fits,
        "optimum_lr",
        "Fitted optimal peak learning rate",
        output / "p2a_optimal_lr_vs_tokens",
        report,
        "p2a_optimal_lr_vs_tokens",
    )
    wd_law = _p2_trend_figure(
        fits,
        "optimum_wd",
        "Fitted optimal weight decay",
        output / "p2a_optimal_wd_vs_tokens",
        report,
        "p2a_optimal_wd_vs_tokens",
    )

    comparison = []
    for fit in fits:
        tokens = fit["tokens"]
        joint_rows = [row for row in data["p2a"] if row["tokens"] == tokens]
        p1_rows = p1_by_budget[tokens]
        joint = min(joint_rows, key=lambda row: row["final_val_loss"])
        p1_best = min(p1_rows, key=lambda row: row["final_val_loss"])
        comparison.append(
            {
                "tokens": tokens,
                "joint_loss": joint["final_val_loss"],
                "p1_loss": p1_best["final_val_loss"],
                "difference_p1_minus_joint": p1_best["final_val_loss"] - joint["final_val_loss"],
                "joint_run_id": joint["run_id"],
                "p1_run_id": p1_best["run_id"],
            }
        )
    apply_style()
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    axes[0].plot(
        [row["tokens"] for row in comparison],
        [row["joint_loss"] for row in comparison],
        marker="o",
        label="joint LR–WD tuning",
    )
    axes[0].plot(
        [row["tokens"] for row in comparison],
        [row["p1_loss"] for row in comparison],
        marker="s",
        label="P1 LR-only tuning",
    )
    prepare_axis(axes[0], "Training tokens", "Best measured validation loss", xlog=True)
    axes[0].legend()
    axes[1].plot(
        [row["tokens"] for row in comparison],
        [row["difference_p1_minus_joint"] for row in comparison],
        marker="D",
        color="#c0392b",
    )
    axes[1].axhline(0, color="#444444", linestyle=":", linewidth=0.9)
    prepare_axis(axes[1], "Training tokens", "P1 loss − joint-tuning loss", xlog=True)
    paths = save_bundle(
        output / "p2a_joint_vs_p1_best_losses",
        figure,
        comparison,
        {"comparison": comparison, "positive_difference_favors_joint_tuning": True},
    )
    _record(report, "p2a_joint_vs_p1_best_losses", paths)

    for fit in fits:
        tokens = fit["tokens"]
        rows = [
            {
                "learning_rate": row["learning_rate"],
                "weight_decay": row["weight_decay"],
                "final_val_loss": row["final_val_loss"],
                "run_id": row["run_id"],
            }
            for row in data["p2a"]
            if row["tokens"] == tokens
        ]
        paths = contour_figure(
            rows,
            fit,
            product=fit["optimum_product"],
            path=output / f"p2b_constant_product_{format_tokens(tokens)}",
        )
        _record(report, f"p2b_constant_product_{format_tokens(tokens)}", paths)

    product_law = power_law(
        [fit["tokens"] for fit in fits], [fit["optimum_product"] for fit in fits]
    )
    normalized_series = []
    for key, label in (
        ("optimum_lr", "optimal LR"),
        ("optimum_wd", "optimal WD"),
        ("optimum_product", "optimal LR × WD"),
    ):
        first = fits[0][key]
        normalized_series.append(
            {
                "label": label,
                "points": [
                    {"x": fit["tokens"], "y": fit[key] / first, "note": "normalized to first budget"}
                    for fit in fits
                ],
                "fit": True,
            }
        )
    paths = scaling_figure(
        normalized_series,
        xlabel="Training tokens",
        ylabel="Optimum normalized at 153.6M tokens",
        path=output / "p2b_lr_wd_product_power_laws",
    )
    _record(report, "p2b_lr_wd_product_power_laws", paths)

    product_fit = data["fits"]["product"]
    predicted = _one(
        data["p2c_target"],
        label="P2c prediction",
        learning_rate=PRODUCT_TARGET_LR,
        weight_decay=product_fit["predicted_wd"],
    )
    wd_grid = [
        _one(
            data["p2c_target"],
            label="P2c WD grid",
            learning_rate=PRODUCT_TARGET_LR,
            weight_decay=wd,
        )
        for wd in P2C_GRID_WDS
    ]
    fixed_lr_best = min(wd_grid, key=lambda row: row["final_val_loss"])
    small_best = min(
        (row for row in data["p2a"] if row["tokens"] == PRODUCT_SOURCE_TOKENS[0]),
        key=lambda row: row["final_val_loss"],
    )
    transferred = _one(
        data["p2c_target"],
        label="P2c transferred source best",
        learning_rate=small_best["learning_rate"],
        weight_decay=small_best["weight_decay"],
    )
    p2c_items = [
        {
            "label": "product-rule prediction",
            "learning_rate": predicted["learning_rate"],
            "weight_decay": predicted["weight_decay"],
            "loss": predicted["final_val_loss"],
            "gap_to_best_grid": 0.0,
            "run_id": predicted["run_id"],
        },
        {
            "label": "153.6M pair transfer",
            "learning_rate": transferred["learning_rate"],
            "weight_decay": transferred["weight_decay"],
            "loss": transferred["final_val_loss"],
            "gap_to_best_grid": transferred["final_val_loss"] - predicted["final_val_loss"],
            "run_id": transferred["run_id"],
        },
        {
            "label": "best fixed-LR WD grid",
            "learning_rate": fixed_lr_best["learning_rate"],
            "weight_decay": fixed_lr_best["weight_decay"],
            "loss": fixed_lr_best["final_val_loss"],
            "gap_to_best_grid": fixed_lr_best["final_val_loss"] - predicted["final_val_loss"],
            "run_id": fixed_lr_best["run_id"],
        },
    ]
    _comparison_bars(
        p2c_items,
        output / "p2c_predicted_rule_vs_baselines",
        title="P2c: product rule versus both baselines",
        report=report,
        name="p2c_predicted_rule_vs_baselines",
    )
    return {
        "p2a": {
            "budget_fits": fits,
            "lr_power_law": lr_law,
            "wd_power_law": wd_law,
            "best_loss_comparison": comparison,
        },
        "p2b": {
            "lr_power_law": lr_law,
            "wd_power_law": wd_law,
            "product_power_law": product_law,
            "r_squared_comparison": {
                "learning_rate": lr_law["r_squared"],
                "weight_decay": wd_law["r_squared"],
                "product": product_law["r_squared"],
            },
        },
        "p2c": {
            "product_fit": product_fit,
            "prediction": predicted,
            "transferred_source_pair": transferred,
            "fixed_lr_best": fixed_lr_best,
            "loss_gap_to_transferred": predicted["final_val_loss"] - transferred["final_val_loss"],
            "loss_gap_to_fixed_lr_best": predicted["final_val_loss"] - fixed_lr_best["final_val_loss"],
        },
    }


def _findings(summary: Mapping) -> list[str]:
    p1a_law = summary["fits"]["p1"]["p1a"]["scaling"]["power_law"]
    all6 = summary["fits"]["p1"]["p1c"]["all_six"]["power_law"]
    large3 = summary["fits"]["p1"]["p1c"]["large_three"]["power_law"]
    p1c_predictions = summary["fits"]["p1"]["p1c"]["predictions"]
    hb = summary["fits"]["p1"]["p1d"]["hyperball"]["power_law"]
    scores = summary["fits"]["p2"]["p2b"]["r_squared_comparison"]
    p2c = summary["fits"]["p2"]["p2c"]
    better_p1c = min(p1c_predictions, key=lambda item: item["gap_to_best_grid"])
    tightest = max(scores, key=scores.get)
    return [
        (
            f"P1a optimal LR has a weak source-budget trend "
            f"(exponent {p1a_law['exponent']:.3f}, R² {p1a_law['r_squared']:.3f})."
        ),
        (
            f"P1c source choice changes the extrapolation: all-six exponent "
            f"{all6['exponent']:.3f} versus large-three {large3['exponent']:.3f}; "
            f"{better_p1c['label']} had the smaller measured target gap "
            f"({better_p1c['gap_to_best_grid']:.6f})."
        ),
        (
            f"Hyperball source optima follow a comparatively regular decreasing law "
            f"(exponent {hb['exponent']:.3f}, R² {hb['r_squared']:.3f})."
        ),
        (
            f"Among P2 fitted coordinates, {tightest} has the tightest power law "
            f"(R² {scores[tightest]:.3f})."
        ),
        (
            f"At 2.4576B tokens the product-rule run's loss gaps are "
            f"{p2c['loss_gap_to_transferred']:.6f} to source-pair transfer and "
            f"{p2c['loss_gap_to_fixed_lr_best']:.6f} to the best fixed-LR WD baseline "
            f"(negative favors the product rule)."
        ),
    ]


def generate() -> dict:
    data = collect_data()
    report = {
        "status": "complete",
        "data_sources": {
            "supplied_parts": ["P1a", "P1b", "P1d", "P2a"],
            "queried_completed_subparts": ["P1c", "P1d", "P2c"],
        },
        "figures": {},
        "fits": {},
    }
    report["fits"]["p1"] = plot_p1(data, report)
    report["fits"]["p2"] = plot_p2(data, report)
    report["findings"] = _findings(report)
    report["bundle_count"] = len(report["figures"])
    report["artifact_count"] = sum(len(paths) for paths in report["figures"].values())
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(_json_ready(report), indent=2) + "\n")
    report["summary_path"] = str(SUMMARY_PATH)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        report = generate()
    except Exception as exc:
        failure = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(failure, indent=2))
        return 1
    print(json.dumps(_json_ready(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
