"""Materialize P1 tables, summaries, and figures from the measured JSON.

    uv run python -m experiments.a3_optimization.analyze_p1

The script never launches training. It preserves direct measurements separately
from the displayed fitted curves.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.a3_optimization.p1_mode_connectivity import (
    BRANCH_STEPS,
    PREFIX_STEPS,
    RESULTS_PATH,
)


OUTPUT_DIR = RESULTS_PATH.parent / "a3-mode-connectivity-analysis"
TABLE_PATH = OUTPUT_DIR / "barrier-table.csv"
SUMMARY_PATH = OUTPUT_DIR / "barrier-summary.json"
PLOTS_DIR = OUTPUT_DIR / "plots"


def _cells() -> list[dict]:
    measured = json.loads(RESULTS_PATH.read_text())
    expected = len(PREFIX_STEPS) * len(BRANCH_STEPS)
    if len(measured) != expected:
        raise ValueError(f"Expected {expected} measured cells, found {len(measured)}")

    cells = []
    for (x, y), row in zip(
        ((x, y) for x in PREFIX_STEPS for y in BRANCH_STEPS),
        measured,
        strict=True,
    ):
        interpolation = row["interpolation"]
        endpoint_mean = 0.5 * (interpolation[0] + interpolation[-1])
        cells.append(
            {
                "x": x,
                "y": y,
                "endpoint_1": interpolation[0],
                "endpoint_2": interpolation[-1],
                "endpoint_mean": endpoint_mean,
                "weight_average": interpolation[2],
                "ensemble": row["ensemble"],
                "barrier": interpolation[2] - endpoint_mean,
                "ensemble_avg_gap": interpolation[2] - row["ensemble"],
                "interpolation": interpolation,
            }
        )
    return cells


def _write_tables(cells: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = (
        "x",
        "y",
        "endpoint_1",
        "endpoint_2",
        "endpoint_mean",
        "weight_average",
        "ensemble",
        "barrier",
        "ensemble_avg_gap",
    )
    with TABLE_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in cells)

    barriers = np.array([row["barrier"] for row in cells])
    failures = [
        {"x": row["x"], "y": row["y"], "barrier": row["barrier"]}
        for row in cells
        if row["barrier"] > 0.01
    ]
    summary = {
        "kind": "measured",
        "source": str(RESULTS_PATH.relative_to(RESULTS_PATH.parents[2])),
        "cells": len(cells),
        "barrier_min": float(barriers.min()),
        "barrier_max": float(barriers.max()),
        "barrier_mean": float(barriers.mean()),
        "averaging_works_count": int((barriers <= 0.01).sum()),
        "criterion": "barrier <= 0.01",
        "failures": failures,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")


def _plot(cells: list[dict]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    lookup = {(row["x"], row["y"]): row for row in cells}
    matrix = np.array(
        [[lookup[(x, y)]["barrier"] for y in BRANCH_STEPS] for x in PREFIX_STEPS]
    )
    fig, ax = plt.subplots(figsize=(7.3, 4.8))
    image = ax.imshow(matrix, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(BRANCH_STEPS)), BRANCH_STEPS)
    ax.set_yticks(range(len(PREFIX_STEPS)), PREFIX_STEPS)
    ax.set_xlabel("Branch duration y (updates)")
    ax.set_ylabel("Shared prefix x (updates)")
    ax.set_title("Measured midpoint barrier B(x, y)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="Barrier B")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "B_heatmap.png", dpi=180)
    plt.close(fig)

    fields = (
        ("endpoint_1", "Endpoint 1 validation loss"),
        ("endpoint_2", "Endpoint 2 validation loss"),
        ("ensemble", "Prediction-ensemble validation loss"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.5), sharex=True, sharey=True)
    values = np.array(
        [
            lookup[(x, y)][field]
            for field, _ in fields
            for x in PREFIX_STEPS
            for y in BRANCH_STEPS
        ]
    )
    vmin, vmax = float(values.min()), float(values.max())
    for ax, (field, title) in zip(axes, fields, strict=True):
        matrix = np.array(
            [[lookup[(x, y)][field] for y in BRANCH_STEPS] for x in PREFIX_STEPS]
        )
        image = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(BRANCH_STEPS)), BRANCH_STEPS)
        ax.set_yticks(range(len(PREFIX_STEPS)), PREFIX_STEPS)
        ax.set_xlabel("Branch duration y")
        ax.set_title(title)
    axes[0].set_ylabel("Shared prefix x")
    fig.colorbar(image, ax=axes, label="Validation loss", shrink=0.85)
    fig.suptitle("Endpoint and ensemble losses across the mode-connectivity grid")
    fig.savefig(PLOTS_DIR / "endpoint_ensemble_heatmaps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.3, 4.8))
    for x in PREFIX_STEPS:
        values = [lookup[(x, y)]["barrier"] for y in BRANCH_STEPS]
        ax.plot(BRANCH_STEPS, values, marker="o", label=f"x={x}")
    y_dense = np.linspace(100, 500, 200)
    ax.plot(
        y_dense,
        -0.1901 + 0.001628 * y_dense,
        linestyle="--",
        color="black",
        label="Fit A growth regime (x=0)",
    )
    ax.axhline(0.01, color="grey", linestyle=":", label="B=0.01")
    ax.set_xlabel("Branch duration y (updates)")
    ax.set_ylabel("Barrier B")
    ax.set_title("Barrier growth after the fork")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "B_vs_y_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.3, 4.8))
    observed = np.array([lookup[(x, 500)]["barrier"] for x in PREFIX_STEPS])
    ax.scatter(PREFIX_STEPS, observed, label="MEASURED", zorder=3)
    x_dense = np.linspace(0, 1000, 300)
    ax.plot(
        x_dense,
        -0.0663 + 0.6870 * np.exp(-0.02316 * x_dense),
        label="FITTED: -0.0663 + 0.687 exp(-0.02316 x)",
    )
    ax.axhline(0.01, color="grey", linestyle=":", label="B=0.01")
    ax.set_xlabel("Shared prefix x (updates)")
    ax.set_ylabel("Barrier B at y=500")
    ax.set_title("Shared training collapses the long-branch barrier")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "B_vs_x_y500.png", dpi=180)
    plt.close(fig)

    barriers = np.array([row["barrier"] for row in cells])
    weight_average = np.array([row["weight_average"] for row in cells])
    ensemble = np.array([row["ensemble"] for row in cells])
    gap = weight_average - ensemble
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].scatter(ensemble, weight_average, c=barriers, cmap="coolwarm")
    bounds = [min(ensemble.min(), weight_average.min()), max(ensemble.max(), weight_average.max())]
    axes[0].plot(bounds, bounds, color="grey", linestyle=":")
    axes[0].set_xlabel("Prediction-ensemble loss")
    axes[0].set_ylabel("Weight-average loss")
    axes[0].set_title("Function vs parameter averaging")
    axes[1].scatter(barriers, gap, label="MEASURED")
    b_dense = np.linspace(barriers.min(), barriers.max(), 200)
    axes[1].plot(b_dense, 0.0897 + 1.0328 * b_dense, label="FITTED Fit C")
    axes[1].set_xlabel("Barrier B")
    axes[1].set_ylabel("Gap G")
    axes[1].set_title("Ensemble advantage tracks barrier")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "ensemble_vs_avg.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    alphas = (0, 0.25, 0.5, 0.75, 1)
    for cell in ((0, 500), (50, 500), (1000, 500)):
        row = lookup[cell]
        ax.plot(alphas, row["interpolation"], marker="o",
                label=f"(x,y)={cell}, B={row['barrier']:.3f}")
        ax.axhline(row["ensemble"], linestyle=":", alpha=0.6)
    ax.set_xlabel(r"Interpolation coefficient $\alpha$")
    ax.set_ylabel("Validation loss")
    ax.set_title("Representative parameter-interpolation curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "interpolation_curves.png", dpi=180)
    plt.close(fig)


def main() -> int:
    cells = _cells()
    _write_tables(cells)
    _plot(cells)
    print(f"Wrote {TABLE_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote six plots under {PLOTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
