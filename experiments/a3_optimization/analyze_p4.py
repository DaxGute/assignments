"""Generate P4 trajectory figures from measured Hessian JSON files.

    uv run python -m experiments.a3_optimization.analyze_p4
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb

from experiments.a3_optimization import hessian as hs
from experiments.a3_optimization.p4_inside_the_hessian import RUNS
from train import checked_train_config, training_run_name


RESULTS_DIR = Path(__file__).resolve().parent / "results" / "a3-inside-the-hessian"
PLOTS_DIR = RESULTS_DIR / "plots"
STEPS = (0, 3, 15, 61, 153, 305, 763, 1526, 3052, 4578, 6104, 7630, 9375)


def _load() -> list[dict]:
    rows = []
    for step in STEPS:
        path = RESULTS_DIR / f"a-adamw-{step}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(json.loads(path.read_text()))
    return rows


def main() -> int:
    rows = _load()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    tokens = [row["tokens"] for row in rows]
    normalized = [
        float(np.asarray(row["energy"]["family"]["per_probe"]).mean(0).sum())
        / row["size"]
        for row in rows
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(tokens, [row["lambda_max"] for row in rows], marker="o")
    axes[0].axhspan(4.5, 6.0, alpha=0.15, color="green",
                   label="Handout final band")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\lambda_{\max}$ (log scale)")
    axes[0].set_title("Hessian sharpness")
    axes[0].legend()
    axes[1].plot(tokens, normalized, marker="o")
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"$\mathrm{tr}(H^2)/d$ (log scale)")
    axes[1].set_title("Normalized squared sharpness")
    for ax in axes:
        ax.set_xlabel("Training tokens")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "p4_lambda_max_early.png", dpi=180)
    plt.close(fig)

    families = tuple(rows[-1]["families"])
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for family in families:
        values = [row["families"][family]["lambda_max"] for row in rows]
        ax.plot(tokens, values, marker="o", markersize=3, label=family)
    ax.set_yscale("log")
    ax.set_xlabel("Training tokens")
    ax.set_ylabel(r"Restricted $\lambda_{\max}$ (log scale)")
    ax.set_title("Curvature by parameter family")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "p4_family_lambda_early.png", dpi=180)
    plt.close(fig)

    density_rows = [row for row in rows if row.get("density_runs")]
    if len(density_rows) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
        for ax, row in zip(axes.flat, density_rows, strict=True):
            runs = row["density_runs"]
            lo = min(min(run["ritz"]) for run in runs)
            hi = max(max(run["ritz"]) for run in runs)
            span = max(hi - lo, 1e-8)
            grid = np.linspace(lo - 0.05 * span, hi + 0.05 * span, 800)
            curve = hs.density(runs, grid, width=0.015 * span)
            ax.plot(grid, curve)
            ax.axvline(0, color="grey", linestyle=":")
            ax.set_title(f"{row['tokens'] / 1e6:.0f}M tokens")
            ax.set_xlabel("Hessian eigenvalue")
            ax.set_ylabel("Estimated spectral density")
        fig.suptitle("Lanczos eigenvalue-density estimates")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "p4_spectrum_density.png", dpi=180)
        plt.close(fig)

    layer_rows = [rows[i] for i in (4, 7, 12)]  # 10M, 100M, 614M
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for row in layer_rows:
        energy = row["energy"]["layer"]
        keys = energy["blocks"]
        mean = np.asarray(energy["per_probe"]).mean(0)
        indices = [keys.index(f"L{i}") for i in range(8)]
        diagonal = [mean[i, i] for i in indices]
        ax.plot(range(8), diagonal, marker="o",
                label=f"{row['tokens'] / 1e6:.0f}M tokens")
    ax.set_yscale("log")
    ax.set_xlabel("Transformer layer depth")
    ax.set_ylabel(r"Within-layer $\|H_{ll}\|_F^2$")
    ax.set_title("Curvature depth profile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "p4_layer_curvature_depth.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for part in ("family", "layer"):
        fractions = []
        for row in rows:
            probes = np.asarray(row["energy"][part]["per_probe"])
            fractions.append(hs.within_frac_from_probes(probes))
        ax.plot(tokens, fractions, marker="o", label=f"structured {part}")
    for step in (1526, 9375):
        path = RESULTS_DIR / f"a-random-energy-{step}.json"
        if not path.exists():
            continue
        random_row = json.loads(path.read_text())
        for part, marker in (("family", "x"), ("layer", "+")):
            ax.scatter(
                random_row["tokens"],
                random_row["energy"][part]["random_within_frac"],
                marker=marker,
                s=70,
                label=f"matched random {part}" if step == 1526 else None,
            )
    ax.set_xlabel("Training tokens")
    ax.set_ylabel(r"Within-block fraction of $\mathrm{tr}(H^2)$")
    ax.set_title("Structured versus matched-random Hessian block energy")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "p4_block_energy.png", dpi=180)
    plt.close(fig)

    rescale = []
    for scale, name in (
        (0.25, "b-rescale-head-x0.25.json"),
        (0.5, "b-rescale-head-x0.5.json"),
        (1.0, "b-rescale-reference.json"),
        (2.0, "b-rescale-head-x2.json"),
    ):
        path = RESULTS_DIR / name
        if path.exists():
            rescale.append((scale, json.loads(path.read_text())))
    if len(rescale) == 4:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
        scales = [scale for scale, _ in rescale]
        axes[0].plot(
            scales, [row["probe_loss"] for _, row in rescale], marker="o"
        )
        axes[1].plot(
            scales, [row["lambda_max"] for _, row in rescale], marker="o"
        )
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.axvline(1, linestyle=":", color="grey")
            ax.set_xlabel("LM-head scale")
        axes[0].set_ylabel("Probe loss")
        axes[0].set_title("Head scaling changes the function")
        axes[1].set_ylabel(r"$\lambda_{\max}$")
        axes[1].set_title("Sharpness is not a quality proxy")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "p4_head_rescale_response.png", dpi=180)
        plt.close(fig)

    reference_lambda = json.loads(
        (RESULTS_DIR / "b-rescale-reference.json").read_text()
    )["lambda_max"]
    training_rows = []
    for raw_config in RUNS:
        config = checked_train_config(raw_config)
        run_name = training_run_name(config)
        path = RESULTS_DIR / f"b-final-{run_name}.json"
        if not path.exists():
            continue
        measurement = json.loads(path.read_text())
        candidates = wandb.Api().runs(
            f"{config.wandb_entity}/{config.wandb_project}",
            filters={"display_name": run_name},
        )
        finished = [run for run in candidates if run.state == "finished"]
        if not finished:
            continue
        run = max(finished, key=lambda item: item.created_at)
        val_loss = float(run.summary["val_loss"])
        ratio = measurement["lambda_max"] / reference_lambda
        training_rows.append(
            {
                "kind": "measured",
                "run_name": run_name,
                "wandb_id": run.id,
                "wandb_url": run.url,
                "intervention": (
                    "frozen-norm"
                    if "frozen-norm" in (config.run_name_suffix or "")
                    else "persistent-head"
                    if "persistent-head" in (config.run_name_suffix or "")
                    else "persistent"
                    if "persistent" in (config.run_name_suffix or "")
                    else "initial-only"
                ),
                "gauge_factor": config.model_builder_kwargs.get(
                    "gauge_factor",
                    config.model_builder_kwargs.get("head_scale"),
                ),
                "val_loss": val_loss,
                "lambda_max": measurement["lambda_max"],
                "lambda_ratio": ratio,
                "assignment_criterion": (
                    2.925 <= val_loss <= 2.931
                    and (ratio >= 2.0 or ratio <= 0.5)
                ),
            }
        )
    training_payload = {
        "kind": "measured",
        "reference_lambda_max": reference_lambda,
        "rows": training_rows,
        "assignment_criterion_met": any(
            row["assignment_criterion"] for row in training_rows
        ),
    }
    (RESULTS_DIR / "a3-p4-training-analysis.json").write_text(
        json.dumps(training_payload, indent=2) + "\n"
    )
    if training_rows:
        labels = [
            f"{row['intervention']} c={row['gauge_factor']:g}"
            for row in training_rows
        ]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
        axes[0].bar(labels, [row["val_loss"] for row in training_rows])
        axes[0].axhspan(2.925, 2.931, alpha=0.18, color="green",
                       label="required validation band")
        axes[0].set_ylabel("Final validation loss")
        axes[0].legend(fontsize=8)
        axes[1].bar(labels, [row["lambda_ratio"] for row in training_rows])
        axes[1].axhline(2, linestyle="--", color="green", label="2× target")
        axes[1].axhline(0.5, linestyle="--", color="green")
        axes[1].set_ylabel("Final λmax / reference λmax")
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.tick_params(axis="x", rotation=35)
        fig.suptitle("Training-induced parameterization and final sharpness")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "p4_training_sharpness.png", dpi=180)
        plt.close(fig)

    continues = {}
    refreshes = {}
    for arm in ("full", "top", "removed"):
        path = RESULTS_DIR / f"c-{arm}.jsonl"
        if path.exists():
            records = [
                json.loads(line) for line in path.read_text().splitlines() if line
            ]
            continues[arm] = [row for row in records if row.get("kind") == "step"]
            refreshes[arm] = [row for row in records if row.get("kind") == "refresh"]
    if len(continues) == 3:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
        final_vals = []
        for arm, records in continues.items():
            x = np.array([row["step"] - records[0]["step"] + 1 for row in records])
            loss = np.array([row["loss"] for row in records])
            fraction = np.array([row["top_fraction"] for row in records])
            window = 20
            kernel = np.ones(window) / window
            smooth_x = x[window - 1:]
            axes[0].plot(smooth_x, np.convolve(loss, kernel, mode="valid"),
                         label=arm)
            axes[1].plot(
                smooth_x, np.convolve(fraction, kernel, mode="valid"), label=arm
            )
            final_vals.append(records[-1]["val_loss"])
        axes[0].set(
            xlabel="Continuation update",
            ylabel="Training loss (20-step mean)",
            title="Subspace-restricted optimization",
        )
        axes[1].set(
            xlabel="Continuation update",
            ylabel=r"$\|P_{10}g\|/\|g\|$ (20-step mean)",
            title="Gradient fraction in refreshed top-10",
        )
        axes[0].legend(fontsize=8)
        axes[1].legend(fontsize=8)
        axes[2].bar(tuple(continues), final_vals)
        axes[2].set(ylabel="Final validation loss", title="Endpoint ranking")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "p4_subspace_continue.png", dpi=180)
        plt.close(fig)

    if len(refreshes) == 3:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for arm, records in refreshes.items():
            ax.plot(
                [row["step"] for row in records],
                [row["lambda_max"] for row in records],
                marker="o",
                label=f"{arm} λmax",
            )
        baseline = refreshes["full"]
        ax.plot(
            [row["step"] for row in baseline],
            [row["heavy_ball_threshold"] for row in baseline],
            linestyle="--",
            color="black",
            label="heavy-ball threshold",
        )
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel(r"$\lambda_{\max}$")
        ax.set_title("Subspace continuation sharpness versus stability threshold")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "p4_subspace_sharpness.png", dpi=180)
        plt.close(fig)

    k30 = {}
    for arm in ("top", "removed"):
        path = RESULTS_DIR / f"c-k30-{arm}.jsonl"
        if path.exists():
            records = [
                json.loads(line) for line in path.read_text().splitlines() if line
            ]
            k30[arm] = [row for row in records if row.get("kind") == "step"]
    if len(continues) == 3 and len(k30) == 2:
        labels = ("full k10", "top k10", "removed k10", "top k30", "removed k30")
        records = (
            continues["full"],
            continues["top"],
            continues["removed"],
            k30["top"],
            k30["removed"],
        )
        mean_fractions = [np.mean([r["top_fraction"] for r in rows]) for rows in records]
        final_losses = [rows[-1]["val_loss"] for rows in records]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
        axes[0].bar(labels, mean_fractions)
        axes[0].axhspan(0.12, 0.25, alpha=0.15, color="green",
                       label="Locked k30 band")
        axes[0].set(ylabel="Mean logged top fraction", title="k=10 vs k=30")
        axes[0].legend(fontsize=8)
        axes[1].bar(labels, final_losses)
        axes[1].set(ylabel="Final validation loss", title="Subspace endpoint regime")
        for ax in axes:
            ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "p4_subspace_k30.png", dpi=180)
        plt.close(fig)

    print(f"Wrote P4 diagnostic plots under {PLOTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
