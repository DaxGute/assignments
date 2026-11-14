"""Summarize and fit the complete 22-run P3 edge-of-stability study.

Writes direct measurements and simple fitted relationships separately. Missing
or running W&B runs remain explicit; they are never promoted to measurements.

    uv run python -m experiments.a3_optimization.analyze_p3
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
import wandb

from experiments.a3_optimization.p3_edge_of_stability import (
    FULL_BATCH_RUNS,
    MINIBATCH_RUNS,
    EXPLORATION_RUNS,
    FRACTION,
)
from experiments.a3_optimization.sharpness import history, summarize
from train import checked_train_config, training_run_name


RESULTS_PATH = Path(__file__).resolve().parent / "results" / "a3-p3-analysis.json"
PLOTS_DIR = RESULTS_PATH.parent / "a3-p3-analysis-plots"


def _run_state(config) -> tuple[str, str | None]:
    name = training_run_name(config)
    runs = wandb.Api().runs(
        f"{config.wandb_entity}/{config.wandb_project}",
        filters={"display_name": name},
    )
    if not runs:
        return "missing", None
    run = max(runs, key=lambda item: item.created_at)
    return run.state, run.url


def _median(values):
    return statistics.median(values) if values else None


def _cv(values):
    if not values:
        return None
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else None


def _full_batch(config, rows) -> dict:
    late = [row for row in rows if 1200 <= row["step"] <= 1500]
    return {
        "kind": "measured",
        "optimizer": config.optimizer_name,
        "num_sequences": config.num_train_sequences,
        "learning_rate": config.learning_rate,
        "lr_schedule": config.lr_schedule,
        "late_n": len(late),
        "late_lambda_max_median": _median(
            [row["lambda_max"] for row in late if row.get("lambda_max") is not None]
        ),
        "late_eta_lambda_median": _median(
            [row["eta_lambda"] for row in late if row.get("eta_lambda") is not None]
        ),
        "edge_summary": summarize(rows),
    }


def _minibatch(config, rows) -> dict:
    total_updates = int(FRACTION * (600_000 // config.batch_size))
    warmup_updates = round(total_updates * config.warmup_percent)
    post = [
        row["eta_lambda_pre"]
        for row in rows
        if row["step"] >= warmup_updates and row.get("eta_lambda_pre") is not None
    ]
    threshold = 2 * (1 + config.beta1) / (1 - config.beta1)
    median = _median(post)
    return {
        "kind": "measured",
        "optimizer": config.optimizer_name,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "beta1": config.beta1,
        "threshold": threshold,
        "post_warmup_n": len(post),
        "post_warmup_eta_lambda_pre_median": median,
        "post_warmup_eta_lambda_pre_cv": _cv(post),
        "threshold_normalized_median": None if median is None else median / threshold,
    }


def _power_fit(xs, ys) -> dict | None:
    pairs = sorted((x, y) for x, y in zip(xs, ys) if x and y and x > 0 and y > 0)
    if len(pairs) < 3:
        return None
    lx, ly = np.log(np.array(pairs, dtype=float)).T
    slope, intercept = np.polyfit(lx, ly, 1)
    fitted = intercept + slope * lx
    ss_res = float(np.square(ly - fitted).sum())
    ss_tot = float(np.square(ly - ly.mean()).sum())
    local_exponents = [
        float(np.log(y1 / y0) / np.log(x1 / x0))
        for (x0, y0), (x1, y1) in zip(pairs, pairs[1:])
    ]
    return {
        "kind": "fitted",
        "form": "y = coefficient * x ** exponent",
        "coefficient": float(np.exp(intercept)),
        "exponent": float(slope),
        "r2_log_space": None if ss_tot == 0 else 1 - ss_res / ss_tot,
        "n": len(pairs),
        "adjacent_pair_exponents": local_exponents,
        "adjacent_pair_exponent_range": [
            min(local_exponents),
            max(local_exponents),
        ],
    }


def _plots(measured: dict, fits: dict, histories: dict) -> None:
    """Write diagnostics only for axes with enough completed measurements."""
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    static = sorted(
        (
            row
            for row in measured.values()
            if row["optimizer"] == "sgd"
            and row["num_sequences"] == 32
            and row["lr_schedule"] == "constant"
        ),
        key=lambda row: row["learning_rate"],
    )
    if len(static) >= 2:
        lrs = np.array([row["learning_rate"] for row in static])
        lambdas = np.array([row["late_lambda_max_median"] for row in static])
        products = np.array([row["late_eta_lambda_median"] for row in static])

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
        axes[0].loglog(lrs, lambdas, marker="o", label="MEASURED")
        fit = fits.get("full_batch_lambda_vs_lr_n32")
        if fit:
            dense = np.geomspace(lrs.min(), lrs.max(), 200)
            axes[0].loglog(
                dense,
                fit["coefficient"] * dense ** fit["exponent"],
                linestyle="--",
                label=f"FITTED exponent {fit['exponent']:.3f}",
            )
        axes[0].loglog(lrs, 2 / lrs, linestyle=":", label=r"Ideal $2/\eta$")
        axes[0].set_xlabel(r"Learning rate $\eta$")
        axes[0].set_ylabel(r"Late median $\lambda_{\max}$")
        axes[0].set_title("Static-LR edge scaling")
        axes[0].legend(fontsize=8)

        axes[1].semilogx(lrs, products, marker="o")
        axes[1].axhline(2, linestyle=":", color="grey", label="Quadratic edge 2")
        axes[1].set_xlabel(r"Learning rate $\eta$")
        axes[1].set_ylabel(r"Late median $\eta\lambda_{\max}$")
        axes[1].set_title("Edge product and regime exit")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "p3_full_lr_scaling.png", dpi=180)
        plt.close(fig)

    by_n = sorted(
        (
            row
            for row in measured.values()
            if row["optimizer"] == "sgd"
            and row["learning_rate"] == 0.01
            and row["lr_schedule"] == "constant"
        ),
        key=lambda row: row["num_sequences"],
    )
    if len(by_n) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
        axes[0].plot(
            [row["num_sequences"] for row in by_n],
            [row["late_lambda_max_median"] for row in by_n],
            marker="o",
        )
        axes[0].set_xlabel("Full-batch sample count N")
        axes[0].set_ylabel(r"Late median $\lambda_{\max}$")
        axes[0].set_title("Sample-count sharpness")
        axes[1].plot(
            [row["num_sequences"] for row in by_n],
            [row["late_eta_lambda_median"] for row in by_n],
            marker="o",
        )
        axes[1].axhspan(1.5, 3.2, alpha=0.12, color="green", label="Locked band")
        axes[1].axhline(2, linestyle=":", color="grey", label="Quadratic edge")
        axes[1].set_xlabel("Full-batch sample count N")
        axes[1].set_ylabel(r"Late median $\eta\lambda_{\max}$")
        axes[1].set_title("Sample-count edge product")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "p3_full_sample_scaling.png", dpi=180)
        plt.close(fig)

    minibatch = sorted(
        (row for row in measured.values() if row["optimizer"] != "sgd"),
        key=lambda row: (row["batch_size"], row["learning_rate"], row["beta1"]),
    )
    if len(minibatch) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
        for batch in sorted({row["batch_size"] for row in minibatch}):
            rows = [
                row
                for row in minibatch
                if row["batch_size"] == batch and row["beta1"] == 0.9
            ]
            if rows:
                axes[0].plot(
                    [row["learning_rate"] for row in rows],
                    [row["post_warmup_eta_lambda_pre_median"] for row in rows],
                    marker="o",
                    label=f"batch={batch}",
                )
        axes[0].set_xscale("log")
        axes[0].axhline(38, linestyle=":", color="grey", label="β1=.9 threshold")
        axes[0].set_xlabel("Learning rate")
        axes[0].set_ylabel(r"Median $\eta\lambda^{pre}$")
        axes[0].set_title("AdamW LR × batch grid")
        axes[0].legend(fontsize=8)

        momentum = [
            row
            for row in minibatch
            if row["batch_size"] == 64 and row["learning_rate"] == 0.003
        ]
        axes[1].scatter(
            [row["beta1"] for row in momentum],
            [row["threshold_normalized_median"] for row in momentum],
        )
        axes[1].axhline(1, linestyle=":", color="grey")
        axes[1].set_xlabel(r"Momentum $\beta_1$")
        axes[1].set_ylabel("Median / predicted threshold")
        axes[1].set_title("Momentum-normalized edge")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "p3_minibatch_scaling.png", dpi=180)
        plt.close(fig)

    core_labels = (
        (FULL_BATCH_RUNS[0], "constant η=.01"),
        (FULL_BATCH_RUNS[1], "η×0.5 at 750"),
        (FULL_BATCH_RUNS[2], "η×2 at 750"),
    )
    core = []
    for raw_config, label in core_labels:
        config = checked_train_config(raw_config)
        rows = histories.get(training_run_name(config))
        if rows:
            core.append((rows, label))
    if len(core) == 3:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for rows, label in core:
            steps = [row["step"] for row in rows]
            axes[0, 0].plot(steps, [row["eta_lambda"] for row in rows], label=label)
            axes[0, 1].plot(steps, [row["lambda_max"] for row in rows], label=label)
            axes[1, 0].plot(
                [step for step, _ in rows.losses],
                [loss for _, loss in rows.losses],
                label=label,
                alpha=0.8,
            )
        clean = core[0][0]
        axes[1, 1].scatter(
            [row["train_loss"] for row in clean],
            [row["lambda_max"] for row in clean],
            c=[row["step"] for row in clean],
            s=18,
            cmap="viridis",
        )
        axes[0, 0].axhline(2, linestyle=":", color="black", label="edge=2")
        for ax in axes.flat[:3]:
            ax.axvline(750, linestyle="--", color="grey", alpha=0.5)
            ax.legend(fontsize=8)
        axes[0, 0].set(xlabel="Update", ylabel=r"$\eta\lambda_{\max}$",
                       title="Edge product and LR interventions")
        axes[0, 1].set(xlabel="Update", ylabel=r"$\lambda_{\max}$",
                       title="Sharpness retargeting")
        axes[1, 0].set(xlabel="Update", ylabel="Training loss",
                       title="Loss while riding the edge", yscale="log")
        axes[1, 1].set(xlabel="Training loss", ylabel=r"$\lambda_{\max}$",
                       title="Clean run: sharpening versus loss")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "p3_full_batch_trajectories.png", dpi=180)
        plt.close(fig)

    base_config = checked_train_config(MINIBATCH_RUNS[0])
    base = histories.get(training_run_name(base_config))
    if base:
        total_updates = int(FRACTION * (600_000 // base_config.batch_size))
        warmup_updates = round(total_updates * base_config.warmup_percent)
        post = [row for row in base if row["step"] >= warmup_updates]
        tokens_per_update = (
            base_config.batch_size * base_config.train_dataset.context_length
        )
        tokens_m = [row["step"] * tokens_per_update / 1e6 for row in post]
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
        axes[0].plot(tokens_m, [row["eta_lambda_pre"] for row in post], marker="o")
        axes[0].axhline(38, linestyle=":", color="black", label="β1=.9 threshold")
        axes[0].set(
            xlabel="Training tokens (millions)",
            ylabel=r"$\eta\lambda_{\max}^{pre}$",
            title="AdamW preconditioned edge",
        )
        axes[0].legend(fontsize=8)
        axes[1].scatter(
            [row["train_loss"] for row in post],
            [row["lambda_max_pre"] for row in post],
            c=tokens_m,
            s=22,
            cmap="viridis",
        )
        axes[1].set(
            xlabel="Training loss",
            ylabel=r"$\lambda_{\max}^{pre}$",
            title="Preconditioned sharpening versus loss",
            yscale="log",
        )
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / "p3_adamw_preconditioned_trajectory.png", dpi=180)
        plt.close(fig)


def main() -> int:
    configs = FULL_BATCH_RUNS + MINIBATCH_RUNS + EXPLORATION_RUNS
    measured, incomplete, histories = {}, {}, {}
    for raw_config in configs:
        config = checked_train_config(raw_config)
        name = training_run_name(config)
        state, url = _run_state(config)
        if state != "finished":
            incomplete[name] = {"state": state, "url": url}
            continue
        rows = history(config)
        histories[name] = rows
        summary = (
            _full_batch(config, rows)
            if config.optimizer_name == "sgd"
            else _minibatch(config, rows)
        )
        measured[name] = {"state": state, "url": url, **summary}

    static_n32 = [
        row
        for row in measured.values()
        if row["optimizer"] == "sgd"
        and row["num_sequences"] == 32
        and row["lr_schedule"] == "constant"
    ]
    adam_beta09 = [
        row
        for row in measured.values()
        if row["optimizer"] != "sgd" and row["beta1"] == 0.9
    ]
    momentum = [
        row
        for row in measured.values()
        if row["optimizer"] != "sgd"
        and row["batch_size"] == 64
        and row["learning_rate"] == 0.003
    ]
    fits = {
        "full_batch_lambda_vs_lr_n32": _power_fit(
            [row["learning_rate"] for row in static_n32],
            [row["late_lambda_max_median"] for row in static_n32],
        ),
        **{
            f"adam_product_vs_lr_batch{batch}": _power_fit(
                [row["learning_rate"] for row in adam_beta09
                 if row["batch_size"] == batch],
                [row["post_warmup_eta_lambda_pre_median"] for row in adam_beta09
                 if row["batch_size"] == batch],
            )
            for batch in sorted({row["batch_size"] for row in adam_beta09})
        },
        **{
            f"adam_product_vs_batch_lr{learning_rate}": _power_fit(
                [row["batch_size"] for row in adam_beta09
                 if row["learning_rate"] == learning_rate],
                [row["post_warmup_eta_lambda_pre_median"] for row in adam_beta09
                 if row["learning_rate"] == learning_rate],
            )
            for learning_rate in sorted(
                {row["learning_rate"] for row in adam_beta09}
            )
        },
        "adam_product_vs_momentum_threshold": _power_fit(
            [row["threshold"] for row in momentum],
            [row["post_warmup_eta_lambda_pre_median"] for row in momentum],
        ),
    }
    payload = {
        "measured": measured,
        "incomplete": incomplete,
        "fits": fits,
        "counts": {
            "expected": len(configs),
            "measured": len(measured),
            "incomplete": len(incomplete),
        },
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    _plots(measured, fits, histories)
    print(json.dumps(payload["counts"], indent=2))
    print(f"Wrote {RESULTS_PATH}")
    print(f"Wrote available diagnostics under {PLOTS_DIR}")
    return 0 if not incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
