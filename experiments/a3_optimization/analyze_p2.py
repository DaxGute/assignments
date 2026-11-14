"""Materialize P2 histories, metrics, and figures from finished W&B runs.

    uv run python -m experiments.a3_optimization.analyze_p2

This is analysis-only: it never submits training.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import wandb

from experiments.a3_optimization.p2_river_valley import (
    HORIZON_BUDGET_RUNS,
    HORIZON_FREE_RUNS,
    REWARM_RUNS,
    SCHEDULE_RUNS,
    SCHEDULE_TUNING_RUNS,
)
from train import checked_train_config, training_run_name


RESULTS_PATH = Path(__file__).resolve().parent / "results" / "a3-p2-analysis.json"
PLOTS_DIR = RESULTS_PATH.parent / "a3-p2-analysis-plots"
KEYS = (
    "train_loss",
    "val_loss",
    "val_ema0.99_loss",
    "val_ema0.999_loss",
    "learning_rate",
)
CORE_LABELS = ("wsd0.0", "linear", "cos", "wsd0.2")
RUN_SPECS = [
    *zip(CORE_LABELS, SCHEDULE_RUNS, strict=True),
    *[
        (f"tune-{config.lr_schedule}-lr{config.learning_rate:g}", config)
        for config in SCHEDULE_TUNING_RUNS
    ],
    *zip(("dip", "dip-deep", "dip-shallow", "dip-long"), REWARM_RUNS, strict=True),
    *zip(("invsqrt", "sgdr2000"), HORIZON_FREE_RUNS, strict=True),
    *[
        (
            f"budget{config.num_train_sequences * 1024 // 1_000_000}M-"
            f"{config.lr_schedule}",
            config,
        )
        for config in HORIZON_BUDGET_RUNS
    ],
]


def _latest_run(config):
    runs = wandb.Api().runs(
        f"{config.wandb_entity}/{config.wandb_project}",
        filters={"display_name": training_run_name(config)},
    )
    if not runs:
        return None
    return max(runs, key=lambda item: item.created_at)


def _series(run, key: str) -> list[list[float]]:
    points = []
    for row in run.scan_history(keys=["optimizer_step", key], page_size=10_000):
        step, value = row.get("optimizer_step"), row.get(key)
        if step is None or value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            points.append([int(step) + 1, value])
    return points


def _final(history: dict, key: str):
    values = history.get(key, [])
    return values[-1][1] if values else None


def _at_step(series: list[list[float]], step: int):
    exact = {point_step: value for point_step, value in series}
    return exact.get(step)


def _plots(runs: dict) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for label in ("wsd0.0", "linear", "cos", "wsd0.2"):
        history = runs[label]["history"]
        for ax, key in zip(
            axes, ("learning_rate", "train_loss", "val_loss"), strict=True
        ):
            ax.plot(
                [x for x, _ in history[key]],
                [y for _, y in history[key]],
                label=label,
            )
    for ax, title, ylabel in zip(
        axes,
        ("Learning-rate schedules", "Training loss", "Validation loss"),
        ("Learning rate", "Next-token loss", "Validation loss"),
        strict=True,
    ):
        ax.set_title(title)
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "p2_schedule_diagnostics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for label in ("wsd0.0", "linear", "cos", "wsd0.2"):
        history = runs[label]["history"]
        axes[0].plot(
            [x for x, _ in history["val_loss"]],
            [y for _, y in history["val_loss"]],
            label=label,
        )
        best_key = min(
            ("val_ema0.99_loss", "val_ema0.999_loss"),
            key=lambda key: _final(history, key),
        )
        axes[1].plot(
            [x for x, _ in history[best_key]],
            [y for _, y in history[best_key]],
            label=f"{label} ({best_key[7:-5]})",
        )
    axes[0].set_title("Raw validation loss")
    axes[1].set_title("Best final EMA per schedule")
    for ax in axes:
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Validation loss")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "p2_schedule_ema.png", dpi=180)
    plt.close(fig)

    dip = runs["dip"]["history"]
    control = runs["wsd0.2"]["history"]
    control_by_step = dict(control["val_loss"])
    regrets = [
        (step, value - control_by_step[step])
        for step, value in dip["val_loss"]
        if step in control_by_step
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot([x for x, _ in regrets], [y for _, y in regrets])
    axes[0].axhline(0, linestyle=":", color="grey")
    axes[0].set_title("Temporary-dip regret vs matched control")
    axes[0].set_xlabel("Optimizer step")
    axes[0].set_ylabel("Validation-loss regret")
    for label in ("wsd0.2", "dip"):
        series = runs[label]["history"]["learning_rate"]
        axes[1].plot([x for x, _ in series], [y for _, y in series], label=label)
    axes[1].set_title("Control and dip learning rates")
    axes[1].set_xlabel("Optimizer step")
    axes[1].set_ylabel("Learning rate")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "p2_dip_rewarm.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for schedule in CORE_LABELS:
        points = [
            (runs[schedule]["config"]["learning_rate"],
             runs[schedule]["final"]["val_loss"])
        ]
        for lr in (0.0015, 0.006):
            row = runs[f"tune-{schedule}-lr{lr:g}"]
            points.append((row["config"]["learning_rate"], row["final"]["val_loss"]))
        points.sort()
        ax.plot([x for x, _ in points], [y for _, y in points],
                marker="o", label=schedule)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Peak learning rate")
    ax.set_ylabel("Final validation loss")
    ax.set_title("Schedule conclusions after peak-LR tuning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "p2_lr_tuning.png", dpi=180)
    plt.close(fig)

    control_by_step = dict(runs["wsd0.2"]["history"]["val_loss"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label in ("dip", "dip-deep", "dip-shallow", "dip-long"):
        series = runs[label]["history"]["val_loss"]
        regret = [
            (step, value - control_by_step[step])
            for step, value in series
            if step in control_by_step
        ]
        ax.plot([x for x, _ in regret], [y for _, y in regret], label=label)
    ax.axhline(0, linestyle=":", color="grey")
    ax.set_xlabel("Optimizer step")
    ax.set_ylabel("Validation-loss regret")
    ax.set_title("Cooldown depth and duration sweep")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "p2_dip_sweep.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    budgets = (153, 307, 614)
    for schedule in ("linear", "cos", "invsqrt", "sgdr2000"):
        values = [
            runs[f"budget{budget}M-{schedule}"]["final"]["val_loss"]
            for budget in budgets[:2]
        ]
        values.append(runs[schedule]["final"]["val_loss"])
        ax.plot(budgets, values, marker="o", label=schedule)
    ax.set_xlabel("Training tokens (millions)")
    ax.set_ylabel("Final validation loss")
    ax.set_title("Fixed horizon-free schedules across stopping budgets")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "p2_horizon_budgets.png", dpi=180)
    plt.close(fig)


def main() -> int:
    loaded, incomplete = {}, {}
    for label, raw_config in RUN_SPECS:
        config = checked_train_config(raw_config)
        run = _latest_run(config)
        if run is None or run.state != "finished":
            incomplete[label] = {
                "state": None if run is None else run.state,
                "url": None if run is None else run.url,
            }
            continue
        history = {key: _series(run, key) for key in KEYS}
        loaded[label] = {
            "kind": "measured",
            "name": run.name,
            "run_id": run.id,
            "url": run.url,
            "config": {
                "lr_schedule": config.lr_schedule,
                "learning_rate": config.learning_rate,
                "num_train_sequences": config.num_train_sequences,
                "lr_dip": config.lr_dip,
                "ema_decays": list(config.ema_decays),
                "warmup_steps": config.warmup_steps,
                "min_lr_ratio": config.min_lr_ratio,
            },
            "final": {key: _final(history, key) for key in KEYS},
            "history": history,
        }

    derived = {}
    if not incomplete:
        linear = loaded["linear"]["final"]["val_loss"]
        wsd = loaded["wsd0.2"]["final"]["val_loss"]
        derived["schedule_regret_vs_linear"] = {
            label: loaded[label]["final"]["val_loss"] - linear
            for label in ("wsd0.0", "linear", "cos", "wsd0.2")
        }
        best_ema = {
            label: min(
                loaded[label]["final"]["val_ema0.99_loss"],
                loaded[label]["final"]["val_ema0.999_loss"],
            )
            for label in ("wsd0.0", "linear", "cos", "wsd0.2")
        }
        raw_gap = loaded["wsd0.0"]["final"]["val_loss"] - linear
        ema_gap = best_ema["wsd0.0"] - best_ema["linear"]
        derived["ema"] = {
            "best_final_by_schedule": best_ema,
            "raw_gap": raw_gap,
            "best_ema_gap": ema_gap,
            "gap_closed_fraction": 1 - ema_gap / raw_gap,
        }
        dip_series = loaded["dip"]["history"]["val_loss"]
        control_series = loaded["wsd0.2"]["history"]["val_loss"]
        derived["dip"] = {
            "final_regret": loaded["dip"]["final"]["val_loss"] - wsd,
            "step_4230_regret": _at_step(dip_series, 4230)
            - _at_step(control_series, 4230),
            "step_4700_regret": _at_step(dip_series, 4700)
            - _at_step(control_series, 4700),
        }
        derived["horizon_free"] = {
            "invsqrt_penalty_vs_linear": loaded["invsqrt"]["final"]["val_loss"]
            - linear,
            "sgdr_penalty_vs_linear": loaded["sgdr2000"]["final"]["val_loss"]
            - linear,
            "absolute_gap": abs(
                loaded["sgdr2000"]["final"]["val_loss"]
                - loaded["invsqrt"]["final"]["val_loss"]
            ),
        }
        derived["schedule_lr_tuning"] = {}
        for schedule in CORE_LABELS:
            losses = {
                "0.003": loaded[schedule]["final"]["val_loss"],
                **{
                    f"{lr:g}": loaded[f"tune-{schedule}-lr{lr:g}"]["final"]["val_loss"]
                    for lr in (0.0015, 0.006)
                },
            }
            best_lr = min(losses, key=losses.get)
            derived["schedule_lr_tuning"][schedule] = {
                "loss_by_lr": losses,
                "best_lr": float(best_lr),
                "best_loss": losses[best_lr],
            }

        control_by_step = dict(loaded["wsd0.2"]["history"]["val_loss"])
        derived["dip_sweep"] = {}
        for label in ("dip", "dip-deep", "dip-shallow", "dip-long"):
            regrets = [
                value - control_by_step[step]
                for step, value in loaded[label]["history"]["val_loss"]
                if step in control_by_step
            ]
            derived["dip_sweep"][label] = {
                "minimum_regret": min(regrets),
                "final_regret": loaded[label]["final"]["val_loss"] - wsd,
            }

        derived["horizon_budgets"] = {}
        for budget in (153, 307):
            losses = {
                schedule: loaded[f"budget{budget}M-{schedule}"]["final"]["val_loss"]
                for schedule in ("linear", "cos", "invsqrt", "sgdr2000")
            }
            tuned = min(losses["linear"], losses["cos"])
            derived["horizon_budgets"][f"{budget}M"] = {
                "losses": losses,
                "best_tuned_loss": tuned,
                "invsqrt_penalty": losses["invsqrt"] - tuned,
                "sgdr_penalty": losses["sgdr2000"] - tuned,
                "hf_absolute_gap": abs(losses["sgdr2000"] - losses["invsqrt"]),
            }
        _plots(loaded)

    payload = {
        "runs": loaded,
        "incomplete": incomplete,
        "derived": derived,
        "counts": {"expected": len(RUN_SPECS), "measured": len(loaded)},
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"counts": payload["counts"], "derived": derived}, indent=2))
    print(f"Wrote {RESULTS_PATH}")
    if not incomplete:
        print(f"Wrote six plots under {PLOTS_DIR}")
    return 0 if not incomplete else 2


if __name__ == "__main__":
    raise SystemExit(main())
