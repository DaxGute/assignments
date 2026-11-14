"""Targeted P1(d) controls on the hard zero-prefix, long-branch cell.

Predictions are locked under ``P1-d-*`` in the prediction ledger.

    uv run python -m experiments.a3_optimization.p1_training_controls runs
    uv run python -m experiments.a3_optimization.p1_training_controls runs --execute
    uv run python -m experiments.a3_optimization.p1_training_controls status
    uv run python -m experiments.a3_optimization.p1_training_controls measure --execute
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from train import TrainConfig, training_run_name


EXPERIMENT_KEY = "a3-mode-connectivity-controls"
ORDER_SEEDS = (1, 2)
BASELINE_B = 0.6237311363220215
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "a3-mode-connectivity-controls.json"


def _branches(base: TrainConfig, stop_step: int) -> list[TrainConfig]:
    return [
        replace(
            base,
            batch_order_seed=seed,
            stop_at_step=stop_step,
            keep_checkpoint_steps=(stop_step,),
            wandb_tags=(EXPERIMENT_KEY,),
        )
        for seed in ORDER_SEEDS
    ]


ARMS = {
    "lr-half": _branches(
        TrainConfig(lr_schedule="wsd0.0", learning_rate=0.0015),
        500,
    ),
    "lr-double": _branches(
        TrainConfig(lr_schedule="wsd0.0", learning_rate=0.006),
        500,
    ),
    # Match the baseline's 32.768M post-fork tokens: 250 * 128 * 1024.
    "batch2x-matched-tokens": _branches(
        TrainConfig(
            lr_schedule="wsd0.0",
            learning_rate=0.003,
            batch_size=128,
            num_micro_batches=2,
        ),
        250,
    ),
    # Unlike stop_at_step on a full 9375-update recipe, this makes cosine reach
    # zero at update 500. warmup_steps preserves the baseline warmup duration.
    "cos-branch": _branches(
        TrainConfig(
            lr_schedule="cos",
            learning_rate=0.003,
            num_train_sequences=32_000,
            num_epochs=1.0,
            warmup_steps=94,
        ),
        500,
    ),
}


def _pair(runs: list[TrainConfig]):
    step = runs[0].keep_checkpoint_steps[0]
    return (
        (training_run_name(runs[0]), step),
        (training_run_name(runs[1]), step),
    )


def _summarize(row: dict) -> dict:
    interpolation = row["interpolation"]
    endpoint_mean = 0.5 * (interpolation[0] + interpolation[-1])
    barrier = interpolation[2] - endpoint_mean
    return {
        "barrier": barrier,
        "delta_barrier_vs_baseline": BASELINE_B - barrier,
        "endpoint_mean_loss": endpoint_mean,
        "ensemble_loss": row["ensemble"],
        "ensemble_avg_gap": interpolation[2] - row["ensemble"],
        "row": row,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("runs", "status", "measure"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=2)
    args = parser.parse_args(argv)

    for arm, runs in ARMS.items():
        print(arm, _pair(runs))
    if args.stage == "status":
        from experiments.a3_optimization.helpers.results import classify_train_configs

        runs = [run for arm_runs in ARMS.values() for run in arm_runs]
        statuses = classify_train_configs(runs)
        print(f"completed={list(statuses.values()).count('completed')} / {len(runs)}")
        for name, status in statuses.items():
            print(status, name)
        return 0
    if not args.execute:
        print(f"[dry-run] {sum(map(len, ARMS.values()))} runs; no Modal submit")
        return 0

    if args.stage == "runs":
        from experiments.a3_optimization.helpers.results import (
            classify_train_configs,
            pending_configs,
        )
        from modal_train import launch_training_jobs

        runs = [run for arm_runs in ARMS.values() for run in arm_runs]
        pending = pending_configs(runs, classify_train_configs(runs))
        print(f"{len(runs) - len(pending)} complete; {len(pending)} missing")
        if pending:
            launch_training_jobs(pending, max_parallel_runs=args.max_parallel)
        return 0

    from experiments.a3_optimization.connectivity import measure_pairs_on_modal

    names = list(ARMS)
    rows = measure_pairs_on_modal([_pair(ARMS[name]) for name in names])
    payload = {
        "baseline": {
            "cell": {"x": 0, "y": 500, "batch_size": 64, "learning_rate": 0.003},
            "barrier": BASELINE_B,
            "kind": "measured",
        },
        "arms": {name: _summarize(row) for name, row in zip(names, rows)},
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
