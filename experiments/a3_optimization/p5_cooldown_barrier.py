"""P5 exploratory: post-separation cooldown barrier (arms R / C / K).

Maps to ledger ``P5-*`` (see docs/a3-p5-exploratory-design.md). Arm R reuses
Batch 1 MEASURED ``B_R``; arms C/K continue from frozen parent checkpoints.

    uv run python -m experiments.a3_optimization.p5_cooldown_barrier continues --dry-run
    uv run python -m experiments.a3_optimization.p5_cooldown_barrier continues --execute
    uv run python -m experiments.a3_optimization.p5_cooldown_barrier measure --dry-run
    uv run python -m experiments.a3_optimization.p5_cooldown_barrier measure
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from train import TrainConfig, training_run_name

# ---------------------------------------------------------------------------
# FROZEN — edit only before first --execute; then lock (internal/p5-run-prep.md §3)
# ---------------------------------------------------------------------------
COOL_STEPS = 100
FORK_STEP = 500
STOP_STEP = FORK_STEP + COOL_STEPS  # 600
ORDER_SEEDS = (1, 2)
LEARNING_RATE = 0.003
# Schedule horizon for lr_dip fractions (default d8: 600_000 seq / batch 64).
TOTAL_STEPS = 600_000 // 64  # 9375
# Arm C: dip peak→0 over the continue window; end=1 so no rewarm before STOP_STEP.
COOL_LR_DIP = (FORK_STEP / TOTAL_STEPS, STOP_STEP / TOTAL_STEPS, 1.0, 0.0)
PARENT_RUNS = {
    1: "model-d8-lr0.003-tok614M-wsd0.0-order1-stop500",
    2: "model-d8-lr0.003-tok614M-wsd0.0-order2-stop500",
}
# Arm R — Batch 1 MEASURED cell (0,500); not a new measurement.
B_R = 0.6237
L_R_MEAN = None  # optional; filled from Batch 1 JSON by operator if needed for dL
EXPERIMENT_KEY = "a3-p5-exploratory"
RESULTS_PATH = Path(__file__).resolve().parent / "results" / "a3-p5-cooldown-barrier.json"
LEDGER_IDS = (
    "P5-B-cool-x0",
    "P5-dB-cool",
    "P5-B-const-ctrl",
    "P5-rank-arms",
    "P5-G-cool-fitC",
    "P5-L-cool-dir",
    "P5-emergent-basin",
)

BASE = TrainConfig(lr_schedule="wsd0.0", learning_rate=LEARNING_RATE)


def _continue(seed: int, *, arm: str) -> TrainConfig:
    tags = (EXPERIMENT_KEY, "PREDICTION", "cooldown" if arm == "C" else "const-control")
    kwargs = dict(
        batch_order_seed=seed,
        fork_from_run=PARENT_RUNS[seed],
        fork_from_step=FORK_STEP,
        stop_at_step=STOP_STEP,
        keep_checkpoint_steps=(STOP_STEP,),
        wandb_tags=tags,
    )
    if arm == "C":
        kwargs["lr_dip"] = COOL_LR_DIP
    return replace(BASE, **kwargs)


COOL_RUNS = [_continue(s, arm="C") for s in ORDER_SEEDS]
CONST_RUNS = [_continue(s, arm="K") for s in ORDER_SEEDS]
CONTINUE_RUNS = COOL_RUNS + CONST_RUNS


def cool_pair():
    return (
        (training_run_name(COOL_RUNS[0]), STOP_STEP),
        (training_run_name(COOL_RUNS[1]), STOP_STEP),
    )


def const_pair():
    return (
        (training_run_name(CONST_RUNS[0]), STOP_STEP),
        (training_run_name(CONST_RUNS[1]), STOP_STEP),
    )


def _barrier_and_gap(row: dict) -> tuple[float, float, float, float]:
    """Return B, G, L1, L2 from a connectivity measure row."""
    interp = row["interpolation"]
    l0, l_mid, l1 = interp[0], interp[2], interp[-1]
    barrier = l_mid - 0.5 * (l0 + l1)
    gap = l_mid - row["ensemble"]
    return barrier, gap, l0, l1


def print_frozen():
    print("FROZEN", json.dumps({
        "COOL_STEPS": COOL_STEPS,
        "FORK_STEP": FORK_STEP,
        "STOP_STEP": STOP_STEP,
        "ORDER_SEEDS": list(ORDER_SEEDS),
        "LEARNING_RATE": LEARNING_RATE,
        "TOTAL_STEPS": TOTAL_STEPS,
        "COOL_LR_DIP": list(COOL_LR_DIP),
        "PARENT_RUNS": PARENT_RUNS,
        "B_R": B_R,
        "LEDGER_IDS": list(LEDGER_IDS),
        "RESULTS_PATH": str(RESULTS_PATH),
    }, indent=2))
    for arm, runs in (("C", COOL_RUNS), ("K", CONST_RUNS)):
        for cfg in runs:
            print(f"  arm={arm}  {training_run_name(cfg)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("continues", "measure"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=2)
    args = parser.parse_args(argv)
    if args.execute and args.dry_run:
        parser.error("pass only one of --dry-run and --execute")
    # Default dry-run unless --execute (no accidental Modal submits).
    execute = bool(args.execute)
    print_frozen()
    if args.stage == "continues":
        if not execute:
            print(
                f"[dry-run] would launch {len(CONTINUE_RUNS)} continue jobs "
                f"(2 cooldown + 2 const); no Modal submit."
            )
            return 0
        from modal_train import launch_training_jobs

        launch_training_jobs(CONTINUE_RUNS, max_parallel_runs=args.max_parallel)
        return 0

    # measure
    pairs = [cool_pair(), const_pair()]
    print(f"measure pairs: cool={pairs[0]}  const={pairs[1]}")
    print(f"B_R={B_R}  (Batch 1 MEASURED; not re-estimated here)")
    if not execute:
        print("[dry-run] skip Modal measure; no B_cool/B_const invented.")
        return 0
    from experiments.a3_optimization.connectivity import measure_pairs_on_modal

    rows = measure_pairs_on_modal(pairs)
    b_c, g_c, l_c1, l_c2 = _barrier_and_gap(rows[0])
    b_k, g_k, l_k1, l_k2 = _barrier_and_gap(rows[1])
    d_b = B_R - b_c
    d_l = None if L_R_MEAN is None else L_R_MEAN - 0.5 * (l_c1 + l_c2)
    emergent = "same-valley" if (b_c <= 0.01 and b_k > 0.01) else (
        "true-modes" if (b_c > 0.05 and b_k > 0.05) else "other"
    )
    summary = {
        "ledger_ids": list(LEDGER_IDS),
        "B_R": B_R,
        "B_cool": b_c,
        "B_const": b_k,
        "G_cool": g_c,
        "G_const": g_k,
        "dB": d_b,
        "dL_endpoints": d_l,
        "emergent": emergent,
        "cool_pair": pairs[0],
        "const_pair": pairs[1],
        "rows": rows,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"B_R={B_R} B_cool={b_c} B_const={b_k} G_cool={g_c} "
        f"dB={d_b} dL_endpoints={d_l} emergent={emergent}"
    )
    print(f"Wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
