"""Problem 2: schedules, parameter averaging, rewarming, and horizon-free schedules.

    uv run python -m experiments.a3_optimization.p2_river_valley
"""

from modal_train import launch_training_jobs
from train import TrainConfig


EXPERIMENT_KEY = "a3-river-valley"
# Each run logs val_ema{decay}_loss next to val_loss at every evaluation.
EMA_DECAYS = (0.99, 0.999)

# Parts (a) and (b): one run per schedule, with evaluation-only EMAs.
SCHEDULE_RUNS = [
    TrainConfig(lr_schedule=lr_schedule, ema_decays=EMA_DECAYS, wandb_tags=(EXPERIMENT_KEY,))
    for lr_schedule in (
        "wsd0.0",  # constant after warmup
        "linear",
        "cos",
        "wsd0.2",  # plateau, then decay over the last 20% of training
    )
]

# Robustness sweep for part (a): bracket the default peak LR for every schedule.
SCHEDULE_TUNING_RUNS = [
    TrainConfig(
        lr_schedule=lr_schedule,
        learning_rate=learning_rate,
        ema_decays=EMA_DECAYS,
        wandb_tags=(EXPERIMENT_KEY, "p2-a-lr-tuning"),
    )
    for lr_schedule in ("wsd0.0", "linear", "cos", "wsd0.2")
    for learning_rate in (0.0015, 0.006)
]

# Part (c): the LR falls to 0.1x between 40% and 45% of training and is back at
# full LR by 50%, before the same 20% cooldown as the wsd0.2 control above.
REWARM_RUNS = [
    TrainConfig(lr_schedule="wsd0.2", lr_dip=(0.4, 0.45, 0.5, 0.1), wandb_tags=(EXPERIMENT_KEY,)),
    TrainConfig(lr_schedule="wsd0.2", lr_dip=(0.4, 0.45, 0.5, 0.0),
                wandb_tags=(EXPERIMENT_KEY, "p2-c-dip-sweep")),
    TrainConfig(lr_schedule="wsd0.2", lr_dip=(0.4, 0.45, 0.5, 0.5),
                wandb_tags=(EXPERIMENT_KEY, "p2-c-dip-sweep")),
    TrainConfig(lr_schedule="wsd0.2", lr_dip=(0.3, 0.45, 0.6, 0.1),
                wandb_tags=(EXPERIMENT_KEY, "p2-c-dip-sweep")),
]

# Part (d): horizon-free schedules. warmup_steps fixes the warmup in updates,
# so it does not depend on the budget; sgdrT restarts every T updates.
HORIZON_FREE_RUNS = [
    TrainConfig(lr_schedule="invsqrt", warmup_steps=94, wandb_tags=(EXPERIMENT_KEY,)),
    TrainConfig(lr_schedule="sgdr2000", warmup_steps=94, min_lr_ratio=0.1,
                wandb_tags=(EXPERIMENT_KEY,)),
]

HORIZON_BUDGET_RUNS = [
    TrainConfig(
        num_train_sequences=num_sequences,
        lr_schedule=schedule,
        warmup_steps=94 if schedule in ("invsqrt", "sgdr2000") else None,
        min_lr_ratio=0.1 if schedule == "sgdr2000" else TrainConfig.min_lr_ratio,
        wandb_tags=(EXPERIMENT_KEY, "p2-d-budget-sweep"),
    )
    for num_sequences in (150_000, 300_000)
    for schedule in ("linear", "cos", "invsqrt", "sgdr2000")
]

RUNS = (
    SCHEDULE_RUNS
    + SCHEDULE_TUNING_RUNS
    + REWARM_RUNS
    + HORIZON_FREE_RUNS
    + HORIZON_BUDGET_RUNS
)


def main():
    launch_training_jobs(RUNS)


if __name__ == "__main__":
    main()
