"""Problem 3: sharpness and the edge of stability.

    uv run python -m experiments.a3_optimization.p3_edge_of_stability

Both helpers attach sharpness_logging.SharpnessLogger, which logs to W&B under
logging/sharpness/*. Read the results back with experiments.a3_optimization.sharpness.
"""

from metric_logging import AFTER_BACKWARD, MetricLogger
from modal_train import launch_training_jobs
from sharpness_logging import SharpnessLogger
from train import TrainConfig


EXPERIMENT_KEY = "a3-edge-of-stability"
FRACTION = 0.32  # of an epoch: 3,000 updates, 197M tokens at batch size 64


def full_batch(num_sequences, updates, every=25, **changes):
    """Full-batch gradient descent on the first `num_sequences` training sequences.

    Every update uses all of them, so one epoch is one update. lambda_max is
    measured on the same sequences every `every` updates. Keyword arguments
    override any setting, e.g. learning_rate=0.02 or optimizer_name="adamw".
    """
    optimizer_name = changes.get("optimizer_name", "sgd")
    settings = dict(
        num_train_sequences=num_sequences,
        batch_size=num_sequences,
        num_epochs=float(updates),
        optimizer_name=optimizer_name,
        # plain gradient descent for sgd (its default beta1 is momentum); AdamW keeps 0.9
        beta1=0.0 if optimizer_name == "sgd" else TrainConfig.beta1,
        lr_schedule="constant",
        weight_decay=0.0,
        grad_norm=None,
        embedding_init_scale=16.0,
        num_evals=10,
        metric_loggers=TrainConfig.metric_loggers
        + (MetricLogger(AFTER_BACKWARD, SharpnessLogger(every=every)),),
        wandb_tags=(EXPERIMENT_KEY,),
    )
    settings.update(changes)
    return TrainConfig(**settings)


def measurement_steps(batch_size):
    """Every 16 updates to update 208, then every 100 to 1,500, then every 300, at
    batch size 64; other batch sizes measure at the same token counts."""
    at_64 = [*range(0, 209, 16), *range(300, 1501, 100), *range(1800, 3001, 300)]
    total = int(FRACTION * (600_000 // batch_size))
    return sorted({round(u * 64 / batch_size) for u in at_64} & set(range(total)))


def minibatch(batch_size=64, **changes):
    """The course recipe with a constant learning rate after warmup, cut at 0.32
    epochs, measuring the preconditioned sharpness on 64 held-out sequences.

    Warmup is 3.125% of the run: 93 updates at batch size 64, and the same
    6M tokens at every batch size.
    """
    settings = dict(
        num_epochs=FRACTION,
        warmup_percent=0.03125,
        lr_schedule="wsd0.0",
        batch_size=batch_size,
        num_micro_batches=max(1, batch_size // 64),
        metric_loggers=TrainConfig.metric_loggers
        + (MetricLogger(AFTER_BACKWARD, SharpnessLogger(
            steps=measurement_steps(batch_size), probe="heldout", kinds=("pre",))),),
        wandb_tags=(EXPERIMENT_KEY,),
    )
    settings.update(changes)
    return TrainConfig(**settings)


# Part (a): full-batch training.
FULL_BATCH_RUNS = [
    full_batch(32, 1500, learning_rate=0.01),  # (a) i: the clean case
    # (a) ii: intervene only after the clean run has reached the edge. These
    # preserve the original 1,500-update budget and measurement cadence.
    full_batch(32, 1500, learning_rate=0.01, lr_schedule="step750x0.5"),
    full_batch(32, 1500, learning_rate=0.01, lr_schedule="step750x2"),
]

# Part (b): minibatch training.
MINIBATCH_RUNS = [
    minibatch(),  # (b): the base recipe
]

# Part (c): 18 exploration runs (22 total with parts a/b), matching the handout.
# The grid varies only learning rate, batch size, sample count, and beta1.
# Quantitative predictions are locked under P3-c-* in the prediction ledger.
FULL_BATCH_EXPLORATION_RUNS = [
    full_batch(32, 1500, learning_rate=0.005),
    full_batch(32, 1500, learning_rate=0.02),
    full_batch(16, 1500, learning_rate=0.01),
    full_batch(64, 1500, learning_rate=0.01),
    # Interpolate the observed N=32/64 sustained-edge regime and resolve the
    # preregistered N=48 quiz prediction.
    full_batch(48, 1500, learning_rate=0.01),
]

MINIBATCH_GRID_RUNS = [
    minibatch(batch_size=batch_size, learning_rate=learning_rate)
    for batch_size in (16, 64, 256)
    for learning_rate in (0.0015, 0.003, 0.006)
    if (batch_size, learning_rate) != (64, 0.003)
]

MOMENTUM_RUNS = [
    minibatch(beta1=beta1)
    for beta1 in (0.0, 0.5)
]

INTERPOLATION_RUNS = [
    # Locate the strong batch-size transition between the original powers of four.
    minibatch(batch_size=32, learning_rate=0.003),
    minibatch(batch_size=128, learning_rate=0.003),
    # Test whether the weak empirical threshold exponent survives near beta1=1.
    minibatch(beta1=0.99),
]

EXPLORATION_RUNS = (
    FULL_BATCH_EXPLORATION_RUNS
    + MINIBATCH_GRID_RUNS
    + MOMENTUM_RUNS
    + INTERPOLATION_RUNS
)
RUNS = FULL_BATCH_RUNS + MINIBATCH_RUNS + EXPLORATION_RUNS


def main():
    launch_training_jobs(RUNS)


if __name__ == "__main__":
    main()
