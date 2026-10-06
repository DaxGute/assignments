from modal_train import launch_training_jobs
from train import TrainConfig

EXPERIMENT_KEY = "a1-p1-covarying-v1"

LR_BATCH = {
    "run_name_suffix": EXPERIMENT_KEY,
    "wandb_tags": (EXPERIMENT_KEY, "learning-rate-batch-size"),
    "save_model": False,
}
LR_WARMUP = {
    "run_name_suffix": EXPERIMENT_KEY,
    "wandb_tags": (EXPERIMENT_KEY, "learning-rate-warmup"),
    "save_model": False,
}
LR_WEIGHT_DECAY = {
    "run_name_suffix": EXPERIMENT_KEY,
    "wandb_tags": (EXPERIMENT_KEY, "learning-rate-weight-decay"),
    "save_model": False,
}

RUNS = [
    # Learning rate x batch size (8 runs)
    TrainConfig(**LR_BATCH, learning_rate=1e-3, batch_size=32),
    TrainConfig(**LR_BATCH, learning_rate=3e-3, batch_size=32),
    TrainConfig(**LR_BATCH, learning_rate=1e-2, batch_size=32),
    TrainConfig(**LR_BATCH, learning_rate=3e-2, batch_size=32),
    TrainConfig(**LR_BATCH, learning_rate=1e-3, batch_size=128),
    TrainConfig(**LR_BATCH, learning_rate=3e-3, batch_size=128),
    TrainConfig(**LR_BATCH, learning_rate=1e-2, batch_size=128),
    TrainConfig(**LR_BATCH, learning_rate=3e-2, batch_size=128),
    # Learning rate x warmup (6 runs)
    TrainConfig(**LR_WARMUP, learning_rate=1e-3, warmup_percent=0.003),
    TrainConfig(**LR_WARMUP, learning_rate=3e-3, warmup_percent=0.003),
    TrainConfig(**LR_WARMUP, learning_rate=1e-2, warmup_percent=0.003),
    TrainConfig(**LR_WARMUP, learning_rate=1e-3, warmup_percent=0.03),
    TrainConfig(**LR_WARMUP, learning_rate=3e-3, warmup_percent=0.03),
    TrainConfig(**LR_WARMUP, learning_rate=1e-2, warmup_percent=0.03),
    # Learning rate x weight decay (6 runs)
    TrainConfig(**LR_WEIGHT_DECAY, learning_rate=1e-3, weight_decay=0.03),
    TrainConfig(**LR_WEIGHT_DECAY, learning_rate=3e-3, weight_decay=0.03),
    TrainConfig(**LR_WEIGHT_DECAY, learning_rate=1e-2, weight_decay=0.03),
    TrainConfig(**LR_WEIGHT_DECAY, learning_rate=1e-3, weight_decay=0.3),
    TrainConfig(**LR_WEIGHT_DECAY, learning_rate=3e-3, weight_decay=0.3),
    TrainConfig(**LR_WEIGHT_DECAY, learning_rate=1e-2, weight_decay=0.3),
]


def main():
    launch_training_jobs(RUNS, max_parallel_runs=2)


if __name__ == "__main__":
    main()
