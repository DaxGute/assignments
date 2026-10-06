from modal_train import launch_training_jobs
from train import TrainConfig

EXPERIMENT_KEY = "a1-p1-1d-v1"
RUN_METADATA = {
    "run_name_suffix": EXPERIMENT_KEY,
    "wandb_tags": (EXPERIMENT_KEY,),
    "save_model": False,
}

RUNS = [
    TrainConfig(**RUN_METADATA),
    TrainConfig(**RUN_METADATA, warmup_percent=0.001),
    TrainConfig(**RUN_METADATA, warmup_percent=0.003),
    TrainConfig(**RUN_METADATA, warmup_percent=0.03),
    TrainConfig(**RUN_METADATA, warmup_percent=0.1),
    TrainConfig(**RUN_METADATA, learning_rate=3e-4),
    TrainConfig(**RUN_METADATA, learning_rate=1e-3),
    TrainConfig(**RUN_METADATA, learning_rate=1e-2),
    TrainConfig(**RUN_METADATA, learning_rate=3e-2),
    TrainConfig(**RUN_METADATA, batch_size=16),
    TrainConfig(**RUN_METADATA, batch_size=32),
    TrainConfig(**RUN_METADATA, batch_size=128),
    TrainConfig(**RUN_METADATA, batch_size=256),
    TrainConfig(**RUN_METADATA, weight_decay=0.01),
    TrainConfig(**RUN_METADATA, weight_decay=0.03),
    TrainConfig(**RUN_METADATA, weight_decay=0.3),
    TrainConfig(**RUN_METADATA, weight_decay=1.0),
]


def main():
    launch_training_jobs(RUNS, max_parallel_runs=2)


if __name__ == "__main__":
    main()
