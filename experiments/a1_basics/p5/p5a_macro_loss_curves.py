from modal_train import launch_training_jobs
from train import TrainConfig

EXPERIMENT_KEY = "a1-p5a-macro-v1"
COMMON = {
    "run_name_suffix": EXPERIMENT_KEY,
    "wandb_tags": (EXPERIMENT_KEY,),
    "save_model": False,
}

RUNS = [
    TrainConfig(**COMMON),
    TrainConfig(**COMMON, learning_rate=0.01),
    TrainConfig(**COMMON, beta1=0.8),
    TrainConfig(**COMMON, batch_size=128),
    TrainConfig(**COMMON, lr_schedule="constant"),
]


if __name__ == "__main__":
    launch_training_jobs(RUNS, max_parallel_runs=2)
