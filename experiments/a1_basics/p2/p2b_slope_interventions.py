from modal_train import launch_training_jobs
from model_config import depth_model_config
from train import TrainConfig

EXPERIMENT_KEY = "a1-p2b-slopes-v1"
COMMON = {
    "run_name_suffix": EXPERIMENT_KEY,
    "save_model": False,
}
INTERVENTION = {"batch_size": 128, "warmup_percent": 0.03}

RUNS = [
    *[
        TrainConfig(
            **COMMON,
            **INTERVENTION,
            model_config=depth_model_config(depth),
            wandb_tags=(EXPERIMENT_KEY, "model-scaling"),
        )
        for depth in range(4, 9)
    ],
    *[
        TrainConfig(
            **COMMON,
            **INTERVENTION,
            num_train_sequences=num_sequences,
            wandb_tags=(EXPERIMENT_KEY, "data-scaling"),
        )
        for num_sequences in (37_500, 75_000, 150_000, 300_000, 600_000)
    ],
]


if __name__ == "__main__":
    launch_training_jobs(RUNS, max_parallel_runs=2)
