from modal_train import launch_training_jobs
from model_config import depth_model_config
from train import TrainConfig

EXPERIMENT_KEY = "a1-p2c-breaking-v1"
COMMON = {
    "run_name_suffix": EXPERIMENT_KEY,
    "save_model": False,
}
INTERVENTIONS = (
    ("high-lr-no-clip", {"learning_rate": 0.1, "grad_norm": None}),
    ("sgd", {"optimizer_name": "sgd", "learning_rate": 0.03}),
)
DATA_INTERVENTIONS = (
    ("high-lr-no-clip", {"learning_rate": 0.1, "grad_norm": None}),
    ("dropout-constant", {"dropout": 0.5, "lr_schedule": "constant"}),
)
DATA_SCALES = (37_500, 75_000, 150_000, 300_000, 600_000)

RUNS = [
    *[
        TrainConfig(
            **COMMON,
            **changes,
            model_config=depth_model_config(depth),
            wandb_tags=(EXPERIMENT_KEY, "model-scaling", intervention),
        )
        for intervention, changes in INTERVENTIONS
        for depth in range(4, 9)
    ],
    *[
        TrainConfig(
            **COMMON,
            **changes,
            num_train_sequences=num_sequences,
            wandb_tags=(EXPERIMENT_KEY, "data-scaling", intervention),
        )
        for intervention, changes in DATA_INTERVENTIONS
        for num_sequences in DATA_SCALES
    ],
]


if __name__ == "__main__":
    launch_training_jobs(RUNS, max_parallel_runs=2)
