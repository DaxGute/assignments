from modal_train import launch_training_jobs
from model_config import depth_model_config
from train import TrainConfig

EXPERIMENT_KEY = "a1-p3c-hyperparameters-v1"
CONDITIONS = (
    ("baseline", {}),
    ("high-lr", {"learning_rate": 0.01}),
    ("dropout", {"dropout": 0.2}),
    ("constant-lr", {"lr_schedule": "constant"}),
    ("no-clipping", {"grad_norm": None}),
    ("larger-d10", {"model_config": depth_model_config(10)}),
    ("longer-2epochs", {"num_epochs": 2.0}),
)

RUNS = [
    TrainConfig(
        **changes,
        model_seed=seed,
        data_seed=seed,
        run_name_suffix=EXPERIMENT_KEY,
        wandb_tags=(EXPERIMENT_KEY, condition),
        save_model=False,
    )
    for condition, changes in CONDITIONS
    for seed in (51, 52)
]


if __name__ == "__main__":
    launch_training_jobs(RUNS, gpu="H100", max_parallel_runs=2)
