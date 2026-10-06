from modal_train import launch_training_jobs
from train import TrainConfig

EXPERIMENT_KEY = "a1-p6c-norm-controls-v1"
COMMON = {
    "run_name_suffix": EXPERIMENT_KEY,
    "save_model": False,
}
INTERVENTIONS = (
    ("low-lr-large-batch", {"learning_rate": 3e-4, "batch_size": 256}),
    (
        "high-lr-small-batch",
        {"learning_rate": 3e-2, "batch_size": 16, "grad_norm": None},
    ),
    ("no-qk-norm", {"qk_norm": False}),
    ("heavy-dropout", {"dropout": 0.5}),
    (
        "sgd-no-clip",
        {"optimizer_name": "sgd", "learning_rate": 3e-2, "grad_norm": None},
    ),
)

RUNS = [
    TrainConfig(
        **COMMON,
        **changes,
        wandb_tags=(EXPERIMENT_KEY, intervention),
    )
    for intervention, changes in INTERVENTIONS
]


if __name__ == "__main__":
    launch_training_jobs(RUNS, max_parallel_runs=2)
