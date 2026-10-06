from modal_train import launch_training_jobs
from train import TrainConfig

EXPERIMENT_KEY = "a1-p4a-perturbation-v1"
COMMON = {
    "deterministic": True,
    "run_name_suffix": EXPERIMENT_KEY,
    "save_model": False,
}

RUNS = [
    TrainConfig(**COMMON, wandb_tags=(EXPERIMENT_KEY, "baseline")),
    TrainConfig(
        **COMMON,
        perturb_one_token=True,
        perturb_num_tokens=1,
        wandb_tags=(EXPERIMENT_KEY, "1-token"),
    ),
    TrainConfig(
        **COMMON,
        perturb_one_token=True,
        perturb_num_tokens=1024,
        wandb_tags=(EXPERIMENT_KEY, "1024-tokens"),
    ),
]


if __name__ == "__main__":
    launch_training_jobs(RUNS, max_parallel_runs=2)
