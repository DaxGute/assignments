from modal_train import launch_training_jobs
from train import TrainConfig

EXPERIMENT_KEY = "a1-p4b-amplification-v1"
MAGNITUDES = (1, 16, 256)
PERTURB_STEPS = (0, 2500, 5000, 7500)

RUNS = [
    TrainConfig(
        deterministic=True,
        perturb_one_token=True,
        perturb_num_tokens=magnitude,
        perturb_step=step,
        run_name_suffix=EXPERIMENT_KEY,
        wandb_tags=(
            EXPERIMENT_KEY,
            f"magnitude-{magnitude}",
            f"perturb-step-{step}",
        ),
        save_model=False,
    )
    for magnitude in MAGNITUDES
    for step in PERTURB_STEPS
]


if __name__ == "__main__":
    launch_training_jobs(RUNS, max_parallel_runs=2)
