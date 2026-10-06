from modal_train import launch_training_jobs
from train import TrainConfig

EXPERIMENT_KEY = "a1-p3a-full-variation-v1"


def run(seed, gpu):
    return TrainConfig(
        model_seed=seed,
        data_seed=seed,
        run_name_suffix=EXPERIMENT_KEY,
        wandb_tags=(EXPERIMENT_KEY, gpu.lower()),
        save_model=False,
    )


if __name__ == "__main__":
    launch_training_jobs([run(101, "H100"), run(102, "H100")], gpu="H100")
    launch_training_jobs([run(103, "A100")], gpu="A100")
