from modal_train import launch_training_jobs
from train import TrainConfig

EXPERIMENT_KEY = "a1-p3b-isolated-v1"


def run(label, source, **changes):
    return TrainConfig(
        run_name_suffix=f"{EXPERIMENT_KEY}-{label}",
        wandb_tags=(EXPERIMENT_KEY, source, label),
        save_model=False,
        **changes,
    )


H100_RUNS = [
    run("deterministic-reference-1", "reference", deterministic=True),
    run("deterministic-reference-2", "reference", deterministic=True),
    run("model-seed-43", "model-seed", deterministic=True, model_seed=43),
    run("model-seed-44", "model-seed", deterministic=True, model_seed=44),
    run("data-seed-43", "data-seed", deterministic=True, data_seed=43),
    run("data-seed-44", "data-seed", deterministic=True, data_seed=44),
    run("hardware-h100", "hardware", model_seed=42, data_seed=42),
]
A100_RUNS = [run("hardware-a100", "hardware", model_seed=42, data_seed=42)]


if __name__ == "__main__":
    launch_training_jobs(H100_RUNS, gpu="H100", max_parallel_runs=2)
    launch_training_jobs(A100_RUNS, gpu="A100")
