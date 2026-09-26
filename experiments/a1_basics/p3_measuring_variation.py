from modal_train import launch_training_jobs
from train import TrainConfig


RUNS = [
    TrainConfig(
        deterministic=True,
        run_name_suffix="deterministic-reference-1",
    ),
    TrainConfig(
        deterministic=True,
        run_name_suffix="deterministic-reference-2",
    ),
]


# TODO: Vary model_seed and data_seed separately and together.
# For hardware nondeterminism, run the same config on at least two GPU
# types by passing `gpu=...` to launch_training_jobs from a local experiment.


def main():
    launch_training_jobs(RUNS)


if __name__ == "__main__":
    main()
