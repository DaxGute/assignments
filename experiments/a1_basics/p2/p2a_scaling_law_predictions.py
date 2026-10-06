import argparse

from modal_train import launch_training_jobs
from model_config import depth_model_config
from train import TrainConfig

EXPERIMENT_KEY = "a1-p2a-scaling-v1"
RECIPES = (
    ("baseline", {}),
    ("constant-lr", {"lr_schedule": "constant"}),
    ("dropout-0.2", {"dropout": 0.2}),
    ("lr-0.03", {"learning_rate": 0.03}),
)


def runs(depths):
    return [
        TrainConfig(
            model_config=depth_model_config(depth),
            run_name_suffix=EXPERIMENT_KEY,
            wandb_tags=(EXPERIMENT_KEY, recipe),
            save_model=False,
            **changes,
        )
        for depth in depths
        for recipe, changes in RECIPES
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("fit", "test", "all"),
        default="fit",
        nargs="?",
        help="fit=d4-d7, test=d8-d9 after preregistration",
    )
    stage = parser.parse_args().stage
    depths = {"fit": range(4, 8), "test": range(8, 10), "all": range(4, 10)}[stage]
    launch_training_jobs(runs(depths), max_parallel_runs=2)
