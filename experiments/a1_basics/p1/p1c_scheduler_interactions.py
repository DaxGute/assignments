from modal_train import launch_training_jobs
from train import TrainConfig

EXPERIMENT_KEY = "a1-p1-schedulers-v2"
RUN_METADATA = {
    "run_name_suffix": EXPERIMENT_KEY,
    "wandb_tags": (EXPERIMENT_KEY,),
    "save_model": False,
}

RUNS = [
    # Scheduler family at the default learning rate (5 runs)
    TrainConfig(**RUN_METADATA, lr_schedule="linear"),
    TrainConfig(**RUN_METADATA, lr_schedule="cos"),
    TrainConfig(**RUN_METADATA, lr_schedule="constant"),
    TrainConfig(**RUN_METADATA, lr_schedule="wsd0.1"),
    TrainConfig(**RUN_METADATA, lr_schedule="wsd0.2"),
    # Scheduler x learning rate (5 additional runs)
    TrainConfig(**RUN_METADATA, lr_schedule="linear", learning_rate=1e-2),
    TrainConfig(**RUN_METADATA, lr_schedule="cos", learning_rate=1e-2),
    TrainConfig(**RUN_METADATA, lr_schedule="constant", learning_rate=1e-2),
    TrainConfig(**RUN_METADATA, lr_schedule="wsd0.1", learning_rate=1e-2),
    TrainConfig(**RUN_METADATA, lr_schedule="wsd0.2", learning_rate=1e-2),
    # Cosine scheduler with the Part A/B hyperparameters (3 runs)
    TrainConfig(**RUN_METADATA, lr_schedule="cos", batch_size=128),
    TrainConfig(**RUN_METADATA, lr_schedule="cos", weight_decay=0.3),
    TrainConfig(**RUN_METADATA, lr_schedule="cos", warmup_percent=0.03),
    # Optimizer family and SGD learning rate (2 runs)
    TrainConfig(
        **RUN_METADATA, optimizer_name="sgd", lr_schedule="cos", learning_rate=3e-3
    ),
    TrainConfig(
        **RUN_METADATA, optimizer_name="sgd", lr_schedule="cos", learning_rate=3e-2
    ),
    # Adam momentum coefficients (4 runs)
    TrainConfig(**RUN_METADATA, lr_schedule="cos", beta1=0.8),
    TrainConfig(**RUN_METADATA, lr_schedule="cos", beta1=0.95),
    TrainConfig(**RUN_METADATA, lr_schedule="cos", beta2=0.9),
    TrainConfig(**RUN_METADATA, lr_schedule="cos", beta2=0.99),
    # Gradient clipping (1 run; default is grad_norm=1.0)
    TrainConfig(**RUN_METADATA, lr_schedule="cos", grad_norm=None),
]


def main():
    launch_training_jobs(RUNS, max_parallel_runs=2)


if __name__ == "__main__":
    main()
