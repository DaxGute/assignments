"""Queue the remaining P3/P4 runs. Default is the transformer wave only.

Both course GPUs may be busy. Pass --gpus 1 to take a single free device.
Does not launch unless called as __main__.
"""
from experiments.a4_stability.sweeps import VARIANTS

# Two rates per new variant. The running pilots are omitted:
# a4-d8-half-lr0.003 and a4-gru-lr0.003.
TRANSFORMER_WAVE = {
    "d8-half": (0.0015, 0.006),
    "d8-ema01": (0.0015, 0.003),
    "d8-ema09": (0.0015, 0.003),
    "d8-uniform": (0.0015, 0.003),
    "d8-attention": (0.0015, 0.003),
    "d8-block2": (0.003, 0.009),
    "d8-block4": (0.0015, 0.003),
    "d8-identity_mix": (0.0015, 0.003),
}

# Held until a GPU is free for multi-hour RNN runs.
RNN_WAVE = {
    "rnn-half": (0.001, 0.002),
    "rnn-ema01": (0.001, 0.002),
    "rnn-ema09": (0.001, 0.002),
    "rnn-uniform": (0.001, 0.002),
    "rnn-attention": (0.001, 0.002),
    "rnn-block2": (0.001, 0.002),
    "rnn-block4": (0.001, 0.002),
    "rnn-identity_mix": (0.001, 0.002),
    "gru": (0.001, 0.006),
    "gru-no-reset": (0.001, 0.003),
    "gru-no-update": (0.001, 0.003),
    "gru-retain": (0.001, 0.003),
    "gru-time-residual": (0.001, 0.003),
}


def build(wave):
    missing = [name for name in wave if name not in VARIANTS]
    if missing:
        raise SystemExit(f"unknown variants: {missing}")
    from experiments.a4_stability.sweeps import config
    return [config(name, lr) for name, rates in wave.items() for lr in rates]


if __name__ == "__main__":
    import sys
    from modal_train import launch_training_jobs
    wave_name = sys.argv[1] if len(sys.argv) > 1 else "transformer"
    gpus = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    wave = TRANSFORMER_WAVE if wave_name == "transformer" else RNN_WAVE
    jobs = build(wave)
    print(f"{wave_name}: {len(jobs)} runs, max_parallel={gpus}")
    for cfg in jobs:
        print(cfg.run_name, cfg.learning_rate)
    if "--dry-run" in sys.argv:
        raise SystemExit(0)
    # One app. max_parallel must stay within the course limit of 2 GPUs,
    # counting containers already running in other apps.
    launch_training_jobs(jobs, max_parallel_runs=gpus)
