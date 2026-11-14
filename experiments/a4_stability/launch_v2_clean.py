"""Clean retrain for contaminated cells: block2@0.003 and gru-retain.

Uses unique run_names (*-v2-*) so W&B/ckpt paths cannot collide with prior runs.
"""
from dataclasses import replace

from experiments.a4_stability.sweeps import config


JOBS = [
    replace(config("d8-block2", 0.003), run_name="a4-d8-block2-v2-lr0.003", force_run=True),
    replace(config("gru-retain", 0.001), run_name="a4-gru-retain-v2-lr0.001", force_run=True),
    replace(config("gru-retain", 0.003), run_name="a4-gru-retain-v2-lr0.003", force_run=True),
]


if __name__ == "__main__":
    import sys
    from modal_train import launch_training_jobs

    which = sys.argv[1] if len(sys.argv) > 1 else "retain"
    gpus = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    if which == "block2":
        jobs = [JOBS[0]]
    elif which == "retain":
        jobs = JOBS[1:]
    elif which == "all":
        jobs = JOBS
    else:
        raise SystemExit("usage: launch_v2_clean.py [block2|retain|all] [gpus]")
    for j in jobs:
        print(j.run_name, j.learning_rate)
    if "--dry-run" in sys.argv:
        raise SystemExit(0)
    launch_training_jobs(jobs, max_parallel_runs=gpus)
