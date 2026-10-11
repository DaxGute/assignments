"""Thin entrypoint for P5 cooldown-barrier exploratory runs.

    uv run python -m experiments.a3_optimization.launch_p5_cooldown continues --dry-run
    uv run python -m experiments.a3_optimization.launch_p5_cooldown measure --dry-run
"""

from experiments.a3_optimization.p5_cooldown_barrier import main


if __name__ == "__main__":
    raise SystemExit(main())
