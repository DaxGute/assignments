"""Plot amplification by perturbation magnitude and remaining training steps."""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import wandb

from utils import WANDB_ENTITY, WANDB_PROJECT

EXPERIMENT_KEY = "a1-p4b-amplification-v1"
BASELINE_KEY = "a1-p4a-perturbation-v1"
PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
OUT_PATH = Path(__file__).resolve().parent / "plots" / f"{EXPERIMENT_KEY}.png"
MAGNITUDES = (1, 16, 256)
PERTURB_STEPS = (0, 2500, 5000, 7500)
COLORS = {1: "#168aad", 16: "#7030A0", 256: "#e74c3c"}


def tagged_int(tags, prefix) -> int:
    tag = next(tag for tag in tags if tag.startswith(prefix))
    return int(tag.removeprefix(prefix))


def train_loss_history(run) -> dict[int, float]:
    return {
        int(row["optimizer_step"]): float(row["train_loss"])
        for row in run.scan_history(keys=["optimizer_step", "train_loss"])
        if row.get("optimizer_step") is not None and row.get("train_loss") is not None
    }


def main() -> None:
    api = wandb.Api()
    baseline_runs = api.runs(
        PROJECT_PATH,
        filters={"tags": {"$in": [BASELINE_KEY]}, "state": "finished"},
        order="-created_at",
    )
    baseline = next(
        (run for run in baseline_runs if "baseline" in set(run.tags or ())),
        None,
    )
    if baseline is None:
        raise RuntimeError("Run part 4a's deterministic baseline first.")
    baseline_val_loss = float(baseline.summary["val_loss"])
    baseline_history = train_loss_history(baseline)

    runs = api.runs(
        PROJECT_PATH,
        filters={"tags": {"$in": [EXPERIMENT_KEY]}, "state": "finished"},
        order="-created_at",
    )
    collected = {}
    for run in runs:
        tags = set(run.tags or ())
        magnitude = tagged_int(tags, "magnitude-")
        step = tagged_int(tags, "perturb-step-")
        collected.setdefault((magnitude, step), run)

    missing = [
        (magnitude, step)
        for magnitude in MAGNITUDES
        for step in PERTURB_STEPS
        if (magnitude, step) not in collected
    ]
    if missing:
        raise RuntimeError(f"Missing finished runs: {missing}")

    terminal_delta = np.array(
        [
            [
                abs(
                    float(collected[(magnitude, step)].summary["val_loss"])
                    - baseline_val_loss
                )
                for step in PERTURB_STEPS
            ]
            for magnitude in MAGNITUDES
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    image = axes[0].imshow(
        np.log10(np.maximum(terminal_delta, 1e-12)),
        cmap="magma",
        aspect="auto",
    )
    axes[0].set(
        title="Terminal variation",
        xlabel="Perturbation step T",
        ylabel="Perturbed tokens",
    )
    axes[0].set_xticks(range(len(PERTURB_STEPS)), PERTURB_STEPS)
    axes[0].set_yticks(range(len(MAGNITUDES)), MAGNITUDES)
    for row in range(len(MAGNITUDES)):
        for column in range(len(PERTURB_STEPS)):
            axes[0].text(
                column,
                row,
                f"{terminal_delta[row, column]:.2e}",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
            )
    fig.colorbar(image, ax=axes[0], label="log10 |Δ validation loss|")

    for (magnitude, step), run in collected.items():
        history = train_loss_history(run)
        shared_steps = sorted(set(history) & set(baseline_history))
        after = [value for value in shared_steps if value >= step]
        delta = [
            abs(history[value] - baseline_history[value]) + 1e-12 for value in after
        ]
        axes[1].plot(
            np.asarray(after) - step,
            delta,
            color=COLORS[magnitude],
            alpha=0.45,
            label=f"{magnitude} tokens" if step == PERTURB_STEPS[0] else None,
        )
    axes[1].set(
        title="Amplification after T",
        xlabel="Optimizer steps since perturbation",
        ylabel="|Δ training loss|",
        yscale="log",
    )
    axes[1].grid(True, which="both", linestyle=":", alpha=0.35)
    axes[1].legend(frameon=False)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=300)
    print(OUT_PATH.resolve())


if __name__ == "__main__":
    main()
