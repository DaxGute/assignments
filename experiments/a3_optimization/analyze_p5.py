"""Generate reproducible diagnostics for the measured P5 cooldown experiment.

    uv run python -m experiments.a3_optimization.analyze_p5
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS = Path(__file__).resolve().parent / "results"
SOURCE = RESULTS / "a3-p5-cooldown-barrier.json"
PLOTS = RESULTS / "a3-p5-analysis-plots"


def main() -> int:
    result = json.loads(SOURCE.read_text())
    cooldown, constant = result["rows"]
    PLOTS.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))

    axes[0].plot(
        cooldown["alphas"], cooldown["interpolation"], marker="o",
        label="cooldown",
    )
    axes[0].plot(
        constant["alphas"], constant["interpolation"], marker="o",
        label="constant continue",
    )
    axes[0].axhline(cooldown["ensemble"], linestyle=":", label="cool ensemble")
    axes[0].set(
        xlabel=r"Interpolation $\alpha$",
        ylabel="Validation loss",
        title="Parameter path remains disconnected",
    )
    axes[0].legend(fontsize=8)

    labels = ("raw P1", "cooldown", "constant")
    barriers = (result["B_R"], result["B_cool"], result["B_const"])
    axes[1].bar(labels, barriers)
    axes[1].axhline(0.01, linestyle=":", color="black", label="connected threshold")
    axes[1].set(ylabel="Midpoint barrier B", title="Cooldown does not close barrier")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(fontsize=8)

    endpoint_labels = ("raw", "cooldown")
    endpoint_losses = (
        result["L_R_endpoint_mean"],
        result["L_C_endpoint_mean"],
    )
    axes[2].bar(endpoint_labels, endpoint_losses)
    axes[2].set(
        ylabel="Mean endpoint validation loss",
        title=f"Cooldown improves loss by {result['dL_endpoints']:.3f}",
    )

    fig.tight_layout()
    output = PLOTS / "p5_cooldown_barrier.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
