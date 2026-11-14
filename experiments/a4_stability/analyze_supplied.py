"""Plots and tables from the supplied P1/P2 manifest and local probes.

Does not train. Numbers come from manifest.json and results/a4_stability/probes/.
"""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DELIV = Path(__file__).resolve().parent / "deliverables"
FIG = DELIV / "figures"
MANIFEST = DELIV / "manifest.json"
PROBES = ROOT / "results" / "a4_stability" / "probes"


def load_manifest():
    data = json.loads(MANIFEST.read_text())
    rows = []
    for run in data["runs"]:
        rows.append({
            "run_name": run["run_name"],
            "variant": run["variant"],
            "problem": run["problem"],
            "lr": run["lr"],
            "status": run["status"],
            "validation_loss": run.get("validation_loss"),
            "url": run.get("url"),
            "checkpoint": run.get("checkpoint") or "",
        })
    return rows


def write_table(rows):
    path = DELIV / "supplied_outcomes.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["problem"], r["variant"], r["lr"])))
    return path


def plot_loss_vs_lr(rows):
    FIG.mkdir(parents=True, exist_ok=True)
    families = {1: "Transformer variants", 2: "RNN variants"}
    for problem, title in families.items():
        fig, ax = plt.subplots(figsize=(8.2, 5.2))
        variants = sorted({r["variant"] for r in rows if r["problem"] == problem})
        for variant in variants:
            cells = sorted((r for r in rows if r["variant"] == variant), key=lambda r: r["lr"])
            xs, ys = [], []
            for cell in cells:
                if cell["validation_loss"] is None:
                    continue
                if cell["validation_loss"] > 6:
                    continue
                xs.append(cell["lr"])
                ys.append(cell["validation_loss"])
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", label=variant)
        ax.set_xscale("log")
        ax.set_xlabel("learning rate")
        ax.set_ylabel("final validation loss")
        ax.set_title(title + " (dead runs at ~6.875 omitted)")
        ax.legend(fontsize=7, ncol=2, frameon=False)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / f"p{problem}_loss_vs_lr.pdf")
        fig.savefig(FIG / f"p{problem}_loss_vs_lr.png", dpi=140)
        plt.close(fig)


def plot_temporal():
    wanted = {
        "a4-rnn-lr0.002": "modern",
        "a4-rnn1-lr0.003": "plain 1-layer",
        "a4-rnn4-lr0.001": "plain 4-layer",
        "a4-rnn-no-residual-lr0.001": "no depth residual",
        "a4-rnn-no-prenorm-lr0.002": "no pre-norm",
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    for ax, layer, title in zip(axes, (0, -1), ("bottom recurrent layer", "top recurrent layer")):
        for name, label in wanted.items():
            blob = json.loads((PROBES / f"{name}.json").read_text())
            grads = blob["last"]["hidden_gradient_rms"]
            series = grads[layer]
            lags = list(range(1, len(series)))
            values = series[-2::-1]
            ax.plot(lags, values, label=label)
        ax.set_yscale("log")
        ax.set_xlim(1, 256)
        ax.set_xlabel("lag before the final token")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("last-token hidden-gradient RMS")
    axes[1].legend(fontsize=7, frameon=False)
    fig.suptitle("One validation sequence. Last-token gradient versus time lag.")
    fig.tight_layout()
    fig.savefig(FIG / "p2_last_token_grad_vs_lag.pdf")
    fig.savefig(FIG / "p2_last_token_grad_vs_lag.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    rows = load_manifest()
    print(write_table(rows))
    plot_loss_vs_lr(rows)
    plot_temporal()
    print(FIG)
