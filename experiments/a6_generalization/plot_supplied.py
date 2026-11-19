"""Figures for the supplied reference runs. No training."""

from pathlib import Path
import json
import numpy as np

from experiments.a6_generalization.analysis import power_law, predict
from experiments.a6_generalization.common import runs

OUT = Path('experiments/a6_generalization/deliverables/figures')


def _style():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            'font.family': ['Palatino', 'DejaVu Serif'],
            'font.size': 11,
            'axes.spines.top': False,
            'axes.spines.right': False,
        }
    )
    return plt


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches='tight')
    return path


def parameters(plt):
    rows = runs('parameter_scaling')['runs']
    rows = sorted(rows, key=lambda row: row['parameters'])
    n = np.array([row['parameters'] / 1e6 for row in rows])
    y = np.array([row['validation_nll'] for row in rows])
    flops = np.array([row['training_flops'] for row in rows])
    fit = power_law(n, y)
    xx = np.geomspace(n.min(), n.max(), 200)
    colors = ['#7030A0' if row['epochs'] == 32 else '#5178D5' for row in rows]
    loss_offsets = {4: (7, -12), 5: (6, 2), 6: (6, -10), 7: (-8, 8), 8: (6, 4), 9: (6, 5), 10: (-8, 8)}
    flop_offsets = {4: (4, 4), 5: (4, 6), 6: (4, 4), 7: (4, -10), 8: (4, 4), 9: (-18, 6), 10: (4, -11)}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    axes[0].scatter(n, y, c=colors, s=36, zorder=3)
    axes[0].plot(xx, predict(fit, xx), '--', color='#7030A0', lw=1.2)
    axes[1].scatter(n, flops, c=colors, s=36, zorder=3)
    for row, x, loss, cost in zip(rows, n, y, flops):
        label = f"d{row['depth']}"
        ox, oy = loss_offsets[row['depth']]
        axes[0].annotate(
            label, (x, loss), textcoords='offset points', xytext=(ox, oy),
            ha='left' if ox >= 0 else 'right', va='center', fontsize=8,
        )
        fx, fy = flop_offsets[row['depth']]
        axes[1].annotate(
            label, (x, cost), textcoords='offset points', xytext=(fx, fy),
            ha='left' if fx >= 0 else 'right', va='center', fontsize=8,
        )
    axes[1].scatter([], [], c='#7030A0', s=36, label='32 epochs')
    axes[1].scatter([], [], c='#5178D5', s=36, label='16 epochs')
    axes[1].legend(frameon=False, fontsize=8)
    axes[0].set(xscale='log', xlabel='Parameters (millions)', ylabel='Validation NLL')
    axes[1].set(xscale='log', yscale='log', xlabel='Parameters (millions)', ylabel='Training FLOPs')
    for ax in axes:
        ax.grid(True, ls=':', alpha=0.3)
    return save(fig, 'p1_parameters.png'), fit


def ensemble(plt):
    from experiments.a6_generalization.analysis import chunk_bootstrap
    from experiments.a6_generalization.p2_regularization import ensemble_curve

    curve = ensemble_curve()
    mean = np.array(curve['ensemble_validation_nll'], dtype=float)
    samples = chunk_bootstrap(np.array(curve['chunk_losses'], dtype=float), repeats=4000, seed=42)
    low = np.quantile(samples, 0.05, axis=1)
    high = np.quantile(samples, 0.95, axis=1)
    k = np.arange(1, len(mean) + 1)
    fit = power_law(k.astype(float), mean)
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    ax.errorbar(
        k,
        mean,
        yerr=[mean - low, high - mean],
        fmt='o',
        color='#7030A0',
        capsize=3,
        label='Subset mean, 90% marginal interval',
    )
    xx = np.linspace(1, 5, 80)
    ax.plot(xx, predict(fit, xx), '--', color='#7030A0', label='Fit on K=1..5')
    ax.set(xlabel='Ensemble members K', ylabel='Validation NLL', xticks=k)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, ls=':', alpha=0.3)
    return save(fig, 'p2_ensemble.png')


def batch(plt):
    rows = sorted(runs('batch_scaling')['runs'], key=lambda row: row['batch'])
    b = np.array([row['batch'] for row in rows])
    y = np.array([row['validation_nll'] for row in rows])
    sharp = []
    for row in rows:
        directional = ((row.get('sharpness') or {}).get('directional') or {}).get('0.05') or {}
        value = directional.get('max_loss_increase')
        if value is None and row.get('batch') == 64:
            measured = json.loads(
                Path('results/a6_generalization/b64_sharpness.json').read_text()
            )
            value = measured['0.05']['max_loss_increase']
        sharp.append(value)
    fig, ax = plt.subplots(figsize=(5.2, 2.85))
    ax.plot(b, y, 'o-', color='#7030A0')
    ax.set(xscale='log', xlabel='Batch size (sequences)', ylabel='Validation NLL')
    ax.grid(True, ls=':', alpha=0.3)
    path = save(fig, 'p3_batch.png')
    fig, ax = plt.subplots(figsize=(5.2, 2.85))
    xb, yb = [], []
    for batch_size, loss, value in zip(b, y, sharp):
        if value is not None:
            xb.append(value)
            yb.append(loss)
            ax.annotate(str(batch_size), (value, loss), textcoords='offset points', xytext=(4, 4), fontsize=8)
    ax.scatter(xb, yb, color='#2B8D6E')
    ax.set(xlabel='Directional sharpness, radius 0.05', ylabel='Validation NLL')
    ax.grid(True, ls=':', alpha=0.3)
    return path, save(fig, 'p3_sharpness.png')


def schedules(plt):
    bundle = runs('schedules')
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    long = max(bundle['runs']['constant'], key=lambda row: row['epochs'])
    history = long['history']
    ax.plot(
        [point['epochs'] for point in history],
        [point['validation_nll'] for point in history],
        color='#5178D5',
        label='Constant LR 3.33e-4, one trajectory',
    )
    linear = sorted(bundle['runs']['linear'], key=lambda row: row['epochs'])
    ax.scatter(
        [row['epochs'] for row in linear],
        [row['validation_nll'] for row in linear],
        color='#7030A0',
        zorder=3,
        label='Linear decay, LR selected per horizon',
    )
    ax.set(xlabel='Epochs over the same 30k chunks', ylabel='Validation NLL')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, ls=':', alpha=0.3)
    return save(fig, 'p3_schedules.png')


def main():
    plt = _style()
    _, fit = parameters(plt)
    ensemble(plt)
    batch(plt)
    schedules(plt)
    print(json.dumps({'parameter_fit_millions': fit}, indent=2))


if __name__ == '__main__':
    main()
