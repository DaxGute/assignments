"""Figures for runs this assignment trained. A curve is final only at the target epoch."""

from pathlib import Path
import json

OUT = Path('experiments/a6_generalization/deliverables/figures')
DENSE_D8 = 1.2627138263769448


def final_nll(history, target_epochs):
    if not history or float(history[-1]['epochs']) < float(target_epochs) - 0.05:
        return None
    return float(history[-1]['validation_nll'])


def finished_history(path, target_epochs=32):
    """A curve is drawn as a finished replication only at the target epoch."""
    file = Path(path)
    if file.is_dir():
        file = file / 'history.json'
    if not file.is_file():
        return None
    history = json.loads(file.read_text())
    if final_nll(history, target_epochs) is None:
        return None
    return history


def endpoint_record(history, baseline, target_epochs=32):
    """The prescribed score is the last epoch, not the best logged checkpoint."""
    final = final_nll(history, target_epochs)
    if final is None:
        return None
    best = min(history, key=lambda row: float(row['validation_nll']))
    return {
        'final': final,
        'delta': final - float(baseline),
        'best': float(best['validation_nll']),
        'best_epoch': float(best['epochs']),
        'endpoint_is_minimum': abs(float(best['validation_nll']) - final) <= 1e-12,
    }


def curves(plt, series, baseline=DENSE_D8, name='trained_curves.png'):
    """series: list of (label, history). Partial histories are drawn but not labeled as finals."""
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for label, history in series:
        ax.plot(
            [row['epochs'] for row in history],
            [row['validation_nll'] for row in history],
            marker='o',
            ms=3.5,
            label=label,
        )
    if baseline is not None:
        ax.axhline(baseline, color='#444444', lw=0.8, ls='--', label='dense d8, 1.2627')
    ax.set_xlabel('Epochs over the same 30,000 chunks')
    ax.set_ylabel('Validation NLL')
    ax.legend(frameon=False, fontsize=8)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches='tight')
    return path


def load(path):
    return json.loads(Path(path).read_text())


RUNS = Path('results/a6_generalization/runs')


def _history(name):
    return load(RUNS / name / 'history.json')


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


def wave1_figures():
    """Curves and the one-epoch fine-tuning grid. Numbers come from history.json."""
    plt = _style()
    made = []
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for ax, names, title in (
        (axes[0], (0.001, 0.003, 0.01), 'Four experts'),
        (axes[1], (0.001, 0.003, 0.01), 'Six-pass loop'),
    ):
        prefix = 'a6-moe' if ax is axes[0] else 'a6-loop'
        for lr in names:
            history = _history(f'{prefix}-lr{lr:g}-s42')
            ax.plot(
                [row['epochs'] for row in history],
                [row['validation_nll'] for row in history],
                marker='o', ms=3, label=f'LR {lr:g}',
            )
        if prefix == 'a6-moe':
            replica, label = 'a6-moe-lr0.003-s43', 'LR 0.003, seed 43'
        else:
            replica, label = 'a6-loop-lr0.001-s43', 'LR 0.001, seed 43'
        history = finished_history(RUNS / replica / 'history.json')
        if history is not None:
            ax.plot(
                [row['epochs'] for row in history],
                [row['validation_nll'] for row in history],
                color='#9B2226', ls='--', marker='o', ms=3, label=label,
            )
        ax.axhline(DENSE_D8, color='#444444', lw=0.8, ls=':', label='dense d8')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation NLL')
        ax.legend(frameon=False, fontsize=8)
    made.append(save_fig(fig, 'p1_trained_curves.png'))

    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for lr in (0.003, 0.01, 0.03):
        history = _history(f'a6-hyperball-lr{lr:g}-s42')
        ax.plot(
            [row['epochs'] for row in history],
            [row['validation_nll'] for row in history],
            marker='o', ms=3, label=f'LR {lr:g}',
        )
    history = finished_history(RUNS / 'a6-hyperball-lr0.03-s43' / 'history.json')
    if history is not None:
        ax.plot(
            [row['epochs'] for row in history],
            [row['validation_nll'] for row in history],
            color='#9B2226', ls='--', marker='o', ms=3, label='LR 0.03, seed 43',
        )
    ax.axhline(DENSE_D8, color='#444444', lw=0.8, ls='--', label='AdamW WD 3.2')
    ax.axhline(1.4057004241179676, color='#888888', lw=0.8, ls=':', label='AdamW WD 0.1')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation NLL')
    ax.legend(frameon=False, fontsize=8)
    made.append(save_fig(fig, 'p2_hyperball.png'))

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    inits = ('related', 'dclm', 'scratch', 'noise')
    lrs = (1e-4, 3e-4, 1e-3, 3e-3)
    import numpy as np

    width = 0.18
    positions = np.arange(len(inits))
    for index, lr in enumerate(lrs):
        heights = []
        for init in inits:
            history = _history(f'a6-ft-{init}-n30000-e1-lr{lr:g}-wd3.2-s42')
            heights.append(history[-1]['validation_nll'])
        axes[0].bar(positions + (index - 1.5) * width, heights, width=width, label=f'{lr:g}')
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(['Related', 'DCLM', 'Scratch', 'Noise'])
    axes[0].set_ylabel('Validation NLL after 1 epoch')
    axes[0].set_ylim(1.20, 1.40)
    axes[0].legend(title='Learning rate', frameon=False, fontsize=7, title_fontsize=8)
    axes[0].set_title('Endpoint')
    for lr in lrs:
        history = _history(f'a6-ft-related-n30000-e1-lr{lr:g}-wd3.2-s42')
        axes[1].plot(
            [row['epochs'] for row in history],
            [row['validation_nll'] for row in history],
            marker='o', ms=2.5, label=f'{lr:g}',
        )
    start = json.loads(
        (RUNS / 'a6-ft-related-n30000-e1-lr0.0003-wd3.2-s42' / 'initial.json').read_text()
    )
    axes[1].axhline(start['validation_nll'], color='#444444', lw=0.8, ls='--', label='step 0')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation NLL')
    axes[1].set_title('Related DNA during the pass')
    axes[1].legend(title='Learning rate', frameon=False, fontsize=7, title_fontsize=8)
    made.append(save_fig(fig, 'p4_one_epoch.png'))
    made.append(pool_scaling_figure(plt))
    return made


def pool_scaling_figure(plt):
    """Sixteen-pass endpoints against unique sequences. Both panels are the same four curves."""
    import numpy as np
    from experiments.a6_generalization.analysis import power_law, predict

    counts = np.array([1000, 3000, 10000, 30000], dtype=float)
    series = {
        'Related': np.array([3.4052941386699676, 2.806183057717979, 1.417181362107396, 1.2647764340471477]),
        'Scratch': np.array([1.311475142739713, 1.3125724094435574, 1.2936271058283746, 1.2636805066037924]),
        'DCLM': np.array([1.3111210491843521, 1.3602688429858536, 1.2909381982572377, 1.2612853994648903]),
        'Noise': np.array([1.3417727962136268, 1.3032283446490764, 1.285362226407975, 1.2631211679652332]),
    }
    colors = {'Related': '#9B2226', 'Scratch': '#1D3557', 'DCLM': '#2A9D8F', 'Noise': '#9A7D0A'}
    fit = power_law(counts, series['Noise'])
    grid = np.geomspace(1000, 30000, 80)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    for ax in axes:
        for label, values in series.items():
            ax.plot(counts, values, marker='o', ms=4, color=colors[label], label=label)
        ax.plot(grid, predict(fit, grid), color=colors['Noise'], ls='--', lw=1, label='Noise fit')
        ax.set_xscale('log')
        ax.set_xticks(counts)
        ax.set_xticklabels(['1k', '3k', '10k', '30k'])
        ax.set_xlabel('Unique sequences')
        ax.set_ylabel('Validation NLL')
    axes[0].set_title('Full range')
    axes[1].set_ylim(1.24, 1.45)
    axes[1].set_title('Same endpoints, expanded')
    axes[0].legend(frameon=False, fontsize=7)
    return save_fig(fig, 'p4_pools.png')


def save_fig(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches='tight')
    return path


if __name__ == '__main__':
    for path in wave1_figures():
        print(path)
