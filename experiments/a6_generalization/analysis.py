"""Small NumPy helpers for parameter, ensemble, and transfer scaling laws."""

import itertools
import numpy as np


def power_law(x, y):
    """Descriptive L(x)=C+A*x**(-alpha); short curves may not identify C."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or x.shape != y.shape or np.any(x <= 0) or not np.isfinite(y).all():
        raise ValueError('Provide at least three finite losses and positive x values.')
    best = None
    for exponent in np.geomspace(0.02, 4.0, 400):
        basis = x**-exponent
        a, c = np.linalg.lstsq(np.column_stack((basis, np.ones_like(x))), y, rcond=None)[0]
        if a < 0 or c < 0 or c > y.min():
            continue
        error = np.mean((c + a * basis - y) ** 2)
        if best is None or error < best[0]:
            best = (error, float(a), float(exponent), float(c))
    if best is None:
        raise ValueError('No decreasing positive-offset fit; inspect the measured curve.')
    return dict(amplitude=best[1], exponent=best[2], offset=best[3])


def predict(fit, x):
    return fit['offset'] + fit['amplitude'] * np.asarray(x, dtype=float) ** -fit['exponent']


def equivalent_samples(sample_counts, scratch_losses, target_loss):
    """How many scratch samples match a pretrained loss, inside the measured curve.

    The mapping is log-linear between the two surrounding scratch points. Losses
    outside that curve are not assigned a sample count.
    """
    n = np.asarray(sample_counts, dtype=float)
    y = np.asarray(scratch_losses, dtype=float)
    target = float(target_loss)
    if len(n) < 2 or n.shape != y.shape or np.any(n <= 0) or not np.isfinite(y).all():
        raise ValueError('Need at least two finite scratch losses and positive sample counts.')
    order = np.argsort(n)
    n, y = n[order], y[order]
    if np.any(np.diff(y) >= 0):
        return dict(status='not_monotone', bracket=None, interpolated=None)
    if target >= y[0]:
        return dict(status='worse_than_all_scratch', bracket=None, interpolated=None)
    if target <= y[-1]:
        return dict(status='better_than_all_scratch', bracket=None, interpolated=None)
    index = int(np.searchsorted(-y, -target)) - 1
    n0, n1 = float(n[index]), float(n[index + 1])
    y0, y1 = float(y[index]), float(y[index + 1])
    fraction = (y0 - target) / (y0 - y1)
    interpolated = float(np.exp(np.log(n0) + fraction * (np.log(n1) - np.log(n0))))
    return dict(
        status='inside',
        bracket=[n0, n1],
        interpolated=interpolated,
        scratch_losses=[y0, y1],
    )


def ensemble_losses(logits, labels):
    """Mean chunk NLL over ALL subsets, one row per member count K.

    logits: list of (chunks, positions, vocabulary) arrays; labels: next tokens.
    Use small chunk slices of memory-mapped arrays to bound memory.
    """
    probabilities = []
    for z in logits:
        z = np.asarray(z, dtype=np.float64)
        z = z - z.max(-1, keepdims=True)
        p = np.exp(z)
        probabilities.append(p / p.sum(-1, keepdims=True))
    losses = []
    for k in range(1, len(logits) + 1):
        members = []
        for ids in itertools.combinations(range(len(logits)), k):
            p = np.mean([probabilities[i] for i in ids], axis=0)
            correct = np.take_along_axis(p, labels[..., None], axis=-1)[..., 0]
            members.append(-np.log(np.maximum(correct, 1e-30)).mean(-1))
        losses.append(np.mean(members, axis=0))
    return np.asarray(losses)


def validation_token_labels(rows):
    """BOS is context, not a target. ``rows`` are the saved validation sequences."""
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] < 2:
        raise ValueError('Validation rows must be sequences with a leading BOS token.')
    return rows[:, 1:]


def probability_ensemble_curve(logits, labels, sequences_per_slice=64):
    """Validation NLL at each member count, in sequence slices.

    The value at each K is the mean over sequences of the subset-averaged
    per-sequence NLL. Sequences have equal length, so this is the token mean.
    It matches ``ensemble_losses(...).mean(-1)`` and does not average logits
    or average the members' losses.
    """
    labels = np.asarray(labels)
    n = int(labels.shape[0])
    if any(len(z) != n for z in logits):
        raise ValueError('Every member must cover the same sequences.')
    totals = None
    seen = 0
    for start in range(0, n, sequences_per_slice):
        stop = min(n, start + sequences_per_slice)
        piece = ensemble_losses(
            [np.asarray(z[start:stop]) for z in logits],
            labels[start:stop],
        )
        totals = piece.sum(-1) if totals is None else totals + piece.sum(-1)
        seen += piece.shape[-1]
    return totals / seen


def chunk_bootstrap(chunks, repeats=1000, seed=42):
    """Resample columns together across K/methods; seed uncertainty is separate."""
    a = np.asarray(chunks)
    indices = np.random.default_rng(seed).integers(0, a.shape[-1], (repeats, a.shape[-1]))
    return a[..., indices].mean(-1)
