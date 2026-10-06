"""Scaling-law fits for Assignment 2.

Every coefficient is computed from the supplied sweep CSV. Nothing in this
module is a hand-copied learning rate or weight decay.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from experiments.a2.provided_sweeps import PARTS, load


P1_TOKENS = (
    153_600_000,
    307_200_000,
    614_400_000,
    1_228_800_000,
    1_843_200_000,
    2_457_600_000,
)
P1_LARGE_TOKENS = (1_228_800_000, 1_843_200_000, 2_457_600_000)
P1_TARGET_TOKENS = 4_915_200_000
HYPERBALL_SOURCE_TOKENS = (153_600_000, 307_200_000, 614_400_000)
HYPERBALL_TARGET_TOKENS = 1_228_800_000
PRODUCT_SOURCE_TOKENS = (153_600_000, 307_200_000, 614_400_000, 1_228_800_000)
PRODUCT_TARGET_TOKENS = 2_457_600_000
PRODUCT_TARGET_LR = 0.003


def load_supplied():
    """Unique supplied runs. A run may belong to more than one part."""
    rows = []
    seen = set()
    for part in PARTS:
        for row in load(part):
            if row["run_id"] in seen:
                continue
            seen.add(row["run_id"])
            rows.append(row)
    return rows


def _group_tokens(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["tokens"])].append(row)
    return dict(grouped)


def quadratic_log_optimum(values, losses):
    """Least-squares loss ~ a + b log(x) + c log(x)^2, then the vertex.

    With three samples this interpolates the measured points. The vertex is
    used only when the parabola opens upwards. Otherwise the best measured
    sample is the optimum.
    """
    x = np.log(np.asarray(values, dtype=float))
    y = np.asarray(losses, dtype=float)
    if len(x) < 3:
        raise ValueError("a quadratic optimum needs at least three samples")
    curvature, slope, intercept = (float(v) for v in np.polyfit(x, y, 2))
    best = int(np.argmin(y))
    record = {
        "coefficients": {
            "log_quadratic": curvature,
            "log_linear": slope,
            "intercept": intercept,
        },
        "samples": [
            {"value": float(value), "loss": float(loss)}
            for value, loss in zip(values, losses)
        ],
        "best_sampled_value": float(values[best]),
        "best_sampled_loss": float(y[best]),
    }
    if curvature <= 0 or not math.isfinite(curvature):
        record.update(
            optimum=float(values[best]),
            method="best_sampled",
            vertex_inside_sample_range=True,
        )
        return record
    log_vertex = -slope / (2 * curvature)
    optimum = float(math.exp(log_vertex))
    record.update(
        optimum=optimum,
        method="log_quadratic_vertex",
        vertex_inside_sample_range=bool(x.min() <= log_vertex <= x.max()),
        fitted_loss_at_optimum=float(
            intercept + slope * log_vertex + curvature * log_vertex**2
        ),
    )
    return record


def power_law(controls, responses):
    """response = prefactor * control ** exponent, fit in log space."""
    x = np.log(np.asarray(controls, dtype=float))
    y = np.log(np.asarray(responses, dtype=float))
    exponent, intercept = (float(v) for v in np.polyfit(x, y, 1))
    predicted = intercept + exponent * x
    residual = y - predicted
    total = y - y.mean()
    total_ss = float(np.dot(total, total))
    r_squared = 1.0 if total_ss == 0 else 1.0 - float(np.dot(residual, residual)) / total_ss
    return {
        "exponent": exponent,
        "log_prefactor": intercept,
        "prefactor": float(math.exp(intercept)),
        "r_squared": r_squared,
        "controls": [float(v) for v in controls],
        "responses": [float(v) for v in responses],
    }


def predict_power_law(fit, control):
    return float(fit["prefactor"] * float(control) ** fit["exponent"])


def _optima(rows, value_key):
    grouped = _group_tokens(rows)
    optima = []
    for tokens in sorted(grouped):
        group = sorted(grouped[tokens], key=lambda row: row[value_key])
        fit = quadratic_log_optimum(
            [row[value_key] for row in group],
            [row["final_val_loss"] for row in group],
        )
        optima.append({"tokens": int(tokens), "n_runs": len(group), **fit})
    return optima


def bivariate_optimum(rows):
    """Fit L(log LR, log WD) and return the critical point when it is a minimum."""
    if len(rows) < 6:
        raise ValueError("the joint quadratic has six coefficients")
    log_lr = np.log(np.asarray([row["learning_rate"] for row in rows], dtype=float))
    log_wd = np.log(np.asarray([row["weight_decay"] for row in rows], dtype=float))
    loss = np.asarray([row["final_val_loss"] for row in rows], dtype=float)
    design = np.column_stack(
        [
            np.ones(len(rows)),
            log_lr,
            log_wd,
            log_lr**2,
            log_lr * log_wd,
            log_wd**2,
        ]
    )
    coefficients, _, rank, _ = np.linalg.lstsq(design, loss, rcond=None)
    a, b, c, d, e, f = (float(v) for v in coefficients)
    hessian = np.array([[2 * d, e], [e, 2 * f]], dtype=float)
    eigenvalues = np.linalg.eigvalsh(hessian)
    best = min(rows, key=lambda row: row["final_val_loss"])
    record = {
        "coefficients": {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f},
        "hessian_eigenvalues": [float(v) for v in eigenvalues],
        "fit_rank": int(rank),
        "n_runs": len(rows),
        "best_sampled_lr": float(best["learning_rate"]),
        "best_sampled_wd": float(best["weight_decay"]),
        "best_sampled_loss": float(best["final_val_loss"]),
        "best_sampled_run_id": best["run_id"],
    }
    minimum = bool(np.all(eigenvalues > 1e-12)) and math.isfinite(float(np.linalg.det(hessian)))
    if not minimum:
        record.update(
            method="best_sampled",
            optimum_lr=float(best["learning_rate"]),
            optimum_wd=float(best["weight_decay"]),
        )
    else:
        log_lr_star, log_wd_star = np.linalg.solve(
            hessian, np.array([-b, -c], dtype=float)
        )
        record.update(
            method="critical_point",
            optimum_lr=float(math.exp(log_lr_star)),
            optimum_wd=float(math.exp(log_wd_star)),
            vertex_inside_log_range=bool(
                log_lr.min() <= log_lr_star <= log_lr.max()
                and log_wd.min() <= log_wd_star <= log_wd.max()
            ),
        )
    record["optimum_product"] = float(record["optimum_lr"] * record["optimum_wd"])
    return record


def fit_token_lr(rows, token_budgets):
    selected = [row for row in rows if int(row["tokens"]) in token_budgets]
    found = {int(row["tokens"]) for row in selected}
    missing = [tokens for tokens in token_budgets if tokens not in found]
    if missing:
        raise ValueError(f"missing token budgets {missing}")
    optima = _optima(selected, "learning_rate")
    law = power_law(
        [row["tokens"] for row in optima],
        [row["optimum"] for row in optima],
    )
    return {
        "optima": optima,
        "power_law": law,
        "prediction_tokens": P1_TARGET_TOKENS,
        "predicted_lr": predict_power_law(law, P1_TARGET_TOKENS),
    }


def fit_hyperball(rows):
    optima = _optima(rows, "learning_rate")
    if [row["tokens"] for row in optima] != list(HYPERBALL_SOURCE_TOKENS):
        raise ValueError(f"unexpected Hyperball budgets: {optima}")
    law = power_law(
        [row["tokens"] for row in optima],
        [row["optimum"] for row in optima],
    )
    return {
        "optima": optima,
        "power_law": law,
        "prediction_tokens": HYPERBALL_TARGET_TOKENS,
        "predicted_lr": predict_power_law(law, HYPERBALL_TARGET_TOKENS),
    }


def fit_lr_wd_product(rows):
    grouped = _group_tokens(rows)
    if sorted(grouped) != list(PRODUCT_SOURCE_TOKENS):
        raise ValueError(f"unexpected joint-sweep budgets: {sorted(grouped)}")
    budgets = []
    for tokens in PRODUCT_SOURCE_TOKENS:
        budgets.append({"tokens": int(tokens), **bivariate_optimum(grouped[tokens])})
    law = power_law(
        [row["tokens"] for row in budgets],
        [row["optimum_product"] for row in budgets],
    )
    predicted_product = predict_power_law(law, PRODUCT_TARGET_TOKENS)
    return {
        "budgets": budgets,
        "power_law": law,
        "prediction_tokens": PRODUCT_TARGET_TOKENS,
        "predicted_product": predicted_product,
        "target_peak_lr": PRODUCT_TARGET_LR,
        "predicted_wd": float(predicted_product / PRODUCT_TARGET_LR),
    }


def best_measured(rows):
    return min(rows, key=lambda row: row["final_val_loss"])


def fit_batch1_sources():
    """All Batch 1 source fits. Safe to call on a laptop; it only reads the CSV."""
    p1 = load("P1a") + load("P1b")
    p1a = load("P1a")
    p2a = load("P2a")
    hyperball = load("P1d")
    small = [row for row in p1a if row["tokens"] == 153_600_000]
    joint_small = [row for row in p2a if row["tokens"] == 153_600_000]
    global_small = small + [
        row for row in joint_small if row["run_id"] not in {item["run_id"] for item in small}
    ]
    best_problem1 = best_measured(small)
    best_joint = best_measured(joint_small)
    best_global = best_measured(global_small)
    return {
        "p1c_all6": fit_token_lr(p1, P1_TOKENS),
        "p1c_large3": fit_token_lr(p1, P1_LARGE_TOKENS),
        "p1d_hyperball": fit_hyperball(hyperball),
        "p2c_product": fit_lr_wd_product(p2a),
        "p1_1536m_best_sampled": {
            "learning_rate": float(best_problem1["learning_rate"]),
            "weight_decay": float(best_problem1["weight_decay"]),
            "final_val_loss": float(best_problem1["final_val_loss"]),
            "run_id": best_problem1["run_id"],
            "run_url": best_problem1["run_url"],
            "tokens": int(best_problem1["tokens"]),
        },
        "p2_1536m_best_joint": {
            "learning_rate": float(best_joint["learning_rate"]),
            "weight_decay": float(best_joint["weight_decay"]),
            "final_val_loss": float(best_joint["final_val_loss"]),
            "run_id": best_joint["run_id"],
            "run_url": best_joint["run_url"],
            "tokens": int(best_joint["tokens"]),
            "global_best_run_id": best_global["run_id"],
            "global_best_loss": float(best_global["final_val_loss"]),
        },
    }
