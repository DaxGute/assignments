"""Numerical fits already used to build the A2 batches.

These functions call ``experiments.a2.scaling``. They do not define a second
fitting formula.
"""

import math

from experiments.a2.scaling import (
    P1_LARGE_TOKENS,
    P1_TARGET_TOKENS,
    P1_TOKENS,
    PRODUCT_SOURCE_TOKENS,
    PRODUCT_TARGET_LR,
    PRODUCT_TARGET_TOKENS,
    HYPERBALL_SOURCE_TOKENS,
    HYPERBALL_TARGET_TOKENS,
    best_measured,
    bivariate_optimum,
    fit_batch1_sources,
    fit_hyperball,
    fit_lr_wd_product,
    fit_token_lr,
    power_law,
    predict_power_law,
    quadratic_log_optimum,
)


def persisted_batch1_fits():
    """Fits written by Batch 1, or the same fit function if the file is absent."""
    import json

    from experiments.a2.helpers.paths import MANIFEST_DIR

    path = MANIFEST_DIR / "batch1_fits.json"
    if path.is_file():
        return json.loads(path.read_text())
    return fit_batch1_sources()


def log_quadratic_loss(fit, value):
    coefficients = fit["coefficients"]
    log_value = math.log(float(value))
    return float(
        coefficients["intercept"]
        + coefficients["log_linear"] * log_value
        + coefficients["log_quadratic"] * log_value**2
    )


def bivariate_loss(fit, learning_rate, weight_decay):
    coefficients = fit["coefficients"]
    log_lr = math.log(float(learning_rate))
    log_wd = math.log(float(weight_decay))
    return float(
        coefficients["a"]
        + coefficients["b"] * log_lr
        + coefficients["c"] * log_wd
        + coefficients["d"] * log_lr**2
        + coefficients["e"] * log_lr * log_wd
        + coefficients["f"] * log_wd**2
    )


def r_squared(fit):
    if fit is None:
        return None
    if "r_squared" in fit:
        return fit["r_squared"]
    law = fit.get("power_law") if isinstance(fit, dict) else None
    if isinstance(law, dict):
        return law.get("r_squared")
    return None
