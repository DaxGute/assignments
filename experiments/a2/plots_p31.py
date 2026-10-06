"""Run and plot the complete Assignment 2 Problem 3.1 noisy quadratic study.

The default command is the assignment-quality experiment (512 independent
Monte Carlo trajectories per point).  ``--quick`` exists only as a smoke test.
All random streams are derived deterministically from ``--seed`` and common
random numbers are used across learning rates within a sweep.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


N_EXAMPLES = 8_192
SOURCE_BATCHES = (1, 2, 4, 8, 16, 32, 64)
ALL_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
TARGET_BATCH = 256
SIGMAS = (1.0, 10.0, 100.0, 300.0)
MOMENTUM_BATCHES = (16, 256)
DEFAULT_SAMPLES = 512
DEFAULT_SEED = 42
ADAM_BETA2 = 0.95
EPSILON = 1e-8

REPO_ROOT = Path(__file__).resolve().parents[2]
PLOT_ROOT = REPO_ROOT / "outputs" / "a2" / "plots" / "p31"
RESULT_PATH = REPO_ROOT / "outputs" / "a2" / "data" / "p31" / "nqm_results.json"

COLORS = ("#1f4e79", "#c0392b", "#0e7c66", "#b86e00", "#6c3483", "#117a65")


def _grid(lo: float, hi: float, count: int) -> np.ndarray:
    """A reproducible, endpoint-inclusive logarithmic LR grid."""
    return np.geomspace(lo, hi, count, dtype=np.float64)


def _choices(quick: bool) -> dict[str, Any]:
    # Upper bounds stay inside the deterministic stability range for the
    # highest-curvature direction. Adaptive methods tolerate the wider range.
    count = 17 if quick else 49
    return {
        # Stay just inside the deterministic stability limit
        # 2 / lambda_max = 0.2. The 49-point default grid still brackets the
        # large-batch optimum without introducing divergent source runs.
        "sgd_lrs": _grid(1e-5, 0.195, count),
        "adaptive_lrs": _grid(1e-5, 0.8, count),
        "momentum_lrs": _grid(1e-5, 0.35, count),
        "beta1_values": np.asarray(
            (0.0, 0.5, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995), dtype=np.float64
        ),
        "fit_window": 2,
        "curve_points": 65 if quick else 129,
    }


def _stream_seed(seed: int, *keys: int) -> np.random.SeedSequence:
    """Stable named stream without Python's process-randomized hash()."""
    return np.random.SeedSequence([int(seed), *(int(key) for key in keys)])


def _model(model: str) -> tuple[np.ndarray, np.ndarray]:
    if model == "2d":
        return np.asarray((1.0, 10.0)), np.asarray((1.0, 0.1))
    if model == "scalar":
        return np.asarray((1.0,)), np.asarray((2.0,))
    raise ValueError(f"unknown curvature model {model!r}")


def simulate_sweep(
    *,
    optimizer: str,
    batch: int,
    lrs: np.ndarray,
    samples: int,
    seed: int,
    sigma: float = 1.0,
    model: str = "2d",
    beta1: float | None = None,
    momentum: float = 0.0,
    record_curve: bool = False,
    curve_points: int = 129,
) -> dict[str, Any]:
    """Vectorized Monte Carlo sweep over LR and independent trajectories.

    The only Python loop is over optimizer updates. LR, sample, and parameter
    axes are NumPy-vectorized. Each LR sees the same independent initializations
    and per-update noise, reducing comparison variance without changing any
    point's Monte Carlo distribution.
    """
    if N_EXAMPLES % batch:
        raise ValueError("batch must divide 8192")
    curvature, init_std = _model(model)
    dims = len(curvature)
    steps = N_EXAMPLES // batch
    lrs = np.asarray(lrs, dtype=np.float64)
    if np.any(lrs <= 0):
        raise ValueError("learning rates must be positive")

    optimizer_ids = {"sgd": 1, "rmsprop": 2, "adam": 3}
    model_ids = {"2d": 1, "scalar": 2}
    beta_key = round(10_000 * (0.0 if beta1 is None else beta1))
    momentum_key = round(10_000 * momentum)
    rng = np.random.default_rng(
        _stream_seed(
            seed,
            optimizer_ids[optimizer],
            model_ids[model],
            batch,
            round(sigma * 1_000),
            beta_key,
            momentum_key,
        )
    )
    initial = rng.normal(size=(samples, dims)) * init_std
    w = np.broadcast_to(initial, (len(lrs), samples, dims)).copy()
    lr_view = lrs[:, None, None]
    noise_scale = float(sigma) / math.sqrt(batch)

    velocity = np.zeros_like(w) if optimizer == "sgd" and momentum else None
    first = np.zeros_like(w) if optimizer in {"rmsprop", "adam"} else None
    second = np.zeros_like(w) if optimizer in {"rmsprop", "adam"} else None
    if optimizer == "rmsprop":
        beta1 = 0.0
    elif optimizer == "adam" and beta1 is None:
        beta1 = 0.9

    checkpoints: set[int] = set()
    curve: list[dict[str, Any]] = []
    if record_curve:
        checkpoints = set(
            int(value)
            for value in np.unique(np.rint(np.linspace(0, steps, min(curve_points, steps + 1))))
        )

        def append_curve(step: int) -> None:
            losses = 0.5 * np.sum(curvature * w * w, axis=2)
            curve.append(
                {
                    "step": int(step),
                    "examples_seen": int(step * batch),
                    "mean_losses": losses.mean(axis=1).tolist(),
                    "standard_errors": (
                        losses.std(axis=1, ddof=1) / math.sqrt(samples)
                    ).tolist(),
                }
            )

        append_curve(0)

    for step in range(1, steps + 1):
        noise = rng.normal(size=(samples, dims)) * noise_scale
        gradient = curvature * w + noise[None, :, :]
        if optimizer == "sgd":
            if velocity is None:
                direction = gradient
            else:
                velocity *= momentum
                velocity += gradient
                direction = velocity
        else:
            assert first is not None and second is not None and beta1 is not None
            first *= beta1
            first += (1.0 - beta1) * gradient
            second *= ADAM_BETA2
            second += (1.0 - ADAM_BETA2) * gradient * gradient
            first_hat = first / (1.0 - beta1**step)
            second_hat = second / (1.0 - ADAM_BETA2**step)
            direction = first_hat / (np.sqrt(second_hat) + EPSILON)
        w -= lr_view * direction
        if record_curve and step in checkpoints:
            append_curve(step)

    losses = 0.5 * np.sum(curvature * w * w, axis=2)
    means = losses.mean(axis=1)
    standard_errors = losses.std(axis=1, ddof=1) / math.sqrt(samples)
    return {
        "optimizer": optimizer,
        "model": model,
        "batch": int(batch),
        "steps": int(steps),
        "sigma": float(sigma),
        "beta1": None if beta1 is None else float(beta1),
        "beta2": ADAM_BETA2 if optimizer != "sgd" else None,
        "momentum": float(momentum) if optimizer == "sgd" else None,
        "learning_rates": lrs.tolist(),
        "mean_final_losses": means.tolist(),
        "standard_errors": standard_errors.tolist(),
        "samples": int(samples),
        "curve": curve,
    }


def _estimate_optimum(sweep: dict[str, Any], fit_window: int = 2) -> dict[str, Any]:
    """Estimate the LR optimum with a local quadratic in log LR.

    The vertex is accepted only for positive curvature and only inside the
    local fit interval; otherwise the best sampled point is reported.
    """
    lrs = np.asarray(sweep["learning_rates"], dtype=float)
    losses = np.asarray(sweep["mean_final_losses"], dtype=float)
    best = int(np.nanargmin(losses))
    left = max(0, best - fit_window)
    right = min(len(lrs), best + fit_window + 1)
    fit_indices = np.arange(left, right)
    result = {
        "best_index": best,
        "best_sampled_lr": float(lrs[best]),
        "best_sampled_loss": float(losses[best]),
        "best_sampled_standard_error": float(sweep["standard_errors"][best]),
        "fit_indices": fit_indices.tolist(),
        "fit_method": "best_sampled",
        "optimum_lr": float(lrs[best]),
        "estimated_minimum_loss": float(losses[best]),
        "boundary_optimum": bool(best in (0, len(lrs) - 1)),
    }
    if len(fit_indices) < 3:
        return result
    x = np.log(lrs[fit_indices])
    y = losses[fit_indices]
    quadratic, linear, intercept = (float(v) for v in np.polyfit(x, y, 2))
    result["fit_coefficients"] = {
        "log_lr_squared": quadratic,
        "log_lr": linear,
        "intercept": intercept,
    }
    if quadratic <= 0 or not math.isfinite(quadratic):
        return result
    vertex = -linear / (2.0 * quadratic)
    if not (x.min() <= vertex <= x.max()):
        return result
    result.update(
        fit_method="local_log_quadratic",
        optimum_lr=float(math.exp(vertex)),
        estimated_minimum_loss=float(intercept + linear * vertex + quadratic * vertex**2),
    )
    return result


def _power_law(batches: list[int] | tuple[int, ...], optima: list[float]) -> dict[str, Any]:
    x = np.log(np.asarray(batches, dtype=float))
    y = np.log(np.asarray(optima, dtype=float))
    exponent, log_prefactor = (float(v) for v in np.polyfit(x, y, 1))
    fitted = log_prefactor + exponent * x
    residual_ss = float(np.sum((y - fitted) ** 2))
    total_ss = float(np.sum((y - y.mean()) ** 2))
    return {
        "exponent": exponent,
        "prefactor": float(math.exp(log_prefactor)),
        "log_prefactor": log_prefactor,
        "r_squared": 1.0 if total_ss == 0 else 1.0 - residual_ss / total_ss,
        "source_batches": [int(value) for value in batches],
        "source_optimal_lrs": [float(value) for value in optima],
    }


def _predict(fit: dict[str, Any], batch: int) -> float:
    return float(fit["prefactor"] * batch ** fit["exponent"])


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#dddddd",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def _save_bundle(
    name: str, figure: plt.Figure, rows: list[dict[str, Any]], metadata: dict[str, Any]
) -> list[str]:
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    base = PLOT_ROOT / name
    paths = [base.with_suffix(suffix) for suffix in (".png", ".pdf", ".csv", ".json")]
    figure.savefig(paths[0])
    figure.savefig(paths[1])
    plt.close(figure)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with paths[2].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    paths[3].write_text(json.dumps(_json_ready(metadata), indent=2) + "\n")
    return [str(path) for path in paths]


def _run_family(
    *,
    optimizer: str,
    batches: tuple[int, ...],
    lrs: np.ndarray,
    samples: int,
    seed: int,
    sigma: float = 1.0,
    model: str = "2d",
    beta1: float | None = None,
    momentum: float = 0.0,
    fit_window: int = 2,
) -> dict[str, Any]:
    sweeps: dict[str, Any] = {}
    optima: dict[str, Any] = {}
    for batch in batches:
        sweep = simulate_sweep(
            optimizer=optimizer,
            batch=batch,
            lrs=lrs,
            samples=samples,
            seed=seed,
            sigma=sigma,
            model=model,
            beta1=beta1,
            momentum=momentum,
        )
        sweeps[str(batch)] = sweep
        optima[str(batch)] = _estimate_optimum(sweep, fit_window)
    source_lrs = [optima[str(batch)]["optimum_lr"] for batch in SOURCE_BATCHES]
    law = _power_law(SOURCE_BATCHES, source_lrs)
    return {"sweeps": sweeps, "optima": optima, "source_power_law": law}


def _measure_prediction(
    family: dict[str, Any],
    *,
    optimizer: str,
    samples: int,
    seed: int,
    sigma: float,
    model: str,
    beta1: float | None,
) -> None:
    predicted_lr = _predict(family["source_power_law"], TARGET_BATCH)
    measured = simulate_sweep(
        optimizer=optimizer,
        batch=TARGET_BATCH,
        lrs=np.asarray([predicted_lr]),
        samples=samples,
        seed=seed,
        sigma=sigma,
        model=model,
        beta1=beta1,
    )
    tuned = family["optima"][str(TARGET_BATCH)]
    family["batch_256_prediction_test"] = {
        "predicted_lr": predicted_lr,
        "measured_loss": measured["mean_final_losses"][0],
        "standard_error": measured["standard_errors"][0],
        "tuned_optimum_lr": tuned["optimum_lr"],
        "best_tuned_sampled_lr": tuned["best_sampled_lr"],
        "best_tuned_sampled_loss": tuned["best_sampled_loss"],
        "prediction_loss_gap": float(
            measured["mean_final_losses"][0] - tuned["best_sampled_loss"]
        ),
        "lr_ratio_predicted_to_tuned": float(predicted_lr / tuned["optimum_lr"]),
        "raw_prediction_sweep": measured,
    }


def _plot_part_a(sgd: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    _style()
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    batches = np.asarray(ALL_BATCHES)
    optimum_lr = np.asarray([sgd["optima"][str(b)]["optimum_lr"] for b in batches])
    best_loss = np.asarray([sgd["optima"][str(b)]["best_sampled_loss"] for b in batches])
    law = sgd["source_power_law"]
    fit_x = np.geomspace(1, 512, 200)
    fit_y = law["prefactor"] * fit_x ** law["exponent"]
    axes[0].loglog(batches, optimum_lr, "o-", color=COLORS[0], label="tuned LR")
    axes[0].loglog(
        fit_x,
        fit_y,
        "--",
        color=COLORS[1],
        label=f"source fit $p={law['exponent']:.3f}$",
    )
    prediction = _predict(law, TARGET_BATCH)
    axes[0].scatter([TARGET_BATCH], [prediction], marker="*", s=100, color="#111111", label="B=256 prediction")
    axes[0].set(xlabel="Batch size B", ylabel="Optimal learning rate", title="SGD LR scaling")
    axes[0].legend(fontsize=8)
    axes[1].loglog(batches, best_loss, "o-", color=COLORS[2])
    axes[1].set(
        xlabel="Batch size B",
        ylabel="Best tuned expected final loss",
        title="SGD tuned performance",
    )
    rows = [
        {
            "batch": int(batch),
            "optimal_lr": float(lr),
            "best_sampled_loss": float(loss),
            "source_fit": bool(batch in SOURCE_BATCHES),
            "predicted_lr_at_256": prediction if batch == TARGET_BATCH else "",
        }
        for batch, lr, loss in zip(batches, optimum_lr, best_loss)
    ]
    return _save_bundle("p31a_sgd_scaling", figure, rows, {**metadata, "source_power_law": law})


def _plot_part_b(adaptive: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    _style()
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    rows: list[dict[str, Any]] = []
    for index, optimizer in enumerate(("rmsprop", "adam")):
        family = adaptive[optimizer]
        law = family["source_power_law"]
        source_y = [family["optima"][str(batch)]["optimum_lr"] for batch in SOURCE_BATCHES]
        fit_x = np.geomspace(1, TARGET_BATCH, 150)
        axes[0].loglog(SOURCE_BATCHES, source_y, "o", color=COLORS[index], label=optimizer.upper())
        axes[0].loglog(
            fit_x,
            law["prefactor"] * fit_x ** law["exponent"],
            "--",
            color=COLORS[index],
            label=f"{optimizer.upper()} fit $p={law['exponent']:.3f}$",
        )
        target = family["batch_256_prediction_test"]
        axes[1].bar(
            np.asarray([0, 1]) + index * 0.36,
            [target["measured_loss"], target["best_tuned_sampled_loss"]],
            width=0.34,
            color=COLORS[index],
            alpha=(0.65 if index == 0 else 0.9),
            label=optimizer.upper(),
        )
        for batch, lr in zip(SOURCE_BATCHES, source_y):
            rows.append({"optimizer": optimizer, "batch": batch, "kind": "source_optimum", "learning_rate": lr, "loss": ""})
        rows.extend(
            (
                {"optimizer": optimizer, "batch": 256, "kind": "predicted", "learning_rate": target["predicted_lr"], "loss": target["measured_loss"]},
                {"optimizer": optimizer, "batch": 256, "kind": "tuned", "learning_rate": target["best_tuned_sampled_lr"], "loss": target["best_tuned_sampled_loss"]},
            )
        )
    axes[0].set(xlabel="Batch size B", ylabel="Optimal learning rate", title="Adaptive source fits")
    axes[0].legend(fontsize=8)
    axes[1].set_xticks([0.18, 1.18], ["Predicted LR", "Tuned LR"])
    axes[1].set(ylabel="Expected final loss", title="Batch 256 transfer")
    axes[1].legend(fontsize=8)
    return _save_bundle("p31b_adaptive_transfer", figure, rows, metadata)


def _plot_part_c(noise: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    _style()
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    rows: list[dict[str, Any]] = []
    for optimizer_index, optimizer in enumerate(("rmsprop", "adam")):
        for model_index, model in enumerate(("2d", "scalar")):
            exponents = [
                noise[model][optimizer][str(sigma)]["source_power_law"]["exponent"]
                for sigma in SIGMAS
            ]
            axes[0].semilogx(
                SIGMAS,
                exponents,
                marker=("o" if model == "2d" else "s"),
                linestyle=("-" if model == "2d" else "--"),
                color=COLORS[optimizer_index],
                label=f"{optimizer.upper()}, {model}",
            )
            gaps = [
                noise[model][optimizer][str(sigma)]["batch_256_prediction_test"][
                    "prediction_loss_gap"
                ]
                for sigma in SIGMAS
            ]
            axes[1].semilogx(
                SIGMAS,
                gaps,
                marker=("o" if model == "2d" else "s"),
                linestyle=("-" if model == "2d" else "--"),
                color=COLORS[optimizer_index],
                label=f"{optimizer.upper()}, {model}",
            )
            for sigma, exponent, gap in zip(SIGMAS, exponents, gaps):
                rows.append(
                    {
                        "optimizer": optimizer,
                        "curvature_model": model,
                        "sigma": sigma,
                        "source_exponent": exponent,
                        "batch_256_prediction_loss_gap": gap,
                    }
                )
    axes[0].set(xlabel=r"Noise scale $\sigma$", ylabel="Fitted LR exponent p", title="Signal-to-noise changes scaling")
    axes[1].axhline(0.0, color="#444444", linewidth=0.8)
    axes[1].set(xlabel=r"Noise scale $\sigma$", ylabel="Predicted minus tuned loss", title="Curvature and transfer")
    for axis in axes:
        axis.legend(fontsize=7)
    return _save_bundle("p31c_noise_curvature", figure, rows, metadata)


def _curve_for_choice(
    *,
    optimizer: str,
    batch: int,
    lr: float,
    samples: int,
    seed: int,
    curve_points: int,
    beta1: float | None = None,
    momentum: float = 0.0,
) -> dict[str, Any]:
    return simulate_sweep(
        optimizer=optimizer,
        batch=batch,
        lrs=np.asarray([lr]),
        samples=samples,
        seed=seed,
        beta1=beta1,
        momentum=momentum,
        record_curve=True,
        curve_points=curve_points,
    )


def _plot_part_d(momentum_results: dict[str, Any], metadata: dict[str, Any]) -> list[str]:
    written: list[str] = []
    _style()
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    rows: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(MOMENTUM_BATCHES):
        values = []
        for mu in (0.0, 0.9):
            optimum = momentum_results["sgd"][str(batch)][str(mu)]["optimum"]
            values.append(optimum["best_sampled_loss"])
            rows.append(
                {
                    "optimizer": "sgd",
                    "batch": batch,
                    "momentum": mu,
                    "best_lr": optimum["best_sampled_lr"],
                    "best_loss": optimum["best_sampled_loss"],
                }
            )
        axes[0].bar(np.asarray([0, 1]) + batch_index * 0.36, values, width=0.34, color=COLORS[batch_index], label=f"B={batch}")
    axes[0].set_xticks([0.18, 1.18], [r"$\mu=0$", r"$\mu=0.9$"])
    axes[0].set(ylabel="Best expected final loss", title="SGD momentum, LR retuned")
    axes[0].legend(fontsize=8)
    for batch_index, batch in enumerate(MOMENTUM_BATCHES):
        beta_rows = momentum_results["adam"][str(batch)]["beta1_sweeps"]
        beta_values = [float(beta) for beta in beta_rows]
        losses = [beta_rows[str(beta)]["optimum"]["best_sampled_loss"] for beta in beta_values]
        axes[1].semilogy(beta_values, losses, "o-", color=COLORS[batch_index], label=f"B={batch}")
        for beta, loss in zip(beta_values, losses):
            optimum = beta_rows[str(beta)]["optimum"]
            rows.append(
                {
                    "optimizer": "adam",
                    "batch": batch,
                    "beta1": beta,
                    "best_lr": optimum["best_sampled_lr"],
                    "best_loss": loss,
                }
            )
    axes[1].set(xlabel=r"Adam $\beta_1$", ylabel="Best expected final loss", title=r"Adam $\beta_1$, LR retuned")
    axes[1].legend(fontsize=8)
    written.extend(_save_bundle("p31d_momentum_retuning", figure, rows, metadata))

    _style()
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.1), sharex=True)
    curve_rows: list[dict[str, Any]] = []
    for axis, batch in zip(axes, MOMENTUM_BATCHES):
        comparisons = momentum_results["learning_curves"][str(batch)]
        for index, (label, curve) in enumerate(comparisons.items()):
            points = curve["curve"]
            x = [point["examples_seen"] for point in points]
            y = [point["mean_losses"][0] for point in points]
            axis.semilogy(x, y, color=COLORS[index], linestyle=("-" if "best" in label else "--"), label=label.replace("_", " "))
            for point, loss in zip(points, y):
                curve_rows.append({"batch": batch, "series": label, "examples_seen": point["examples_seen"], "mean_loss": loss})
        axis.set(xlabel="Examples seen", ylabel="Expected loss", title=f"B={batch}")
        axis.legend(fontsize=7)
    written.extend(_save_bundle("p31d_learning_curves", figure, curve_rows, metadata))
    return written


def generate(*, samples: int = DEFAULT_SAMPLES, seed: int = DEFAULT_SEED, quick: bool = False) -> dict[str, Any]:
    """Generate all P3.1 simulations, raw data, summaries, and figure bundles."""
    if not quick and samples < 512:
        raise ValueError("assignment-quality generation requires at least 512 samples")
    if samples < 2:
        raise ValueError("at least two samples are required")
    started = time.time()
    choices = _choices(quick)
    common = {
        "samples": int(samples),
        "seed": int(seed),
        "quick": bool(quick),
        "n_examples": N_EXAMPLES,
        "source_batches": list(SOURCE_BATCHES),
        "all_batches": list(ALL_BATCHES),
        "target_batch": TARGET_BATCH,
        "gradient_noise": "N(0, sigma^2 I / B), independent across updates and samples",
        "initialization_2d": "w1 ~ N(0,1), w2 ~ N(0,0.1)",
        "initialization_scalar": "w ~ N(0,2)",
        "curvature_2d": [1.0, 10.0],
        "curvature_scalar": [1.0],
        "adam_beta2": ADAM_BETA2,
        "epsilon": EPSILON,
        "common_random_numbers_across_lrs": True,
        "lr_optimum_estimator": "five-point local quadratic in log LR; constrained to local interval; fallback to best sampled",
        "power_law_fit": "ordinary least squares: log(optimal LR) = log(c) + p log(B)",
        "grid_choices": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in choices.items()
        },
    }

    # (a) SGD through batch 512 and a source-only prediction tested at B=256.
    sgd = _run_family(
        optimizer="sgd",
        batches=ALL_BATCHES,
        lrs=choices["sgd_lrs"],
        samples=samples,
        seed=seed,
        fit_window=choices["fit_window"],
    )
    _measure_prediction(
        sgd,
        optimizer="sgd",
        samples=samples,
        seed=seed,
        sigma=1.0,
        model="2d",
        beta1=None,
    )

    # (b) Adaptive source fits and held-out B=256 prediction-versus-tuning.
    adaptive: dict[str, Any] = {}
    for optimizer, beta1 in (("rmsprop", 0.0), ("adam", 0.9)):
        family = _run_family(
            optimizer=optimizer,
            batches=SOURCE_BATCHES + (TARGET_BATCH,),
            lrs=choices["adaptive_lrs"],
            samples=samples,
            seed=seed,
            beta1=beta1,
            fit_window=choices["fit_window"],
        )
        _measure_prediction(
            family,
            optimizer=optimizer,
            samples=samples,
            seed=seed,
            sigma=1.0,
            model="2d",
            beta1=beta1,
        )
        adaptive[optimizer] = family

    # (c) Both curvature models, all four sigma values, and explicit B=256 tests.
    noise: dict[str, Any] = {"2d": {}, "scalar": {}}
    for model in ("2d", "scalar"):
        for optimizer, beta1 in (("rmsprop", 0.0), ("adam", 0.9)):
            noise[model][optimizer] = {}
            for sigma in SIGMAS:
                if model == "2d" and sigma == 1.0:
                    family = adaptive[optimizer]
                else:
                    family = _run_family(
                        optimizer=optimizer,
                        batches=SOURCE_BATCHES + (TARGET_BATCH,),
                        lrs=choices["adaptive_lrs"],
                        samples=samples,
                        seed=seed,
                        sigma=sigma,
                        model=model,
                        beta1=beta1,
                        fit_window=choices["fit_window"],
                    )
                    _measure_prediction(
                        family,
                        optimizer=optimizer,
                        samples=samples,
                        seed=seed,
                        sigma=sigma,
                        model=model,
                        beta1=beta1,
                    )
                noise[model][optimizer][str(sigma)] = family

    # (d) Momentum comparisons always retune LR.
    momentum_results: dict[str, Any] = {"sgd": {}, "adam": {}, "learning_curves": {}}
    for batch in MOMENTUM_BATCHES:
        momentum_results["sgd"][str(batch)] = {}
        for mu in (0.0, 0.9):
            sweep = simulate_sweep(
                optimizer="sgd",
                batch=batch,
                # Heavy-ball momentum expands the deterministic stability
                # interval to 2(1 + mu) / lambda_max. Keep the mu=0 control
                # on the ordinary SGD grid so every simulated point is finite.
                lrs=choices["sgd_lrs"] if mu == 0.0 else choices["momentum_lrs"],
                samples=samples,
                seed=seed,
                momentum=mu,
            )
            momentum_results["sgd"][str(batch)][str(mu)] = {
                "sweep": sweep,
                "optimum": _estimate_optimum(sweep, choices["fit_window"]),
            }
        beta_sweeps: dict[str, Any] = {}
        for beta1 in choices["beta1_values"]:
            sweep = simulate_sweep(
                optimizer="adam",
                batch=batch,
                lrs=choices["adaptive_lrs"],
                samples=samples,
                seed=seed,
                beta1=float(beta1),
            )
            beta_sweeps[str(float(beta1))] = {
                "sweep": sweep,
                "optimum": _estimate_optimum(sweep, choices["fit_window"]),
            }
        best_beta = min(
            beta_sweeps,
            key=lambda beta: beta_sweeps[beta]["optimum"]["best_sampled_loss"],
        )
        momentum_results["adam"][str(batch)] = {
            "beta1_sweeps": beta_sweeps,
            "best_beta1": float(best_beta),
        }

        sgd_no = momentum_results["sgd"][str(batch)]["0.0"]["optimum"]
        sgd_best = min(
            momentum_results["sgd"][str(batch)].items(),
            key=lambda item: item[1]["optimum"]["best_sampled_loss"],
        )
        adam_no = beta_sweeps["0.0"]["optimum"]
        adam_best = beta_sweeps[best_beta]["optimum"]
        momentum_results["learning_curves"][str(batch)] = {
            "sgd_no_momentum": _curve_for_choice(
                optimizer="sgd", batch=batch, lr=sgd_no["best_sampled_lr"],
                samples=samples, seed=seed, curve_points=choices["curve_points"],
            ),
            "sgd_best_momentum": _curve_for_choice(
                optimizer="sgd", batch=batch,
                lr=sgd_best[1]["optimum"]["best_sampled_lr"], samples=samples,
                seed=seed, curve_points=choices["curve_points"],
                momentum=float(sgd_best[0]),
            ),
            "adam_no_momentum": _curve_for_choice(
                optimizer="adam", batch=batch, lr=adam_no["best_sampled_lr"],
                samples=samples, seed=seed, curve_points=choices["curve_points"], beta1=0.0,
            ),
            "adam_best_momentum": _curve_for_choice(
                optimizer="adam", batch=batch, lr=adam_best["best_sampled_lr"],
                samples=samples, seed=seed, curve_points=choices["curve_points"],
                beta1=float(best_beta),
            ),
        }

    summary = {
        "sgd_source_exponent": sgd["source_power_law"]["exponent"],
        "rmsprop_source_exponent": adaptive["rmsprop"]["source_power_law"]["exponent"],
        "adam_source_exponent": adaptive["adam"]["source_power_law"]["exponent"],
        "sgd_batch_256_prediction": sgd["batch_256_prediction_test"],
        "adaptive_batch_256_predictions": {
            optimizer: adaptive[optimizer]["batch_256_prediction_test"]
            for optimizer in ("rmsprop", "adam")
        },
        "noise_exponents": {
            model: {
                optimizer: {
                    str(sigma): noise[model][optimizer][str(sigma)][
                        "source_power_law"
                    ]["exponent"]
                    for sigma in SIGMAS
                }
                for optimizer in ("rmsprop", "adam")
            }
            for model in ("2d", "scalar")
        },
        "momentum_best_settings": {
            str(batch): {
                "sgd": min(
                    (
                        {
                            "momentum": float(mu),
                            "learning_rate": row["optimum"]["best_sampled_lr"],
                            "loss": row["optimum"]["best_sampled_loss"],
                        }
                        for mu, row in momentum_results["sgd"][str(batch)].items()
                    ),
                    key=lambda row: row["loss"],
                ),
                "adam": {
                    "beta1": momentum_results["adam"][str(batch)]["best_beta1"],
                    "learning_rate": momentum_results["adam"][str(batch)][
                        "beta1_sweeps"
                    ][str(momentum_results["adam"][str(batch)]["best_beta1"])][
                        "optimum"
                    ][
                        "best_sampled_lr"
                    ],
                    "loss": momentum_results["adam"][str(batch)]["beta1_sweeps"][
                        str(momentum_results["adam"][str(batch)]["best_beta1"])
                    ]["optimum"]["best_sampled_loss"],
                },
            }
            for batch in MOMENTUM_BATCHES
        },
    }
    results = {
        "schema_version": 1,
        "experiment": "Assignment 2 Problem 3.1 noisy quadratic model",
        "configuration": common,
        "summary": summary,
        "part_a_sgd": sgd,
        "part_b_adaptive": adaptive,
        "part_c_noise_and_curvature": noise,
        "part_d_momentum": momentum_results,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(_json_ready(results), indent=2) + "\n")

    plot_metadata = {"configuration": common, "raw_results": str(RESULT_PATH)}
    written: list[str] = []
    written.extend(_plot_part_a(sgd, plot_metadata))
    written.extend(_plot_part_b(adaptive, plot_metadata))
    written.extend(_plot_part_c(noise, plot_metadata))
    written.extend(_plot_part_d(momentum_results, plot_metadata))
    elapsed = time.time() - started
    report = {
        "name": "p3.1",
        "written": written,
        "raw_results": str(RESULT_PATH),
        "samples": samples,
        "seed": seed,
        "quick": quick,
        "elapsed_seconds": elapsed,
        "summary": summary,
    }
    return _json_ready(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=None, help="independent trajectories per LR (default: 512; quick: 64)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--quick", action="store_true", help="smoke-test grids only; not assignment-quality")
    args = parser.parse_args(argv)
    samples = args.samples if args.samples is not None else (64 if args.quick else DEFAULT_SAMPLES)
    if not args.quick and samples < 512:
        parser.error("--samples must be at least 512 unless --quick is used")
    report = generate(samples=samples, seed=args.seed, quick=args.quick)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
