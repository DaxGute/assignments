from dataclasses import dataclass

import numpy as np
import wandb

from utils import WANDB_ENTITY, WANDB_PROJECT

PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"


@dataclass
class Curve:
    run: object
    steps: np.ndarray
    losses: np.ndarray


def load_curves(filters=None, limit=None) -> list[Curve]:
    query = {"state": "finished", **(filters or {})}
    runs = wandb.Api().runs(PROJECT_PATH, filters=query, order="-created_at")
    curves = []
    for run in runs:
        rows = list(run.scan_history(keys=["optimizer_step", "train_loss"]))
        points = [
            (row.get("optimizer_step"), row.get("train_loss"))
            for row in rows
            if row.get("optimizer_step") is not None
            and row.get("train_loss") is not None
            and np.isfinite(row["train_loss"])
        ]
        if len(points) < 20:
            continue
        steps, losses = zip(*points, strict=True)
        curves.append(
            Curve(
                run=run,
                steps=np.asarray(steps, dtype=float),
                losses=np.asarray(losses, dtype=float),
            )
        )
        if limit is not None and len(curves) >= limit:
            break
    return curves


def rolling_mean(values: np.ndarray, window=21) -> np.ndarray:
    window = min(window, len(values))
    if window < 2:
        return values.copy()
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def smoothness(curve: Curve) -> float:
    log_losses = np.log(np.maximum(curve.losses, 1e-12))
    residual = log_losses - rolling_mean(log_losses)
    return float(1.4826 * np.median(np.abs(residual - np.median(residual))))


def macro_deviation(curve: Curve) -> float:
    progress = (curve.steps - curve.steps.min()) / max(np.ptp(curve.steps), 1.0)
    log_losses = np.log(np.maximum(rolling_mean(curve.losses), 1e-12))
    fit = np.polyval(np.polyfit(progress, log_losses, 2), progress)
    return float(np.sqrt(np.mean((log_losses - fit) ** 2)))


def config_value(run, key, default=None):
    if key == "learning_rate":
        return run.config.get("learning_rate", run.config.get("optim_lr", default))
    return run.config.get(key, default)
