"""Light completion helpers for A3 batch launchers.

Official starters already skip finished Modal volume outputs for many stages.
These helpers only classify TrainConfig jobs the same way A2 Batch 1 does.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def classify_train_configs(configs: Sequence) -> dict[str, str]:
    """Map training_run_name → completed|missing using the Modal volume.

    Does not query W&B. In-flight / failed distinction needs W&B (optional later).
    """
    from modal_train import _completed_model_exists
    from train import training_run_name

    statuses = {}
    for config in configs:
        name = training_run_name(config)
        exists, _ = _completed_model_exists(config)
        statuses[name] = "completed" if exists else "missing"
    return statuses


def pending_configs(configs: Sequence, statuses: dict[str, str] | None = None) -> list:
    from train import training_run_name

    if statuses is None:
        statuses = classify_train_configs(configs)
    return [config for config in configs if statuses.get(training_run_name(config)) != "completed"]


def format_status_lines(label: str, statuses: dict[str, str]) -> list[str]:
    values = list(statuses.values())
    return [
        label,
        f"  Already completed: {values.count('completed')}",
        f"  New / missing: {values.count('missing')}",
        f"  Total named jobs: {len(values)}",
    ]


def flatten(groups: Iterable[Sequence]) -> list:
    out = []
    for group in groups:
        out.extend(group)
    return out
