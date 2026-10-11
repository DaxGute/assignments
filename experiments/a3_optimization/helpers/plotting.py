"""Minimal plotting helpers. Problem-specific figures live under plots/ or outputs/a3/plots/."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from experiments.a3_optimization.helpers.paths import plot_directory


class MissingData(RuntimeError):
    """A figure cannot be drawn because required completed results are absent."""


@dataclass
class PlotReport:
    name: str
    written: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def figure_path(problem: str, name: str, suffix: str = "png") -> Path:
    return plot_directory(problem) / f"{name}.{suffix}"
