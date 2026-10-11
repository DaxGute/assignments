"""Light result / counterfactual bookkeeping for A3 (no fabricated measurements)."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.a3_optimization.helpers.paths import data_directory, problem_directory


class ResultsError(RuntimeError):
    """Required completed results are missing or ambiguous."""


def counterfactual_path(problem: str, name: str) -> Path:
    """JSON path for a pre-registered counterfactual under pN/counterfactuals/."""
    root = problem_directory(problem) / "counterfactuals"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{name}.json"


def write_counterfactual(problem: str, name: str, payload: dict) -> Path:
    """Write a counterfactual record. Caller must set status predicted before launch."""
    if "status" not in payload:
        payload = {**payload, "status": "predicted"}
    if payload.get("observed") is None:
        payload.setdefault("observed", None)
    path = counterfactual_path(problem, name)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_counterfactuals(problem: str) -> list[dict]:
    root = problem_directory(problem) / "counterfactuals"
    if not root.exists():
        return []
    rows = []
    for path in sorted(root.glob("*.json")):
        rows.append(json.loads(path.read_text()))
    return rows


def save_json(problem: str, name: str, payload: dict) -> Path:
    path = data_directory(problem) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    local = problem_directory(problem) / "results" / f"{name}.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(json.dumps(payload, indent=2) + "\n")
    return path
