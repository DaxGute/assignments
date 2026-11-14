"""Result / plot locations for A3, kept inside the assignment package.

Hand-in PDFs live under ``experiments/a3_optimization/deliverables/``.
Measured JSON and analysis plots live under ``experiments/a3_optimization/results/``.
Do not write a top-level repo ``outputs/`` tree.
"""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_DIR = PACKAGE_ROOT / "manifests"
PACKAGE_PLOTS = PACKAGE_ROOT / "plots"
STARTER_RESULTS = PACKAGE_ROOT / "results"
# Legacy name kept for callers; now package-local under results/.
OUTPUT_ROOT = STARTER_RESULTS


def problem_root(problem: str) -> Path:
    """``results/p1`` … ``results/p5`` (accepts ``1``, ``p1``, ``P1``)."""
    text = _problem_id(problem)
    path = OUTPUT_ROOT / f"p{text}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def plot_directory(problem: str, *parts: str, counterfactual: bool = False) -> Path:
    """Plots under ``experiments/a3_optimization/plots/pN/``.

    Also ensures a mirror under ``results/plots/pN/`` for analysis dumps.
    Counterfactuals nest under the problem.
    """
    text = _problem_id(problem)
    package = PACKAGE_PLOTS / f"p{text}"
    mirrored = OUTPUT_ROOT / "plots" / f"p{text}"
    if counterfactual:
        package = package / "counterfactual"
        mirrored = mirrored / "counterfactual"
    if parts:
        package = package.joinpath(*parts)
        mirrored = mirrored.joinpath(*parts)
    package.mkdir(parents=True, exist_ok=True)
    mirrored.mkdir(parents=True, exist_ok=True)
    return package


def data_directory(problem: str, *parts: str, counterfactual: bool = False) -> Path:
    path = problem_root(problem) / "data"
    if counterfactual:
        path = problem_root(problem) / "counterfactual" / "data"
    if parts:
        path = path.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def counterfactual_directory(problem: str, *parts: str) -> Path:
    """Counterfactuals nest under the problem (`…/counterfactual/`), not a parallel tree."""
    text = _problem_id(problem)
    path = PACKAGE_PLOTS / f"p{text}" / "counterfactual"
    mirrored = OUTPUT_ROOT / "plots" / f"p{text}" / "counterfactual"
    if parts:
        path = path.joinpath(*parts)
        mirrored = mirrored.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    mirrored.mkdir(parents=True, exist_ok=True)
    return path


def _problem_id(problem: str) -> str:
    text = str(problem).strip().lower()
    if text.startswith("p"):
        text = text[1:]
    if text not in {"1", "2", "3", "4", "5"}:
        raise ValueError(f"problem must be 1–5, got {problem!r}")
    return text
