"""Output locations for A3 figures, measurement dumps, and per-problem folders."""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_DIR = PACKAGE_ROOT / "manifests"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "a3"
PLOT_ROOT = OUTPUT_ROOT / "plots"
DATA_ROOT = OUTPUT_ROOT / "data"

# Per-problem trees under the package (counterfactuals + local results/plots).
PROBLEM_DIRS = {
    f"p{n}": PACKAGE_ROOT / f"p{n}" for n in (1, 2, 3, 4, 5)
}


def plot_directory(*parts: str) -> Path:
    path = PLOT_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_directory(*parts: str) -> Path:
    path = DATA_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def problem_directory(problem: str) -> Path:
    key = problem if problem.startswith("p") else f"p{problem}"
    path = PROBLEM_DIRS[key]
    for sub in ("counterfactuals", "results", "plots"):
        (path / sub).mkdir(parents=True, exist_ok=True)
    return path


def ensure_layout() -> dict[str, Path]:
    """Create manifests/, outputs/a3/..., and p1–p5/{counterfactuals,results,plots}."""
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    created = {"manifests": MANIFEST_DIR, "plots": PLOT_ROOT, "data": DATA_ROOT}
    for key in PROBLEM_DIRS:
        created[key] = problem_directory(key)
    return created
