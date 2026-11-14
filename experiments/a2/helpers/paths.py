"""Result locations for A2 figures and saved plot data.

Authoritative store lives under ``experiments/a2/results/`` (not a top-level
``outputs/`` folder). Hand-in PDFs live under ``experiments/a2/deliverables/``.
"""

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_DIR = PACKAGE_ROOT / "manifests"
OUTPUT_ROOT = PACKAGE_ROOT / "results"
PLOT_ROOT = OUTPUT_ROOT / "plots"
DATA_ROOT = OUTPUT_ROOT / "data"
NQM_RESULTS = DATA_ROOT / "p31" / "nqm_results.json"
STRESS_RESULTS = DATA_ROOT / "stress"
DIAGNOSTIC_RESULTS = DATA_ROOT / "diagnostics"


def plot_directory(*parts: str) -> Path:
    path = PLOT_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
