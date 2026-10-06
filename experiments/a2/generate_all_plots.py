"""Generate every required Assignment 2 figure from completed experiments.

Problem-specific modules own their data validation and figures. This entry point
runs them in assignment order and writes one machine-readable artifact index.
"""

from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.a2.helpers.paths import OUTPUT_ROOT, PLOT_ROOT


MODULES = (
    ("p1-p2", "experiments.a2.plots_p12"),
    ("p3.1", "experiments.a2.plots_p31"),
    ("p3.2", "experiments.a2.plots_p32"),
    ("p4", "experiments.a2.plots_p4"),
)


def _paths_under(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(str(path.relative_to(OUTPUT_ROOT)) for path in root.rglob("*") if path.is_file())


def generate(*, samples: int = 512, seed: int = 42) -> dict:
    reports = {}
    for label, module_name in MODULES:
        module = importlib.import_module(module_name)
        if label == "p3.1":
            report = module.generate(samples=samples, seed=seed)
        else:
            report = module.generate()
        reports[label] = report

    from experiments.a2.create_intuition_sheet import generate as generate_intuition_sheet

    intuition_pdf = generate_intuition_sheet()
    index = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "nqm_samples": samples,
        "nqm_seed": seed,
        "reports": reports,
        "artifacts": _paths_under(PLOT_ROOT),
        "intuition_sheet": str(intuition_pdf),
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / "artifact_index.json"
    path.write_text(json.dumps(index, indent=2, default=str) + "\n")
    return index


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.samples < 512:
        parser.error("--samples must be at least 512 for assignment figures")
    index = generate(samples=args.samples, seed=args.seed)
    print(f"Wrote {len(index['artifacts'])} plot artifacts beneath {PLOT_ROOT}")
    print(f"Wrote {OUTPUT_ROOT / 'artifact_index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
