"""A3 Batch 2 — P2 river-valley + P3 edge-of-stability (parallel tracks).

Independent of Batch 1. Wraps official starters; does not invent grids.

    uv run python -m experiments.a3_optimization.batches.batch2 --dry-run
    uv run python -m experiments.a3_optimization.batches.batch2 --execute
    uv run python -m experiments.a3_optimization.launch_batch2 --execute

Use ``--only p2`` / ``--only p3`` to run one track. Completed volume runs are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.a3_optimization.helpers.metadata import membership, membership_tags
from experiments.a3_optimization.helpers.paths import MANIFEST_DIR
from experiments.a3_optimization.helpers.results import (
    classify_train_configs,
    format_status_lines,
    pending_configs,
)


BATCH = "batch2"


def _p2_runs():
    from experiments.a3_optimization.p2_river_valley import RUNS

    return list(RUNS)


def _p3_runs():
    from experiments.a3_optimization.p3_edge_of_stability import RUNS

    return list(RUNS)


def build_plan():
    jobs = []
    for config in _p2_runs():
        record = membership(
            problem="2",
            subpart="a",
            family="river_valley",
            config_role=str(config.lr_schedule),
            batch=BATCH,
        )
        jobs.append(
            {
                "label": f"a3-b2-p2-{config.lr_schedule}",
                "track": "p2",
                "kind": "lm",
                "config": config,
                "memberships": [record],
                "tags": membership_tags(record),
            }
        )
    for index, config in enumerate(_p3_runs()):
        subpart = "a" if getattr(config, "optimizer_name", "") == "sgd" else "b"
        record = membership(
            problem="3",
            subpart=subpart,
            family="edge_of_stability",
            config_role="starter",
            batch=BATCH,
        )
        jobs.append(
            {
                "label": f"a3-b2-p3-{index}-{config.optimizer_name}",
                "track": "p3",
                "kind": "lm",
                "config": config,
                "memberships": [record],
                "tags": membership_tags(record),
            }
        )
    return {
        "batch": BATCH,
        "jobs": jobs,
        "notes": [
            "P2 and P3 are independent; launch both when GPU budget allows.",
            "Starter RUNS are the required core; expand p3 exploration in the starter before Batch 2 execute if aiming for ~22 runs.",
            "Counterfactuals stay out of this required manifest (Batch 3 gated).",
        ],
    }


def write_manifest(plan, directory=None):
    directory = Path(directory or MANIFEST_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    public = [
        {
            "label": job["label"],
            "track": job["track"],
            "kind": job["kind"],
            "tags": job["tags"],
            "memberships": job["memberships"],
        }
        for job in plan["jobs"]
    ]
    payload = {"batch": BATCH, "jobs": public, "notes": plan["notes"]}
    json_path = directory / "batch2_manifest.json"
    csv_path = directory / "batch2_manifest.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    fields = ("label", "track", "problem", "subpart", "family", "config_role")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for job in public:
            primary = job["memberships"][0]
            writer.writerow(
                {
                    "label": job["label"],
                    "track": job["track"],
                    "problem": primary["problem"],
                    "subpart": primary["subpart"],
                    "family": primary["family"],
                    "config_role": primary["config_role"],
                }
            )
    return json_path, csv_path


def format_operator_summary(plan, statuses=None) -> str:
    p2 = sum(1 for job in plan["jobs"] if job["track"] == "p2")
    p3 = sum(1 for job in plan["jobs"] if job["track"] == "p3")
    lines = [
        "A3 Batch 2 (P2 + P3)",
        "====================",
        "",
        "Prerequisites: Batch 1 not required (independent)",
        f"P2 jobs: {p2}",
        f"P3 jobs: {p3} (expand starter for full ~22-run exploration)",
        "",
    ]
    for note in plan["notes"]:
        lines.append(f"- {note}")
    if statuses:
        lines.append("")
        lines.extend(format_status_lines("Train configs", statuses))
    return "\n".join(lines)


def run_batch2(
    *,
    dry_run: bool = False,
    max_parallel: int = 2,
    only: str | None = None,
    manifest_dir: Path | None = None,
) -> int:
    plan = build_plan()
    if only in {"p2", "p3"}:
        plan = {**plan, "jobs": [job for job in plan["jobs"] if job["track"] == only]}
    paths = write_manifest(plan, directory=manifest_dir)
    configs = [job["config"] for job in plan["jobs"]]
    statuses = None
    try:
        statuses = classify_train_configs(configs)
    except Exception as exc:
        print(f"Volume completion check unavailable: {type(exc).__name__}: {exc}")
        if not dry_run:
            print("Refusing to launch without a completion check.")
            return 1
    print(format_operator_summary(plan, statuses))
    print()
    print(f"Wrote {paths[0]}")
    print(f"Wrote {paths[1]}")
    pending = pending_configs(configs, statuses) if statuses is not None else list(configs)
    if dry_run:
        print(f"Dry run only. {len(pending)} job(s) would launch.")
        return 0
    if not pending:
        print("Nothing to submit.")
        return 0
    from modal_train import launch_training_jobs

    launch_training_jobs(pending, max_parallel_runs=max_parallel)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--only", choices=("p2", "p3"), help="Launch a single track.")
    args = parser.parse_args(argv)
    if args.execute and args.dry_run:
        parser.error("pass only one of --dry-run and --execute")
    if args.max_parallel < 1:
        parser.error("--max-parallel must be positive")
    return run_batch2(
        dry_run=args.dry_run,
        max_parallel=args.max_parallel,
        only=args.only,
        manifest_dir=args.manifest_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
