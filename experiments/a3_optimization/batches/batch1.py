"""A3 Batch 1 — P1 mode connectivity only (sequential).

Wraps ``p1_mode_connectivity``; does not redefine the (x, y) grid.

    uv run python -m experiments.a3_optimization.batches.batch1 --dry-run
    uv run python -m experiments.a3_optimization.batches.batch1 --execute
    uv run python -m experiments.a3_optimization.launch_batch1 --execute

Stages: prefix → wait → branches → wait → measure.
Resume: ``--stage {prefix|branches|measure}``; completed volume runs are skipped
by the shared Modal launcher / prefix checkpoint check (same as ad hoc launches).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from experiments.a3_optimization.helpers.metadata import membership, membership_tags
from experiments.a3_optimization.helpers.paths import MANIFEST_DIR
from experiments.a3_optimization.helpers.results import (
    classify_train_configs,
    format_status_lines,
    pending_configs,
)


BATCH = "batch1"
STAGES = ("prefix", "branches", "measure")


def _p1():
    from experiments.a3_optimization import p1_mode_connectivity as p1

    return p1


def build_plan():
    p1 = _p1()
    branches = [run for runs in p1.BRANCHES.values() for run in runs]
    jobs = [
        {
            "label": "a3-b1-p1-prefix",
            "stage": "prefix",
            "kind": "p1_stage",
            "count": 1,
            "memberships": [
                membership(
                    problem="1",
                    subpart="a",
                    family="mode_connectivity",
                    config_role="prefix",
                    batch=BATCH,
                    stage="prefix",
                )
            ],
            "config": p1.PREFIX,
        },
        {
            "label": "a3-b1-p1-branches",
            "stage": "branches",
            "kind": "p1_stage",
            "count": len(branches),
            "memberships": [
                membership(
                    problem="1",
                    subpart="a",
                    family="mode_connectivity",
                    config_role="branch",
                    batch=BATCH,
                    stage="branches",
                )
            ],
            "configs": branches,
        },
        {
            "label": "a3-b1-p1-measure",
            "stage": "measure",
            "kind": "p1_stage",
            "count": len(p1.pairs()),
            "memberships": [
                membership(
                    problem="1",
                    subpart="b",
                    family="mode_connectivity",
                    config_role="measure",
                    batch=BATCH,
                    stage="measure",
                )
            ],
            "results_path": str(p1.RESULTS_PATH),
        },
    ]
    for job in jobs:
        tags = []
        for record in job["memberships"]:
            tags.extend(membership_tags(record))
        job["tags"] = list(dict.fromkeys(tags))
    return {
        "batch": BATCH,
        "problem": "1",
        "jobs": jobs,
        "notes": [
            "Batch 1 is P1 only. P2+P3 are Batch 2; P4 (+ exploratory) is Batch 3.",
            "Stages are sequential: prefix → branches → measure.",
            "Ad hoc official launches fold in: skip completed, leave in-flight alone.",
            "P1(d) counterfactuals are Batch 3 (gated), not Batch 1.",
        ],
    }


def write_manifest(plan, directory=None):
    directory = Path(directory or MANIFEST_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    public = []
    for job in plan["jobs"]:
        public.append(
            {
                "label": job["label"],
                "stage": job["stage"],
                "kind": job["kind"],
                "count": job["count"],
                "tags": job["tags"],
                "memberships": job["memberships"],
                "results_path": job.get("results_path"),
            }
        )
    payload = {"batch": BATCH, "problem": "1", "jobs": public, "notes": plan["notes"]}
    json_path = directory / "batch1_manifest.json"
    csv_path = directory / "batch1_manifest.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    fields = ("label", "stage", "kind", "problem", "subpart", "config_role", "count")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for job in public:
            primary = job["memberships"][0]
            writer.writerow(
                {
                    "label": job["label"],
                    "stage": job["stage"],
                    "kind": job["kind"],
                    "problem": primary["problem"],
                    "subpart": primary["subpart"],
                    "config_role": primary["config_role"],
                    "count": job["count"],
                }
            )
    return json_path, csv_path


def format_operator_summary(plan, statuses=None) -> str:
    lines = [
        "A3 Batch 1 (P1 only)",
        "====================",
        "",
        "Prerequisites: Modal + course volume",
        "Contents: P1 prefix → branches → measure",
        "Not included: P2, P3, P4, P1(d), P5",
        "",
    ]
    for note in plan["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    if statuses:
        lines.extend(format_status_lines("Train configs (prefix+branches)", statuses))
        lines.append("")
    for job in plan["jobs"]:
        lines.append(f"{job['label']:<28} stage={job['stage']:<10} n={job['count']}")
    return "\n".join(lines)


def _run_official_stage(stage: str) -> None:
    p1 = _p1()
    old = sys.argv
    try:
        sys.argv = [old[0], stage]
        p1.main()
    finally:
        sys.argv = old


def run_batch1(
    *,
    dry_run: bool = False,
    max_parallel: int = 2,
    stage: str | None = None,
    manifest_dir: Path | None = None,
) -> int:
    del max_parallel  # official p1 launcher sets parallelism; kept for A2 CLI parity
    plan = build_plan()
    paths = write_manifest(plan, directory=manifest_dir)
    train_configs = [plan["jobs"][0]["config"], *plan["jobs"][1]["configs"]]
    statuses = None
    try:
        statuses = classify_train_configs(train_configs)
    except Exception as exc:
        print(f"Volume completion check unavailable: {type(exc).__name__}: {exc}")
    print(format_operator_summary(plan, statuses))
    print()
    print(f"Wrote {paths[0]}")
    print(f"Wrote {paths[1]}")
    print()

    stages = STAGES if stage is None else (stage,)
    if dry_run:
        for name in stages:
            job = next(item for item in plan["jobs"] if item["stage"] == name)
            if name == "prefix":
                print(f"[dry-run] P1 prefix keep_checkpoint_steps={job['config'].keep_checkpoint_steps}")
            elif name == "branches":
                pending = (
                    pending_configs(job["configs"], statuses)
                    if statuses is not None
                    else job["configs"]
                )
                print(f"[dry-run] P1 branches: {len(pending)}/{job['count']} would launch")
            else:
                print(f"[dry-run] P1 measure: {job['count']} cells → {job['results_path']}")
        print("Dry run only. Re-run with --execute (or omit --dry-run) to submit.")
        return 0

    # Execute one stage at a time so the operator waits (official README discipline).
    for name in stages:
        print(f"--- P1 {name} ---")
        _run_official_stage(name)
        if stage is None and name != "measure":
            nxt = "branches" if name == "prefix" else "measure"
            print(
                f"Finished submit/check for P1 {name}. "
                f"After it completes on Modal, run:\n"
                f"  uv run python -m experiments.a3_optimization.launch_batch1 --stage {nxt}"
            )
            return 0
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit stages (default when --dry-run is omitted).",
    )
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--stage", choices=STAGES, help="Resume a single P1 stage.")
    args = parser.parse_args(argv)
    if args.execute and args.dry_run:
        parser.error("pass only one of --dry-run and --execute")
    if args.max_parallel < 1:
        parser.error("--max-parallel must be positive")
    return run_batch1(
        dry_run=args.dry_run,
        max_parallel=args.max_parallel,
        stage=args.stage,
        manifest_dir=args.manifest_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
