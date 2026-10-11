"""A3 Batch 3 — P4 Hessian (+ gated exploratory).

Required core wraps ``p4_inside_the_hessian`` stages. Counterfactual / P1(d) / P5
jobs are listed only under ``exploratory`` after ledger pre-registration and are
never mixed unmarked into the required manifest.

    uv run python -m experiments.a3_optimization.batches.batch3 --dry-run
    uv run python -m experiments.a3_optimization.batches.batch3 --execute --stage a
    uv run python -m experiments.a3_optimization.launch_batch3 --stage fetch
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from experiments.a3_optimization.helpers.metadata import membership, membership_tags
from experiments.a3_optimization.helpers.paths import MANIFEST_DIR


BATCH = "batch3"
REQUIRED_STAGES = ("a", "b-rescale", "b-train", "b-measure", "c", "fetch")


def _p4():
    from experiments.a3_optimization import p4_inside_the_hessian as p4

    return p4


def build_plan(*, include_exploratory: bool = False):
    p4 = _p4()
    jobs = []
    for stage, stage_jobs in p4.STAGES.items():
        for item in stage_jobs:
            record = membership(
                problem="4",
                subpart=stage[0],
                family="inside_hessian",
                config_role=stage,
                batch=BATCH,
                stage=stage,
            )
            jobs.append(
                {
                    "label": f"a3-b3-p4-{item['name']}",
                    "stage": stage,
                    "kind": "hessian",
                    "required": True,
                    "hessian_job": item,
                    "memberships": [record],
                    "tags": membership_tags(record),
                }
            )
    for config in p4.RUNS:
        record = membership(
            problem="4",
            subpart="b",
            family="inside_hessian",
            config_role="b-train",
            batch=BATCH,
            stage="b-train",
        )
        jobs.append(
            {
                "label": f"a3-b3-p4-train-{config.lr_schedule}",
                "stage": "b-train",
                "kind": "lm",
                "required": True,
                "config": config,
                "memberships": [record],
                "tags": membership_tags(record),
            }
        )
    exploratory = []
    if include_exploratory:
        # Placeholder section only — real CF jobs are appended after ledger rows exist.
        exploratory.append(
            {
                "label": "a3-b3-exploratory-placeholder",
                "stage": "exploratory",
                "kind": "gated",
                "required": False,
                "note": (
                    "Add P1(d)/P5/counterfactual jobs here only after "
                    "docs/a3-prediction-ledger.md rows are pre-registered."
                ),
                "memberships": [
                    membership(
                        problem="5",
                        subpart="a",
                        family="exploratory",
                        config_role="gated",
                        batch=BATCH,
                        stage="exploratory",
                    )
                ],
                "tags": ["a3", "batch3", "counterfactual", "gated"],
            }
        )
    return {
        "batch": BATCH,
        "jobs": jobs,
        "exploratory": exploratory,
        "notes": [
            "Required: P4 stages a → fetch, then b-*, then c (see official starter).",
            "Needs shared volume hard-dl-dclm-v1 for provided checkpoints.",
            "Exploratory/counterfactual jobs require --include-exploratory and ledger pre-reg.",
        ],
    }


def write_manifest(plan, directory=None):
    directory = Path(directory or MANIFEST_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    public_jobs = []
    for job in plan["jobs"] + plan.get("exploratory", []):
        public_jobs.append(
            {
                "label": job["label"],
                "stage": job["stage"],
                "kind": job["kind"],
                "required": job.get("required", False),
                "tags": job.get("tags", []),
                "memberships": job.get("memberships", []),
                "hessian_name": None
                if job.get("hessian_job") is None
                else job["hessian_job"]["name"],
                "note": job.get("note"),
            }
        )
    payload = {
        "batch": BATCH,
        "jobs": [job for job in public_jobs if job["required"]],
        "exploratory": [job for job in public_jobs if not job["required"]],
        "notes": plan["notes"],
    }
    json_path = directory / "batch3_manifest.json"
    csv_path = directory / "batch3_manifest.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    fields = ("label", "stage", "kind", "required", "problem", "subpart", "hessian_name")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for job in public_jobs:
            primary = (job.get("memberships") or [{"problem": "", "subpart": ""}])[0]
            writer.writerow(
                {
                    "label": job["label"],
                    "stage": job["stage"],
                    "kind": job["kind"],
                    "required": job["required"],
                    "problem": primary.get("problem", ""),
                    "subpart": primary.get("subpart", ""),
                    "hessian_name": job.get("hessian_name") or "",
                }
            )
    return json_path, csv_path


def format_operator_summary(plan) -> str:
    required = len(plan["jobs"])
    exploratory = len(plan.get("exploratory") or [])
    lines = [
        "A3 Batch 3 (P4 + gated exploratory)",
        "===================================",
        "",
        "Prerequisites: shared Hessian volume; prefer Batch 1–2 baselines before CFs",
        f"Required P4 jobs listed: {required}",
        f"Exploratory (gated) listed: {exploratory}",
        "",
    ]
    for note in plan["notes"]:
        lines.append(f"- {note}")
    by_stage = {}
    for job in plan["jobs"]:
        by_stage[job["stage"]] = by_stage.get(job["stage"], 0) + 1
    lines.append("")
    for stage in REQUIRED_STAGES:
        if stage in by_stage or stage == "fetch":
            lines.append(f"  {stage}: {by_stage.get(stage, 0)} jobs (fetch downloads results)")
    return "\n".join(lines)


def _run_official_stage(stage: str, *, local: bool = False) -> None:
    p4 = _p4()
    old = sys.argv
    argv = [old[0], stage]
    if local:
        argv.append("--local")
    try:
        sys.argv = argv
        p4.main()
    finally:
        sys.argv = old


def run_batch3(
    *,
    dry_run: bool = False,
    stage: str = "a",
    local: bool = False,
    include_exploratory: bool = False,
    manifest_dir: Path | None = None,
) -> int:
    plan = build_plan(include_exploratory=include_exploratory)
    paths = write_manifest(plan, directory=manifest_dir)
    print(format_operator_summary(plan))
    print()
    print(f"Wrote {paths[0]}")
    print(f"Wrote {paths[1]}")
    print()
    if dry_run:
        print(f"[dry-run] Would run official P4 stage {stage!r} (local={local}).")
        if include_exploratory:
            print("[dry-run] Exploratory section present but not auto-launched.")
        print("Dry run only.")
        return 0
    if stage == "exploratory":
        print(
            "Exploratory jobs are gated. Add concrete configs after ledger "
            "pre-registration; this wrapper does not invent CF launches."
        )
        return 1
    print(f"--- P4 {stage} ---")
    _run_official_stage(stage, local=local)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--stage",
        choices=(*REQUIRED_STAGES, "exploratory"),
        default="a",
        help="Official P4 stage (default: a).",
    )
    parser.add_argument("--local", action="store_true", help="Pass --local to P4 measure stages.")
    parser.add_argument(
        "--include-exploratory",
        action="store_true",
        help="List gated exploratory placeholder in the manifest (does not launch CFs).",
    )
    parser.add_argument("--manifest-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.execute and args.dry_run:
        parser.error("pass only one of --dry-run and --execute")
    return run_batch3(
        dry_run=args.dry_run,
        stage=args.stage,
        local=args.local,
        include_exploratory=args.include_exploratory,
        manifest_dir=args.manifest_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
