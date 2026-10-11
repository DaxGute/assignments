"""Batch 3 — P1 measure, P4 Hessian stages, optional P5 slot.

    uv run python -m experiments.a3_optimization.batches.batch3 --dry-run
    uv run python -m experiments.a3_optimization.batches.batch3 --stage p1-measure
    uv run python -m experiments.a3_optimization.batches.batch3 --stage p4-a
    uv run python -m experiments.a3_optimization.batches.batch3 --stage p4-fetch
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.a3_optimization import p1_mode_connectivity as p1
from experiments.a3_optimization import p4_inside_the_hessian as p4
from experiments.a3_optimization.helpers.paths import MANIFEST_DIR, ensure_layout
from experiments.a3_optimization.helpers.results import write_counterfactual


BATCH = "batch3"
ASSIGNMENT = "a3"


def register_p4_p5_slots():
    return [
        write_counterfactual(
            "p4",
            "p4b-sharpen-without-loss",
            {
                "id": "p4b-sharpen-without-loss",
                "problem": "4",
                "subpart": "b",
                "change": "Move final lambda_max by >=2x; keep val loss in [2.925, 2.931]",
                "predicted": None,
                "observed": None,
                "status": "predicted",
            },
        ),
        write_counterfactual(
            "p5",
            "p5-custom-prediction",
            {
                "id": "p5-custom-prediction",
                "problem": "5",
                "subpart": "a",
                "change": "Optional staff-quiz-style prediction problem",
                "predicted": None,
                "observed": None,
                "status": "predicted",
                "note": "Not required for core grade; design after P1–P4 intuition.",
            },
        ),
    ]


def build_batch3():
    ensure_layout()
    slots = register_p4_p5_slots()
    jobs = [
        {
            "label": "p1-measure",
            "kind": "measure",
            "stage": "p1-measure",
            "module": "experiments.a3_optimization.p1_mode_connectivity",
            "action": "measure",
            "n_pairs": len(p1.pairs()),
            "results_path": str(p1.RESULTS_PATH),
            "memberships": [
                {
                    "assignment": ASSIGNMENT,
                    "batch": BATCH,
                    "problem": "1",
                    "subpart": "b",
                    "family": "mode_connectivity",
                    "config_role": "measure",
                    "stage": "p1-measure",
                }
            ],
            "hyperparameters": {},
        },
        {
            "label": "p4-a-look",
            "kind": "hessian",
            "stage": "p4-a",
            "module": "experiments.a3_optimization.p4_inside_the_hessian",
            "action": "a",
            "n_jobs": len(p4.LOOK_JOBS),
            "memberships": [
                {
                    "assignment": ASSIGNMENT,
                    "batch": BATCH,
                    "problem": "4",
                    "subpart": "a",
                    "family": "hessian",
                    "config_role": "look",
                    "stage": "p4-a",
                }
            ],
            "hyperparameters": {"checkpoints": list(p4.CHECKPOINTS)},
            "note": "Reads course volume hard-dl-dclm-v1 a3_hessian runs.",
        },
        {
            "label": "p4-c-subspace",
            "kind": "hessian",
            "stage": "p4-c",
            "module": "experiments.a3_optimization.p4_inside_the_hessian",
            "action": "c",
            "n_jobs": len(p4.SUBSPACE_JOBS),
            "memberships": [
                {
                    "assignment": ASSIGNMENT,
                    "batch": BATCH,
                    "problem": "4",
                    "subpart": "c",
                    "family": "hessian",
                    "config_role": "subspace",
                    "stage": "p4-c",
                }
            ],
            "hyperparameters": {"arms": ["full", "top", "removed"]},
        },
        {
            "label": "p4-fetch",
            "kind": "fetch",
            "stage": "p4-fetch",
            "module": "experiments.a3_optimization.p4_inside_the_hessian",
            "action": "fetch",
            "memberships": [
                {
                    "assignment": ASSIGNMENT,
                    "batch": BATCH,
                    "problem": "4",
                    "subpart": "a",
                    "family": "hessian",
                    "config_role": "fetch",
                    "stage": "p4-fetch",
                }
            ],
            "hyperparameters": {},
        },
    ]
    return {
        "assignment": ASSIGNMENT,
        "batch": BATCH,
        "jobs": jobs,
        "counterfactual_slots": [str(path) for path in slots],
        "notes": [
            "P1 measure requires all Batch 1 branch checkpoints.",
            "P4 needs Modal access to hard-dl-dclm-v1.",
            "P4(b) train/rescale jobs are student-filled on the official starter; not auto-listed until configured.",
        ],
        "references": {},
        "fits": {},
    }


def write_manifest(plan, directory=MANIFEST_DIR):
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "assignment": plan["assignment"],
        "batch": plan["batch"],
        "notes": plan["notes"],
        "counterfactual_slots": plan["counterfactual_slots"],
        "jobs": plan["jobs"],
        "fits": plan["fits"],
        "references": plan["references"],
    }
    json_path = directory / "batch3_manifest.json"
    csv_path = directory / "batch3_manifest.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    fields = ("label", "kind", "stage", "problem", "subpart", "action", "module")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for job in plan["jobs"]:
            primary = job["memberships"][0]
            writer.writerow(
                {
                    "label": job["label"],
                    "kind": job["kind"],
                    "stage": job["stage"],
                    "problem": primary["problem"],
                    "subpart": primary["subpart"],
                    "action": job.get("action", ""),
                    "module": job.get("module", ""),
                }
            )
    return json_path, csv_path


def launch_stage(plan, stage: str):
    selected = [job for job in plan["jobs"] if job["stage"] == stage]
    if not selected:
        print(f"No jobs for stage={stage!r}.")
        return None
    job = selected[0]
    if stage == "p1-measure":
        from experiments.a3_optimization.connectivity import measure_pairs_on_modal

        rows = measure_pairs_on_modal(p1.pairs())
        p1.RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        p1.RESULTS_PATH.write_text(json.dumps(rows, indent=2) + "\n")
        # Mirror into package p1/results and outputs/a3/data/p1
        from experiments.a3_optimization.helpers.results import save_json

        save_json("p1", "mode-connectivity", {"n_cells": len(rows), "rows": rows})
        print(f"Wrote {len(rows)} cells to {p1.RESULTS_PATH}")
        return rows
    if stage.startswith("p4"):
        from experiments.a3_optimization.hessian_jobs import fetch, launch
        from experiments.a3_optimization.p4_inside_the_hessian import STAGES

        action = job["action"]
        if action == "fetch":
            names = [j["name"] for jobs in STAGES.values() for j in jobs]
            print(fetch(names))
            return names
        return launch(STAGES[action], local=False)
    raise ValueError(f"Unhandled stage {stage!r}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("all", "p1-measure", "p4-a", "p4-c", "p4-fetch"),
        default="all",
    )
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    args = parser.parse_args(argv)
    if args.execute and args.dry_run:
        parser.error("pass only one of --dry-run and --execute")
    plan = build_batch3()
    paths = write_manifest(plan, args.manifest_dir)
    execute = not args.dry_run
    print(f"Batch 3: {len(plan['jobs'])} jobs")
    for job in plan["jobs"]:
        print(f"  {job['label']}  stage={job['stage']}  kind={job['kind']}")
    for slot in plan["counterfactual_slots"]:
        print(f"Counterfactual slot: {slot}")
    for path in paths:
        print(f"Wrote {path}")
    if not execute:
        print("Dry run only.")
        return 0
    if args.stage == "all":
        print("Refusing --stage all on execute; pick p1-measure / p4-a / p4-c / p4-fetch.")
        return 1
    launch_stage(plan, args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
