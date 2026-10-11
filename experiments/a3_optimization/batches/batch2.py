"""Batch 2 — follow-ons and counterfactual slots (after Batch 1 / P1 prefix).

Includes P2 rewarm + horizon-free schedules, P3 mid-run η steps, and registered
P1(d)/P2/P3 counterfactual *slots* (configs attached only after ledger
pre-registration). Dry-run writes manifests; does not invent measured losses.

    uv run python -m experiments.a3_optimization.batches.batch2 --dry-run
    uv run python -m experiments.a3_optimization.batches.batch2 --stage p2cd
    uv run python -m experiments.a3_optimization.batches.batch2 --stage p3aii
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from train import TrainConfig, training_run_name

from experiments.a3_optimization import p2_river_valley as p2
from experiments.a3_optimization import p3_edge_of_stability as p3
from experiments.a3_optimization.helpers.metadata import membership_tags
from experiments.a3_optimization.helpers.paths import MANIFEST_DIR, ensure_layout
from experiments.a3_optimization.helpers.results import write_counterfactual


BATCH = "batch2"
ASSIGNMENT = "a3"


def _membership(problem, subpart, family, config_role, stage=None, hypothesis=None):
    row = {
        "assignment": ASSIGNMENT,
        "batch": BATCH,
        "problem": problem,
        "subpart": subpart,
        "family": family,
        "config_role": config_role,
    }
    if stage:
        row["stage"] = stage
    if hypothesis:
        row["hypothesis"] = hypothesis
    return row


def _tag_config(config: TrainConfig, memberships: list[dict]) -> TrainConfig:
    tags = list(config.wandb_tags)
    for membership in memberships:
        for tag in membership_tags(membership, batch=BATCH):
            if tag not in tags:
                tags.append(tag)
    return replace(config, wandb_tags=tuple(tags))


def _lm_job(label, config, memberships, *, stage="train"):
    tagged = _tag_config(config, memberships)
    primary = memberships[0]
    return {
        "label": label,
        "aliases": [label, training_run_name(tagged)],
        "kind": "lm",
        "stage": stage,
        "config": tagged,
        "tags": list(tagged.wandb_tags),
        "memberships": memberships,
        "metadata": {
            "assignment": ASSIGNMENT,
            "batch": BATCH,
            "problem": primary["problem"],
            "subpart": primary["subpart"],
            "family": primary["family"],
            "config_role": primary["config_role"],
            "stage": primary.get("stage") or stage,
            "hypothesis": primary.get("hypothesis"),
        },
        "hyperparameters": {
            "peak_lr": tagged.learning_rate,
            "weight_decay": tagged.weight_decay,
            "batch_size": tagged.batch_size,
            "lr_schedule": tagged.lr_schedule,
            "token_budget": tagged.num_train_sequences * 1024,
            "optimizer": tagged.optimizer_name,
            "lr_dip": tagged.lr_dip,
            "warmup_steps": tagged.warmup_steps,
            "min_lr_ratio": tagged.min_lr_ratio,
            "ema_decays": list(tagged.ema_decays),
        },
    }


def register_default_counterfactual_slots():
    """Pre-register empty counterfactual JSON under each problem (predicted only)."""
    slots = [
        (
            "p1",
            "p1d-lr-ablation",
            {
                "id": "p1d-lr-ablation",
                "problem": "1",
                "subpart": "d",
                "change": "Vary peak LR vs baseline cell; measure barrier B",
                "predicted": None,
                "observed": None,
                "status": "predicted",
                "note": "Fill predicted before launching; attach TrainConfig in batch2 COUNTERFACTUAL_RUNS.",
            },
        ),
        (
            "p2",
            "p2c-dip-deeper",
            {
                "id": "p2c-dip-deeper",
                "problem": "2",
                "subpart": "c",
                "change": "Deeper/longer lr_dip than starter (0.4,0.45,0.5,0.1)",
                "predicted": None,
                "observed": None,
                "status": "predicted",
            },
        ),
        (
            "p3",
            "p3c-batch-or-lr",
            {
                "id": "p3c-batch-or-lr",
                "problem": "3",
                "subpart": "c",
                "change": "Open hparam probe affecting edge entry (batch or LR)",
                "predicted": None,
                "observed": None,
                "status": "predicted",
            },
        ),
    ]
    paths = []
    for problem, name, payload in slots:
        paths.append(write_counterfactual(problem, name, payload))
    return paths


# Attach concrete TrainConfigs here only after ledger pre-registration.
COUNTERFACTUAL_RUNS: list[tuple[str, TrainConfig, list[dict]]] = []


def build_batch2():
    ensure_layout()
    cf_paths = register_default_counterfactual_slots()
    jobs = []

    for config in p2.REWARM_RUNS:
        jobs.append(
            _lm_job(
                "p2-rewarm-dip",
                config,
                [
                    _membership(
                        "2", "c", "river_valley", "lr_dip", stage="p2cd"
                    )
                ],
                stage="p2cd",
            )
        )
    for config in p2.HORIZON_FREE_RUNS:
        jobs.append(
            _lm_job(
                f"p2-horizon-{config.lr_schedule}",
                config,
                [
                    _membership(
                        "2", "d", "river_valley", "horizon_free", stage="p2cd"
                    )
                ],
                stage="p2cd",
            )
        )

    # P3 (a)ii mid-run LR changes (handout TODO on starter).
    for sched, label in (
        ("step750x0.5", "p3-fullbatch-half-lr"),
        ("step750x2", "p3-fullbatch-double-lr"),
    ):
        config = p3.full_batch(32, 1500, learning_rate=0.01, lr_schedule=sched)
        jobs.append(
            _lm_job(
                label,
                config,
                [
                    _membership(
                        "3",
                        "a",
                        "edge_of_stability",
                        "full_batch_step",
                        stage="p3aii",
                        hypothesis=sched,
                    )
                ],
                stage="p3aii",
            )
        )

    for label, config, memberships in COUNTERFACTUAL_RUNS:
        jobs.append(_lm_job(label, config, memberships, stage="counterfactual"))

    return {
        "assignment": ASSIGNMENT,
        "batch": BATCH,
        "jobs": jobs,
        "counterfactual_slots": [str(path) for path in cf_paths],
        "notes": [
            "Depends on Batch 1 for schedule/EMA reference comparisons.",
            "COUNTERFACTUAL_RUNS starts empty — fill after prediction ledger pre-registration.",
            "P1 measure stays in Batch 3 (needs all branch checkpoints).",
        ],
        "references": {},
        "fits": {},
    }


def _public_job(job):
    return {
        "label": job["label"],
        "aliases": job["aliases"],
        "wandb_name": training_run_name(job["config"]),
        "kind": job["kind"],
        "stage": job["stage"],
        "tags": job["tags"],
        "metadata": job["metadata"],
        "memberships": job["memberships"],
        "hyperparameters": job["hyperparameters"],
    }


def write_manifest(plan, directory=MANIFEST_DIR):
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "assignment": plan["assignment"],
        "batch": plan["batch"],
        "notes": plan["notes"],
        "counterfactual_slots": plan["counterfactual_slots"],
        "fits": plan["fits"],
        "references": plan["references"],
        "jobs": [_public_job(job) for job in plan["jobs"]],
    }
    json_path = directory / "batch2_manifest.json"
    csv_path = directory / "batch2_manifest.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    fields = (
        "label",
        "stage",
        "problem",
        "subpart",
        "family",
        "config_role",
        "hypothesis",
        "peak_lr",
        "lr_schedule",
        "wandb_name",
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for job in payload["jobs"]:
            primary = job["memberships"][0]
            hp = job["hyperparameters"]
            writer.writerow(
                {
                    "label": job["label"],
                    "stage": job["stage"],
                    "problem": primary["problem"],
                    "subpart": primary["subpart"],
                    "family": primary["family"],
                    "config_role": primary["config_role"],
                    "hypothesis": primary.get("hypothesis") or "",
                    "peak_lr": hp.get("peak_lr"),
                    "lr_schedule": hp.get("lr_schedule"),
                    "wandb_name": job["wandb_name"],
                }
            )
    return json_path, csv_path


def jobs_for_stage(plan, stage: str):
    if stage == "all":
        return [job for job in plan["jobs"] if job["stage"] != "counterfactual"]
    return [job for job in plan["jobs"] if job["stage"] == stage]


def launch_stage(plan, stage: str, *, max_parallel=2):
    from modal_train import launch_training_jobs

    selected = jobs_for_stage(plan, stage)
    if not selected:
        print(f"No jobs for stage={stage!r}.")
        return []
    configs = [job["config"] for job in selected]
    print(f"Launching {len(configs)} jobs for stage={stage!r}")
    return launch_training_jobs(configs, max_parallel_runs=max_parallel)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("all", "p2cd", "p3aii", "counterfactual"),
        default="all",
    )
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    args = parser.parse_args(argv)
    if args.execute and args.dry_run:
        parser.error("pass only one of --dry-run and --execute")
    plan = build_batch2()
    paths = write_manifest(plan, args.manifest_dir)
    execute = not args.dry_run
    selected = jobs_for_stage(plan, args.stage)
    print(f"Batch 2: {len(plan['jobs'])} jobs; {len(selected)} for stage={args.stage!r}")
    for job in selected:
        print(f"  {job['label']}  stage={job['stage']}  sched={job['hyperparameters']['lr_schedule']}")
    for slot in plan["counterfactual_slots"]:
        print(f"Counterfactual slot: {slot}")
    for path in paths:
        print(f"Wrote {path}")
    if not execute:
        print("Dry run only.")
        return 0
    launch_stage(plan, args.stage, max_parallel=args.max_parallel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
