"""Batch 1 — required baselines that do not depend on prior A3 measurements.

Mirrors ``experiments.a2.batches.batch1``: dry-run writes manifests; execute
submits via Modal (when credentials exist). Official starters stay the source of
truth for TrainConfig construction.

Stages in this batch (order matters for P1)::

    uv run python -m experiments.a3_optimization.batches.batch1 --dry-run
    uv run python -m experiments.a3_optimization.batches.batch1 --stage prefix
    # after prefix checkpoints exist on the volume:
    uv run python -m experiments.a3_optimization.batches.batch1 --stage branches
    uv run python -m experiments.a3_optimization.batches.batch1 --stage p2a
    uv run python -m experiments.a3_optimization.batches.batch1 --stage p3ab

Default ``--stage all`` launches independent LM jobs only (P2a schedules + P3
a-i/b). P1 prefix/branches remain explicit stages so Mac operators do not
accidentally fork before the prefix finishes.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from train import TrainConfig, training_run_name

from experiments.a3_optimization import p1_mode_connectivity as p1
from experiments.a3_optimization import p2_river_valley as p2
from experiments.a3_optimization import p3_edge_of_stability as p3
from experiments.a3_optimization.helpers.metadata import membership_tags
from experiments.a3_optimization.helpers.paths import MANIFEST_DIR, ensure_layout


BATCH = "batch1"
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


def _lm_job(label, config, memberships, *, kind="lm", stage="train"):
    tagged = _tag_config(config, memberships)
    primary = memberships[0]
    return {
        "label": label,
        "aliases": [label, training_run_name(tagged)],
        "kind": kind,
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
        },
        "hyperparameters": {
            "peak_lr": tagged.learning_rate,
            "weight_decay": tagged.weight_decay,
            "batch_size": tagged.batch_size,
            "lr_schedule": tagged.lr_schedule,
            "token_budget": tagged.num_train_sequences * 1024,
            "optimizer": tagged.optimizer_name,
            "stop_at_step": tagged.stop_at_step,
            "fork_from_step": tagged.fork_from_step,
            "batch_order_seed": tagged.batch_order_seed,
            "ema_decays": list(tagged.ema_decays),
        },
    }


def build_batch1():
    """Return the Batch 1 plan. Does not launch jobs or invent measurements."""
    ensure_layout()
    jobs = []

    # P1 prefix (must finish before branches).
    jobs.append(
        _lm_job(
            "p1-prefix",
            p1.PREFIX,
            [
                _membership(
                    "1", "a", "mode_connectivity", "prefix", stage="prefix"
                )
            ],
            stage="prefix",
        )
    )

    # P1 branches (one job per (x, order_seed)); stage gated.
    for x, runs in p1.BRANCHES.items():
        for config, seed in zip(runs, p1.BRANCH_ORDER_SEEDS):
            jobs.append(
                _lm_job(
                    f"p1-branch-x{x}-order{seed}",
                    config,
                    [
                        _membership(
                            "1",
                            "a",
                            "mode_connectivity",
                            "branch",
                            stage="branches",
                        )
                    ],
                    stage="branches",
                )
            )

    # P2 (a)(b) schedule + EMA baselines.
    for config in p2.SCHEDULE_RUNS:
        jobs.append(
            _lm_job(
                f"p2-schedule-{config.lr_schedule}",
                config,
                [
                    _membership(
                        "2", "a", "river_valley", "schedule", stage="p2a"
                    ),
                    _membership(
                        "2", "b", "river_valley", "ema", stage="p2a"
                    ),
                ],
                stage="p2a",
            )
        )

    # P3 (a)i clean full-batch + (b) minibatch base.
    for config in p3.FULL_BATCH_RUNS:
        jobs.append(
            _lm_job(
                "p3-fullbatch-clean",
                config,
                [
                    _membership(
                        "3", "a", "edge_of_stability", "full_batch", stage="p3ab"
                    )
                ],
                stage="p3ab",
            )
        )
    for config in p3.MINIBATCH_RUNS:
        jobs.append(
            _lm_job(
                "p3-minibatch-base",
                config,
                [
                    _membership(
                        "3", "b", "edge_of_stability", "minibatch", stage="p3ab"
                    )
                ],
                stage="p3ab",
            )
        )

    return {
        "assignment": ASSIGNMENT,
        "batch": BATCH,
        "jobs": jobs,
        "notes": [
            "Wraps official p1_mode_connectivity / p2_river_valley / p3_edge_of_stability.",
            "P1 branches require prefix checkpoints on the Modal volume.",
            "No counterfactual / open-ended exploration jobs in Batch 1.",
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
        "fits": plan["fits"],
        "references": plan["references"],
        "jobs": [_public_job(job) for job in plan["jobs"]],
    }
    json_path = directory / "batch1_manifest.json"
    csv_path = directory / "batch1_manifest.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    fields = (
        "label",
        "kind",
        "stage",
        "problem",
        "subpart",
        "family",
        "config_role",
        "peak_lr",
        "lr_schedule",
        "batch_size",
        "tokens",
        "optimizer",
        "stop_at_step",
        "fork_from_step",
        "batch_order_seed",
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
                    "kind": job["kind"],
                    "stage": job["stage"],
                    "problem": primary["problem"],
                    "subpart": primary["subpart"],
                    "family": primary["family"],
                    "config_role": primary["config_role"],
                    "peak_lr": hp.get("peak_lr"),
                    "lr_schedule": hp.get("lr_schedule"),
                    "batch_size": hp.get("batch_size"),
                    "tokens": hp.get("token_budget"),
                    "optimizer": hp.get("optimizer"),
                    "stop_at_step": hp.get("stop_at_step"),
                    "fork_from_step": hp.get("fork_from_step"),
                    "batch_order_seed": hp.get("batch_order_seed"),
                    "wandb_name": job["wandb_name"],
                }
            )
    return json_path, csv_path


def jobs_for_stage(plan, stage: str):
    if stage == "all":
        # Independent LM jobs only — never auto-start P1 forks.
        return [job for job in plan["jobs"] if job["stage"] in {"p2a", "p3ab"}]
    return [job for job in plan["jobs"] if job["stage"] == stage]


def launch_stage(plan, stage: str, *, max_parallel=2):
    """Submit TrainConfigs for one stage. P1 prefix uses the official check helper for branches."""
    from modal_train import launch_training_jobs

    selected = jobs_for_stage(plan, stage)
    if not selected:
        print(f"No jobs for stage={stage!r}.")
        return []
    if stage == "branches":
        p1.check_prefix_checkpoints()
    configs = [job["config"] for job in selected]
    print(f"Launching {len(configs)} jobs for stage={stage!r}")
    return launch_training_jobs(configs, max_parallel_runs=max_parallel)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit jobs (default when --dry-run is omitted).",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "prefix", "branches", "p2a", "p3ab"),
        default="all",
        help="all = p2a+p3ab only; use prefix/branches explicitly for P1.",
    )
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    args = parser.parse_args(argv)
    if args.execute and args.dry_run:
        parser.error("pass only one of --dry-run and --execute")
    plan = build_batch1()
    paths = write_manifest(plan, args.manifest_dir)
    execute = not args.dry_run
    selected = jobs_for_stage(plan, args.stage)
    print(f"Batch 1: {len(plan['jobs'])} jobs in plan; {len(selected)} for stage={args.stage!r}")
    for job in selected:
        hp = job["hyperparameters"]
        print(
            f"  {job['label']}  stage={job['stage']}  "
            f"sched={hp['lr_schedule']}  lr={hp['peak_lr']}  "
            f"stop={hp['stop_at_step']}  fork={hp['fork_from_step']}"
        )
    for path in paths:
        print(f"Wrote {path}")
    if not execute:
        print("Dry run only. Re-run without --dry-run (and with Modal auth) to submit.")
        return 0
    launch_stage(plan, args.stage, max_parallel=args.max_parallel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
