"""Submit one Batch 1 plan. Imported only when batch1 is run with --execute."""

from __future__ import annotations

from pathlib import PurePosixPath

from data import DEFAULT_DATASET_DIR_NAME
from modal_utils import (
    MODAL_ENVIRONMENT,
    MODAL_SHARED_DATASETS_DIR,
    VOLUME_MOUNTS,
    app,
    build_image,
    secrets,
    timestamped_modal_app_name,
    user_volume,
)


STRESS_DIR = "/a2-stress"
STRESS_TIMEOUT_SECONDS = 4 * 60 * 60


@app.function(
    image=build_image(),
    volumes=VOLUME_MOUNTS,
    gpu="H100",
    retries=0,
    max_containers=2,
    timeout=STRESS_TIMEOUT_SECONDS,
)
def _run_stress(payload):
    """One five-step job. Depth 1000 keeps microbatch 8; rerun that job alone if it OOMs."""
    from pathlib import Path

    from experiments.a2.policies import initialize_stress, stress_groups
    from experiments.a2.stress import StressConfig, load_tokens, run

    label = payload["label"]
    output = Path("/root/data") / STRESS_DIR.strip("/") / f"{label}.json"
    if output.exists():
        print(f"Skipping existing stress diagnostics {output}")
        return {"label": label, "skipped": True, "output": str(output)}
    spec = payload["stress"]
    config = StressConfig(
        width=spec["width"],
        depth=spec["depth"],
        head_dim=spec["head_dim"],
        microbatch=spec["microbatch"],
        seed=spec["seed"],
        precision=spec["precision"],
        steps=spec["steps"],
        batch=spec["batch_size"],
    )
    policy = {
        "policy": spec["policy"],
        "reference_width": spec["reference_width"],
        "reference_depth": spec["reference_depth"],
    }

    def initialize(model):
        initialize_stress(model, **policy)

    def groups(model, base_lr):
        return stress_groups(model, base_lr, **policy)

    data = PurePosixPath(MODAL_SHARED_DATASETS_DIR) / DEFAULT_DATASET_DIR_NAME
    train = load_tokens(data / "train", config.steps * config.batch, config.context)
    val = load_tokens(data / "val", config.batch, config.context)
    try:
        result = run(
            config,
            train,
            val,
            base_lr=spec["base_lr"],
            initialize_fn=initialize,
            groups_fn=groups,
            output=output,
        )
        import wandb
        from experiments.a2.wandb_diagnostics import log_records
        from utils import WANDB_ENTITY, WANDB_PROJECT

        with wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
            name=label,
            tags=list(payload["tags"]),
            config={
                **payload["metadata"],
                "parameter_groups": result["parameter_groups"],
                "data_sha256": result["data_sha256"],
                "output_multiplier": result["output_multiplier"],
                "residual_multipliers": result["residual_multipliers"],
            },
        ) as run_handle:
            wandb.define_metric("optimizer_step")
            wandb.define_metric("*", step_metric="optimizer_step")
            for row in result["history"]:
                wandb.log(
                    {
                        "optimizer_step": row["step"],
                        **{
                            key: row[key]
                            for key in ("val_loss", "train_loss", "logit_rms")
                            if key in row
                        },
                    }
                )
            log_records(wandb, result["alignment"], result["history"])
            wandb.save(str(output), base_path=str(output.parent), policy="now")
            print("WANDB_RUN_URL=" + run_handle.url)
        return {"label": label, "skipped": False, "output": str(output)}
    finally:
        user_volume.commit()


def _existing_stress_labels():
    try:
        entries = user_volume.listdir(STRESS_DIR)
    except Exception as exc:
        print(f"Could not list existing stress outputs: {exc}")
        return set()
    labels = set()
    for entry in entries:
        path = getattr(entry, "path", entry)
        name = PurePosixPath(str(path)).name
        if name.endswith(".json"):
            labels.add(name[: -len(".json")])
    return labels


def launch_stress(jobs, max_parallel):
    import modal

    existing = _existing_stress_labels()
    pending = []
    for job in jobs:
        if job["label"] in existing or any(alias in existing for alias in job["aliases"]):
            print(f"Skipping existing stress run {job['label']}")
            continue
        pending.append(
            {
                "label": job["label"],
                "tags": list(job["tags"]),
                "metadata": job["metadata"],
                "stress": {
                    key: job["stress"][key]
                    for key in (
                        "policy",
                        "width",
                        "depth",
                        "base_lr",
                        "precision",
                        "reference_width",
                        "reference_depth",
                        "seed",
                        "steps",
                        "batch_size",
                        "microbatch",
                        "head_dim",
                    )
                },
            }
        )
    if not pending:
        print("No five-step jobs to submit.")
        return []
    results = []
    with modal.enable_output():
        with app.run(
            name=timestamped_modal_app_name("a2-batch1-stress"),
            detach=True,
            environment_name=MODAL_ENVIRONMENT,
        ):
            remote = _run_stress.with_options(
                max_containers=max_parallel,
                secrets=secrets(include_wandb=True),
            )
            for payload in pending:
                call = remote.spawn(payload)
                results.append({"label": payload["label"], "modal_call_id": call.object_id})
                print(f"Submitted stress {payload['label']} call_id={call.object_id}")
    return results


def launch(plan, max_parallel=2):
    """Language-model jobs first, then five-step jobs. Both apps are detached."""
    if max_parallel < 1:
        raise ValueError(f"max_parallel must be positive, got {max_parallel}.")
    language_model = [job["config"] for job in plan["jobs"] if job["kind"] == "lm"]
    stress = [job for job in plan["jobs"] if job["kind"] == "stress"]
    launched = {}
    if language_model:
        from experiments.a2.modal_launcher import launch_training_jobs

        launched["language_model"] = launch_training_jobs(
            language_model,
            max_parallel_runs=max_parallel,
        )
    else:
        print("No language-model jobs to submit.")
    if stress:
        launched["stress"] = launch_stress(stress, max_parallel)
    else:
        print("No five-step jobs in this plan.")
    return launched
