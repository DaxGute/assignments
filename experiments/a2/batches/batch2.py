"""Batch 2 of the CS312 A2 hyperparameter-scaling experiments.

The normal command reads finished Batch 1 runs, fits the targets, and submits
them. ``--dry-run`` does not submit jobs::

    uv run python -m experiments.a2.batches.batch2 --dry-run
    uv run python -m experiments.a2.batches.batch2

No learning rate or weight decay is typed in here. ``plan_batch2`` reads
measurements. Each measurement needs ``final_val_loss`` plus ``problem``,
``subpart``, ``hypothesis``, ``batch_size``, ``peak_lr``, ``weight_decay``,
``width``, ``parameterization``, and ``tokens``. Supplied CSV rows fill source
cells that were not retrained:

- Problem 3.2 at B=64, when the LR-WD pair is in the supplied sweeps
- the width-512 curve at 153.6M tokens and weight decay 0.1

That width-512 curve is the course baseline. At the reference width it is
also the standard-muP curve, so both Problem 4.2(c) prescriptions use it.

Local target factors, chosen here rather than in the handout:

- Problem 3.2(b) at B=128 and B=256: the power-law prediction times
  ``2 ** -0.5``, ``1``, and ``2 ** 0.5``
- Problem 4.2(c) at width 1024: the power-law prediction, the transferred
  Problem 1 sampled learning rate, and the prediction times ``2 ** k`` for
  ``k`` in ``{-1, 0, 1}``

Values that land on each other become one config.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from experiments.a2.batches.batch1 import (
    P32A_LEARNING_RATES,
    P32B_LEARNING_RATE,
    P32B_WEIGHT_DECAYS,
    P32_BATCHES,
    P32_TOKENS,
    P42_TOKENS,
    coalesce,
    fmt_float,
    same_float,
)
from experiments.a2.policies import long_scaling
from experiments.a2.provided_sweeps import load
from experiments.a2.scaling import power_law, predict_power_law, quadratic_log_optimum


P32_TARGET_BATCHES = (128, 256)
P42C_WIDTH = 1024
P42C_SOURCE_WIDTHS = (128, 256, 512)
P32_LOCAL_FACTORS = (2**-0.5, 1.0, 2**0.5)
P42C_LOCAL_EXPONENTS = (-1, 0, 1)
LOSS_ATOL = 1e-4
# Keep manifests in experiments/a2/manifests after this module moved into batches/.
MANIFEST_DIR = Path(__file__).resolve().parents[1] / "manifests"


class Batch2Error(ValueError):
    pass


def _not_ready(detail):
    raise Batch2Error(
        "Batch 2 cannot run yet:\n"
        "required Batch 1 source experiments are missing or incomplete.\n\n"
        f"{detail}"
    )


def _stop(detail):
    raise Batch2Error(f"Batch 2 cannot run:\n{detail}")


def _loss(row):
    value = row.get("final_val_loss", row.get("val_loss"))
    if value is None or not math.isfinite(float(value)):
        raise Batch2Error(f"measurement is missing a finite loss: {row.get('run_id') or row.get('label')}")
    return float(value)


def _match(row, **expected):
    for key, value in expected.items():
        actual = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if actual is None or not same_float(actual, value):
                return False
        elif actual != value:
            return False
    return True


_MEMBERSHIP_FIELDS = (
    "problem",
    "subpart",
    "hypothesis",
    "family",
    "config_role",
    "parameterization",
    "stage",
    "batch",
)


def _candidates(row):
    yield row
    for membership in row.get("memberships") or []:
        if not isinstance(membership, dict):
            continue
        expanded = dict(row)
        for key in _MEMBERSHIP_FIELDS:
            value = membership.get(key)
            if value not in (None, ""):
                expanded[key] = value
        yield expanded


def _match_any(row, **expected):
    return any(_match(candidate, **expected) for candidate in _candidates(row))


def _supplied_measurements():
    rows = []
    for part in ("P1a", "P1b", "P2a"):
        for row in load(part):
            rows.append(
                {
                    "problem": "supplied",
                    "subpart": part,
                    "hypothesis": None,
                    "batch_size": int(row["batch_size"]),
                    "peak_lr": float(row["learning_rate"]),
                    "weight_decay": float(row["weight_decay"]),
                    "width": 512,
                    "depth": 8,
                    "parameterization": "baseline",
                    "tokens": int(row["tokens"]),
                    "final_val_loss": float(row["final_val_loss"]),
                    "optimizer": row["optimizer"],
                    "lr_schedule": row["lr_schedule"],
                    "seed": int(row["seed"]),
                    "run_id": row["run_id"],
                }
            )
    return rows


def _load_supplied():
    try:
        return _supplied_measurements()
    except FileNotFoundError as exc:
        raise Batch2Error(
            "ERROR:\n"
            "The supplied sweep CSV is missing, so Batch 2 cannot read the "
            "B=64 source cells or the width-512 learning-rate curve.\n"
            f"Expected worksheets/hparam_invariants/data/provided_sweeps.csv.\n"
            f"{exc}"
        ) from exc


def _collect(measurements, *, description, **expected):
    found = [row for row in measurements if _match_any(row, **expected)]
    if not found:
        lines = "\n".join(f"{key}={value}" for key, value in expected.items())
        _not_ready(
            f"Expected source sweep for\n{description}\n{lines}\n"
            "but no completed source run was found."
        )
    losses = [_loss(row) for row in found]
    if max(losses) - min(losses) > LOSS_ATOL:
        identities = [row.get("run_id") or row.get("label") for row in found]
        _stop(
            f"Ambiguous source for {description}: {expected}\n"
            f"Finished runs disagree by more than {LOSS_ATOL}: "
            f"{list(zip(identities, losses))}"
        )
    return found


def _optimum(rows, value_key):
    ordered = sorted(rows, key=lambda row: float(row[value_key]))
    fit = quadratic_log_optimum(
        [float(row[value_key]) for row in ordered],
        [_loss(row) for row in ordered],
    )
    if fit["method"] != "log_quadratic_vertex":
        raise Batch2Error(
            "Batch 2 cannot run:\n"
            f"The {value_key} curve is not convex in log space "
            f"(fit method {fit['method']!r}). "
            "Refusing to substitute the best sampled value."
        )
    return fit


def _deduped_values(values):
    chosen = []
    for value in values:
        chosen.append(coalesce(float(value), chosen))
    # Preserve the first occurrence of each coalesced value.
    unique = []
    for value in chosen:
        if not any(same_float(value, previous) for previous in unique):
            unique.append(value)
    return unique


def _job(label, membership, **hyperparameters):
    return {
        "label": label,
        "membership": membership,
        "hyperparameters": hyperparameters,
        "scheduled": True,
    }


def _unique_by_value(rows, value_key):
    chosen = []
    seen = []
    for row in sorted(rows, key=lambda item: float(item[value_key])):
        if any(same_float(row[value_key], previous) for previous in seen):
            earlier = next(
                item for item in chosen if same_float(item[value_key], row[value_key])
            )
            if abs(_loss(earlier) - _loss(row)) > LOSS_ATOL:
                _stop(
                    f"Two finished runs share {value_key}={row[value_key]} "
                    f"but their losses differ by more than {LOSS_ATOL}: "
                    f"{_loss(earlier)} and {_loss(row)}."
                )
            continue
        seen.append(float(row[value_key]))
        chosen.append(row)
    return chosen


def _source_record(law, **fields):
    return {
        "exponent": float(law["exponent"]),
        "prefactor": float(law["prefactor"]),
        "r_squared": float(law["r_squared"]),
        **fields,
    }


def plan_p32b(measurements, supplied=None):
    """Predict LR and WD at B=128 and B=256 from the completed source sweeps."""
    if supplied is None:
        supplied = _load_supplied()
    pool = [
        row
        for row in list(measurements) + supplied
        if row.get("optimizer") in {None, "adamw"}
        and row.get("lr_schedule", "linear") == "linear"
        and row.get("seed", 42) == 42
    ]
    hypotheses = []
    for hypothesis, value_key, fixed_key, fixed_value, grid in (
        ("fixed_wd_scale_lr", "peak_lr", "weight_decay", 0.1, P32A_LEARNING_RATES),
        (
            "fixed_lr_scale_wd",
            "weight_decay",
            "peak_lr",
            P32B_LEARNING_RATE,
            P32B_WEIGHT_DECAYS,
        ),
    ):
        optima = []
        source = []
        for batch_size in P32_BATCHES:
            rows = []
            for value in grid:
                rows.extend(
                    _collect(
                        pool,
                        description="Problem 3.2 source cell",
                        batch_size=batch_size,
                        tokens=P32_TOKENS,
                        **{value_key: float(value), fixed_key: float(fixed_value)},
                    )[:1]
                )
            fit = _optimum(rows, value_key)
            optima.append(fit["optimum"])
            source.append({"batch_size": batch_size, "optimum": fit})
        law = power_law(P32_BATCHES, optima)
        jobs = []
        for batch_size in P32_TARGET_BATCHES:
            predicted = predict_power_law(law, batch_size)
            if not math.isfinite(predicted) or predicted <= 0:
                raise Batch2Error(
                    f"Problem 3.2 {hypothesis} prediction at B={batch_size} is {predicted}."
                )
            values = _deduped_values(predicted * factor for factor in P32_LOCAL_FACTORS)
            for value in values:
                fields = {
                    "batch_size": batch_size,
                    "tokens": P32_TOKENS,
                    "peak_lr": P32B_LEARNING_RATE if value_key == "weight_decay" else value,
                    "weight_decay": 0.1 if value_key == "peak_lr" else value,
                    "optimizer": "adamw",
                    "width": 512,
                    "depth": 8,
                }
                role = "predicted" if same_float(value, predicted) else "nearby_validation"
                jobs.append(
                    _job(
                        (
                            f"a2-b2-p32b-{hypothesis}-b{batch_size:03d}"
                            f"-lr{fmt_float(fields['peak_lr'])}"
                            f"-wd{fmt_float(fields['weight_decay'])}-{role}"
                        ),
                        {
                            "problem": "3.2",
                            "subpart": "b",
                            "family": "batch_scaling_target",
                            "config_role": role,
                            "hypothesis": hypothesis,
                            "stage": "target",
                            "batch": "batch2",
                            "source_batch": "batch1",
                            "parameterization": "baseline",
                            "prediction_source": _source_record(
                                law,
                                kind="batch_power_law",
                                hypothesis=hypothesis,
                                varied=value_key,
                                held_fixed={fixed_key: float(fixed_value)},
                                source_batches=list(P32_BATCHES),
                                source_optima=[float(item) for item in optima],
                                predicted=float(predicted),
                                trained_value=float(value),
                            ),
                        },
                        **fields,
                    )
                )
        hypotheses.append(
            {
                "hypothesis": hypothesis,
                "source_optima": source,
                "power_law": law,
                "jobs": jobs,
            }
        )
    return {"problem": "3.2", "subpart": "b", "hypotheses": hypotheses}


def _transferred_learning_rate():
    from experiments.a2.scaling import fit_batch1_sources

    try:
        record = fit_batch1_sources()["p1_1536m_best_sampled"]
    except FileNotFoundError as exc:
        raise Batch2Error(
            "ERROR:\n"
            "The supplied Problem 1 sweep is missing, so Batch 2 cannot recover "
            "the directly transferred learning rate.\n"
            f"{exc}"
        ) from exc
    return record


def plan_p42c(measurements, transferred=None, supplied=None):
    """Predict the width-1024 learning rate from widths 128, 256, and 512."""
    if transferred is None:
        transferred = _transferred_learning_rate()
    if supplied is None:
        supplied = _load_supplied()
    transferred_lr = float(transferred["learning_rate"])
    pool = list(measurements) + supplied
    prescriptions = []
    for parameterization in ("baseline", "mup"):
        optima = []
        source = []
        for width in P42C_SOURCE_WIDTHS:
            wanted = "baseline" if width == 512 else parameterization
            rows = [
                row
                for row in pool
                if _match_any(
                    row,
                    width=width,
                    depth=8,
                    tokens=P42_TOKENS,
                    batch_size=64,
                    parameterization=wanted,
                    optimizer="adamw",
                    lr_schedule="linear",
                    seed=42,
                    weight_decay=0.1,
                )
            ]
            if width == 512:
                rows = [
                    row
                    for row in rows
                    if row.get("problem") in {None, "supplied", "1", "4.2"}
                    or _match_any(row, problem="supplied")
                    or _match_any(row, problem="1")
                    or _match_any(row, problem="4.2")
                ]
            else:
                rows = [row for row in rows if _match_any(row, problem="4.2", subpart="a")]
            rows = _unique_by_value(rows, "peak_lr")
            if len(rows) < 3:
                _not_ready(
                    "Expected source sweep for\n"
                    "problem=4.2\n"
                    "subpart=a\n"
                    f"parameterization={wanted}\n"
                    f"width={width}\n"
                    f"tokens={P42_TOKENS}\n"
                    "weight_decay=0.1\n"
                    f"but found {len(rows)} completed learning-rate points; need at least 3."
                )
            fit = _optimum(rows, "peak_lr")
            optima.append(fit["optimum"])
            source.append({"width": width, "optimum": fit})
        law = power_law(P42C_SOURCE_WIDTHS, optima)
        predicted = predict_power_law(law, P42C_WIDTH)
        if not math.isfinite(predicted) or predicted <= 0:
            raise Batch2Error(f"width-1024 {parameterization} prediction is {predicted}.")
        candidates = [("powerlaw_prediction", predicted, "width_power_law")]
        candidates.append(("direct_transfer", transferred_lr, "problem1_best_sampled_lr"))
        for exponent in P42C_LOCAL_EXPONENTS:
            candidates.append(("local_sweep", predicted * 2.0**exponent, "width_power_law"))
        chosen = []
        jobs = []
        for role, learning_rate, kind in candidates:
            learning_rate = coalesce(learning_rate, chosen)
            chosen.append(learning_rate)
            scaling = long_scaling(parameterization, P42C_WIDTH, 8)
            provenance = _source_record(
                law,
                kind=kind,
                parameterization=parameterization,
                source_widths=list(P42C_SOURCE_WIDTHS),
                source_optima=[float(item) for item in optima],
                predicted=float(predicted),
                transferred_lr=transferred_lr,
                transferred_run_id=transferred.get("run_id"),
                trained_lr=float(learning_rate),
            )
            jobs.append(
                _job(
                    (
                        f"a2-b2-p42c-width-{parameterization}-w{P42C_WIDTH:04d}"
                        f"-lr{fmt_float(learning_rate)}-{role}"
                    ),
                    {
                        "problem": "4.2",
                        "subpart": "c",
                        "family": "heldout_width",
                        "config_role": role,
                        "parameterization": parameterization,
                        "stage": "target",
                        "batch": "batch2",
                        "source_batch": "batch1",
                        "prediction_source": provenance,
                    },
                    tokens=P42_TOKENS,
                    batch_size=64,
                    width=P42C_WIDTH,
                    depth=8,
                    peak_lr=learning_rate,
                    weight_decay=0.1,
                    optimizer="adamw",
                    parameterization=parameterization,
                    scaling=scaling,
                )
            )
        prescriptions.append(
            {
                "parameterization": parameterization,
                "source_optima": source,
                "power_law": law,
                "predicted_lr": predicted,
                "transferred_lr": transferred_lr,
                "transferred_run_id": transferred.get("run_id"),
                "jobs": jobs,
            }
        )
    return {"problem": "4.2", "subpart": "c", "prescriptions": prescriptions}


def plan_batch2(measurements, supplied=None):
    """Build the Batch 2 target plan. ``measurements`` come from finished Batch 1 runs."""
    if not isinstance(measurements, list):
        raise Batch2Error("measurements must be a list of result dicts")
    return {
        "batch": "batch2",
        "p32b": plan_p32b(measurements, supplied=supplied),
        "p42c": plan_p42c(measurements, supplied=supplied),
    }


def measurements_from_results(records):
    """Normalize manifest jobs or W&B summaries into the measurement dicts above.

    ``final_val_loss`` has to already be attached. This does not query W&B.
    """
    measurements = []
    for record in records:
        metadata = record.get("metadata", record)
        hyperparameters = record.get("hyperparameters", metadata)
        loss = record.get("final_val_loss", metadata.get("final_val_loss"))
        measurements.append(
            {
                "label": record.get("label", metadata.get("label")),
                "problem": metadata.get("problem"),
                "subpart": metadata.get("subpart"),
                "hypothesis": metadata.get("hypothesis"),
                "batch_size": hyperparameters.get("batch_size"),
                "peak_lr": hyperparameters.get("peak_lr"),
                "weight_decay": hyperparameters.get("weight_decay"),
                "width": hyperparameters.get("width"),
                "depth": hyperparameters.get("depth"),
                "parameterization": metadata.get("parameterization"),
                "tokens": hyperparameters.get("token_budget", hyperparameters.get("tokens")),
                "optimizer": hyperparameters.get("optimizer"),
                "lr_schedule": hyperparameters.get("lr_schedule", metadata.get("lr_schedule")),
                "seed": hyperparameters.get("seed", metadata.get("seed")),
                "beta1": hyperparameters.get("beta1", metadata.get("beta1")),
                "beta2": hyperparameters.get("beta2", metadata.get("beta2")),
                "run_id": record.get("run_id", metadata.get("run_id")),
                "final_val_loss": loss,
            }
        )
    return measurements


def _planned_jobs(plan):
    for hypothesis in plan["p32b"]["hypotheses"]:
        for job in hypothesis["jobs"]:
            yield job
    for prescription in plan["p42c"]["prescriptions"]:
        for job in prescription["jobs"]:
            yield job


def _processed_tokens(batch_size, token_budget, context=1024):
    sequences = int(token_budget) // context
    micro = min(int(batch_size), 64)
    num_micro = int(batch_size) // micro
    steps = (sequences // micro) // num_micro
    return steps * num_micro * micro * context


def schedule(plan):
    """Turn analytical targets into one training config per physical setup."""
    from dataclasses import replace

    from experiments.a2.batches.batch1 import Catalog, add_lm

    catalog = Catalog([])
    for planned in _planned_jobs(plan):
        hyperparameters = planned["hyperparameters"]
        membership = planned["membership"]
        parameterization = hyperparameters.get(
            "parameterization", membership.get("parameterization", "baseline")
        )
        add_lm(
            catalog,
            label=planned["label"],
            membership=membership,
            spec={
                "tokens": hyperparameters["tokens"],
                "batch_size": hyperparameters["batch_size"],
                "learning_rate": hyperparameters["peak_lr"],
                "weight_decay": hyperparameters["weight_decay"],
                "width": hyperparameters["width"],
                "depth": hyperparameters["depth"],
                "parameterization": parameterization,
                "optimizer_name": "adamw",
            },
        )
    jobs = []
    for job in catalog.scheduled():
        roles = []
        hypotheses = []
        sources = []
        for membership in job["memberships"]:
            role = membership.get("config_role")
            if role and role not in roles:
                roles.append(role)
            hypothesis = membership.get("hypothesis")
            if hypothesis and hypothesis not in hypotheses:
                hypotheses.append(hypothesis)
            source = membership.get("prediction_source")
            if source is not None:
                sources.append(source)
        metadata = dict(job["metadata"])
        metadata["assignment"] = "a2"
        metadata["batch"] = "batch2"
        metadata["source_batch"] = "batch1"
        metadata["config_roles"] = roles
        metadata["hypotheses"] = hypotheses
        metadata["prediction_sources"] = sources
        metadata["processed_tokens"] = _processed_tokens(
            metadata["batch_size"], metadata["token_budget"]
        )
        from train import training_model_name

        metadata["model_name"] = training_model_name(job["config"])
        tags = []
        for tag in job["tags"]:
            if tag == "batch1":
                tag = "batch2"
            if tag not in tags:
                tags.append(tag)
        if "batch2" not in tags:
            tags.append("batch2")
        if "source_batch1" not in tags:
            tags.append("source_batch1")
        job["tags"] = tags
        job["metadata"] = metadata
        job["config"] = replace(
            job["config"],
            save_model=False,
            wandb_tags=tuple(tags),
            experiment_metadata=metadata,
        )
        job["reused"] = None
        jobs.append(job)
    return jobs


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Batch2Error("refusing to write a non-finite fit value")
        return value
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return _json_ready(value.item())
    return value


def _public_job(job):
    from train import training_run_name

    hyperparameters = job["metadata"]
    return {
        "label": job["label"],
        "aliases": list(job["aliases"]),
        "wandb_name": training_run_name(job["config"]),
        "tags": list(job["tags"]),
        "metadata": _json_ready(job["metadata"]),
        "prediction_sources": _json_ready(job["metadata"].get("prediction_sources", [])),
        "reused_run_id": None if job.get("reused") is None else job["reused"].get("run_id"),
        "reused_run_url": None if job.get("reused") is None else job["reused"].get("run_url"),
        "peak_lr": hyperparameters.get("peak_lr"),
        "weight_decay": hyperparameters.get("weight_decay"),
        "batch_size": hyperparameters.get("batch_size"),
        "width": hyperparameters.get("width"),
        "depth": hyperparameters.get("depth"),
        "parameterization": hyperparameters.get("parameterization"),
        "token_budget": hyperparameters.get("token_budget"),
        "processed_tokens": hyperparameters.get("processed_tokens"),
    }


def write_manifest(plan, jobs, directory=MANIFEST_DIR):
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "batch": "batch2",
        "source_batch": "batch1",
        "fits": _json_ready({"p32b": plan["p32b"], "p42c": plan["p42c"]}),
        "jobs": [_public_job(job) for job in jobs],
        "discretionary": {
            "p32b_local_factors": list(P32_LOCAL_FACTORS),
            "p42c_local_exponents": list(P42C_LOCAL_EXPONENTS),
            "note": (
                "Problem 3.2(b) trains the prediction and half an octave on either side. "
                "Problem 4.2(c) trains the power-law prediction, the Problem 1 sampled "
                "learning rate, and one octave on either side of the prediction. "
                "Matching values are one physical run with every analytical role kept."
            ),
        },
    }
    json_path = directory / "batch2_manifest.json"
    csv_path = directory / "batch2_manifest.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    fields = (
        "label",
        "aliases",
        "problem",
        "subpart",
        "family",
        "config_role",
        "hypothesis",
        "parameterization",
        "batch_size",
        "width",
        "peak_lr",
        "weight_decay",
        "token_budget",
        "processed_tokens",
        "reused_run_id",
        "wandb_name",
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for job in jobs:
            metadata = job["metadata"]
            primary = job["memberships"][0]
            writer.writerow(
                {
                    "label": job["label"],
                    "aliases": "|".join(job["aliases"]),
                    "problem": primary["problem"],
                    "subpart": primary["subpart"],
                    "family": primary["family"],
                    "config_role": "|".join(metadata.get("config_roles") or []),
                    "hypothesis": "|".join(metadata.get("hypotheses") or []),
                    "parameterization": metadata.get("parameterization"),
                    "batch_size": metadata.get("batch_size"),
                    "width": metadata.get("width"),
                    "peak_lr": metadata.get("peak_lr"),
                    "weight_decay": metadata.get("weight_decay"),
                    "token_budget": metadata.get("token_budget"),
                    "processed_tokens": metadata.get("processed_tokens"),
                    "reused_run_id": "" if job.get("reused") is None else job["reused"].get("run_id"),
                    "wandb_name": training_run_name_of(job),
                }
            )
    return json_path, csv_path


def training_run_name_of(job):
    from train import training_run_name

    return training_run_name(job["config"])


def _fmt_law(law):
    return (
        f"exponent={law['exponent']:.8g}  prefactor={law['prefactor']:.8g}  "
        f"r^2={law['r_squared']:.8g}"
    )


def format_summary(plan, jobs, measurements):
    proposed = list(_planned_jobs(plan))
    reused = [job for job in jobs if job.get("reused")]
    fresh = [job for job in jobs if not job.get("reused")]
    lines = [
        "Batch 2",
        f"Measurements used: {len(measurements)}",
        f"Analytical targets: {len(proposed)}",
        f"Deduped training configs: {len(jobs)}",
        f"Already finished, will not relaunch: {len(reused)}",
        f"Unique GPU jobs to submit: {len(fresh)}",
        "",
        "Problem 3.2(b)",
    ]
    for hypothesis in plan["p32b"]["hypotheses"]:
        law = hypothesis["power_law"]
        lines.append(f"  {hypothesis['hypothesis']}  {_fmt_law(law)}")
        for item in hypothesis["source_optima"]:
            fit = item["optimum"]
            lines.append(
                f"    B={item['batch_size']} optimum={fit['optimum']:.8g} "
                f"method={fit['method']} inside_range={fit['vertex_inside_sample_range']}"
            )
        for batch_size in P32_TARGET_BATCHES:
            predicted = predict_power_law(law, batch_size)
            lines.append(f"    B={batch_size} predicted={predicted:.8g}")
    lines.extend(["", "Problem 4.2(c)"])
    for prescription in plan["p42c"]["prescriptions"]:
        law = prescription["power_law"]
        lines.append(f"  {prescription['parameterization']}  {_fmt_law(law)}")
        for item in prescription["source_optima"]:
            fit = item["optimum"]
            lines.append(
                f"    width={item['width']} optimum={fit['optimum']:.8g} "
                f"method={fit['method']}"
            )
        lines.append(
            f"    width={P42C_WIDTH} powerlaw_prediction={prescription['predicted_lr']:.8g}"
        )
        lines.append(
            f"    width={P42C_WIDTH} direct_transfer={prescription['transferred_lr']:.8g} "
            f"run={prescription.get('transferred_run_id')}"
        )
    lines.extend(["", "Deduped runs"])
    for job in jobs:
        metadata = job["metadata"]
        roles = ",".join(metadata.get("config_roles") or [])
        state = "reused" if job.get("reused") else "new"
        lines.append(
            f"  {job['label']}  {state}  roles={roles}  "
            f"lr={fmt_float(metadata['peak_lr'])}  wd={fmt_float(metadata['weight_decay'])}  "
            f"B={metadata['batch_size']}  width={metadata['width']}  "
            f"param={metadata.get('parameterization')}"
        )
        if job["aliases"]:
            lines.append(f"    aliases: {', '.join(job['aliases'])}")
    return "\n".join(lines)


def _number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _whole(value):
    number = _number(value)
    if number is None or not float(number).is_integer():
        return None
    return int(number)


def _first_present(sources, *keys):
    for source in sources:
        if not isinstance(source, dict):
            continue
        for key in keys:
            if key in source and source[key] not in (None, ""):
                return source[key]
    return None


def _run_loss(run):
    summary = dict(getattr(run, "summary", {}) or {})
    for key in ("final_val_loss", "val_loss"):
        number = _number(summary.get(key))
        if number is not None:
            return number
    last = None
    try:
        history = run.scan_history(keys=["val_loss", "final_val_loss"])
    except Exception:
        return None
    for row in history:
        for key in ("final_val_loss", "val_loss"):
            number = _number(row.get(key))
            if number is not None:
                last = number
    return last


def _sources_of(config):
    sources = []
    for key in ("experiment_metadata", "experiment", "metadata", "a2_metadata"):
        value = config.get(key)
        if isinstance(value, dict):
            sources.append(value)
    sources.append(config)
    return sources


def _is_batch1(config, tags):
    batch = _first_present(_sources_of(config), "batch")
    if batch in {1, "1", "batch1"}:
        return True
    return "batch1" in tags


def _measurement_from_run(run):
    config = dict(run.config or {})
    tags = set(run.tags or [])
    sources = _sources_of(config)
    protocol = _first_present(sources, "protocol")
    if protocol == "five_step":
        return None
    loss = _run_loss(run)
    peak_candidates = [
        _number(_first_present(sources, "peak_lr")),
        _number(_first_present(sources, "learning_rate", "optim_lr")),
    ]
    peaks = [value for value in peak_candidates if value is not None]
    if len(peaks) >= 2 and not same_float(peaks[0], peaks[1]):
        raise Batch2Error(
            "ERROR:\n"
            f"Run {run.id} logs peak_lr={peaks[0]} and learning_rate={peaks[1]}. "
            "Those values disagree, so the source cell is ambiguous."
        )
    peak_lr = peaks[0] if peaks else None
    weight_decay = _number(_first_present(sources, "weight_decay"))
    batch_size = _whole(_first_present(sources, "batch_size"))
    tokens = _whole(_first_present(sources, "token_budget", "tokens", "train_tokens"))
    width = _whole(_first_present(sources, "width"))
    if width is None:
        model_config = config.get("model_config")
        if isinstance(model_config, dict):
            width = _whole(model_config.get("hidden_size"))
    depth = _whole(_first_present(sources, "depth"))
    if depth is None and width == 512:
        model_config = config.get("model_config")
        if isinstance(model_config, dict):
            depth = _whole(model_config.get("num_hidden_layers"))
    parameterization = _first_present(sources, "parameterization")
    if isinstance(parameterization, list):
        parameterization = parameterization[0] if len(parameterization) == 1 else None
    record = {
        "run_id": run.id,
        "run_url": getattr(run, "url", None),
        "label": getattr(run, "name", None),
        "problem": _first_present(sources, "problem"),
        "subpart": _first_present(sources, "subpart"),
        "hypothesis": _first_present(sources, "hypothesis"),
        "batch_size": batch_size,
        "peak_lr": peak_lr,
        "weight_decay": weight_decay,
        "width": width,
        "depth": depth,
        "parameterization": parameterization,
        "tokens": tokens,
        "optimizer": _first_present(sources, "optimizer") or config.get("optimizer_name"),
        "lr_schedule": _first_present(sources, "lr_schedule") or config.get("lr_schedule"),
        "seed": _whole(_first_present(sources, "seed", "model_seed")) or config.get("model_seed"),
        "beta1": _number(_first_present(sources, "beta1")) or _number(config.get("beta1")),
        "beta2": _number(_first_present(sources, "beta2")) or _number(config.get("beta2")),
        "final_val_loss": loss,
        "is_batch1": _is_batch1(config, tags),
    }
    if record["peak_lr"] is None or record["batch_size"] is None or record["tokens"] is None:
        return None
    if record["final_val_loss"] is None:
        return None
    return record


def load_source_runs():
    """Read A2 runs from W&B.

    The first list is completed Batch 1 measurements used to fit Batch 2.
    The second list is every readable run, including ones still in progress,
    so a rerun can skip them.
    """
    from experiments.a2.helpers.results import ResultsError, query_a2_runs

    try:
        described = query_a2_runs()
    except ResultsError as exc:
        _not_ready(str(exc))
    inconsistent = [
        record
        for record in described
        if record.get("inconsistencies")
        and record.get("state_completed")
        and record.get("problem") in {"3.2", "4.2"}
    ]
    if inconsistent:
        lines = ["Finished runs have inconsistent metadata:"]
        for record in inconsistent:
            label = record.get("run_id") or record.get("name") or "(unnamed)"
            lines.append(f"  {label}: {'; '.join(record['inconsistencies'])}")
        _stop("\n".join(lines))
    measurements = []
    catalog = []
    for record in described:
        if record.get("protocol") == "five_step":
            continue
        if record.get("peak_lr") is None or record.get("batch_size") is None or record.get("tokens") is None:
            continue
        item = dict(record)
        item["is_batch1"] = record.get("batch") == "batch1"
        catalog.append(item)
        if not record.get("completed") or not item["is_batch1"]:
            continue
        if record.get("problem") not in {"3.2", "4.2", "1"}:
            continue
        if record.get("subpart") == "c":
            continue
        measurements.append(item)
    return measurements, catalog


def classify_launch(jobs, catalog):
    from experiments.a2.helpers.results import FAILED_STATES, FINISHED_STATES

    for job in jobs:
        matches = [record for record in catalog if _same_training(job, record)]
        completed = [record for record in matches if record.get("completed")]
        if completed:
            losses = [float(record["final_val_loss"]) for record in completed]
            if max(losses) - min(losses) > LOSS_ATOL:
                _stop(
                    f"Finished runs already exist for {job['label']} but their losses disagree: "
                    f"{[(record.get('run_id'), record.get('final_val_loss')) for record in completed]}"
                )
            completed.sort(key=lambda record: record.get("run_id") or "")
            job["reused"] = completed[0]
            job["launch_status"] = "completed"
            continue
        if any(
            record.get("state") and record.get("state") not in FINISHED_STATES | FAILED_STATES
            for record in matches
        ):
            job["reused"] = None
            job["launch_status"] = "running"
            continue
        if matches:
            job["reused"] = None
            job["launch_status"] = "failed"
            continue
        job["reused"] = None
        job["launch_status"] = "missing"
    return jobs


def _same_training(job, record):
    from experiments.a2.helpers.results import canonicalize_parameterization

    metadata = job["metadata"]
    parameterization = metadata.get("parameterization")
    if isinstance(parameterization, list):
        parameterization = parameterization[0] if len(parameterization) == 1 else None
    other = record.get("parameterization")
    if isinstance(other, list):
        other = other[0] if len(other) == 1 else None
    if canonicalize_parameterization(parameterization) != canonicalize_parameterization(other):
        return False
    try:
        return (
            int(metadata["width"]) == int(record["width"])
            and int(metadata["depth"]) == int(record["depth"])
            and int(metadata["batch_size"]) == int(record["batch_size"])
            and int(metadata["token_budget"]) == int(record["tokens"])
            and same_float(metadata["peak_lr"], record["peak_lr"])
            and same_float(metadata["weight_decay"], record["weight_decay"])
            and (metadata.get("optimizer") or "adamw") == (record.get("optimizer") or "adamw")
        )
    except (TypeError, ValueError):
        return False


def mark_finished(jobs, catalog):
    return classify_launch(jobs, catalog)


def get_best_p32_config(batch_size=256):
    """Best finished Problem 3.2 LR-WD pair at ``batch_size``.

    This reads the same W&B fields Batch 2 writes. The course model name is
    ``a2-d8``. Subpart c and any run that is not the 614.4M AdamW setup are
    excluded. The winner is the lowest final validation loss.
    """
    if batch_size not in {8, 16, 32, 64, 128, 256}:
        raise Batch2Error(f"batch_size must be a Problem 3.2 batch, got {batch_size}")
    _measurements, catalog = load_source_runs()
    from experiments.a2.helpers.results import ResultsError, get_best_run

    try:
        chosen = get_best_run(catalog, batch_size=int(batch_size))
    except ResultsError as exc:
        raise Batch2Error(str(exc)) from exc
    winner = chosen["winner"]
    return {
        "batch_size": int(batch_size),
        "peak_lr": float(winner["peak_lr"]),
        "weight_decay": float(winner["weight_decay"]),
        "final_val_loss": float(winner["final_val_loss"]),
        "run_id": winner.get("run_id"),
        "run_url": winner.get("run_url"),
        "token_budget": P32_TOKENS,
        "beta1": 0.9,
        "beta2": 0.95,
    }


def _parabola(value, center):
    return (math.log(float(value) / center) ** 2) + 2.0


def _synthetic_measurements():
    rows = []
    for batch_size in P32_BATCHES:
        for learning_rate in P32A_LEARNING_RATES:
            rows.append(
                {
                    "problem": "3.2",
                    "subpart": "a",
                    "batch_size": batch_size,
                    "tokens": P32_TOKENS,
                    "peak_lr": float(learning_rate),
                    "weight_decay": 0.1,
                    "width": 512,
                    "depth": 8,
                    "parameterization": "baseline",
                    "optimizer": "adamw",
                    "lr_schedule": "linear",
                    "seed": 42,
                    "final_val_loss": _parabola(learning_rate, 0.003),
                }
            )
        for weight_decay in P32B_WEIGHT_DECAYS:
            rows.append(
                {
                    "problem": "3.2",
                    "subpart": "b",
                    "batch_size": batch_size,
                    "tokens": P32_TOKENS,
                    "peak_lr": float(P32B_LEARNING_RATE),
                    "weight_decay": float(weight_decay),
                    "width": 512,
                    "depth": 8,
                    "parameterization": "baseline",
                    "optimizer": "adamw",
                    "lr_schedule": "linear",
                    "seed": 42,
                    "final_val_loss": _parabola(weight_decay, 0.2),
                }
            )
    for parameterization in ("baseline", "mup"):
        for width in (128, 256):
            for learning_rate in (0.001, 0.003, 0.006):
                rows.append(
                    {
                        "problem": "4.2",
                        "subpart": "a",
                        "batch_size": 64,
                        "tokens": P42_TOKENS,
                        "peak_lr": learning_rate,
                        "weight_decay": 0.1,
                        "width": width,
                        "depth": 8,
                        "parameterization": parameterization,
                        "optimizer": "adamw",
                        "lr_schedule": "linear",
                        "seed": 42,
                        "final_val_loss": _parabola(learning_rate, 0.003),
                    }
                )
    for learning_rate in (0.001, 0.003, 0.006):
        rows.append(
            {
                "problem": "1",
                "subpart": "a",
                "batch_size": 64,
                "tokens": P42_TOKENS,
                "peak_lr": learning_rate,
                "weight_decay": 0.1,
                "width": 512,
                "depth": 8,
                "parameterization": "baseline",
                "optimizer": "adamw",
                "lr_schedule": "linear",
                "seed": 42,
                "final_val_loss": _parabola(learning_rate, 0.003),
            }
        )
    return rows


def self_check():
    """Fit a convex synthetic sweep and check dedup, metadata, and muP wiring."""
    transferred = {
        "learning_rate": 0.003,
        "weight_decay": 0.1,
        "run_id": "synthetic-p1",
        "tokens": P42_TOKENS,
    }
    measurements = _synthetic_measurements()
    plan = {
        "batch": "batch2",
        "p32b": plan_p32b(measurements, supplied=[]),
        "p42c": plan_p42c(measurements, transferred=transferred, supplied=[]),
    }
    jobs = schedule(plan)
    analytical = list(_planned_jobs(plan))
    if len(analytical) != 22:
        raise AssertionError(f"expected 22 analytical targets, got {len(analytical)}")
    if len(jobs) != 18:
        raise AssertionError(f"expected 18 unique training configs, got {len(jobs)}")
    p32 = [job for job in jobs if job["memberships"][0]["problem"] == "3.2"]
    p42 = [job for job in jobs if job["memberships"][0]["problem"] == "4.2"]
    if len(p32) != 12 or len(p42) != 6:
        raise AssertionError(f"expected 12 Problem 3.2 and 6 Problem 4.2 jobs, got {len(p32)} and {len(p42)}")
    shared = [job for job in p42 if len(job["memberships"]) > 1]
    if len(shared) != 2:
        raise AssertionError(f"expected the shared prediction/transfer run for both prescriptions, got {len(shared)}")
    for job in jobs:
        metadata = job["config"].experiment_metadata
        if metadata["batch"] != "batch2" or metadata["source_batch"] != "batch1":
            raise AssertionError(f"{job['label']} is missing batch2 metadata")
        if "batch1" in job["config"].wandb_tags or "batch2" not in job["config"].wandb_tags:
            raise AssertionError(f"{job['label']} has tags {job['config'].wandb_tags}")
        if job["config"].save_model:
            raise AssertionError(f"{job['label']} would save a checkpoint")
        logger_text = repr(job["config"].metric_loggers)
        for path in (
            "experiments.a2.probes:features",
            "experiments.a2.probes:gradients",
            "experiments.a2.alignment:before_update",
            "experiments.a2.alignment:after_update",
        ):
            if path not in logger_text:
                raise AssertionError(f"{job['label']} is missing diagnostic logger {path}")
    baseline = next(job for job in p32 if job["metadata"]["batch_size"] == 128)
    model = baseline["config"].model_config
    if model is None or model.name != "a2-d8":
        raise AssertionError("Problem 3.2 must keep the course model name a2-d8")
    if baseline["metadata"]["token_budget"] != P32_TOKENS:
        raise AssertionError("Problem 3.2 token_budget must stay 614400000")
    if baseline["metadata"]["processed_tokens"] != 614_334_464:
        raise AssertionError(baseline["metadata"]["processed_tokens"])
    mup = next(
        job
        for job in p42
        if job["metadata"].get("parameterization") == "mup"
        and len(job["memberships"]) > 1
    )
    scaling = mup["config"].model_builder_kwargs
    if mup["config"].model_builder != "experiments.a2.policies:build_scaled_model":
        raise AssertionError(mup["config"].model_builder)
    if mup["config"].optimizer_builder != "experiments.a2.policies:build_scaled_optimizer":
        raise AssertionError(mup["config"].optimizer_builder)
    if not same_float(scaling["hidden_lr_multiplier"], 0.5):
        raise AssertionError(scaling["hidden_lr_multiplier"])
    if not same_float(scaling["output_multiplier"], 0.5):
        raise AssertionError(scaling["output_multiplier"])
    roles = {membership["config_role"] for membership in mup["memberships"]}
    if roles != {"powerlaw_prediction", "direct_transfer", "local_sweep"}:
        raise AssertionError(roles)
    print(format_summary(plan, jobs, measurements))
    print("self-check passed")
    return plan, jobs


def format_operator_summary(plan, jobs, measurements):
    def count(problem, subpart):
        return sum(
            1
            for job in jobs
            if any(
                membership["problem"] == problem and membership["subpart"] == subpart
                for membership in job["memberships"]
            )
        )

    statuses = [job.get("launch_status", "missing") for job in jobs]
    predictions = len(plan["p32b"]["hypotheses"]) * len(P32_TARGET_BATCHES)
    predictions += len(plan["p42c"]["prescriptions"])
    lines = [
        "A2 Batch 2",
        "==========",
        "",
        "Prerequisites: OK",
        "",
        f"Source runs found: {len(measurements)}",
        f"Predictions computed: {predictions}",
        f"Already completed: {statuses.count('completed')}",
        f"Currently running: {statuses.count('running')}",
        f"Failed, will relaunch: {statuses.count('failed')}",
        f"New jobs to launch: {statuses.count('missing')}",
        "",
        f"P3.2b: {count('3.2', 'b')}",
        f"P4.2c: {count('4.2', 'c')}",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fit, write the manifest, and do not submit jobs.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit new Batch 2 jobs. This is the default when --dry-run is omitted.",
    )
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Fit a synthetic sweep and check the schedule. Does not read W&B or launch.",
    )
    args = parser.parse_args(argv)
    if args.self_check:
        self_check()
        return 0
    if args.execute and args.dry_run:
        parser.error("pass only one of --dry-run and --execute")
    if args.max_parallel < 1:
        parser.error("--max-parallel must be positive")
    execute = not args.dry_run
    measurements, catalog = load_source_runs()
    plan = plan_batch2(measurements)
    jobs = classify_launch(schedule(plan), catalog)
    paths = write_manifest(plan, jobs, args.manifest_dir)
    print(format_operator_summary(plan, jobs, measurements))
    print()
    if args.dry_run:
        print(format_summary(plan, jobs, measurements))
        print()
    print(f"Wrote {paths[0]}")
    print(f"Wrote {paths[1]}")
    pending = [job for job in jobs if job.get("launch_status") in {"missing", "failed"}]
    for job in jobs:
        if job.get("launch_status") == "failed":
            print(f"Relaunching failed run {job['label']}")
        elif job.get("launch_status") == "running":
            print(f"Leaving running run {job['label']}")
    if not execute:
        print("Dry run only. Run without --dry-run to submit these jobs.")
        return 0
    if not pending:
        print("Nothing to submit.")
        return 0
    from experiments.a2.modal_launcher import launch_training_jobs

    launch_training_jobs(
        [job["config"] for job in pending],
        max_parallel_runs=args.max_parallel,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Batch2Error as exc:
        print(exc)
        raise SystemExit(1) from exc
