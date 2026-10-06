"""Batch 1 of the CS312 A2 hyperparameter-scaling experiments.

The normal command submits the jobs. ``--dry-run`` only writes the manifest::

    uv run python -m experiments.a2.batches.batch1 --dry-run
    uv run python -m experiments.a2.batches.batch1

Scheduled runs are the configurations that can be fixed from the handout and
the supplied sweeps. Shared training configurations are launched once and
carry every analysis in ``memberships`` and in W&B tags. Query shared runs
with those, for example::

    any(m["problem"] == "3.2" and m["subpart"] == "a" for m in run.config["memberships"])

The flat ``problem`` and ``subpart`` fields are the primary analysis. A run
that also serves another subpart still has that subpart in ``memberships``
and in its tags (``p32a``, ``p32b``, and so on).

Completed runs are skipped. Failed runs are relaunched. Runs that are still
in progress are left alone. Batch 2 and Batch 3 read these runs from W&B
metadata; their hyperparameters are not invented here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from model_config import LMConfig

from experiments.a2.hyperball import ADAMH_DEFAULT_ADAM_LR_RATIO
from experiments.a2.policies import (
    LONG_REFERENCE_DEPTH,
    LONG_REFERENCE_WIDTH,
    STRESS_REFERENCE_DEPTH,
    STRESS_REFERENCE_WIDTH,
    long_scaling,
    stress_identity,
)
from experiments.a2.helpers.results import canonicalize_parameterization
from experiments.a2.scaling import (
    P1_TARGET_TOKENS,
    PRODUCT_TARGET_LR,
    PRODUCT_TARGET_TOKENS,
    fit_batch1_sources,
    load_supplied,
)


# Problem 1's AdamW grid. B=64 at 614.4M and WD 0.1 is already supplied, so
# the nine new Problem 3.2a runs are the other three batch sizes.
P32A_LEARNING_RATES = (0.0015, 0.003, 0.006)
P32_BATCHES = (8, 16, 32, 64)
P32_TOKENS = 614_400_000
# The handout does not list the WD grid. 0.1, 0.2, and 0.4 are the supplied
# 614.4M joint-sweep values, so the B=64 points at those WDs are reused.
# 0.8 continues one factor of two past that grid: at B=64 and LR 0.0015 the
# measured loss is still decreasing at WD 0.4.
P32B_WEIGHT_DECAYS = (0.1, 0.2, 0.4, 0.8)
P32B_LEARNING_RATE = 0.0015
P2C_COMPARISON_WDS = (0.05, 0.1, 0.2)
# About 30 five-step width runs: 5 base LRs x 3 widths x 2 prescriptions.
P41A_BASE_LRS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
P41A_WIDTHS = (640, 2560, 5120)
# About 25 five-step depth runs. Depth 2 makes the three prescriptions
# identical, so 4 LRs x 7 distinct configs = 28 unique jobs.
P41C_BASE_LRS = (3e-4, 1e-3, 3e-3, 1e-2)
P41C_DEPTHS = (2, 100, 1000)
P42A_WIDTHS = (128, 256)
P42A_EXPONENTS = (-2, -1, 0, 1, 2)
P42D_DEPTHS = (4, 16)
P42D_EXPONENTS = (-1, 0, 1)
P42_TOKENS = 153_600_000
# One octave below the prediction, plus half an octave on either side.
HYPERBALL_NEARBY_FACTORS = (0.5, 2**-0.5, 2**0.5)
# Keep manifests in experiments/a2/manifests after this module moved into batches/.
MANIFEST_DIR = Path(__file__).resolve().parents[1] / "manifests"


def fmt_float(value):
    return format(float(value), ".8g")


def _csv_cell(value):
    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)
    return "" if value is None else value


def same_float(left, right):
    return math.isclose(float(left), float(right), rel_tol=1e-8, abs_tol=0.0)


def coalesce(value, anchors):
    """Reuse an earlier hyperparameter when the new value matches it."""
    for anchor in anchors:
        if same_float(value, anchor):
            return anchor
    return value


def problem_tags(problem, subpart):
    compact = problem.replace(".", "")
    return [f"p{compact}", f"p{compact}{subpart}"]


def membership_tags(membership):
    tags = ["a2", "batch1", *problem_tags(membership["problem"], membership["subpart"])]
    for key in ("family", "config_role", "hypothesis", "stage", "parameterization"):
        value = membership.get(key)
        if value:
            tags.append(str(value))
    return tags


def unique_strings(values):
    found = []
    for value in values:
        if value not in found:
            found.append(value)
    return found


def architecture(width, depth, parameterization):
    if width % 64:
        raise ValueError(f"width {width} is not divisible by the head dimension 64")
    heads = width // 64
    return LMConfig(
        name=f"a2-w{width:04d}-d{depth:03d}-{parameterization}",
        vocab_size=4096,
        context_length=1024,
        hidden_size=width,
        intermediate_size=int(width * 3.5),
        num_hidden_layers=depth,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        head_dim=64,
    )


def find_supplied(rows, *, tokens, batch_size, learning_rate, weight_decay, optimizer, lr_schedule):
    hits = [
        row
        for row in rows
        if row["tokens"] == tokens
        and row["batch_size"] == batch_size
        and row["optimizer"] == optimizer
        and row["lr_schedule"] == lr_schedule
        and row["seed"] == 42
        and same_float(row["learning_rate"], learning_rate)
        and same_float(row["weight_decay"], weight_decay)
        and same_float(row["beta1"], 0.9)
        and same_float(row["beta2"], 0.95)
    ]
    if not hits:
        return None
    hits.sort(key=lambda row: row["run_id"])
    return hits[0]


def lm_identity(config):
    model = config.model_config.to_dict() if config.model_config is not None else {"name": config.model_name}
    payload = {
        "kind": "lm",
        "model": model,
        "model_builder": config.model_builder,
        "model_builder_kwargs": config.model_builder_kwargs,
        "optimizer_name": config.optimizer_name,
        "optimizer_builder": config.optimizer_builder,
        "optimizer_kwargs": config.optimizer_kwargs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "batch_size": config.batch_size,
        "num_micro_batches": config.num_micro_batches,
        "num_train_sequences": config.num_train_sequences,
        "num_epochs": config.num_epochs,
        "lr_schedule": config.lr_schedule,
        "warmup_percent": config.warmup_percent,
        "grad_norm": config.grad_norm,
        "precision": config.precision,
        "data_seed": config.data_seed,
        "model_seed": config.model_seed,
        "dropout": config.dropout,
        "qk_norm": config.qk_norm,
        "tie_word_embeddings": config.tie_word_embeddings,
        "train_path": config.train_dataset.path,
        "val_path": config.val_dataset.path,
    }
    return json.dumps(payload, sort_keys=True, default=str)


def stress_key(policy, width, depth, base_lr, precision, reference_width, reference_depth):
    payload = {
        "kind": "stress",
        "width": width,
        "depth": depth,
        "precision": precision,
        "seed": 42,
        "steps": 5,
        "batch_size": 64,
        "microbatch": 8,
        "head_dim": 64,
        "reference_width": reference_width,
        "reference_depth": reference_depth,
        **stress_identity(policy, width, depth, base_lr, reference_width, reference_depth),
    }
    return json.dumps(payload, sort_keys=True)


class Catalog:
    def __init__(self, supplied_rows):
        self.supplied_rows = supplied_rows
        self.jobs = {}
        self.order = []
        self.references = {}

    def add(self, *, label, kind, membership, identity, hyperparameters, config=None, stress=None, supplied_query=None):
        if supplied_query is not None:
            row = find_supplied(self.supplied_rows, **supplied_query)
            if row is not None:
                self._add_reference(row, label, membership, hyperparameters)
                return
        if identity in self.jobs:
            job = self.jobs[identity]
            job["aliases"].append(label)
            job["memberships"].append(membership)
            self._finish(job)
            return
        job = {
            "label": label,
            "aliases": [],
            "kind": kind,
            "identity": identity,
            "memberships": [membership],
            "hyperparameters": hyperparameters,
            "config": config,
            "stress": stress,
        }
        self._finish(job)
        self.jobs[identity] = job
        self.order.append(identity)

    def _add_reference(self, row, label, membership, hyperparameters):
        reference = self.references.setdefault(
            row["run_id"],
            {
                "run_id": row["run_id"],
                "run_url": row["run_url"],
                "parts": row["parts"],
                "final_val_loss": row["final_val_loss"],
                "hyperparameters": {
                    "tokens": row["tokens"],
                    "batch_size": row["batch_size"],
                    "learning_rate": row["learning_rate"],
                    "weight_decay": row["weight_decay"],
                    "optimizer": row["optimizer"],
                    "lr_schedule": row["lr_schedule"],
                    "width": 512,
                    "depth": 8,
                    "parameterization": "baseline",
                },
                "aliases": [],
                "memberships": [],
            },
        )
        reference["aliases"].append(label)
        reference["memberships"].append(membership)
        reference["hyperparameters"].update(
            {key: value for key, value in hyperparameters.items() if key not in reference["hyperparameters"]}
        )

    def _finish(self, job):
        tags = []
        for membership in job["memberships"]:
            tags.extend(membership_tags(membership))
        job["tags"] = unique_strings(tags)
        parameterizations = unique_strings(
            membership["parameterization"]
            for membership in job["memberships"]
            if membership.get("parameterization")
        )
        primary = job["memberships"][0]
        hyperparameters = job["hyperparameters"]
        canonical_parameterizations = [
            canonicalize_parameterization(item) for item in parameterizations
        ]
        job["metadata"] = {
            "assignment": "a2",
            "batch": "batch1",
            "label": job["label"],
            "aliases": list(job["aliases"]),
            "memberships": list(job["memberships"]),
            "problem": primary["problem"],
            "subpart": primary["subpart"],
            "family": primary["family"],
            "config_role": primary["config_role"],
            "hypothesis": primary.get("hypothesis"),
            "stage": primary.get("stage"),
            "parameterization": (
                canonical_parameterizations[0]
                if len(canonical_parameterizations) == 1
                else canonical_parameterizations
            ),
            "equivalent_parameterizations": canonical_parameterizations,
            "source_type": primary.get("source_type"),
            "protocol": "five_step" if job["kind"] == "stress" else "language_model",
            **hyperparameters,
        }
        if job["config"] is not None:
            config = job["config"]
            job["metadata"].update(
                {
                    "optimizer": config.optimizer_name,
                    "optimizer_name": config.optimizer_name,
                    "lr_schedule": config.lr_schedule,
                    "warmup_percent": config.warmup_percent,
                    "beta1": config.beta1,
                    "beta2": config.beta2,
                    "weight_decay": config.weight_decay,
                    "batch_size": config.batch_size,
                    "seed": config.model_seed,
                    "peak_lr": config.learning_rate,
                }
            )
        else:
            job["metadata"].update(
                {
                    "optimizer_name": hyperparameters.get("optimizer"),
                    "lr_schedule": hyperparameters.get("lr_schedule", "constant"),
                    "warmup_percent": hyperparameters.get("warmup_percent", 0.0),
                }
            )
        if job["config"] is not None:
            from dataclasses import replace

            job["config"] = replace(
                job["config"],
                run_name_suffix=job["label"],
                wandb_tags=tuple(job["tags"]),
                experiment_metadata=job["metadata"],
            )
        if job["stress"] is not None:
            job["stress"]["label"] = job["label"]
            job["stress"]["tags"] = list(job["tags"])
            job["stress"]["metadata"] = job["metadata"]

    def scheduled(self):
        return [self.jobs[key] for key in self.order]


def _positive(value, label):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be a positive finite number, got {value}")
    return float(value)


def build_lm_config(spec):
    from experiments.a2.modal_launcher import config as a2_config

    kwargs = {
        "tokens": spec["tokens"],
        "batch": spec["batch_size"],
        "learning_rate": spec["learning_rate"],
        "weight_decay": spec["weight_decay"],
        "optimizer_name": spec["optimizer_name"],
        "model_seed": 42,
    }
    baseline_course = (
        spec["parameterization"] == "baseline"
        and spec["width"] == LONG_REFERENCE_WIDTH
        and spec["depth"] == 8
    )
    if not baseline_course:
        kwargs["model_config"] = architecture(spec["width"], spec["depth"], spec["parameterization"])
    if spec["parameterization"] != "baseline":
        scaling = long_scaling(
            spec["parameterization"],
            spec["width"],
            spec["depth"],
            reference_width=LONG_REFERENCE_WIDTH,
            reference_depth=LONG_REFERENCE_DEPTH,
        )
        kwargs["model_builder"] = "experiments.a2.policies:build_scaled_model"
        kwargs["model_builder_kwargs"] = scaling
        kwargs["optimizer_builder"] = "experiments.a2.policies:build_scaled_optimizer"
    return a2_config(**kwargs)


def add_lm(catalog, *, label, membership, spec):
    spec = {
        "optimizer_name": "adamw",
        "weight_decay": 0.1,
        "width": LONG_REFERENCE_WIDTH,
        "depth": 8,
        "parameterization": "baseline",
        **spec,
    }
    config = build_lm_config(spec)
    hyperparameters = {
        "token_budget": spec["tokens"],
        "target_tokens": spec["tokens"],
        "batch_size": spec["batch_size"],
        "width": spec["width"],
        "depth": spec["depth"],
        "peak_lr": spec["learning_rate"],
        "base_lr": spec["learning_rate"],
        "weight_decay": config.weight_decay,
        "optimizer": spec["optimizer_name"],
        "beta1": config.beta1,
        "beta2": config.beta2,
        "seed": 42,
        "num_updates": None,
    }
    if spec["optimizer_name"] == "adamh":
        hyperparameters["hyperball_adam_lr_ratio"] = ADAMH_DEFAULT_ADAM_LR_RATIO
        hyperparameters["hyperball_adam_lr"] = spec["learning_rate"] * ADAMH_DEFAULT_ADAM_LR_RATIO
        hyperparameters["nearby_factor"] = float(spec["nearby_factor"])
    if spec["parameterization"] != "baseline":
        scaling = config.model_builder_kwargs
        hyperparameters["hidden_lr_multiplier"] = scaling["hidden_lr_multiplier"]
        hyperparameters["output_multiplier"] = scaling["output_multiplier"]
        hyperparameters["residual_multiplier"] = scaling["residual_multiplier"]
        hyperparameters["block_eps_multiplier"] = scaling["block_eps_multiplier"]
    supplied_query = None
    if spec["parameterization"] == "baseline" and spec["width"] == 512 and spec["depth"] == 8:
        supplied_query = {
            "tokens": spec["tokens"],
            "batch_size": spec["batch_size"],
            "learning_rate": spec["learning_rate"],
            "weight_decay": config.weight_decay,
            "optimizer": "adamh" if spec["optimizer_name"] == "adamh" else "adamw",
            "lr_schedule": "linear",
        }
    catalog.add(
        label=label,
        kind="lm",
        membership=membership,
        identity=lm_identity(config),
        hyperparameters=hyperparameters,
        config=config,
        supplied_query=supplied_query,
    )


def add_stress(catalog, *, label, membership, policy, width, depth, base_lr, precision, reference_width, reference_depth):
    identity_numbers = stress_identity(policy, width, depth, base_lr, reference_width, reference_depth)
    stress = {
        "policy": policy,
        "width": width,
        "depth": depth,
        "base_lr": base_lr,
        "precision": precision,
        "reference_width": reference_width,
        "reference_depth": reference_depth,
        "seed": 42,
        "steps": 5,
        "batch_size": 64,
        "microbatch": 8,
        "head_dim": 64,
    }
    catalog.add(
        label=label,
        kind="stress",
        membership=membership,
        identity=stress_key(policy, width, depth, base_lr, precision, reference_width, reference_depth),
        hyperparameters={
            "token_budget": 5 * 64 * 1024,
            "batch_size": 64,
            "width": width,
            "depth": depth,
            "peak_lr": base_lr,
            "base_lr": base_lr,
            "hidden_lr": identity_numbers["hidden_lr"],
            "weight_decay": 0.0,
            "optimizer": "adam",
            "beta1": 0.9,
            "beta2": 0.95,
            "seed": 42,
            "num_updates": 5,
            "precision": precision,
            "warmup_percent": 0.0,
            "grad_norm": None,
            "lr_schedule": "constant",
            "output_multiplier": identity_numbers["output_multiplier"],
            "residual_multiplier": identity_numbers["residual_multiplier"],
            "reference_width": reference_width,
            "reference_depth": reference_depth,
        },
        stress=stress,
    )


def _membership(**fields):
    parameterization = fields.get("parameterization")
    if parameterization:
        parameterization = canonicalize_parameterization(parameterization)
    record = {
        "problem": fields["problem"],
        "subpart": fields["subpart"],
        "family": fields["family"],
        "config_role": fields["config_role"],
        "hypothesis": fields.get("hypothesis"),
        "stage": fields.get("stage"),
        "parameterization": parameterization,
        "source_type": fields.get("source_type"),
    }
    return record


def _assert_plan(jobs):
    labels = [job["label"] for job in jobs]
    if len(labels) != len(set(labels)):
        raise AssertionError("Batch 1 labels are not unique")
    names = [name for job in jobs for name in (job["label"], *job["aliases"])]
    if len(names) != len(set(names)):
        raise AssertionError("Batch 1 labels and aliases overlap")
    identities = [job["identity"] for job in jobs]
    if len(identities) != len(set(identities)):
        raise AssertionError("Batch 1 training identities are not unique")
    for job in jobs:
        for membership in job["memberships"]:
            if membership["problem"] == "3.2" and membership["subpart"] == "c":
                raise AssertionError("Problem 3.2c is Batch 3")
            if membership["problem"] == "4.2" and membership["subpart"] == "c":
                raise AssertionError("Problem 4.2c is Batch 2")
            if membership["problem"] == "3.1":
                raise AssertionError("Problem 3.1 is a local CPU simulation")
        hyperparameters = job["hyperparameters"]
        if hyperparameters.get("batch_size") in {128, 256} and any(
            membership["problem"] == "3.2" for membership in job["memberships"]
        ):
            raise AssertionError("B=128 and B=256 targets belong to Batch 2")
        if hyperparameters.get("width") == 1024:
            raise AssertionError("width 1024 belongs to Batch 2")


def build_batch1(include_p2c_transfer=True):
    fits = fit_batch1_sources()
    catalog = Catalog(load_supplied())
    predicted_all6 = _positive(fits["p1c_all6"]["predicted_lr"], "six-budget LR")
    predicted_large3 = _positive(fits["p1c_large3"]["predicted_lr"], "three-budget LR")
    scheduled_lrs = []
    for learning_rate, role in (
        *((lr, "target_grid") for lr in (0.0015, 0.003, 0.006)),
        (predicted_all6, "prediction_all6"),
        (predicted_large3, "prediction_large3"),
    ):
        learning_rate = coalesce(learning_rate, scheduled_lrs)
        scheduled_lrs.append(learning_rate)
        add_lm(
            catalog,
            label=f"a2-b1-p1c-lr{fmt_float(learning_rate)}-{role}",
            membership=_membership(
                problem="1",
                subpart="c",
                family="token_lr_scaling_target",
                config_role=role,
                source_type="provided",
                parameterization="baseline",
            ),
            spec={
                "tokens": P1_TARGET_TOKENS,
                "batch_size": 64,
                "learning_rate": learning_rate,
                "weight_decay": 0.1,
            },
        )

    predicted_hyperball = _positive(fits["p1d_hyperball"]["predicted_lr"], "Hyperball LR")
    hyperball_grid = [(predicted_hyperball, "predicted", 1.0)]
    for factor in HYPERBALL_NEARBY_FACTORS:
        hyperball_grid.append((predicted_hyperball * factor, "nearby", factor))
    scheduled_lrs = []
    for learning_rate, role, factor in hyperball_grid:
        learning_rate = coalesce(learning_rate, scheduled_lrs)
        scheduled_lrs.append(learning_rate)
        add_lm(
            catalog,
            label=f"a2-b1-p1d-hyperball-lr{fmt_float(learning_rate)}-{role}",
            membership=_membership(
                problem="1",
                subpart="d",
                family="hyperball_lr_scaling",
                config_role=role,
                source_type="provided",
                parameterization="baseline",
            ),
            spec={
                "tokens": 1_228_800_000,
                "batch_size": 64,
                "learning_rate": learning_rate,
                "weight_decay": 0.0,
                "optimizer_name": "adamh",
                "nearby_factor": factor,
            },
        )

    predicted_wd = _positive(fits["p2c_product"]["predicted_wd"], "predicted WD")
    scheduled_wds = []
    for weight_decay, role in (
        *((wd, "wd_grid") for wd in P2C_COMPARISON_WDS),
        (predicted_wd, "predicted"),
    ):
        weight_decay = coalesce(weight_decay, scheduled_wds)
        scheduled_wds.append(weight_decay)
        add_lm(
            catalog,
            label=f"a2-b1-p2c-lr{fmt_float(PRODUCT_TARGET_LR)}-wd{fmt_float(weight_decay)}-{role}",
            membership=_membership(
                problem="2",
                subpart="c",
                family="lr_wd_product_rule",
                config_role=role,
                source_type="provided",
                parameterization="baseline",
            ),
            spec={
                "tokens": PRODUCT_TARGET_TOKENS,
                "batch_size": 64,
                "learning_rate": PRODUCT_TARGET_LR,
                "weight_decay": weight_decay,
            },
        )

    best_joint = fits["p2_1536m_best_joint"]
    transfer = {
        "label": (
            f"a2-b1-p2c-transfer-lr{fmt_float(best_joint['learning_rate'])}"
            f"-wd{fmt_float(best_joint['weight_decay'])}"
        ),
        "learning_rate": best_joint["learning_rate"],
        "weight_decay": best_joint["weight_decay"],
        "source_run_id": best_joint["run_id"],
        "source_run_url": best_joint["run_url"],
        "source_loss": best_joint["final_val_loss"],
        "scheduled": include_p2c_transfer,
        "reason": (
            "Problem 2c retrains the supplied 153.6M LR-WD pair at 2.4576B "
            "tokens. The source run itself is not repeated. This target is "
            "part of the default Batch 1 launch."
        ),
    }
    if include_p2c_transfer:
        add_lm(
            catalog,
            label=transfer["label"],
            membership=_membership(
                problem="2",
                subpart="c",
                family="lr_wd_product_rule",
                config_role="transferred_source_best",
                source_type="provided",
                parameterization="baseline",
            ),
            spec={
                "tokens": PRODUCT_TARGET_TOKENS,
                "batch_size": 64,
                "learning_rate": best_joint["learning_rate"],
                "weight_decay": best_joint["weight_decay"],
            },
        )

    for batch_size in P32_BATCHES:
        for learning_rate in P32A_LEARNING_RATES:
            add_lm(
                catalog,
                label=(
                    f"a2-b1-p32a-lr-b{batch_size:03d}-lr{fmt_float(learning_rate)}-wd0.1"
                ),
                membership=_membership(
                    problem="3.2",
                    subpart="a",
                    family="batch_lr_source",
                    config_role="source",
                    stage="source",
                    parameterization="baseline",
                ),
                spec={
                    "tokens": P32_TOKENS,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": 0.1,
                },
            )
            add_lm(
                catalog,
                label=(
                    f"a2-b1-p32b-lr-b{batch_size:03d}-lr{fmt_float(learning_rate)}-wd0.1"
                ),
                membership=_membership(
                    problem="3.2",
                    subpart="b",
                    family="batch_scaling",
                    config_role="source",
                    hypothesis="fixed_wd_scale_lr",
                    stage="source",
                    parameterization="baseline",
                ),
                spec={
                    "tokens": P32_TOKENS,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": 0.1,
                },
            )
    for batch_size in P32_BATCHES:
        for weight_decay in P32B_WEIGHT_DECAYS:
            add_lm(
                catalog,
                label=(
                    f"a2-b1-p32b-wd-b{batch_size:03d}-lr{fmt_float(P32B_LEARNING_RATE)}"
                    f"-wd{fmt_float(weight_decay)}"
                ),
                membership=_membership(
                    problem="3.2",
                    subpart="b",
                    family="batch_scaling",
                    config_role="source",
                    hypothesis="fixed_lr_scale_wd",
                    stage="source",
                    parameterization="baseline",
                ),
                spec={
                    "tokens": P32_TOKENS,
                    "batch_size": batch_size,
                    "learning_rate": P32B_LEARNING_RATE,
                    "weight_decay": weight_decay,
                },
            )

    for policy in ("kaiming", "mup"):
        for width in P41A_WIDTHS:
            for base_lr in P41A_BASE_LRS:
                add_stress(
                    catalog,
                    label=f"a2-b1-p41a-width-{policy}-w{width:04d}-lr{fmt_float(base_lr)}",
                    membership=_membership(
                        problem="4.1",
                        subpart="a",
                        family="width_5step",
                        config_role="lr_sweep",
                        parameterization=policy,
                    ),
                    policy=policy,
                    width=width,
                    depth=2,
                    base_lr=base_lr,
                    precision="fp32",
                    reference_width=STRESS_REFERENCE_WIDTH,
                    reference_depth=STRESS_REFERENCE_DEPTH,
                )
    for policy in ("mup", "depth_mup", "completep"):
        for depth in P41C_DEPTHS:
            for base_lr in P41C_BASE_LRS:
                add_stress(
                    catalog,
                    label=f"a2-b1-p41c-depth-{policy}-d{depth:04d}-lr{fmt_float(base_lr)}",
                    membership=_membership(
                        problem="4.1",
                        subpart="c",
                        family="depth_5step",
                        config_role="lr_sweep",
                        parameterization=policy,
                    ),
                    policy=policy,
                    width=64,
                    depth=depth,
                    base_lr=base_lr,
                    precision="mp",
                    reference_width=64,
                    reference_depth=STRESS_REFERENCE_DEPTH,
                )

    transferred_lr = _positive(fits["p1_1536m_best_sampled"]["learning_rate"], "transferred width LR")
    for parameterization in ("baseline", "mup"):
        for width in P42A_WIDTHS:
            for exponent in P42A_EXPONENTS:
                learning_rate = transferred_lr * 2.0**exponent
                role = "transferred" if exponent == 0 else "local_sweep"
                add_lm(
                    catalog,
                    label=(
                        f"a2-b1-p42a-width-{parameterization}-w{width:04d}"
                        f"-lr{fmt_float(learning_rate)}-{role}"
                    ),
                    membership=_membership(
                        problem="4.2",
                        subpart="a",
                        family="long_width_transfer",
                        config_role=role,
                        parameterization=parameterization,
                        source_type="provided",
                    ),
                    spec={
                        "tokens": P42_TOKENS,
                        "batch_size": 64,
                        "learning_rate": learning_rate,
                        "weight_decay": 0.1,
                        "width": width,
                        "depth": 8,
                        "parameterization": parameterization,
                    },
                )
    for parameterization in ("mup", "depth_mup", "completep"):
        for depth in P42D_DEPTHS:
            for exponent in P42D_EXPONENTS:
                learning_rate = transferred_lr * 2.0**exponent
                role = "transferred" if exponent == 0 else "local_sweep"
                add_lm(
                    catalog,
                    label=(
                        f"a2-b1-p42d-depth-{parameterization}-d{depth:04d}"
                        f"-lr{fmt_float(learning_rate)}-{role}"
                    ),
                    membership=_membership(
                        problem="4.2",
                        subpart="d",
                        family="long_depth_transfer",
                        config_role=role,
                        parameterization=parameterization,
                        source_type="provided",
                    ),
                    spec={
                        "tokens": P42_TOKENS,
                        "batch_size": 64,
                        "learning_rate": learning_rate,
                        "weight_decay": 0.1,
                        "width": LONG_REFERENCE_WIDTH,
                        "depth": depth,
                        "parameterization": parameterization,
                    },
                )

    from experiments.a2.provided_sweeps import reference_diagnostics

    reference = reference_diagnostics()
    jobs = catalog.scheduled()
    _assert_plan(jobs)
    return {
        "fits": fits,
        "jobs": jobs,
        "references": list(catalog.references.values()),
        "optional_p2c_transfer": transfer,
        "width512_reference": {
            "run_id": reference["run_id"],
            "run_url": reference["run_url"],
            "tokens": reference["tokens"],
            "learning_rate": reference["learning_rate"],
            "weight_decay": reference["weight_decay"],
            "width": reference["width"],
            "depth": 8,
            "note": (
                "Supplied width-512 diagnostics for Problem 4.2. At width 512 and "
                "depth 8, standard muP, Depth-muP, and CompleteP match this course "
                "baseline, so it is not retrained."
            ),
        },
    }


def _public_job(job):
    from train import training_run_name

    wandb_name = training_run_name(job["config"]) if job["config"] is not None else job["label"]
    return {
        "label": job["label"],
        "aliases": job["aliases"],
        "wandb_name": wandb_name,
        "kind": job["kind"],
        "tags": job["tags"],
        "metadata": job["metadata"],
        "memberships": job["memberships"],
        "hyperparameters": job["hyperparameters"],
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("refusing to write a non-finite fit value")
        return value
    return value


def write_manifest(plan, directory=MANIFEST_DIR):
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "fits": _json_ready(plan["fits"]),
        "jobs": [_public_job(job) for job in plan["jobs"]],
        "references": _json_ready(plan["references"]),
        "optional_p2c_transfer": _json_ready(plan["optional_p2c_transfer"]),
        "width512_reference": plan["width512_reference"],
    }
    json_path = directory / "batch1_manifest.json"
    fits_path = directory / "batch1_fits.json"
    csv_path = directory / "batch1_manifest.csv"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    fits_path.write_text(json.dumps(payload["fits"], indent=2) + "\n")
    fields = (
        "label",
        "kind",
        "problem",
        "subpart",
        "family",
        "config_role",
        "hypothesis",
        "parameterization",
        "tokens",
        "batch_size",
        "width",
        "depth",
        "peak_lr",
        "weight_decay",
        "optimizer",
        "aliases",
        "wandb_name",
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for job in payload["jobs"]:
            hyperparameters = job["hyperparameters"]
            primary = job["memberships"][0]
            writer.writerow(
                {
                    "label": job["label"],
                    "kind": job["kind"],
                    "problem": primary["problem"],
                    "subpart": primary["subpart"],
                    "family": primary["family"],
                    "config_role": primary["config_role"],
                    "hypothesis": primary.get("hypothesis") or "",
                    "parameterization": _csv_cell(job["metadata"].get("parameterization")),
                    "tokens": hyperparameters.get("token_budget"),
                    "batch_size": hyperparameters.get("batch_size"),
                    "width": hyperparameters.get("width"),
                    "depth": hyperparameters.get("depth"),
                    "peak_lr": hyperparameters.get("peak_lr"),
                    "weight_decay": hyperparameters.get("weight_decay"),
                    "optimizer": hyperparameters.get("optimizer"),
                    "aliases": "|".join(job["aliases"]),
                    "wandb_name": job["wandb_name"],
                }
            )
    return json_path, csv_path, fits_path


def _bucket(job):
    primary = job["memberships"][0]
    return f"P{primary['problem']}{primary['subpart']}"


def job_tracking_name(job):
    if job.get("config") is not None:
        from train import training_run_name

        return training_run_name(job["config"])
    return job["label"]


def _membership_counts(plan):
    counts = {}
    for job in plan["jobs"]:
        seen = set()
        for membership in job["memberships"]:
            bucket = f"P{membership['problem']}.{membership['subpart']}"
            if bucket in seen:
                continue
            seen.add(bucket)
            counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def _volume_complete(jobs):
    complete = {}
    stress_jobs = [job for job in jobs if job["kind"] == "stress"]
    language_jobs = [job for job in jobs if job["kind"] == "lm"]
    if stress_jobs:
        from experiments.a2.launch_batch1 import _existing_stress_labels

        existing = _existing_stress_labels()
        for job in stress_jobs:
            name = job_tracking_name(job)
            complete[name] = job["label"] in existing or any(
                alias in existing for alias in job["aliases"]
            )
    if language_jobs:
        from modal_train import _completed_model_exists

        for job in language_jobs:
            exists, _run_name = _completed_model_exists(job["config"])
            complete[job_tracking_name(job)] = bool(exists)
    return complete


def completion_status(plan, *, check_volume=False):
    """Classify each Batch 1 job. Raises ResultsError if W&B cannot be read."""
    from experiments.a2.helpers.results import classify_named_runs, query_a2_runs

    described = query_a2_runs()
    names = [job_tracking_name(job) for job in plan["jobs"]]
    volume = _volume_complete(plan["jobs"]) if check_volume else {}
    by_name = classify_named_runs(names, described, volume_complete=volume)
    return {job["label"]: by_name[job_tracking_name(job)] for job in plan["jobs"]}


def format_operator_summary(plan, statuses=None, status_note=None):
    counts = _membership_counts(plan)
    unique = len(plan["jobs"])
    lines = [
        "A2 Batch 1",
        "==========",
        "",
        "Prerequisites: none",
        "",
    ]
    if statuses is None:
        lines.append(f"Unique jobs: {unique}")
        lines.append("Already completed: unknown")
        lines.append(f"New jobs to launch: {unique} planned")
        if status_note:
            lines.append(status_note)
    else:
        lines.append(f"Already completed: {sum(status == 'completed' for status in statuses.values())}")
        lines.append(f"Currently running: {sum(status == 'running' for status in statuses.values())}")
        lines.append(f"Failed, will relaunch: {sum(status == 'failed' for status in statuses.values())}")
        lines.append(f"New jobs to launch: {sum(status == 'missing' for status in statuses.values())}")
    lines.append("")
    for bucket in sorted(counts):
        lines.append(f"{bucket}: {counts[bucket]}")
    lines.append(f"Unique training jobs: {unique}")
    return "\n".join(lines)


def format_summary(plan):
    counts = {}
    kind_counts = {}
    for job in plan["jobs"]:
        bucket = _bucket(job)
        counts[bucket] = counts.get(bucket, 0) + 1
        kind_counts[bucket] = kind_counts.get(bucket, {})
        kind_counts[bucket][job["kind"]] = kind_counts[bucket].get(job["kind"], 0) + 1
    lines = ["Batch 1"]
    for bucket in sorted(counts):
        lines.append(f"{bucket:<8} {counts[bucket]}")
    lines.append(f"{'TOTAL':<8} {len(plan['jobs'])}")
    shared = [job for job in plan["jobs"] if len(job["memberships"]) > 1 or job["aliases"]]
    lines.append("")
    lines.append(f"Shared or aliased configs: {len(shared)}")
    for job in shared:
        roles = [
            f"P{membership['problem']}{membership['subpart']}:{membership['config_role']}"
            for membership in job["memberships"]
        ]
        lines.append(f"  {job['label']}  <- {', '.join(roles)}")
    fits = plan["fits"]
    lines.extend(
        [
            "",
            "Fits from the supplied sweeps",
            f"  P1c all-six LR at 4.9152B: {fits['p1c_all6']['predicted_lr']:.8g}",
            f"  P1c large-three LR at 4.9152B: {fits['p1c_large3']['predicted_lr']:.8g}",
            f"  P1d Hyperball LR at 1.2288B: {fits['p1d_hyperball']['predicted_lr']:.8g}",
            (
                f"  P2c product at 2.4576B: {fits['p2c_product']['predicted_product']:.8g}"
                f"  WD at LR {PRODUCT_TARGET_LR:g}: {fits['p2c_product']['predicted_wd']:.8g}"
            ),
            (
                "  P1 153.6M best sampled LR: "
                f"{fits['p1_1536m_best_sampled']['learning_rate']:.8g}"
            ),
            (
                "  P2 153.6M best joint pair: "
                f"LR {fits['p2_1536m_best_joint']['learning_rate']:.8g} "
                f"WD {fits['p2_1536m_best_joint']['weight_decay']:.8g} "
                f"({fits['p2_1536m_best_joint']['run_id']})"
            ),
            f"Supplied runs referenced instead of retrained: {len(plan['references'])}",
            f"Optional P2c transfer scheduled: {plan['optional_p2c_transfer']['scheduled']}",
        ]
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the manifest and do not submit jobs.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit Batch 1. This is the default when --dry-run is omitted.",
    )
    parser.add_argument(
        "--include-p2c-transfer",
        action="store_true",
        help="Retained for compatibility. The Problem 2c transfer is already included.",
    )
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    args = parser.parse_args(argv)
    if args.execute and args.dry_run:
        parser.error("pass only one of --dry-run and --execute")
    if args.max_parallel < 1:
        parser.error("--max-parallel must be positive")
    plan = build_batch1(include_p2c_transfer=True)
    paths = write_manifest(plan, args.manifest_dir)
    execute = not args.dry_run
    statuses = None
    status_note = None
    if execute:
        try:
            statuses = completion_status(plan, check_volume=True)
        except Exception as exc:
            print("Refusing to launch Batch 1.")
            print("Completion status could not be checked, so finished runs might be duplicated.")
            print(f"{type(exc).__name__}: {exc}")
            return 1
    else:
        try:
            statuses = completion_status(plan, check_volume=False)
        except Exception as exc:
            status_note = f"Completion check unavailable: {type(exc).__name__}: {exc}"
    print(format_operator_summary(plan, statuses, status_note))
    print()
    if args.dry_run:
        print(format_summary(plan))
        print()
        for job in plan["jobs"]:
            hyperparameters = job["hyperparameters"]
            print(
                f"{job['label']}  kind={job['kind']}  "
                f"lr={fmt_float(hyperparameters['peak_lr'])}  "
                f"wd={fmt_float(hyperparameters['weight_decay'])}  "
                f"batch={hyperparameters['batch_size']}  "
                f"width={hyperparameters['width']}  depth={hyperparameters['depth']}  "
                f"opt={hyperparameters['optimizer']}"
            )
        print()
    print(f"Wrote {paths[0]}")
    print(f"Wrote {paths[1]}")
    print(f"Wrote {paths[2]}")
    if not execute:
        print("Dry run only. Run without --dry-run to submit these jobs.")
        return 0
    pending = [
        job
        for job in plan["jobs"]
        if statuses[job["label"]] in {"missing", "failed"}
    ]
    for label, status in statuses.items():
        if status == "failed":
            print(f"Relaunching failed run {label}")
        elif status == "running":
            print(f"Leaving running run {label}")
    completed = sum(status == "completed" for status in statuses.values())
    if completed:
        print(f"Skipping {completed} completed runs.")
    if not pending:
        print("Nothing to submit.")
        return 0
    from experiments.a2.launch_batch1 import launch

    launch({**plan, "jobs": pending}, max_parallel=args.max_parallel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
