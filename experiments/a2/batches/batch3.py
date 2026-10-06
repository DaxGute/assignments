"""Problem 3.2(c): momentum ablation at the best measured LR-WD pairs.

Batch 3 is the last required GPU stage. It does not launch until completed
Problem 3.2 runs exist for B=8 and B=256. Batch 1 and Batch 2 are not imported.
The result store is W&B config and tags.

Metadata contract expected from earlier Problem 3.2 runs
-------------------------------------------------------
Preferred: top-level W&B config fields, or the same fields inside one of
``experiment_metadata``, ``experiment``, ``metadata``, or ``a2_metadata``.

Also accepted: tags ``a2.<key>.<value>``, ``<key>=<value>``, or ``<key>:<value>``.

Fields:
    assignment = "a2"
    problem = "3.2"
    subpart
    batch                  # "batch1" / "batch2", not the sequence batch size
    family
    config_role
    batch_size
    peak_lr                # aliases: learning_rate, optim_lr
    weight_decay
    beta1
    beta2
    token_budget           # alias: train_tokens
    final_val_loss         # summary.final_val_loss, else summary.val_loss
    status                 # alias: run.state; "finished" counts as completed

A source run is eligible only when it is a completed measured LR-WD run on the
Problem 3.2 language-model setup (default d8, 614.4M tokens, linear schedule,
1% warmup, AdamW, beta1=0.9, beta2=0.95). Momentum ablations are ignored.
The winner is the lowest measured final validation loss. Identical LR-WD pairs
are deduplicated first. Numerical ties break by smaller peak LR, then smaller
weight decay, then lexicographically smallest run id.

If Batch 1 / Batch 2 log these fields under different names, change
``METADATA_CONTAINERS`` and ``FIELD_ALIASES`` only.

There is no Batch 4. Problem 3.2(d) and Problem 4.2(e) are analysis.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from data import CONTEXT_LENGTH
from metric_logging import AFTER_BACKWARD, MetricLogger
from module_rms_logging import log_module_rms
from train import TrainConfig, checked_train_config, training_run_name
from utils import WANDB_ENTITY, WANDB_PROJECT


ASSIGNMENT = "a2"
PROBLEM = "3.2"
SUBPART = "c"
BATCH_NAME = "batch3"
FAMILY = "momentum_ablation"
CONFIG_ROLE = "momentum_sweep"
EXPERIMENT_KEY = "a2-b3-p32c"

SOURCE_BATCH_SIZES = (8, 256)
# No Problem 3.2(c) grid is defined in the starter repo. Problem 3.1(d) says
# to include beta1=0, beta1=0.9, and values closer to 1. Six values times the
# two batch sizes match the suggested 12-run budget.
BETA1_GRID = (0.0, 0.8, 0.9, 0.95, 0.98, 0.99)
BETA2 = 0.95
SOURCE_BETA1 = 0.9
WARMUP_PERCENT = 0.01
LR_SCHEDULE = "linear"
OPTIMIZER_NAME = "adamw"
MODEL_NAME = "a2-d8"
COURSE_MODEL_NAMES = {"a2-d8", "d8"}
NUM_TRAIN_SEQUENCES = 600_000
TOKEN_BUDGET = NUM_TRAIN_SEQUENCES * CONTEXT_LENGTH
MICROBATCH_SEQUENCE_CAP = 64
MAX_PARALLEL_RUNS = 2

TIE_ATOL = 1e-6
DUPLICATE_LOSS_ATOL = 1e-4
SETUP_ATOL = 1e-8

PROJECT_PATH = f"{WANDB_ENTITY}/{WANDB_PROJECT}"
METADATA_CONTAINERS = (
    "experiment_metadata",
    "experiment",
    "metadata",
    "a2_metadata",
)
FIELD_ALIASES = {
    "assignment": ("assignment",),
    "problem": ("problem",),
    "subpart": ("subpart",),
    "batch": ("batch", "experiment_batch"),
    "family": ("family",),
    "config_role": ("config_role",),
    "batch_size": ("batch_size",),
    "peak_lr": ("peak_lr", "learning_rate", "optim_lr"),
    "weight_decay": ("weight_decay",),
    "beta1": ("beta1",),
    "beta2": ("beta2",),
    "token_budget": ("token_budget",),
    "lr_schedule": ("lr_schedule",),
    "warmup_percent": ("warmup_percent",),
    "model_name": ("model_name",),
    "optimizer_name": ("optimizer_name", "optimizer"),
    "momentum_role": ("momentum_role",),
    "status": ("status",),
    "num_train_sequences": ("num_train_sequences",),
}
FLOAT_FIELDS = {
    "peak_lr",
    "weight_decay",
    "beta1",
    "beta2",
    "warmup_percent",
    "token_budget",
    "batch_size",
    "num_train_sequences",
}
FINISHED_STATES = {"finished", "completed", "success"}
MOMENTUM_MARKERS = ("momentum",)
TAG_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
TAG_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
RUN_PREFIX = "a2."

if TOKEN_BUDGET != 614_400_000:
    raise RuntimeError(
        f"Expected the Problem 3.2 token budget of 614.4M, got {TOKEN_BUDGET}."
    )


class Batch3Error(RuntimeError):
    def __init__(self, message: str, *, kind: str = "error"):
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class FieldValue:
    source: str
    text: str
    number: float | None


@dataclass(frozen=True)
class SourceRun:
    run_id: str
    name: str
    state: str
    assignment: str | None
    problem: str | None
    subpart: str | None
    batch: str | None
    family: str | None
    config_role: str | None
    batch_size: int | None
    peak_lr: float | None
    weight_decay: float | None
    beta1: float | None
    beta2: float | None
    token_budget: int | None
    lr_schedule: str | None
    warmup_percent: float | None
    model_name: str | None
    optimizer_name: str | None
    momentum_role: str | None
    status: str | None
    final_val_loss: float | None
    width: int | None = None
    depth: int | None = None
    parameterization: str | None = None
    inconsistencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class MeasuredConfig:
    batch_size: int
    peak_lr: float
    weight_decay: float
    source_run_id: str
    source_subpart: str | None
    source_batch: str | None
    source_final_val_loss: float
    source_name: str
    deduped_run_ids: tuple[str, ...]


@dataclass
class BatchSelection:
    batch_size: int
    eligible: tuple[SourceRun, ...]
    unique_pairs: tuple[SourceRun, ...]
    winner: SourceRun
    tie_break_used: bool
    tied_run_ids: tuple[str, ...]
    deduped_groups: tuple[tuple[str, ...], ...]
    excluded: tuple[tuple[str, str], ...] = ()

    def measured(self) -> MeasuredConfig:
        winner = self.winner
        deduped = next(
            (group for group in self.deduped_groups if winner.run_id in group),
            (winner.run_id,),
        )
        return MeasuredConfig(
            batch_size=self.batch_size,
            peak_lr=float(winner.peak_lr),
            weight_decay=float(winner.weight_decay),
            source_run_id=winner.run_id,
            source_subpart=winner.subpart,
            source_batch=winner.batch,
            source_final_val_loss=float(winner.final_val_loss),
            source_name=winner.name,
            deduped_run_ids=tuple(sorted(deduped)),
        )


@dataclass(frozen=True)
class MomentumJob:
    config: TrainConfig
    batch_size: int
    beta1: float
    peak_lr: float
    weight_decay: float
    beta2: float
    momentum_role: str
    run_name_suffix: str
    training_name: str
    metadata: dict
    source: MeasuredConfig


@dataclass
class Batch3Plan:
    selections: dict[int, BatchSelection]
    jobs: tuple[MomentumJob, ...]
    dropped_duplicates: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def configs(self) -> list[TrainConfig]:
        return [job.config for job in self.jobs]


def same_number(left: float, right: float, *, atol: float = SETUP_ATOL) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=atol)


def hp_key(value: float) -> float:
    return float(f"{float(value):.8e}")


def format_hp(value: float) -> str:
    return f"{float(value):.6g}"


def format_loss(value: float) -> str:
    return f"{float(value):.6f}"


def format_beta1(value: float) -> str:
    return f"{float(value):.3f}"


def plain_data(value):
    if isinstance(value, Mapping):
        return {str(key): plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_data(item) for item in value]
    return value


def _is_blank(value) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _number(value) -> float | None:
    if isinstance(value, bool) or _is_blank(value):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(number):
        return None
    return number


def _text(value) -> str | None:
    if _is_blank(value):
        return None
    if isinstance(value, float) and math.isfinite(value):
        return f"{value:.12g}"
    return str(value).strip()


def normalize_problem(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip().lower()
    for prefix in ("problem", "p"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip(" _-:")
    return text


def normalize_token(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().lower()


def _config_layers(config: Mapping) -> list[tuple[str, Mapping]]:
    layers = [("config", config)]
    for name in METADATA_CONTAINERS:
        nested = config.get(name)
        if isinstance(nested, Mapping):
            layers.append((f"config.{name}", nested))
    return layers


def _tag_fields(tags: Sequence[str]) -> dict[str, list[FieldValue]]:
    found: dict[str, list[FieldValue]] = defaultdict(list)
    known = set(FIELD_ALIASES)
    for tag in tags:
        key = None
        raw = None
        source = f"tag:{tag}"
        if tag.startswith(RUN_PREFIX):
            rest = tag[len(RUN_PREFIX) :]
            parsed_key, sep, parsed_value = rest.partition(".")
            if sep and parsed_key in known:
                key, raw = parsed_key, parsed_value
        else:
            for separator in ("=", ":"):
                if separator not in tag:
                    continue
                parsed_key, parsed_value = tag.split(separator, 1)
                if parsed_key in known and parsed_value != "":
                    key, raw = parsed_key, parsed_value
                    break
        if key is None or raw is None:
            continue
        found[key].append(FieldValue(source, raw, _number(raw)))
    return found


def _collect_fields(config: Mapping, tags: Sequence[str]) -> dict[str, list[FieldValue]]:
    collected: dict[str, list[FieldValue]] = defaultdict(list)
    layers = _config_layers(config)
    for field_name, aliases in FIELD_ALIASES.items():
        for layer_name, layer in layers:
            for alias in aliases:
                if alias not in layer or _is_blank(layer.get(alias)):
                    continue
                raw_value = layer.get(alias)
                text = _text(raw_value)
                if text is None:
                    continue
                collected[field_name].append(
                    FieldValue(f"{layer_name}.{alias}", text, _number(raw_value))
                )
        for item in _tag_fields(tags).get(field_name, ()):
            collected[field_name].append(item)
    return collected


def _unique_field(
    values: Sequence[FieldValue],
    *,
    numeric: bool,
) -> tuple[FieldValue | None, str | None]:
    usable = []
    for item in values:
        if numeric:
            if item.number is None:
                return None, f"{item.source} is not numeric ({item.text!r})"
            if any(same_number(item.number, previous.number) for previous in usable):
                continue
        else:
            if any(item.text.lower() == previous.text.lower() for previous in usable):
                continue
        usable.append(item)
    if not usable:
        return None, None
    if len(usable) > 1:
        rendered = ", ".join(f"{item.source}={item.text}" for item in usable)
        return None, f"conflicting values: {rendered}"
    return usable[0], None


def _context_length(config: Mapping) -> int | None:
    dataset = config.get("train_dataset")
    if isinstance(dataset, Mapping):
        length = _number(dataset.get("context_length"))
        if length is not None:
            return int(round(length))
    length = _number(config.get("context_length"))
    if length is None:
        return None
    return int(round(length))


def _loss_from_summary(summary: Mapping) -> float | None:
    for key in ("final_val_loss", "val_loss"):
        parsed = _number(summary.get(key))
        if parsed is not None:
            return parsed
    return None


def _loss_from_history(run) -> float | None:
    scan = getattr(run, "scan_history", None)
    if scan is None:
        return None
    latest_step = -1
    latest_value = None
    for row in scan(keys=["optimizer_step", "val_loss", "final_val_loss"]):
        value = _number(row.get("final_val_loss"))
        if value is None:
            value = _number(row.get("val_loss"))
        if value is None:
            continue
        step_number = _number(row.get("optimizer_step"))
        step = latest_step + 1 if step_number is None else int(step_number)
        if step >= latest_step:
            latest_step = step
            latest_value = value
    return latest_value


def _source_from_record(record: Mapping) -> SourceRun:
    return SourceRun(
        run_id=str(record.get("run_id") or ""),
        name=str(record.get("name") or ""),
        state=str(record.get("state") or ""),
        assignment=record.get("assignment"),
        problem=record.get("problem"),
        subpart=record.get("subpart"),
        batch=record.get("batch"),
        family=record.get("family"),
        config_role=record.get("config_role"),
        batch_size=record.get("batch_size"),
        peak_lr=record.get("peak_lr"),
        weight_decay=record.get("weight_decay"),
        beta1=record.get("beta1"),
        beta2=record.get("beta2"),
        token_budget=record.get("token_budget"),
        lr_schedule=record.get("lr_schedule"),
        warmup_percent=record.get("warmup_percent"),
        model_name=record.get("model_name"),
        optimizer_name=record.get("optimizer") or record.get("optimizer_name"),
        momentum_role=record.get("momentum_role"),
        status=record.get("status"),
        final_val_loss=record.get("final_val_loss"),
        width=record.get("width"),
        depth=record.get("depth"),
        parameterization=record.get("parameterization"),
        inconsistencies=tuple(record.get("inconsistencies") or ()),
    )


def _record_from_source(run: SourceRun) -> dict:
    finished = _finished(run)
    inconsistencies = list(run.inconsistencies)
    return {
        "run_id": run.run_id,
        "name": run.name,
        "state": run.state,
        "assignment": run.assignment,
        "problem": run.problem,
        "subpart": run.subpart,
        "batch": run.batch,
        "family": run.family,
        "config_role": run.config_role,
        "batch_size": run.batch_size,
        "peak_lr": run.peak_lr,
        "weight_decay": run.weight_decay,
        "beta1": run.beta1,
        "beta2": run.beta2,
        "token_budget": run.token_budget,
        "lr_schedule": run.lr_schedule,
        "warmup_percent": run.warmup_percent,
        "model_name": run.model_name,
        "optimizer": run.optimizer_name,
        "momentum_role": run.momentum_role,
        "parameterization": run.parameterization,
        "width": run.width,
        "depth": run.depth,
        "final_val_loss": run.final_val_loss,
        "status": run.status,
        "inconsistencies": inconsistencies,
        "completed": finished is True and run.final_val_loss is not None and not inconsistencies,
    }


def parse_source_run(run) -> SourceRun:
    from experiments.a2.helpers.results import describe_run

    return _source_from_record(describe_run(run))


def _is_momentum_ablation(run: SourceRun) -> bool:
    if run.subpart in {"c", "3.2c"}:
        return True
    if run.batch == BATCH_NAME:
        return True
    if run.momentum_role in {"momentum", "no_momentum"}:
        return True
    markers = (run.family, run.config_role)
    return any(
        marker is not None and any(token in marker for token in MOMENTUM_MARKERS)
        for marker in markers
    )


def _finished(run: SourceRun) -> bool | None:
    """Return True/False, or None when recorded status disagrees with run state."""
    state = normalize_token(run.state)
    status = normalize_token(run.status)
    state_known = state not in {None, ""}
    status_known = status not in {None, ""}
    state_done = state in FINISHED_STATES
    status_done = status in FINISHED_STATES
    if state_known and status_known and state_done != status_done:
        return None
    if state_known:
        return state_done
    if status_known:
        return status_done
    return False


def _reject_if_present(problems: list[str], actual, expected, label: str) -> None:
    if actual is None:
        return
    if isinstance(expected, float):
        if not same_number(actual, expected):
            problems.append(f"{label}={format_hp(actual)}, expected {format_hp(expected)}")
        return
    if actual != expected:
        problems.append(f"{label}={actual}, expected {expected}")


def _setup_problems(run: SourceRun) -> list[str]:
    """Reasons a finished run cannot be a measured Problem 3.2 LR-WD source."""
    problems = []
    _reject_if_present(problems, run.assignment, ASSIGNMENT, "assignment")
    _reject_if_present(problems, run.model_name, MODEL_NAME, "model_name")
    _reject_if_present(problems, run.lr_schedule, LR_SCHEDULE, "lr_schedule")
    _reject_if_present(problems, run.optimizer_name, OPTIMIZER_NAME, "optimizer_name")
    if run.warmup_percent is not None and not same_number(run.warmup_percent, WARMUP_PERCENT):
        problems.append(
            f"warmup_percent={format_hp(run.warmup_percent)}, expected {WARMUP_PERCENT}"
        )
    if run.token_budget is None:
        problems.append("token_budget is missing")
    elif run.token_budget != TOKEN_BUDGET:
        problems.append(
            f"token_budget={run.token_budget}, expected {TOKEN_BUDGET} (614.4M)"
        )
    if run.beta2 is None:
        problems.append("beta2 is missing")
    elif not same_number(run.beta2, BETA2):
        problems.append(f"beta2={format_hp(run.beta2)}, expected {BETA2}")
    if run.beta1 is None:
        problems.append("beta1 is missing")
    elif not same_number(run.beta1, SOURCE_BETA1):
        problems.append(
            f"beta1={format_hp(run.beta1)}, expected the LR-WD measurement "
            f"value {SOURCE_BETA1}"
        )
    if run.peak_lr is None:
        problems.append("peak_lr is missing")
    if run.weight_decay is None:
        problems.append("weight_decay is missing")
    return problems


def _query_runs(api, filters: dict) -> list:
    return list(api.runs(PROJECT_PATH, filters=filters))


def load_p32_source_runs():
    from experiments.a2.helpers.results import ResultsError, query_a2_runs

    try:
        described = query_a2_runs()
    except ResultsError as exc:
        raise Batch3Error(
            "Batch 3 cannot run yet:\n"
            "required Batch 1 or Batch 2 source experiments are missing or incomplete.\n\n"
            f"{exc}",
            kind="missing",
        ) from exc
    return tuple(_source_from_record(record) for record in described)


def select_measured_lr_wd(runs: Sequence[SourceRun], batch_size: int) -> BatchSelection:
    from experiments.a2.helpers.results import ResultsError, get_best_run

    measurements = [
        _record_from_source(run) if isinstance(run, SourceRun) else dict(run) for run in runs
    ]
    try:
        chosen = get_best_run(measurements, batch_size=batch_size)
    except ResultsError as exc:
        raise Batch3Error(str(exc), kind=exc.kind) from exc
    originals = {run.run_id: run for run in runs if isinstance(run, SourceRun)}

    def restore(item):
        if isinstance(item, SourceRun):
            return item
        return originals.get(item.get("run_id")) or _source_from_record(item)

    def ordered(items):
        restored = [restore(item) for item in items]
        return tuple(
            sorted(
                restored,
                key=lambda run: (
                    float(run.final_val_loss),
                    float(run.peak_lr),
                    float(run.weight_decay),
                    run.run_id,
                ),
            )
        )

    return BatchSelection(
        batch_size=batch_size,
        eligible=ordered(chosen["eligible"]),
        unique_pairs=ordered(chosen["unique_pairs"]),
        winner=restore(chosen["winner"]),
        tie_break_used=bool(chosen["tie_break_used"]),
        tied_run_ids=tuple(chosen["tied_run_ids"]),
        deduped_groups=tuple(chosen["deduped_groups"]),
        excluded=tuple(chosen["excluded"]),
    )


def _selection_from_eligible(
    batch_size: int,
    eligible: tuple[SourceRun, ...],
    excluded: tuple[tuple[str, str], ...],
) -> BatchSelection:
    groups: dict[tuple[float, float], list[SourceRun]] = defaultdict(list)
    for run in eligible:
        groups[(hp_key(run.peak_lr), hp_key(run.weight_decay))].append(run)

    representatives = []
    deduped_groups = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda run: (run.run_id, run.name))
        losses = [float(run.final_val_loss) for run in members]
        if max(losses) - min(losses) > DUPLICATE_LOSS_ATOL:
            rendered = ", ".join(
                f"{run.run_id} loss={format_loss(run.final_val_loss)}" for run in members
            )
            peak_lr = members[0].peak_lr
            weight_decay = members[0].weight_decay
            raise Batch3Error(
                "ERROR:\n"
                "Cannot construct Batch 3.\n\n"
                "Multiple completed runs use the same LR-WD pair but disagree "
                "on final validation loss:\n"
                f"problem = {PROBLEM}\n"
                f"batch_size = {batch_size}\n"
                f"peak_lr = {format_hp(peak_lr)}\n"
                f"weight_decay = {format_hp(weight_decay)}\n"
                f"  {rendered}",
                kind="inconsistent",
            )
        representatives.append(members[0])
        if len(members) > 1:
            deduped_groups.append(tuple(run.run_id for run in members))

    best_loss = min(float(run.final_val_loss) for run in representatives)
    tied = [
        run
        for run in representatives
        if same_number(run.final_val_loss, best_loss, atol=TIE_ATOL)
    ]
    winner = min(
        tied,
        key=lambda run: (float(run.peak_lr), float(run.weight_decay), run.run_id),
    )
    ordered_eligible = tuple(
        sorted(
            eligible,
            key=lambda run: (
                float(run.final_val_loss),
                float(run.peak_lr),
                float(run.weight_decay),
                run.run_id,
            ),
        )
    )
    ordered_pairs = tuple(
        sorted(
            representatives,
            key=lambda run: (
                float(run.final_val_loss),
                float(run.peak_lr),
                float(run.weight_decay),
                run.run_id,
            ),
        )
    )
    return BatchSelection(
        batch_size=batch_size,
        eligible=ordered_eligible,
        unique_pairs=ordered_pairs,
        winner=winner,
        tie_break_used=len(tied) > 1,
        tied_run_ids=tuple(sorted(run.run_id for run in tied)),
        deduped_groups=tuple(deduped_groups),
        excluded=excluded,
    )


def get_best_p32_config(batch_size: int, runs: Sequence[SourceRun] | None = None) -> MeasuredConfig:
    """Return the best measured Problem 3.2 LR-WD pair at ``batch_size``."""
    if runs is None:
        runs = load_p32_source_runs()
    return select_measured_lr_wd(runs, batch_size).measured()


def resolve_source_selections(
    runs: Sequence[SourceRun],
) -> dict[int, BatchSelection]:
    selections = {}
    errors = []
    missing = []
    for batch_size in SOURCE_BATCH_SIZES:
        try:
            selections[batch_size] = select_measured_lr_wd(runs, batch_size)
        except Batch3Error as exc:
            errors.append(exc)
            if exc.kind == "missing":
                missing.append(batch_size)
    if not errors:
        return selections
    if len(missing) == len(errors) and missing:
        lines = [
            "Batch 3 cannot run yet:",
            "required Batch 1 or Batch 2 source experiments are missing or incomplete.",
            "",
            "No completed measured LR-WD configurations were found for:",
            f"problem = {PROBLEM}",
        ]
        lines.extend(f"batch_size = {batch_size}" for batch_size in missing)
        lines.append("")
        if missing == [8]:
            lines.append("Run Batch 1 first.")
        elif missing == [256]:
            lines.append("Run Batch 2 first.")
        else:
            lines.append("Run Batch 1 and Batch 2 first.")
        details = []
        for exc in errors:
            body = str(exc).split("Related runs that were not eligible:", 1)
            if len(body) == 2:
                details.append(body[1].strip())
        if details:
            lines.extend(["", "Related runs that were not eligible:"])
            for detail in details:
                lines.extend(detail.splitlines())
        raise Batch3Error("\n".join(lines), kind="missing")
    raise Batch3Error(
        "ERROR:\nCannot construct Batch 3.\n\n"
        + "\n\n".join(str(exc).split("\n", 3)[-1] for exc in errors)
    )


def num_micro_batches_for(batch_size: int) -> int:
    microbatch_sequences = min(batch_size, MICROBATCH_SEQUENCE_CAP)
    if microbatch_sequences <= 0 or batch_size % microbatch_sequences != 0:
        raise Batch3Error(
            f"batch_size={batch_size} cannot be split into microbatches of "
            f"min(B, {MICROBATCH_SEQUENCE_CAP}) sequences."
        )
    return batch_size // microbatch_sequences


def momentum_role_for(beta1: float) -> str:
    if same_number(beta1, 0.0):
        return "no_momentum"
    return "momentum"


def run_name_suffix(batch_size: int, beta1: float) -> str:
    return (
        f"{EXPERIMENT_KEY}-momentum-"
        f"b{batch_size:03d}-b1{format_beta1(beta1)}"
    )


def _format_meta_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return format(value, ".12g")
    return str(value)


def metadata_tags(metadata: Mapping) -> tuple[str, ...]:
    tags = []
    for key in sorted(metadata):
        value = metadata[key]
        if value is None or value == "":
            continue
        tag = f"{RUN_PREFIX}{key}.{_format_meta_value(value)}"
        if TAG_KEY_RE.fullmatch(key) is None or TAG_RE.fullmatch(tag) is None:
            raise Batch3Error(f"Metadata tag {tag!r} is not a valid W&B tag.")
        tags.append(tag)
    return tuple(tags)


def job_metadata(job_fields: Mapping) -> dict:
    return {
        "assignment": ASSIGNMENT,
        "batch": BATCH_NAME,
        "problem": PROBLEM,
        "subpart": SUBPART,
        "family": FAMILY,
        "config_role": CONFIG_ROLE,
        "batch_size": int(job_fields["batch_size"]),
        "peak_lr": float(job_fields["peak_lr"]),
        "weight_decay": float(job_fields["weight_decay"]),
        "beta1": float(job_fields["beta1"]),
        "beta2": float(job_fields["beta2"]),
        "token_budget": TOKEN_BUDGET,
        "momentum_role": job_fields["momentum_role"],
        "source_run_id": job_fields["source_run_id"],
        "source_subpart": job_fields["source_subpart"] or "unknown",
        "source_batch": job_fields["source_batch"] or "unknown",
        "source_final_val_loss": float(job_fields["source_final_val_loss"]),
        "source_lr_wd_run_id": job_fields["source_run_id"],
        "source_lr_wd_batch": job_fields["source_batch"] or "unknown",
        "source_lr_wd_loss": float(job_fields["source_final_val_loss"]),
        "lr_schedule": LR_SCHEDULE,
        "warmup_percent": WARMUP_PERCENT,
        "model_name": MODEL_NAME,
        "optimizer_name": OPTIMIZER_NAME,
        "optimizer": OPTIMIZER_NAME,
        "width": 512,
        "depth": 8,
        "parameterization": "baseline",
        "seed": 42,
    }


def promote_experiment_metadata(context):
    """Metric hook. Structured fields are written from tags at startup."""
    del context
    return {}


PROMOTED_METADATA_KEYS = set(FIELD_ALIASES) | {
    "source_run_id",
    "source_subpart",
    "source_batch",
    "source_final_val_loss",
    "source_lr_wd_run_id",
    "source_lr_wd_batch",
    "source_lr_wd_loss",
    "momentum_role",
    "token_budget",
}
INTEGER_METADATA_KEYS = {"batch_size", "token_budget", "num_train_sequences"}


def _promote_experiment_metadata_setup(context) -> None:
    import wandb

    if wandb.run is None:
        return
    metadata = {}
    for tag in context.config.wandb_tags:
        if not str(tag).startswith(RUN_PREFIX):
            continue
        key, sep, raw = str(tag)[len(RUN_PREFIX) :].partition(".")
        if not sep or key not in PROMOTED_METADATA_KEYS:
            continue
        number = _number(raw)
        if key in INTEGER_METADATA_KEYS and number is not None:
            metadata[key] = int(round(number))
        elif key in FLOAT_FIELDS or key in {"source_final_val_loss", "source_lr_wd_loss"}:
            metadata[key] = number if number is not None else raw
        else:
            metadata[key] = raw
    if not metadata:
        return
    existing = set(wandb.config.keys())
    fresh = {key: value for key, value in metadata.items() if key not in existing}
    if fresh:
        wandb.config.update(fresh, allow_val_change=False)


promote_experiment_metadata.setup = _promote_experiment_metadata_setup


def experiment_metric_loggers() -> tuple[MetricLogger, ...]:
    return (
        MetricLogger(event=AFTER_BACKWARD, fn=log_module_rms),
        MetricLogger(event=AFTER_BACKWARD, fn=promote_experiment_metadata),
    )


def build_momentum_jobs(
    selections: Mapping[int, BatchSelection],
) -> tuple[MomentumJob, ...]:
    jobs = []
    seen = set()
    for batch_size in SOURCE_BATCH_SIZES:
        source = selections[batch_size].measured()
        for beta1 in BETA1_GRID:
            role = momentum_role_for(beta1)
            suffix = run_name_suffix(batch_size, beta1)
            metadata = job_metadata(
                {
                    "batch_size": batch_size,
                    "peak_lr": source.peak_lr,
                    "weight_decay": source.weight_decay,
                    "beta1": beta1,
                    "beta2": BETA2,
                    "momentum_role": role,
                    "source_run_id": source.source_run_id,
                    "source_subpart": source.source_subpart,
                    "source_batch": source.source_batch,
                    "source_final_val_loss": source.source_final_val_loss,
                }
            )
            identity = (
                batch_size,
                hp_key(source.peak_lr),
                hp_key(source.weight_decay),
                hp_key(beta1),
                hp_key(BETA2),
                TOKEN_BUDGET,
                LR_SCHEDULE,
            )
            if identity in seen:
                continue
            seen.add(identity)
            role_tag = "no-momentum" if role == "no_momentum" else "momentum"
            from experiments.a2.modal_launcher import config as a2_config

            config = a2_config(
                tokens=TOKEN_BUDGET,
                batch=batch_size,
                learning_rate=source.peak_lr,
                weight_decay=source.weight_decay,
                beta1=beta1,
                beta2=BETA2,
                save_model=False,
                run_name_suffix=suffix,
                wandb_tags=(
                    EXPERIMENT_KEY,
                    "a2",
                    "batch3",
                    FAMILY.replace("_", "-"),
                    role_tag,
                    *metadata_tags(metadata),
                ),
                experiment_metadata=metadata,
            )
            checked_train_config(config)
            jobs.append(
                MomentumJob(
                    config=config,
                    batch_size=batch_size,
                    beta1=beta1,
                    peak_lr=source.peak_lr,
                    weight_decay=source.weight_decay,
                    beta2=BETA2,
                    momentum_role=role,
                    run_name_suffix=suffix,
                    training_name=training_run_name(config),
                    metadata=metadata,
                    source=source,
                )
            )
    return tuple(jobs)


def build_plan(runs: Sequence[SourceRun] | None = None) -> Batch3Plan:
    if runs is None:
        runs = load_p32_source_runs()
    selections = resolve_source_selections(runs)
    jobs = build_momentum_jobs(selections)
    proposed = len(SOURCE_BATCH_SIZES) * len(BETA1_GRID)
    notes = []
    if len(BETA1_GRID) * len(SOURCE_BATCH_SIZES) != 12:
        notes.append(
            "The starter repo does not define a Problem 3.2(c) beta1 grid. "
            f"This sweep uses {len(BETA1_GRID)} beta1 values "
            f"{', '.join(format_beta1(value) for value in BETA1_GRID)} "
            f"at batch sizes {', '.join(str(size) for size in SOURCE_BATCH_SIZES)}."
        )
    return Batch3Plan(
        selections=selections,
        jobs=jobs,
        dropped_duplicates=proposed - len(jobs),
        notes=tuple(notes),
    )


def _format_source_block(selection: BatchSelection) -> list[str]:
    winner = selection.winner
    lines = [
        f"B={selection.batch_size}",
        "Eligible measured runs:",
    ]
    if not selection.eligible:
        lines.append("  (none)")
    for run in selection.eligible:
        lines.append(
            "  "
            f"peak_lr={format_hp(run.peak_lr)}  "
            f"wd={format_hp(run.weight_decay)}  "
            f"loss={format_loss(run.final_val_loss)}  "
            f"run={run.run_id}  "
            f"subpart={run.subpart or '-'}  "
            f"batch={run.batch or '-'}"
        )
    if selection.deduped_groups:
        lines.append("Deduplicated identical LR-WD pairs (kept the smallest run id):")
        for group in selection.deduped_groups:
            lines.append(f"  {', '.join(group)}")
    else:
        lines.append("Deduplicated identical LR-WD pairs: none")
    lines.extend(
        [
            "Selected:",
            f"  peak LR: {format_hp(winner.peak_lr)}",
            f"  WD: {format_hp(winner.weight_decay)}",
            f"  source run: {winner.run_id}",
            f"  source subpart: {winner.subpart or '-'}",
            f"  source batch: {winner.batch or '-'}",
            f"  final val loss: {format_loss(winner.final_val_loss)}",
        ]
    )
    if selection.tie_break_used:
        lines.append(
            "  selection: numerically tied final validation loss "
            f"(abs tol {TIE_ATOL:g}); "
            "chose the smaller peak LR, then the smaller weight decay, "
            "then the lexicographically smallest run id"
        )
        lines.append(f"  tied runs: {', '.join(selection.tied_run_ids)}")
    else:
        lines.append("  selection: lowest final validation loss")
    if selection.excluded:
        lines.append("Excluded:")
        for label, reason in selection.excluded:
            lines.append(f"  {label}: {reason}")
    return lines


def render_manifest(plan: Batch3Plan) -> str:
    lines = [
        "A2 Batch 3",
        "============",
        "",
        "Selected source configuration",
        "",
    ]
    for batch_size in SOURCE_BATCH_SIZES:
        lines.extend(_format_source_block(plan.selections[batch_size]))
        lines.append("")
    lines.extend(
        [
            "Tie break: if final validation losses agree within "
            f"{TIE_ATOL:g}, choose the smaller peak LR, then the smaller "
            "weight decay, then the lexicographically smallest run id.",
            "Duplicate runs of one LR-WD pair are collapsed before that comparison "
            f"when their losses agree within {DUPLICATE_LOSS_ATOL:g}.",
            "",
            "Momentum sweep",
            "",
            "Held fixed within each batch size: peak LR, weight decay, and beta2.",
            f"beta2 = {BETA2}",
            f"schedule = {LR_SCHEDULE}",
            f"warmup = {WARMUP_PERCENT:g} (1% of optimizer updates)",
            f"tokens = {TOKEN_BUDGET} (614.4M)",
            f"model = {MODEL_NAME}",
            "microbatch sequences = min(B, 64)",
            "",
            "beta1 grid: " + ", ".join(format_beta1(value) for value in BETA1_GRID),
            "The starter repo does not define a Problem 3.2(c) grid.",
            "Included beta1 = 0 (no-momentum control), beta1 = 0.9 (AdamW default),",
            "beta1 = 0.8 (used elsewhere in this codebase), and 0.95, 0.98, 0.99",
            "(closer to 1, as in Problem 3.1(d)).",
            f"{len(BETA1_GRID)} beta1 values x {len(SOURCE_BATCH_SIZES)} batch sizes "
            f"= {len(BETA1_GRID) * len(SOURCE_BATCH_SIZES)} proposed runs.",
            "",
        ]
    )
    for batch_size in SOURCE_BATCH_SIZES:
        lines.append(f"B={batch_size}")
        for job in plan.jobs:
            if job.batch_size != batch_size:
                continue
            lines.append(
                f"  beta1 = {format_beta1(job.beta1)}  "
                f"momentum_role = {job.momentum_role}  "
                f"peak_lr = {format_hp(job.peak_lr)}  "
                f"wd = {format_hp(job.weight_decay)}"
            )
        lines.append("")
    lines.append("Jobs")
    for job in plan.jobs:
        lines.append(
            f"  {job.run_name_suffix}  "
            f"name={job.training_name}  "
            f"microbatches={job.config.num_micro_batches}"
        )
    lines.append("")
    if plan.dropped_duplicates:
        lines.append(
            f"Removed {plan.dropped_duplicates} duplicate Batch 3 configurations."
        )
    else:
        lines.append("Deduplication removed 0 identical Batch 3 configurations.")
    lines.append(f"Unique GPU jobs: {len(plan.jobs)}")
    if len(plan.jobs) != 12:
        lines.append(
            "Unique GPU jobs is not 12 because the beta1 grid or deduplication "
            "changed the suggested Problem 3.2(c) budget."
        )
    for note in plan.notes:
        lines.append(note)
    lines.extend(
        [
            "",
            "Plot fields: batch_size, beta1, final_val_loss (summary val_loss), "
            "peak_lr, weight_decay.",
            "Compare beta1 = 0 (momentum_role=no_momentum) with beta1 = 0.9 "
            "and the other momentum settings at B=8 and B=256.",
            "No Batch 4. Problem 3.2(d) is analysis of these runs.",
        ]
    )
    return "\n".join(lines)


def plot_fields(run: SourceRun) -> dict:
    """Fields for a beta1-vs-loss plot, one curve per batch size."""
    return {
        "batch_size": run.batch_size,
        "beta1": run.beta1,
        "final_val_loss": run.final_val_loss,
        "peak_lr": run.peak_lr,
        "weight_decay": run.weight_decay,
        "momentum_role": run.momentum_role,
    }


def format_operator_summary(plan: Batch3Plan, statuses: Mapping[str, str] | None = None) -> str:
    lines = [
        "A2 Batch 3",
        "==========",
        "",
        "Prerequisites: OK",
        "",
    ]
    for batch_size in SOURCE_BATCH_SIZES:
        winner = plan.selections[batch_size].winner
        lines.append(
            f"B={batch_size} best: peak_lr={format_hp(winner.peak_lr)}  "
            f"wd={format_hp(winner.weight_decay)}  "
            f"loss={format_loss(winner.final_val_loss)}"
        )
    lines.append("")
    if statuses is None:
        lines.append(f"New jobs to launch: {len(plan.jobs)} planned")
    else:
        values = list(statuses.values())
        lines.append(f"Already completed: {values.count('completed')}")
        lines.append(f"Currently running: {values.count('running')}")
        lines.append(f"Failed, will relaunch: {values.count('failed')}")
        lines.append(f"New jobs to launch: {values.count('missing')}")
    lines.append("")
    lines.append(f"P3.2c: {len(plan.jobs)}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A2 Batch 3: Problem 3.2(c) momentum ablation."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Query Problem 3.2 results and print the Batch 3 manifest.",
    )
    parser.add_argument(
        "--max-parallel-runs",
        "--max-parallel",
        type=int,
        default=MAX_PARALLEL_RUNS,
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.max_parallel_runs < 1:
        print("Batch 3 cannot run:\nmax-parallel must be positive.")
        return 1
    from experiments.a2.helpers.results import ResultsError, classify_named_runs, query_a2_runs

    try:
        described = query_a2_runs()
    except ResultsError as exc:
        print(
            "Batch 3 cannot run yet:\n"
            "required Batch 1 or Batch 2 source experiments are missing or incomplete.\n\n"
            f"{exc}"
        )
        return 1
    runs = tuple(_source_from_record(record) for record in described)
    try:
        plan = build_plan(runs)
    except Batch3Error as exc:
        print(exc)
        return 1
    statuses = classify_named_runs(
        [job.training_name for job in plan.jobs],
        described,
    )
    print(format_operator_summary(plan, statuses))
    print()
    if args.dry_run:
        print(render_manifest(plan))
        print()
        print("Dry run only. Run without --dry-run to submit these jobs.")
        return 0
    pending = [
        job
        for job in plan.jobs
        if statuses[job.training_name] in {"missing", "failed"}
    ]
    for job in plan.jobs:
        status = statuses[job.training_name]
        if status == "failed":
            print(f"Relaunching failed run {job.run_name_suffix}")
        elif status == "running":
            print(f"Leaving running run {job.run_name_suffix}")
    if not pending:
        print("Nothing to submit.")
        return 0
    from experiments.a2.modal_launcher import launch_training_jobs

    launch_training_jobs(
        [job.config for job in pending],
        max_parallel_runs=args.max_parallel_runs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
