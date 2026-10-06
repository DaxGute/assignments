"""Shared A2 result discovery.

Batch 2 and Batch 3 read completed runs through this module. Selection uses
structured metadata, not run names. A missing or ambiguous source set raises
``ResultsError`` instead of substituting a default.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence


ASSIGNMENT = "a2"
METADATA_CONTAINERS = (
    "experiment_metadata",
    "experiment",
    "metadata",
    "a2_metadata",
)
RUN_PREFIX = "a2."

# Requested Problem 3.2 budget. Processed tokens can be slightly smaller when
# the last incomplete accumulation group is dropped.
P32_TOKEN_BUDGET = 614_400_000
P32_MODEL_NAMES = {"a2-d8", "d8"}
P32_WIDTH = 512
P32_DEPTH = 8
P32_BETA1 = 0.9
P32_BETA2 = 0.95
P32_OPTIMIZER = "adamw"
P32_SCHEDULE = "linear"
P32_WARMUP = 0.01
P32_PARAMETERIZATION = "baseline"
PROCESSED_TOKEN_TOLERANCE = 0.001
LOSS_ATOL = 1e-4
TIE_ATOL = 1e-6
NUMBER_ATOL = 1e-8

FINISHED_STATES = {"finished", "completed", "success"}
FAILED_STATES = {"failed", "crashed", "killed", "canceled", "cancelled", "error"}
FIELD_ALIASES = {
    "assignment": ("assignment",),
    "batch": ("batch", "experiment_batch"),
    "problem": ("problem",),
    "subpart": ("subpart",),
    "family": ("family",),
    "config_role": ("config_role",),
    "hypothesis": ("hypothesis",),
    "token_budget": ("token_budget",),
    "batch_size": ("batch_size",),
    "peak_lr": ("peak_lr", "learning_rate", "optim_lr"),
    "weight_decay": ("weight_decay",),
    "beta1": ("beta1",),
    "beta2": ("beta2",),
    "width": ("width",),
    "depth": ("depth",),
    "parameterization": ("parameterization",),
    "optimizer": ("optimizer", "optimizer_name"),
    "seed": ("seed", "model_seed"),
    "lr_schedule": ("lr_schedule",),
    "warmup_percent": ("warmup_percent",),
    "model_name": ("model_name",),
    "momentum_role": ("momentum_role",),
    "status": ("status",),
}
INTEGER_FIELDS = {"token_budget", "batch_size", "width", "depth", "seed"}
FLOAT_FIELDS = {"peak_lr", "weight_decay", "beta1", "beta2", "warmup_percent"}

PARAMETERIZATION_ALIASES = {
    "baseline": "baseline",
    "course_baseline": "baseline",
    "default": "baseline",
    "mup": "mup",
    "µp": "mup",
    "μp": "mup",
    "standard_mup": "mup",
    "standard-mup": "mup",
    "depth_mup": "depth_mup",
    "depth-mup": "depth_mup",
    "depthmup": "depth_mup",
    "completep": "completep",
    "complete_p": "completep",
    "complete-p": "completep",
    "kaiming": "kaiming",
}
BATCH_ALIASES = {
    "1": "batch1",
    "batch1": "batch1",
    "batch_1": "batch1",
    "2": "batch2",
    "batch2": "batch2",
    "batch_2": "batch2",
    "3": "batch3",
    "batch3": "batch3",
    "batch_3": "batch3",
}


class ResultsError(RuntimeError):
    def __init__(self, message: str, *, kind: str = "error"):
        super().__init__(message)
        self.kind = kind


def same_number(left, right, *, atol: float = NUMBER_ATOL) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-8, abs_tol=atol)


def canonicalize_parameterization(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace(" ", "_")
    if text in PARAMETERIZATION_ALIASES:
        return PARAMETERIZATION_ALIASES[text]
    return text


def canonicalize_batch(value) -> str | None:
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    return BATCH_ALIASES.get(text, text)


def normalize_problem(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    for prefix in ("problem", "p"):
        if text.startswith(prefix) and len(text) > len(prefix):
            rest = text[len(prefix) :].strip(" _-:")
            if rest[:1].isdigit():
                text = rest
                break
    return text


def get_loss(record) -> float | None:
    """Finite final validation loss, from ``final_val_loss`` or ``val_loss``."""
    if record is None:
        return None
    if isinstance(record, Mapping):
        for key in ("final_val_loss", "val_loss"):
            parsed = _as_float(record.get(key))
            if parsed is not None:
                return parsed
        summary = record.get("summary")
        if isinstance(summary, Mapping):
            return get_loss(summary)
        return None
    return _as_float(record)


def _as_float(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _as_int(value) -> int | None:
    number = _as_float(value)
    if number is None or not number.is_integer():
        return None
    return int(number)


def _text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and math.isfinite(value):
        return format(value, ".12g")
    if isinstance(value, int):
        return str(value)
    return None


def _layers(config: Mapping) -> list[tuple[str, Mapping]]:
    layers = [("config", config)]
    for name in METADATA_CONTAINERS:
        nested = config.get(name)
        if isinstance(nested, Mapping):
            layers.append((f"config.{name}", nested))
    return layers


def _tag_values(tags: Sequence[str]) -> dict[str, list[tuple[str, str]]]:
    found = defaultdict(list)
    alias_to_field = {
        alias: field for field, aliases in FIELD_ALIASES.items() for alias in aliases
    }
    for tag in tags:
        text = str(tag)
        key = None
        raw = None
        if text.startswith(RUN_PREFIX):
            parsed_key, sep, parsed_value = text[len(RUN_PREFIX) :].partition(".")
            if sep and parsed_key in alias_to_field and parsed_value != "":
                key, raw = alias_to_field[parsed_key], parsed_value
        else:
            for separator in ("=", ":"):
                if separator not in text:
                    continue
                parsed_key, parsed_value = text.split(separator, 1)
                if parsed_key in alias_to_field and parsed_value != "":
                    key, raw = alias_to_field[parsed_key], parsed_value
                    break
        if key is not None and raw is not None:
            found[key].append((f"tag:{text}", raw))
    return found


def _canonicalize_text(field: str, text: str) -> str:
    if field == "parameterization":
        return canonicalize_parameterization(text) or text
    if field == "batch":
        return canonicalize_batch(text) or text
    if field == "problem":
        return normalize_problem(text) or text
    if field in {"assignment", "subpart", "family", "config_role", "hypothesis", "optimizer", "lr_schedule", "model_name", "momentum_role", "status"}:
        return text.strip().lower()
    return text


def _unique_values(field: str, found: list[tuple[str, str]]) -> tuple[object | None, str | None]:
    chosen = []
    for source, raw in found:
        if field in INTEGER_FIELDS:
            number = _as_int(raw)
            if number is None:
                return None, f"{field}: {source} is not an integer ({raw!r})"
            if any(previous[1] == number for previous in chosen):
                continue
            chosen.append((source, number))
        elif field in FLOAT_FIELDS:
            number = _as_float(raw)
            if number is None:
                return None, f"{field}: {source} is not numeric ({raw!r})"
            if any(same_number(previous[1], number) for previous in chosen):
                continue
            chosen.append((source, number))
        else:
            text = _canonicalize_text(field, raw)
            if any(previous[1] == text for previous in chosen):
                continue
            chosen.append((source, text))
    if not chosen:
        return None, None
    if len(chosen) > 1:
        rendered = ", ".join(f"{source}={value}" for source, value in chosen)
        return None, f"{field}: conflicting values: {rendered}"
    return chosen[0][1], None


def _processed_tokens_compatible(requested: int, processed: int) -> bool:
    if processed == requested:
        return True
    if processed > requested or requested <= 0:
        return False
    return (requested - processed) / requested <= PROCESSED_TOKEN_TOLERANCE


def _collect_raw(config: Mapping, tags: Sequence[str]) -> dict[str, list[tuple[str, str]]]:
    found = defaultdict(list)
    for field, aliases in FIELD_ALIASES.items():
        for layer_name, layer in _layers(config):
            for alias in aliases:
                if alias not in layer:
                    continue
                raw_value = layer.get(alias)
                if field == "parameterization" and isinstance(raw_value, (list, tuple)):
                    texts = [item for item in (_text(item) for item in raw_value) if item]
                    canonical = []
                    for item in texts:
                        name = canonicalize_parameterization(item)
                        if name and name not in canonical:
                            canonical.append(name)
                    if len(canonical) == 1:
                        found[field].append((f"{layer_name}.{alias}", canonical[0]))
                    elif len(canonical) > 1:
                        found[field].append((f"{layer_name}.{alias}", "|".join(canonical)))
                    continue
                text = _text(raw_value)
                if text is None:
                    continue
                found[field].append((f"{layer_name}.{alias}", text))
        for item in _tag_values(tags).get(field, ()):
            found[field].append(item)
    model_config = config.get("model_config")
    if isinstance(model_config, Mapping):
        name = _text(model_config.get("name"))
        if name is not None:
            found["model_name"].append(("config.model_config.name", name))
    return found


def _run_summary(run) -> dict:
    """Summary already returned with the run list.

    ``run.summary`` is a W&B ``HTTPSummary``, not a dict. Treating it as one
    drops ``val_loss`` and forces a full history download.
    """
    metrics = getattr(run, "summary_metrics", None)
    if isinstance(metrics, Mapping):
        return _mapping(metrics)
    summary = getattr(run, "summary", None)
    if isinstance(summary, Mapping):
        return _mapping(summary)
    raw = getattr(summary, "_json_dict", None)
    if isinstance(raw, Mapping):
        return _mapping(raw)
    return {}


def _loss_from_history(run) -> float | None:
    """Last logged validation loss, from the tail of history only."""
    scan = getattr(run, "scan_history", None)
    if scan is None:
        return None
    try:
        last_step = int(getattr(run, "lastHistoryStep"))
    except (TypeError, ValueError):
        return None
    if last_step < 0:
        return None
    end = last_step + 1
    # Walk backward one page at a time. Stop at the first page that has a loss.
    for _ in range(8):
        start = max(0, end - 1000)
        try:
            rows = scan(keys=["val_loss"], min_step=start, max_step=end, page_size=1000)
        except Exception:
            return None
        page_step = -1
        page_value = None
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            value = _as_float(row.get("final_val_loss"))
            if value is None:
                value = _as_float(row.get("val_loss"))
            if value is None:
                continue
            step_number = _as_float(row.get("_step"))
            step = page_step + 1 if step_number is None else int(step_number)
            if step >= page_step:
                page_step = step
                page_value = value
        if page_value is not None:
            return page_value
        if start == 0:
            break
        end = start
    return None


def _mapping(value) -> dict:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return {}


def read_metadata(config, tags=(), summary=None, *, state="", run_id="", name="", history_loss=None) -> dict:
    """Normalize one run's config, tags, and summary into the A2 field contract."""
    config = _mapping(config)
    summary = _mapping(summary)
    tags = tuple(str(tag) for tag in (tags or ()))
    found = _collect_raw(config, tags)
    inconsistencies = []
    resolved = {}
    for field in FIELD_ALIASES:
        value, error = _unique_values(field, found.get(field, []))
        if error is not None:
            inconsistencies.append(error)
        resolved[field] = value

    processed_values = []
    for layer_name, layer in _layers(config):
        if "train_tokens" not in layer:
            continue
        processed = _as_int(layer.get("train_tokens"))
        if processed is None:
            inconsistencies.append(f"train_tokens: {layer_name}.train_tokens is not an integer")
        else:
            processed_values.append(processed)
    processed_unique = []
    for value in processed_values:
        if value not in processed_unique:
            processed_unique.append(value)
    if len(processed_unique) > 1:
        inconsistencies.append(f"train_tokens: conflicting values: {processed_unique}")
    processed = processed_unique[0] if len(processed_unique) == 1 else None
    requested = resolved["token_budget"]
    if requested is not None and processed is not None and not _processed_tokens_compatible(requested, processed):
        inconsistencies.append(
            f"token_budget: requested {requested} but processed train_tokens {processed}"
        )
    elif requested is None and processed is not None:
        resolved["token_budget"] = processed

    loss = get_loss(summary)
    if loss is None:
        for _, layer in _layers(config):
            loss = get_loss(layer)
            if loss is not None:
                break
    if loss is None:
        loss = _as_float(history_loss)

    state_text = (state or "").strip().lower()
    status_text = resolved["status"] or state_text or None
    state_done = state_text in FINISHED_STATES
    status_done = status_text in FINISHED_STATES if status_text else False
    if state_text and status_text and state_text != status_text and state_done != status_done:
        inconsistencies.append(
            f"status {status_text!r} disagrees with state {state_text!r}"
        )
        state_completed = False
    else:
        state_completed = state_done or (not state_text and status_done)

    memberships = []
    for _, layer in _layers(config):
        raw_memberships = layer.get("memberships")
        if isinstance(raw_memberships, (list, tuple)) and raw_memberships:
            memberships = [item for item in raw_memberships if isinstance(item, Mapping)]
            break

    record = {
        "run_id": str(run_id or ""),
        "name": str(name or ""),
        "state": state_text,
        "assignment": resolved["assignment"],
        "batch": resolved["batch"],
        "problem": resolved["problem"],
        "subpart": resolved["subpart"],
        "family": resolved["family"],
        "config_role": resolved["config_role"],
        "hypothesis": resolved["hypothesis"],
        "token_budget": resolved["token_budget"],
        "tokens": resolved["token_budget"],
        "batch_size": resolved["batch_size"],
        "peak_lr": resolved["peak_lr"],
        "weight_decay": resolved["weight_decay"],
        "beta1": resolved["beta1"],
        "beta2": resolved["beta2"],
        "width": resolved["width"],
        "depth": resolved["depth"],
        "parameterization": resolved["parameterization"],
        "optimizer": resolved["optimizer"],
        "seed": resolved["seed"],
        "lr_schedule": resolved["lr_schedule"],
        "warmup_percent": resolved["warmup_percent"],
        "model_name": resolved["model_name"],
        "momentum_role": resolved["momentum_role"],
        "status": status_text,
        "final_val_loss": loss,
        "memberships": memberships,
        "inconsistencies": inconsistencies,
        "state_completed": state_completed,
        "completed": state_completed and loss is not None and not inconsistencies,
    }
    return record


def describe_run(run) -> dict:
    """Read one W&B run, including history when the summary has no loss."""
    config = _mapping(getattr(run, "config", {}) or {})
    summary = _run_summary(run)
    tags = tuple(getattr(run, "tags", ()) or ())
    state = str(getattr(run, "state", "") or "")
    loss = get_loss(summary)
    if loss is None and state.strip().lower() in FINISHED_STATES:
        name = getattr(run, "name", None) or getattr(run, "id", "")
        print(f"  {name}: summary has no validation loss; reading the last logged point", flush=True)
        loss = _loss_from_history(run)
    return read_metadata(
        config,
        tags,
        summary,
        state=state,
        run_id=str(getattr(run, "id", "") or ""),
        name=str(getattr(run, "name", "") or getattr(run, "display_name", "") or ""),
        history_loss=loss,
    )


def query_a2_runs():
    """Return every A2 run currently visible in the configured W&B project."""
    try:
        import wandb

        from utils import WANDB_ENTITY, WANDB_PROJECT

        api = wandb.Api()
        runs = list(
            api.runs(
                f"{WANDB_ENTITY}/{WANDB_PROJECT}",
                filters={
                    "$or": [
                        {"config.assignment": ASSIGNMENT},
                        {"tags": {"$in": [ASSIGNMENT, "batch1", "batch2", "batch3"]}},
                    ]
                },
            )
        )
    except Exception as exc:
        raise ResultsError(
            "Could not query W&B for A2 runs.\n"
            f"{type(exc).__name__}: {exc}"
        ) from exc
    print(f"Reading {len(runs)} A2 runs from W&B", flush=True)
    return [describe_run(run) for run in runs]


def _filter_match(run: Mapping, filters: Mapping) -> bool:
    for key, expected in filters.items():
        if expected is None:
            continue
        actual = run.get(key)
        if key in {"problem", "subpart", "batch", "parameterization"}:
            candidates = [actual]
            for membership in run.get("memberships") or []:
                if isinstance(membership, Mapping):
                    candidates.append(membership.get(key))
            normalized = []
            for candidate in candidates:
                if key == "problem":
                    normalized.append(normalize_problem(candidate))
                elif key == "batch":
                    normalized.append(canonicalize_batch(candidate))
                elif key == "parameterization":
                    normalized.append(canonicalize_parameterization(candidate))
                else:
                    normalized.append(None if candidate is None else str(candidate).strip().lower())
            if expected not in normalized:
                return False
            continue
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            if actual is None or not same_number(actual, expected):
                return False
        elif actual != expected:
            return False
    return True


def get_completed_a2_runs(runs: Sequence[Mapping] | None = None, **filters) -> list[dict]:
    """Completed A2 runs with a finite loss and consistent metadata.

    ``filters`` compare canonical fields. Problem, subpart, batch, and
    parameterization also match any entry in ``memberships``.
    """
    if runs is None:
        runs = query_a2_runs()
    selected = []
    for run in runs:
        if not run.get("completed"):
            continue
        if run.get("assignment") not in {None, ASSIGNMENT}:
            continue
        if filters and not _filter_match(run, filters):
            continue
        selected.append(dict(run))
    return selected


def _is_momentum(run: Mapping) -> bool:
    if run.get("subpart") in {"c", "3.2c"}:
        return True
    if run.get("batch") == "batch3":
        return True
    if run.get("momentum_role") in {"momentum", "no_momentum"}:
        return True
    for key in ("family", "config_role"):
        value = run.get(key) or ""
        if "momentum" in str(value):
            return True
    return False


def _setup_reason(run: Mapping) -> str | None:
    problems = []
    if run.get("assignment") not in {None, ASSIGNMENT}:
        problems.append(f"assignment={run.get('assignment')}")
    model_name = run.get("model_name")
    if model_name not in P32_MODEL_NAMES:
        problems.append(f"model_name={model_name}, expected one of {sorted(P32_MODEL_NAMES)}")
    if run.get("lr_schedule") != P32_SCHEDULE:
        problems.append(f"lr_schedule={run.get('lr_schedule')}, expected {P32_SCHEDULE}")
    if run.get("optimizer") != P32_OPTIMIZER:
        problems.append(f"optimizer={run.get('optimizer')}, expected {P32_OPTIMIZER}")
    warmup = run.get("warmup_percent")
    if warmup is None or not same_number(warmup, P32_WARMUP):
        problems.append(f"warmup_percent={warmup}, expected {P32_WARMUP}")
    if run.get("token_budget") != P32_TOKEN_BUDGET:
        problems.append(
            f"token_budget={run.get('token_budget')}, expected {P32_TOKEN_BUDGET}"
        )
    if run.get("width") != P32_WIDTH:
        problems.append(f"width={run.get('width')}, expected {P32_WIDTH}")
    if run.get("depth") != P32_DEPTH:
        problems.append(f"depth={run.get('depth')}, expected {P32_DEPTH}")
    if run.get("parameterization") != P32_PARAMETERIZATION:
        problems.append(
            f"parameterization={run.get('parameterization')}, expected {P32_PARAMETERIZATION}"
        )
    if run.get("beta2") is None or not same_number(run.get("beta2"), P32_BETA2):
        problems.append(f"beta2={run.get('beta2')}, expected {P32_BETA2}")
    if run.get("beta1") is None or not same_number(run.get("beta1"), P32_BETA1):
        problems.append(f"beta1={run.get('beta1')}, expected {P32_BETA1}")
    if run.get("peak_lr") is None:
        problems.append("peak_lr is missing")
    if run.get("weight_decay") is None:
        problems.append("weight_decay is missing")
    if not problems:
        return None
    return "; ".join(problems)


def _choose_winner(eligible: Sequence[Mapping], *, batch_size: int) -> dict:
    groups = defaultdict(list)
    for run in eligible:
        key = (float(f"{float(run['peak_lr']):.8e}"), float(f"{float(run['weight_decay']):.8e}"))
        groups[key].append(run)
    representatives = []
    deduped_groups = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda run: (str(run.get("run_id") or ""), str(run.get("name") or "")))
        losses = [float(run["final_val_loss"]) for run in members]
        if max(losses) - min(losses) > LOSS_ATOL:
            rendered = ", ".join(
                f"{run.get('run_id')} loss={float(run['final_val_loss']):.6f}" for run in members
            )
            raise ResultsError(
                "Batch 3 cannot run:\n"
                "completed runs share an LR-WD pair but disagree on final validation loss.\n\n"
                f"problem = 3.2\n"
                f"batch_size = {batch_size}\n"
                f"peak_lr = {members[0]['peak_lr']}\n"
                f"weight_decay = {members[0]['weight_decay']}\n"
                f"  {rendered}",
                kind="inconsistent",
            )
        representatives.append(members[0])
        if len(members) > 1:
            deduped_groups.append(tuple(str(run.get("run_id") or "") for run in members))
    best_loss = min(float(run["final_val_loss"]) for run in representatives)
    tied = [
        run
        for run in representatives
        if same_number(run["final_val_loss"], best_loss, atol=TIE_ATOL)
    ]
    winner = min(
        tied,
        key=lambda run: (
            float(run["peak_lr"]),
            float(run["weight_decay"]),
            str(run.get("run_id") or ""),
        ),
    )
    return {
        "winner": winner,
        "unique_pairs": representatives,
        "tie_break_used": len(tied) > 1,
        "tied_run_ids": tuple(sorted(str(run.get("run_id") or "") for run in tied)),
        "deduped_groups": tuple(deduped_groups),
    }


def get_best_run(runs: Sequence[Mapping], *, batch_size: int) -> dict:
    """Best measured Problem 3.2 LR-WD pair at ``batch_size``.

    Identical LR-WD pairs are collapsed first. The winner is the lowest final
    validation loss. Numerical ties break by smaller peak LR, then smaller
    weight decay, then the smallest run id.
    """
    inconsistent = []
    excluded = []
    eligible = []
    for run in runs:
        if normalize_problem(run.get("problem")) != "3.2":
            continue
        if canonicalize_batch(run.get("batch")) == "batch3" or _is_momentum(run):
            continue
        if run.get("batch_size") != batch_size:
            continue
        label = run.get("run_id") or run.get("name") or "(unnamed)"
        if run.get("inconsistencies"):
            inconsistent.append(f"{label}: {'; '.join(run['inconsistencies'])}")
            continue
        if not run.get("completed"):
            state = run.get("state") or run.get("status") or "unknown"
            excluded.append((label, f"status={state}"))
            continue
        reason = _setup_reason(run)
        if reason:
            excluded.append((label, reason))
            continue
        if run.get("final_val_loss") is None:
            inconsistent.append(f"{label}: final validation loss is missing")
            continue
        eligible.append(run)

    if inconsistent:
        lines = [
            "Batch 3 cannot run:",
            "completed Problem 3.2 runs have ambiguous or inconsistent metadata.",
            "",
            f"problem = 3.2",
            f"batch_size = {batch_size}",
        ]
        lines.extend(f"  {item}" for item in inconsistent)
        raise ResultsError("\n".join(lines), kind="inconsistent")
    if not eligible:
        lines = [
            "Batch 3 cannot run yet:",
            "required Batch 1 or Batch 2 source experiments are missing or incomplete.",
            "",
            "No completed measured LR-WD configurations were found for:",
            "problem = 3.2",
            f"batch_size = {batch_size}",
            "",
            "Run Batch 2 first." if batch_size == 256 else "Run Batch 1 first.",
        ]
        if excluded:
            lines.extend(["", "Related runs that were not eligible:"])
            for label, reason in excluded:
                lines.append(f"  {label}: {reason}")
        raise ResultsError("\n".join(lines), kind="missing")
    chosen = _choose_winner(eligible, batch_size=batch_size)
    chosen.update(
        batch_size=batch_size,
        eligible=tuple(eligible),
        excluded=tuple(excluded),
    )
    return chosen


def classify_named_runs(names: Sequence[str], runs: Sequence[Mapping], volume_complete: Mapping | None = None) -> dict[str, str]:
    """Map a tracking name to completed, running, failed, or missing.

    A finished run with a loss is completed. An in-progress run is left alone.
    A failed run, or a finished run with no loss, is reported as failed so the
    caller can relaunch it. ``volume_complete`` marks names whose saved model
    or stress output already exists.
    """
    volume_complete = volume_complete or {}
    grouped = defaultdict(list)
    for run in runs:
        name = run.get("name")
        if name:
            grouped[name].append(run)
    statuses = {}
    for name in names:
        if volume_complete.get(name):
            statuses[name] = "completed"
            continue
        records = grouped.get(name, [])
        if not records:
            statuses[name] = "missing"
            continue
        finished = [record for record in records if str(record.get("state") or "") in FINISHED_STATES]
        if any(record.get("final_val_loss") is not None for record in finished):
            statuses[name] = "completed"
            continue
        if any(str(record.get("state") or "") not in FINISHED_STATES | FAILED_STATES for record in records):
            statuses[name] = "running"
            continue
        if finished or any(str(record.get("state") or "") in FAILED_STATES for record in records):
            statuses[name] = "failed"
            continue
        statuses[name] = "running"
    return statuses


def get_a2_runs():
    """Every visible A2 run. Same records as ``query_a2_runs``."""
    return query_a2_runs()


def get_completed_runs(runs: Sequence[Mapping] | None = None, **filters) -> list[dict]:
    """Completed runs. ``filters`` use the same comparison as ``get_completed_a2_runs``."""
    return get_completed_a2_runs(runs, **filters)


def get_runs_for_subpart(problem, subpart, runs: Sequence[Mapping] | None = None, **filters) -> list[dict]:
    """Completed runs for one problem subpart, including shared memberships."""
    return get_completed_a2_runs(runs, problem=problem, subpart=subpart, **filters)


def get_best_measured_run(runs: Sequence[Mapping], *, batch_size: int) -> dict:
    """Best measured Problem 3.2 LR-WD pair. Same rule as ``get_best_run``."""
    return get_best_run(runs, batch_size=batch_size)


def _flatten(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _field_matches(key: str, actual, expected) -> bool:
    if isinstance(actual, (list, tuple)):
        return any(_field_matches(key, item, expected) for item in actual)
    if key == "problem":
        return normalize_problem(actual) == normalize_problem(expected)
    if key == "batch":
        return canonicalize_batch(actual) == canonicalize_batch(expected)
    if key == "parameterization":
        return canonicalize_parameterization(actual) == canonicalize_parameterization(expected)
    if key == "subpart":
        if actual is None or expected is None:
            return False
        return str(actual).strip().lower() == str(expected).strip().lower()
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return actual is not None and same_number(actual, expected)
    return actual == expected


def matching_runs(runs: Sequence[Mapping], *, only_completed: bool = True, **filters) -> list[dict]:
    """Filter already loaded runs.

    Problem, subpart, batch, parameterization, hypothesis, family, config_role,
    and stage match either the top-level field or any membership. This does not
    change ``get_best_run`` or ``get_completed_a2_runs``.
    """
    selected = []
    for run in runs:
        if only_completed and not run.get("completed"):
            continue
        matched = True
        for key, expected in filters.items():
            if expected is None:
                continue
            candidates = _flatten(run.get(key))
            for membership in run.get("memberships") or []:
                if isinstance(membership, Mapping):
                    candidates.extend(_flatten(membership.get(key)))
            if not any(_field_matches(key, candidate, expected) for candidate in candidates):
                matched = False
                break
        if matched:
            selected.append(dict(run))
    return selected
