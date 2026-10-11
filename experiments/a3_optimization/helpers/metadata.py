"""Canonical A3 experiment metadata (A2 membership pattern)."""

from __future__ import annotations

ASSIGNMENT = "a3"
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


def canonicalize_batch(value) -> str | None:
    if value is None or value == "":
        return None
    return BATCH_ALIASES.get(str(value).strip().lower(), str(value).strip().lower())


def membership(
    *,
    problem: str,
    subpart: str,
    family: str,
    config_role: str,
    batch: str = "batch1",
    hypothesis: str | None = None,
    stage: str | None = None,
    source_type: str | None = None,
) -> dict:
    record = {
        "assignment": ASSIGNMENT,
        "batch": canonicalize_batch(batch) or batch,
        "problem": str(problem),
        "subpart": str(subpart),
        "family": family,
        "config_role": config_role,
        "hypothesis": hypothesis,
        "stage": stage,
        "source_type": source_type,
    }
    return {key: value for key, value in record.items() if value is not None}


def membership_tags(record: dict) -> list[str]:
    batch = canonicalize_batch(record.get("batch")) or "batch1"
    problem = str(record.get("problem", "")).replace(".", "")
    subpart = str(record.get("subpart", ""))
    tags = [ASSIGNMENT, batch, f"p{problem}", f"p{problem}{subpart}"]
    for key in ("family", "config_role", "hypothesis", "stage"):
        value = record.get(key)
        if value:
            tags.append(str(value).replace("_", "-"))
    # Preserve order, drop duplicates.
    seen = []
    for tag in tags:
        if tag not in seen:
            seen.append(tag)
    return seen
