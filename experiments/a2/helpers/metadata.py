"""Canonical A2 metadata fields.

The aliases and normalizers are the ones result loading already uses.
Hyperball is the ``adamh`` optimizer, not a parameterization name.
"""

from experiments.a2.helpers.results import (  # noqa: F401
    ASSIGNMENT,
    BATCH_ALIASES,
    FIELD_ALIASES,
    PARAMETERIZATION_ALIASES,
    canonicalize_batch,
    canonicalize_parameterization,
    normalize_problem,
)


CANONICAL_FIELDS = (
    "assignment",
    "batch",
    "problem",
    "subpart",
    "family",
    "config_role",
    "parameterization",
)
CANONICAL_PARAMETERIZATIONS = (
    "baseline",
    "kaiming",
    "mup",
    "depth_mup",
    "completep",
)
CANONICAL_OPTIMIZERS = ("adamw", "adam", "adamh", "muon")
HYPERBALL_OPTIMIZER = "adamh"
DISPLAY_NAMES = {
    "baseline": "course baseline",
    "kaiming": "Kaiming",
    "mup": "µP",
    "depth_mup": "Depth-µP",
    "completep": "CompleteP",
    "adamw": "AdamW",
    "adamh": "Hyperball",
    "adam": "Adam",
}


def display_name(value: str) -> str:
    canonical = canonicalize_parameterization(value) or str(value)
    return DISPLAY_NAMES.get(canonical, canonical)
