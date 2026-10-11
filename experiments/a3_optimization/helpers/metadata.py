"""Canonical A3 metadata fields (mirrors A2 helpers/metadata patterns)."""

ASSIGNMENT = "a3"

CANONICAL_FIELDS = (
    "assignment",
    "batch",
    "problem",
    "subpart",
    "family",
    "config_role",
    "stage",
    "hypothesis",
)

CANONICAL_FAMILIES = (
    "mode_connectivity",
    "river_valley",
    "edge_of_stability",
    "hessian",
    "prediction_problem",
)


def problem_tags(problem: str, subpart: str | None = None) -> list[str]:
    compact = str(problem).replace(".", "")
    tags = [f"p{compact}"]
    if subpart:
        tags.append(f"p{compact}{subpart}")
    return tags


def membership_tags(membership: dict, *, batch: str) -> list[str]:
    tags = ["a3", batch, *problem_tags(membership["problem"], membership.get("subpart"))]
    for key in ("family", "config_role", "hypothesis", "stage"):
        value = membership.get(key)
        if value:
            tags.append(str(value))
    return tags


def display_name(value: str) -> str:
    return {
        "mode_connectivity": "mode connectivity",
        "river_valley": "river-valley",
        "edge_of_stability": "edge of stability",
        "hessian": "Hessian",
        "wsd0.0": "constant after warmup",
        "wsd0.2": "plateau-then-decay 20%",
        "invsqrt": "inverse-sqrt",
        "sgdr2000": "SGDR T=2000",
    }.get(str(value), str(value))
