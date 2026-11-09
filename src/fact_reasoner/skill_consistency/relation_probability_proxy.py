"""Opt-in heuristic relation weights; never inferred calibrated probabilities.

These mappings are experiment assumptions. They do not change the registered
NLI prompts or any native FactReasoner probability calculation.
"""

from types import MappingProxyType

LEVELS = (
    "unambiguous",
    "minor_ambiguity",
    "moderate_ambiguity",
    "substantial_ambiguity",
    "closely_competing_relations",
)
POLICY_VERSION = "0.1.0-experimental"
POLICIES = MappingProxyType(
    {
        name: MappingProxyType(dict(zip(LEVELS, values)))
        for name, values in {
            "provisional": (0.90, 0.75, 0.60, 0.45, 0.35),
            "flatter": (0.80, 0.70, 0.60, 0.45, 0.35),
            "sharper": (0.95, 0.85, 0.70, 0.45, 0.35),
        }.items()
    }
)


def relation_probability_proxy(
    relation: str, ambiguity: str, *, policy: str = "provisional"
) -> float:
    """Look up a category weight, equally for all three NLI relations.

    Unknown labels and policies fail explicitly; no fallback weight is supplied.
    """
    if relation not in ("entailment", "contradiction", "neutral"):
        raise ValueError(f"Unknown NLI relation: {relation!r}")
    if policy not in POLICIES:
        raise ValueError(f"Unknown probability-proxy policy: {policy!r}")
    if ambiguity not in LEVELS:
        raise ValueError(f"Unknown relation-ambiguity label: {ambiguity!r}")
    return POLICIES[policy][ambiguity]


def native_edge_weight(
    relation: str, ambiguity: str, *, policy: str = "provisional"
) -> float | None:
    """Adapt only compatible proxy values to native signed relation factors.

    Neutral remains recorded at judgment level, but produces no native edge.
    Below 0.5, a native entailment/contradiction factor reverses its direction;
    that is not the intended meaning of ambiguity. Reject rather than silently
    invert, clamp, or invent a probability redistribution.
    """
    value = relation_probability_proxy(relation, ambiguity, policy=policy)
    if relation == "neutral":
        return None
    if value < 0.5:
        raise ValueError(
            "Proxy below 0.5 would reverse a native relation edge; a different factor interpretation is required"
        )
    return value
