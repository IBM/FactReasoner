"""Versioned relation-ambiguity judgments and high-ambiguity abstention.

The prompt protocol is the frozen September 2026 documentation-consistency
protocol.  Ambiguity categories are ordinal experiment assumptions, not
calibrated probabilities.  Substantial and closely competing judgments retain
all raw provenance but are withheld rather than relabeled neutral.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import resources
import json
from types import MappingProxyType
from typing import Any, Literal, Mapping

from .comprehensiveness import FactRelation
from .factreasoner_prompt_adapter import (
    FORWARD_DOCUMENTATION_TO_CODE_V1_3,
    REVERSE_CODE_TO_REQUIREMENT_V1_1,
    PromptDirection,
)
from .relation_probability_proxy import LEVELS, relation_probability_proxy


RelationLabel = Literal["entailment", "contradiction", "neutral"]
AmbiguityLevel = Literal[
    "unambiguous",
    "minor_ambiguity",
    "moderate_ambiguity",
    "substantial_ambiguity",
    "closely_competing_relations",
]
EffectiveStatus = Literal["accepted", "withheld_high_ambiguity"]

AMBIGUITY_PROTOCOL_VERSION = "relation-ambiguity-0.2.0-pilot"
AMBIGUITY_GATE_POLICY_VERSION = "high-ambiguity-abstention-1.0.0"
AMBIGUITY_EXAMPLES_RESOURCE = "relation_ambiguity_examples_v2.json"
AMBIGUITY_EXAMPLES_SHA256 = (
    "5e1d23f327c41f0d850f79b26880761e9773c03bb8336c5e275881ac1effdfee"
)
DIRECTIONAL_PROMPT_SHA256 = MappingProxyType(
    {
        "documentation_to_code": (
            "3ea8b32c3344730e6710f88e4e25eafb53acf026a3cf6b405a65d682cbbc33cf"
        ),
        "code_to_requirement": (
            "ca42ba49911be7d381f647344d75d0acb00d27339d47748ea5ca2197ec0e157c"
        ),
    }
)
ACCEPTED_AMBIGUITY_LEVELS = frozenset(
    {"unambiguous", "minor_ambiguity", "moderate_ambiguity"}
)
WITHHELD_AMBIGUITY_LEVELS = frozenset(
    {"substantial_ambiguity", "closely_competing_relations"}
)
RELATION_LABELS = frozenset({"entailment", "contradiction", "neutral"})
_FORMAT_PARAGRAPH = (
    "Explain the decisive evidence briefly. End with exactly one label in square\n"
    "brackets: [entailment], [contradiction], or [neutral]."
)

RELATION_AMBIGUITY_SCHEMA: Mapping[str, Any] = MappingProxyType(
    {
        "type": "object",
        "properties": {
            "relation": {"type": "string", "enum": sorted(RELATION_LABELS)},
            "relation_ambiguity": {"type": "string", "enum": list(LEVELS)},
            "basis": {"type": "string"},
        },
        "required": ["relation", "relation_ambiguity", "basis"],
        "additionalProperties": False,
    }
)


def _examples_bytes() -> bytes:
    value = (
        resources.files(__package__)
        .joinpath("prompts", AMBIGUITY_EXAMPLES_RESOURCE)
        .read_bytes()
    )
    observed = hashlib.sha256(value).hexdigest()
    if observed != AMBIGUITY_EXAMPLES_SHA256:
        raise RuntimeError(
            "Relation-ambiguity examples changed without a protocol version update: "
            f"expected {AMBIGUITY_EXAMPLES_SHA256}, found {observed}"
        )
    return value


def ambiguity_examples() -> dict[str, Any]:
    """Load and verify the frozen relation-ambiguity example bank."""

    value = json.loads(_examples_bytes())
    if tuple(value.get("labels", ())) != LEVELS:
        raise RuntimeError("Frozen ambiguity labels do not match the scoring policy")
    if {item["output"]["relation_ambiguity"] for item in value["examples"]} != set(
        LEVELS
    ):
        raise RuntimeError("Frozen examples do not cover every ambiguity level")
    return value


def ambiguity_prompt(direction: PromptDirection) -> str:
    """Build one byte-verified frozen directional ambiguity prompt.

    Only the two documentation-consistency directions have a frozen ambiguity
    protocol.  The returned text is the stable prefix; callers append the actual
    premise and hypothesis exactly as the original experiment did.
    """

    if direction not in DIRECTIONAL_PROMPT_SHA256:
        raise ValueError(
            "Relation ambiguity is frozen only for documentation_to_code and "
            "code_to_requirement"
        )
    text = (
        FORWARD_DOCUMENTATION_TO_CODE_V1_3.text
        if direction == "documentation_to_code"
        else REVERSE_CODE_TO_REQUIREMENT_V1_1.text
    )
    if text.count(_FORMAT_PARAGRAPH) != 1:
        raise RuntimeError(
            "Directional prompt no longer has the frozen output contract"
        )
    definitions = text.split(_FORMAT_PARAGRAPH)[0].strip()
    bank = ambiguity_examples()
    instruction = (
        "Return only a JSON object with relation, relation_ambiguity, and basis, in "
        "that order. Select the relation first under the directional rules. Then "
        "classify ambiguity in assigning that relation. Do not give a numeric "
        "probability.\n"
    )
    instruction += "\n".join(f"{key}: {value}" for key, value in bank["labels"].items())
    instruction += (
        "\nMissing coverage can clearly establish neutral: do not confuse neutral "
        "with evaluator ambiguity. An unambiguous relation can be entailment, "
        "contradiction, or neutral. Identify a concrete competing interpretation "
        "when ambiguity is present. Do not infer ambiguity merely from absence of "
        "evidence. The examples illustrate generic NLI; apply the directional rules "
        "to the actual pair."
    )
    examples = []
    for item in bank["examples"]:
        examples.append(
            f"PREMISE: {item['premise']}\nHYPOTHESIS: {item['hypothesis']}\n"
            f"OUTPUT: {json.dumps(item['output'])}"
        )
    prompt = (
        definitions
        + "\n\n"
        + instruction
        + "\n\nIN-CONTEXT EXAMPLES\n\n"
        + "\n\n".join(examples)
    )
    observed = hashlib.sha256(prompt.encode()).hexdigest()
    expected = DIRECTIONAL_PROMPT_SHA256[direction]
    if observed != expected:
        raise RuntimeError(
            f"Frozen {direction} ambiguity prompt changed: expected {expected}, "
            f"found {observed}"
        )
    return prompt


def ambiguity_prompt_with_pair(
    direction: PromptDirection, *, premise: str, hypothesis: str
) -> str:
    """Append one actual pair using the frozen experiment wire format."""

    return (
        ambiguity_prompt(direction)
        + "\n\nACTUAL PAIR\nPREMISE: "
        + premise
        + "\nHYPOTHESIS: "
        + hypothesis
    )


def parse_ambiguity_output(text: str) -> dict[str, str]:
    """Strictly parse one structured relation-plus-ambiguity response."""

    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {
        "relation",
        "relation_ambiguity",
        "basis",
    }:
        raise ValueError("Output schema mismatch")
    if value["relation"] not in RELATION_LABELS:
        raise ValueError(f"Invalid relation: {value['relation']!r}")
    if value["relation_ambiguity"] not in LEVELS:
        raise ValueError(f"Invalid relation ambiguity: {value['relation_ambiguity']!r}")
    if not isinstance(value["basis"], str) or not value["basis"].strip():
        raise ValueError("Relation basis must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class AmbiguityJudgment:
    """One raw ambiguity judgment plus its versioned effective disposition."""

    source_id: str
    target_id: str
    relation: RelationLabel
    relation_ambiguity: AmbiguityLevel
    basis: str
    raw_relation_score: float | None = None
    raw_relation_score_method: str | None = None
    effective_status: EffectiveStatus = "accepted"
    effective_relation: RelationLabel | None = None
    discrete_weight: float = 0.0
    gate_policy_version: str = AMBIGUITY_GATE_POLICY_VERSION

    def __post_init__(self) -> None:
        if not self.source_id or not self.target_id:
            raise ValueError("Judgment source and target identifiers must not be empty")
        if self.relation not in RELATION_LABELS:
            raise ValueError(f"Unknown relation: {self.relation!r}")
        if self.relation_ambiguity not in LEVELS:
            raise ValueError(f"Unknown ambiguity level: {self.relation_ambiguity!r}")
        if not self.basis.strip():
            raise ValueError("Judgment basis must not be empty")
        if (
            self.raw_relation_score is not None
            and not 0.0 <= self.raw_relation_score <= 1.0
        ):
            raise ValueError("Raw relation score must be between 0 and 1")
        if (self.raw_relation_score is None) != (
            self.raw_relation_score_method is None
        ):
            raise ValueError("Raw relation score and method must be recorded together")
        expected_status: EffectiveStatus = (
            "accepted"
            if self.relation_ambiguity in ACCEPTED_AMBIGUITY_LEVELS
            else "withheld_high_ambiguity"
        )
        expected_relation = self.relation if expected_status == "accepted" else None
        expected_weight = relation_probability_proxy(
            self.relation, self.relation_ambiguity
        )
        if self.effective_status != expected_status:
            raise ValueError("Effective status does not match the ambiguity gate")
        if self.effective_relation != expected_relation:
            raise ValueError("Effective relation does not match the ambiguity gate")
        if self.discrete_weight != expected_weight:
            raise ValueError("Discrete weight does not match the frozen proxy policy")
        if self.gate_policy_version != AMBIGUITY_GATE_POLICY_VERSION:
            raise ValueError("Unknown ambiguity gate policy version")

    @classmethod
    def from_parsed(
        cls,
        *,
        source_id: str,
        target_id: str,
        parsed: Mapping[str, Any],
        raw_relation_score: float | None = None,
        raw_relation_score_method: str | None = None,
    ) -> AmbiguityJudgment:
        """Apply the frozen gate to one strictly shaped parsed response."""

        normalized = parse_ambiguity_output(json.dumps(dict(parsed)))
        ambiguity = normalized["relation_ambiguity"]
        accepted = ambiguity in ACCEPTED_AMBIGUITY_LEVELS
        return cls(
            source_id=source_id,
            target_id=target_id,
            relation=normalized["relation"],  # type: ignore[arg-type]
            relation_ambiguity=ambiguity,  # type: ignore[arg-type]
            basis=normalized["basis"],
            raw_relation_score=raw_relation_score,
            raw_relation_score_method=raw_relation_score_method,
            effective_status="accepted" if accepted else "withheld_high_ambiguity",
            effective_relation=normalized["relation"] if accepted else None,  # type: ignore[arg-type]
            discrete_weight=relation_probability_proxy(
                normalized["relation"], ambiguity
            ),
        )

    @property
    def is_withheld(self) -> bool:
        return self.effective_status == "withheld_high_ambiguity"

    def to_fact_relation(self, *, ungated: bool = False) -> FactRelation | None:
        """Convert accepted evidence, or explicitly request an ungated sensitivity."""

        relation = self.relation if ungated else self.effective_relation
        if relation is None:
            return None
        return FactRelation(
            source_id=self.source_id,
            target_id=self.target_id,
            label=relation,
            probability=self.discrete_weight,
            raw_label=self.relation,
            raw_probability=self.raw_relation_score,
            policy_action=(
                "ungated_high_ambiguity_sensitivity"
                if ungated and self.is_withheld
                else "ambiguity_gate_accepted"
            ),
            policy_reasons=(
                ("high_ambiguity_retained_for_sensitivity",)
                if ungated and self.is_withheld
                else ()
            ),
            review_flags=(("withheld_high_ambiguity",) if self.is_withheld else ()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "raw_relation": self.relation,
            "relation_ambiguity": self.relation_ambiguity,
            "basis": self.basis,
            "raw_relation_score": self.raw_relation_score,
            "raw_relation_score_method": self.raw_relation_score_method,
            "discrete_weight": self.discrete_weight,
            "effective_status": self.effective_status,
            "effective_relation": self.effective_relation,
            "gate_policy_version": self.gate_policy_version,
        }
