"""Auditable task guards for project-specific FR2 relation labels.

The NLI model remains the semantic classifier.  These rules only reject labels
that violate explicit project invariants (for example, treating read-only
inspection as a contradiction of a no-modification rule).  A rejected label is
changed to neutral and its original confidence is retained as raw provenance;
it is never reinterpreted as confidence in the replacement label.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import re
from typing import Literal, Sequence
from urllib.parse import parse_qsl, urlsplit

from .comprehensiveness import FactAtom, FactRelation


RelationDirection = Literal["forward", "reverse"]
RELATION_POLICY_ID = "skill-documentation-relation-guards"
RELATION_POLICY_VERSION = "1.2.0"

_GENERAL_CAPABILITIES = {
    "behavioral_configuration",
    "control_flow_constraint",
    "general_behavior",
    "model_instruction",
    "output_generation",
    "requirement_enforcement",
}
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_MUTATION_PROHIBITION_RE = re.compile(
    r"\b(?:do not|don't|never|must not)\b[^.]{0,80}"
    r"\b(?:changes?|edits?|modif(?:y|ies|ications?)|write|patch|apply fixes?)\b",
    re.IGNORECASE,
)
_MUTATION_ACTION_RE = re.compile(
    r"\b(?:changes?|edits?|modif(?:y|ies|ications?)|write|patch|apply(?:ing)? (?:a )?fix)\b",
    re.IGNORECASE,
)
_READ_ONLY_ACTION_RE = re.compile(
    r"\b(?:analy[sz](?:e|es|ed|ing)|audits?|checks?|diffs?|extracts?|"
    r"gathers?|inspects?|lists?|pars(?:e|es|ed|ing)|reads?|reports?|reviews?|"
    r"search(?:es|ed|ing)?|verif(?:y|ies|ied|ying))\b",
    re.IGNORECASE,
)
_EXCLUSIVE_RE = re.compile(
    r"\b(?:only|must exclusively|never|must not|cannot|can't)\b",
    re.IGNORECASE,
)
_ACTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "collect": re.compile(
        r"\b(?:extract|gather|identify|list|enumerate)(?:s|ed|ing)?\b", re.I
    ),
    "classify": re.compile(r"\bclassif(?:y|ies|ied|ication)\b", re.I),
    "request": re.compile(r"\b(?:curl|download|fetch|http|request)\b", re.I),
    "read": re.compile(r"\b(?:diff|inspect|read)\b", re.I),
    "search": re.compile(r"\bsearch\b", re.I),
    "analyze": re.compile(
        r"\b(?:analy[sz]e|audit|check|review|verify|find bugs?|security analysis)\b",
        re.I,
    ),
    "report": re.compile(r"\b(?:findings?|output|report|say)\b", re.I),
    "provide": re.compile(r"\b(?:include|provide|supply|state)\b", re.I),
    "configure": re.compile(r"\b(?:configur(?:e|es)|sets?)\b", re.I),
    "mutate": _MUTATION_ACTION_RE,
    "execute": re.compile(r"\b(?:execute|invoke|run)\b", re.I),
    "fabricate": re.compile(r"\b(?:fabricat(?:e|es|ing)|invent(?:s|ing)?)\b", re.I),
    "ask": re.compile(r"\b(?:ask|request clarification)\b", re.I),
}

# Explicit negative obligations must be supported in their own action scope.
# Merely sharing "output" or "report" cannot support a no-edit/no-invention rule.
_NEGATIVE_CLAUSE_RE = re.compile(
    r"\b(?:do not|don't|never|must not|cannot|can't|rather than|without)\b([^.;\n,]+)",
    re.I,
)
_PROHIBITION_ACTIONS = {"mutate", "fabricate", "ask"}


def _prohibited_actions(text: str) -> set[str]:
    result: set[str] = set()
    for clause in _NEGATIVE_CLAUSE_RE.findall(text):
        # A trailing positive instruction is a different obligation.
        clause = re.split(r"\s+(?:but|just|instead|rather)\s+|\s+-\s+|,", clause)[0]
        result.update(_action_frames(clause) & _PROHIBITION_ACTIONS)
    if re.search(r"\bread.only(?:_mode)?\b(?:\s+as\s+True)?", text, re.I):
        # A false configuration is not a prohibition.
        if not re.search(r"\bread.only(?:_mode)?\b\s+as\s+False", text, re.I):
            result.add("mutate")
    return result


def _enumerated_scope(text: str) -> tuple[str, ...]:
    """Literal category collections in deterministic configuration/schema claims.

    Only explicit comma-separated prose lists are checked. This does not infer
    missing semantics from arbitrary lexical overlap or code syntax.
    """
    match = re.search(r"(?:\bas\s+|\bfor:\s*)([^\n]+)\.?$", text)
    if not match:
        return ()
    value = match.group(1).rstrip(".")
    if any(char in value for char in "{}[]='\"_"):
        return ()
    parts = tuple(
        re.sub(r"^(?:and|or)\s+", "", part.strip(), flags=re.I)
        for part in value.split(",")
    )
    return parts if len(parts) >= 3 and all(parts) else ()


def _normalized_scope(text: str) -> str:
    return " ".join(re.findall(r"[\w]+", text.casefold()))


@dataclass(frozen=True, slots=True)
class RelationPolicyResult:
    """Guarded relations plus a complete adjustment summary."""

    relations: tuple[FactRelation, ...]
    adjusted_count: int
    reason_counts: tuple[tuple[str, int], ...]
    review_flag_counts: tuple[tuple[str, int], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": RELATION_POLICY_ID,
            "policy_version": RELATION_POLICY_VERSION,
            "semantics": (
                "deterministic guards may neutralize an invalid entailment or "
                "contradiction; they never create entailment or contradiction"
            ),
            "confidence_policy": (
                "an adjusted relation has probability=null because raw confidence "
                "cannot be transformed into confidence in the replacement label"
            ),
            "adjusted_count": self.adjusted_count,
            "reason_counts": dict(self.reason_counts),
            "review_flag_counts": dict(self.review_flag_counts),
        }


def apply_relation_policy(
    nodes: Sequence[FactAtom],
    relations: Sequence[FactRelation],
    *,
    direction: RelationDirection,
) -> RelationPolicyResult:
    """Neutralize model labels that violate declared task semantics."""

    if direction not in {"forward", "reverse"}:
        raise ValueError(f"Unknown relation direction: {direction!r}")
    node_by_id = {item.id: item for item in nodes}
    if len(node_by_id) != len(nodes):
        raise ValueError("Node identifiers must be unique")
    unknown = {
        endpoint
        for relation in relations
        for endpoint in (relation.source_id, relation.target_id)
        if endpoint not in node_by_id
    }
    if unknown:
        raise ValueError(f"Relations reference unknown nodes: {sorted(unknown)}")

    guarded: list[FactRelation] = []
    reason_counts: Counter[str] = Counter()
    review_flag_counts: Counter[str] = Counter()
    for relation in relations:
        premise = node_by_id[relation.source_id]
        hypothesis = node_by_id[relation.target_id]
        reasons = _guard_reasons(premise, hypothesis, relation)
        review_flags = _review_flags(premise, hypothesis, relation)
        review_flag_counts.update(review_flags)
        if reasons and relation.label != "neutral":
            reason_counts.update(reasons)
            guarded.append(
                replace(
                    relation,
                    label="neutral",
                    probability=None,
                    raw_label=relation.label,
                    raw_probability=relation.probability,
                    policy_action="neutralized",
                    policy_reasons=tuple(sorted(reasons)),
                    review_flags=tuple(
                        sorted(set(relation.review_flags) | review_flags)
                    ),
                )
            )
        else:
            guarded.append(
                replace(
                    relation,
                    review_flags=tuple(
                        sorted(set(relation.review_flags) | review_flags)
                    ),
                )
            )
    return RelationPolicyResult(
        relations=tuple(guarded),
        adjusted_count=sum(item.policy_action == "neutralized" for item in guarded),
        reason_counts=tuple(sorted(reason_counts.items())),
        review_flag_counts=tuple(sorted(review_flag_counts.items())),
    )


def _guard_reasons(
    premise: FactAtom,
    hypothesis: FactAtom,
    relation: FactRelation,
) -> set[str]:
    if relation.label == "neutral":
        return set()

    shared_capabilities = _substantive_capabilities(
        premise
    ) & _substantive_capabilities(hypothesis)
    shared_entities = set(premise.entities) & set(hypothesis.entities)
    entity_conflict = _conflicting_explicit_entities(premise, hypothesis)

    if relation.label in {"entailment", "equivalence"}:
        reasons: set[str] = set()
        if premise.semantic_role == "example":
            reasons.add("example_does_not_entail_internal_behavior")
        if entity_conflict:
            reasons.add("conflicting_explicit_entity")
        if (
            hypothesis.granularity == "mechanism"
            and premise.granularity != "mechanism"
            and not shared_entities
        ):
            reasons.add("broad_statement_does_not_entail_mechanism")
        missing_prohibitions = _prohibited_actions(
            hypothesis.text
        ) - _prohibited_actions(premise.text)
        if missing_prohibitions:
            reasons.add("missing_explicit_prohibition_scope")
        required_actions = {
            action
            for clause in re.findall(
                r"\bmust\s+(?:only\s+)?([^.;\n]+)", hypothesis.text, re.I
            )
            if not clause.lower().startswith("not ")
            for action in _action_frames(clause)
            if action in {"collect", "classify", "read", "search"}
        }
        if required_actions - _action_frames(premise.text):
            reasons.add("missing_required_action_scope")
        categories = _enumerated_scope(hypothesis.text)
        premise_scope = _normalized_scope(premise.text)
        if categories and any(
            _normalized_scope(category) not in premise_scope for category in categories
        ):
            reasons.add("incomplete_enumerated_scope")
        premise_actions = _action_frames(premise.text)
        hypothesis_actions = _action_frames(hypothesis.text)
        if (
            premise_actions
            and hypothesis_actions
            and premise_actions.isdisjoint(hypothesis_actions)
            and not (
                _prohibited_actions(premise.text) & _prohibited_actions(hypothesis.text)
            )
            and not _opposite_polarity(premise, hypothesis)
        ):
            reasons.add("different_action_scope")
        return reasons

    # Contradiction requires evidence that both statements cannot hold.  Mere
    # topical difference, extra detail, or read-only analysis is insufficient.
    if _read_only_compatible(premise.text, hypothesis.text):
        return {"read_only_activity_is_compatible_with_no_modification"}
    if not shared_capabilities and not shared_entities:
        return {"no_shared_scope_for_contradiction"}
    if (
        not _opposite_polarity(premise, hypothesis)
        and not entity_conflict
        and not (
            _EXCLUSIVE_RE.search(premise.text) or _EXCLUSIVE_RE.search(hypothesis.text)
        )
    ):
        return {"no_explicit_incompatibility"}
    return set()


def _substantive_capabilities(node: FactAtom) -> set[str]:
    return set(node.capability_types) - _GENERAL_CAPABILITIES


def _urls(node: FactAtom) -> set[str]:
    return {item.rstrip("/") for item in node.entities if _URL_RE.match(item)}


def _conflicting_explicit_entities(first: FactAtom, second: FactAtom) -> bool:
    first_urls = _urls(first)
    second_urls = _urls(second)
    return bool(
        first_urls
        and second_urls
        and not any(
            _urls_compatible(left, right)
            for left in first_urls
            for right in second_urls
        )
    )


def _review_flags(
    first: FactAtom,
    second: FactAtom,
    relation: FactRelation,
) -> set[str]:
    """Flag weak routing agreement without changing the model verdict."""

    if relation.label not in {"entailment", "equivalence"}:
        return set()
    first_capabilities = set(first.capability_types) - {"general_behavior"}
    second_capabilities = set(second.capability_types) - {"general_behavior"}
    if (
        first_capabilities
        and second_capabilities
        and first_capabilities.isdisjoint(second_capabilities)
    ):
        return {"no_shared_capability_type"}
    return set()


def _urls_compatible(first: str, second: str) -> bool:
    """Compare exact URLs while allowing a declared dynamic template slot."""

    left = urlsplit(first.rstrip("/"))
    right = urlsplit(second.rstrip("/"))
    if (left.scheme.casefold(), left.netloc.casefold()) != (
        right.scheme.casefold(),
        right.netloc.casefold(),
    ):
        return False

    def segments(path: str) -> tuple[str, ...]:
        return tuple(item for item in path.split("/") if item)

    left_segments = segments(left.path)
    right_segments = segments(right.path)
    if len(left_segments) != len(right_segments):
        return False
    for left_item, right_item in zip(left_segments, right_segments, strict=True):
        if left_item == right_item:
            continue
        if not any(
            marker in left_item + right_item for marker in ("{", "}", "$", "<", ">")
        ):
            return False
    left_keys = {key for key, _ in parse_qsl(left.query, keep_blank_values=True)}
    right_keys = {key for key, _ in parse_qsl(right.query, keep_blank_values=True)}
    return not left_keys or not right_keys or left_keys == right_keys


def _opposite_polarity(first: FactAtom, second: FactAtom) -> bool:
    return {first.polarity, second.polarity} == {"positive", "negative"}


def _read_only_compatible(first: str, second: str) -> bool:
    pairs = ((first, second), (second, first))
    return any(
        _MUTATION_PROHIBITION_RE.search(rule)
        and _READ_ONLY_ACTION_RE.search(activity)
        and not _MUTATION_ACTION_RE.search(activity)
        for rule, activity in pairs
    )


def _action_frames(text: str) -> set[str]:
    return {
        action for action, pattern in _ACTION_PATTERNS.items() if pattern.search(text)
    }
