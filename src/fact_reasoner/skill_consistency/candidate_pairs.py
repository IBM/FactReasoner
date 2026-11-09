"""Deterministic, auditable candidate selection for project FR2 runs.

The adaptive selector has no fixed number of candidates per claim.  It retains
every pair with a declared semantic anchor and statistically screens the
remaining lexical overlaps with a hypergeometric null model and Benjamini-
Hochberg false-discovery-rate control.  The same rule is used in both FR2
directions.

The selector sees normalized statements, typed capabilities, and typed entities.
It never sees code excerpts, compiler mappings, expected labels, NLI scores, or
previous graph outcomes.  Selection changes which relations are evaluated; it
does not create or modify an NLI label.  A pair that is not selected has no NLI
label, and a claim with no selected pair is explicitly unassessed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Any, Literal, Sequence
from urllib.parse import parse_qsl, urlsplit

from .comprehensiveness import FactAtom, FactRelation, PairKind, RelationPair


PairingMode = Literal["adaptive", "exhaustive"]
PAIR_SELECTOR_ID = "semantic-anchor-lexical-fdr"
PAIR_SELECTOR_VERSION = "2.0.0"
DEFAULT_LEXICAL_FALSE_DISCOVERY_RATE = 0.05
DEFAULT_SELECTOR_SENSITIVITY_RATES = (0.01, 0.05, 0.10)


@dataclass(frozen=True, slots=True)
class PairSelectorPolicy:
    """Versioned policy for adaptive relation eligibility.

    The FDR is an interpretable statistical error-control level, not a request
    budget.  ``MAX_NOTEBOOK_NLI_JUDGMENTS`` remains a separate abort-only guard.
    """

    selector_id: str = PAIR_SELECTOR_ID
    selector_version: str = PAIR_SELECTOR_VERSION
    lexical_null_model: str = "hypergeometric_token_set_overlap"
    multiple_testing: str = "benjamini_hochberg"
    lexical_false_discovery_rate: float = DEFAULT_LEXICAL_FALSE_DISCOVERY_RATE
    unconditional_anchors: tuple[str, ...] = (
        "shared_non_general_capability",
        "shared_exact_typed_entity",
        "compatible_dynamic_url_entity",
    )
    forced_fallback: bool = False
    budget_behavior: str = "abort_without_truncation"

    def __post_init__(self) -> None:
        if not 0.0 < self.lexical_false_discovery_rate < 1.0:
            raise ValueError(
                "lexical_false_discovery_rate must be strictly between 0 and 1"
            )

    @property
    def policy_sha256(self) -> str:
        payload = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "policy_sha256": self.policy_sha256}


DEFAULT_PAIR_SELECTOR_POLICY = PairSelectorPolicy()

_GENERAL_CAPABILITIES = {
    "behavioral_configuration",
    "control_flow_constraint",
    "general_behavior",
    "model_instruction",
    "output_generation",
    "requirement_enforcement",
    "user_interaction",
}
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "code",
    "does",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "requirement",
    "skill",
    "that",
    "the",
    "their",
    "this",
    "to",
    "uses",
    "using",
    "with",
}
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.:/{}$<>-]*", re.IGNORECASE)
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)


def select_candidate_relation_pairs(
    evidence_nodes: Sequence[FactAtom],
    claim_nodes: Sequence[FactAtom],
    *,
    kind: PairKind,
    false_discovery_rate: float = DEFAULT_LEXICAL_FALSE_DISCOVERY_RATE,
) -> tuple[RelationPair, ...]:
    """Select a variable-size evidence set for every claim hypothesis.

    Semantic anchors are always retained.  Unanchored pairs are retained only
    when their token-set overlap survives Benjamini-Hochberg FDR control across
    all unanchored pairs in this direction.  Capability disagreement alone does
    not exclude a pair: an independently significant lexical relation remains
    eligible.  No fallback pair is fabricated for an unmatched claim.
    """

    if kind not in {"context_atom", "atom_context"}:
        raise ValueError("Candidate FR2 pairing supports context-atom directions only")
    _validate_false_discovery_rate(false_discovery_rate)
    expected_evidence_kind, expected_claim_kind = (
        ("context", "atom") if kind == "context_atom" else ("atom", "context")
    )
    if any(item.kind != expected_evidence_kind for item in evidence_nodes):
        raise ValueError(f"{kind} received an invalid evidence partition")
    if any(item.kind != expected_claim_kind for item in claim_nodes):
        raise ValueError(f"{kind} received an invalid claim partition")
    identifiers = [item.id for item in (*evidence_nodes, *claim_nodes)]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Candidate pairing requires globally unique node IDs")
    if not evidence_nodes or not claim_nodes:
        return ()

    evidence_tokens = {item.id: _tokens(item.text) for item in evidence_nodes}
    claim_tokens = {item.id: _tokens(item.text) for item in claim_nodes}
    document_frequency = Counter(
        token for tokens in evidence_tokens.values() for token in tokens
    )
    corpus_size = len(evidence_nodes)
    token_universe = set().union(*evidence_tokens.values(), *claim_tokens.values())
    universe_size = max(1, len(token_universe))

    anchored: dict[tuple[str, str], tuple[float, tuple[str, ...]]] = {}
    lexical_tests: list[tuple[float, str, str, float, tuple[str, ...]]] = []
    for claim in claim_nodes:
        for evidence in evidence_nodes:
            score, reasons = _candidate_score(
                evidence,
                claim,
                evidence_tokens[evidence.id],
                claim_tokens[claim.id],
                document_frequency,
                corpus_size,
            )
            pair_id = (evidence.id, claim.id)
            if _has_semantic_anchor(reasons):
                anchored[pair_id] = (score, reasons)
                continue
            overlap = len(evidence_tokens[evidence.id] & claim_tokens[claim.id])
            p_value = _hypergeometric_overlap_probability(
                universe_size,
                len(evidence_tokens[evidence.id]),
                len(claim_tokens[claim.id]),
                overlap,
            )
            lexical_tests.append((p_value, evidence.id, claim.id, score, reasons))

    lexical_selected = _benjamini_hochberg_selected(
        lexical_tests, false_discovery_rate=false_discovery_rate
    )
    selected: dict[tuple[str, str], tuple[float, tuple[str, ...]]] = dict(anchored)
    for p_value, evidence_id, claim_id, score, reasons in lexical_tests:
        pair_id = (evidence_id, claim_id)
        if pair_id not in lexical_selected:
            continue
        significance = -math.log10(max(p_value, 1e-300))
        selected[pair_id] = (
            max(score, significance),
            (
                *reasons,
                f"lexical_overlap_p:{p_value:.12g}",
                f"benjamini_hochberg_fdr:{false_discovery_rate:.6g}",
            ),
        )

    result: list[RelationPair] = []
    for claim in claim_nodes:
        ranked = sorted(
            (
                (-score, evidence_id, reasons)
                for (evidence_id, claim_id), (score, reasons) in selected.items()
                if claim_id == claim.id
            ),
            key=lambda item: (item[0], item[1]),
        )
        for rank, (negative_score, evidence_id, reasons) in enumerate(ranked, start=1):
            result.append(
                RelationPair(
                    premise_id=evidence_id,
                    hypothesis_id=claim.id,
                    kind=kind,
                    selection_method=(f"{PAIR_SELECTOR_ID}:v{PAIR_SELECTOR_VERSION}"),
                    selection_rank=rank,
                    selection_score=round(-negative_score, 6),
                    selection_reasons=reasons,
                )
            )
    return tuple(result)


def pairing_report(
    pairs: Sequence[RelationPair],
    *,
    possible_pair_count: int,
    claim_nodes: Sequence[FactAtom],
    mode: PairingMode,
    false_discovery_rate: float | None,
) -> dict[str, Any]:
    """Return an auditable cost and target-coverage report."""

    counts = Counter(item.hypothesis_id for item in pairs)
    claim_ids = [item.id for item in claim_nodes]
    selected = len(pairs)
    possible = max(0, possible_pair_count)
    return {
        "mode": mode,
        "selector_id": PAIR_SELECTOR_ID if mode == "adaptive" else None,
        "selector_version": PAIR_SELECTOR_VERSION if mode == "adaptive" else None,
        "selector_policy": (
            PairSelectorPolicy(
                lexical_false_discovery_rate=(
                    false_discovery_rate
                    if false_discovery_rate is not None
                    else DEFAULT_LEXICAL_FALSE_DISCOVERY_RATE
                )
            ).to_dict()
            if mode == "adaptive"
            else None
        ),
        "selector_inputs": (
            "normalized statements, capability types, entities, polarity"
            if mode == "adaptive"
            else "not applicable"
        ),
        "selector_excluded_inputs": (
            "code excerpts, compiler mappings, expected labels, NLI outputs, scores"
            if mode == "adaptive"
            else "not applicable"
        ),
        "lexical_false_discovery_rate": false_discovery_rate,
        "fixed_candidates_per_claim": None,
        "possible_pairs": possible,
        "selected_pairs": selected,
        "pruned_pairs": max(0, possible - selected),
        "selected_fraction": selected / possible if possible else None,
        "claim_count": len(claim_ids),
        "claims_with_candidates": sum(counts[item] > 0 for item in claim_ids),
        "claims_without_candidates": [item for item in claim_ids if counts[item] == 0],
        "forced_fallback_pair_count": 0,
        "target_retrieval_coverage": (
            sum(counts[item] > 0 for item in claim_ids) / len(claim_ids)
            if claim_ids
            else None
        ),
        "interpretation": (
            "adaptive-mode scores are conservative lower bounds; a target with no "
            "selected pair is unassessed, and selector recall must be validated "
            "against an exhaustive same-model relation ledger"
            if mode == "adaptive"
            else "all declared cross-partition pairs were evaluated"
        ),
        "per_claim_candidate_counts": {item: counts[item] for item in claim_ids},
        "selected_pair_provenance": [item.to_dict() for item in pairs],
    }


def selector_sensitivity_report(
    evidence_nodes: Sequence[FactAtom],
    claim_nodes: Sequence[FactAtom],
    *,
    kind: PairKind,
    false_discovery_rates: Sequence[float] = DEFAULT_SELECTOR_SENSITIVITY_RATES,
) -> tuple[dict[str, Any], ...]:
    """Report cost and target coverage across predeclared FDR values.

    This is an offline selector diagnostic.  It does not call NLI and does not
    treat a larger selected set as semantically more accurate.
    """

    possible = len(evidence_nodes) * len(claim_nodes)
    rows: list[dict[str, Any]] = []
    for false_discovery_rate in false_discovery_rates:
        pairs = select_candidate_relation_pairs(
            evidence_nodes,
            claim_nodes,
            kind=kind,
            false_discovery_rate=false_discovery_rate,
        )
        report = pairing_report(
            pairs,
            possible_pair_count=possible,
            claim_nodes=claim_nodes,
            mode="adaptive",
            false_discovery_rate=false_discovery_rate,
        )
        rows.append(
            {
                "lexical_false_discovery_rate": false_discovery_rate,
                "selected_pairs": report["selected_pairs"],
                "selected_fraction": report["selected_fraction"],
                "claim_count": report["claim_count"],
                "claims_with_candidates": report["claims_with_candidates"],
                "target_retrieval_coverage": report["target_retrieval_coverage"],
                "claims_without_candidates": report["claims_without_candidates"],
            }
        )
    return tuple(rows)


def calibrate_pair_selector(
    evidence_nodes: Sequence[FactAtom],
    claim_nodes: Sequence[FactAtom],
    exhaustive_relations: Sequence[FactRelation],
    *,
    kind: PairKind,
    false_discovery_rates: Sequence[float] = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.10,
        0.20,
    ),
    minimum_decisive_recall: float = 0.99,
) -> dict[str, Any]:
    """Choose the least-cost FDR policy meeting a declared recall target.

    The reference is an exhaustive run of the *same NLI model and prompt*.
    Consequently this calibrates pruning fidelity, not semantic accuracy.  A
    calibration result should be evaluated again on a held-out exhaustive
    ledger before it is treated as validated.
    """

    if not 0.0 < minimum_decisive_recall <= 1.0:
        raise ValueError("minimum_decisive_recall must be in (0, 1]")
    rows: list[dict[str, Any]] = []
    for false_discovery_rate in false_discovery_rates:
        pairs = select_candidate_relation_pairs(
            evidence_nodes,
            claim_nodes,
            kind=kind,
            false_discovery_rate=false_discovery_rate,
        )
        recall = evaluate_candidate_recall(pairs, exhaustive_relations)
        rows.append(
            {
                "lexical_false_discovery_rate": false_discovery_rate,
                "selected_pair_count": len(pairs),
                "target_retrieval_coverage": (
                    len({item.hypothesis_id for item in pairs}) / len(claim_nodes)
                    if claim_nodes
                    else None
                ),
                "decisive_relation_count": recall["decisive_relation_count"],
                "decisive_relation_recall": recall["decisive_relation_recall"],
                "by_label": recall["by_label"],
            }
        )
    eligible = [
        item
        for item in rows
        if item["decisive_relation_recall"] is not None
        and item["decisive_relation_recall"] >= minimum_decisive_recall
    ]
    chosen = min(
        eligible,
        key=lambda item: (
            item["selected_pair_count"],
            item["lexical_false_discovery_rate"],
        ),
        default=None,
    )
    return {
        "calibration_id": "same-model-exhaustive-selector-calibration",
        "calibration_version": "1.0.0",
        "human_gold": False,
        "semantic_accuracy_estimate": False,
        "reference": "same-model exhaustive relation ledger",
        "selector_id": PAIR_SELECTOR_ID,
        "selector_version": PAIR_SELECTOR_VERSION,
        "minimum_decisive_relation_recall": minimum_decisive_recall,
        "status": (
            "selected" if chosen is not None else "no_tested_policy_meets_recall_target"
        ),
        "chosen_policy": (
            PairSelectorPolicy(
                lexical_false_discovery_rate=chosen["lexical_false_discovery_rate"]
            ).to_dict()
            if chosen is not None
            else None
        ),
        "grid_results": rows,
        "interpretation": (
            "This result measures whether pruning retains decisive labels from "
            "the same model. It is not human-gold NLI accuracy, and the chosen "
            "policy requires held-out exhaustive validation."
        ),
    }


def evaluate_candidate_recall(
    candidate_pairs: Sequence[RelationPair],
    exhaustive_relations: Sequence[FactRelation],
) -> dict[str, Any]:
    """Measure whether a candidate plan retains an exhaustive run's decisions.

    This is a selector-recall diagnostic, not semantic accuracy: the exhaustive
    model ledger is a counterfactual reference, not human ground truth.
    """

    selected = {(item.premise_id, item.hypothesis_id) for item in candidate_pairs}
    decisive = [item for item in exhaustive_relations if item.label != "neutral"]
    retained = [
        item for item in decisive if (item.source_id, item.target_id) in selected
    ]
    by_label: dict[str, dict[str, Any]] = {}
    for label in ("entailment", "contradiction", "equivalence"):
        members = [item for item in decisive if item.label == label]
        kept = [
            item for item in members if (item.source_id, item.target_id) in selected
        ]
        by_label[label] = {
            "exhaustive_count": len(members),
            "retained_count": len(kept),
            "recall": len(kept) / len(members) if members else None,
        }
    return {
        "reference": "same-model exhaustive relation ledger",
        "human_gold": False,
        "interpretation": (
            "candidate-selector recall of decisive model relations; not NLI accuracy"
        ),
        "decisive_relation_count": len(decisive),
        "retained_decisive_relation_count": len(retained),
        "decisive_relation_recall": (
            len(retained) / len(decisive) if decisive else None
        ),
        "by_label": by_label,
        "missed_decisive_relations": [
            item.to_dict()
            for item in decisive
            if (item.source_id, item.target_id) not in selected
        ],
    }


def _validate_false_discovery_rate(value: float) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError("false_discovery_rate must be strictly between 0 and 1")


def _has_semantic_anchor(reasons: Sequence[str]) -> bool:
    return any(
        reason.startswith(
            (
                "shared_capability:",
                "shared_entity:",
                "compatible_dynamic_entity:",
            )
        )
        for reason in reasons
    )


def _hypergeometric_overlap_probability(
    universe_size: int,
    evidence_token_count: int,
    claim_token_count: int,
    observed_overlap: int,
) -> float:
    """Return P[X >= observed] under random token-set overlap.

    Token sets are sampled without replacement from the vocabulary shared by
    this direction.  Zero overlap has probability one and therefore can never
    pass the lexical selector by itself.
    """

    if observed_overlap <= 0:
        return 1.0
    if universe_size < 1:
        return 1.0
    evidence_token_count = min(evidence_token_count, universe_size)
    claim_token_count = min(claim_token_count, universe_size)
    upper = min(evidence_token_count, claim_token_count)
    denominator = math.comb(universe_size, claim_token_count)
    if denominator == 0:
        return 1.0
    numerator = 0
    for overlap in range(observed_overlap, upper + 1):
        remaining = claim_token_count - overlap
        unavailable = universe_size - evidence_token_count
        if remaining < 0 or remaining > unavailable:
            continue
        numerator += math.comb(evidence_token_count, overlap) * math.comb(
            unavailable, remaining
        )
    return min(1.0, numerator / denominator)


def _benjamini_hochberg_selected(
    tests: Sequence[tuple[float, str, str, float, tuple[str, ...]]],
    *,
    false_discovery_rate: float,
) -> set[tuple[str, str]]:
    """Return pair IDs passing a deterministic BH step-up procedure."""

    _validate_false_discovery_rate(false_discovery_rate)
    ordered = sorted(tests, key=lambda item: (item[0], item[1], item[2]))
    test_count = len(ordered)
    cutoff: float | None = None
    for rank, item in enumerate(ordered, start=1):
        p_value = item[0]
        if p_value <= false_discovery_rate * rank / test_count:
            cutoff = p_value
    if cutoff is None:
        return set()
    return {
        (evidence_id, claim_id)
        for p_value, evidence_id, claim_id, _, _ in ordered
        if p_value <= cutoff
    }


def _candidate_score(
    evidence: FactAtom,
    claim: FactAtom,
    evidence_tokens: set[str],
    claim_tokens: set[str],
    document_frequency: Counter[str],
    corpus_size: int,
) -> tuple[float, tuple[str, ...]]:
    reasons: list[str] = []
    score = 0.0

    evidence_capabilities = set(evidence.capability_types)
    claim_capabilities = set(claim.capability_types)
    substantive_overlap = (evidence_capabilities - _GENERAL_CAPABILITIES) & (
        claim_capabilities - _GENERAL_CAPABILITIES
    )
    general_overlap = (evidence_capabilities & claim_capabilities) - substantive_overlap
    if substantive_overlap:
        score += 5.0 * len(substantive_overlap)
        reasons.extend(
            f"shared_capability:{item}" for item in sorted(substantive_overlap)
        )
    if general_overlap:
        score += 0.5 * len(general_overlap)
        reasons.extend(
            f"shared_general_type:{item}" for item in sorted(general_overlap)
        )

    exact_entities, compatible_entities = _shared_entities(evidence, claim)
    if exact_entities:
        score += 12.0 * len(exact_entities)
        reasons.extend(f"shared_entity:{item}" for item in exact_entities)
    if compatible_entities:
        score += 10.0 * len(compatible_entities)
        reasons.extend(
            f"compatible_dynamic_entity:{item}" for item in compatible_entities
        )

    shared_tokens = evidence_tokens & claim_tokens
    if shared_tokens:
        lexical_weight = sum(
            math.log((corpus_size + 1) / (document_frequency[token] + 1)) + 1.0
            for token in shared_tokens
        )
        normalization = math.sqrt(
            max(1, len(evidence_tokens)) * max(1, len(claim_tokens))
        )
        score += 10.0 * lexical_weight / normalization
        reasons.append(f"idf_token_overlap:{len(shared_tokens)}")

    if evidence.polarity != claim.polarity and (
        substantive_overlap or exact_entities or compatible_entities or shared_tokens
    ):
        score += 1.0
        reasons.append("opposite_polarity_same_scope")

    return score, tuple(reasons)


def _tokens(text: str) -> set[str]:
    result: set[str] = set()
    for raw_token in _TOKEN_RE.findall(text.casefold()):
        candidates = [
            raw_token,
            *re.split(r"[_.:/{}$<>-]+", raw_token),
        ]
        for token in candidates:
            if token in _STOPWORDS or len(token) <= 1:
                continue
            result.add(token)
            if len(token) > 4 and token.endswith("ies"):
                result.add(token[:-3] + "y")
            elif len(token) > 4 and token.endswith("s"):
                result.add(token[:-1])
    return result


def _shared_entities(
    first: FactAtom, second: FactAtom
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    first_entities = {
        item.strip().casefold(): item for item in first.entities if item.strip()
    }
    second_entities = {
        item.strip().casefold(): item for item in second.entities if item.strip()
    }
    exact_keys = sorted(set(first_entities) & set(second_entities))
    exact = tuple(first_entities[item] for item in exact_keys)

    compatible: list[str] = []
    first_urls = [item for item in first.entities if _URL_RE.match(item)]
    second_urls = [item for item in second.entities if _URL_RE.match(item)]
    for left in first_urls:
        for right in second_urls:
            if left.casefold().rstrip("/") == right.casefold().rstrip("/"):
                continue
            if _urls_compatible(left, right):
                compatible.append(f"{left}<->{right}")
    return exact, tuple(sorted(set(compatible)))


def _urls_compatible(first: str, second: str) -> bool:
    left = urlsplit(first.rstrip("/"))
    right = urlsplit(second.rstrip("/"))
    if (left.scheme.casefold(), left.netloc.casefold()) != (
        right.scheme.casefold(),
        right.netloc.casefold(),
    ):
        return False
    left_segments = tuple(item for item in left.path.split("/") if item)
    right_segments = tuple(item for item in right.path.split("/") if item)
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
