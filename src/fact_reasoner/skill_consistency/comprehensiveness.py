"""Graph construction and interpretation for documentation consistency.

The primary workflow runs forward FR2 for explicit documentation support and
reverse FR2 for direct requirement coverage. The accepted specification corpus
is the evaluation universe; this module does not perform model-based relevance
filtering. Paper-compatible context-context/SCC helpers remain available for
research comparison, but reverse FR2 does not claim those outputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence
from urllib.parse import parse_qsl, urlsplit

from .models import SemanticUnit


AtomKind = Literal["atom", "context"]
RelationLabel = Literal["entailment", "contradiction", "neutral", "equivalence"]
PairKind = Literal["context_atom", "atom_context", "context_context"]
EvidenceStatus = Literal[
    "supported",
    "contradicted",
    "mixed",
    "undetermined",
    "partial",
    "unresolved_high_ambiguity",
    "unassessed",
]


@dataclass(frozen=True, slots=True)
class FactAtom:
    """One revised statement in the relation graph.

    ``kind`` is the stable project role used for provenance and visualization:
    ``atom`` identifies a statement produced from code and ``context`` identifies
    a statement produced from the skill specification.  A reverse FR2 run swaps
    their *FactReasoner* claim/evidence roles without changing these project roles.
    """

    id: str
    text: str
    kind: AtomKind
    source: str | None = None
    relevance: float | None = None
    component: str | None = None
    semantic_role: str | None = None
    source_file: str | None = None
    source_lines: tuple[int, ...] = ()
    excerpt: str | None = None
    observation_ids: tuple[str, ...] = ()
    claim_id: str | None = None
    claim_ids: tuple[str, ...] = ()
    unit_id: str | None = None
    grounding_status: str = "not_applicable"
    capability_types: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    granularity: str = "behavior"
    assembly_rule: str | None = None
    polarity: str = "positive"
    content_form: str | None = None
    documentation_role: str = "not_applicable"
    atomization_rule: str | None = None
    semantic_member_count: int | None = None
    source_spans: tuple[dict[str, Any], ...] = ()
    scope_binding: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Atom identifiers must not be empty")
        if not self.text.strip():
            raise ValueError(f"Atom {self.id!r} has no text")
        if self.kind not in {"atom", "context"}:
            raise ValueError(f"Unknown atom kind: {self.kind!r}")
        if self.relevance is not None and not 1.0 <= self.relevance <= 5.0:
            raise ValueError("Relevance scores must be between 1 and 5")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "kind": self.kind,
            "source": self.source,
            "relevance": self.relevance,
            "component": self.component,
            "semantic_role": self.semantic_role,
            "source_file": self.source_file,
            "source_lines": list(self.source_lines),
            "excerpt": self.excerpt,
            "source_spans": list(self.source_spans),
            "scope_binding": self.scope_binding,
            "observation_ids": list(self.observation_ids),
            "claim_id": self.claim_id,
            "claim_ids": list(self.claim_ids),
            "unit_id": self.unit_id,
            "grounding_status": self.grounding_status,
            "capability_types": list(self.capability_types),
            "entities": list(self.entities),
            "granularity": self.granularity,
            "assembly_rule": self.assembly_rule,
            "polarity": self.polarity,
            "content_form": self.content_form,
            "documentation_role": self.documentation_role,
            "atomization_rule": self.atomization_rule,
            "semantic_member_count": self.semantic_member_count,
        }


@dataclass(frozen=True, slots=True)
class RelationPair:
    """One premise and hypothesis pair required by the NLI metric."""

    premise_id: str
    hypothesis_id: str
    kind: PairKind
    selection_method: str = "exhaustive"
    selection_rank: int | None = None
    selection_score: float | None = None
    selection_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "premise_id": self.premise_id,
            "hypothesis_id": self.hypothesis_id,
            "kind": self.kind,
            "selection_method": self.selection_method,
            "selection_rank": self.selection_rank,
            "selection_score": self.selection_score,
            "selection_reasons": list(self.selection_reasons),
        }


@dataclass(frozen=True, slots=True)
class FactRelation:
    """A directed relation from ``source_id`` to ``target_id``."""

    source_id: str
    target_id: str
    label: RelationLabel
    probability: float | None = None
    raw_label: RelationLabel | None = None
    raw_probability: float | None = None
    policy_action: str = "unchanged"
    policy_reasons: tuple[str, ...] = ()
    review_flags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.label not in {
            "entailment",
            "contradiction",
            "neutral",
            "equivalence",
        }:
            raise ValueError(f"Unknown relation label: {self.label!r}")
        if self.probability is not None and not 0.0 <= self.probability <= 1.0:
            raise ValueError("Relation probabilities must be between 0 and 1")
        if self.raw_label is not None and self.raw_label not in {
            "entailment",
            "contradiction",
            "neutral",
            "equivalence",
        }:
            raise ValueError(f"Unknown raw relation label: {self.raw_label!r}")
        if self.raw_probability is not None and not 0.0 <= self.raw_probability <= 1.0:
            raise ValueError("Raw relation probabilities must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "label": self.label,
            "probability": self.probability,
            "raw_label": self.raw_label,
            "raw_probability": self.raw_probability,
            "policy_action": self.policy_action,
            "policy_reasons": list(self.policy_reasons),
            "review_flags": list(self.review_flags),
        }


@dataclass(frozen=True, slots=True)
class CondensedComponent:
    """A strongly connected component in the entailment graph."""

    id: str
    representative_id: str
    member_ids: tuple[str, ...]
    atom_representative_id: str | None
    context_representative_id: str | None

    @property
    def contains_atom(self) -> bool:
        return self.atom_representative_id is not None

    @property
    def contains_context(self) -> bool:
        return self.context_representative_id is not None


@dataclass(frozen=True, slots=True)
class CondensedGraph:
    """The acyclic graph obtained after merging equivalent statements."""

    components: tuple[CondensedComponent, ...]
    edges: tuple[tuple[str, str], ...]
    component_by_atom_id: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "components": [asdict(component) for component in self.components],
            "edges": [list(edge) for edge in self.edges],
            "component_by_atom_id": dict(self.component_by_atom_id),
        }


@dataclass(frozen=True, slots=True)
class ComprehensivenessResult:
    """Covered, missing, and minimal missing specification statements."""

    score: float | None
    covered_context_ids: tuple[str, ...]
    missing_context_ids: tuple[str, ...]
    missing_basis_ids: tuple[str, ...]
    context_component_count: int
    graph: CondensedGraph

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "covered_context_ids": list(self.covered_context_ids),
            "missing_context_ids": list(self.missing_context_ids),
            "missing_basis_ids": list(self.missing_basis_ids),
            "covered_count": len(self.covered_context_ids),
            "missing_count": len(self.missing_context_ids),
            "context_component_count": self.context_component_count,
            "graph": self.graph.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class CodeSupportResult:
    """A separate factuality-style score for code statements."""

    score: float | None
    supported_atom_ids: tuple[str, ...]
    unsupported_atom_ids: tuple[str, ...]
    contradicted_atom_ids: tuple[str, ...]
    atom_component_count: int
    graph: CondensedGraph

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "supported_atom_ids": list(self.supported_atom_ids),
            "unsupported_atom_ids": list(self.unsupported_atom_ids),
            "contradicted_atom_ids": list(self.contradicted_atom_ids),
            "supported_count": len(self.supported_atom_ids),
            "unsupported_count": len(self.unsupported_atom_ids),
            "atom_component_count": self.atom_component_count,
            "graph": self.graph.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class AtomEvidenceAssessment:
    """Conservative interpretation of documentation evidence for one atom."""

    atom_id: str
    status: EvidenceStatus
    entailing_context_ids: tuple[str, ...]
    contradicting_context_ids: tuple[str, ...]
    neutral_context_ids: tuple[str, ...]
    support_review_flags: tuple[str, ...] = ()
    native_fr2_marginal_probability: float | None = None
    evaluated_relation_count: int = 0
    withheld_high_ambiguity_context_ids: tuple[str, ...] = ()

    @property
    def posterior_probability(self) -> float | None:
        """Compatibility alias; the value never determines the guarded state."""

        return self.native_fr2_marginal_probability

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in (
            "entailing_context_ids",
            "contradicting_context_ids",
            "neutral_context_ids",
            "support_review_flags",
            "withheld_high_ambiguity_context_ids",
        ):
            value[key] = list(value[key])
        value["native_fr2_diagnostic_note"] = (
            "raw FR2 marginal; not transformed by project relation guards and "
            "not used to assign the guarded evidence status"
        )
        return value


@dataclass(frozen=True, slots=True)
class UnitEvidenceAssessment:
    """Support aggregated over the required atoms of one behavior."""

    unit_id: str
    status: EvidenceStatus
    atom_ids: tuple[str, ...]
    supported_atom_ids: tuple[str, ...]
    contradicted_atom_ids: tuple[str, ...]
    undetermined_atom_ids: tuple[str, ...]
    unresolved_high_ambiguity_atom_ids: tuple[str, ...]
    unassessed_atom_ids: tuple[str, ...]
    support_fraction: float | None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in (
            "atom_ids",
            "supported_atom_ids",
            "contradicted_atom_ids",
            "undetermined_atom_ids",
            "unresolved_high_ambiguity_atom_ids",
            "unassessed_atom_ids",
        ):
            value[key] = list(value[key])
        return value


@dataclass(frozen=True, slots=True)
class Fr2SupportInterpretation:
    """Evidence states for forward-FR2 atoms and behavior-level support."""

    atom_assessments: tuple[AtomEvidenceAssessment, ...]
    behavior_assessments: tuple[UnitEvidenceAssessment, ...]
    atom_explicit_support_rate: float | None
    behavior_strict_support_rate: float | None
    behavior_fractional_support_rate: float | None
    grounded_atom_count: int
    unassessed_atom_count: int
    grounded_behavior_count: int
    unassessed_behavior_count: int
    pairing_mode: str = "exhaustive"

    @property
    def behavior_strict_support_lower_bound(self) -> float | None:
        denominator = len(self.behavior_assessments)
        return (
            sum(item.status == "supported" for item in self.behavior_assessments)
            / denominator
            if denominator
            else None
        )

    @property
    def behavior_strict_support_upper_bound(self) -> float | None:
        denominator = len(self.behavior_assessments)
        return (
            sum(
                item.status in {"supported", "unresolved_high_ambiguity", "unassessed"}
                for item in self.behavior_assessments
            )
            / denominator
            if denominator
            else None
        )

    @property
    def behavior_assessed_fraction(self) -> float | None:
        denominator = len(self.behavior_assessments)
        return (
            sum(
                item.status not in {"unresolved_high_ambiguity", "unassessed"}
                for item in self.behavior_assessments
            )
            / denominator
            if denominator
            else None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "atom_assessments": [item.to_dict() for item in self.atom_assessments],
            "behavior_assessments": [
                item.to_dict() for item in self.behavior_assessments
            ],
            "atom_explicit_support_rate": self.atom_explicit_support_rate,
            "behavior_strict_support_rate": self.behavior_strict_support_rate,
            "behavior_fractional_support_rate": (self.behavior_fractional_support_rate),
            "behavior_strict_support_lower_bound": (
                self.behavior_strict_support_lower_bound
            ),
            "behavior_strict_support_upper_bound": (
                self.behavior_strict_support_upper_bound
            ),
            "behavior_assessed_fraction": self.behavior_assessed_fraction,
            "grounded_atom_count": self.grounded_atom_count,
            "unassessed_atom_count": self.unassessed_atom_count,
            "grounded_behavior_count": self.grounded_behavior_count,
            "unassessed_behavior_count": self.unassessed_behavior_count,
            "pairing_mode": self.pairing_mode,
            "interpretation_note": (
                "neutral means evaluated documentation evidence is insufficient, not "
                "that the code behavior is false or unauthorized. A node with no "
                "evaluated candidate relation is unassessed, not neutral. A target "
                "whose selected judgments were all withheld is unresolved due to "
                "high ambiguity, not neutral or unassessed"
            ),
        }


@dataclass(frozen=True, slots=True)
class RequirementCoverageAssessment:
    """Strict and fractional implementation coverage for one requirement unit."""

    requirement_id: str
    status: Literal[
        "covered",
        "partial",
        "mixed",
        "not_evidenced",
        "contradicted",
        "unresolved_high_ambiguity",
        "unassessed",
    ]
    context_ids: tuple[str, ...]
    covered_context_ids: tuple[str, ...]
    contradicted_context_ids: tuple[str, ...]
    mixed_context_ids: tuple[str, ...]
    evidence_atom_ids: tuple[str, ...]
    coverage_fraction: float | None
    documentation_role: str = "not_applicable"
    parent_context_id: str | None = None
    parent_status: str = "unassessed"
    member_context_ids: tuple[str, ...] = ()
    covered_member_context_ids: tuple[str, ...] = ()
    contradicted_member_context_ids: tuple[str, ...] = ()
    mixed_member_context_ids: tuple[str, ...] = ()
    member_coverage_fraction: float | None = None
    native_fr2_diagnostics: Mapping[str, Any] | None = None
    evaluated_parent_relation_count: int = 0
    evaluated_member_relation_count: int = 0
    withheld_high_ambiguity_context_ids: tuple[str, ...] = ()
    withheld_high_ambiguity_evidence_atom_ids: tuple[str, ...] = ()

    @property
    def posterior_min_probability(self) -> float | None:
        diagnostics = self.native_fr2_diagnostics or {}
        value = diagnostics.get("minimum_marginal_probability")
        return float(value) if value is not None else None

    @property
    def posterior_mean_probability(self) -> float | None:
        diagnostics = self.native_fr2_diagnostics or {}
        value = diagnostics.get("mean_marginal_probability")
        return float(value) if value is not None else None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in (
            "context_ids",
            "covered_context_ids",
            "contradicted_context_ids",
            "mixed_context_ids",
            "evidence_atom_ids",
            "member_context_ids",
            "covered_member_context_ids",
            "contradicted_member_context_ids",
            "mixed_member_context_ids",
            "withheld_high_ambiguity_context_ids",
            "withheld_high_ambiguity_evidence_atom_ids",
        ):
            value[key] = list(value[key])
        return value


@dataclass(frozen=True, slots=True)
class RequirementComprehensivenessResult:
    """Requirement-level comprehensiveness without relevance filtering."""

    strict_parent_score: float | None
    member_fractional_score: float | None
    assessments: tuple[RequirementCoverageAssessment, ...]
    requirement_count: int
    context_count: int
    by_documentation_role: tuple[dict[str, Any], ...] = ()
    pairing_mode: str = "exhaustive"

    @property
    def strict_score(self) -> float | None:
        """Compatibility alias for the explicitly named parent score."""

        return self.strict_parent_score

    @property
    def fractional_score(self) -> float | None:
        """Compatibility alias for the explicitly named member score."""

        return self.member_fractional_score

    @property
    def strict_parent_upper_bound(self) -> float | None:
        return (
            sum(
                item.status in {"covered", "unresolved_high_ambiguity", "unassessed"}
                for item in self.assessments
            )
            / self.requirement_count
            if self.requirement_count
            else None
        )

    @property
    def assessed_requirement_fraction(self) -> float | None:
        return (
            sum(
                item.status not in {"unresolved_high_ambiguity", "unassessed"}
                for item in self.assessments
            )
            / self.requirement_count
            if self.requirement_count
            else None
        )

    def to_dict(self) -> dict[str, Any]:
        conflicts = [
            item.requirement_id
            for item in self.assessments
            if item.contradicted_context_ids or item.mixed_context_ids
        ]
        return {
            "strict_parent_requirement_score": self.strict_parent_score,
            "strict_parent_requirement_lower_bound": self.strict_parent_score,
            "strict_parent_requirement_upper_bound": self.strict_parent_upper_bound,
            "assessed_requirement_fraction": self.assessed_requirement_fraction,
            "member_level_fractional_score": self.member_fractional_score,
            "strict_score": self.strict_parent_score,
            "fractional_score": self.member_fractional_score,
            "assessments": [item.to_dict() for item in self.assessments],
            "requirement_count": self.requirement_count,
            "context_count": self.context_count,
            "by_documentation_role": list(self.by_documentation_role),
            "pairing_mode": self.pairing_mode,
            "requirements_with_conflicts": conflicts,
            "conflict_count": len(conflicts),
            "denominator_policy": (
                "all structurally accepted requirement units; no model relevance filter"
            ),
            "conflict_policy": (
                "the complete parent requirement determines strict coverage; member "
                "relations determine fractional coverage. A mixed or contradicted "
                "parent/member prevents strict coverage"
            ),
            "interpretation_note": (
                "guarded relations determine project scores. Native FR2 marginals are "
                "stored separately as diagnostics and are never transformed into states. "
                "Adaptive ranges are conditional sensitivity ranges under the "
                "unassessed-unit convention, not confidence intervals or guaranteed "
                "exhaustive-graph bounds. Selector recall remains unvalidated"
            ),
        }


def required_relation_pairs(
    atoms: Sequence[FactAtom],
    contexts: Sequence[FactAtom],
) -> list[RelationPair]:
    """Enumerate all relation checks required by NLI comprehensiveness.

    The returned list contains both directions for every code and context pair,
    plus both ordered directions between distinct context atoms.  Code to code
    pairs are intentionally absent.
    """

    _validate_partition(atoms, contexts)
    pairs: list[RelationPair] = []
    for atom in atoms:
        for context in contexts:
            pairs.append(RelationPair(context.id, atom.id, "context_atom"))
            pairs.append(RelationPair(atom.id, context.id, "atom_context"))
    for source in contexts:
        for target in contexts:
            if source.id != target.id:
                pairs.append(RelationPair(source.id, target.id, "context_context"))
    return pairs


def required_fr2_relation_pairs(
    atoms: Sequence[FactAtom],
    contexts: Sequence[FactAtom],
) -> list[RelationPair]:
    """Enumerate the forward FR2 specification-context to code-atom pairs."""

    _validate_partition(atoms, contexts)
    return [
        RelationPair(context.id, atom.id, "context_atom")
        for atom in atoms
        for context in contexts
    ]


def required_reverse_fr2_relation_pairs(
    atoms: Sequence[FactAtom],
    contexts: Sequence[FactAtom],
) -> list[RelationPair]:
    """Enumerate reverse FR2 code-evidence to requirement-claim pairs.

    Project roles stay unchanged so code nodes remain ``atom`` nodes and
    documentation nodes remain ``context`` nodes.  The FR2 adapter swaps those
    roles only in its serialized payload.
    """

    _validate_partition(atoms, contexts)
    return [
        RelationPair(atom.id, context.id, "atom_context")
        for atom in atoms
        for context in contexts
    ]


def relations_from_results(
    pairs: Sequence[RelationPair],
    results: Sequence[Mapping[str, Any]],
) -> list[FactRelation]:
    """Join positionally aligned NLI results to their statement identifiers."""

    if len(pairs) != len(results):
        raise ValueError(
            f"Expected {len(pairs)} NLI results but received {len(results)}"
        )
    relations: list[FactRelation] = []
    for pair, result in zip(pairs, results, strict=True):
        label = str(result.get("label", "")).strip().lower()
        if label not in {"entailment", "contradiction", "neutral", "equivalence"}:
            raise ValueError(
                f"Unknown NLI label {label!r} for {pair.premise_id} -> "
                f"{pair.hypothesis_id}"
            )
        probability_value = result.get("probability")
        probability = (
            float(probability_value) if probability_value is not None else None
        )
        relations.append(
            FactRelation(
                source_id=pair.premise_id,
                target_id=pair.hypothesis_id,
                label=label,  # type: ignore[arg-type]
                probability=probability,
            )
        )
    return relations


def condense_entailment_graph(
    atoms: Sequence[FactAtom],
    relations: Sequence[FactRelation],
) -> CondensedGraph:
    """Merge strongly connected entailment components into a directed acyclic graph."""

    atom_by_id = _validate_graph(atoms, relations)
    order = {atom.id: index for index, atom in enumerate(atoms)}
    adjacency = _entailment_adjacency(atoms, relations)
    raw_components = _strongly_connected_components(
        [atom.id for atom in atoms], adjacency
    )
    raw_components.sort(key=lambda members: min(order[item] for item in members))

    components: list[CondensedComponent] = []
    component_by_atom: dict[str, str] = {}
    for index, members in enumerate(raw_components):
        ordered_members = tuple(sorted(members, key=order.__getitem__))
        context_members = tuple(
            item for item in ordered_members if atom_by_id[item].kind == "context"
        )
        atom_members = tuple(
            item for item in ordered_members if atom_by_id[item].kind == "atom"
        )
        component_id = f"scc_{index:03d}"
        representative = context_members[0] if context_members else atom_members[0]
        component = CondensedComponent(
            id=component_id,
            representative_id=representative,
            member_ids=ordered_members,
            atom_representative_id=atom_members[0] if atom_members else None,
            context_representative_id=(context_members[0] if context_members else None),
        )
        components.append(component)
        for member in ordered_members:
            component_by_atom[member] = component_id

    condensed_edges: set[tuple[str, str]] = set()
    for source, targets in adjacency.items():
        source_component = component_by_atom[source]
        for target in targets:
            target_component = component_by_atom[target]
            if source_component != target_component:
                condensed_edges.add((source_component, target_component))

    component_order = {
        component.id: index for index, component in enumerate(components)
    }
    edges = tuple(
        sorted(
            condensed_edges,
            key=lambda edge: (
                component_order[edge[0]],
                component_order[edge[1]],
            ),
        )
    )
    return CondensedGraph(
        components=tuple(components),
        edges=edges,
        component_by_atom_id=component_by_atom,
    )


def score_nli_comprehensiveness(
    atoms: Sequence[FactAtom],
    relations: Sequence[FactRelation],
) -> ComprehensivenessResult:
    """Measure the share of accepted specification SCCs reached from atoms."""

    graph = condense_entailment_graph(atoms, relations)
    adjacency = _component_adjacency(graph)
    response_components = {
        component.id for component in graph.components if component.contains_atom
    }
    context_components = [
        component for component in graph.components if component.contains_context
    ]

    reachable_from_response: set[str] = set()
    for component_id in response_components:
        reachable_from_response.update(_reachable(component_id, adjacency))

    covered = [
        component
        for component in context_components
        if component.id in reachable_from_response
    ]
    missing = [
        component
        for component in context_components
        if component.id not in reachable_from_response
    ]

    missing_ids = {component.id for component in missing}
    basis: list[CondensedComponent] = []
    for target in missing:
        has_more_specific_missing = any(
            source_id != target.id and target.id in _reachable(source_id, adjacency)
            for source_id in missing_ids
        )
        if not has_more_specific_missing:
            basis.append(target)

    denominator = len(context_components)
    score = len(covered) / denominator if denominator else None
    return ComprehensivenessResult(
        score=score,
        covered_context_ids=tuple(
            component.context_representative_id  # type: ignore[misc]
            for component in covered
        ),
        missing_context_ids=tuple(
            component.context_representative_id  # type: ignore[misc]
            for component in missing
        ),
        missing_basis_ids=tuple(
            component.context_representative_id  # type: ignore[misc]
            for component in basis
        ),
        context_component_count=denominator,
        graph=graph,
    )


def score_code_support(
    atoms: Sequence[FactAtom],
    relations: Sequence[FactRelation],
) -> CodeSupportResult:
    """Measure the share of code facts entailed by the skill specification.

    This is a factuality-style companion measure.  It is not the reverse form
    of the paper's comprehensiveness score.
    """

    graph = condense_entailment_graph(atoms, relations)
    adjacency = _component_adjacency(graph)
    context_components = {
        component.id for component in graph.components if component.contains_context
    }
    atom_components = [
        component for component in graph.components if component.contains_atom
    ]

    reachable_from_context: set[str] = set()
    for component_id in context_components:
        reachable_from_context.update(_reachable(component_id, adjacency))

    supported = [
        component
        for component in atom_components
        if component.id in reachable_from_context
    ]
    unsupported = [
        component
        for component in atom_components
        if component.id not in reachable_from_context
    ]

    contradicted_components: set[str] = set()
    atom_by_id = {atom.id: atom for atom in atoms}
    for relation in relations:
        if relation.label != "contradiction":
            continue
        source = atom_by_id[relation.source_id]
        target = atom_by_id[relation.target_id]
        if source.kind == "atom":
            contradicted_components.add(graph.component_by_atom_id[source.id])
        if target.kind == "atom":
            contradicted_components.add(graph.component_by_atom_id[target.id])

    denominator = len(atom_components)
    score = len(supported) / denominator if denominator else None
    return CodeSupportResult(
        score=score,
        supported_atom_ids=tuple(
            component.atom_representative_id  # type: ignore[misc]
            for component in supported
        ),
        unsupported_atom_ids=tuple(
            component.atom_representative_id  # type: ignore[misc]
            for component in unsupported
        ),
        contradicted_atom_ids=tuple(
            component.atom_representative_id  # type: ignore[misc]
            for component in atom_components
            if component.id in contradicted_components
        ),
        atom_component_count=denominator,
        graph=graph,
    )


def interpret_fr2_support(
    nodes: Sequence[FactAtom],
    relations: Sequence[FactRelation],
    *,
    behavior_units: Sequence[SemanticUnit] = (),
    posterior_by_atom_id: Mapping[str, float] | None = None,
    pairing_mode: str = "exhaustive",
    withheld_high_ambiguity_pairs: Sequence[RelationPair] = (),
) -> Fr2SupportInterpretation:
    """Interpret forward FR2 results without treating neutral as contradiction.

    Direct context-to-atom labels provide auditable evidence sets.  Optional
    Merlin marginals are retained as diagnostics, but do not erase a mixed or
    undetermined relation state.
    """

    node_by_id = _validate_graph(nodes, relations)
    posterior = posterior_by_atom_id or {}
    withheld_by_atom: dict[str, set[str]] = {}
    for pair in withheld_high_ambiguity_pairs:
        withheld_by_atom.setdefault(pair.hypothesis_id, set()).add(pair.premise_id)
    atom_nodes = [item for item in nodes if item.kind == "atom"]
    assessments: list[AtomEvidenceAssessment] = []
    for atom in atom_nodes:
        entailing: set[str] = set()
        contradicting: set[str] = set()
        neutral: set[str] = set()
        review_flags: set[str] = set()
        evaluated_relation_count = 0
        for relation in relations:
            if relation.target_id != atom.id:
                continue
            source = node_by_id[relation.source_id]
            if source.kind != "context":
                continue
            evaluated_relation_count += 1
            if relation.label in {"entailment", "equivalence"}:
                flags = _support_review_flags(source, atom)
                review_flags.update(flags)
                if "conflicting_explicit_url" in flags:
                    neutral.add(source.id)
                    review_flags.add("entailment_downgraded_by_entity_guard")
                else:
                    entailing.add(source.id)
            elif relation.label == "contradiction":
                contradicting.add(source.id)
            elif relation.label == "neutral":
                neutral.add(source.id)

        withheld_contexts = tuple(sorted(withheld_by_atom.get(atom.id, ())))
        if atom.grounding_status not in {"grounded", "not_applicable"}:
            status: EvidenceStatus = "unassessed"
        elif evaluated_relation_count == 0 and withheld_contexts:
            status = "unresolved_high_ambiguity"
        elif evaluated_relation_count == 0:
            status = "unassessed"
        elif entailing and contradicting:
            status = "mixed"
        elif entailing:
            status = "supported"
        elif contradicting:
            status = "contradicted"
        else:
            status = "undetermined"
        assessments.append(
            AtomEvidenceAssessment(
                atom_id=atom.id,
                status=status,
                entailing_context_ids=tuple(sorted(entailing)),
                contradicting_context_ids=tuple(sorted(contradicting)),
                neutral_context_ids=tuple(sorted(neutral)),
                support_review_flags=tuple(sorted(review_flags)),
                native_fr2_marginal_probability=posterior.get(atom.id),
                evaluated_relation_count=evaluated_relation_count,
                withheld_high_ambiguity_context_ids=withheld_contexts,
            )
        )

    assessment_by_atom = {item.atom_id: item for item in assessments}
    node_ids_by_claim: dict[str, list[str]] = {}
    for node in atom_nodes:
        member_claim_ids = node.claim_ids or (node.claim_id or node.id,)
        for claim_id in member_claim_ids:
            node_ids_by_claim.setdefault(claim_id, []).append(node.id)

    unit_assessments: list[UnitEvidenceAssessment] = []
    for unit in behavior_units:
        member_ids = tuple(
            dict.fromkeys(
                node_id
                for claim_id in unit.claim_ids
                for node_id in node_ids_by_claim.get(claim_id, ())
            )
        )
        if not member_ids:
            member_ids = tuple(
                node.id for node in atom_nodes if node.unit_id == unit.id
            )
        member_results = [
            assessment_by_atom[item]
            for item in member_ids
            if item in assessment_by_atom
        ]
        supported = tuple(
            item.atom_id for item in member_results if item.status == "supported"
        )
        contradicted = tuple(
            item.atom_id
            for item in member_results
            if item.status in {"contradicted", "mixed"}
        )
        undetermined = tuple(
            item.atom_id for item in member_results if item.status == "undetermined"
        )
        unresolved = tuple(
            item.atom_id
            for item in member_results
            if item.status == "unresolved_high_ambiguity"
        )
        unassessed = tuple(
            item.atom_id for item in member_results if item.status == "unassessed"
        )
        statuses = {item.status for item in member_results}
        if (
            unit.grounding_status != "grounded"
            or not member_results
            or statuses == {"unassessed"}
        ):
            unit_status: EvidenceStatus = "unassessed"
        elif "mixed" in statuses or (supported and contradicted):
            unit_status = "mixed"
        elif contradicted:
            unit_status = "contradicted"
        elif supported and len(supported) == len(member_results):
            unit_status = "supported"
        elif supported:
            unit_status = "partial"
        elif statuses == {"unresolved_high_ambiguity"}:
            unit_status = "unresolved_high_ambiguity"
        else:
            unit_status = "undetermined"
        support_fraction = (
            len(supported) / len(member_results)
            if unit_status != "unassessed" and member_results
            else None
        )
        unit_assessments.append(
            UnitEvidenceAssessment(
                unit_id=unit.id,
                status=unit_status,
                atom_ids=member_ids,
                supported_atom_ids=supported,
                contradicted_atom_ids=contradicted,
                undetermined_atom_ids=undetermined,
                unresolved_high_ambiguity_atom_ids=unresolved,
                unassessed_atom_ids=unassessed,
                support_fraction=support_fraction,
            )
        )

    grounded = [item for item in assessments if item.status != "unassessed"]
    atom_rate = (
        sum(item.status == "supported" for item in grounded) / len(grounded)
        if grounded
        else None
    )
    assessed_units = [item for item in unit_assessments if item.status != "unassessed"]
    behavior_rate = (
        sum(item.status == "supported" for item in assessed_units) / len(assessed_units)
        if assessed_units
        else None
    )
    behavior_fractional_rate = (
        sum(item.support_fraction or 0.0 for item in assessed_units)
        / len(assessed_units)
        if assessed_units
        else None
    )
    return Fr2SupportInterpretation(
        atom_assessments=tuple(assessments),
        behavior_assessments=tuple(unit_assessments),
        atom_explicit_support_rate=atom_rate,
        behavior_strict_support_rate=behavior_rate,
        behavior_fractional_support_rate=behavior_fractional_rate,
        grounded_atom_count=len(grounded),
        unassessed_atom_count=len(assessments) - len(grounded),
        grounded_behavior_count=len(assessed_units),
        unassessed_behavior_count=len(unit_assessments) - len(assessed_units),
        pairing_mode=pairing_mode,
    )


def score_requirement_comprehensiveness(
    nodes: Sequence[FactAtom],
    relations: Sequence[FactRelation],
    requirement_units: Sequence[SemanticUnit],
    *,
    posterior_by_context_id: Mapping[str, float] | None = None,
    pairing_mode: str = "exhaustive",
    withheld_high_ambiguity_pairs: Sequence[RelationPair] = (),
) -> RequirementComprehensivenessResult:
    """Score complete parent requirements and their atomic members separately.

    Guarded relations alone determine the project states. ``posterior_by_context_id``
    is copied into a clearly named native-FR2 diagnostic object and never affects a
    status or score.  Legacy graphs without explicit parent/member nodes retain the
    previous all-members-strict behavior.
    """

    node_by_id = _validate_graph(nodes, relations)
    native_marginals = posterior_by_context_id or {}
    context_nodes = [item for item in nodes if item.kind == "context"]
    parent_by_unit = {
        item.unit_id: item
        for item in context_nodes
        if item.unit_id and item.granularity == "requirement_unit"
    }
    members_by_claim: dict[str, list[FactAtom]] = {}
    legacy_by_claim: dict[str, list[FactAtom]] = {}
    for node in context_nodes:
        member_claim_ids = node.claim_ids or (node.claim_id or node.id,)
        if node.granularity == "requirement_member":
            destination = members_by_claim
        elif node.granularity != "requirement_unit":
            destination = legacy_by_claim
        else:
            continue
        for claim_id in member_claim_ids:
            destination.setdefault(claim_id, []).append(node)

    incoming: dict[str, list[FactRelation]] = {node.id: [] for node in context_nodes}
    withheld_by_context: dict[str, set[str]] = {}
    for pair in withheld_high_ambiguity_pairs:
        withheld_by_context.setdefault(pair.hypothesis_id, set()).add(pair.premise_id)
    for relation in relations:
        source = node_by_id[relation.source_id]
        target = node_by_id[relation.target_id]
        if source.kind == "atom" and target.kind == "context":
            incoming[target.id].append(relation)

    def state_for(context: FactAtom) -> tuple[str, set[str]]:
        relevant = [
            item
            for item in incoming[context.id]
            if node_by_id[item.source_id].grounding_status
            in {"grounded", "not_applicable"}
        ]
        if not relevant:
            if withheld_by_context.get(context.id):
                return "unresolved_high_ambiguity", set()
            return "unassessed", set()
        supporting: list[FactRelation] = []
        conflicts: list[FactRelation] = []
        for relation in relevant:
            if relation.label in {"entailment", "equivalence"}:
                flags = _support_review_flags(context, node_by_id[relation.source_id])
                if "conflicting_explicit_url" not in flags:
                    supporting.append(relation)
            elif relation.label == "contradiction":
                conflicts.append(relation)
        evidence = {item.source_id for item in (*supporting, *conflicts)}
        if supporting and conflicts:
            return "mixed", evidence
        if supporting:
            return "covered", evidence
        if conflicts:
            return "contradicted", evidence
        return "not_evidenced", evidence

    assessments: list[RequirementCoverageAssessment] = []
    all_context_ids: set[str] = set()
    for unit in requirement_units:
        parent = parent_by_unit.get(unit.id)
        member_nodes = [
            node
            for claim_id in unit.claim_ids
            for node in members_by_claim.get(claim_id, ())
        ]
        member_nodes = list({item.id: item for item in member_nodes}.values())
        if not member_nodes:
            # An atomic parent may retain multiple duplicate provenance IDs.
            # Only explicitly constructed member nodes create a conjunction.
            if parent is not None and (
                parent.semantic_member_count == 1
                or (parent.semantic_member_count is None and len(unit.claim_ids) == 1)
            ):
                member_nodes = [parent]
            elif parent is None:
                member_nodes = [
                    node
                    for claim_id in unit.claim_ids
                    for node in legacy_by_claim.get(claim_id, ())
                ]
                member_nodes = list({item.id: item for item in member_nodes}.values())

        context_ids = tuple(
            dict.fromkeys(
                ([parent.id] if parent is not None else [])
                + [item.id for item in member_nodes]
            )
        )
        all_context_ids.update(context_ids)
        if not context_ids:
            assessments.append(
                RequirementCoverageAssessment(
                    requirement_id=unit.id,
                    status="unassessed",
                    context_ids=(),
                    covered_context_ids=(),
                    contradicted_context_ids=(),
                    mixed_context_ids=(),
                    evidence_atom_ids=(),
                    coverage_fraction=None,
                    documentation_role=unit.documentation_role,
                )
            )
            continue

        parent_state, parent_evidence = (
            state_for(parent) if parent is not None else ("unassessed", set())
        )
        member_states = {item.id: state_for(item) for item in member_nodes}
        covered_members = tuple(
            item for item, (state, _) in member_states.items() if state == "covered"
        )
        contradicted_members = tuple(
            item
            for item, (state, _) in member_states.items()
            if state == "contradicted"
        )
        mixed_members = tuple(
            item for item, (state, _) in member_states.items() if state == "mixed"
        )
        member_fraction = (
            len(covered_members) / len(member_nodes) if member_nodes else None
        )

        if parent is not None:
            strict_covered = (
                parent_state == "covered"
                and bool(member_nodes)
                and len(covered_members) == len(member_nodes)
                and not (contradicted_members or mixed_members)
            )
            member_state_values = {state for state, _ in member_states.values()}
            if parent_state == "mixed" or mixed_members:
                status = "mixed"
            elif parent_state == "contradicted" or contradicted_members:
                status = "contradicted"
            elif strict_covered:
                status = "covered"
            elif covered_members:
                status = "partial"
            elif parent_state == "unresolved_high_ambiguity" or member_state_values == {
                "unresolved_high_ambiguity"
            }:
                status = "unresolved_high_ambiguity"
            elif parent_state == "unassessed" or member_state_values == {"unassessed"}:
                status = "unassessed"
            else:
                status = "not_evidenced"
        else:
            strict_covered = bool(member_nodes) and len(covered_members) == len(
                member_nodes
            )
            if mixed_members or (covered_members and contradicted_members):
                status = "mixed"
            elif contradicted_members:
                status = "contradicted"
            elif strict_covered:
                status = "covered"
            elif covered_members:
                status = "partial"
            elif member_states and {state for state, _ in member_states.values()} == {
                "unresolved_high_ambiguity"
            }:
                status = "unresolved_high_ambiguity"
            elif member_states and {state for state, _ in member_states.values()} == {
                "unassessed"
            }:
                status = "unassessed"
            else:
                status = "not_evidenced"

        evidence = set(parent_evidence)
        for _, member_evidence in member_states.values():
            evidence.update(member_evidence)
        covered_contexts = tuple(
            dict.fromkeys(
                (
                    [parent.id]
                    if parent is not None and parent_state == "covered"
                    else []
                )
                + list(covered_members)
            )
        )
        contradicted_contexts = tuple(
            dict.fromkeys(
                (
                    [parent.id]
                    if parent is not None and parent_state == "contradicted"
                    else []
                )
                + list(contradicted_members)
            )
        )
        mixed_contexts = tuple(
            dict.fromkeys(
                ([parent.id] if parent is not None and parent_state == "mixed" else [])
                + list(mixed_members)
            )
        )
        probabilities = [
            float(native_marginals[item])
            for item in context_ids
            if item in native_marginals
        ]
        diagnostics: dict[str, Any] | None = None
        if probabilities:
            diagnostics = {
                "source": "native_factreasoner_v2_marginals",
                "used_for_guarded_state": False,
                "minimum_marginal_probability": min(probabilities),
                "mean_marginal_probability": sum(probabilities) / len(probabilities),
                "values_by_context_id": {
                    item: float(native_marginals[item])
                    for item in context_ids
                    if item in native_marginals
                },
            }
        withheld_context_ids = tuple(
            item for item in context_ids if withheld_by_context.get(item)
        )
        withheld_evidence_ids = tuple(
            sorted(
                {
                    source_id
                    for context_id in withheld_context_ids
                    for source_id in withheld_by_context[context_id]
                }
            )
        )
        assessments.append(
            RequirementCoverageAssessment(
                requirement_id=unit.id,
                status=status,  # type: ignore[arg-type]
                context_ids=context_ids,
                covered_context_ids=covered_contexts,
                contradicted_context_ids=contradicted_contexts,
                mixed_context_ids=mixed_contexts,
                evidence_atom_ids=tuple(sorted(evidence)),
                coverage_fraction=member_fraction,
                documentation_role=unit.documentation_role,
                parent_context_id=parent.id if parent is not None else None,
                parent_status=parent_state,
                member_context_ids=tuple(item.id for item in member_nodes),
                covered_member_context_ids=covered_members,
                contradicted_member_context_ids=contradicted_members,
                mixed_member_context_ids=mixed_members,
                member_coverage_fraction=member_fraction,
                native_fr2_diagnostics=diagnostics,
                evaluated_parent_relation_count=(
                    len(incoming[parent.id]) if parent is not None else 0
                ),
                evaluated_member_relation_count=sum(
                    len(incoming[item.id]) for item in member_nodes
                ),
                withheld_high_ambiguity_context_ids=withheld_context_ids,
                withheld_high_ambiguity_evidence_atom_ids=withheld_evidence_ids,
            )
        )

    denominator = len(requirement_units)
    strict_parent_score = (
        sum(item.status == "covered" for item in assessments) / denominator
        if denominator
        else None
    )
    member_fractional_score = (
        sum(item.member_coverage_fraction or 0.0 for item in assessments) / denominator
        if denominator
        else None
    )
    role_rows: list[dict[str, Any]] = []
    for role in sorted({item.documentation_role for item in assessments}):
        members = [item for item in assessments if item.documentation_role == role]
        role_rows.append(
            {
                "documentation_role": role,
                "requirement_count": len(members),
                "strict_parent_requirement_score": (
                    sum(item.status == "covered" for item in members) / len(members)
                ),
                "strict_parent_requirement_upper_bound": (
                    sum(
                        item.status
                        in {"covered", "unresolved_high_ambiguity", "unassessed"}
                        for item in members
                    )
                    / len(members)
                ),
                "assessed_requirement_fraction": (
                    sum(
                        item.status not in {"unresolved_high_ambiguity", "unassessed"}
                        for item in members
                    )
                    / len(members)
                ),
                "member_level_fractional_score": (
                    sum(item.member_coverage_fraction or 0.0 for item in members)
                    / len(members)
                ),
            }
        )
    return RequirementComprehensivenessResult(
        strict_parent_score=strict_parent_score,
        member_fractional_score=member_fractional_score,
        assessments=tuple(assessments),
        requirement_count=denominator,
        context_count=len(all_context_ids),
        by_documentation_role=tuple(role_rows),
        pairing_mode=pairing_mode,
    )


def _support_review_flags(context: FactAtom, atom: FactAtom) -> set[str]:
    flags: set[str] = set()
    shared = set(context.capability_types) & set(atom.capability_types)
    shared.discard("general_behavior")
    if context.capability_types and atom.capability_types and not shared:
        flags.add("no_shared_capability_type")
    context_urls = {
        item for item in context.entities if item.startswith(("http://", "https://"))
    }
    atom_urls = {
        item for item in atom.entities if item.startswith(("http://", "https://"))
    }
    if (
        context_urls
        and atom_urls
        and not any(
            _comparable_url(left, right) for left in context_urls for right in atom_urls
        )
    ):
        flags.add("conflicting_explicit_url")
    if context.granularity != atom.granularity:
        flags.add("cross_granularity_support")
    return flags


def _comparable_url(first: str, second: str) -> bool:
    """Allow one dynamic path/query value while preserving endpoint identity."""

    left = urlsplit(first.rstrip("/"))
    right = urlsplit(second.rstrip("/"))
    if (left.scheme.casefold(), left.netloc.casefold()) != (
        right.scheme.casefold(),
        right.netloc.casefold(),
    ):
        return False
    left_parts = tuple(item for item in left.path.split("/") if item)
    right_parts = tuple(item for item in right.path.split("/") if item)
    if len(left_parts) != len(right_parts):
        return False
    for left_item, right_item in zip(left_parts, right_parts, strict=True):
        if left_item == right_item:
            continue
        if not any(
            marker in left_item + right_item for marker in ("{", "}", "$", "<", ">")
        ):
            return False
    left_keys = {key for key, _ in parse_qsl(left.query, keep_blank_values=True)}
    right_keys = {key for key, _ in parse_qsl(right.query, keep_blank_values=True)}
    return not left_keys or not right_keys or left_keys == right_keys


def _validate_partition(
    atoms: Sequence[FactAtom], contexts: Sequence[FactAtom]
) -> None:
    if any(atom.kind != "atom" for atom in atoms):
        raise ValueError("The response partition may contain only code atoms")
    if any(context.kind != "context" for context in contexts):
        raise ValueError("The context partition may contain only specification atoms")
    identifiers = [item.id for item in (*atoms, *contexts)]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Atom identifiers must be unique")


def _validate_graph(
    atoms: Sequence[FactAtom], relations: Sequence[FactRelation]
) -> dict[str, FactAtom]:
    atom_by_id = {atom.id: atom for atom in atoms}
    if len(atom_by_id) != len(atoms):
        raise ValueError("Atom identifiers must be unique")
    unknown = {
        endpoint
        for relation in relations
        for endpoint in (relation.source_id, relation.target_id)
        if endpoint not in atom_by_id
    }
    if unknown:
        raise ValueError(f"Relations reference unknown atoms: {sorted(unknown)}")
    return atom_by_id


def _entailment_adjacency(
    atoms: Sequence[FactAtom], relations: Sequence[FactRelation]
) -> dict[str, set[str]]:
    adjacency = {atom.id: set() for atom in atoms}
    for relation in relations:
        if relation.label == "entailment":
            adjacency[relation.source_id].add(relation.target_id)
        elif relation.label == "equivalence":
            adjacency[relation.source_id].add(relation.target_id)
            adjacency[relation.target_id].add(relation.source_id)
    return adjacency


def _strongly_connected_components(
    nodes: Iterable[str], adjacency: Mapping[str, set[str]]
) -> list[list[str]]:
    index = 0
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in adjacency[node]:
            if neighbor not in indices:
                visit(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif neighbor in on_stack:
                lowlink[node] = min(lowlink[node], indices[neighbor])

        if lowlink[node] != indices[node]:
            return
        component: list[str] = []
        while stack:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == node:
                break
        components.append(component)

    for node in nodes:
        if node not in indices:
            visit(node)
    return components


def _component_adjacency(graph: CondensedGraph) -> dict[str, set[str]]:
    adjacency = {component.id: set() for component in graph.components}
    for source, target in graph.edges:
        adjacency[source].add(target)
    return adjacency


def _reachable(source: str, adjacency: Mapping[str, set[str]]) -> set[str]:
    visited: set[str] = set()
    pending = [source]
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        pending.extend(adjacency[current] - visited)
    return visited
