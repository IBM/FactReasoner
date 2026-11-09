"""Materialize MICR evidence as explicit FactReasoner analysis views."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Iterable, Sequence

from .candidate_pairs import (
    DEFAULT_LEXICAL_FALSE_DISCOVERY_RATE,
    PairingMode,
    pairing_report,
    select_candidate_relation_pairs,
)
from .comprehensiveness import (
    FactAtom,
    RelationPair,
    required_fr2_relation_pairs,
    required_reverse_fr2_relation_pairs,
)
from .models import EvidenceClaim, SemanticUnit
from .skill_atoms import SkillAtomExtraction, eligible_requirement_claims


@dataclass(frozen=True)
class AnalysisView:
    """One declared partitioning of evidence for a specific question."""

    id: str
    question: str
    atoms: tuple[FactAtom, ...]
    contexts: tuple[FactAtom, ...]
    requirement_contexts: tuple[FactAtom, ...] = ()
    requirement_member_contexts: tuple[FactAtom, ...] = ()
    documentation_bundles: tuple[FactAtom, ...] = ()
    comprehensiveness_bundles: tuple[FactAtom, ...] = ()
    implementation_units: tuple[SemanticUnit, ...] = ()
    requirement_units: tuple[SemanticUnit, ...] = ()
    reverse_evidence_override: tuple[FactAtom, ...] | None = None
    pairing_mode: PairingMode = "exhaustive"
    lexical_false_discovery_rate: float = DEFAULT_LEXICAL_FALSE_DISCOVERY_RATE

    def __post_init__(self) -> None:
        if self.pairing_mode not in {"adaptive", "exhaustive"}:
            raise ValueError(f"Unknown pairing mode: {self.pairing_mode!r}")
        if not 0.0 < self.lexical_false_discovery_rate < 1.0:
            raise ValueError(
                "lexical_false_discovery_rate must be strictly between 0 and 1"
            )

    @property
    def nodes(self) -> tuple[FactAtom, ...]:
        """Stable project nodes used by the forward FR2 graph."""

        return (*self.atoms, *self.forward_contexts)

    @property
    def forward_contexts(self) -> tuple[FactAtom, ...]:
        """Individual declarations plus structurally justified conjunctions."""

        return (*self.contexts, *self.documentation_bundles)

    @property
    def forward_nodes(self) -> tuple[FactAtom, ...]:
        """Code atoms plus documentation contexts for forward FR2."""

        return self.nodes

    @property
    def comprehensiveness_atoms(self) -> tuple[FactAtom, ...]:
        """Code evidence used as contexts inside the reverse FR2 run."""

        if self.reverse_evidence_override is not None:
            return self.reverse_evidence_override
        return (*self.atoms, *self.comprehensiveness_bundles)

    @property
    def comprehensiveness_nodes(self) -> tuple[FactAtom, ...]:
        """Stable project nodes used by the reverse FR2 graph."""

        return (*self.comprehensiveness_atoms, *self.reverse_contexts)

    @property
    def reverse_contexts(self) -> tuple[FactAtom, ...]:
        """Complete requirement units used as reverse-FR2 claims."""

        if self.requirement_contexts:
            return (*self.requirement_contexts, *self.requirement_member_contexts)
        return self.contexts

    @property
    def reverse_nodes(self) -> tuple[FactAtom, ...]:
        """Alias naming the reverse FR2 graph without changing node identities."""

        return self.comprehensiveness_nodes

    @property
    def forward_fr2_pairs(self) -> tuple[RelationPair, ...]:
        if self.pairing_mode == "adaptive":
            return select_candidate_relation_pairs(
                self.forward_contexts,
                self.atoms,
                kind="context_atom",
                false_discovery_rate=self.lexical_false_discovery_rate,
            )
        return tuple(required_fr2_relation_pairs(self.atoms, self.forward_contexts))

    @property
    def reverse_fr2_pairs(self) -> tuple[RelationPair, ...]:
        if self.pairing_mode == "adaptive":
            return select_candidate_relation_pairs(
                self.comprehensiveness_atoms,
                self.reverse_contexts,
                kind="atom_context",
                false_discovery_rate=self.lexical_false_discovery_rate,
            )
        return tuple(
            required_reverse_fr2_relation_pairs(
                self.comprehensiveness_atoms, self.reverse_contexts
            )
        )

    @property
    def forward_pairing_report(self) -> dict[str, object]:
        return pairing_report(
            self.forward_fr2_pairs,
            possible_pair_count=len(self.atoms) * len(self.forward_contexts),
            claim_nodes=self.atoms,
            mode=self.pairing_mode,
            false_discovery_rate=(
                self.lexical_false_discovery_rate
                if self.pairing_mode == "adaptive"
                else None
            ),
        )

    @property
    def reverse_pairing_report(self) -> dict[str, object]:
        return pairing_report(
            self.reverse_fr2_pairs,
            possible_pair_count=(
                len(self.comprehensiveness_atoms) * len(self.reverse_contexts)
            ),
            claim_nodes=self.reverse_contexts,
            mode=self.pairing_mode,
            false_discovery_rate=(
                self.lexical_false_discovery_rate
                if self.pairing_mode == "adaptive"
                else None
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "question": self.question,
            "atoms": [item.to_dict() for item in self.atoms],
            "contexts": [item.to_dict() for item in self.contexts],
            "forward_contexts": [item.to_dict() for item in self.forward_contexts],
            "documentation_bundles": [
                item.to_dict() for item in self.documentation_bundles
            ],
            "requirement_contexts": [
                item.to_dict() for item in self.requirement_contexts
            ],
            "requirement_member_contexts": [
                item.to_dict() for item in self.requirement_member_contexts
            ],
            "reverse_contexts": [item.to_dict() for item in self.reverse_contexts],
            "reverse_evidence_override": (
                None
                if self.reverse_evidence_override is None
                else [n.to_dict() for n in self.reverse_evidence_override]
            ),
            "comprehensiveness_bundles": [
                item.to_dict() for item in self.comprehensiveness_bundles
            ],
            "implementation_units": [
                item.to_dict() for item in self.implementation_units
            ],
            "requirement_units": [item.to_dict() for item in self.requirement_units],
            "pairing": {
                "mode": self.pairing_mode,
                "lexical_false_discovery_rate": (self.lexical_false_discovery_rate),
                "forward": self.forward_pairing_report,
                "reverse": self.reverse_pairing_report,
            },
            "relation_counts": {
                "forward_fr2": len(self.forward_fr2_pairs),
                "reverse_fr2": len(self.reverse_fr2_pairs),
                "bidirectional_total": (
                    len(self.forward_fr2_pairs) + len(self.reverse_fr2_pairs)
                ),
            },
        }


def _node_from_claim(claim: EvidenceClaim, kind: str) -> FactAtom:
    return FactAtom(
        id=claim.id,
        text=claim.assertion,
        kind=kind,  # type: ignore[arg-type]
        source=f"{claim.source_file}:{','.join(map(str, claim.source_lines))}",
        component=claim.component,
        semantic_role=claim.semantic_role,
        source_file=claim.source_file,
        source_lines=claim.source_lines,
        excerpt=claim.excerpt,
        observation_ids=claim.observation_ids,
        claim_id=claim.id,
        claim_ids=(claim.id,),
        unit_id=claim.parent_unit_id,
        grounding_status=claim.grounding_status,
        capability_types=claim.behavior_families,
        entities=claim.entities,
        granularity=claim.granularity,
        polarity=claim.polarity,
        content_form=claim.content_form,
        documentation_role=claim.documentation_role,
        atomization_rule=claim.atomization_rule,
    )


def _nodes_from_claims(
    claims: Sequence[EvidenceClaim], kind: str
) -> tuple[FactAtom, ...]:
    """Consolidate exact claim text without losing any source occurrence."""

    order: list[str] = []
    grouped: dict[str, list[EvidenceClaim]] = {}
    for claim in claims:
        key = " ".join(claim.assertion.split()).casefold()
        if key not in grouped:
            order.append(key)
        grouped.setdefault(key, []).append(claim)

    nodes: list[FactAtom] = []
    for key in order:
        members = grouped[key]
        first = members[0]
        if len(members) == 1:
            nodes.append(_node_from_claim(first, kind))
            continue
        source_files = tuple(dict.fromkeys(item.source_file for item in members))
        excerpts = [
            f"{item.source_file}:{','.join(map(str, item.source_lines))}\n{item.excerpt}"
            for item in members
        ]
        nodes.append(
            FactAtom(
                id=first.id,
                text=first.assertion,
                kind=kind,  # type: ignore[arg-type]
                source="; ".join(source_files),
                component=(
                    first.component
                    if all(member.component == first.component for member in members)
                    else "mixed"
                ),
                semantic_role=(
                    first.semantic_role
                    if all(
                        item.semantic_role == first.semantic_role for item in members
                    )
                    else "mixed"
                ),
                source_file="; ".join(source_files),
                source_lines=tuple(
                    sorted({line for item in members for line in item.source_lines})
                ),
                excerpt="\n\n".join(excerpts),
                observation_ids=tuple(
                    dict.fromkeys(
                        observation_id
                        for item in members
                        for observation_id in item.observation_ids
                    )
                ),
                claim_ids=tuple(item.id for item in members),
                grounding_status=(
                    "grounded"
                    if all(item.grounding_status == "grounded" for item in members)
                    else "uncertain"
                ),
                capability_types=tuple(
                    sorted(
                        {
                            capability
                            for item in members
                            for capability in item.behavior_families
                        }
                    )
                ),
                entities=tuple(
                    sorted({entity for item in members for entity in item.entities})
                ),
                granularity=(
                    first.granularity
                    if all(item.granularity == first.granularity for item in members)
                    else "mixed"
                ),
                polarity=(
                    first.polarity
                    if all(item.polarity == first.polarity for item in members)
                    else "mixed"
                ),
                content_form=(
                    first.content_form
                    if all(item.content_form == first.content_form for item in members)
                    else "mixed"
                ),
                documentation_role=(
                    first.documentation_role
                    if all(
                        item.documentation_role == first.documentation_role
                        for item in members
                    )
                    else "mixed"
                ),
                atomization_rule=(
                    first.atomization_rule
                    if all(
                        item.atomization_rule == first.atomization_rule
                        for item in members
                    )
                    else "exact semantic duplicates consolidated"
                ),
            )
        )
    return tuple(nodes)


def node_from_unit(
    unit: SemanticUnit,
    claims_by_id: dict[str, EvidenceClaim],
) -> FactAtom:
    """Represent an explicit implementation bundle with full lineage."""

    members = [claims_by_id[item] for item in unit.claim_ids if item in claims_by_id]
    excerpts = [
        f"{item.source_file}:{','.join(map(str, item.source_lines))}\n{item.excerpt}"
        for item in members
    ]
    return FactAtom(
        id=unit.id,
        text=unit.statement,
        kind="atom",
        source="; ".join(unit.source_files),
        component=unit.component,
        semantic_role="implementation_bundle",
        source_file="; ".join(unit.source_files),
        source_lines=tuple(
            sorted({line for item in members for line in item.source_lines})
        ),
        excerpt="\n\n".join(excerpts) if excerpts else None,
        observation_ids=unit.observation_ids,
        claim_ids=unit.claim_ids,
        unit_id=unit.id,
        grounding_status=unit.grounding_status,
        capability_types=unit.capability_types,
        entities=tuple(
            sorted({entity for item in members for entity in item.entities})
        ),
        granularity="implementation_bundle",
        assembly_rule=unit.assembly_rule,
        polarity=(
            members[0].polarity
            if members and all(item.polarity == members[0].polarity for item in members)
            else "mixed"
        ),
        content_form="structural_bundle",
        documentation_role="not_applicable",
        atomization_rule="child claims supplied by structural bundle",
    )


def requirement_node_from_unit(
    unit: SemanticUnit,
    claims_by_id: dict[str, EvidenceClaim],
) -> FactAtom:
    """Represent one complete documentation requirement with child provenance."""

    members = [claims_by_id[item] for item in unit.claim_ids if item in claims_by_id]
    if not members:
        raise ValueError(f"Requirement unit {unit.id!r} has no known claims")
    identifier = "R_" + re.sub(r"[^A-Za-z0-9]+", "_", unit.id).strip("_")
    excerpts = [
        f"{item.source_file}:{','.join(map(str, item.source_lines))}\n{item.excerpt}"
        for item in members
    ]
    # Claim IDs record source occurrences, not independent truth conditions.
    by_statement: dict[str, EvidenceClaim] = {}
    for member in members:
        by_statement.setdefault(" ".join(member.assertion.split()).casefold(), member)
    distinct_members = list(by_statement.values())
    statement = (
        unit.statement
        if len(distinct_members) == 1
        else "Joint documentation requirement:\n"
        + "\n".join(f"- {item.assertion}" for item in distinct_members)
    )
    return FactAtom(
        id=identifier,
        text=statement,
        kind="context",
        source="; ".join(unit.source_files),
        component=unit.component,
        semantic_role=(
            members[0].semantic_role
            if all(item.semantic_role == members[0].semantic_role for item in members)
            else "mixed"
        ),
        source_file="; ".join(unit.source_files),
        source_lines=tuple(
            sorted({line for item in members for line in item.source_lines})
        ),
        excerpt="\n\n".join(excerpts),
        claim_ids=unit.claim_ids,
        unit_id=unit.id,
        grounding_status="not_applicable",
        capability_types=unit.capability_types,
        entities=tuple(
            sorted({entity for item in members for entity in item.entities})
        ),
        granularity="requirement_unit",
        assembly_rule=unit.assembly_rule,
        polarity=(
            members[0].polarity
            if all(item.polarity == members[0].polarity for item in members)
            else "mixed"
        ),
        content_form="documentation_requirement_unit",
        documentation_role=unit.documentation_role,
        atomization_rule="member claims combined as one strict parent requirement",
        semantic_member_count=len(distinct_members),
    )


def documentation_bundle_node_from_unit(
    unit: SemanticUnit,
    claims_by_id: dict[str, EvidenceClaim],
) -> FactAtom:
    """Represent a source-structural conjunction for forward documentation support."""

    node = requirement_node_from_unit(unit, claims_by_id)
    identifier = "D_" + re.sub(r"[^A-Za-z0-9]+", "_", unit.id).strip("_")
    return replace(
        node,
        id=identifier,
        semantic_role="documentation_bundle",
        granularity="documentation_bundle",
        content_form="structural_documentation_bundle",
        atomization_rule="child documentation claims combined as a conjunctive premise",
    )


def requirement_member_node(node: FactAtom) -> FactAtom:
    """Create an unambiguous reverse-FR2 node for one requirement member."""

    identifier = "RM_" + re.sub(r"[^A-Za-z0-9]+", "_", node.id).strip("_")
    return replace(
        node,
        id=identifier,
        text=f"Requirement member:\n{node.text}",
        granularity="requirement_member",
        content_form="documentation_requirement_member",
    )


def _has_multiple_distinct_claims(
    unit: SemanticUnit,
    claims_by_id: dict[str, EvidenceClaim],
) -> bool:
    statements = {
        " ".join(claims_by_id[claim_id].assertion.split()).casefold()
        for claim_id in unit.claim_ids
        if claim_id in claims_by_id
    }
    return len(statements) > 1


def build_documentation_code_view(
    skill: SkillAtomExtraction,
    code_claims: Sequence[EvidenceClaim],
    *,
    include_mechanisms: bool = True,
    implementation_units: Sequence[SemanticUnit] = (),
    pairing_mode: PairingMode = "adaptive",
    lexical_false_discovery_rate: float = DEFAULT_LEXICAL_FALSE_DISCOVERY_RATE,
) -> AnalysisView:
    """Ask whether full documented evidence supports grounded code behavior."""

    context_claims = eligible_requirement_claims(skill)
    selected_code = tuple(
        claim
        for claim in code_claims
        if claim.scoring_eligible
        and (include_mechanisms or claim.granularity != "mechanism")
    )
    scoring_claim_ids = {item.id for item in selected_code}
    selected_units = tuple(
        unit
        for unit in implementation_units
        if unit.unit_kind == "behavior"
        and any(claim_id in scoring_claim_ids for claim_id in unit.claim_ids)
    )
    requirement_units = _consolidate_exact_semantic_units(
        skill.requirement_units,
        assembly_rule=(
            "exact normalized requirement consolidation; every source "
            "occurrence is retained"
        ),
    )
    documentation_claims_by_id = {item.id: item for item in context_claims}
    return AnalysisView(
        id="documentation_supports_code",
        question="What grounded code behavior is explicitly supported by the skill?",
        atoms=_nodes_from_claims(selected_code, "atom"),
        contexts=_nodes_from_claims(context_claims, "context"),
        documentation_bundles=tuple(
            documentation_bundle_node_from_unit(unit, documentation_claims_by_id)
            for unit in requirement_units
            if _has_multiple_distinct_claims(unit, documentation_claims_by_id)
        ),
        implementation_units=selected_units,
        requirement_units=requirement_units,
        pairing_mode=pairing_mode,
        lexical_false_discovery_rate=lexical_false_discovery_rate,
    )


def build_metadata_instruction_view(skill: SkillAtomExtraction) -> AnalysisView:
    """Build the BIV-compatible M-to-I disclosure view."""

    contexts = tuple(
        item
        for item in skill.metadata
        if item.scoring_eligible
        and item.semantic_role in {"declaration", "requirement", "constraint"}
    )
    atoms = tuple(
        item
        for item in skill.instructions
        if item.scoring_eligible
        and item.semantic_role in {"operation", "requirement", "constraint"}
    )
    return AnalysisView(
        id="metadata_discloses_instructions",
        question="Which operational instructions are disclosed by skill metadata?",
        atoms=_nodes_from_claims(atoms, "atom"),
        contexts=_nodes_from_claims(contexts, "context"),
    )


def build_comprehensiveness_view(
    skill: SkillAtomExtraction,
    implementation_units: Sequence[SemanticUnit],
    code_claims: Sequence[EvidenceClaim],
    *,
    pairing_mode: PairingMode = "adaptive",
    lexical_false_discovery_rate: float = DEFAULT_LEXICAL_FALSE_DISCOVERY_RATE,
) -> AnalysisView:
    """Build the stable nodes for forward and reverse FR2.

    Every accepted requirement claim remains a project context. Forward FR2 uses
    those requirements as evidence for frozen code-atom claims. Reverse FR2 swaps
    the internal FactReasoner roles and treats grounded code atoms plus explicit
    structural bundles as evidence for requirement claims.
    """

    scoring_claims = tuple(item for item in code_claims if item.scoring_eligible)
    scoring_claim_ids = {item.id for item in scoring_claims}
    behavior_units = tuple(
        unit
        for unit in implementation_units
        if unit.unit_kind == "behavior"
        if any(claim_id in scoring_claim_ids for claim_id in unit.claim_ids)
    )
    bundle_units = _consolidate_exact_semantic_units(
        tuple(
            unit
            for unit in implementation_units
            if unit.unit_kind == "implementation_bundle"
        ),
        assembly_rule=(
            "exact normalized bundle consolidation; every code occurrence and "
            "child claim is retained"
        ),
    )
    claims_by_id = {item.id: item for item in code_claims}
    atoms = _nodes_from_claims(scoring_claims, "atom")
    bundles = tuple(node_from_unit(item, claims_by_id) for item in bundle_units)
    requirement_units = _consolidate_exact_semantic_units(
        skill.requirement_units,
        assembly_rule=(
            "exact normalized requirement consolidation; every source "
            "occurrence is retained"
        ),
    )
    requirement_claims = eligible_requirement_claims(skill)
    contexts = _nodes_from_claims(requirement_claims, "context")
    requirement_claims_by_id = {item.id: item for item in requirement_claims}
    requirement_contexts = tuple(
        requirement_node_from_unit(item, requirement_claims_by_id)
        for item in requirement_units
    )
    conjunctive_claim_ids = {
        claim_id
        for unit in requirement_units
        if _has_multiple_distinct_claims(unit, requirement_claims_by_id)
        for claim_id in unit.claim_ids
    }
    requirement_member_contexts = tuple(
        requirement_member_node(item)
        for item in contexts
        if conjunctive_claim_ids.intersection(
            item.claim_ids or (item.claim_id or item.id,)
        )
    )
    documentation_bundles = tuple(
        documentation_bundle_node_from_unit(item, requirement_claims_by_id)
        for item in requirement_units
        if _has_multiple_distinct_claims(item, requirement_claims_by_id)
    )
    return AnalysisView(
        id="requirement_comprehensiveness",
        question="Which declared requirements have grounded implementations?",
        atoms=atoms,
        contexts=contexts,
        requirement_contexts=requirement_contexts,
        requirement_member_contexts=requirement_member_contexts,
        documentation_bundles=documentation_bundles,
        comprehensiveness_bundles=bundles,
        implementation_units=behavior_units,
        requirement_units=requirement_units,
        pairing_mode=pairing_mode,
        lexical_false_discovery_rate=lexical_false_discovery_rate,
    )


def _consolidate_exact_semantic_units(
    units: Sequence[SemanticUnit],
    *,
    assembly_rule: str,
) -> tuple[SemanticUnit, ...]:
    """Evaluate identical semantic units once while retaining every occurrence."""

    order: list[str] = []
    grouped: dict[str, list[SemanticUnit]] = {}
    for unit in units:
        key = " ".join(unit.statement.split()).casefold()
        if key not in grouped:
            order.append(key)
        grouped.setdefault(key, []).append(unit)

    result: list[SemanticUnit] = []
    for key in order:
        members = grouped[key]
        first = members[0]
        if len(members) == 1:
            result.append(first)
            continue
        result.append(
            SemanticUnit(
                id=first.id,
                unit_kind=first.unit_kind,
                statement=first.statement,
                claim_ids=tuple(
                    dict.fromkeys(
                        claim_id for member in members for claim_id in member.claim_ids
                    )
                ),
                component=(
                    first.component
                    if all(member.component == first.component for member in members)
                    else "mixed"
                ),
                capability_types=tuple(
                    sorted(
                        {
                            capability
                            for member in members
                            for capability in member.capability_types
                        }
                    )
                ),
                observation_ids=tuple(
                    dict.fromkeys(
                        observation_id
                        for member in members
                        for observation_id in member.observation_ids
                    )
                ),
                source_files=tuple(
                    sorted(
                        {
                            source_file
                            for member in members
                            for source_file in member.source_files
                        }
                    )
                ),
                grounding_status=(
                    first.grounding_status
                    if all(
                        member.grounding_status == first.grounding_status
                        for member in members
                    )
                    else "uncertain"
                ),
                assembly_rule=assembly_rule,
                documentation_role=(
                    first.documentation_role
                    if all(
                        member.documentation_role == first.documentation_role
                        for member in members
                    )
                    else "mixed"
                ),
            )
        )
    return tuple(result)


def claims_by_id(
    claim_groups: Iterable[Sequence[EvidenceClaim]],
) -> dict[str, EvidenceClaim]:
    """Index claim groups and reject identifiers that would break lineage."""

    result: dict[str, EvidenceClaim] = {}
    for group in claim_groups:
        for claim in group:
            if claim.id in result:
                raise ValueError(f"Duplicate evidence claim identifier: {claim.id}")
            result[claim.id] = claim
    return result
