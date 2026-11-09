"""Ground deterministic code claims in syntax observations and behavior units."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
from pathlib import Path
import re
from typing import Mapping, Sequence

from .ast_facts import extract_python_facts
from .models import EvidenceClaim, SemanticUnit, SourceObservation


OBSERVED_NODES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Assign,
    ast.AnnAssign,
    ast.AugAssign,
    ast.Call,
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.Raise,
    ast.Return,
)


def _id_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_") or "source"


@dataclass(frozen=True)
class CodeLineage:
    """Static observations, grounded atoms, and their implementation units."""

    observations: tuple[SourceObservation, ...]
    atoms: tuple[EvidenceClaim, ...]
    behavior_units: tuple[SemanticUnit, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "observations": [item.to_dict() for item in self.observations],
            "atoms": [item.to_dict() for item in self.atoms],
            "behavior_units": [item.to_dict() for item in self.behavior_units],
        }


def _source_lines(node: ast.AST) -> tuple[int, ...]:
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", start)
    return tuple(range(start, end + 1))


def _excerpt(node: ast.AST, source: str) -> str:
    value = ast.get_source_segment(source, node)
    return value.strip() if value else node.__class__.__name__


def _structural_path(node: ast.AST, containers: Sequence[ast.AST]) -> tuple[str, ...]:
    members: list[str] = []
    node_start = getattr(node, "lineno", 1)
    node_end = getattr(node, "end_lineno", node_start)
    for item in containers:
        start = getattr(item, "lineno", 1)
        end = getattr(item, "end_lineno", start)
        if start <= node_start and node_end <= end:
            if isinstance(item, ast.ClassDef):
                members.append(f"class:{item.name}")
            elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                members.append(f"function:{item.name}")
    return tuple(members or ("module",))


def extract_python_observations(
    path: Path, *, source_file: str | None = None
) -> tuple[SourceObservation, ...]:
    """Record reviewable AST observations without importing target code."""

    path = Path(path)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    relative = source_file or path.name
    prefix = _id_slug(relative)
    containers = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        key=lambda item: (
            getattr(item, "lineno", 0),
            -getattr(item, "end_lineno", 0),
        ),
    )
    nodes = sorted(
        (node for node in ast.walk(tree) if isinstance(node, OBSERVED_NODES)),
        key=lambda item: (
            getattr(item, "lineno", 0),
            getattr(item, "col_offset", 0),
            item.__class__.__name__,
        ),
    )
    observations: list[SourceObservation] = []
    for index, node in enumerate(nodes, start=1):
        lines = _source_lines(node)
        observations.append(
            SourceObservation(
                id=f"O_{prefix}_{index:04d}",
                component="code",
                source_file=relative,
                source_lines=lines,
                excerpt=_excerpt(node, source),
                syntax_kind=node.__class__.__name__,
                structural_path=_structural_path(node, containers),
                derivation="Python ast.parse; target was not imported or executed",
            )
        )
    return tuple(observations)


def _best_observation(
    claim: EvidenceClaim, observations: Sequence[SourceObservation]
) -> SourceObservation | None:
    claim_lines = set(claim.source_lines)
    candidates = [
        item
        for item in observations
        if item.source_file == claim.source_file
        and claim_lines
        and claim_lines <= set(item.source_lines)
    ]
    if not candidates:
        candidates = [
            item
            for item in observations
            if item.source_file == claim.source_file
            and claim_lines.intersection(item.source_lines)
        ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            len(item.source_lines),
            0 if item.syntax_kind == "Call" else 1,
            item.id,
        ),
    )


def _structural_claims(
    observations: Sequence[SourceObservation],
) -> list[EvidenceClaim]:
    """Describe control, failure, and return syntax without inferring intent."""

    claims: list[EvidenceClaim] = []
    for observation in observations:
        excerpt = " ".join(observation.excerpt.split())
        assertion: str | None = None
        kind: str | None = None
        families: tuple[str, ...] = ()
        role = "mechanism"
        polarity = "positive"
        if observation.syntax_kind == "Raise":
            raised = excerpt.removeprefix("raise ").strip()
            assertion = f"Bundled code raises {raised}."
            kind = "raised_exception"
            families = ("error_handling",)
        elif observation.syntax_kind == "If":
            condition = excerpt.split(":", 1)[0].removeprefix("if ").strip()
            assertion = f"Bundled code conditionally executes when {condition}."
            kind = "control_flow_condition"
            families = ("control_flow_constraint",)
            role = "constraint"
        elif observation.syntax_kind == "Return":
            returned = excerpt.removeprefix("return").strip() or "without a value"
            assertion = f"Bundled code returns {returned}."
            kind = "return_value"
            families = ("output_generation",)
        if assertion is None or kind is None:
            continue
        claims.append(
            EvidenceClaim(
                id="pending",
                kind=kind,
                assertion=assertion,
                source_file=observation.source_file,
                source_lines=observation.source_lines,
                excerpt=observation.excerpt,
                tags=(),
                derivation="deterministic Python AST observation and fixed template",
                behavior_families=families,
                granularity="mechanism",
                component="code",
                semantic_role=role,
                actor="local_process",
                content_form="python_source",
                grounding_status="grounded",
                observation_ids=(observation.id,),
                parent_unit_id=f"BEH:{observation.source_file}:{observation.id}",
                polarity=polarity,
                scoring_eligible=False,
            )
        )
    return claims


def ground_python_atoms(
    path: Path,
    *,
    source_file: str | None = None,
    external_consumed_names: frozenset[str] = frozenset(),
) -> CodeLineage:
    """Attach every recognized code atom to an exact AST observation.

    ``external_consumed_names`` carries configuration names that sibling modules
    feed into behavior, so a definition-only module is not mistaken for dead
    configuration.
    """

    relative = source_file or Path(path).name
    prefix = _id_slug(relative)
    observations = extract_python_observations(path, source_file=relative)
    extracted = [
        *extract_python_facts(
            Path(path),
            source_file=relative,
            external_consumed_names=external_consumed_names,
        ),
        *_structural_claims(observations),
    ]
    extracted.sort(
        key=lambda item: (
            item.source_lines[0] if item.source_lines else 0,
            item.kind,
            item.assertion,
        )
    )
    atoms: list[EvidenceClaim] = []
    for index, claim in enumerate(extracted, start=1):
        observation = next(
            (item for item in observations if item.id in claim.observation_ids),
            None,
        ) or _best_observation(claim, observations)
        observation_ids = (observation.id,) if observation else ()
        parent = (
            f"B_{prefix}_{observation.id.rsplit('_', 1)[-1]}"
            if observation
            else f"B_{prefix}_unresolved_{index:04d}"
        )
        atoms.append(
            replace(
                claim,
                id=f"C_{prefix}_{index:04d}",
                observation_ids=observation_ids,
                parent_unit_id=parent,
                grounding_status="grounded" if observation else "uncertain",
            )
        )

    claims_by_unit: dict[str, list[EvidenceClaim]] = {}
    for claim in atoms:
        assert claim.parent_unit_id is not None
        claims_by_unit.setdefault(claim.parent_unit_id, []).append(claim)
    observation_by_id = {item.id: item for item in observations}

    units: list[SemanticUnit] = []
    for unit_id, members in claims_by_unit.items():
        behavior_claims = [item for item in members if item.granularity == "behavior"]
        selected = behavior_claims or members
        statement = " AND ".join(item.assertion for item in selected)
        observation_ids = tuple(
            dict.fromkeys(
                observation_id
                for item in members
                for observation_id in item.observation_ids
            )
        )
        grounding = (
            "grounded"
            if observation_ids
            and all(item.grounding_status == "grounded" for item in members)
            else "uncertain"
        )
        units.append(
            SemanticUnit(
                id=unit_id,
                unit_kind="behavior",
                statement=statement,
                claim_ids=tuple(item.id for item in members),
                component="code",
                capability_types=tuple(
                    sorted(
                        {
                            family
                            for item in members
                            for family in item.behavior_families
                        }
                    )
                ),
                observation_ids=observation_ids,
                source_files=tuple(sorted({item.source_file for item in members})),
                grounding_status=grounding,
                assembly_rule=(
                    "claims emitted from the same smallest enclosing AST observation"
                ),
            )
        )
    units.sort(
        key=lambda unit: (
            min(
                (
                    observation_by_id[item].source_lines[0]
                    for item in unit.observation_ids
                    if item in observation_by_id
                ),
                default=0,
            ),
            unit.id,
        )
    )
    return CodeLineage(observations, tuple(atoms), tuple(units))


def assemble_implementation_bundle(
    identifier: str,
    units: Sequence[SemanticUnit],
    claims: Mapping[str, EvidenceClaim],
    *,
    assembly_rule: str,
) -> SemanticUnit:
    """Create an explicit conjunctive premise from structurally related units.

    Only scoring-eligible behavioral claims enter the NLI premise. Their full
    lineage is retained even when identical assertion text is listed once. The
    caller must supply the structural reason (for example, one function or one
    verified data-flow path); arbitrary semantic similarity is not sufficient.
    """

    if not units:
        raise ValueError("An implementation bundle needs at least one unit")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", identifier):
        raise ValueError("Bundle identifiers must be FactReasoner-safe")
    if not assembly_rule.strip():
        raise ValueError("An implementation bundle needs an assembly rule")
    missing = {
        claim_id
        for unit in units
        for claim_id in unit.claim_ids
        if claim_id not in claims
    }
    if missing:
        raise ValueError(f"Unknown bundle claims: {sorted(missing)}")
    claim_ids = tuple(
        dict.fromkeys(
            claim_id
            for unit in units
            for claim_id in unit.claim_ids
            if claims[claim_id].scoring_eligible
            and claims[claim_id].granularity == "behavior"
        )
    )
    if not claim_ids:
        raise ValueError(
            "An implementation bundle needs at least one scoring-eligible claim"
        )
    grounding = (
        "grounded"
        if all(
            claims[claim_id].grounding_status == "grounded" for claim_id in claim_ids
        )
        else "uncertain"
    )
    statements = list(
        dict.fromkeys(claims[claim_id].assertion for claim_id in claim_ids)
    )
    return SemanticUnit(
        id=identifier,
        unit_kind="implementation_bundle",
        statement="Joint code evidence:\n"
        + "\n".join(f"- {statement}" for statement in statements),
        claim_ids=claim_ids,
        component="code",
        capability_types=tuple(
            sorted(
                {
                    family
                    for claim_id in claim_ids
                    for family in claims[claim_id].behavior_families
                }
            )
        ),
        observation_ids=tuple(
            dict.fromkeys(
                observation_id
                for claim_id in claim_ids
                for observation_id in claims[claim_id].observation_ids
            )
        ),
        source_files=tuple(
            sorted({claims[claim_id].source_file for claim_id in claim_ids})
        ),
        grounding_status=grounding,
        assembly_rule=assembly_rule,
    )


def assemble_structural_bundle(
    identifier: str,
    lineages: Sequence[CodeLineage],
    selectors: Sequence[Mapping[str, object]],
    *,
    assembly_rule: str,
) -> SemanticUnit:
    """Assemble a bundle from auditable file/AST-path selectors.

    Each selector needs ``source_file`` and ``structural_path`` values. Optional
    ``claim_kinds``, ``line_start``, ``line_end``, and ``assertion_contains``
    fields narrow the evidence. Only behavior-granularity claims are selected;
    mechanisms remain independently auditable atoms. Every selector must
    resolve, so an upstream source change fails visibly instead of silently
    weakening the evidence bundle.
    """

    if not selectors:
        raise ValueError("A structural bundle needs at least one selector")
    claims: dict[str, EvidenceClaim] = {}
    observations: list[SourceObservation] = []
    units: list[SemanticUnit] = []
    for lineage in lineages:
        for claim in lineage.atoms:
            if claim.id in claims:
                raise ValueError(f"Duplicate code claim identifier: {claim.id}")
            claims[claim.id] = claim
        observations.extend(lineage.observations)
        units.extend(lineage.behavior_units)

    chosen_claims_by_unit: dict[str, list[str]] = {}
    unit_order: list[str] = []
    for selector in selectors:
        source_file = str(selector.get("source_file", "")).strip()
        path_member = str(selector.get("structural_path", "")).strip()
        if not source_file or not path_member:
            raise ValueError(
                "Every structural selector needs source_file and structural_path"
            )
        line_start_raw = selector.get("line_start")
        line_end_raw = selector.get("line_end")
        line_start = int(line_start_raw) if line_start_raw is not None else None
        line_end = int(line_end_raw) if line_end_raw is not None else None
        if line_start is not None and line_start < 1:
            raise ValueError("line_start must be positive")
        if line_end is not None and (
            line_end < 1 or (line_start is not None and line_end < line_start)
        ):
            raise ValueError("line_end must be positive and not precede line_start")

        kind_value = selector.get("claim_kinds", ())
        claim_kinds = (
            {str(kind_value)}
            if isinstance(kind_value, str)
            else {str(item) for item in kind_value}  # type: ignore[union-attr]
        )
        contains_value = selector.get("assertion_contains", ())
        contains = (
            (str(contains_value).casefold(),)
            if isinstance(contains_value, str)
            else tuple(  # type: ignore[arg-type]
                str(item).casefold() for item in contains_value
            )
        )

        observation_ids = {
            observation.id
            for observation in observations
            if observation.source_file == source_file
            and path_member in observation.structural_path
            and (
                line_start is None
                or max(observation.source_lines, default=0) >= line_start
            )
            and (
                line_end is None
                or min(observation.source_lines, default=line_end + 1) <= line_end
            )
        }
        matched_claim_ids = {
            claim.id
            for claim in claims.values()
            if claim.scoring_eligible
            and claim.granularity == "behavior"
            and set(claim.observation_ids) & observation_ids
            and (not claim_kinds or claim.kind in claim_kinds)
            and (
                not contains
                or any(item in claim.assertion.casefold() for item in contains)
            )
        }
        if not matched_claim_ids:
            raise ValueError(
                "Structural selector resolved no scoring behavior: "
                f"{source_file} / {path_member}"
            )
        for unit in units:
            selected = [
                claim_id for claim_id in unit.claim_ids if claim_id in matched_claim_ids
            ]
            if not selected:
                continue
            if unit.id not in chosen_claims_by_unit:
                chosen_claims_by_unit[unit.id] = []
                unit_order.append(unit.id)
            chosen_claims_by_unit[unit.id].extend(selected)

    unit_by_id = {item.id: item for item in units}
    chosen = [
        replace(
            unit_by_id[unit_id],
            claim_ids=tuple(dict.fromkeys(chosen_claims_by_unit[unit_id])),
        )
        for unit_id in unit_order
    ]

    return assemble_implementation_bundle(
        identifier,
        chosen,
        claims,
        assembly_rule=assembly_rule,
    )


def lineage_integrity_errors(lineage: CodeLineage) -> tuple[str, ...]:
    """Return audit failures instead of silently dropping provenance."""

    observation_ids = {item.id for item in lineage.observations}
    atom_ids = {item.id for item in lineage.atoms}
    errors: list[str] = []
    for atom in lineage.atoms:
        if atom.grounding_status == "grounded" and not atom.observation_ids:
            errors.append(f"{atom.id}: grounded atom has no observation")
        unknown = set(atom.observation_ids) - observation_ids
        if unknown:
            errors.append(f"{atom.id}: unknown observations {sorted(unknown)}")
    covered_atoms = {
        claim_id for unit in lineage.behavior_units for claim_id in unit.claim_ids
    }
    missing = atom_ids - covered_atoms
    if missing:
        errors.append(f"atoms missing from behavior units: {sorted(missing)}")
    return tuple(errors)
