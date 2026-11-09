"""Shared serializable evidence and lineage types.

The source component (metadata, instruction, code, or activated resource) is
deliberately independent from the role a claim has in a particular graph.  A
single instruction claim can therefore be a context in a documentation-to-code
analysis and an atom in a metadata-to-instructions analysis without losing its
identity or provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


ArtifactComponent = Literal["metadata", "instruction", "code", "resource"]
SemanticRole = Literal[
    "identity",
    "declaration",
    "requirement",
    "operation",
    "constraint",
    "example",
    "mechanism",
]
GroundingStatus = Literal["grounded", "uncertain", "refuted", "not_applicable"]
DocumentationRole = Literal[
    "metadata_declaration",
    "normative_requirement",
    "constraint",
    "executable_example",
    "illustrative_example",
    "reference_definition",
    "activated_resource",
    "not_applicable",
    "mixed",
]


@dataclass(frozen=True)
class EvidenceClaim:
    """A claim whose wording and provenance were derived deterministically."""

    id: str
    kind: str
    assertion: str
    source_file: str
    source_lines: tuple[int, ...]
    excerpt: str
    tags: tuple[str, ...]
    derivation: str
    behavior_families: tuple[str, ...] = ()
    granularity: str = "behavior"
    occurrences: tuple[dict[str, Any], ...] = ()
    # Neutral defaults protect legacy/imported records from being silently
    # misclassified as executable, grounded code.  Current extractors set all
    # five fields explicitly.
    component: ArtifactComponent | str = "unknown"
    semantic_role: SemanticRole | str = "declaration"
    actor: str = "unknown"
    content_form: str = "unknown"
    grounding_status: GroundingStatus | str = "not_applicable"
    observation_ids: tuple[str, ...] = ()
    parent_unit_id: str | None = None
    entities: tuple[str, ...] = ()
    condition: str | None = None
    polarity: str = "positive"
    activated_by: tuple[str, ...] = ()
    scoring_eligible: bool = True
    documentation_role: DocumentationRole | str = "not_applicable"
    atomization_rule: str = "one source proposition"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_lines"] = list(self.source_lines)
        value["tags"] = list(self.tags)
        value["behavior_families"] = list(self.behavior_families)
        value["observation_ids"] = list(self.observation_ids)
        value["entities"] = list(self.entities)
        value["activated_by"] = list(self.activated_by)
        value["occurrences"] = list(self.occurrences) or [
            {
                "source_file": self.source_file,
                "source_lines": list(self.source_lines),
                "excerpt": self.excerpt,
            }
        ]
        return value


@dataclass(frozen=True)
class SourceObservation:
    """A syntax-level observation made without executing the target artifact."""

    id: str
    component: ArtifactComponent | str
    source_file: str
    source_lines: tuple[int, ...]
    excerpt: str
    syntax_kind: str
    structural_path: tuple[str, ...] = ()
    derivation: str = "deterministic source parse"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["source_lines"] = list(self.source_lines)
        value["structural_path"] = list(self.structural_path)
        return value


@dataclass(frozen=True)
class SemanticUnit:
    """A requirement or implementation behavior assembled from one or more claims."""

    id: str
    unit_kind: Literal["requirement", "behavior", "implementation_bundle"] | str
    statement: str
    claim_ids: tuple[str, ...]
    component: ArtifactComponent | str
    capability_types: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()
    source_files: tuple[str, ...] = ()
    grounding_status: GroundingStatus | str = "grounded"
    assembly_rule: str = "one claim"
    documentation_role: DocumentationRole | str = "not_applicable"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in (
            "claim_ids",
            "capability_types",
            "observation_ids",
            "source_files",
        ):
            value[key] = list(value[key])
        return value
