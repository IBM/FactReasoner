"""Exact claim ownership over the supported specification heading hierarchy.

Structure is not a claim of entailment. Every claim has one direct owner; subtree
membership is an explicit union, not an additional coverage denominator.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

from .subtree_coverage import Section, specification_sections

VERSION = "documentation-tree-0.1.0"


def section_parents(sections: Sequence[Section]) -> tuple[int | None, ...]:
    parents = []
    stack: list[int] = []
    for index, section in enumerate(sections):
        while stack and sections[stack[-1]].level >= section.level:
            stack.pop()
        parents.append(stack[-1] if stack else None)
        stack.append(index)
    return tuple(parents)


@dataclass(frozen=True)
class DocumentationNode:
    id: str
    section: Section
    parent_id: str | None
    children: tuple[str, ...]
    own_claim_ids: tuple[str, ...]
    subtree_claim_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DocumentationTree:
    nodes: tuple[DocumentationNode, ...]
    unowned_claim_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"version": VERSION, **asdict(self)}


def build_documentation_tree(
    text: str, claims: Sequence[Mapping[str, object]]
) -> DocumentationTree:
    sections = specification_sections(text)
    parents = section_parents(sections)
    ids = [f"D_{section.heading_line:04d}" for section in sections]
    claim_ids = [str(item["id"]) for item in claims]
    if len(claim_ids) != len(set(claim_ids)):
        raise ValueError(
            "Claim IDs must be unique; preserve distinct source occurrences"
        )
    owned: list[list[str]] = [[] for _ in sections]
    loose = []
    for claim, claim_id in zip(claims, claim_ids):
        lines = tuple(claim.get("source_lines") or ())
        candidates = [
            i
            for i, s in enumerate(sections)
            if lines and all(s.contains(int(line)) for line in lines)
        ]
        if not candidates:
            loose.append(claim_id)
        else:
            owned[candidates[-1]].append(claim_id)
    children = [
        [j for j, parent in enumerate(parents) if parent == i]
        for i in range(len(sections))
    ]
    members = [list(item) for item in owned]
    for index in reversed(range(len(sections))):
        if parents[index] is not None:
            members[parents[index]].extend(members[index])
    nodes = tuple(
        DocumentationNode(
            ids[i],
            section,
            ids[parents[i]] if parents[i] is not None else None,
            tuple(ids[j] for j in children[i]),
            tuple(owned[i]),
            tuple(sorted(set(members[i]))),
        )
        for i, section in enumerate(sections)
    )
    return DocumentationTree(nodes, tuple(loose))
