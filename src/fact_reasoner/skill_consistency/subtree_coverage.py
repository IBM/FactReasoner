"""Subtree-versus-group coverage: a prototype unit model.

The per-claim model pairs one documentation claim with one code atom. When the
true relationship is many-to-many - four schema fields realising one requirement,
or one configuration table constraining six claims - every individual pair is
correctly neutral and the aggregate reads as uncovered. The unit, not the
judgment, is wrong.

This module builds the two composite units:

``DocumentationGroup``
    Claims sharing a specification section, derived from markdown heading
    structure. Structure is the author's own partition and is read exactly; no
    claim is reworded, dropped, or invented.

``CodeSubtree``
    One behavior-boundary call plus the definitions its arguments resolve to.
    Composition matters: a requirement declaration promising five checks, bound
    to a validator testing one, is misleading read alone and accurate read
    together.

    The boundary call, not the enclosing function, is the unit. Descending from
    an entry point instead was tried and abandoned: in a pipeline package
    ``main`` reaches ``run_pipeline`` reaches everything, so the descent returns
    the whole package as one or two indistinguishable subtrees. Call structure
    there is a chain, not a tree, and a chain has no informative partition. A
    boundary call has one instruction text, one output contract, and its own
    requirements, so distinct calls stay distinct units.

Both sides are deterministic. Relating them is left to the caller, because that
is a model judgment and belongs behind the versioned prompt machinery.

Status: prototype. It computes units and a relation matrix; it assigns no score
and replaces nothing. The assumption that subtree-level judging resolves cases
that per-pair judging leaves neutral is untested, and until it is measured this
must not feed a reported number.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal, Mapping, Sequence

from .call_graph import CallGraph, Reference
from .source_identity import source_module

GROUPING_ID = "specification-section-grouping"
GROUPING_VERSION = "0.2.0-prototype"
SUBTREE_ID = "behavior-boundary-call-subtree"
SUBTREE_VERSION = "0.3.0-prototype"
DEFAULT_SUBTREE_DEPTH = 3

# The call spellings that mark a crossing from program plumbing into declared
# model or validation behavior. Kept identical to the extractor's own boundary
# predicate in ``ast_facts`` so the unit and the atoms agree on what a boundary
# is; a second, divergent definition here would silently partition atoms by a
# rule the extractor never applied.
BOUNDARY_ATTRIBUTE_SUFFIXES = (".instruct",)
BOUNDARY_TERMINAL_NAMES = ("instruct", "req", "Requirement")

RelationLabel = Literal["entailment", "contradiction", "neutral"]
CellState = Literal[
    "entailed",
    "contradicted",
    "mixed",
    "neutral",
    "unassessed",
]
GroupState = Literal[
    "covered",
    "contradicted",
    "mixed_evidence",
    "not_evidenced",
    "unassessed",
]

_HEADING_RE = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?)|[ \t]*)$")


@dataclass(frozen=True, slots=True)
class Section:
    """One specification section, from its heading to the next of equal rank."""

    title: str
    level: int
    heading_line: int
    first_line: int
    last_line: int

    def contains(self, line: int) -> bool:
        return self.first_line <= line <= self.last_line

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "level": self.level,
            "heading_line": self.heading_line,
            "first_line": self.first_line,
            "last_line": self.last_line,
        }


def specification_sections(text: str) -> tuple[Section, ...]:
    """Read ATX headings outside fenced code/frontmatter, retaining exact spans.

    Setext headings and headings nested in blockquotes/lists are not supported.
    This is the declared heading dialect of the prototype, not a full Markdown
    renderer. Four-space indented code is never recognized as an ATX heading.
    """

    lines = text.splitlines()
    headings: list[tuple[int, int, str]] = []
    fence: tuple[str, int] | None = None
    frontmatter = bool(lines and lines[0].strip() == "---")
    for number, line in enumerate(lines, start=1):
        if frontmatter:
            if number > 1 and line.strip() in {"---", "..."}:
                frontmatter = False
            continue
        if fence is not None:
            if re.fullmatch(
                r" {0,3}" + re.escape(fence[0]) + "{" + str(fence[1]) + r",}[ \t]*",
                line,
            ):
                fence = None
            continue
        opener = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if opener and not (opener[1][0] == "`" and "`" in opener[2]):
            fence = (opener[1][0], len(opener[1]))
            continue
        match = _HEADING_RE.match(line)
        if match is not None:
            title = re.sub(r"[ \t]+#+[ \t]*$", "", match[2] or "").strip()
            headings.append((number, len(match[1]), title))

    sections: list[Section] = []
    for index, (heading_line, level, title) in enumerate(headings):
        last = len(lines)
        for following_line, following_level, _title in headings[index + 1 :]:
            if following_level <= level:
                last = following_line - 1
                break
        sections.append(
            Section(
                title=title,
                level=level,
                heading_line=heading_line,
                first_line=heading_line,
                last_line=last,
            )
        )
    return tuple(sections)


@dataclass(frozen=True, slots=True)
class DocumentationGroup:
    """Claims sharing one specification section."""

    id: str
    title: str
    level: int
    source_file: str
    first_line: int
    last_line: int
    claim_ids: tuple[str, ...]
    statements: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "level": self.level,
            "source_file": self.source_file,
            "source_lines": [self.first_line, self.last_line],
            "claim_ids": list(self.claim_ids),
            "claim_count": len(self.claim_ids),
            "statements": list(self.statements),
            "grouping_id": GROUPING_ID,
            "grouping_version": GROUPING_VERSION,
        }


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_")
    return (cleaned or "section")[:48]


def documentation_groups(
    specification_text: str,
    claims: Sequence[Mapping[str, object]],
    *,
    source_file: str,
    level: int = 2,
) -> tuple[tuple[DocumentationGroup, ...], tuple[str, ...]]:
    """Partition claims by enclosing section at ``level``.

    Returns the groups and the ids of claims lying outside every section, so a
    claim is never silently absent from the partition.
    """

    sections = [
        item
        for item in specification_sections(specification_text)
        if item.level <= level
    ]
    assigned: dict[int, list[Mapping[str, object]]] = {
        index: [] for index in range(len(sections))
    }
    ungrouped: list[str] = []
    for claim in claims:
        lines = list(claim.get("source_lines") or ())
        if not lines:
            ungrouped.append(str(claim.get("id")))
            continue
        line = int(lines[0])
        owner: int | None = None
        for index, section in enumerate(sections):
            if section.contains(line):
                owner = index
        if owner is None:
            ungrouped.append(str(claim.get("id")))
            continue
        assigned[owner].append(claim)

    groups: list[DocumentationGroup] = []
    for index, section in enumerate(sections):
        members = assigned[index]
        if not members:
            continue
        groups.append(
            DocumentationGroup(
                id=f"G_{section.heading_line:04d}_{_slug(section.title)}",
                title=section.title,
                level=section.level,
                source_file=source_file,
                first_line=section.first_line,
                last_line=section.last_line,
                claim_ids=tuple(str(item.get("id")) for item in members),
                statements=tuple(
                    str(item.get("assertion") or item.get("text") or "")
                    for item in members
                ),
            )
        )
    return tuple(groups), tuple(ungrouped)


@dataclass(frozen=True, slots=True)
class BoundaryCall:
    """One call crossing from plumbing into declared model or validation behavior."""

    module: str
    line: int
    end_line: int
    spelling: str
    container: tuple[str, ...]

    @property
    def source_lines(self) -> tuple[int, ...]:
        return tuple(range(self.line, self.end_line + 1))

    def to_dict(self) -> dict[str, object]:
        return {
            "module": self.module,
            "line": self.line,
            "end_line": self.end_line,
            "spelling": self.spelling,
            "container": list(self.container),
        }


def _call_spelling(node: ast.Call) -> str:
    try:
        return ast.unparse(node.func)
    except Exception:  # pragma: no cover - unparse is total for parsed trees
        return ""


def _is_boundary_call(node: ast.Call) -> bool:
    """Match the extractor's boundary predicate, by spelling alone.

    Deliberately syntactic. Deciding that a call enters a model requires knowing
    what the callee does, and this project never executes target code, so the
    spelling is the only evidence available. It over-matches a local helper
    coincidentally named ``instruct`` and under-matches a framework that spells
    the boundary differently; both are visible in the emitted spelling.
    """

    spelling = _call_spelling(node)
    if not spelling:
        return False
    if any(spelling.endswith(suffix) for suffix in BOUNDARY_ATTRIBUTE_SUFFIXES):
        return True
    return spelling.rsplit(".", 1)[-1] in BOUNDARY_TERMINAL_NAMES


def _enclosing(node: ast.AST, tree: ast.Module) -> tuple[str, ...]:
    path: list[str] = []
    for item in ast.walk(tree):
        if not isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = item.lineno
        finish = getattr(item, "end_lineno", start)
        if start <= getattr(node, "lineno", 0) <= finish:
            prefix = "class" if isinstance(item, ast.ClassDef) else "function"
            path.append(f"{prefix}:{item.name}")
    return tuple(path)


def boundary_calls(
    files: Iterable[tuple[object, str]],
) -> tuple[BoundaryCall, ...]:
    """Locate every behavior-boundary call across declared production files.

    ``files`` pairs a readable path with the module name used in atom
    provenance, matching ``build_call_graph``.
    """

    found: list[BoundaryCall] = []
    for path, module in files:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _is_boundary_call(node):
                continue
            found.append(
                BoundaryCall(
                    module=module,
                    line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    spelling=_call_spelling(node),
                    container=_enclosing(node, tree),
                )
            )
    return tuple(sorted(found, key=lambda item: (item.module, item.line)))


@dataclass(frozen=True, slots=True)
class CodeSubtree:
    """A boundary call plus the definitions its arguments resolve to."""

    id: str
    entry_module: str
    entry_line: int
    entry_kind: str
    depth: int
    atom_ids: tuple[str, ...]
    statements: tuple[str, ...]
    spelling: str = ""
    container: tuple[str, ...] = ()
    edges: tuple[Mapping[str, object], ...] = field(default=())

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "entry_module": self.entry_module,
            "entry_line": self.entry_line,
            "entry_kind": self.entry_kind,
            "spelling": self.spelling,
            "container": list(self.container),
            "depth": self.depth,
            "atom_ids": list(self.atom_ids),
            "atom_count": len(self.atom_ids),
            "statements": list(self.statements),
            "edges": [dict(item) for item in self.edges],
            "subtree_id": SUBTREE_ID,
            "subtree_version": SUBTREE_VERSION,
        }


def _atoms_at(
    atoms: Sequence[Mapping[str, object]],
    module: str,
    lines: Iterable[int],
    modules: Sequence[str] = (),
) -> list[Mapping[str, object]]:
    wanted = set(lines)
    found = []
    for atom in atoms:
        source = str(atom.get("source_file") or "")
        if source_module(source, modules or (module,)) != module:
            continue
        if wanted & set(atom.get("source_lines") or ()):
            found.append(atom)
    return found


def _definition_at(graph: CallGraph, module: str, line: int) -> object | None:
    return next(
        (
            item
            for item in graph.definitions
            if item.module == module and item.source_lines[0] == line
        ),
        None,
    )


def code_subtrees(
    graph: CallGraph,
    atoms: Sequence[Mapping[str, object]],
    calls: Sequence[BoundaryCall],
    *,
    max_depth: int = DEFAULT_SUBTREE_DEPTH,
) -> tuple[tuple[CodeSubtree, ...], tuple[str, ...]]:
    """Assemble one subtree per behavior-boundary call.

    A subtree holds the atoms on the call itself - its instruction text - plus
    the atoms of every definition reachable from the call's own lines within
    ``max_depth`` hops: the output schema through ``format=``, the declared
    requirements through ``requirements=``, and each requirement's validator one
    hop further.

    Returns the subtrees and the ids of atoms belonging to no subtree, so an
    atom is never silently absent. Unattached atoms are a real and expected
    outcome - configuration a boundary call does not consume, or a schema no
    call declares as its format - and they stay visible rather than being
    assigned to an arbitrary unit.
    """

    subtrees: list[CodeSubtree] = []
    attached: set[str] = set()
    for call in calls:
        members = _atoms_at(
            atoms, call.module, call.source_lines, graph.declared_modules
        )
        reached: list[Reference] = []
        for line in call.source_lines:
            reached.extend(graph.reachable(call.module, line, max_depth=max_depth))
        reached = list(dict.fromkeys(reached))
        for reference in reached:
            target = _definition_at(graph, reference.to_module, reference.to_line)
            if target is None:
                continue
            members.extend(
                _atoms_at(
                    atoms, target.module, target.source_lines, graph.declared_modules
                )
            )

        seen: set[str] = set()
        ordered: list[Mapping[str, object]] = []
        for atom in members:
            key = str(atom.get("id"))
            if key in seen:
                continue
            seen.add(key)
            ordered.append(atom)
        if not ordered:
            continue
        attached.update(str(item.get("id")) for item in ordered)
        subtrees.append(
            CodeSubtree(
                id=f"S_{_slug(call.module)}_{call.line:04d}",
                entry_module=call.module,
                entry_line=call.line,
                entry_kind="boundary_call",
                spelling=call.spelling,
                container=call.container,
                depth=max_depth,
                atom_ids=tuple(str(item.get("id")) for item in ordered),
                statements=tuple(
                    str(item.get("text") or item.get("assertion") or "")
                    for item in ordered
                ),
                edges=tuple(item.to_dict() for item in reached),
            )
        )

    unattached = tuple(
        str(item.get("id")) for item in atoms if str(item.get("id")) not in attached
    )
    return tuple(subtrees), unattached


def containment(
    subtrees: Sequence[CodeSubtree],
) -> tuple[tuple[str, str], ...]:
    """Pairs ``(inner, outer)`` where one subtree's atoms are a subset of another's.

    Nesting is expected and is not an error. A requirement declared at module
    level is its own boundary call, and it is also part of every ``instruct``
    call that passes it, so both units exist and both are worth judging: a
    documented checklist may be met by the requirement text alone. But the two
    are not independent evidence, so any aggregate that sums over subtrees must
    consult this relation and count each atom once.
    """

    members = {subtree.id: frozenset(subtree.atom_ids) for subtree in subtrees}
    pairs = [
        (inner, outer)
        for inner, inner_atoms in members.items()
        for outer, outer_atoms in members.items()
        if inner != outer and inner_atoms < outer_atoms
    ]
    return tuple(sorted(pairs))


def maximal_subtrees(
    subtrees: Sequence[CodeSubtree],
) -> tuple[CodeSubtree, ...]:
    """Subtrees not strictly contained in another; they may still overlap.

    Use ``subtree_membership`` for unique coverage and shared evidence. Maximal
    sets are not a partition and must never be summed as independent witnesses.
    """

    contained = {inner for inner, _outer in containment(subtrees)}
    return tuple(item for item in subtrees if item.id not in contained)


def subtree_membership(subtrees: Sequence[CodeSubtree]) -> dict[str, object]:
    """Record all owners, including equal and overlapping non-nested groups."""
    ids = [item.id for item in subtrees]
    if len(ids) != len(set(ids)):
        raise ValueError("Subtree IDs must be unique")
    owners: dict[str, list[str]] = {}
    for subtree in subtrees:
        for atom_id in dict.fromkeys(subtree.atom_ids):
            owners.setdefault(atom_id, []).append(subtree.id)
    shared = {key: value for key, value in owners.items() if len(value) > 1}
    return {
        "atom_owners": owners,
        "shared_atoms": shared,
        "unique_atom_ids": sorted(owners),
        "unique_atom_count": len(owners),
        "membership_count": sum(len(value) for value in owners.values()),
        "independent_evidence": False if shared else None,
        "policy": "membership is structural; overlap never supplies independent evidence",
    }


@dataclass(frozen=True, slots=True)
class MatrixCell:
    """One subtree judged against one documentation group."""

    subtree_id: str
    group_id: str
    state: CellState
    labels: tuple[RelationLabel, ...] = ()
    note: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "subtree_id": self.subtree_id,
            "group_id": self.group_id,
            "state": self.state,
            "labels": list(self.labels),
            "note": self.note,
        }


def cell_state(labels: Sequence[RelationLabel]) -> CellState:
    """Reduce judgments for one cell without inventing a decisive answer."""

    if not labels:
        return "unassessed"
    distinct = set(labels)
    if "entailment" in distinct and "contradiction" in distinct:
        return "mixed"
    if "entailment" in distinct:
        return "entailed"
    if "contradiction" in distinct:
        return "contradicted"
    return "neutral"


def group_state(cells: Sequence[MatrixCell]) -> GroupState:
    """Derive the reverse view from the same matrix as the forward view.

    A group without entailment is not one state. It may have been judged and
    unmet, judged and contradicted, or never judged at all, and those must stay
    distinguishable: silence is missing coverage, contradiction requires explicit
    opposition, and an unjudged group is unassessed rather than uncovered.
    """

    if not cells or all(item.state == "unassessed" for item in cells):
        return "unassessed"
    states = {item.state for item in cells}
    if "mixed" in states:
        return "mixed_evidence"
    if "entailed" in states and "contradicted" in states:
        return "mixed_evidence"
    if "contradicted" in states:
        return "contradicted"
    if "entailed" in states:
        return "covered"
    return "not_evidenced"


def coverage_matrix(
    subtrees: Sequence[CodeSubtree],
    groups: Sequence[DocumentationGroup],
    judgments: Mapping[tuple[str, str], Sequence[RelationLabel]],
) -> dict[str, object]:
    """Assemble both directions from one subtree-by-group judgment set.

    ``judgments`` maps ``(subtree_id, group_id)`` to the labels observed for that
    cell. A missing key is unassessed, never neutral.
    """

    cells = [
        MatrixCell(
            subtree_id=subtree.id,
            group_id=group.id,
            state=cell_state(judgments.get((subtree.id, group.id), ())),
            labels=tuple(judgments.get((subtree.id, group.id), ())),
        )
        for subtree in subtrees
        for group in groups
    ]
    by_group: dict[str, list[MatrixCell]] = {}
    by_subtree: dict[str, list[MatrixCell]] = {}
    for cell in cells:
        by_group.setdefault(cell.group_id, []).append(cell)
        by_subtree.setdefault(cell.subtree_id, []).append(cell)

    group_states = {
        group.id: group_state(by_group.get(group.id, [])) for group in groups
    }
    subtree_states = {
        subtree.id: cell_state(
            tuple(
                label
                for cell in by_subtree.get(subtree.id, [])
                for label in cell.labels
            )
        )
        for subtree in subtrees
    }
    conflicting = [cell.to_dict() for cell in cells if cell.state == "mixed"]
    multiply_entailed = [
        group_id
        for group_id, entries in by_group.items()
        if sum(1 for item in entries if item.state == "entailed") > 1
    ]
    return {
        "status": "prototype",
        "interpretation": (
            "unit-model prototype; assigns no score and replaces no reported "
            "measure. A mixed cell needs review rather than being a verdict: it "
            "may mean the code violates one requirement while satisfying "
            "another, that two groups overlap, or that one judgment is wrong"
        ),
        "grouping_id": GROUPING_ID,
        "grouping_version": GROUPING_VERSION,
        "subtree_id": SUBTREE_ID,
        "subtree_version": SUBTREE_VERSION,
        "membership": subtree_membership(subtrees),
        "cells": [item.to_dict() for item in cells],
        "group_states": group_states,
        "subtree_states": subtree_states,
        "conflicting_cells": conflicting,
        "groups_entailed_by_multiple_subtrees": sorted(multiply_entailed),
        "unassessed_cells": sum(1 for item in cells if item.state == "unassessed"),
        "cell_count": len(cells),
    }
