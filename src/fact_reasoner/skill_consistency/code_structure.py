"""Structural ownership for all grounded code, with shared dependency context.

Containment supplies an inspection hierarchy, not a conjunctive semantic premise.
Each source atom has one direct owner; dependencies reference the same atom IDs.
File/class/function/block/call nodes never imply that a behavior executes.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .call_graph import CallGraph, build_call_graph
from .source_identity import source_module
from .subtree_coverage import _is_boundary_call

VERSION = "code-structure-0.1.0"


@dataclass(frozen=True)
class CodeNode:
    id: str
    module: str
    kind: str
    source_lines: tuple[int, ...]
    parent_id: str | None
    children: tuple[str, ...]
    own_atom_ids: tuple[str, ...]
    subtree_atom_ids: tuple[str, ...]
    context_atom_ids: tuple[str, ...]
    references: tuple[dict, ...]
    unresolved_references: tuple[dict, ...]


@dataclass(frozen=True)
class CodeStructure:
    nodes: tuple[CodeNode, ...]
    unassigned_atom_ids: tuple[str, ...]
    dependency_depth: int

    def to_dict(self) -> dict:
        return {
            "version": VERSION,
            "policy": "unique direct ownership; shared context is not independent evidence",
            **asdict(self),
        }


def build_code_structure(
    files: Sequence[tuple[Path, str]],
    atoms: Sequence[Mapping[str, object]],
    *,
    graph: CallGraph | None = None,
    max_depth: int = 3,
) -> CodeStructure:
    if max_depth < 1:
        raise ValueError("max_depth must be positive")
    graph = graph or build_call_graph(files)
    modules = [module for _, module in files]
    if len(modules) != len(set(modules)):
        raise ValueError("Declared modules must be unique")
    atom_ids = [str(atom["id"]) for atom in atoms]
    if len(atom_ids) != len(set(atom_ids)):
        raise ValueError("Atom IDs must uniquely identify source observations")
    nodes: list[dict] = []

    def add(module, kind, start, end, column, parent):
        identifier = f"C:{module}:{start}:{column}:{kind}"
        if any(n["id"] == identifier for n in nodes):
            raise ValueError(f"Ambiguous structural node: {identifier}")
        node = dict(
            id=identifier,
            module=module,
            kind=kind,
            source_lines=tuple(range(start, end + 1)),
            parent_id=parent,
            children=[],
            own_atom_ids=[],
            subtree_atom_ids=[],
        )
        nodes.append(node)
        if parent is not None:
            next(n for n in nodes if n["id"] == parent)["children"].append(identifier)
        return identifier

    def walk(node, module, parent):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            start = min([node.lineno, *[d.lineno for d in node.decorator_list]])
            current = add(module, kind, start, node.end_lineno, node.col_offset, parent)
        elif isinstance(node, ast.Call) and _is_boundary_call(node):
            current = add(
                module,
                "boundary_call",
                node.lineno,
                node.end_lineno,
                node.col_offset,
                parent,
            )
        else:
            current = parent
        # Statement suites are separate branches; the condition remains with
        # the enclosing structural node. Opposite branches are not flattened.
        branch = isinstance(
            node,
            (
                ast.If,
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.Try,
                ast.TryStar,
                ast.With,
                ast.AsyncWith,
                ast.ExceptHandler,
                ast.match_case,
            ),
        )
        for field, value in ast.iter_fields(node):
            if (
                branch
                and field in {"body", "orelse", "finalbody"}
                and isinstance(value, list)
                and value
            ):
                first, last = value[0], value[-1]
                suite = add(
                    module,
                    f"{type(node).__name__.lower()}_{field}",
                    first.lineno,
                    last.end_lineno,
                    first.col_offset,
                    current,
                )
                for child in value:
                    walk(child, module, suite)
            elif isinstance(value, ast.AST):
                walk(value, module, current)
            elif isinstance(value, list):
                for child in value:
                    if isinstance(child, ast.AST):
                        walk(child, module, current)

    for path, module in files:
        text = Path(path).read_text(encoding="utf-8")
        root = add(module, "file", 1, max(1, len(text.splitlines())), 0, None)
        tree = ast.parse(text, filename=str(path))
        walk(tree, module, root)
    lookup = {node["id"]: node for node in nodes}
    depth = {}
    for node in nodes:
        depth[node["id"]] = (
            0 if node["parent_id"] is None else depth[node["parent_id"]] + 1
        )
    unassigned = []
    atom_modules = {}
    for atom in atoms:
        identifier = str(atom["id"])
        lines = set(atom.get("source_lines") or ())
        module = source_module(str(atom.get("source_file") or ""), modules)
        atom_modules[identifier] = module
        candidates = [
            n
            for n in nodes
            if n["module"] == module and lines and lines <= set(n["source_lines"])
        ]
        if not candidates:
            unassigned.append(identifier)
            continue
        # Without columns, multiple calls on one line cannot be distinguished.
        deepest = max(depth[n["id"]] for n in candidates)
        best = [n for n in candidates if depth[n["id"]] == deepest]
        if len(best) != 1:
            # Own the atom at the nearest shared ancestor, not an arbitrary call.
            ancestors = []
            for n in best:
                chain = set()
                while n is not None:
                    chain.add(n["id"])
                    n = lookup.get(n["parent_id"])
                ancestors.append(chain)
            common = set.intersection(*ancestors)
            owner = lookup[max(common, key=depth.get)]
        else:
            owner = best[0]
        owner["own_atom_ids"].append(identifier)
    for node in reversed(nodes):
        node["subtree_atom_ids"].extend(node["own_atom_ids"])
        if node["parent_id"] is not None:
            lookup[node["parent_id"]]["subtree_atom_ids"].extend(
                node["subtree_atom_ids"]
            )
    result = []
    for node in nodes:
        reached = tuple(
            dict.fromkeys(
                ref
                for line in node["source_lines"]
                for ref in graph.reachable(node["module"], line, max_depth=max_depth)
            )
        )
        targets = {(ref.to_module, ref.to_line) for ref in reached}
        spans = [
            (d.module, set(d.source_lines))
            for d in graph.definitions
            if (d.module, d.source_lines[0]) in targets
        ]
        context = {
            str(a["id"])
            for a in atoms
            for module, lines in spans
            if atom_modules[str(a["id"])] == module
            and lines.intersection(a.get("source_lines") or ())
        }
        context.difference_update(node["subtree_atom_ids"])
        result.append(
            CodeNode(
                node["id"],
                node["module"],
                node["kind"],
                node["source_lines"],
                node["parent_id"],
                tuple(node["children"]),
                tuple(node["own_atom_ids"]),
                tuple(sorted(set(node["subtree_atom_ids"]))),
                tuple(sorted(context)),
                tuple(r.to_dict() for r in reached),
                tuple(
                    r.to_dict()
                    for r in graph.unresolved
                    if r.from_module == node["module"]
                    and r.from_line in node["source_lines"]
                ),
            )
        )
    return CodeStructure(tuple(result), tuple(unassigned), max_depth)
