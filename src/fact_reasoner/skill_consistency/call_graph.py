"""Resolve intra-package references between declared production files.

``ast.parse`` is per-file: it records that a call is spelled ``run_pipeline``,
not which definition that name refers to. Python resolves names at runtime, and
this project never imports or executes target code, so resolution is done
statically here by reading import statements and matching names against
definitions in sibling modules.

Three reference kinds are recorded, because the informative ones are not all
calls:

``call``
    A call whose target resolves to a package-local definition.
``argument``
    A package-local definition passed as an argument, such as
    ``format=ParsedProblem`` or ``validation_fn=_check_objective_identified``.
    These carry the output contract and the validator binding, which is where a
    declared requirement meets the code that actually enforces it.
``annotation``
    A package-local definition named in a type annotation, such as
    ``sentences: list[SentenceLabel]``. A nested model is part of the output
    contract that names it; reading only the outer class would describe a field
    as ``list[SentenceLabel]`` while leaving every field of ``SentenceLabel``
    attached to nothing.

Argument resolution also follows a local name bound once in the same function to
a literal container, because a package that assembles its grounding context into
a dict before passing it - ``grounding_context=grounding_rules`` - otherwise
hides every constant in that dict behind one unresolvable name.

Resolution is conservative static evidence, not a complete Python interpreter.
Lexical bindings and declared imports are followed; shadowed, rebound, dynamic
and unsupported references remain unresolved. Resolved edges describe syntax
dependencies, not proof of runtime execution or joint semantic support.
"""

from __future__ import annotations

import ast
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Literal

RESOLVER_ID = "intra-package-static-reference-resolver"
RESOLVER_VERSION = "1.3.0"

ReferenceKind = Literal["call", "argument", "annotation"]


@dataclass(frozen=True, slots=True)
class Definition:
    """One module-level definition that a sibling module can reference."""

    name: str
    module: str
    kind: str
    source_lines: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "module": self.module,
            "kind": self.kind,
            "source_lines": list(self.source_lines),
        }


@dataclass(frozen=True, slots=True)
class Reference:
    """One resolved reference from a source location to a definition."""

    kind: ReferenceKind
    name: str
    from_module: str
    from_line: int
    from_container: tuple[str, ...]
    to_module: str
    to_line: int
    to_kind: str
    keyword: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "name": self.name,
            "keyword": self.keyword,
            "from_module": self.from_module,
            "from_line": self.from_line,
            "from_container": list(self.from_container),
            "to_module": self.to_module,
            "to_line": self.to_line,
            "to_kind": self.to_kind,
            "resolver_id": RESOLVER_ID,
            "resolver_version": RESOLVER_VERSION,
        }


@dataclass(frozen=True, slots=True)
class Unresolved:
    """A reference this resolver cannot follow without executing the target."""

    expression: str
    reason: str
    from_module: str
    from_line: int

    def to_dict(self) -> dict[str, object]:
        return {
            "expression": self.expression,
            "reason": self.reason,
            "from_module": self.from_module,
            "from_line": self.from_line,
        }


@dataclass(frozen=True, slots=True)
class CallGraph:
    """Resolved references across one declared production package."""

    definitions: tuple[Definition, ...]
    references: tuple[Reference, ...]
    unresolved: tuple[Unresolved, ...]
    declared_modules: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "resolver_id": RESOLVER_ID,
            "resolver_version": RESOLVER_VERSION,
            "policy": (
                "static intra-package resolution only; dynamic dispatch and "
                "computed access are reported as unresolved, so resolved edges "
                "are a conservative syntactic lower bound, not runtime proof"
            ),
            "declared_modules": list(self.declared_modules),
            "definitions": [item.to_dict() for item in self.definitions],
            "references": [item.to_dict() for item in self.references],
            "unresolved": [item.to_dict() for item in self.unresolved],
            "definition_count": len(self.definitions),
            "reference_count": len(self.references),
            "unresolved_count": len(self.unresolved),
        }

    def references_from(self, module: str, line: int) -> tuple[Reference, ...]:
        return tuple(
            item
            for item in self.references
            if item.from_module == module and item.from_line == line
        )

    def reachable(
        self, module: str, line: int, *, max_depth: int = 3
    ) -> tuple[Reference, ...]:
        """Follow references transitively up to ``max_depth`` hops.

        A bound is required: unbounded closure would pull an entire package into
        every neighbourhood and make the notion of related evidence vacuous.
        """

        if max_depth < 1:
            raise ValueError("max_depth must be positive")
        seen: set[tuple[str, int]] = set()
        frontier = deque([(module, line, 0)])
        collected: list[Reference] = []
        while frontier:
            current_module, current_line, depth = frontier.popleft()
            if depth >= max_depth:
                continue
            for reference in self.references:
                if (
                    reference.from_module != current_module
                    or reference.from_line != current_line
                ):
                    continue
                collected.append(reference)
                target = (reference.to_module, reference.to_line)
                if target in seen:
                    continue
                seen.add(target)
                for body_line in _definition_body_lines(self, *target):
                    frontier.append((reference.to_module, body_line, depth + 1))
        return tuple(dict.fromkeys(collected))


def _definition_body_lines(graph: CallGraph, module: str, line: int) -> tuple[int, ...]:
    """Lines inside the definition at ``module:line`` that make references."""

    definition = next(
        (
            item
            for item in graph.definitions
            if item.module == module and item.source_lines[0] == line
        ),
        None,
    )
    if definition is None:
        return ()
    span = set(definition.source_lines)
    return tuple(
        sorted(
            {
                item.from_line
                for item in graph.references
                if item.from_module == module and item.from_line in span
            }
        )
    )


@dataclass
class _Binding:
    kind: str
    value: object = None


@dataclass
class _Scope:
    node: ast.AST
    parent: _Scope | None
    kind: str
    bindings: dict[str, list[_Binding]] = field(default_factory=dict)
    redirected: set[str] = field(default_factory=set)
    wildcard_import: bool = False

    def bind(self, name: str, binding: _Binding) -> None:
        self.bindings.setdefault(name, []).append(binding)


class _Bindings(ast.NodeVisitor):
    """Collect lexical ownership without executing a target module.

    Multiple assignments and global/nonlocal mutation are deliberately unresolved.
    A spelling elsewhere in the package never establishes a binding here.
    """

    def __init__(self, module: str, tree: ast.Module):
        self.module = module
        self.root = _Scope(tree, None, "module")
        self.scope = self.root
        self.scopes: dict[ast.AST, _Scope] = {}
        self.definitions: list[Definition] = []
        self.visit(tree)

    def visit(self, node: ast.AST):
        self.scopes[node] = self.scope
        return super().visit(node)

    def definition(self, node: ast.AST, name: str, kind: str) -> Definition:
        value = Definition(
            name, self.module, kind, tuple(range(node.lineno, node.end_lineno + 1))
        )
        self.definitions.append(value)
        return value

    def named_scope(self, node, kind: str):
        self.scope.bind(
            node.name, _Binding("definition", self.definition(node, node.name, kind))
        )
        for item in node.decorator_list:
            self.visit(item)
        if isinstance(node, ast.ClassDef):
            for item in [*node.bases, *node.keywords]:
                self.visit(item)
        else:
            for item in [*node.args.defaults, *[x for x in node.args.kw_defaults if x]]:
                self.visit(item)
            for arg in [
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
                *([node.args.vararg] if node.args.vararg else []),
                *([node.args.kwarg] if node.args.kwarg else []),
            ]:
                if arg.annotation:
                    self.visit(arg.annotation)
            if node.returns:
                self.visit(node.returns)
        previous = self.scope
        self.scope = _Scope(node, previous, kind)
        if not isinstance(node, ast.ClassDef):
            self.parameters(node.args)
        for item in node.body:
            self.visit(item)
        self.scope = previous

    def parameters(self, args):
        for arg in [
            *args.posonlyargs,
            *args.args,
            *args.kwonlyargs,
            *([args.vararg] if args.vararg else []),
            *([args.kwarg] if args.kwarg else []),
        ]:
            self.scope.bind(arg.arg, _Binding("dynamic"))

    def visit_FunctionDef(self, node):
        self.named_scope(node, "function")

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self.named_scope(node, "class")

    def visit_Lambda(self, node):
        for default in [*node.args.defaults, *[x for x in node.args.kw_defaults if x]]:
            self.visit(default)
        previous = self.scope
        self.scope = _Scope(node, previous, "function")
        self.parameters(node.args)
        self.visit(node.body)
        self.scope = previous

    def comprehension(self, node):
        # The first iterable is evaluated in the enclosing scope.
        self.visit(node.generators[0].iter)
        previous = self.scope
        self.scope = _Scope(node, previous, "function")
        for index, generator in enumerate(node.generators):
            if index:
                self.visit(generator.iter)
            self.visit(generator.target)
            for condition in generator.ifs:
                self.visit(condition)
        for name in ("elt", "key", "value"):
            if hasattr(node, name):
                self.visit(getattr(node, name))
        self.scope = previous

    visit_ListComp = comprehension
    visit_SetComp = comprehension
    visit_DictComp = comprehension
    visit_GeneratorExp = comprehension

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.scope.bind(node.id, _Binding("dynamic"))

    def assignment(self, node, targets, value):
        for target in targets:
            if isinstance(target, ast.Name):
                if self.scope.kind in {"module", "class"}:
                    binding = _Binding(
                        "definition", self.definition(node, target.id, "constant")
                    )
                elif isinstance(value, (ast.Dict, ast.List, ast.Tuple, ast.Set)):
                    binding = _Binding("literal", value)
                else:
                    binding = _Binding("dynamic")
                self.scope.bind(target.id, binding)
                self.scopes[target] = self.scope
            else:
                self.visit(target)
        if value is not None:
            self.visit(value)

    def visit_Assign(self, node):
        self.assignment(node, node.targets, node.value)

    def visit_AnnAssign(self, node):
        self.assignment(node, [node.target], node.value)
        self.visit(node.annotation)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.scope.bind(node.name, _Binding("dynamic"))
        self.generic_visit(node)

    def visit_MatchAs(self, node):
        if node.name:
            self.scope.bind(node.name, _Binding("dynamic"))
        self.generic_visit(node)

    def visit_MatchStar(self, node):
        if node.name:
            self.scope.bind(node.name, _Binding("dynamic"))

    def visit_MatchMapping(self, node):
        if node.rest:
            self.scope.bind(node.rest, _Binding("dynamic"))
        self.generic_visit(node)

    def visit_Global(self, node):
        self.scope.redirected.update(node.names)

    visit_Nonlocal = visit_Global

    def visit_ImportFrom(self, node):
        for alias in node.names:
            if alias.name == "*":
                self.scope.wildcard_import = True
            else:
                self.scope.bind(
                    alias.asname or alias.name,
                    _Binding("import", (node.level, node.module or "", alias.name)),
                )

    def visit_Import(self, node):
        for alias in node.names:
            # Without 'as', Python binds the first package segment.
            self.scope.bind(
                alias.asname or alias.name.split(".")[0],
                _Binding(
                    "module", alias.name if alias.asname else alias.name.split(".")[0]
                ),
            )


def _lookup(name: str, scope: _Scope) -> tuple[_Binding | None, str]:
    current: _Scope | None = scope
    while current is not None:
        if name in current.redirected:
            return None, "global/nonlocal binding requires flow analysis"
        bindings = current.bindings.get(name, ())
        if bindings:
            if len(bindings) != 1:
                return None, "rebound name has no unique static definition"
            value = bindings[0]
            if value.kind == "dynamic":
                return None, "local/parameter binding requires runtime data"
            return value, ""
        if current.wildcard_import:
            return None, "wildcard import may change the binding"
        parent = current.parent
        # Class namespaces are not enclosing lexical scopes, including for a
        # nested class or comprehension. The current class body was checked above.
        while parent is not None and parent.kind == "class":
            parent = parent.parent
        current = parent
    return None, "name has no declared lexical binding"


def _container_for(scope: _Scope) -> tuple[str, ...]:
    names = []
    while scope is not None:
        if isinstance(
            scope.node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            names.append(f"{scope.kind}:{scope.node.name}")
        scope = scope.parent
    return tuple(reversed(names))


def build_call_graph(files: Iterable[tuple[Path, str]]) -> CallGraph:
    """Resolve scoped, explicitly imported references across declared files only."""
    parsed: dict[str, _Bindings] = {}
    roots: set[str] = set()
    for path, module in files:
        module = Path(module).as_posix()
        if module in parsed:
            raise ValueError(f"Duplicate declared module: {module}")
        tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
        parsed[module] = _Bindings(module, tree)
        parent = Path(path).parent
        for _ in range(len(Path(module).parts) - 1):
            parent = parent.parent
        roots.add(parent.name)
    package_root = next(iter(roots)) if len(roots) == 1 else None
    definitions = tuple(d for entry in parsed.values() for d in entry.definitions)

    def module_path(parts: list[str]) -> str | None:
        name = "/".join(parts)
        for candidate in (
            f"{name}.py",
            f"{name}/__init__.py" if name else "__init__.py",
        ):
            if candidate in parsed:
                return candidate
        return None

    def import_parts(module: str, level: int, spelling: str) -> list[str] | None:
        suffix = spelling.split(".") if spelling else []
        if level:
            parent = list(Path(module).parent.parts)
            if level - 1 > len(parent):
                return None
            return parent[: len(parent) - level + 1] + suffix
        if not package_root or not suffix or suffix[0] != package_root:
            return None
        return suffix[1:]

    def resolve(expr: ast.AST, module: str, scope: _Scope, visited=frozenset()):
        if isinstance(expr, ast.Name):
            key = (module, id(scope), expr.id)
            if key in visited:
                return None, "cyclic static binding"
            visited = visited | {key}
            binding, reason = _lookup(expr.id, scope)
            if binding is None:
                return None, reason
            if binding.kind == "definition":
                return binding.value, ""
            if binding.kind == "literal":
                return binding, ""
            if binding.kind == "module":
                parts = import_parts(module, 0, binding.value)
                if parts is not None:
                    return ("module", tuple(parts)), ""
                return None, "external module"
            level, spelling, original = binding.value
            parts = import_parts(module, level, spelling)
            if parts is None:
                return None, "external or undeclared import"
            target = module_path(parts)
            if target:
                value, reason = resolve(
                    ast.Name(id=original, ctx=ast.Load()),
                    target,
                    parsed[target].root,
                    visited,
                )
                if value is not None:
                    return value, reason
            if module_path([*parts, original]):
                return ("module", tuple([*parts, original])), ""
            return None, "imported symbol has no unique declared definition"
        if isinstance(expr, ast.Attribute):
            base, reason = resolve(expr.value, module, scope, visited)
            if isinstance(base, tuple) and base[0] == "module":
                parts = list(base[1])
                target = module_path(parts)
                if target:
                    value, reason = resolve(
                        ast.Name(id=expr.attr, ctx=ast.Load()),
                        target,
                        parsed[target].root,
                        visited,
                    )
                    if value is not None:
                        return value, reason
                if module_path([*parts, expr.attr]):
                    return ("module", tuple([*parts, expr.attr])), ""
            return None, reason or "attribute target requires runtime binding"
        return None, "computed target requires runtime binding"

    references: list[Reference] = []
    unresolved: list[Unresolved] = []
    for module, index in parsed.items():

        def emit(expr, origin, kind, keyword=None, trail=frozenset()):
            scope = index.scopes.get(expr, index.scopes[origin])
            target, reason = resolve(expr, module, scope)
            if isinstance(target, Definition):
                references.append(
                    Reference(
                        kind,
                        ast.unparse(expr),
                        module,
                        origin.lineno,
                        _container_for(index.scopes[origin]),
                        target.module,
                        target.source_lines[0],
                        target.kind,
                        keyword,
                    )
                )
            elif isinstance(target, _Binding) and target.kind == "literal":
                if id(target.value) not in trail:
                    for inner in expressions(target.value):
                        emit(inner, origin, kind, keyword, trail | {id(target.value)})
            else:
                unresolved.append(
                    Unresolved(
                        ast.unparse(expr),
                        reason or "module is not a definition",
                        module,
                        origin.lineno,
                    )
                )

        def expressions(root):
            # An attribute is one expression; do not separately resolve its base.
            if isinstance(root, (ast.Name, ast.Attribute)):
                yield root
            else:
                for child in ast.iter_child_nodes(root):
                    yield from expressions(child)

        for node in ast.walk(index.root.node):
            if isinstance(node, ast.AnnAssign):
                for expr in expressions(node.annotation):
                    emit(expr, node, "annotation")
            if not isinstance(node, ast.Call):
                continue
            emit(node.func, node, "call")
            for position, argument in enumerate(node.args):
                for expr in expressions(argument):
                    emit(expr, node, "argument", f"positional:{position}")
            for keyword in node.keywords:
                for expr in expressions(keyword.value):
                    emit(expr, node, "argument", keyword.arg)
    return CallGraph(
        definitions,
        tuple(dict.fromkeys(references)),
        tuple(dict.fromkeys(unresolved)),
        tuple(sorted(parsed)),
    )
