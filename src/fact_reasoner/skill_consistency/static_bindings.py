"""Bounded source-only binding expansion; no target evaluation or imports.

Reuses the project's lexical index, adding literal requirement-list aliases and
pure, same-argument predicate helpers. Unknown effects stay explicit gaps.
"""

from __future__ import annotations
import ast
from copy import deepcopy
from pathlib import Path
from .call_graph import _Bindings, _lookup, Reference, _container_for

VERSION = "connected-static-bindings-1.0.0"


class StaticBindings:
    def __init__(self, files):
        self.paths = {name: Path(path) for path, name in files}
        self.sources = {name: path.read_text() for name, path in self.paths.items()}
        self.index = {
            name: _Bindings(name, ast.parse(source))
            for name, source in self.sources.items()
        }
        self.parents = {
            name: {
                child: node
                for node in ast.walk(index.root.node)
                for child in ast.iter_child_nodes(node)
            }
            for name, index in self.index.items()
        }
        roots = set()
        for name, path in self.paths.items():
            root = path.parent
            for _ in range(len(Path(name).parts) - 1):
                root = root.parent
            roots.add(root.name)
        self.package = next(iter(roots)) if len(roots) == 1 else None

    def module_path(self, parts):
        name = "/".join(parts)
        return next(
            (
                x
                for x in (
                    name + ".py",
                    name + "/__init__.py" if name else "__init__.py",
                )
                if x in self.index
            ),
            None,
        )

    def import_parts(self, module, level, spelling):
        suffix = spelling.split(".") if spelling else []
        if level:
            parent = list(Path(module).parent.parts)
            return (
                None
                if level - 1 > len(parent)
                else parent[: len(parent) - level + 1] + suffix
            )
        return (
            suffix[1:]
            if self.package and suffix and suffix[0] == self.package
            else None
        )

    def context(self, module, node):
        rows = []
        parent = self.parents[module]
        current = node
        while current in parent:
            child = current
            current = parent[current]
            if isinstance(current, ast.If):
                branch = "else" if child in current.orelse else "then"
                rows.append(
                    dict(
                        kind="if",
                        condition=ast.unparse(current.test),
                        branch=branch,
                        line=current.lineno,
                    )
                )
            elif isinstance(
                current,
                (
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                    ast.Try,
                    ast.TryStar,
                    ast.With,
                    ast.AsyncWith,
                    ast.ExceptHandler,
                    ast.Match,
                ),
            ):
                suite = next(
                    (
                        field
                        for field, value in ast.iter_fields(current)
                        if isinstance(value, list) and child in value
                    ),
                    "nested",
                )
                rows.append(
                    dict(
                        kind=type(current).__name__,
                        suite=suite,
                        line=current.lineno,
                        condition=ast.unparse(current.test)
                        if isinstance(current, ast.While)
                        else ast.unparse(current.iter)
                        if isinstance(current, (ast.For, ast.AsyncFor))
                        else "execution governed by this control scope",
                    )
                )
        return list(reversed(rows))

    def resolve(self, module, expr, origin=None, trail=frozenset()):
        """Return (module, original AST definition/value); reject uncertain bindings."""
        index = self.index[module]
        origin = origin or expr
        scope = index.scopes.get(expr, index.scopes.get(origin, index.root))
        if isinstance(expr, ast.Name):
            key = (module, id(scope), expr.id)
            if key in trail:
                raise ValueError("cyclic static binding")
            trail = trail | {key}
            binding, reason = _lookup(expr.id, scope)
            if binding is None:
                raise ValueError(reason)
            if binding.kind == "definition":
                d = binding.value
                matches = [
                    n
                    for n in ast.walk(self.index[d.module].root.node)
                    if getattr(n, "lineno", None) == d.source_lines[0]
                    and isinstance(
                        n,
                        (
                            ast.FunctionDef,
                            ast.AsyncFunctionDef,
                            ast.ClassDef,
                            ast.Assign,
                            ast.AnnAssign,
                        ),
                    )
                    and getattr(n, "end_lineno", None) == d.source_lines[-1]
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "multiple definitions share an ambiguous source span"
                    )
                node = matches[0]
                if self.context(d.module, node):
                    raise ValueError("definition depends on a control-flow branch")
                return d.module, node
            if binding.kind == "literal":
                return module, binding.value
            if binding.kind == "module":
                parts = self.import_parts(module, 0, binding.value)
                if parts is not None:
                    return "@module", parts
                raise ValueError("external module")
            if binding.kind != "import":
                raise ValueError("unsupported lexical binding")
            level, spelling, original = binding.value
            parts = self.import_parts(module, level, spelling)
            if parts is None:
                raise ValueError("external or undeclared import")
            target = self.module_path(parts)
            if target:
                try:
                    return self.resolve(
                        target,
                        ast.Name(id=original, ctx=ast.Load()),
                        self.index[target].root.node,
                        trail,
                    )
                except ValueError:
                    pass
            if self.module_path([*parts, original]):
                return "@module", parts + [original]
            raise ValueError("imported symbol has no unique static definition")
        if isinstance(expr, ast.Attribute):
            owner, parts = self.resolve(module, expr.value, origin, trail)
            if owner != "@module":
                raise ValueError("attribute requires runtime object binding")
            target = self.module_path(parts)
            if target:
                try:
                    return self.resolve(
                        target,
                        ast.Name(id=expr.attr, ctx=ast.Load()),
                        self.index[target].root.node,
                        trail,
                    )
                except ValueError:
                    pass
            if self.module_path([*parts, expr.attr]):
                return "@module", parts + [expr.attr]
        raise ValueError("computed target requires runtime binding")

    def edge(self, module, origin, expr, keyword):
        target, node = self.resolve(module, expr, origin)
        if target == "@module":
            raise ValueError("module is not a definition")
        kind = (
            "function"
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            else "class"
            if isinstance(node, ast.ClassDef)
            else "constant"
        )
        return Reference(
            "argument",
            ast.unparse(expr),
            module,
            origin.lineno,
            _container_for(
                self.index[module].scopes.get(origin, self.index[module].root)
            ),
            target,
            node.lineno,
            kind,
            keyword,
        )

    def dereference_container(self, module, expr, trail=frozenset()):
        if isinstance(expr, (ast.List, ast.Tuple)):
            return module, expr, []
        target, node = self.resolve(module, expr)
        key = (target, getattr(node, "lineno", None), getattr(node, "col_offset", None))
        if key in trail:
            raise ValueError("cyclic requirement-list alias")
        if isinstance(node, (ast.List, ast.Tuple)):
            return target, node, [(target, node)]
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            raise ValueError("requirements reference is not a literal sequence")
        value = node.value
        dest, sequence, spans = self.dereference_container(target, value, trail | {key})
        return dest, sequence, [(target, node), *spans]

    def requirements(self, module, expr):
        target, sequence, spans = self.dereference_container(module, expr)
        # A declared mutable list may be inspected only when every syntactically
        # identifiable alias is used as a list alias or an instruct requirement.
        if isinstance(sequence, ast.List):
            for name, index in self.index.items():
                for use in ast.walk(index.root.node):
                    if not isinstance(use, (ast.Name, ast.Attribute)) or not isinstance(
                        getattr(use, "ctx", None), ast.Load
                    ):
                        continue
                    try:
                        _, resolved, _ = self.dereference_container(name, use)
                    except ValueError:
                        continue
                    if resolved is not sequence:
                        continue
                    parent = self.parents[name].get(use)
                    if (
                        isinstance(parent, (ast.Assign, ast.AnnAssign))
                        and parent.value is use
                    ):
                        continue
                    if isinstance(parent, ast.keyword) and parent.arg == "requirements":
                        call = self.parents[name].get(parent)
                        if (
                            isinstance(call, ast.Call)
                            and isinstance(call.func, ast.Attribute)
                            and call.func.attr == "instruct"
                        ):
                            continue
                    if isinstance(parent, ast.Attribute) and parent.value is use:
                        raise ValueError(
                            "requirement list may be mutated or inspected through a method"
                        )
                    # Import-qualified names contain their module Name as a child;
                    # only the complete expression resolves to the list.
                    raise ValueError(
                        "requirement list escapes the supported static binding grammar"
                    )
        if any(not isinstance(e, (ast.Name, ast.Attribute)) for e in sequence.elts):
            raise ValueError(
                "requirement sequence contains computed or starred members"
            )
        return [(target, e) for e in sequence.elts], spans

    def require_unmodified_definition(self, module, node):
        for name, index in self.index.items():
            for target in ast.walk(index.root.node):
                if not isinstance(
                    target, (ast.Attribute, ast.Subscript)
                ) or not isinstance(target.ctx, (ast.Store, ast.Del)):
                    continue
                try:
                    dest, definition = self.resolve(name, target.value, target)
                except ValueError:
                    continue
                if dest == module and definition is node:
                    raise ValueError(
                        "bound definition has attribute or subscript mutation"
                    )

    def expand_predicate(self, module, function):
        """Inline pure helper returns only when the exact input argument is forwarded.

        Returns a synthetic AST solely for the existing deterministic formatter;
        all provenance points to original functions. No AST is evaluated.
        """
        provenance = []
        edges = []

        def expand(name, fn, trail):
            key = (name, fn.lineno)
            if key in trail:
                raise ValueError("cyclic predicate helper")
            args = fn.args
            params = [*args.posonlyargs, *args.args]
            if (
                fn.decorator_list
                or len(params) != 1
                or args.vararg
                or args.kwarg
                or args.kwonlyargs
                or args.defaults
            ):
                raise ValueError(
                    "predicate helper has unsupported signature or decorators"
                )
            body = list(fn.body)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]
            if (
                len(body) != 1
                or not isinstance(body[0], ast.Return)
                or body[0].value is None
            ):
                raise ValueError("predicate helper has control flow or mutation")
            provenance.append((name, fn))
            parameter = params[0].arg

            def visit(expr):
                if isinstance(expr, ast.Call) and isinstance(
                    expr.func, (ast.Name, ast.Attribute)
                ):
                    # Keep recognized operations on this exact parameter intact.
                    if (
                        isinstance(expr.func, ast.Attribute)
                        and isinstance(expr.func.value, ast.Name)
                        and expr.func.value.id == parameter
                    ):
                        return deepcopy(expr)
                    if (
                        len(expr.args) != 1
                        or expr.keywords
                        or not isinstance(expr.args[0], ast.Name)
                        or expr.args[0].id != parameter
                    ):
                        raise ValueError(
                            "helper does not forward the exact checked object"
                        )
                    dest, helper = self.resolve(name, expr.func, expr)
                    if not isinstance(helper, ast.FunctionDef):
                        raise ValueError(
                            "predicate helper is not a synchronous function"
                        )
                    edges.append(
                        self.edge(name, expr, expr.func, "predicate_helper").to_dict()
                    )
                    rewritten = expand(dest, helper, trail | {key})
                    other = (
                        helper.args.args[0].arg
                        if helper.args.args
                        else helper.args.posonlyargs[0].arg
                    )

                    class Rename(ast.NodeTransformer):
                        def visit_Name(self, node):
                            return (
                                ast.copy_location(
                                    ast.Name(id=parameter, ctx=ast.Load()), node
                                )
                                if node.id == other
                                else node
                            )

                    return Rename().visit(rewritten)
                new = deepcopy(expr)
                for field, value in ast.iter_fields(expr):
                    if isinstance(value, ast.AST):
                        setattr(new, field, visit(value))
                    elif isinstance(value, list):
                        setattr(
                            new,
                            field,
                            [visit(v) if isinstance(v, ast.AST) else v for v in value],
                        )
                return new

            return visit(body[0].value)

        expression = expand(module, function, frozenset())
        result = deepcopy(function)
        result.body = [
            ast.copy_location(ast.Return(value=expression), function.body[-1])
        ]
        return ast.fix_missing_locations(result), provenance, edges
