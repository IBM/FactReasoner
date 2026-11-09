"""Convert selected Python AST operations into fixed natural-language facts.

The extractor parses source text only. It never imports or executes the target.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Iterable

from .doc_claims import behavior_families_for_text, tags_for_text
from .models import EvidenceClaim


URL_RE = re.compile(r"https?://[^\s'\"`<>]+", re.IGNORECASE)


# A clause whose only content is a label introducing a template placeholder is a
# variable-interpolation fragment, not a behavior: ``Problem text: {{ x }}``
# states nothing the documentation could support or contradict. A clause with a
# predicate is kept even when it interpolates, because ``Inspect {path}.`` does
# assert an operation.
_PLACEHOLDER_RE = re.compile(r"\{\{?[^{}]*\}?\}")
_LABEL_THEN_PLACEHOLDER_RE = re.compile(
    r"^[0-9A-Za-z][0-9A-Za-z _-]*:\s*\{\{?[^{}]*\}?\}[\s.]*$"
)


def _is_placeholder_only(text: str) -> bool:
    """True for a bare placeholder, or a label whose only content is one."""

    if _LABEL_THEN_PLACEHOLDER_RE.match(text.strip()):
        return True
    residue = _PLACEHOLDER_RE.sub(" ", text)
    return not re.sub(r"[^0-9A-Za-z]+", " ", residue).strip()


def _atomic_text_parts(text: str) -> tuple[str, ...]:
    """Split only explicit sentence/semicolon boundaries into propositions."""

    parts: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`{])", text):
        for clause in re.split(r"\s*;\s*|\n+", sentence):
            normalized = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", clause)
            normalized = " ".join(normalized.split()).strip()
            if normalized and not _is_placeholder_only(normalized):
                parts.append(normalized)
    return tuple(parts or (" ".join(text.split()),))


def _literal_members(node: ast.AST) -> tuple[str, ...]:
    """Return safe literal collection members, never evaluating target code.

    A mapping decomposes into ``key -> value`` members so a rule table states one
    proposition per entry. Nested mappings render their own entries, because a
    table such as problem type -> properties carries one requirement per row.
    """

    if isinstance(node, ast.Dict):
        members: list[str] = []
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            if key_node is None:
                return ()
            try:
                key = ast.literal_eval(key_node)
                value = ast.literal_eval(value_node)
            except (ValueError, TypeError):
                return ()
            if not isinstance(key, (str, int, float, bool)):
                return ()
            if isinstance(value, dict):
                rendered = ", ".join(
                    f"{inner_key}: {inner_value}"
                    for inner_key, inner_value in value.items()
                )
                members.append(f"{key!r} -> {rendered}")
            elif isinstance(value, (list, tuple, set)):
                rendered = ", ".join(str(item) for item in value)
                members.append(f"{key!r} -> {rendered}")
            elif isinstance(value, (str, int, float, bool)):
                members.append(f"{key!r} -> {value}")
            else:
                return ()
        return tuple(members)
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return ()
    members = []
    for item in node.elts:
        try:
            value = ast.literal_eval(item)
        except (ValueError, TypeError):
            return ()
        if isinstance(value, (str, int, float, bool)):
            members.append(repr(value))
        else:
            return ()
    return tuple(members)


def _attribute_name(node: ast.AST, aliases: dict[str, str]) -> str:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _attribute_name(node.value, aliases)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _imports(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.asname:
                    aliases[item.asname] = item.name
                else:
                    top_level = item.name.split(".")[0]
                    aliases[top_level] = top_level
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"
    return aliases


def _assignments(tree: ast.AST) -> dict[str, ast.AST]:
    values: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                values[node.target.id] = node.value
    return values


def _source_text(node: ast.AST, source: str) -> str:
    segment = ast.get_source_segment(source, node)
    return " ".join(segment.split()) if segment else node.__class__.__name__


def _render_expression(
    node: ast.AST | None,
    source: str,
    assignments: dict[str, ast.AST],
    seen: frozenset[str] = frozenset(),
) -> str:
    if node is None:
        return "<not supplied>"
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Name) and node.id in assignments and node.id not in seen:
        resolved = _render_expression(
            assignments[node.id], source, assignments, seen | {node.id}
        )
        return f"{node.id} = {resolved}"
    if isinstance(node, (ast.List, ast.Tuple)):
        opener, closer = ("[", "]") if isinstance(node, ast.List) else ("(", ")")
        return (
            opener
            + ", ".join(
                _render_expression(item, source, assignments, seen)
                for item in node.elts
            )
            + closer
        )
    return _source_text(node, source)


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    for item in call.keywords:
        if item.arg == name:
            return item.value
    return None


def _is_true(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _line_excerpt(source_lines: list[str], node: ast.AST) -> str:
    start = max(getattr(node, "lineno", 1) - 1, 0)
    end = max(getattr(node, "end_lineno", start + 1), start + 1)
    return "\n".join(source_lines[start:end]).strip()


def _lines(node: ast.AST) -> tuple[int, ...]:
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", start)
    return tuple(range(start, end + 1))


def _instruction_text(node: ast.AST | None, source: str) -> str | None:
    """Return stable prompt/requirement text without evaluating expressions."""

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return " ".join(node.value.split())
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                expression = _source_text(value.value, source)
                parts.append("{" + expression + "}")
        return " ".join("".join(parts).split())
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _instruction_text(node.left, source)
        right = _instruction_text(node.right, source)
        if left is None:
            left = "{" + _source_text(node.left, source) + "}"
        if right is None:
            right = "{" + _source_text(node.right, source) + "}"
        return " ".join(f"{left} {right}".split())
    return None


def _semantic_families(text: str, *base: str) -> tuple[str, ...]:
    """Add coarse routing hints while preserving the full behavioral claim."""

    lowered = text.lower()
    families = set(base) | set(behavior_families_for_text(text))
    hints = {
        "security_analysis": (
            "security",
            "vulnerabil",
            "attack surface",
            "injection",
            "authorization",
            "csrf",
            "cryptograph",
        ),
        "repository_inspection": (
            "repository",
            "diff",
            "changed file",
            "modified file",
            "source file",
        ),
        "filesystem_read": ("read file", "reads lines", "file content"),
        "output_generation": ("output", "report", "return", "result"),
        "error_handling": ("error", "failure", "fallback", "unverified"),
        "user_interaction": ("user", "query", "request"),
    }
    families.update(
        family
        for family, terms in hints.items()
        if any(term in lowered for term in terms)
    )
    return tuple(sorted(families))


def _claim(
    *,
    kind: str,
    assertion: str,
    source_file: str,
    node: ast.AST,
    source_lines: list[str],
    tags: Iterable[str],
    behavior_families: Iterable[str],
    granularity: str,
    actor: str = "local_process",
    semantic_role: str | None = None,
    condition: str | None = None,
    entities: Iterable[str] = (),
    scoring_eligible: bool = True,
    atomization_rule: str = "one deterministic semantic proposition",
) -> EvidenceClaim:
    return EvidenceClaim(
        id="pending",
        kind=kind,
        assertion=assertion,
        source_file=source_file,
        source_lines=_lines(node),
        excerpt=_line_excerpt(source_lines, node),
        tags=tuple(sorted(set(tags))),
        derivation="deterministic Python AST rule and fixed template",
        behavior_families=tuple(sorted(set(behavior_families))),
        granularity=granularity,
        component="code",
        semantic_role=(
            semantic_role
            or ("mechanism" if granularity == "mechanism" else "operation")
        ),
        actor=actor,
        content_form="python_source",
        grounding_status="grounded",
        entities=tuple(sorted(set(entities) | set(URL_RE.findall(assertion)))),
        condition=condition,
        polarity=(
            "negative"
            if re.search(
                r"\b(?:do not|don't|never|must not|cannot|can't|without|no|not)\b",
                assertion,
                re.IGNORECASE,
            )
            else "positive"
        ),
        scoring_eligible=scoring_eligible,
        documentation_role="not_applicable",
        atomization_rule=atomization_rule,
    )


_BEHAVIORAL_CONFIGURATION_MARKERS = (
    "CHECKLIST",
    "ENDPOINT",
    "OUT_OF_SCOPE",
    "PREFIX",
    "PRESET",
    "PRIORITY",
    "READ_ONLY",
    "REQUIRED",
    "REQUIRES",
    "SKIP",
)


def _assignment_target_and_value(
    node: ast.AST,
) -> tuple[str, ast.AST | None] | None:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        return (target.id, node.value) if isinstance(target, ast.Name) else None
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id, node.value
    return None


def _referenced_assignment_names(
    node: ast.AST | None,
    assignments: dict[str, ast.AST],
    *,
    seen: frozenset[str] = frozenset(),
) -> set[str]:
    """Trace names through local assignments without evaluating target code."""

    if node is None:
        return set()
    names = {
        item.id
        for item in ast.walk(node)
        if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
    }
    result = set(names)
    for name in names - seen:
        assigned = assignments.get(name)
        if assigned is not None:
            result.update(
                _referenced_assignment_names(assigned, assignments, seen=seen | {name})
            )
    return result


def _configuration_names_consumed_by_behavior(
    tree: ast.Module,
) -> set[str]:
    """Find constants that flow into declared model/validation behavior."""

    aliases = _imports(tree)
    assignments = _assignments(tree)
    consumed: set[str] = set()
    for call in (item for item in ast.walk(tree) if isinstance(item, ast.Call)):
        name = _attribute_name(call.func, aliases)
        terminal = name.rsplit(".", 1)[-1]
        if not (
            name.endswith(".instruct") or terminal in {"instruct", "req", "Requirement"}
        ):
            continue
        for value in (*call.args, *(item.value for item in call.keywords)):
            consumed.update(_referenced_assignment_names(value, assignments))
    return {name for name in consumed if name.lstrip("_").isupper()}


# Deployment settings select *where* a program runs, not *what* it does, so they
# are not policy the specification would describe. Recognized structurally: a
# scalar whose name denotes a backend, model, credential, timeout or budget.
_RUNTIME_TUNING_NAMES = (
    "BACKEND",
    "BASE_URL",
    "BUDGET",
    "HOST",
    "KEY",
    "LOOP",
    "MODEL_ID",
    "PORT",
    "RETRY",
    "RETRIES",
    "SEED",
    "TEMPERATURE",
    "TIMEOUT",
    "TOKEN",
    "VERSION",
)


def _is_runtime_tuning(normalized_name: str, value: ast.AST | None) -> bool:
    """True for a scalar deployment setting rather than a behavioral policy."""

    if not isinstance(value, ast.Constant):
        return False
    if isinstance(value.value, str) and len(value.value) > 80:
        return False
    parts = set(normalized_name.split("_"))
    return any(
        item in parts or normalized_name == item for item in _RUNTIME_TUNING_NAMES
    )


def _behavioral_configuration_facts(
    tree: ast.Module,
    source: str,
    source_file: str,
    source_lines: list[str],
    external_consumed_names: frozenset[str] = frozenset(),
) -> list[EvidenceClaim]:
    """Expose behavior-affecting constants without treating runtime tuning as policy.

    A compiled package usually declares constants in one module and consumes them
    in another, so consumption observed in sibling modules counts. Without it, a
    definition module looks unused and its policy is silently dropped.
    """

    consumed_names = _configuration_names_consumed_by_behavior(tree) | set(
        external_consumed_names
    )
    facts: list[EvidenceClaim] = []
    for node in tree.body:
        assignment = _assignment_target_and_value(node)
        if assignment is None:
            continue
        name, value = assignment
        normalized_name = name.lstrip("_")
        if not normalized_name.isupper() or _is_runtime_tuning(normalized_name, value):
            continue
        if not (
            name in consumed_names
            or any(
                marker in normalized_name
                for marker in _BEHAVIORAL_CONFIGURATION_MARKERS
            )
        ):
            continue
        if value is None:
            continue
        rendered = _instruction_text(value, source) or _source_text(value, source)
        if not rendered or len(rendered) > 1600:
            continue
        members = _literal_members(value)
        if members:
            assertions = [
                (
                    f"Bundled code configures {name} to include {member}.",
                    "literal configuration collection split into member claims",
                )
                for member in members
            ]
        elif name == "PREFIX_TEXT":
            assertions = [
                (
                    f"Bundled code configures this behavioral policy: {part}",
                    "behavioral policy split at explicit sentence boundaries",
                )
                for part in _atomic_text_parts(rendered)
            ]
        else:
            assertions = [
                (
                    f"Bundled code configures {name} as {rendered}.",
                    "one scalar behavioral configuration",
                )
            ]
        for assertion, atomization_rule in assertions:
            facts.append(
                _claim(
                    kind="behavioral_configuration",
                    assertion=assertion,
                    source_file=source_file,
                    node=node,
                    source_lines=source_lines,
                    tags=(*tags_for_text(assertion), "configuration"),
                    behavior_families=_semantic_families(
                        f"{name} {rendered}", "behavioral_configuration"
                    ),
                    granularity="behavior",
                    semantic_role="constraint",
                    entities=(name,),
                    atomization_rule=atomization_rule,
                )
            )
    return facts


def _field_description(value: ast.AST | None, source: str) -> str | None:
    if not isinstance(value, ast.Call):
        return None
    description = _keyword(value, "description")
    return _instruction_text(description, source)


def _field_is_required(value: ast.AST | None) -> bool:
    if value is None:
        return True
    if not isinstance(value, ast.Call):
        return False
    terminal = _attribute_name(value.func, {}).rsplit(".", 1)[-1]
    if terminal != "Field":
        return False
    if value.args:
        first = value.args[0]
        if not (isinstance(first, ast.Constant) and first.value is ...):
            return False
    return (
        _keyword(value, "default") is None
        and _keyword(value, "default_factory") is None
    )


def validator_predicate_description(
    function: ast.AST, *, module_tree: ast.Module | None = None
) -> str | None:
    """Normalize supported string predicates without executing target code.

    Supported: literal containment, lower(), length comparisons, Boolean
    connectives, and any/all over literal marker lists. A straight-line literal
    assignment may supply that list. Branches, mutation, helpers and decorated
    callbacks remain unresolved. Builtins must not be shadowed in supplied AST.
    """
    if not isinstance(function, ast.FunctionDef) or function.decorator_list:
        return None
    args = function.args
    parameters = [*args.posonlyargs, *args.args]
    if len(parameters) != 1 or args.vararg or args.kwarg or args.kwonlyargs:
        return None
    parameter = parameters[0].arg
    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    literals = {}
    for statement in body[:-1]:
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            return None
        name = statement.targets[0].id
        value = statement.value
        if (
            name == parameter
            or name in literals
            or not isinstance(value, (ast.List, ast.Tuple))
        ):
            return None
        if not all(
            isinstance(v, ast.Constant) and isinstance(v.value, str) for v in value.elts
        ):
            return None
        literals[name] = value.elts
    if not body or not isinstance(body[-1], ast.Return):
        return None

    def builtin(name):
        if module_tree is None or name == parameter or name in literals:
            return False
        # Fail closed even for rebinding in an unrelated lexical scope.
        for n in ast.walk(module_tree):
            if (
                isinstance(n, ast.Name)
                and isinstance(n.ctx, (ast.Store, ast.Del))
                and n.id == name
            ):
                return False
            if isinstance(n, ast.arg) and n.arg == name:
                return False
            if (
                isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and n.name == name
            ):
                return False
            if isinstance(n, (ast.Import, ast.ImportFrom)) and any(
                a.name == "*" or (a.asname or a.name.split(".")[0]) == name
                for a in n.names
            ):
                return False
        return True

    def output(node):
        if isinstance(node, ast.Name) and node.id == parameter:
            return "output"
        if (
            isinstance(node, ast.Call)
            and not node.args
            and not node.keywords
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "lower"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == parameter
        ):
            return "lowercased output"
        return None

    def describe(node, bound=None):
        bound = bound or {}
        if isinstance(node, ast.Constant) and type(node.value) is bool:
            return (
                "true (all outputs accepted)"
                if node.value
                else "false (all outputs rejected)"
            )
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = node.left, node.comparators[0]
            if isinstance(left, ast.Name):
                left = bound.get(left.id, left)
            rendered = output(right)
            if (
                isinstance(left, ast.Constant)
                and isinstance(left.value, str)
                and rendered
                and isinstance(node.ops[0], (ast.In, ast.NotIn))
            ):
                verb = (
                    "contains"
                    if isinstance(node.ops[0], ast.In)
                    else "does not contain"
                )
                return f"the {rendered} {verb} the literal marker {left.value!r}"
            operators = {
                ast.Gt: ">",
                ast.GtE: ">=",
                ast.Lt: "<",
                ast.LtE: "<=",
                ast.Eq: "==",
                ast.NotEq: "!=",
            }
            if (
                isinstance(left, ast.Call)
                and isinstance(left.func, ast.Name)
                and left.func.id == "len"
                and builtin("len")
                and len(left.args) == 1
                and not left.keywords
                and output(left.args[0])
                and isinstance(right, ast.Constant)
                and type(right.value) is int
                and type(node.ops[0]) in operators
            ):
                return f"the {output(left.args[0])} length {operators[type(node.ops[0])]} {right.value}"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            inner = describe(node.operand, bound)
            return None if inner is None else f"NOT ({inner})"
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            parts = [describe(value, bound) for value in node.values]
            if any(part is None for part in parts):
                return None
            return (" AND " if isinstance(node.op, ast.And) else " OR ").join(
                f"({part})" for part in parts
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"any", "all"}
            and builtin(node.func.id)
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.GeneratorExp)
        ):
            gen = node.args[0]
            if len(gen.generators) != 1:
                return None
            clause = gen.generators[0]
            if (
                clause.is_async
                or clause.ifs
                or not isinstance(clause.target, ast.Name)
                or clause.target.id == parameter
            ):
                return None
            values = (
                clause.iter.elts
                if isinstance(clause.iter, (ast.List, ast.Tuple))
                else literals.get(clause.iter.id)
                if isinstance(clause.iter, ast.Name)
                else None
            )
            if values is None or not all(
                isinstance(v, ast.Constant) and isinstance(v.value, str) for v in values
            ):
                return None
            parts = [describe(gen.elt, {**bound, clause.target.id: v}) for v in values]
            if any(part is None for part in parts):
                return None
            if not parts:
                return (
                    "true (all outputs accepted)"
                    if node.func.id == "all"
                    else "false (all outputs rejected)"
                )
            return (" AND " if node.func.id == "all" else " OR ").join(
                f"({part})" for part in parts
            )
        return None

    return describe(body[-1].value)


def _requirement_facts(
    tree: ast.Module,
    source: str,
    source_file: str,
    source_lines: list[str],
) -> list[EvidenceClaim]:
    """Expose Mellea requirement prose and its deterministic predicate separately."""

    aliases = _imports(tree)
    functions = {
        item.name: item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    facts: list[EvidenceClaim] = []
    for node in tree.body:
        assignment = _assignment_target_and_value(node)
        if assignment is None:
            continue
        requirement_name, value = assignment
        if not isinstance(value, ast.Call):
            continue
        if _attribute_name(value.func, aliases).rsplit(".", 1)[-1] != "Requirement":
            continue

        description = _instruction_text(_keyword(value, "description"), source)
        if description:
            for part in _atomic_text_parts(description):
                facts.append(
                    _claim(
                        kind="program_requirement_behavior",
                        assertion=(
                            f"Bundled code declares requirement {requirement_name}: {part}"
                        ),
                        source_file=source_file,
                        node=node,
                        source_lines=source_lines,
                        tags=(*tags_for_text(part), "requirement"),
                        behavior_families=_semantic_families(
                            part, "requirement_enforcement"
                        ),
                        granularity="behavior",
                        actor="requirement_specification",
                        semantic_role="declaration",
                        entities=(requirement_name,),
                        atomization_rule="Requirement description split at explicit sentence boundaries",
                    )
                )

        validation = _keyword(value, "validation_fn")
        predicate_name: str | None = None
        if isinstance(validation, ast.Call) and (
            _attribute_name(validation.func, aliases).rsplit(".", 1)[-1]
            == "simple_validate"
        ):
            predicate = validation.args[0] if validation.args else None
            if isinstance(predicate, ast.Name):
                predicate_name = predicate.id
        predicate_function = functions.get(predicate_name or "")
        if predicate_function is None:
            continue
        predicate = validator_predicate_description(
            predicate_function, module_tree=tree
        )
        if predicate is None:
            # Branches, nested functions, mutations and helper calls are not an
            # unconditional acceptance predicate. Keep their source observations.
            continue
        assertion = f"Bundled runtime validator {predicate_function.name} accepts output exactly when {predicate}."
        facts.append(
            _claim(
                kind="requirement_validation_predicate",
                assertion=assertion,
                source_file=source_file,
                node=predicate_function,
                source_lines=source_lines,
                tags=(*tags_for_text(assertion), "requirement", "validation"),
                behavior_families=_semantic_families(
                    predicate, "requirement_validation"
                ),
                granularity="behavior",
                actor="runtime_validator",
                entities=(requirement_name, predicate_function.name),
                atomization_rule="single pure Boolean validator predicate; no inferred control flow",
            )
        )
    return facts


def _schema_contract_facts(
    tree: ast.Module,
    source: str,
    source_file: str,
    source_lines: list[str],
) -> list[EvidenceClaim]:
    """Ground one behavior atom per declared Pydantic output field."""

    aliases = _imports(tree)
    facts: list[EvidenceClaim] = []
    for class_node in tree.body:
        if not isinstance(class_node, ast.ClassDef):
            continue
        if not any(
            _attribute_name(base, aliases).endswith("BaseModel")
            for base in class_node.bases
        ):
            continue
        for field in class_node.body:
            if not isinstance(field, ast.AnnAssign) or not isinstance(
                field.target, ast.Name
            ):
                continue
            field_name = field.target.id
            field_type = _source_text(field.annotation, source)
            requirement = "required" if _field_is_required(field.value) else "optional"
            description = _field_description(field.value, source)
            base_assertion = (
                f"Bundled code defines {requirement} output field "
                f"{class_node.name}.{field_name} with type {field_type}."
            )
            assertions = [
                (
                    "output_schema_field",
                    base_assertion,
                    "field declaration separated from its description",
                )
            ]
            if description:
                assertions.extend(
                    (
                        "output_schema_meaning",
                        f"Bundled code defines {class_node.name}.{field_name} to mean: {part}",
                        "field description split at explicit sentence boundaries",
                    )
                    for part in _atomic_text_parts(description)
                )
            for claim_kind, assertion, atomization_rule in assertions:
                facts.append(
                    _claim(
                        kind=claim_kind,
                        assertion=assertion,
                        source_file=source_file,
                        node=field,
                        source_lines=source_lines,
                        tags=(*tags_for_text(assertion), "output", "schema"),
                        behavior_families=_semantic_families(
                            f"{field_name} {description or ''}", "output_schema"
                        ),
                        granularity="behavior",
                        entities=(f"{class_node.name}.{field_name}",),
                        atomization_rule=atomization_rule,
                    )
                )
    return facts


def _conditional_outcome_facts(
    tree: ast.Module,
    source: str,
    source_file: str,
    source_lines: list[str],
) -> list[EvidenceClaim]:
    """Ground explicit guard outcomes without inferring paths or intent."""

    facts: list[EvidenceClaim] = []
    assignments = _assignments(tree)
    for branch in ast.walk(tree):
        if not isinstance(branch, ast.If):
            continue
        condition = _source_text(branch.test, source)
        for outcome in branch.body:
            if isinstance(outcome, ast.Return):
                # A literal/user-facing outcome is grounded by its syntax.  A
                # generic ``return value`` is only control flow and remains in
                # the non-scoring structural observation layer.
                rendered = _instruction_text(outcome.value, source)
                if rendered is None:
                    continue
                assertion = (
                    f"When {condition}, bundled code returns this outcome: {rendered}."
                )
                kind = "conditional_return_behavior"
                families = _semantic_families(
                    f"{condition} {rendered}",
                    "control_flow_constraint",
                    "output_generation",
                )
            elif isinstance(outcome, ast.Raise):
                rendered = _render_expression(outcome.exc, source, assignments)
                assertion = f"When {condition}, bundled code raises {rendered}."
                kind = "conditional_error_behavior"
                families = _semantic_families(
                    f"{condition} {rendered}",
                    "control_flow_constraint",
                    "error_handling",
                )
            else:
                continue
            if len(assertion) > 1800:
                continue
            facts.append(
                _claim(
                    kind=kind,
                    assertion=assertion,
                    source_file=source_file,
                    node=outcome,
                    source_lines=source_lines,
                    tags=(*tags_for_text(assertion), "condition"),
                    behavior_families=families,
                    granularity="behavior",
                    semantic_role="constraint",
                    condition=condition,
                )
            )
    return facts


def _url_assignments(
    tree: ast.AST,
    source_file: str,
    source_lines: list[str],
) -> list[EvidenceClaim]:
    facts: list[EvidenceClaim] = []
    for node in ast.walk(tree):
        target: ast.AST | None = None
        value: ast.AST | None = None
        if isinstance(node, ast.Assign) and node.targets:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if target is None or not isinstance(value, ast.Constant):
            continue
        if not isinstance(value.value, str) or not URL_RE.match(value.value):
            continue
        name = _attribute_name(target, {}) or _source_text(target, "")
        facts.append(
            _claim(
                kind="embedded_url_behavior",
                assertion=f"Bundled code references URL {value.value}.",
                source_file=source_file,
                node=node,
                source_lines=source_lines,
                tags=("network", "url"),
                behavior_families=("network_destination",),
                granularity="behavior",
            )
        )
        facts.append(
            _claim(
                kind="embedded_url",
                assertion=f"Bundled code assigns URL {value.value} to {name}.",
                source_file=source_file,
                node=node,
                source_lines=source_lines,
                tags=("network", "url"),
                behavior_families=("network_destination",),
                granularity="mechanism",
            )
        )
    return facts


# Instruction-bearing calls appear in two argument styles across compiled
# packages: the instruction as the first positional argument, or as a declared
# keyword. The recorded semantics are identical, so extraction must not depend
# on which style a generator emitted. Keyword names are pinned here rather than
# inferred; an unlisted keyword yields no atom instead of a guessed one.
INSTRUCTION_ARGUMENT_KEYWORDS = ("description", "prompt", "instruction")
INSTRUCTION_ARGUMENT_POLICY_VERSION = "instruction-argument-style-1.0.0"


def _instruction_argument(call: ast.Call) -> ast.expr | None:
    """Return the instruction expression from either supported argument style."""

    if call.args:
        return call.args[0]
    for keyword in call.keywords:
        if keyword.arg in INSTRUCTION_ARGUMENT_KEYWORDS:
            return keyword.value
    return None


def _call_facts(
    tree: ast.AST,
    source: str,
    source_file: str,
    source_lines: list[str],
) -> list[EvidenceClaim]:
    aliases = _imports(tree)
    assignments = _assignments(tree)
    facts: list[EvidenceClaim] = []

    calls = sorted(
        (node for node in ast.walk(tree) if isinstance(node, ast.Call)),
        key=lambda node: (node.lineno, node.col_offset),
    )
    for call in calls:
        name = _attribute_name(call.func, aliases)
        instruction_argument = _instruction_argument(call)
        first = _render_expression(
            call.args[0] if call.args else None, source, assignments
        )
        lower_command = first.lower()

        if name.endswith(".instruct") or name == "instruct":
            instruction = _instruction_text(instruction_argument, source)
            if instruction:
                for part in _atomic_text_parts(instruction):
                    facts.append(
                        _claim(
                            kind="model_instruction_behavior",
                            assertion=(
                                f"Bundled code instructs a language model to: {part}"
                            ),
                            source_file=source_file,
                            node=call,
                            source_lines=source_lines,
                            tags=(*tags_for_text(part), "model"),
                            behavior_families=_semantic_families(
                                part, "model_instruction"
                            ),
                            granularity="behavior",
                            actor="language_model",
                            atomization_rule="model instruction split at explicit sentence boundaries",
                        )
                    )

        if name.endswith(".req") or name == "req":
            requirement = _instruction_text(instruction_argument, source)
            if requirement:
                for part in _atomic_text_parts(requirement):
                    facts.append(
                        _claim(
                            kind="program_requirement_behavior",
                            assertion=(
                                f"Bundled code declares this output requirement: {part}"
                            ),
                            source_file=source_file,
                            node=call,
                            source_lines=source_lines,
                            tags=(*tags_for_text(part), "requirement"),
                            behavior_families=_semantic_families(
                                part, "requirement_enforcement"
                            ),
                            granularity="behavior",
                            actor="requirement_specification",
                            semantic_role="declaration",
                            atomization_rule="runtime requirement declaration split at explicit sentence boundaries",
                        )
                    )

        terminal_name = name.rsplit(".", 1)[-1]
        if terminal_name in {"search_fn", "_default_search_fn"}:
            facts.append(
                _claim(
                    kind="repository_search_behavior",
                    assertion="Bundled code searches repository content.",
                    source_file=source_file,
                    node=call,
                    source_lines=source_lines,
                    tags=("filesystem", "search"),
                    behavior_families=("repository_inspection",),
                    granularity="behavior",
                )
            )
        if terminal_name in {"read_file_fn", "_default_read_file_fn"}:
            facts.append(
                _claim(
                    kind="repository_file_read_behavior",
                    assertion="Bundled code reads local repository file contents.",
                    source_file=source_file,
                    node=call,
                    source_lines=source_lines,
                    tags=("filesystem", "read"),
                    behavior_families=("filesystem_read", "repository_inspection"),
                    granularity="behavior",
                )
            )

        if name in {"urllib.request.urlopen", "requests.get", "requests.post"}:
            facts.append(
                _claim(
                    kind="network_request_behavior",
                    assertion="Bundled code makes an outbound network request.",
                    source_file=source_file,
                    node=call,
                    source_lines=source_lines,
                    tags=("network", "url"),
                    behavior_families=("network_request",),
                    granularity="behavior",
                )
            )
            facts.append(
                _claim(
                    kind="network_request",
                    assertion=f"Bundled code calls {name} with argument {first}.",
                    source_file=source_file,
                    node=call,
                    source_lines=source_lines,
                    tags=("network", "url"),
                    behavior_families=("network_request",),
                    granularity="mechanism",
                )
            )

        if name == "tempfile.mkdtemp":
            facts.append(
                _claim(
                    kind="temporary_directory_behavior",
                    assertion="Bundled code creates a temporary directory.",
                    source_file=source_file,
                    node=call,
                    source_lines=source_lines,
                    tags=("filesystem", "temporary", "write"),
                    behavior_families=("directory_creation",),
                    granularity="behavior",
                )
            )
            facts.append(
                _claim(
                    kind="temporary_directory",
                    assertion="Bundled code creates a temporary directory with tempfile.mkdtemp.",
                    source_file=source_file,
                    node=call,
                    source_lines=source_lines,
                    tags=("filesystem", "temporary", "write"),
                    behavior_families=("directory_creation",),
                    granularity="mechanism",
                )
            )

        if name in {"os.makedirs", "pathlib.Path.mkdir"} or name.endswith(".mkdir"):
            facts.append(
                _claim(
                    kind="directory_creation_behavior",
                    assertion="Bundled code creates a directory.",
                    source_file=source_file,
                    node=call,
                    source_lines=source_lines,
                    tags=("filesystem", "write"),
                    behavior_families=("directory_creation",),
                    granularity="behavior",
                )
            )
            facts.append(
                _claim(
                    kind="directory_creation",
                    assertion=f"Bundled code creates a directory by calling {name} with {first}.",
                    source_file=source_file,
                    node=call,
                    source_lines=source_lines,
                    tags=("filesystem", "write"),
                    behavior_families=("directory_creation",),
                    granularity="mechanism",
                )
            )

        if name == "json.dump":
            target = _render_expression(
                call.args[1] if len(call.args) > 1 else None, source, assignments
            )
            facts.append(
                _claim(
                    kind="json_write_behavior",
                    assertion="Bundled code writes JSON data to a file.",
                    source_file=source_file,
                    node=call,
                    source_lines=source_lines,
                    tags=("filesystem", "json", "write"),
                    behavior_families=("file_write", "json_write"),
                    granularity="behavior",
                )
            )
            facts.append(
                _claim(
                    kind="json_write",
                    assertion=f"Bundled code serializes JSON to {target} with json.dump.",
                    source_file=source_file,
                    node=call,
                    source_lines=source_lines,
                    tags=("filesystem", "json", "write"),
                    behavior_families=("file_write", "json_write"),
                    granularity="mechanism",
                )
            )

        if name in {"subprocess.run", "subprocess.call", "subprocess.Popen"}:
            curl_download = "curl" in lower_command and bool(
                re.search(
                    r"(?:^|[\s'\",\[])(?:-o|--output)(?:[\s'\",\]]|$)", lower_command
                )
            )
            if curl_download:
                behavior_kind = "download_behavior"
                behavior_assertion = "Bundled code downloads a file from a URL."
                mechanism_kind = "curl_download"
                mechanism_assertion = (
                    f"Bundled code invokes curl through {name}: {first}."
                )
                tags = ("command", "download", "execution", "network")
                family = "download"
            elif "curl" in lower_command:
                behavior_kind = "network_request_behavior"
                urls = URL_RE.findall(first)
                target = (
                    urls[0].rstrip(".,;:!?)]}")
                    if urls
                    else "a dynamically resolved URL"
                )
                behavior_assertion = (
                    "Bundled code makes an outbound HTTP request to "
                    f"{target} using curl."
                )
                mechanism_kind = "curl_request"
                mechanism_assertion = (
                    f"Bundled code invokes curl through {name}: {first}."
                )
                tags = ("command", "execution", "network")
                family = "network_request"
            elif "chmod" in lower_command:
                behavior_kind = "permission_change_behavior"
                behavior_assertion = "Bundled code changes file permissions."
                mechanism_kind = "permission_change"
                mechanism_assertion = (
                    f"Bundled code changes file permissions through {name}: {first}."
                )
                tags = ("command", "execution", "filesystem", "permission")
                family = "permission_change"
            elif "git" in lower_command and "diff" in lower_command:
                behavior_kind = "repository_diff_behavior"
                behavior_assertion = (
                    "Bundled code reads repository changes using git diff."
                )
                if "head" in lower_command and (
                    "defaultbranch" in lower_command or "gh repo view" in lower_command
                ):
                    behavior_assertion = (
                        "Bundled code reads repository changes between the resolved "
                        "default branch and HEAD using git diff."
                    )
                mechanism_kind = "git_diff_execution"
                mechanism_assertion = (
                    f"Bundled code invokes git diff through {name}: {first}."
                )
                tags = ("command", "execution", "filesystem")
                family = "repository_inspection"
            elif "gh" in lower_command and "repo" in lower_command:
                behavior_kind = "repository_metadata_behavior"
                behavior_assertion = (
                    "Bundled code reads repository metadata using the GitHub CLI."
                )
                if (
                    "defaultbranchref" in lower_command
                    or "default branch" in lower_command
                ):
                    behavior_assertion = (
                        "Bundled code resolves the repository default branch name "
                        "using the GitHub CLI."
                    )
                mechanism_kind = "github_cli_execution"
                mechanism_assertion = (
                    f"Bundled code invokes the GitHub CLI through {name}: {first}."
                )
                tags = ("command", "execution", "filesystem")
                family = "repository_inspection"
            else:
                behavior_kind = "process_execution_behavior"
                behavior_assertion = f"Bundled code executes this command: {first}."
                mechanism_kind = "subprocess_execution"
                mechanism_assertion = (
                    f"Bundled code executes a subprocess through {name}: {first}."
                )
                tags = ("command", "execution")
                family = "process_execution"
            facts.append(
                _claim(
                    kind=behavior_kind,
                    assertion=behavior_assertion,
                    source_file=source_file,
                    node=call,
                    source_lines=source_lines,
                    tags=tags,
                    behavior_families=(family,),
                    granularity="behavior",
                )
            )
            facts.append(
                _claim(
                    kind=mechanism_kind,
                    assertion=mechanism_assertion,
                    source_file=source_file,
                    node=call,
                    source_lines=source_lines,
                    tags=tags,
                    behavior_families=(family,),
                    granularity="mechanism",
                )
            )
            if _is_true(_keyword(call, "shell")):
                facts.append(
                    _claim(
                        kind="shell_true",
                        assertion=f"Bundled code calls {name} with shell=True.",
                        source_file=source_file,
                        node=call,
                        source_lines=source_lines,
                        tags=("command", "execution", "shell"),
                        behavior_families=("process_execution",),
                        granularity="mechanism",
                    )
                )

        if name in {"threading.Thread", "Thread"}:
            target = _render_expression(_keyword(call, "target"), source, assignments)
            facts.append(
                _claim(
                    kind="background_execution_behavior",
                    assertion=f"Bundled code runs {target} in the background.",
                    source_file=source_file,
                    node=call,
                    source_lines=source_lines,
                    tags=("background", "execution", "thread"),
                    behavior_families=("background_execution",),
                    granularity="behavior",
                )
            )
            facts.append(
                _claim(
                    kind="background_thread",
                    assertion=f"Bundled code creates a background thread with target {target}.",
                    source_file=source_file,
                    node=call,
                    source_lines=source_lines,
                    tags=("background", "execution", "thread"),
                    behavior_families=("background_execution",),
                    granularity="mechanism",
                )
            )

    return facts


def _generative_function_facts(
    tree: ast.AST,
    source: str,
    source_file: str,
    source_lines: list[str],
) -> list[EvidenceClaim]:
    """Extract grounded behavior encoded by Mellea ``@generative`` slots."""

    aliases = _imports(tree)
    facts: list[EvidenceClaim] = []
    functions = sorted(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    for function in functions:
        decorators = []
        for decorator in function.decorator_list:
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            decorators.append(_attribute_name(target, aliases))
        if not any(item.endswith("generative") for item in decorators):
            continue
        instruction = ast.get_docstring(function, clean=True)
        if not instruction:
            continue
        instruction = " ".join(instruction.split())
        for part in _atomic_text_parts(instruction):
            facts.append(
                _claim(
                    kind="generative_slot_behavior",
                    assertion=(
                        f"Bundled generative function {function.name} instructs a "
                        f"language model to: {part}"
                    ),
                    source_file=source_file,
                    node=function,
                    source_lines=source_lines,
                    tags=(*tags_for_text(part), "model"),
                    behavior_families=_semantic_families(part, "model_instruction"),
                    granularity="behavior",
                    actor="language_model",
                    atomization_rule="generative docstring split at explicit sentence boundaries",
                )
            )
    return facts


def configuration_names_consumed_by_behavior(path: Path) -> frozenset[str]:
    """Names this module feeds into model or validation behavior."""

    path = Path(path)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return frozenset(_configuration_names_consumed_by_behavior(tree))


def extract_python_facts(
    path: Path,
    *,
    source_file: str | None = None,
    external_consumed_names: frozenset[str] = frozenset(),
) -> list[EvidenceClaim]:
    """Parse one Python file and return facts for recognized operations."""

    path = Path(path)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    relative = source_file or path.name
    source_lines = source.splitlines()
    candidates = [
        *_url_assignments(tree, relative, source_lines),
        *_behavioral_configuration_facts(
            tree, source, relative, source_lines, external_consumed_names
        ),
        *_requirement_facts(tree, source, relative, source_lines),
        *_schema_contract_facts(tree, source, relative, source_lines),
        *_conditional_outcome_facts(tree, source, relative, source_lines),
        *_generative_function_facts(tree, source, relative, source_lines),
        *_call_facts(tree, source, relative, source_lines),
    ]
    candidates.sort(
        key=lambda item: (
            item.source_lines[0] if item.source_lines else 0,
            item.kind,
            item.assertion,
        )
    )

    seen: set[tuple[object, ...]] = set()
    facts: list[EvidenceClaim] = []
    for candidate in candidates:
        identity = (
            candidate.kind,
            candidate.source_file,
            candidate.source_lines,
            candidate.assertion,
        )
        if identity in seen:
            continue
        seen.add(identity)
        facts.append(
            EvidenceClaim(
                id=f"code_{len(facts)}",
                kind=candidate.kind,
                assertion=candidate.assertion,
                source_file=candidate.source_file,
                source_lines=candidate.source_lines,
                excerpt=candidate.excerpt,
                tags=candidate.tags,
                derivation=candidate.derivation,
                behavior_families=candidate.behavior_families,
                granularity=candidate.granularity,
                occurrences=candidate.occurrences,
                component=candidate.component,
                semantic_role=candidate.semantic_role,
                actor=candidate.actor,
                content_form=candidate.content_form,
                grounding_status=candidate.grounding_status,
                observation_ids=candidate.observation_ids,
                parent_unit_id=candidate.parent_unit_id,
                entities=candidate.entities,
                condition=candidate.condition,
                polarity=candidate.polarity,
                activated_by=candidate.activated_by,
                scoring_eligible=candidate.scoring_eligible,
                documentation_role=candidate.documentation_role,
                atomization_rule=candidate.atomization_rule,
            )
        )
    return facts
