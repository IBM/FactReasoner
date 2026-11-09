"""Source-linked call scopes and conservative Boolean opposition receipts.

These fields are audit metadata, never model inputs. No expected labels are
accepted. A guard can preserve an NLI contradiction only after a unique call
reference and an attached checker's explicit opposite rule are established.
"""

from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import itertools
import json
import re
import textwrap
from typing import Sequence

from .comprehensiveness import FactAtom

VERSION = "bound-call-scope-1.0.0"
MAX_MARKERS = 8
_NOUNS = {
    "extraction": "extract",
    "summarization": "summarize",
    "classification": "classify",
    "validation": "validate",
    "translation": "translate",
    "conversion": "convert",
    "parsing": "parse",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def text_digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def instruction_reference(instruction):
    """Exact literal instruction identity; case and a terminal period normalize."""
    if not isinstance(instruction, str) or not instruction.strip():
        return None
    return (
        "instruction:"
        + " ".join(instruction.strip().removesuffix(".").split()).casefold()
    )


def call_identity(source_file, source_lines, column, source_sha256):
    return "call:" + digest([source_file, list(source_lines), column, source_sha256])


def marker_predicate(function):
    """A pure Boolean abstraction over string membership; never eval/execute.

    This proof grammar is intentionally smaller than the fact extractor. Other
    predicates remain visible evidence but do not receive this opposition proof.
    """
    if not isinstance(function, ast.FunctionDef) or function.decorator_list:
        return None
    args = function.args
    names = [*args.posonlyargs, *args.args]
    if len(names) != 1 or args.vararg or args.kwarg or args.kwonlyargs:
        return None
    parameter = names[0].arg
    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if len(body) != 1 or not isinstance(body[0], ast.Return):
        return None

    def parse(node):
        if isinstance(node, ast.Constant) and type(node.value) is bool:
            return {"op": "constant", "value": node.value}
        if (
            isinstance(node, ast.Compare)
            and len(node.ops) == 1
            and isinstance(node.ops[0], (ast.In, ast.NotIn))
        ):
            left, right = node.left, node.comparators[0]
            normalization = None
            if isinstance(right, ast.Name) and right.id == parameter:
                normalization = "identity"
            if (
                isinstance(right, ast.Call)
                and isinstance(right.func, ast.Attribute)
                and isinstance(right.func.value, ast.Name)
                and right.func.value.id == parameter
                and right.func.attr == "lower"
                and not right.args
                and not right.keywords
            ):
                normalization = "lower"
            if not (
                isinstance(left, ast.Constant)
                and isinstance(left.value, str)
                and left.value
                and normalization
            ):
                return None
            atom = {
                "op": "contains",
                "marker": left.value,
                "normalization": normalization,
            }
            return (
                atom
                if isinstance(node.ops[0], ast.In)
                else {"op": "not", "value": atom}
            )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            child = parse(node.operand)
            return None if child is None else {"op": "not", "value": child}
        if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
            children = [parse(n) for n in node.values]
            if any(n is None for n in children):
                return None
            return {
                "op": "and" if isinstance(node.op, ast.And) else "or",
                "values": children,
            }
        return None

    return parse(body[0].value)


def _atoms(expression):
    op = expression["op"]
    if op == "constant":
        return set()
    if op == "contains":
        return {(expression["marker"], expression["normalization"])}
    if op == "not":
        return _atoms(expression["value"])
    if op in {"and", "or"}:
        return set().union(*(_atoms(n) for n in expression["values"]))
    raise ValueError("Unsupported Boolean expression")


def _value(expression, assignment):
    op = expression["op"]
    if op == "constant":
        return expression["value"]
    if op == "contains":
        return assignment[(expression["marker"], expression["normalization"])]
    if op == "not":
        return not _value(expression["value"], assignment)
    if op == "and":
        return all(_value(n, assignment) for n in expression["values"])
    if op == "or":
        return any(_value(n, assignment) for n in expression["values"])
    raise ValueError("Unsupported Boolean expression")


def complementary(actual, required):
    """Sufficient opposition proof over every Boolean valuation, capped at 256."""
    if actual is None or required is None:
        return False
    actual_atoms, required_atoms = _atoms(actual), _atoms(required)
    if actual_atoms != required_atoms or not 1 <= len(actual_atoms) <= MAX_MARKERS:
        return False
    atoms = sorted(actual_atoms)
    return all(
        _value(actual, dict(zip(atoms, values)))
        != _value(required, dict(zip(atoms, values)))
        for values in itertools.product((False, True), repeat=len(atoms))
    )


def documentation_contract(text):
    """Parse explicit call references and a bounded literal acceptance grammar.

    No fuzzy match, inferred checker name, ontology tag or expected answer.
    Nominal references ('receipt extraction call') normalize to the corresponding
    literal instruction ('Extract receipt.'); resolution must be globally unique.
    Unrecognized prose remains an unresolved reference or acceptance rule.
    """
    value = " ".join(text.removeprefix("Requirement member:\n").split())
    named = re.fullmatch(r"Configure the ([\w -]+?) call (.+)\.", value, re.I)
    explicit = re.fullmatch(
        r'Configure the call with instruction ([\'"])(.+?)\1 (.+)\.', value, re.I
    )
    reference = None
    selector = None
    tail = None
    if explicit:
        reference = instruction_reference(explicit[2])
        selector = "call with instruction " + explicit[1] + explicit[2] + explicit[1]
        tail = explicit[3]
    elif named:
        subject = named[1].strip()
        tail = named[2]
        selector = subject + " call"
        parts = subject.rsplit(" ", 1)
        if len(parts) == 2 and parts[1].casefold() in _NOUNS:
            reference = instruction_reference(
                _NOUNS[parts[1].casefold()] + " " + parts[0]
            )
    result = {"reference": reference, "selector": selector, "acceptance": None}
    if tail is None:
        return result
    prefix = r"(?:with a checker|to request the [A-Za-z_]\w* output format and attach a checker) whose acceptance rule is that "
    match = re.fullmatch(
        prefix + r"(.+) (?:is|are) present in the (lowercased )?output", tail, re.I
    )
    if not match:
        return result
    condition, normalization = match[1], "lower" if match[2] else "identity"
    condition_match = re.fullmatch(
        r"(both|all|either|any) literal markers (.+)", condition, re.I
    )
    single = re.fullmatch(r"the literal marker (.+)", condition, re.I)
    if condition_match:
        quantifier, values = condition_match[1].casefold(), condition_match[2]
        connector = "or" if quantifier in {"either", "any"} else "and"
    elif single:
        quantifier, values, connector = "single", single[1], "and"
    else:
        return result
    # Consume the entire list. Additional conditions or incompatible connectors
    # make the contract unresolved rather than being silently discarded.
    quoted = r"(?:'[^'\\\n]+'|\"[^\"\\\n]+\")"
    pieces = re.findall(quoted, values)
    if not pieces or len(pieces) > MAX_MARKERS:
        return result
    remainder = re.sub(quoted, "#", values)
    if len(pieces) == 1:
        if quantifier not in {"single", "all", "any"} or remainder != "#":
            return result
    else:
        valid_list = re.fullmatch(
            r"#(?:, #)*(?:,? " + connector + r" #)", remainder, re.I
        )
        if (
            not valid_list
            or quantifier == "single"
            or (quantifier == "both" and len(pieces) != 2)
        ):
            return result
    markers = [p[1:-1] for p in pieces]
    if len(set(markers)) != len(markers):
        return result
    atoms = [
        {"op": "contains", "marker": marker, "normalization": normalization}
        for marker in markers
    ]
    result["acceptance"] = (
        atoms[0] if len(atoms) == 1 else {"op": connector, "values": atoms}
    )
    return result


def attach_code_scopes(bundles: Sequence[FactAtom], records):
    """Use already-resolved source edges; bind the complete declared call catalog."""
    staged = []
    for node, record in zip(bundles, records, strict=True):
        call_span = next(s for s in node.source_spans if s["role"] == "call")
        origin = {
            "source_file": node.source_file,
            "source_lines": list(node.source_lines),
            "column": record["call_column"],
            "sha256": call_span["sha256"],
        }
        staged.append(
            replace(
                node,
                scope_binding={
                    "version": VERSION,
                    "kind": "code_call",
                    "statement_sha256": text_digest(node.text),
                    "call_id": call_identity(
                        origin["source_file"],
                        origin["source_lines"],
                        origin["column"],
                        origin["sha256"],
                    ),
                    "origin": origin,
                    "reference": instruction_reference(record["literal_instruction"]),
                    "validators": record["validators"],
                    "edges": record["edges"],
                },
            )
        )
    catalog = sorted(
        [
            {
                "call_id": n.scope_binding["call_id"],
                "reference": n.scope_binding["reference"],
            }
            for n in staged
        ],
        key=lambda r: r["call_id"],
    )
    catalog_hash = digest(catalog)
    return tuple(
        replace(n, scope_binding={**n.scope_binding, "catalog_sha256": catalog_hash})
        for n in staged
    )


def bind_requirement_scopes(nodes: Sequence[FactAtom], all_bundles: Sequence[FactAtom]):
    catalog = sorted(
        [
            {
                "call_id": n.scope_binding["call_id"],
                "reference": n.scope_binding["reference"],
            }
            for n in all_bundles
            if n.scope_binding and n.scope_binding.get("kind") == "code_call"
        ],
        key=lambda r: r["call_id"],
    )
    catalog_hash = digest(catalog)
    if any(
        n.scope_binding and n.scope_binding.get("catalog_sha256") != catalog_hash
        for n in all_bundles
    ):
        raise ValueError(
            "Requirement resolution needs the entire declared call catalog"
        )
    result = []
    for node in nodes:
        contract = documentation_contract(node.text)
        candidates = [
            r["call_id"]
            for r in catalog
            if contract["reference"] is not None
            and r["reference"] == contract["reference"]
        ]
        result.append(
            replace(
                node,
                scope_binding={
                    "version": VERSION,
                    "kind": "documentation_call",
                    "statement_sha256": text_digest(node.text),
                    "source_file": node.source_file,
                    "source_lines": list(node.source_lines),
                    "contract": contract,
                    "catalog": catalog,
                    "catalog_sha256": catalog_hash,
                    "candidate_call_ids": candidates,
                    "status": "resolved"
                    if len(candidates) == 1
                    else "ambiguous"
                    if candidates
                    else "unresolved",
                },
            )
        )
    return tuple(result)


def bind_view_requirements(view, all_bundles):
    return replace(
        view,
        contexts=bind_requirement_scopes(view.contexts, all_bundles),
        requirement_contexts=bind_requirement_scopes(
            view.requirement_contexts, all_bundles
        ),
        requirement_member_contexts=bind_requirement_scopes(
            view.requirement_member_contexts, all_bundles
        ),
    )


def _span(node, role, source_file, start):
    found = [
        s
        for s in node.source_spans
        if s["role"] == role
        and s["source_file"] == source_file
        and start in s["source_lines"]
    ]
    if len(found) != 1:
        raise ValueError("Missing or ambiguous source span")
    return found[0]


def _validated_scope(premise, hypothesis):
    code, doc = premise.scope_binding, hypothesis.scope_binding
    if not code or not doc:
        return "missing_bound_scope", None
    if (
        code["version"] != VERSION
        or doc["version"] != VERSION
        or code["kind"] != "code_call"
        or doc["kind"] != "documentation_call"
    ):
        return "invalid_bound_scope", None
    if code["statement_sha256"] != text_digest(premise.text) or doc[
        "statement_sha256"
    ] != text_digest(hypothesis.text):
        return "stale_bound_scope", None
    if doc["source_file"] != hypothesis.source_file or doc["source_lines"] != list(
        hypothesis.source_lines
    ):
        return "stale_requirement_source", None
    origin = code["origin"]
    if origin["source_file"] != premise.source_file or origin["source_lines"] != list(
        premise.source_lines
    ):
        return "stale_call_source", None
    call = _span(premise, "call", origin["source_file"], origin["source_lines"][0])
    if (
        call["sha256"] != origin["sha256"]
        or call["source_lines"] != origin["source_lines"]
    ):
        return "stale_call_source", None
    if code["call_id"] != call_identity(
        origin["source_file"],
        origin["source_lines"],
        origin["column"],
        origin["sha256"],
    ):
        return "invalid_call_identity", None
    if (
        doc["catalog_sha256"] != code["catalog_sha256"]
        or digest(doc["catalog"]) != code["catalog_sha256"]
    ):
        return "different_call_catalog", None
    if {"call_id": code["call_id"], "reference": code["reference"]} not in doc[
        "catalog"
    ]:
        return "missing_call_in_catalog", None
    contract = documentation_contract(hypothesis.text)
    if contract != doc["contract"]:
        return "stale_requirement_contract", None
    candidates = [
        r["call_id"]
        for r in doc["catalog"]
        if contract["reference"] is not None and r["reference"] == contract["reference"]
    ]
    if candidates != doc["candidate_call_ids"]:
        return "invalid_requirement_resolution", None
    if len(candidates) != 1:
        return "ambiguous_bound_call" if candidates else "unresolved_bound_call", None
    if candidates[0] != code["call_id"]:
        return "different_bound_call", None
    return None, contract


def bound_opposition(premise: FactAtom, hypothesis: FactAtom):
    """Return an auditable, pair-specific proof disposition; no relation creation."""
    result = {"version": VERSION, "accepted": False, "reason": None}
    try:
        reason, contract = _validated_scope(premise, hypothesis)
        if reason:
            return {**result, "reason": reason}
        code = premise.scope_binding
        if contract["acceptance"] is None:
            return {**result, "reason": "unresolved_acceptance_contract"}
        validators = code["validators"]
        if len(validators) != 1:
            return {**result, "reason": "missing_or_multiple_bound_checkers"}
        checker = validators[0]
        requirement, callback = checker["requirement_edge"], checker["callback_edge"]
        origin = code["origin"]
        if requirement not in code["edges"] or callback not in code["edges"]:
            return {**result, "reason": "missing_checker_binding_edges"}
        if (
            requirement["kind"] != "argument"
            or requirement["keyword"] != "requirements"
            or requirement["from_module"] != origin["source_file"]
            or requirement["from_line"] != origin["source_lines"][0]
            or callback["kind"] != "argument"
            or callback["keyword"] != "validation_fn"
            or callback["from_module"] != requirement["to_module"]
        ):
            return {**result, "reason": "unrelated_checker_binding"}
        req_span = _span(
            premise,
            "requirement declaration",
            requirement["to_module"],
            requirement["to_line"],
        )
        if callback["from_line"] not in req_span["source_lines"]:
            return {**result, "reason": "unrelated_checker_binding"}
        source = _span(premise, "validator", callback["to_module"], callback["to_line"])
        if source["source_lines"][0] != callback["to_line"]:
            return {**result, "reason": "unrelated_checker_definition"}
        parsed = ast.parse(textwrap.dedent(source["excerpt"]))
        if len(parsed.body) != 1:
            return {**result, "reason": "unsupported_checker_predicate"}
        expression = marker_predicate(parsed.body[0])
        if expression is None or expression != checker["predicate"]:
            return {**result, "reason": "unsupported_or_stale_checker_predicate"}
        if not complementary(expression, contract["acceptance"]):
            return {**result, "reason": "no_complementary_bound_acceptance"}
        return {
            **result,
            "accepted": True,
            "call_id": code["call_id"],
            "checker_source": {
                "source_file": source["source_file"],
                "source_lines": source["source_lines"],
                "sha256": source["sha256"],
            },
            "actual": expression,
            "required": contract["acceptance"],
            "proof": "Boolean complements under every valuation of the same literal markers",
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        IndexError,
        SyntaxError,
        AttributeError,
        RecursionError,
    ):
        return {**result, "reason": "invalid_bound_provenance"}


def bound_scope_mismatch(premise, hypothesis):
    """Definitively different resolved calls cannot support one another."""
    try:
        reason, _ = _validated_scope(premise, hypothesis)
        return reason == "different_bound_call"
    except (KeyError, TypeError, ValueError, IndexError, AttributeError):
        return False
