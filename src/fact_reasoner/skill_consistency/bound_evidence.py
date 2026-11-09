"""Static, role-specific evidence for one instruct call at a time.

No target is imported. Only explicit format and requirements arguments are
followed. Unresolved expressions remain gaps, never semantic contradictions.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Sequence

from .analysis_views import AnalysisView
from .ast_facts import (
    _claim,
    _instruction_argument,
    _instruction_text,
    _keyword,
    validator_predicate_description,
)
from .call_graph import build_call_graph
from .code_lineage import CodeLineage, ground_python_atoms
from .comprehensiveness import FactAtom
from .bound_scope import attach_code_scopes, bind_view_requirements, marker_predicate

VERSION = "bound-call-evidence-1.1.0"
ID_VERSION = "bound-call-evidence-1.0.0"  # Preserve call-node identities across metadata upgrades.
RULE = "instruct.format -> schema; instruct.requirements -> Requirement.validation_fn -> simple_validate callback"
BOUND_KINDS = frozenset(
    {
        "model_instruction_behavior",
        "program_requirement_behavior",
        "output_schema_field",
        "output_schema_meaning",
        "requirement_validation_predicate",
    }
)


def _span(node):
    return tuple(range(node.lineno, node.end_lineno + 1))


def _terminal(node):
    return (
        node.id
        if isinstance(node, ast.Name)
        else node.attr
        if isinstance(node, ast.Attribute)
        else ""
    )


def _trusted_import(tree, expression, symbol):
    """Only a unique module-level Mellea import, with no shadowing/rebinding.

    Conservative even when the same name is shadowed in an unrelated scope.
    This is preferable to treating an arbitrary Requirement/simple_validate as
    the Mellea API. It does not establish the receiver type of instruct.
    """
    if not isinstance(expression, ast.Name):
        return False
    name = expression.id
    bindings = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            for alias in n.names:
                if alias.name == "*":
                    bindings.append(False)
                if alias.asname or alias.name:
                    if (alias.asname or alias.name) == name:
                        bindings.append(
                            n in tree.body
                            and n.level == 0
                            and n.module == "mellea.stdlib.requirements"
                            and alias.name == symbol
                        )
        elif isinstance(n, ast.Import):
            if any((a.asname or a.name.split(".")[0]) == name for a in n.names):
                bindings.append(False)
        elif (
            isinstance(n, ast.Name)
            and isinstance(n.ctx, (ast.Store, ast.Del))
            and n.id == name
        ):
            bindings.append(False)
        elif isinstance(n, ast.arg) and n.arg == name:
            bindings.append(False)
        elif (
            isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and n.name == name
        ):
            bindings.append(False)
    return bindings == [True]


@dataclass(frozen=True)
class BoundEvidence:
    bundles: tuple[FactAtom, ...]
    bindings: tuple[dict, ...]
    unresolved: tuple[dict, ...]
    lineages: tuple[CodeLineage, ...]
    call_graph: dict

    def to_dict(self):
        return {
            "version": VERSION,
            "assembly_rule": RULE,
            "bundles": [n.to_dict() for n in self.bundles],
            "bindings": list(self.bindings),
            "unresolved": list(self.unresolved),
            "call_graph": self.call_graph,
            "scope": "Static call configuration only; instruct receiver type and runtime success are not proved.",
            "target_code_imported_or_executed": False,
        }


def build_bound_evidence(
    files: Sequence[tuple[Path, str]], *, lineages: Sequence[CodeLineage] | None = None
) -> BoundEvidence:
    """Build auditable conjunctions along supported syntax edges, never proximity.

    Names must be relative to the declared package root for import resolution.
    Dynamic arguments, same-line call ambiguities, decorated/complex validators,
    and unsupported contracts are explicitly reported. Original members remain
    in the lineage inventory; they are not independent reverse-FR2 contexts.
    """
    graph = build_call_graph(files)
    paths = {name: Path(path) for path, name in files}
    if len(paths) != len(files):
        raise ValueError("Duplicate source name")
    sources = {name: p.read_text() for name, p in paths.items()}
    trees = {name: ast.parse(value) for name, value in sources.items()}
    lineage_list = (
        tuple(lineages)
        if lineages is not None
        else tuple(ground_python_atoms(p, source_file=n) for p, n in files)
    )
    lineage_by_file = {
        obs.source_file: lineage
        for lineage in lineage_list
        for obs in lineage.observations
    }
    definitions = {(d.module, d.source_lines[0]): d for d in graph.definitions}
    calls_by_line = {
        name: Counter(
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and _terminal(n.func) == "instruct"
        )
        for name, tree in trees.items()
    }
    constructor_counts = {
        name: Counter(
            n.lineno
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and _terminal(n.func) == "Requirement"
        )
        for name, tree in trees.items()
    }
    bundles, binding_records, unresolved = [], [], []

    def definition_node(module, line):
        definition = definitions.get((module, line))
        if definition is None:
            return None
        return next(
            (
                n
                for n in ast.walk(trees[module])
                if isinstance(
                    n,
                    (
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                        ast.ClassDef,
                        ast.Assign,
                        ast.AnnAssign,
                    ),
                )
                and n.lineno == line
                and _span(n) == definition.source_lines
            ),
            None,
        )

    for module, tree in trees.items():
        calls = sorted(
            (
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.Call) and _terminal(n.func) == "instruct"
            ),
            key=lambda n: (n.lineno, n.col_offset),
        )
        for call in calls:
            identifier = (
                "BC_"
                + hashlib.sha256(
                    f"{ID_VERSION}:{module}:{call.lineno}:{call.col_offset}".encode()
                ).hexdigest()[:20]
            )
            spans, members, edges, gaps, validators = [], [], [], [], []
            capabilities = {"model_instruction"}
            instruction = _instruction_text(
                _instruction_argument(call), sources[module]
            )
            facts = [
                f"This instruct call requests: {instruction}"
                if instruction
                else "This instruct call has an unresolved instruction expression."
            ]

            def gap(role, reason):
                entry = {
                    "bundle_id": identifier,
                    "source_file": module,
                    "source_lines": list(_span(call)),
                    "role": role,
                    "reason": reason,
                }
                gaps.append(entry)
                unresolved.append(entry)
                facts.append(
                    f"Unresolved {role}: {reason}. No behavior is inferred from this gap."
                )

            def include(name, node, role):
                ls = _span(node)
                lineage = lineage_by_file.get(name)
                relevant = (
                    []
                    if lineage is None
                    else [
                        a
                        for a in lineage.atoms
                        if a.kind in BOUND_KINDS and set(a.source_lines) <= set(ls)
                    ]
                )
                members.extend(relevant)
                observations = (
                    []
                    if lineage is None
                    else [
                        o.id
                        for o in lineage.observations
                        if set(o.source_lines) <= set(ls)
                    ]
                )
                row = {
                    "role": role,
                    "source_file": name,
                    "source_lines": list(ls),
                    "excerpt": "\n".join(sources[name].splitlines()[n - 1] for n in ls),
                    "sha256": hashlib.sha256(paths[name].read_bytes()).hexdigest(),
                    "claim_ids": [a.id for a in relevant],
                    "observation_ids": observations,
                }
                if row not in spans:
                    spans.append(row)
                return relevant

            def references(origin_module, origin, keyword):
                return [
                    e
                    for e in graph.references_from(origin_module, origin.lineno)
                    if e.kind == "argument" and e.keyword == keyword
                ]

            def select_direct(origin_module, origin, keyword, expression):
                # The graph walks nested expressions; require a direct name here
                # so format=factory(Schema) cannot masquerade as format=Schema.
                if not isinstance(expression, (ast.Name, ast.Attribute)):
                    return None
                candidates = [
                    e
                    for e in references(origin_module, origin, keyword)
                    if e.name == ast.unparse(expression)
                ]
                return candidates[0] if len(candidates) == 1 else None

            include(module, call, "call")
            if calls_by_line[module][call.lineno] != 1:
                gap("call binding", "multiple instruct calls share a source line")
            else:
                fmt = _keyword(call, "format")
                edge = select_direct(module, call, "format", fmt)
                if edge is not None:
                    target = definition_node(edge.to_module, edge.to_line)
                    if isinstance(target, ast.ClassDef):
                        edges.append(edge.to_dict())
                        fields = include(edge.to_module, target, "schema")
                        capabilities.add("output_schema")
                        facts.append(
                            f"This call explicitly supplies output format {target.name}. This declares shape, not successful execution."
                        )
                        facts.extend(
                            a.assertion
                            for a in fields
                            if a.kind
                            in {"output_schema_field", "output_schema_meaning"}
                        )
                        if not any(a.kind == "output_schema_field" for a in fields):
                            gap(
                                "schema fields",
                                "no supported direct field declarations",
                            )
                        if any(
                            not (_terminal(base) == "BaseModel")
                            for base in target.bases
                        ):
                            gap(
                                "schema inheritance",
                                "inherited fields are not expanded",
                            )
                    else:
                        gap("format", "format does not resolve to a class")
                elif fmt is not None:
                    gap(
                        "format",
                        "format is not a uniquely resolved direct schema reference",
                    )
                else:
                    facts.append(
                        "No format argument is supplied at this call; no output shape is inferred."
                    )
                reqs = _keyword(call, "requirements")
                if reqs is None:
                    facts.append(
                        "No requirements argument is supplied at this call; no attached checker is inferred."
                    )
                elif not isinstance(reqs, (ast.List, ast.Tuple)):
                    gap(
                        "requirements",
                        "only literal lists or tuples of explicit requirement references are supported",
                    )
                else:
                    for expr in reqs.elts:
                        edge = select_direct(module, call, "requirements", expr)
                        if edge is None:
                            gap(
                                "requirement",
                                "requirement is not a uniquely resolved direct reference",
                            )
                            continue
                        assigned = definition_node(edge.to_module, edge.to_line)
                        req = getattr(assigned, "value", None)
                        rt = trees[edge.to_module]
                        if not isinstance(req, ast.Call) or not _trusted_import(
                            rt, req.func, "Requirement"
                        ):
                            gap(
                                "requirement",
                                "bound value is not a supported Mellea Requirement declaration",
                            )
                            continue
                        if constructor_counts[edge.to_module][req.lineno] > 1:
                            gap(
                                "requirement",
                                "multiple Requirement constructors share a source line",
                            )
                            continue
                        edges.append(edge.to_dict())
                        include(edge.to_module, assigned, "requirement declaration")
                        capabilities.add("requirement_declaration")
                        desc = _instruction_text(
                            _keyword(req, "description"), sources[edge.to_module]
                        )
                        facts.append(
                            f"This call attaches a Requirement whose declared intent is: {desc or 'unresolved description'}. The description itself establishes no implemented check."
                        )
                        wrapper = _keyword(req, "validation_fn")
                        if wrapper is None:
                            facts.append(
                                "This Requirement supplies no validation_fn; no deterministic checker is inferred."
                            )
                            continue
                        if (
                            not isinstance(wrapper, ast.Call)
                            or not _trusted_import(rt, wrapper.func, "simple_validate")
                            or len(wrapper.args) != 1
                            or wrapper.keywords
                        ):
                            gap(
                                "checker",
                                "validation_fn is not a supported simple_validate callback binding",
                            )
                            continue
                        callback = select_direct(
                            edge.to_module, req, "validation_fn", wrapper.args[0]
                        )
                        if callback is None:
                            gap("checker", "callback is unresolved or shadowed")
                            continue
                        function = definition_node(callback.to_module, callback.to_line)
                        if not isinstance(function, ast.FunctionDef):
                            gap(
                                "checker",
                                "callback does not resolve to a synchronous function",
                            )
                            continue
                        edges.append(callback.to_dict())
                        include(callback.to_module, function, "validator")
                        validators.append(
                            {
                                "requirement_edge": edge.to_dict(),
                                "callback_edge": callback.to_dict(),
                                "predicate": marker_predicate(function),
                            }
                        )
                        validator_span = next(
                            s
                            for s in spans
                            if s["role"] == "validator"
                            and s["source_file"] == callback.to_module
                            and s["source_lines"] == list(_span(function))
                        )
                        predicate = validator_predicate_description(
                            function, module_tree=trees[callback.to_module]
                        )
                        if predicate is None:
                            gap(
                                "checker body",
                                "control flow, decorators, mutation or unsupported predicate requires further analysis",
                            )
                            continue
                        capabilities.add("requirement_validation")
                        assertion = f"Bundled runtime validator {function.name} accepts output exactly when {predicate}."
                        linked_claim = _claim(
                            kind="requirement_validation_predicate",
                            assertion=assertion,
                            source_file=callback.to_module,
                            node=function,
                            source_lines=sources[callback.to_module].splitlines(),
                            tags=("requirement", "validation"),
                            behavior_families=("requirement_validation",),
                            granularity="behavior",
                            actor="runtime_validator",
                            entities=(function.name,),
                            atomization_rule="single pure Boolean validator predicate; resolved simple_validate binding",
                        )
                        linked_claim = replace(
                            linked_claim,
                            id="VP_"
                            + hashlib.sha256(
                                f"{callback.to_module}:{callback.to_line}".encode()
                            ).hexdigest()[:20],
                            observation_ids=tuple(validator_span["observation_ids"]),
                        )
                        members.append(linked_claim)
                        if linked_claim.id not in validator_span["claim_ids"]:
                            validator_span["claim_ids"].append(linked_claim.id)
                        facts.append(
                            f"This call attaches checker {function.name} via simple_validate. Its actual predicate accepts output exactly when {predicate}."
                        )
            unique_members = {(a.source_file, a.id): a for a in members}
            node = FactAtom(
                identifier,
                "Connected static evidence for one call:\n- " + "\n- ".join(facts),
                "atom",
                source=module,
                component="code",
                semantic_role="implementation_bundle",
                source_file=module,
                source_lines=_span(call),
                excerpt="\n\n".join(
                    f"{s['role']} — {s['source_file']}:{s['source_lines'][0]}\n{s['excerpt']}"
                    for s in spans
                ),
                observation_ids=tuple(
                    dict.fromkeys(o for span in spans for o in span["observation_ids"])
                ),
                claim_ids=tuple(dict.fromkeys(a.id for a in unique_members.values())),
                unit_id=identifier,
                grounding_status="grounded",
                capability_types=tuple(sorted(capabilities)),
                granularity="implementation_bundle",
                assembly_rule=RULE,
                content_form="bound_call_bundle",
                source_spans=tuple(spans),
            )
            bundles.append(node)
            binding_records.append(
                {
                    "bundle_id": identifier,
                    "edges": edges,
                    "source_spans": spans,
                    "instruction": instruction,
                    "literal_instruction": (
                        _instruction_argument(call).value
                        if isinstance(_instruction_argument(call), ast.Constant)
                        and isinstance(_instruction_argument(call).value, str)
                        else None
                    ),
                    "call_column": call.col_offset,
                    "validators": validators,
                    "member_claims": [a.to_dict() for a in unique_members.values()],
                    "unresolved": gaps,
                }
            )
    bundles = attach_code_scopes(bundles, binding_records)
    return BoundEvidence(
        bundles,
        tuple(binding_records),
        tuple(unresolved),
        lineage_list,
        graph.to_dict(),
    )


def with_bound_evidence(view: AnalysisView, evidence: BoundEvidence) -> AnalysisView:
    """Replace call-related reverse evidence while preserving forward inventory.

    Unbound declarations, schemas and validators remain in the audit inventory.
    Member atoms are not added beside their bundle as independent witnesses.
    Other code operations remain available as before.
    """
    related = {
        a.id
        for lineage in evidence.lineages
        for a in lineage.atoms
        if a.kind in BOUND_KINDS
    }
    remaining = tuple(
        n for n in view.atoms if not related.intersection(n.claim_ids or (n.id,))
    )
    result = replace(
        view,
        comprehensiveness_bundles=evidence.bundles,
        reverse_evidence_override=(*remaining, *evidence.bundles),
    )
    return bind_view_requirements(result, evidence.bundles)
