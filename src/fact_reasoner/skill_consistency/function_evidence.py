"""Conservative operation templates in function/control scopes, never execution.

Function grouping preserves individual conditions and local object bindings. It
is an inspection packet, not a proof of unconditional order or runtime success.
"""

from __future__ import annotations
import ast
import hashlib
from .comprehensiveness import FactAtom
from .static_bindings import StaticBindings

VERSION = "function-operation-evidence-1.0.0"
# Only names with explicit syntax-level templates are emitted. An attribute name
# does not establish its receiver's library/runtime type.
OPERATIONS = {
    "close": ("resource_cleanup", "close"),
    "terminate": ("process_execution", "terminate"),
    "kill": ("process_execution", "kill"),
    "wait": ("process_execution", "wait"),
    "Popen": ("process_execution", "create a process using the named Popen call"),
    "run": ("process_execution", "invoke the named run call"),
    "create_connection": ("network_request", "invoke the named create_connection call"),
    "read_text": ("file_read", "read text using the named read_text method"),
    "write_text": ("file_write", "write text using the named write_text method"),
    "read_bytes": ("file_read", "read bytes using the named read_bytes method"),
    "write_bytes": ("file_write", "write bytes using the named write_bytes method"),
    "unlink": ("file_delete", "invoke the named unlink method"),
    "wait_for_load_state": ("browser_interaction", "wait for a browser load state"),
    "goto": ("browser_interaction", "invoke the named goto method"),
    "launch": ("browser_interaction", "invoke the named launch method"),
}


def build_function_evidence(files, lineages):
    index = StaticBindings(files)
    out = []
    records = []
    by_file = {
        o.source_file: lineage for lineage in lineages for o in lineage.observations
    }
    for module, entry in index.index.items():
        parent = index.parents[module]

        def owner(node):
            while node in parent:
                node = parent[node]
                if isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
                ):
                    return node
            return None

        for fn in ast.walk(entry.root.node):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = [
                n
                for n in ast.walk(fn)
                if isinstance(n, ast.Call)
                and owner(n) is fn
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in OPERATIONS
            ]
            if not calls:
                continue
            assignments = {}
            for n in ast.walk(fn):
                if owner(n) is fn and isinstance(n, (ast.Assign, ast.AnnAssign)):
                    for target in (
                        n.targets if isinstance(n, ast.Assign) else [n.target]
                    ):
                        if isinstance(target, ast.Name):
                            assignments.setdefault(target.id, []).append(n)
            rows = []
            caps = set()
            entities = set()
            for call in sorted(calls, key=lambda x: (x.lineno, x.col_offset)):
                capability, verb = OPERATIONS[call.func.attr]
                caps.add(capability)
                receiver = ast.unparse(call.func.value)
                entities.add(receiver)
                conditions = index.context(module, call)
                expression = ast.unparse(call)
                # Expression text is an exact deterministic syntax observation,
                # not code sent for model-generated interpretation.
                text = (
                    f"Function {fn.name} contains the operation {verb}: {expression}."
                )
                if conditions:
                    text += (
                        " Its enclosing control scopes are "
                        + "; ".join(
                            str({k: v for k, v in c.items() if k != "line"})
                            for c in conditions
                        )
                        + "."
                    )
                else:
                    text += " It is in the function body; this does not prove the function executes."
                if isinstance(call.func.value, ast.Name):
                    definitions = assignments.get(receiver, [])
                    if len(definitions) == 1:
                        definition = definitions[0]
                        text += f" Receiver name {receiver} has one assignment in this function, to {ast.unparse(definition.value)}."
                        if index.context(module, definition):
                            text += " That assignment is conditional; runtime object identity is not established across paths."
                    elif len(definitions) > 1:
                        text += f" Receiver name {receiver} is reassigned; its runtime object identity is unresolved."
                    else:
                        text += f" Receiver name {receiver} has no unique local assignment; runtime type and object identity are unresolved."
                rows.append(
                    dict(
                        text=text,
                        line=call.lineno,
                        conditions=conditions,
                        receiver=receiver,
                    )
                )
            ls = tuple(range(fn.lineno, fn.end_lineno + 1))
            source = index.sources[module]
            digest = hashlib.sha256(index.paths[module].read_bytes()).hexdigest()
            lineage = by_file.get(module)
            call_spans = {tuple(range(c.lineno, c.end_lineno + 1)) for c in calls}
            observations = (
                tuple(
                    o.id
                    for o in lineage.observations
                    if tuple(o.source_lines) in call_spans and o.syntax_kind == "Call"
                )
                if lineage
                else ()
            )
            identity = (
                "CF_"
                + hashlib.sha256(
                    f"{VERSION}:{module}:{fn.lineno}:{fn.col_offset}".encode()
                ).hexdigest()[:20]
            )
            excerpt = "\n".join(source.splitlines()[i - 1] for i in ls)
            span = dict(
                role="function operations",
                source_file=module,
                source_lines=list(ls),
                excerpt=excerpt,
                sha256=digest,
                observation_ids=list(observations),
                claim_ids=[],
            )
            text = (
                "Static operation evidence within one function. Individual conditions apply; presence is not unconditional execution or proof of successful behavior.\n- "
                + "\n- ".join(row["text"] for row in rows)
            )
            out.append(
                FactAtom(
                    identity,
                    text,
                    "atom",
                    source_file=module,
                    source_lines=ls,
                    excerpt=excerpt,
                    source_spans=(span,),
                    component="code",
                    semantic_role="implementation_bundle",
                    grounding_status="grounded",
                    observation_ids=observations,
                    capability_types=tuple(sorted(caps)),
                    entities=tuple(sorted(entities)),
                    granularity="implementation_bundle",
                    assembly_rule="same lexical function; each operation retains its own control conditions and receiver binding",
                    content_form="connected_function_bundle",
                    unit_id=identity,
                )
            )
            records.append(
                dict(
                    id=identity,
                    source_file=module,
                    function=fn.name,
                    operations=rows,
                    condition_or_identity_proofs=False,
                )
            )
    return tuple(out), tuple(records)
