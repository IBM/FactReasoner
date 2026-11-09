"""Adaptive search over frozen documentation and implementation hierarchies.

Source statements and the allowable search space are pinned before NLI. Visited
pairs are chosen during traversal; a neutral coarse comparison never removes
unseen descendant evidence. Independent code roots and requirement scopes remain
scheduled after a contradiction. This is not proof of exhaustive semantics.
"""

from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Callable, Awaitable
from .candidate_pairs import select_candidate_relation_pairs
from .code_structure import build_code_structure
from .documentation_tree import build_documentation_tree
from .comprehensiveness import RelationPair
from .traversal import StepScope, assess_step

VERSION = "adaptive-two-tree-search-1.0.0"


@dataclass(frozen=True)
class SearchSpace:
    view: object
    documentation: object
    implementation: object
    pairs: tuple
    initial_pairs: tuple
    roots: tuple[str, ...]
    evidence_owners: dict

    def to_dict(self):
        return dict(
            version=VERSION,
            documentation=self.documentation.to_dict(),
            implementation=self.implementation.to_dict(),
            pairs=[p.to_dict() for p in self.pairs],
            initial_pairs=[p.to_dict() for p in self.initial_pairs],
            roots=list(self.roots),
            evidence_owners=self.evidence_owners,
            full_target_ids=[n.id for n in self.view.reverse_contexts],
            policy="Source and eligibility frozen; code-node visits and NLI comparisons adaptive. Whole-specification denominator; no forced fallback candidates.",
        )


def prepare_search(view, *, specification, code_files):
    targets = view.reverse_contexts
    evidence = view.comprehensiveness_atoms
    if len({n.id for n in (*targets, *evidence)}) != len(targets) + len(evidence):
        raise ValueError("Search nodes require unique IDs")
    doc = build_documentation_tree(
        specification.read_text(),
        [dict(id=n.id, source_lines=n.source_lines) for n in targets],
    )
    code = build_code_structure(
        code_files,
        [
            dict(id=n.id, source_file=n.source_file, source_lines=n.source_lines)
            for n in evidence
        ],
    )
    by_id = {n.id: n for n in evidence}
    # The same deterministic selector is applied at every structural resolution.
    # Union is an audit/budget universe; no model is called during preparation.
    initial = tuple(view.reverse_fr2_pairs)
    pairs = {(p.premise_id, p.hypothesis_id): p for p in initial}
    owners = {a: n.id for n in code.nodes for a in n.own_atom_ids}
    for node in code.nodes:
        for label, ids in [
            ("subtree", node.subtree_atom_ids),
            ("direct", node.own_atom_ids),
            ("references", node.context_atom_ids),
        ]:
            if not ids:
                continue
            selected = select_candidate_relation_pairs(
                tuple(by_id[i] for i in ids),
                targets,
                kind="atom_context",
                false_discovery_rate=view.lexical_false_discovery_rate,
            )
            for p in selected:
                key = (p.premise_id, p.hypothesis_id)
                if key not in pairs:
                    pairs[key] = replace(
                        p,
                        selection_method=VERSION,
                        selection_reasons=(
                            *p.selection_reasons,
                            "structural_" + label + ":" + node.id,
                        ),
                    )
    # A resolved literal call reference is a source-derived navigation anchor.
    # It does not supply any NLI label or new semantic relation.
    for target in targets:
        scope = target.scope_binding or {}
        calls = scope.get("candidate_call_ids", [])
        if len(calls) != 1:
            continue
        for candidate in evidence:
            if (candidate.scope_binding or {}).get("call_id") == calls[0]:
                key = (candidate.id, target.id)
                pairs.setdefault(
                    key,
                    RelationPair(
                        *key,
                        "atom_context",
                        VERSION,
                        selection_reasons=("same_uniquely_resolved_call",),
                    ),
                )
    if code.unassigned_atom_ids:
        raise ValueError("Every search evidence node must have a structural owner")
    return SearchSpace(
        view,
        doc,
        code,
        tuple(pairs[k] for k in sorted(pairs)),
        initial,
        tuple(n.id for n in code.nodes if n.parent_id is None),
        owners,
    )


def _state(targets, evidence, judgments, identifier="search"):
    scope = StepScope(
        identifier,
        "code_to_documentation",
        tuple(targets),
        tuple(targets),
        tuple(evidence),
    )
    return assess_step(scope, judgments)


async def search(
    space: SearchSpace,
    judge: Callable[[tuple], Awaitable[tuple]],
    *,
    policy="adaptive",
    progress=None,
):
    """Visit both hierarchies. Judge returns validated ordinary NLI judgments.

    Adaptive mode stops a *code branch* once its visible target group is complete
    but always continues after contradictions, other code roots and documentation
    scopes. A full-audit mode visits every eligible pair and exposes all conflicts.
    """
    if policy not in {"adaptive", "expanded_flat", "current_flat", "adaptive_audit"}:
        raise ValueError("Unknown search policy")
    targets = {n.id: n for n in space.view.reverse_contexts}
    evidence = {n.id: n for n in space.view.comprehensiveness_atoms}
    allpairs = {(p.premise_id, p.hypothesis_id): p for p in space.pairs}
    visited = set()
    judgments = {}
    trace = []
    pruned = []

    async def compare(pairs, where, reason):
        fresh = tuple(
            p for p in pairs if (p.premise_id, p.hypothesis_id) not in visited
        )
        if not fresh:
            return ()
        received = await judge(fresh)
        permitted = {(p.premise_id, p.hypothesis_id) for p in fresh}
        for j in received:
            if (
                j.direction != "code_to_documentation"
                or j.target_id not in targets
                or not j.evidence_ids
                or any((e, j.target_id) not in permitted for e in j.evidence_ids)
            ):
                raise ValueError("Judge returned an out-of-scope relation")
            if j.id in judgments and judgments[j.id] != j:
                raise ValueError("Conflicting judgment identity")
            judgments[j.id] = j
        visited.update(permitted)
        trace.append(
            dict(
                action="compare",
                scope=where,
                reason=reason,
                pairs=[p.to_dict() for p in fresh],
                judgment_ids=[j.id for j in received],
                resolved_source_chains={
                    p.premise_id: list(evidence[p.premise_id].source_spans)
                    for p in fresh
                    if evidence[p.premise_id].source_spans
                },
            )
        )
        if progress:
            progress(dict(visited_pairs=len(visited), trace=trace))
        return tuple(received)

    if policy in {"expanded_flat", "current_flat"}:
        pairs = space.pairs if policy == "expanded_flat" else space.initial_pairs
        await compare(pairs, "flat", "complete declared candidate set")
    else:
        code = {n.id: n for n in space.implementation.nodes}

        def rank(node, tids):
            selected = [
                p
                for p in space.pairs
                if p.hypothesis_id in tids and p.premise_id in node.subtree_atom_ids
            ]
            return (
                -len({p.hypothesis_id for p in selected}),
                node.module,
                node.source_lines[0],
                node.id,
            )

        async def visit_code(node_id, tids, inherited, doc_id):
            node = code[node_id]
            eligible = tuple(
                p
                for p in space.pairs
                if p.hypothesis_id in tids and p.premise_id in node.own_atom_ids
            )
            local = await compare(
                eligible,
                (doc_id, node.id),
                "inspect code node and its attached source references",
            )
            applicable = {j.id: j for j in (*inherited, *local) if j.target_id in tids}
            # Reuse previously visited witnesses without assigning new votes.
            for j in judgments.values():
                if j.target_id in tids and set(j.evidence_ids) <= set(
                    node.subtree_atom_ids
                ):
                    applicable[j.id] = j
            assessment = _state(tids, evidence, tuple(applicable.values()), node.id)
            trace.append(
                dict(
                    action="assess",
                    documentation_node=doc_id,
                    code_node=node.id,
                    state=assessment.state,
                    supported_ids=list(assessment.supported_ids),
                    pending_ids=list(assessment.pending_ids),
                    reason="source-scoped categorical assessment; independent branches remain scheduled",
                )
            )
            if assessment.state != "complete" or policy == "adaptive_audit":
                linked = tuple(
                    p
                    for p in space.pairs
                    if p.hypothesis_id in tids
                    and p.premise_id in node.context_atom_ids
                    and (p.premise_id, p.hypothesis_id) not in visited
                )
                if linked:
                    trace.append(
                        dict(
                            action="follow_source_references",
                            code_node=node.id,
                            reason=assessment.state,
                            evidence_ids=sorted({p.premise_id for p in linked}),
                            references=list(node.references),
                            unresolved_references=list(node.unresolved_references),
                        )
                    )
                    found = await compare(
                        linked,
                        (doc_id, node.id),
                        "follow uniquely resolved static source references",
                    )
                    applicable.update((j.id, j) for j in found)
                    assessment = _state(
                        tids, evidence, tuple(applicable.values()), node.id
                    )
            stop = assessment.state == "complete" and policy != "adaptive_audit"
            if stop:
                omitted = [
                    p.to_dict()
                    for p in space.pairs
                    if p.hypothesis_id in tids
                    and p.premise_id
                    in set(node.subtree_atom_ids) - set(node.own_atom_ids)
                    and (p.premise_id, p.hypothesis_id) not in visited
                ]
                if omitted:
                    pruned.append(
                        dict(
                            documentation_node=doc_id,
                            code_node=node.id,
                            reason=assessment.state,
                            pairs=omitted,
                        )
                    )
                return
            for child in sorted(node.children, key=lambda x: rank(code[x], tids)):
                trace.append(
                    dict(
                        action="descend_code",
                        from_node=node.id,
                        to_node=child,
                        reason=assessment.state,
                    )
                )
                await visit_code(child, tids, tuple(applicable.values()), doc_id)

        # Documentation ownership preserves exact compound targets and member
        # identities. A parent title is never substituted for hidden obligations.
        doc_nodes = {n.id: n for n in space.documentation.nodes}
        owned = set()

        async def visit_doc(node):
            ids = tuple(node.own_claim_ids)
            if ids:
                owned.update(ids)
                for root in sorted(space.roots, key=lambda x: rank(code[x], ids)):
                    await visit_code(root, ids, (), node.id)
            for child in node.children:
                trace.append(
                    dict(
                        action="descend_documentation", from_node=node.id, to_node=child
                    )
                )
                await visit_doc(doc_nodes[child])

        for node in space.documentation.nodes:
            if node.parent_id is None:
                await visit_doc(node)
        loose = tuple(i for i in targets if i not in owned)
        if loose:
            for root in sorted(space.roots, key=lambda x: rank(code[x], loose)):
                await visit_code(root, loose, (), "__unowned__")
    final = tuple(judgments.values())
    assessment = _state(targets, evidence, final)
    unvisited = tuple(p for key, p in allpairs.items() if key not in visited)
    states = {}
    for target in targets:
        accepted = [
            j for j in final if j.target_id == target and j.status == "accepted"
        ]
        labels = {j.relation for j in accepted}
        states[target] = (
            "mixed"
            if {"entailment", "contradiction"} <= labels
            else "entailment"
            if "entailment" in labels
            else "contradiction"
            if "contradiction" in labels
            else "neutral"
            if accepted
            else "unassessed"
        )
    return dict(
        version=VERSION,
        policy=policy,
        state=assessment.state,
        target_states=states,
        visited_pairs=[allpairs[k].to_dict() for k in sorted(visited)],
        unvisited_pairs=[p.to_dict() for p in unvisited],
        trace=trace,
        pruned=pruned,
        judgments=final,
        counts=dict(
            eligible_pairs=len(allpairs),
            visited_pairs=len(visited),
            unvisited_pairs=len(unvisited),
            unassessed_targets=sum(v == "unassessed" for v in states.values()),
            supported_targets=sum(v == "entailment" for v in states.values()),
        ),
    )
