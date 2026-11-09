"""Offline, injectable controller for the directional traversal contract.

Plans freeze targets, source evidence, and candidate packets before comparison.
Containment is scheduling only; the comparator must supply every semantic edge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from time import perf_counter
from typing import Awaitable, Callable, MutableMapping, Sequence

from .code_structure import CodeStructure
from .documentation_tree import DocumentationTree
from .traversal import (
    ChildScope,
    Direction,
    Judgment,
    StepAssessment,
    StepScope,
    assess_step,
    summarize_obligations,
)

VERSION = "recursive-traversal-0.4.0"


@dataclass(frozen=True)
class Statement:
    """Frozen normalized text with its original source provenance."""

    id: str
    text: str
    source_file: str
    source_lines: tuple[int, ...]
    source_excerpt: str
    source_sha256: str
    source_spans: tuple[dict, ...] = ()
    scope_binding: dict | None = None


@dataclass(frozen=True)
class TraversalPlan:
    direction: Direction
    scopes: tuple[StepScope, ...]
    root_ids: tuple[str, ...]
    targets: tuple[Statement, ...]
    evidence: tuple[Statement, ...]
    comparator_id: str

    def to_dict(self) -> dict:
        return {"runner_version": VERSION, **asdict(self)}


@dataclass(frozen=True)
class ComparisonRequest:
    scope_id: str
    direction: Direction
    targets: tuple[Statement, ...]
    evidence: tuple[Statement, ...]
    comparator_id: str

    @property
    def key(self) -> str:
        return _hash({"runner_version": VERSION, **asdict(self)})


Comparator = Callable[[ComparisonRequest], Sequence[Judgment]]
Cache = MutableMapping[str, tuple[Judgment, ...]]


@dataclass(frozen=True)
class TraversalVisit:
    path: tuple[str, ...]
    scope: StepScope
    comparison_key: str
    comparison_status: str
    error_type: str | None
    elapsed_seconds: float
    local: StepAssessment
    decision: StepAssessment
    aggregate: StepAssessment


@dataclass(frozen=True)
class PrunedScope:
    scope_id: str
    stopped_at: str
    reason: str


@dataclass(frozen=True)
class TraversalResult:
    plan_sha256: str
    visits: tuple[TraversalVisit, ...]
    pruned: tuple[PrunedScope, ...]
    summary: dict
    metrics: dict

    def to_dict(self) -> dict:
        return {"runner_version": VERSION, **asdict(self)}


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _index(items: Sequence, name: str) -> dict:
    result = {}
    for item in items:
        if not isinstance(item.id, str) or not item.id or item.id in result:
            raise ValueError(f"{name} require unique nonempty IDs")
        result[item.id] = item
    return result


def _statements(items: Sequence[Statement], name: str) -> dict[str, Statement]:
    result = _index(items, name)
    for item in items:
        if (
            not item.text.strip()
            or not item.source_file
            or not item.source_excerpt.strip()
        ):
            raise ValueError(f"{name} require text and explicit source provenance")
        if (
            not item.source_lines
            or len(set(item.source_lines)) != len(item.source_lines)
            or any(type(line) is not int or line < 1 for line in item.source_lines)
        ):
            raise ValueError(f"{name} require unique positive source lines")
        if len(item.source_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in item.source_sha256
        ):
            raise ValueError(f"{name} require SHA-256 source hashes")
        for span in item.source_spans:
            _statements(
                (
                    Statement(
                        item.id,
                        item.text,
                        span["source_file"],
                        tuple(span["source_lines"]),
                        span["excerpt"],
                        span["sha256"],
                    ),
                ),
                "Evidence spans",
            )
    return result


def validate_plan(plan: TraversalPlan) -> None:
    """Validate the entire forest before any comparator call, including pruned nodes."""
    if not plan.comparator_id.strip():
        raise ValueError("A versioned comparator identity is required")
    targets = _statements(plan.targets, "Targets")
    evidence = _statements(plan.evidence, "Evidence")
    scopes = _index(plan.scopes, "Scopes")
    if (
        len(set(plan.root_ids)) != len(plan.root_ids)
        or not set(plan.root_ids) <= scopes.keys()
    ):
        raise ValueError("Roots must identify distinct declared scopes")
    # Even an empty forest must declare a valid direction.
    summarize_obligations(plan.direction, tuple(targets), ())
    parents = {}
    for scope in plan.scopes:
        assess_step(scope, ())
        if scope.direction != plan.direction:
            raise ValueError("All scopes must retain the plan direction")
        if (
            not set(scope.required_ids) <= targets.keys()
            or not set(scope.evidence_ids) <= evidence.keys()
        ):
            raise ValueError("Scope refers to undeclared target or evidence statements")
        for child in scope.children:
            if child.id not in scopes or set(child.required_ids) != set(
                scopes[child.id].required_ids
            ):
                raise ValueError("Child contract must match its declared scope")
            if child.id in parents:
                raise ValueError("A structural scope must have exactly one parent")
            parents[child.id] = scope.id
    if set(plan.root_ids) != set(scopes) - set(parents):
        raise ValueError("Roots must contain every parentless scope")
    visited = set()
    stack = list(plan.root_ids)
    while stack:
        identifier = stack.pop()
        if identifier in visited:
            raise ValueError("Traversal scopes must form an acyclic forest")
        visited.add(identifier)
        stack.extend(child.id for child in scopes[identifier].children)
    if visited != set(scopes):
        raise ValueError("Disconnected or cyclic scopes cannot be omitted")
    root_members = []
    for identifier in plan.root_ids:
        root_members.extend(scopes[identifier].required_ids)
    if len(root_members) != len(set(root_members)) or set(root_members) != set(targets):
        raise ValueError(
            "Roots must partition every frozen target, including unowned obligations"
        )


def _merge(*groups: Sequence[Judgment]) -> tuple[Judgment, ...]:
    result = {}
    for group in groups:
        for judgment in group:
            if judgment.id in result and result[judgment.id] != judgment:
                raise ValueError(
                    "Conflicting judgment identity across traversal levels"
                )
            result[judgment.id] = judgment
    return tuple(result.values())


def _accumulated(scope: StepScope, judgments: Sequence[Judgment]) -> StepAssessment:
    """Reduce already validated witnesses; never pretend they came from this packet."""
    applicable = tuple(j for j in judgments if j.target_id in scope.required_ids)
    visible = tuple(sorted(set(scope.visible_ids) | {j.target_id for j in applicable}))
    evidence = tuple(
        sorted(
            set(scope.evidence_ids) | {e for j in applicable for e in j.evidence_ids}
        )
    )
    return assess_step(
        replace(scope, visible_ids=visible, evidence_ids=evidence), applicable
    )


def _traversal_events(plan: TraversalPlan, *, cache: Cache | None = None):
    """Visit pending children in declared order, then aggregate exact obligations.

    A decision includes ancestor witnesses, so descent cannot erase conflicts.
    Child results do not cancel already scheduled siblings. The final aggregate
    describes observed evidence; it is not an instruction to start another pass.
    Operational comparator failures remain explicit, are not cached, and permit
    descent. Invalid judgments or plans fail closed instead of becoming neutral.
    """
    validate_plan(plan)
    scopes = {s.id: s for s in plan.scopes}
    targets = {s.id: s for s in plan.targets}
    evidence = {s.id: s for s in plan.evidence}
    visits: list[TraversalVisit | None] = []
    pruned = []
    ledger: tuple[Judgment, ...] = ()
    calls = hits = failures = 0

    def prune(identifier: str, stopped_at: str, reason: str) -> None:
        pruned.append(PrunedScope(identifier, stopped_at, reason))
        for child in scopes[identifier].children:
            prune(child.id, stopped_at, reason)

    def visit(
        identifier: str, path: tuple[str, ...], inherited: tuple[Judgment, ...]
    ) -> tuple[Judgment, ...]:
        nonlocal ledger, calls, hits, failures
        scope = scopes[identifier]
        path = (*path, identifier)
        index = len(visits)
        visits.append(None)
        request = ComparisonRequest(
            identifier,
            plan.direction,
            tuple(targets[t] for t in scope.visible_ids),
            tuple(evidence[e] for e in scope.evidence_ids),
            plan.comparator_id,
        )
        key = request.key
        judgments = ()
        error_type = None
        started = perf_counter()
        if not scope.required_ids:
            status = "empty_scope"
        elif not scope.visible_ids:
            status = "no_visible_targets"
        elif not scope.evidence_ids:
            status = "no_candidate_evidence"
        elif cache is not None and key in cache:
            judgments = tuple(cache[key])
            status = "cache_hit"
            hits += 1
        else:
            calls += 1
            response = yield request
            if isinstance(response, _ComparisonFailure):
                error_type = response.error_type
                status = "failed"
                failures += 1
            else:
                judgments = tuple(response)
                status = "compared"
        elapsed = perf_counter() - started
        local = assess_step(scope, judgments)
        # Validate identity globally before persisting a successful cache entry.
        ledger = _merge(ledger, local.judgments)
        if status == "compared" and cache is not None:
            cache[key] = local.judgments
        cumulative = _merge(inherited, local.judgments)
        decision = _accumulated(scope, cumulative)
        subtree = local.judgments
        # Freeze the schedule at this decision: a later contradicted child does
        # not silently suppress evaluation of a sibling that was pending here.
        for child in scope.children:
            if child.id in decision.next_child_ids:
                subtree = _merge(
                    subtree, (yield from visit(child.id, path, cumulative))
                )
            else:
                reason = (
                    f"{decision.state}_scope"
                    if decision.action in {"stop", "skip"}
                    else "obligations_already_settled"
                )
                prune(child.id, identifier, reason)
        aggregate = _accumulated(scope, _merge(inherited, subtree))
        visits[index] = TraversalVisit(
            path, scope, key, status, error_type, elapsed, local, decision, aggregate
        )
        return subtree

    for root in plan.root_ids:
        yield from visit(root, (), ())
    completed = tuple(visits)
    summary = summarize_obligations(
        plan.direction, tuple(targets), [v.local for v in completed]
    )
    unresolved = set(summary["unresolved_ids"])
    summary["terminal_unresolved_ids"] = sorted(
        unresolved & {t for v in completed for t in v.aggregate.terminal_pending_ids}
    )
    summary["pruned_unresolved_ids"] = sorted(
        unresolved & {t for p in pruned for t in scopes[p.scope_id].required_ids}
    )
    metrics = dict(
        planned_scopes=len(scopes),
        visited_scopes=len(completed),
        pruned_scopes=len(pruned),
        comparator_calls=calls,
        cache_hits=hits,
        failed_comparisons=failures,
        unique_judgments=len(ledger),
        unique_witness_ids=len({e for j in ledger for e in j.evidence_ids}),
    )
    return TraversalResult(
        _hash(plan.to_dict()), completed, tuple(pruned), summary, metrics
    )


@dataclass(frozen=True)
class _ComparisonFailure:
    error_type: str


def run_traversal(
    plan: TraversalPlan, comparator: Comparator, *, cache: Cache | None = None
) -> TraversalResult:
    """Run the shared controller with a synchronous, offline comparator."""
    events = _traversal_events(plan, cache=cache)
    response = None
    while True:
        try:
            request = events.send(response)
        except StopIteration as done:
            return done.value
        try:
            response = tuple(comparator(request))
        except Exception as exc:
            response = _ComparisonFailure(type(exc).__name__)


async def run_traversal_async(
    plan: TraversalPlan,
    comparator: Callable[[ComparisonRequest], Awaitable[Sequence[Judgment]]],
    *,
    cache: Cache | None = None,
) -> TraversalResult:
    """Use the identical controller while keeping every provider call on this loop."""
    events = _traversal_events(plan, cache=cache)
    response = None
    try:
        while True:
            try:
                request = events.send(response)
            except StopIteration as done:
                return done.value
            try:
                response = tuple(await comparator(request))
            except Exception as exc:
                response = _ComparisonFailure(type(exc).__name__)
    finally:
        events.close()


def plan_from_structure(
    structure: DocumentationTree | CodeStructure,
    *,
    targets: Sequence[Statement],
    evidence: Sequence[Statement],
    packets: dict[str, tuple[str, ...]],
    comparator_id: str,
) -> TraversalPlan:
    """Expose own and immediate-child obligations; require explicit candidate packets.

    This adapter schedules one target tree per direction. Cross-tree retrieval is
    deliberately injected as frozen packets, not inferred from common containment.
    Unowned targets receive a real terminal scope and remain in the denominator.
    """
    if isinstance(structure, DocumentationTree):
        direction: Direction = "code_to_documentation"
        own = {n.id: n.own_claim_ids for n in structure.nodes}
        members = {n.id: n.subtree_claim_ids for n in structure.nodes}
        loose = structure.unowned_claim_ids
    elif isinstance(structure, CodeStructure):
        direction = "documentation_to_code"
        own = {n.id: n.own_atom_ids for n in structure.nodes}
        members = {n.id: n.subtree_atom_ids for n in structure.nodes}
        loose = structure.unassigned_atom_ids
    else:
        raise TypeError("Expected a canonical documentation or code structure")
    loose_id = "__unowned__"
    nodes = _index(structure.nodes, "Structural nodes")
    scope_ids = set(nodes) | ({loose_id} if loose else set())
    if loose_id in nodes or set(packets) != scope_ids:
        raise ValueError(
            "Provide an explicit packet (possibly empty) for every structural scope"
        )
    scopes = []
    roots = []
    for node in structure.nodes:
        children = tuple(ChildScope(c, members[c]) for c in node.children)
        visible = tuple(
            dict.fromkeys((*own[node.id], *(t for c in node.children for t in own[c])))
        )
        scopes.append(
            StepScope(
                node.id,
                direction,
                members[node.id],
                visible,
                packets[node.id],
                children,
            )
        )
        if node.parent_id is None:
            roots.append(node.id)
    if loose:
        scopes.append(StepScope(loose_id, direction, loose, loose, packets[loose_id]))
        roots.append(loose_id)
    plan = TraversalPlan(
        direction,
        tuple(scopes),
        tuple(roots),
        tuple(targets),
        tuple(evidence),
        comparator_id,
    )
    validate_plan(plan)
    return plan
