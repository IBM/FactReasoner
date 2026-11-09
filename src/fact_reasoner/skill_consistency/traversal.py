"""Offline contract for a directional traversal step; no model or solver calls.

The caller supplies exact obligations, exposed statements, candidate evidence and
accepted/withheld directional judgments. This module validates and reduces that
ledger; it never infers meaning from containment or reverses entailment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence

VERSION = "directional-traversal-contract-0.1.0"
Direction = Literal["code_to_documentation", "documentation_to_code"]
State = Literal["complete", "partial", "contradicted", "unresolved"]


def _unique(values: Sequence[str], name: str) -> set[str]:
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} requires nonempty string IDs")
    result = set(values)
    if len(result) != len(values):
        raise ValueError(f"{name} contains duplicate IDs")
    return result


@dataclass(frozen=True)
class ChildScope:
    id: str
    required_ids: tuple[str, ...]


@dataclass(frozen=True)
class StepScope:
    id: str
    direction: Direction
    required_ids: tuple[str, ...]
    visible_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    children: tuple[ChildScope, ...] = ()


@dataclass(frozen=True)
class Judgment:
    id: str
    direction: Direction
    target_id: str
    evidence_ids: tuple[str, ...]
    relation: Literal["entailment", "contradiction", "neutral"]
    basis: str
    status: Literal["accepted", "withheld"] = "accepted"


@dataclass(frozen=True)
class StepAssessment:
    scope_id: str
    direction: Direction
    state: State
    supported_ids: tuple[str, ...]
    contradicted_ids: tuple[str, ...]
    mixed_ids: tuple[str, ...]
    pending_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    action: Literal["stop", "descend", "terminal", "skip"]
    next_child_ids: tuple[str, ...]
    terminal_pending_ids: tuple[str, ...]
    judgments: tuple[Judgment, ...]

    def to_dict(self) -> dict:
        return {"contract_version": VERSION, **asdict(self)}


def assess_step(scope: StepScope, judgments: Sequence[Judgment]) -> StepAssessment:
    """Reduce a conjunction of exact obligations under one evidence packet.

    Compound obligations retain a parent contract ID *as well as* member IDs.
    All must be supported for complete: isolated members do not establish a
    required ordering, condition, or object binding. Every judgment addresses one
    explicit target; a single future model response can contain many judgments.
    """
    if scope.direction not in {"code_to_documentation", "documentation_to_code"}:
        raise ValueError("A traversal must declare its direction")
    required = _unique(scope.required_ids, "required_ids")
    visible = _unique(scope.visible_ids, "visible_ids")
    evidence = _unique(scope.evidence_ids, "evidence_ids")
    if not visible <= required:
        raise ValueError("Visible targets must belong to the exact required scope")
    _unique(tuple(c.id for c in scope.children), "child IDs")
    child_members: set[str] = set()
    for child in scope.children:
        members = _unique(child.required_ids, "child required_ids")
        if not members <= required or members & child_members:
            raise ValueError(
                "Child scopes must be disjoint subsets of the parent scope"
            )
        child_members.update(members)
    ledger: dict[str, Judgment] = {}
    for judgment in judgments:
        if judgment.direction != scope.direction:
            raise ValueError("Judgment direction cannot be reversed or mixed")
        if judgment.target_id not in visible:
            raise ValueError("A judgment must address an exposed target in this scope")
        witnesses = _unique(judgment.evidence_ids, "judgment evidence_ids")
        if not witnesses or not witnesses <= evidence:
            raise ValueError("A judgment requires identified evidence from this packet")
        if judgment.relation not in {"entailment", "contradiction", "neutral"}:
            raise ValueError("Unknown relation")
        if judgment.status not in {"accepted", "withheld"}:
            raise ValueError("Unknown judgment status")
        if not judgment.id or not judgment.basis.strip():
            raise ValueError("Judgments require identity and an explanatory basis")
        if judgment.id in ledger and ledger[judgment.id] != judgment:
            raise ValueError("A judgment ID cannot identify different records")
        ledger[judgment.id] = judgment
    labels = {target: set() for target in required}
    withheld: set[str] = set()
    for j in ledger.values():
        if j.status == "withheld":
            withheld.add(j.target_id)
        else:
            labels[j.target_id].add(j.relation)
    mixed = {t for t, v in labels.items() if {"entailment", "contradiction"} <= v}
    supported = {t for t, v in labels.items() if "entailment" in v and t not in mixed}
    opposed = {t for t, v in labels.items() if "contradiction" in v and t not in mixed}
    pending = required - supported - opposed
    reasons = []
    if not required:
        state: State = "unresolved"
        reasons.append("empty_scope")
    elif mixed:
        state = "unresolved"
        reasons.append("mixed_evidence")
    elif opposed:
        state = "contradicted"
        reasons.append("explicit_opposition_to_required_obligation")
    elif supported == required:
        state = "complete"
    elif supported:
        state = "partial"
    else:
        state = "unresolved"
    if required - visible:
        reasons.append("unexposed_obligations")
    if any(not labels[t] and t not in withheld for t in pending):
        reasons.append("missing_judgments")
    if pending & withheld:
        reasons.append("withheld_judgments")
    if any(labels[t] == {"neutral"} for t in pending):
        reasons.append("not_evidenced_by_supplied_context")
    if not evidence and required:
        reasons.append("no_candidate_evidence")
    next_children = tuple(
        c.id for c in scope.children if pending.intersection(c.required_ids)
    )
    if not required:
        action = "skip"
        next_children = ()
    elif state in {"complete", "contradicted"}:
        action = "stop"
        next_children = ()
    else:
        action = "descend" if next_children else "terminal"
    # Even when descent continues, unresolved own obligations cannot disappear.
    terminal = pending - set().union(*(set(c.required_ids) for c in scope.children))
    return StepAssessment(
        scope.id,
        scope.direction,
        state,
        tuple(sorted(supported)),
        tuple(sorted(opposed)),
        tuple(sorted(mixed)),
        tuple(sorted(pending)),
        tuple(reasons),
        action,
        next_children,
        tuple(sorted(terminal)),
        tuple(ledger.values()),
    )


def summarize_obligations(
    direction: Direction, required_ids: Sequence[str], steps: Sequence[StepAssessment]
) -> dict:
    """Union original obligations across levels, preserving opposing witnesses.

    The caller freezes the denominator (e.g. atomic requirement members) before
    traversal. Parent contract IDs can be audited separately. Repeated visits or
    shared witnesses do not add weight. This is categorical bookkeeping only;
    it is not FactReasoner inference or a calibrated dependence correction.
    """
    if direction not in {"code_to_documentation", "documentation_to_code"}:
        raise ValueError("A summary must declare its direction")
    required = _unique(required_ids, "required_ids")
    labels = {key: set() for key in required}
    seen: dict[str, Judgment] = {}
    for step in steps:
        if step.direction != direction:
            raise ValueError("Directions require separate coverage summaries")
        for judgment in step.judgments:
            if judgment.id in seen and seen[judgment.id] != judgment:
                raise ValueError(
                    "Conflicting judgment identity across traversal levels"
                )
            seen[judgment.id] = judgment
            if judgment.status == "accepted" and judgment.target_id in required:
                labels[judgment.target_id].add(judgment.relation)
    mixed = {t for t, v in labels.items() if {"entailment", "contradiction"} <= v}
    supported = {t for t, v in labels.items() if "entailment" in v and t not in mixed}
    opposed = {t for t, v in labels.items() if "contradiction" in v and t not in mixed}
    return dict(
        contract_version=VERSION,
        direction=direction,
        denominator_ids=sorted(required),
        supported_ids=sorted(supported),
        contradicted_ids=sorted(opposed),
        mixed_ids=sorted(mixed),
        unresolved_ids=sorted(required - supported - opposed),
        supported_count=len(supported),
        denominator_count=len(required),
        judgment_ids=sorted(seen),
        interpretation="Unique categorical obligations; no probability or independence claim",
    )
