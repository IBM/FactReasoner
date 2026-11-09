"""Recursive scheduling of the existing Mellea NLI and native FR2 pipeline.

Selection is frozen before NLI. The tree controls when pairs are visited; it
neither generates labels nor turns sibling statements into a joint premise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Sequence

from .analysis_views import AnalysisView
from .code_structure import build_code_structure
from .comprehensiveness import RelationPair
from .documentation_tree import build_documentation_tree
from .factreasoner_fr2 import _ambiguity_judgments, run_factreasoner_v2
from .factreasoner_prompt_adapter import (
    build_project_nli_extractor,
    extractor_identity,
    project_nli_cache_identity,
    prompt_for_direction,
    serialize_node_for_nli,
)
from .recursive_traversal import (
    Statement,
    TraversalPlan,
    plan_from_structure,
    run_traversal_async,
    validate_plan,
)
from .relation_policy import apply_relation_policy
from .traversal import Judgment

VERSION = "recursive-mellea-fr2-0.3.0"
SELECTION_POLICY = "frozen-view-pairs-in-structural-scopes-1.0.0"


def _direction(direction: str) -> str:
    if direction == "documentation_to_code":
        return "forward"
    if direction == "code_to_documentation":
        return "reverse"
    raise ValueError("Unknown traversal direction")


def _prompt(direction: str):
    return prompt_for_direction(
        "documentation_to_code"
        if _direction(direction) == "forward"
        else "code_to_requirement"
    )


def _partitions(view: AnalysisView, direction: str):
    if _direction(direction) == "forward":
        return view.atoms, view.forward_contexts, view.forward_fr2_pairs
    return view.reverse_contexts, view.comprehensiveness_atoms, view.reverse_fr2_pairs


@dataclass(frozen=True)
class TraversedView(AnalysisView):
    """Ordinary FR2 roles restricted to pairs actually judged during traversal."""

    traversed_forward: tuple[RelationPair, ...] = ()
    traversed_reverse: tuple[RelationPair, ...] = ()

    @property
    def forward_fr2_pairs(self):
        return self.traversed_forward

    @property
    def reverse_fr2_pairs(self):
        return self.traversed_reverse


def _traversed_view(view, pairs, direction):
    data = {field.name: getattr(view, field.name) for field in fields(AnalysisView)}
    data["pairing_mode"] = (
        "adaptive"  # Unvisited targets remain in the project denominator.
    )
    return TraversedView(**data, **{"traversed_" + direction: tuple(pairs)})


def prepare_traversal_plan(
    view: AnalysisView,
    *,
    direction: str,
    specification: Path,
    code_files: Sequence[tuple[Path, str]],
    sources: dict[str, Path],
) -> TraversalPlan:
    """Freeze existing adaptive pairs into canonical tree packets; no new bundling.

    `sources` resolves the exact source names already stored in the AnalysisView.
    Ambiguous aggregate source locations fail rather than being silently guessed.
    Existing compound requirements and their member IDs remain separate targets.
    """
    targets, evidence, pairs = _partitions(view, direction)
    nodes = (*targets, *evidence)
    if len({n.id for n in nodes}) != len(nodes):
        raise ValueError("Directional node IDs must be unique across both partitions")

    def statement(node):
        if node.source_file not in sources or not node.source_lines:
            raise ValueError(f"Exact source file and lines are required for {node.id}")
        raw = sources[node.source_file].read_bytes()
        lines = raw.decode().splitlines()
        if any(n < 1 or n > len(lines) for n in node.source_lines):
            raise ValueError(f"Source lines outside pinned file for {node.id}")
        for span in node.source_spans:
            path = sources.get(span["source_file"])
            if (
                path is None
                or hashlib.sha256(path.read_bytes()).hexdigest() != span["sha256"]
            ):
                raise ValueError("Related source hash changed or is unavailable")
            related_lines = path.read_text().splitlines()
            if any(n < 1 or n > len(related_lines) for n in span["source_lines"]):
                raise ValueError("Related source lines outside pinned file")
            if (
                "\n".join(related_lines[n - 1] for n in span["source_lines"])
                != span["excerpt"]
            ):
                raise ValueError("Related source excerpt does not match pinned file")
        return Statement(
            node.id,
            node.text,
            node.source_file,
            node.source_lines,
            "\n".join(lines[n - 1] for n in node.source_lines),
            hashlib.sha256(raw).hexdigest(),
            node.source_spans,
            node.scope_binding,
        )

    frozen_targets = tuple(statement(n) for n in targets)
    frozen_evidence = tuple(statement(n) for n in evidence)
    if direction == "code_to_documentation":
        structure = build_documentation_tree(
            specification.read_text(),
            [dict(id=n.id, source_lines=n.source_lines) for n in targets],
        )
        loose = structure.unowned_claim_ids
    else:
        structure = build_code_structure(
            code_files,
            [
                dict(id=n.id, source_file=n.source_file, source_lines=n.source_lines)
                for n in targets
            ],
        )
        loose = structure.unassigned_atom_ids
    empty = {n.id: () for n in structure.nodes}
    if loose:
        empty["__unowned__"] = ()
    plan = plan_from_structure(
        structure,
        targets=frozen_targets,
        evidence=frozen_evidence,
        packets=empty,
        comparator_id=project_nli_cache_identity(_prompt(direction)),
    )
    # A packet is the union of eligible evidence for visible targets. Per-target
    # eligibility still applies inside the comparator; the union is not a bundle.
    scopes = tuple(
        replace(
            s,
            evidence_ids=tuple(
                dict.fromkeys(
                    p.premise_id for p in pairs if p.hypothesis_id in s.visible_ids
                )
            ),
        )
        for s in plan.scopes
    )
    return replace(plan, scopes=scopes)


def verify_plan_view(plan: TraversalPlan, view: AnalysisView) -> None:
    validate_plan(plan)
    targets, evidence, _ = _partitions(view, plan.direction)
    for supplied, nodes in [(plan.targets, targets), (plan.evidence, evidence)]:
        by_id = {n.id: n for n in nodes}
        if set(by_id) != {s.id for s in supplied}:
            raise ValueError("Traversal must preserve every frozen directional node")
        for s in supplied:
            node = by_id[s.id]
            if (
                s.text,
                s.source_file,
                s.source_lines,
                s.source_spans,
                s.scope_binding,
            ) != (
                node.text,
                node.source_file,
                node.source_lines,
                node.source_spans,
                node.scope_binding,
            ):
                raise ValueError(
                    "Traversal statements differ from the FactReasoner view"
                )
    if plan.comparator_id != project_nli_cache_identity(_prompt(plan.direction)):
        raise ValueError(
            "Traversal comparator identity differs from the current directional prompt"
        )


async def run_recursive_factreasoner(
    plan: TraversalPlan,
    view: AnalysisView,
    *,
    backend: object,
    merlin_path: str | Path,
    nli_cache_dir: str | Path,
    nli_method: str = "logprobs",
    batch_size: int = 16,
    max_judgments: int = 1000,
    extractor: object | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Judge visited pairs once and reuse their official cache in normal native FR2.

    `extractor` is a test seam; production uses build_project_nli_extractor, whose
    requests run through Mellea. The fresh cache is written only for valid parsed
    results. Fallbacks are recorded in the ledger and never treated as neutral.
    """
    verify_plan_view(plan, view)
    targets, evidence, pairs = _partitions(view, plan.direction)
    if batch_size < 1 or max_judgments < 1:
        raise ValueError("Budgets and batch size must be positive")
    if len(pairs) > max_judgments:
        raise ValueError(
            f"Frozen pair plan ({len(pairs)}) exceeds ceiling {max_judgments}; not truncated"
        )
    cache_path = Path(nli_cache_dir)
    if cache_path.exists():
        raise FileExistsError("Each traversal requires a fresh NLI cache directory")
    from fact_reasoner.core.nli_cache import NLIVerdictCache

    prompt = _prompt(plan.direction)
    nli = extractor or build_project_nli_extractor(
        backend, prompt=prompt, nli_method=nli_method
    )
    model, method = extractor_identity(nli)
    cache = NLIVerdictCache(
        str(cache_path), prompt_version=project_nli_cache_identity(prompt)
    )
    direction = _direction(plan.direction)
    nodes = {n.id: n for n in (*targets, *evidence)}
    texts = {key: serialize_node_for_nli(n) for key, n in nodes.items()}
    pair_map = {(p.premise_id, p.hypothesis_id): p for p in pairs}
    if len(pair_map) != len(pairs):
        raise ValueError("Frozen pair plan contains duplicate identities")
    records = {}
    text_results = {}
    nli_requests = 0

    def emit(stage):
        if progress is not None:
            progress(
                dict(
                    stage=stage,
                    nli_requests=nli_requests,
                    ledger=[
                        {
                            **r,
                            "judgment": asdict(r["judgment"])
                            if r["judgment"] is not None
                            else None,
                        }
                        for r in records.values()
                    ],
                )
            )

    def convert(pair, raw):
        if raw.get("_factreasoner_fallback") is True:
            return None, None, "fallback"
        if (
            raw.get("_factreasoner_fallback") is not False
            or raw.get("_nli_score_status") == "invalid"
        ):
            raise ValueError("A valid explicit parser disposition is required")
        parsed = _ambiguity_judgments((pair,), (raw,), ambiguity_enabled=True)[0]
        expected_label = "neutral" if parsed.is_withheld else parsed.relation
        expected_weight = 1.0 if parsed.is_withheld else parsed.discrete_weight
        if (
            raw.get("label") != expected_label
            or raw.get("probability") != expected_weight
        ):
            raise ValueError(
                "Native cache value does not match the existing ambiguity/discrete policy"
            )
        relation = parsed.to_fact_relation()
        guarded = (
            None
            if relation is None
            else apply_relation_policy(
                tuple(nodes.values()), (relation,), direction=direction
            ).relations[0]
        )
        judgment = Judgment(
            "NLI:"
            + hashlib.sha256(
                json.dumps(
                    [
                        plan.comparator_id,
                        model,
                        method,
                        pair.premise_id,
                        pair.hypothesis_id,
                    ],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            plan.direction,
            pair.hypothesis_id,
            (pair.premise_id,),
            parsed.relation if parsed.is_withheld else guarded.label,
            parsed.basis,
            "withheld" if parsed.is_withheld else "accepted",
        )
        return (
            judgment,
            parsed.to_dict(),
            "withheld" if parsed.is_withheld else "accepted",
        )

    async def compare(request):
        nonlocal nli_requests
        selected = [
            p
            for p in pairs
            if p.hypothesis_id in {s.id for s in request.targets}
            and p.premise_id in {s.id for s in request.evidence}
        ]
        missing = [
            p for p in selected if (p.premise_id, p.hypothesis_id) not in records
        ]
        grouped = {}
        for pair in missing:
            key = (texts[pair.premise_id], texts[pair.hypothesis_id])
            grouped.setdefault(key, []).append(pair)
        fresh = [key for key in grouped if key not in text_results]
        for start in range(0, len(fresh), batch_size):
            keys = fresh[start : start + batch_size]
            nli_requests += len(keys)
            emit("nli")
            try:
                batch = await nli.run_batch([k[0] for k in keys], [k[1] for k in keys])
                if len(batch) != len(keys):
                    raise ValueError("NLI result count does not match the frozen batch")
            except Exception as exc:
                batch = [
                    {"_factreasoner_fallback": True, "error_type": type(exc).__name__}
                    for _ in keys
                ]
            for key, raw in zip(keys, batch, strict=True):
                # Validate all semantic fields before caching or admitting an edge.
                pair = grouped[key][0]
                try:
                    convert(pair, raw)
                except (
                    ValueError,
                    KeyError,
                    TypeError,
                    IndexError,
                    AttributeError,
                ) as exc:
                    raw = {
                        "_factreasoner_fallback": True,
                        "error_type": type(exc).__name__,
                    }
                if raw.get("_factreasoner_fallback") is not True:
                    text_results[key] = dict(raw)
                    cache.put_many([(cache.make_key(model, method, *key), dict(raw))])
                for occurrence in grouped[key]:
                    records[(occurrence.premise_id, occurrence.hypothesis_id)] = (
                        make_record(occurrence, raw)
                    )
        for key, aliases in grouped.items():
            for pair in aliases:
                identity = (pair.premise_id, pair.hypothesis_id)
                if identity not in records:
                    records[identity] = make_record(pair, text_results[key])
        judgments = []
        for p in selected:
            row = records[(p.premise_id, p.hypothesis_id)]
            if request.scope_id not in row["scope_ids"]:
                row["scope_ids"].append(request.scope_id)
            if row["judgment"] is not None:
                judgments.append(row["judgment"])
        emit("compared")
        return judgments

    def make_record(pair, raw):
        judgment, ambiguity, status = convert(pair, raw)
        return dict(
            pair=pair.to_dict(),
            raw_result=dict(raw),
            ambiguity=ambiguity,
            status=status,
            judgment=judgment,
            scope_ids=[],
        )

    traversal = await run_traversal_async(plan, compare)
    completed = tuple(
        pair_map[k] for k, r in records.items() if r["status"] != "fallback"
    )
    native = None
    if completed:
        emit("factreasoner")
        native = await run_factreasoner_v2(
            _traversed_view(view, completed, direction),
            direction=direction,
            backend=backend,
            merlin_path=merlin_path,
            nli_method=nli_method,
            nli_cache_dir=cache_path,
            nli_batch_size=batch_size,
            prompt=prompt,
            cache_only=True,
        )
    ledger = [
        {**r, "judgment": asdict(r["judgment"]) if r["judgment"] is not None else None}
        for r in records.values()
    ]
    return dict(
        version=VERSION,
        direction=plan.direction,
        selection_policy=SELECTION_POLICY,
        input_pairing_mode=view.pairing_mode,
        traversal=traversal.to_dict(),
        nli_ledger=ledger,
        factreasoner=None if native is None else native.to_dict(),
        native_status="complete" if native else "no_valid_assessed_pairs",
        prompt_provenance=prompt.provenance(model=model, nli_method=method),
        counts=dict(
            planned_pairs=len(pairs),
            visited_pairs=len(records),
            unvisited_pairs=len(pairs) - len(records),
            nli_requests=nli_requests,
            valid_pairs=len(completed),
            fallback_pairs=sum(r["status"] == "fallback" for r in records.values()),
            withheld_pairs=sum(r["status"] == "withheld" for r in records.values()),
            factreasoner_additional_nli_calls=0,
        ),
        interpretation="Traversal uses guarded categorical relations; native FR2 remains the usual discrete-weight diagnostic over raw accepted model relations.",
    )
