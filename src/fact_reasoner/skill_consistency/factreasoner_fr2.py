"""Run stable project nodes through forward and reverse FactReasoner V2."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Literal, Sequence

from .analysis_views import AnalysisView
from .comprehensiveness import (
    FactAtom,
    FactRelation,
    RelationPair,
    relations_from_results,
)
from .factreasoner_prompt_adapter import (
    FORWARD_DOCUMENTATION_TO_CODE,
    REVERSE_CODE_TO_REQUIREMENT,
    REVERSE_CODE_TO_REQUIREMENT_V1_2,
    PromptSpec,
    build_project_nli_extractor,
    extractor_identity,
    prompt_for_direction,
    project_nli_cache_identity,
    serialize_node_for_nli,
)
from .relation_ambiguity import (
    AMBIGUITY_GATE_POLICY_VERSION,
    AMBIGUITY_PROTOCOL_VERSION,
    AmbiguityJudgment,
)
from .relation_policy import apply_relation_policy
from .relation_probability_proxy import (
    POLICY_VERSION as AMBIGUITY_WEIGHT_POLICY_VERSION,
)


Fr2Direction = Literal["forward", "reverse"]


@dataclass(frozen=True)
class FactReasonerV2Run:
    """One FR2 graph with stable project-facing relation orientation."""

    direction: Fr2Direction
    relations: tuple[FactRelation, ...]
    model_relations: tuple[FactRelation, ...]
    network_relations: tuple[FactRelation, ...]
    raw_score: dict[str, Any]
    marginals: tuple[dict[str, Any], ...]
    payload: dict[str, Any]
    nli_runtime: dict[str, Any]
    role_mapping: dict[str, tuple[str, ...]]
    prompt_provenance: dict[str, str]
    relation_policy: dict[str, Any]
    pairing: dict[str, Any]
    ambiguity_judgments: tuple[dict[str, Any], ...] = ()
    withheld_high_ambiguity_pairs: tuple[RelationPair, ...] = ()
    pipeline_version: str = "v2"
    nli_pair_mode: str = "adaptive"

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction,
            "pipeline_version": self.pipeline_version,
            "nli_pair_mode": self.nli_pair_mode,
            "relations": [item.to_dict() for item in self.relations],
            "model_relations": [item.to_dict() for item in self.model_relations],
            "network_relations": [item.to_dict() for item in self.network_relations],
            "raw_score": self.raw_score,
            "marginals": list(self.marginals),
            "payload": self.payload,
            "nli_runtime": self.nli_runtime,
            "prompt_provenance": dict(self.prompt_provenance),
            "relation_policy": dict(self.relation_policy),
            "ambiguity_judgments": [dict(item) for item in self.ambiguity_judgments],
            "withheld_high_ambiguity_pairs": [
                item.to_dict() for item in self.withheld_high_ambiguity_pairs
            ],
            "pairing": dict(self.pairing),
            "role_mapping": {
                key: list(value) for key, value in self.role_mapping.items()
            },
        }


def factreasoner_payload(
    view: AnalysisView, *, direction: Fr2Direction = "forward"
) -> dict[str, Any]:
    """Serialize one FR2 direction without regenerating frozen statements.

    Project kinds never change: code remains ``atom`` and documentation remains
    ``context`` for lineage and visualization.  Only the payload roles change.
    """

    declared_pairs = (
        view.forward_fr2_pairs if direction == "forward" else view.reverse_fr2_pairs
    )
    claim_nodes, evidence_nodes, unassessed_claim_ids = _selected_fr2_partitions(
        view, direction, declared_pairs
    )
    context_ids_by_claim: dict[str, list[str]] = {item.id: [] for item in claim_nodes}
    for pair in declared_pairs:
        context_ids_by_claim[pair.hypothesis_id].append(pair.premise_id)
    contexts = [
        {
            "id": item.id,
            "title": (
                f"{item.component or 'source'} evidence from "
                f"{item.source_file or item.id}"
            ),
            "text": serialize_node_for_nli(item),
            "normalized_statement": item.text,
            # FactReasoner concatenates non-empty snippets with ``text`` before
            # NLI. Keep the exact frozen context as the sole model input and
            # retain the larger source excerpt only as provenance.
            "snippet": "",
            "excerpt": item.excerpt,
            "link": "",
            "component": item.component,
            "semantic_role": item.semantic_role,
            "source_file": item.source_file,
            "source_lines": list(item.source_lines),
            "source_spans": list(item.source_spans),
            "scope_binding": item.scope_binding,
            "project_kind": item.kind,
        }
        for item in evidence_nodes
    ]
    atoms = [
        {
            "id": item.id,
            "text": serialize_node_for_nli(item),
            "original": item.text,
            "normalized_statement": item.text,
            "contexts": context_ids_by_claim[item.id],
            "component": item.component,
            "source_file": item.source_file,
            "source_lines": list(item.source_lines),
            "source_spans": list(item.source_spans),
            "scope_binding": item.scope_binding,
            "observation_ids": list(item.observation_ids),
            "claim_ids": list(item.claim_ids),
            "grounding_status": item.grounding_status,
            "project_kind": item.kind,
        }
        for item in claim_nodes
    ]
    return {
        "input": view.question,
        "output": "\n".join(item.text for item in claim_nodes),
        "topic": f"{view.id}_{direction}_fr2",
        "atoms": atoms,
        "contexts": contexts,
        "pairing": (
            view.forward_pairing_report
            if direction == "forward"
            else view.reverse_pairing_report
        ),
        "unassessed_claim_ids": list(unassessed_claim_ids),
    }


async def run_factreasoner_v2(
    view: AnalysisView,
    *,
    direction: Fr2Direction = "forward",
    backend: object,
    merlin_path: str | Path,
    nli_method: str = "logprobs",
    nli_cache_dir: str | Path | None = None,
    nli_batch_size: int = 256,
    prompt: PromptSpec | None = None,
    cache_only: bool = False,
) -> FactReasonerV2Run:
    """Run FR2 in one declared direction using frozen, provenance-rich nodes.

    Imports stay inside the function so all extraction and tests remain usable
    without FactReasoner, Merlin, Mellea, or credentials.
    """

    validate_fr2_view(view, direction=direction)

    from fact_reasoner import FactReasoner
    from fact_reasoner.core.nli_config import get_pair_config

    payload = factreasoner_payload(view, direction=direction)
    project_prompt = prompt or prompt_for_direction(
        "documentation_to_code" if direction == "forward" else "code_to_requirement"
    )
    declared_pairs = (
        view.forward_fr2_pairs if direction == "forward" else view.reverse_fr2_pairs
    )
    claim_nodes, evidence_nodes, _ = _selected_fr2_partitions(
        view, direction, declared_pairs
    )
    temporary_cache = (
        tempfile.TemporaryDirectory(prefix=f"factreasoner-v2-{direction}-nli-")
        if nli_cache_dir is None
        else None
    )
    cache_path = Path(
        nli_cache_dir if nli_cache_dir is not None else temporary_cache.name
    ).expanduser()
    try:
        extractor = build_project_nli_extractor(
            backend,
            prompt=project_prompt,
            nli_method=nli_method,
        )
        if cache_only:

            def reject_uncached(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("Cache-only FR2 cannot request a new NLI judgment")

            async def reject_uncached_batch(*args: Any, **kwargs: Any) -> Any:
                return reject_uncached(*args, **kwargs)

            extractor.run = reject_uncached
            extractor.run_batch = reject_uncached_batch
        pipeline = FactReasoner(
            nli_extractor=extractor,
            merlin_path=str(merlin_path),
            use_priors=True,
            nli_pair_config=get_pair_config("all_pairs"),
            nli_cache_dir=str(cache_path),
        )
        # The installed cache defaults to FactReasoner's stock prompt version.
        # Override the identity on this cache instance so verdicts from another
        # project prompt can never be reused accidentally.
        pipeline.nli_cache.prompt_version = project_nli_cache_identity(project_prompt)
        pipeline.from_dict_with_contexts(payload)

        # FactReasoner 0.8.0's synchronous relation builder can create a separate
        # ``asyncio.run`` loop when called by Jupyter. A
        # persistent async HTTP backend (notably RITS) then remains bound to the
        # first loop after that loop closes. Prime FactReasoner's official NLI
        # cache from this coroutine instead: every remote call stays on the one
        # persistent notebook loop, while ``pipeline.build`` performs only cache
        # reads and preserves FactReasoner's native FR2 reconciliation.
        model_text_by_id = {
            **{item.id: item.get_text() for item in pipeline.atoms.values()},
            **{item.id: item.get_text() for item in pipeline.contexts.values()},
        }
        ledger_results, prime_stats = await _run_cached_relation_pairs(
            declared_pairs,
            text_by_id=model_text_by_id,
            extractor=extractor,
            cache=pipeline.nli_cache,
            batch_size=nli_batch_size,
        )
        await pipeline.build(
            has_atoms=True,
            has_contexts=True,
            revise_atoms=False,
            summarize_contexts=False,
            # Every claim carries its declared candidate contexts in the payload.
            # Keeping this enabled prevents FactReasoner from silently restoring
            # the full Cartesian product after the project cache is primed.
            contexts_per_atom_only=True,
            rel_atom_context=True,
            rel_context_context=False,
            remove_duplicates=True,
        )
        build_totals = dict(pipeline.nli_stats.get("totals", {}))
        if build_totals.get("llm_calls", 0):
            raise RuntimeError(
                "FactReasoner missed a primed NLI verdict and attempted a "
                "second event-loop-bound backend call"
            )

        raw_score, marginals = pipeline.score()
        network_relations = tuple(
            FactRelation(
                source_id=relation.source.id,
                target_id=relation.target.id,
                label=str(relation.type).lower(),  # type: ignore[arg-type]
                probability=(
                    float(relation.probability)
                    if relation.probability is not None
                    else None
                ),
            )
            for relation in pipeline.relations
            if relation.link == "context_atom"
        )
        ambiguity_enabled = project_prompt in {
            FORWARD_DOCUMENTATION_TO_CODE,
            REVERSE_CODE_TO_REQUIREMENT,
            REVERSE_CODE_TO_REQUIREMENT_V1_2,
        }
        ambiguity_judgments = _ambiguity_judgments(
            declared_pairs,
            ledger_results,
            ambiguity_enabled=ambiguity_enabled,
        )
        model_relations = (
            tuple(
                relation
                for judgment in ambiguity_judgments
                if (relation := judgment.to_fact_relation()) is not None
            )
            if ambiguity_enabled
            else tuple(relations_from_results(declared_pairs, ledger_results))
        )
        policy_result = apply_relation_policy(
            view.forward_nodes if direction == "forward" else view.reverse_nodes,
            model_relations,
            direction=direction,
        )
        model, method = extractor_identity(extractor)
        return FactReasonerV2Run(
            direction=direction,
            relations=policy_result.relations,
            model_relations=model_relations,
            network_relations=network_relations,
            raw_score=_sanitize_native_score(dict(raw_score)),
            marginals=tuple(dict(item) for item in marginals),
            payload=payload,
            nli_runtime={
                "execution_mode": "cache_only_frozen_ledger"
                if cache_only
                else "single_event_loop_cache_prime",
                "prompt_cache_identity": project_nli_cache_identity(project_prompt),
                "parser_version": extractor.parser_version,
                "pairing_mode": view.pairing_mode,
                "all_pairs_counterfactual": (
                    view.forward_pairing_report["possible_pairs"]
                    if direction == "forward"
                    else view.reverse_pairing_report["possible_pairs"]
                ),
                **prime_stats,
                "factreasoner_build_cache_hits": build_totals.get("cache_hits", 0),
                "factreasoner_build_llm_calls": build_totals.get("llm_calls", 0),
            },
            role_mapping={
                "factreasoner_atoms": tuple(item.id for item in claim_nodes),
                "factreasoner_contexts": tuple(item.id for item in evidence_nodes),
                "project_code_nodes": tuple(
                    item.id
                    for item in (
                        view.atoms
                        if direction == "forward"
                        else view.comprehensiveness_atoms
                    )
                ),
                "project_requirement_nodes": tuple(
                    item.id
                    for item in (
                        view.forward_contexts
                        if direction == "forward"
                        else view.reverse_contexts
                    )
                ),
                "project_forward_documentation_bundles": tuple(
                    item.id for item in view.documentation_bundles
                ),
                "project_reverse_requirement_parents": tuple(
                    item.id for item in view.requirement_contexts
                ),
                "project_reverse_requirement_members": tuple(
                    item.id for item in view.requirement_member_contexts
                ),
            },
            prompt_provenance=project_prompt.provenance(
                model=model,
                nli_method=method,
            ),
            relation_policy={
                **policy_result.to_dict(),
                "ambiguity_protocol_version": AMBIGUITY_PROTOCOL_VERSION,
                "ambiguity_weight_policy_version": AMBIGUITY_WEIGHT_POLICY_VERSION,
                "ambiguity_gate_policy_version": AMBIGUITY_GATE_POLICY_VERSION,
                "accepted_weight_rule": "strictly_greater_than_0.5",
                "withheld_high_ambiguity_count": sum(
                    item.is_withheld for item in ambiguity_judgments
                ),
            },
            ambiguity_judgments=tuple(item.to_dict() for item in ambiguity_judgments),
            withheld_high_ambiguity_pairs=tuple(
                pair
                for pair in declared_pairs
                if (pair.premise_id, pair.hypothesis_id)
                in {
                    (judgment.source_id, judgment.target_id)
                    for judgment in ambiguity_judgments
                    if judgment.is_withheld
                }
            ),
            pairing=(
                view.forward_pairing_report
                if direction == "forward"
                else view.reverse_pairing_report
            ),
            nli_pair_mode=view.pairing_mode,
        )
    finally:
        if temporary_cache is not None:
            temporary_cache.cleanup()


def _ambiguity_judgments(
    pairs: Sequence[RelationPair],
    results: Sequence[dict[str, Any]],
    *,
    ambiguity_enabled: bool,
) -> tuple[AmbiguityJudgment, ...]:
    """Convert current ambiguity results while allowing explicit legacy replay."""

    if not ambiguity_enabled:
        return ()
    if len(pairs) != len(results):
        raise ValueError("Ambiguity result count does not match declared pairs")
    judgments = []
    for pair, result in zip(pairs, results, strict=True):
        if result.get("_factreasoner_fallback") is True:
            continue
        if "relation_ambiguity" not in result:
            raise ValueError("Mixed ambiguity and legacy NLI results are not allowed")
        judgments.append(
            AmbiguityJudgment.from_parsed(
                source_id=pair.premise_id,
                target_id=pair.hypothesis_id,
                parsed={
                    "relation": result.get("raw_relation", result.get("label")),
                    "relation_ambiguity": result["relation_ambiguity"],
                    "basis": result["basis"],
                },
                raw_relation_score=(
                    result.get("raw_relation_simbauq_confidence")
                    if "raw_relation_simbauq_confidence" in result
                    else result.get("raw_relation_logprob")
                ),
                raw_relation_score_method=(
                    "simbauq_confidence"
                    if "raw_relation_simbauq_confidence" in result
                    else "label_token_logprob"
                    if "raw_relation_logprob" in result
                    else None
                ),
            )
        )
    return tuple(judgments)


async def _run_cached_relation_pairs(
    pairs: Sequence[RelationPair],
    *,
    text_by_id: dict[str, str],
    extractor: object,
    cache: object | None,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Evaluate missing pairs on the caller's event loop and cache each batch."""

    if batch_size < 1:
        raise ValueError("nli_batch_size must be positive")
    unknown = {
        endpoint
        for pair in pairs
        for endpoint in (pair.premise_id, pair.hypothesis_id)
        if endpoint not in text_by_id
    }
    if unknown:
        raise ValueError(
            f"NLI pairs reference unknown statement IDs: {sorted(unknown)}"
        )

    if cache is not None:
        model_id, method = _extractor_identity(extractor)
        keys = [
            cache.make_key(
                model_id,
                method,
                text_by_id[pair.premise_id],
                text_by_id[pair.hypothesis_id],
            )
            for pair in pairs
        ]
        cached = cache.get_many(keys)
    else:
        keys = [""] * len(pairs)
        cached = {}

    results: list[dict[str, Any] | None] = [cached.get(key) for key in keys]
    missing = [index for index, item in enumerate(results) if item is None]
    for start in range(0, len(missing), batch_size):
        indices = missing[start : start + batch_size]
        batch_results = await extractor.run_batch(
            [text_by_id[pairs[index].premise_id] for index in indices],
            [text_by_id[pairs[index].hypothesis_id] for index in indices],
        )
        if len(batch_results) != len(indices):
            raise RuntimeError(
                "NLI backend returned a result count that does not match the "
                "declared relation batch"
            )
        for index, result in zip(indices, batch_results, strict=True):
            results[index] = dict(result)
        if cache is not None:
            cache.put_many((keys[index], results[index]) for index in indices)

    if any(item is None for item in results):
        raise RuntimeError("NLI evaluation left one or more relation pairs unresolved")
    resolved = [dict(item) for item in results if item is not None]
    return (
        resolved,
        {
            "declared_pairs": len(pairs),
            "direct_llm_calls": len(missing),
            "preexisting_cache_hits": len(pairs) - len(missing),
            "explicit_fallback_results": sum(
                item.get("_factreasoner_fallback") is True for item in resolved
            ),
            "batch_size": batch_size,
        },
    )


def _extractor_identity(extractor: object) -> tuple[str, str]:
    """Read the cache identity without importing FactReasoner at module load."""

    return extractor_identity(extractor)


def _sanitize_native_score(score: dict[str, Any]) -> dict[str, Any]:
    """Remove reference metrics when FactReasoner received no gold labels."""

    references = score.get("references")
    has_references = isinstance(references, dict) and any(
        value is not None for value in references.values()
    )
    result = dict(score)
    result["diagnostic_scope"] = (
        "native FR2 probabilistic inference over raw model relations; project "
        "scores are computed separately from policy-guarded relations"
    )
    result["reference_metrics_available"] = has_references
    if has_references:
        return result
    invalid_reference_fields = (
        "references",
        "gold_factuality_score",
        "gold_true_atoms",
        "true_positive",
        "true_negative",
        "false_positive",
        "false_negative",
        "avg_brier",
    )
    omitted = [key for key in invalid_reference_fields if key in result]
    for key in omitted:
        result.pop(key, None)
    result["reference_metric_fields_omitted"] = omitted
    return result


def run_factreasoner_v2_sync(*args: Any, **kwargs: Any) -> FactReasonerV2Run:
    """Synchronous convenience wrapper for scripts outside an active event loop."""

    return asyncio.run(run_factreasoner_v2(*args, **kwargs))


def validate_fr2_view(
    view: AnalysisView, *, direction: Fr2Direction = "forward"
) -> None:
    """Validate invariants that FR2 would otherwise enforce destructively."""

    claim_nodes, evidence_nodes = _fr2_partitions(view, direction)
    if not claim_nodes:
        raise ValueError(f"{direction} FR2 needs at least one claim")
    if not evidence_nodes:
        raise ValueError(f"{direction} FR2 needs at least one evidence context")
    pairs = view.forward_fr2_pairs if direction == "forward" else view.reverse_fr2_pairs
    if not pairs:
        raise ValueError(f"{direction} FR2 has no adaptively eligible relation pairs")
    paired_claims = {item.hypothesis_id for item in pairs}
    unpaired_claims = [item.id for item in claim_nodes if item.id not in paired_claims]
    if view.pairing_mode == "exhaustive" and unpaired_claims:
        raise ValueError(
            f"{direction} exhaustive FR2 unexpectedly has unpaired claims: "
            f"{unpaired_claims}"
        )
    normalized_contexts = [
        " ".join(item.text.split()).casefold() for item in evidence_nodes
    ]
    if len(normalized_contexts) != len(set(normalized_contexts)):
        raise ValueError(
            "FR2 removes duplicate evidence contexts. Consolidate exact duplicates with their "
            "provenance before running so stable requirement identities are not lost."
        )
    normalized_atoms = [" ".join(item.text.split()).casefold() for item in claim_nodes]
    if len(normalized_atoms) != len(set(normalized_atoms)):
        raise ValueError(
            "FactReasoner removes duplicate claims. Consolidate exact duplicates "
            "with every provenance occurrence before running."
        )


def _fr2_partitions(
    view: AnalysisView, direction: Fr2Direction
) -> tuple[tuple[FactAtom, ...], tuple[FactAtom, ...]]:
    if direction == "forward":
        return view.atoms, view.forward_contexts
    if direction == "reverse":
        return view.reverse_contexts, view.comprehensiveness_atoms
    raise ValueError(f"Unknown FR2 direction: {direction!r}")


def _selected_fr2_partitions(
    view: AnalysisView,
    direction: Fr2Direction,
    pairs: Sequence[RelationPair],
) -> tuple[tuple[FactAtom, ...], tuple[FactAtom, ...], tuple[str, ...]]:
    """Return only nodes participating in the native FR2 runtime graph.

    Project views retain every claim for denominators and audit.  Adaptive
    targets without an eligible relation are deliberately omitted from the
    native FactReasoner payload, then reported as unassessed by the project
    scorer rather than being assigned a fabricated neutral relation.
    """

    all_claim_nodes, all_evidence_nodes = _fr2_partitions(view, direction)
    paired_claim_ids = {item.hypothesis_id for item in pairs}
    selected_evidence_ids = {item.premise_id for item in pairs}
    claim_nodes = tuple(item for item in all_claim_nodes if item.id in paired_claim_ids)
    evidence_nodes = tuple(
        item for item in all_evidence_nodes if item.id in selected_evidence_ids
    )
    unassessed_claim_ids = tuple(
        item.id for item in all_claim_nodes if item.id not in paired_claim_ids
    )
    return claim_nodes, evidence_nodes, unassessed_claim_ids
