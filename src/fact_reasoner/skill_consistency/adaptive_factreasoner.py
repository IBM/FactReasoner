"""Adaptive comparisons through the existing Mellea extractor and native FR2.

Prompts, ambiguity-to-weight mapping and relation guards are unchanged. The
search discovers source-pinned pairs; native FR2 reuses the exact parsed ledger.
"""

from __future__ import annotations
from dataclasses import asdict, fields
import hashlib
import json
import time
from pathlib import Path
from .adaptive_search import search
from .factreasoner_prompt_adapter import (
    REVERSE_CODE_TO_REQUIREMENT,
    build_project_nli_extractor,
    extractor_identity,
    serialize_node_for_nli,
    project_nli_cache_identity,
)
from .factreasoner_fr2 import _ambiguity_judgments, run_factreasoner_v2
from .relation_policy import apply_relation_policy
from .traversal import Judgment
from .traversal_factreasoner import (
    TraversedView,
    prepare_traversal_plan,
    verify_plan_view,
)
from .analysis_views import AnalysisView

VERSION = "adaptive-mellea-fr2-1.0.0"


def pair_view(view, pairs):
    values = {f.name: getattr(view, f.name) for f in fields(AnalysisView)}
    values["pairing_mode"] = "adaptive"
    return TraversedView(**values, traversed_reverse=tuple(pairs))


class JudgmentSession:
    def __init__(self, space, extractor, *, cache_dir, batch_size=16):
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        self.space = space
        self.extractor = extractor
        self.batch_size = batch_size
        self.model, self.method = extractor_identity(extractor)
        self.prompt = REVERSE_CODE_TO_REQUIREMENT
        self.nodes = {n.id: n for n in space.view.reverse_nodes}
        self.texts = {i: serialize_node_for_nli(n) for i, n in self.nodes.items()}
        self.allowed = {(p.premise_id, p.hypothesis_id): p for p in space.pairs}
        self.records = {}
        self.text_results = {}
        self.logical_requests = 0
        self.provider_seconds = 0.0
        self.cache_dir = Path(cache_dir)
        if self.cache_dir.exists():
            raise FileExistsError(
                "Require a fresh native cache for each arm/repetition"
            )
        from fact_reasoner.core.nli_cache import NLIVerdictCache

        self.cache = NLIVerdictCache(
            str(self.cache_dir), prompt_version=project_nli_cache_identity(self.prompt)
        )

    def convert(self, pair, raw):
        if raw.get("_factreasoner_fallback") is True:
            return None, None, "fallback"
        if (
            raw.get("_factreasoner_fallback") is not False
            or raw.get("_nli_score_status") == "invalid"
        ):
            raise ValueError("Invalid parser disposition")
        parsed = _ambiguity_judgments((pair,), (raw,), ambiguity_enabled=True)[0]
        if raw.get("label") != (
            "neutral" if parsed.is_withheld else parsed.relation
        ) or raw.get("probability") != (
            1.0 if parsed.is_withheld else parsed.discrete_weight
        ):
            raise ValueError(
                "Native cache differs from the unchanged discrete-weight policy"
            )
        relation = parsed.to_fact_relation()
        guarded = (
            None
            if relation is None
            else apply_relation_policy(
                tuple(self.nodes.values()), (relation,), direction="reverse"
            ).relations[0]
        )
        identifier = (
            "NLI:"
            + hashlib.sha256(
                json.dumps(
                    [
                        project_nli_cache_identity(self.prompt),
                        self.model,
                        self.method,
                        pair.premise_id,
                        pair.hypothesis_id,
                    ],
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        )
        judgment = Judgment(
            identifier,
            "code_to_documentation",
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

    async def judge(self, pairs):
        if any((p.premise_id, p.hypothesis_id) not in self.allowed for p in pairs):
            raise ValueError("Pair outside frozen allowable search space")
        groups = {}
        for pair in pairs:
            if (pair.premise_id, pair.hypothesis_id) in self.records:
                continue
            key = (self.texts[pair.premise_id], self.texts[pair.hypothesis_id])
            groups.setdefault(key, []).append(pair)
        pending = [k for k in groups if k not in self.text_results]
        for start in range(0, len(pending), self.batch_size):
            keys = pending[start : start + self.batch_size]
            self.logical_requests += len(keys)
            began = time.monotonic()
            try:
                batch = await self.extractor.run_batch(
                    [k[0] for k in keys], [k[1] for k in keys]
                )
                if len(batch) != len(keys):
                    raise ValueError("Incomplete NLI batch")
            except Exception as exc:
                batch = [
                    {"_factreasoner_fallback": True, "error_type": type(exc).__name__}
                    for _ in keys
                ]
            self.provider_seconds += time.monotonic() - began
            if batch and all(v.get("_factreasoner_fallback") is True for v in batch):
                raise RuntimeError(
                    "Whole adaptive NLI batch failed; stop provider traffic"
                )
            for key, raw in zip(keys, batch, strict=True):
                try:
                    self.convert(groups[key][0], raw)
                except (ValueError, KeyError, TypeError, AttributeError, IndexError):
                    raw = {
                        "_factreasoner_fallback": True,
                        "error_type": "invalid_parsed_judgment",
                    }
                # Retain failed results in this run to avoid selective retry.
                # They are not written to the official NLI cache.
                self.text_results[key] = dict(raw)
                if not raw.get("_factreasoner_fallback"):
                    self.cache.put_many(
                        [
                            (
                                self.cache.make_key(self.model, self.method, *key),
                                dict(raw),
                            )
                        ]
                    )
        for key, aliases in groups.items():
            raw = self.text_results[key]
            for p in aliases:
                judgment, ambiguity, status = self.convert(p, raw)
                self.records[p.premise_id, p.hypothesis_id] = dict(
                    pair=p.to_dict(),
                    raw_result=raw,
                    judgment=None if judgment is None else asdict(judgment),
                    ambiguity=ambiguity,
                    status=status,
                )
        result = []
        for p in pairs:
            row = self.records[p.premise_id, p.hypothesis_id]
            if row["judgment"]:
                j = {
                    **row["judgment"],
                    "evidence_ids": tuple(row["judgment"]["evidence_ids"]),
                }
                result.append(Judgment(**j))
        return tuple(result)


async def run_adaptive_factreasoner(
    space,
    *,
    backend,
    merlin_path,
    nli_cache_dir,
    specification,
    code_files,
    sources,
    policy="adaptive",
    extractor=None,
    batch_size=16,
    max_judgments=2500,
    progress=None,
):
    if max_judgments < 1:
        raise ValueError("Request ceiling must be positive")
    nodes = {n.id: n for n in space.view.reverse_nodes}
    unique = {
        (
            serialize_node_for_nli(nodes[p.premise_id]),
            serialize_node_for_nli(nodes[p.hypothesis_id]),
        )
        for p in space.pairs
    }
    if len(unique) > max_judgments:
        raise ValueError(
            "Full allowable search space exceeds ceiling; no target truncation"
        )
    expanded = pair_view(space.view, space.pairs)
    # Reuse the existing exact source/excerpt/hash verification before any call.
    plan = prepare_traversal_plan(
        expanded,
        direction="code_to_documentation",
        specification=specification,
        code_files=code_files,
        sources=sources,
    )
    verify_plan_view(plan, expanded)
    nli = extractor or build_project_nli_extractor(
        backend, prompt=REVERSE_CODE_TO_REQUIREMENT, nli_method="logprobs"
    )
    session = JudgmentSession(
        space, nli, cache_dir=nli_cache_dir, batch_size=batch_size
    )
    began = time.monotonic()
    traversal = await search(space, session.judge, policy=policy, progress=progress)
    # The same evidence IDs survive all visits. Native graph construction sees
    # each actually judged pair once, retaining the full target denominator.
    pairs = tuple(
        space_pair
        for key, space_pair in session.allowed.items()
        if key in session.records and session.records[key]["status"] != "fallback"
    )
    native = None
    native_start = time.monotonic()
    if pairs:
        native = await run_factreasoner_v2(
            pair_view(space.view, pairs),
            direction="reverse",
            backend=backend,
            merlin_path=merlin_path,
            nli_method="logprobs",
            nli_cache_dir=session.cache_dir,
            nli_batch_size=batch_size,
            prompt=REVERSE_CODE_TO_REQUIREMENT,
            cache_only=True,
        )
    native_seconds = time.monotonic() - native_start
    traversal["judgments"] = [asdict(j) for j in traversal["judgments"]]
    return dict(
        version=VERSION,
        traversal=traversal,
        nli_ledger=list(session.records.values()),
        factreasoner=None if native is None else native.to_dict(),
        prompt_provenance=REVERSE_CODE_TO_REQUIREMENT.provenance(
            model=session.model, nli_method=session.method
        ),
        counts=dict(
            **traversal["counts"],
            nli_requests=session.logical_requests,
            fallback_pairs=sum(
                r["status"] == "fallback" for r in session.records.values()
            ),
            withheld_pairs=sum(
                r["status"] == "withheld" for r in session.records.values()
            ),
            factreasoner_additional_nli_calls=0,
        ),
        timing=dict(
            total_seconds=time.monotonic() - began,
            nli_seconds=session.provider_seconds,
            native_seconds=native_seconds,
        ),
        interpretation="Guarded categorical traversal and ordinary discrete-weight native FR2 over raw accepted relations; no calibrated probability or proof of runtime behavior.",
    )
