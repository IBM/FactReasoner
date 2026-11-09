"""Native FR2 integration with injected NLI, plus the shared async controller."""

import asyncio
from dataclasses import replace
import hashlib
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from fact_reasoner.core.nli import NLIExtractor

from fact_reasoner.skill_consistency.analysis_views import AnalysisView
from fact_reasoner.skill_consistency.comprehensiveness import FactAtom
from fact_reasoner.skill_consistency.factreasoner_prompt_adapter import (
    project_nli_cache_identity,
    prompt_for_direction,
)
from fact_reasoner.skill_consistency.recursive_traversal import (
    Statement,
    TraversalPlan,
    run_traversal,
    run_traversal_async,
)
from fact_reasoner.skill_consistency.relation_probability_proxy import (
    relation_probability_proxy,
)
from fact_reasoner.skill_consistency.traversal import StepScope, ChildScope, Judgment
from fact_reasoner.skill_consistency.traversal_factreasoner import (
    run_recursive_factreasoner,
)
import fact_reasoner.skill_consistency.factreasoner_fr2 as fr2

MERLIN = Path(
    os.environ.get("FACT_REASONER_TEST_MERLIN") or shutil.which("merlin") or "merlin"
)


def inputs(direction="documentation_to_code"):
    nodes = tuple(
        FactAtom(
            i,
            text,
            kind,
            source_file=file,
            source_lines=(line,),
            excerpt=text,
            component="I" if kind == "context" else "C",
            semantic_role="requirement" if kind == "context" else "operation",
        )
        for i, text, kind, file, line in [
            ("A1", "Bundled code writes the report to disk.", "atom", "code.py", 1),
            ("A2", "Bundled code reads the configuration.", "atom", "code.py", 2),
            ("D1", "Write the report to disk.", "context", "SKILL.md", 1),
            ("D2", "Read the configuration.", "context", "SKILL.md", 2),
        ]
    )
    view = AnalysisView("fixture", "question", nodes[:2], nodes[2:])
    targets, evidence = (
        (view.atoms, view.contexts)
        if direction == "documentation_to_code"
        else (view.contexts, view.atoms)
    )

    def s(n):
        return Statement(
            n.id,
            n.text,
            n.source_file,
            n.source_lines,
            n.text,
            hashlib.sha256(n.text.encode()).hexdigest(),
        )

    ids = tuple(n.id for n in targets)
    es = tuple(n.id for n in evidence)
    scopes = (
        StepScope(
            "root", direction, ids, ids[:1], es, tuple(ChildScope(i, (i,)) for i in ids)
        ),
        *(StepScope(i, direction, (i,), (i,), es) for i in ids),
    )
    prompt = prompt_for_direction(
        "documentation_to_code"
        if direction == "documentation_to_code"
        else "code_to_requirement"
    )
    return TraversalPlan(
        direction,
        scopes,
        ("root",),
        tuple(s(n) for n in targets),
        tuple(s(n) for n in evidence),
        project_nli_cache_identity(prompt),
    ), view


def raw(label="entailment", ambiguity="minor_ambiguity"):
    weight = relation_probability_proxy(label, ambiguity)
    return {
        "label": label if weight > 0.5 else "neutral",
        "raw_relation": label,
        "probability": weight if weight > 0.5 else 1.0,
        "relation_ambiguity": ambiguity,
        "basis": "Constructed integration judgment.",
        "raw_relation_logprob": 0.999,
        "discrete_weight": weight,
        "_factreasoner_fallback": False,
    }


class Extractor(NLIExtractor):
    model_id = "fixture-model"
    method = "logprobs"
    parser_version = "test-parser"

    def __init__(self, results=None):
        self.calls = []
        self.results = results

    async def run_batch(self, premises, hypotheses):
        self.calls.append((id(asyncio.get_running_loop()), premises, hypotheses))
        if isinstance(self.results, Exception):
            raise self.results
        return [dict(self.results or raw()) for _ in premises]


def execute(
    tmp_path,
    monkeypatch,
    *,
    direction="documentation_to_code",
    result=None,
    plan_change=None,
):
    monkeypatch.chdir(tmp_path)
    plan, view = inputs(direction)
    if plan_change:
        plan = plan_change(plan)
    nli = Extractor(result)
    monkeypatch.setattr(fr2, "build_project_nli_extractor", lambda *a, **kw: nli)
    run = asyncio.run(
        run_recursive_factreasoner(
            plan,
            view,
            backend=SimpleNamespace(model_id=nli.model_id),
            merlin_path=MERLIN,
            nli_cache_dir=tmp_path / "cache",
            extractor=nli,
            batch_size=1,
            max_judgments=4,
        )
    )
    return run, nli


@pytest.mark.skipif(
    not MERLIN.is_file(), reason="Local native Merlin runtime unavailable"
)
@pytest.mark.parametrize(
    "direction", ["documentation_to_code", "code_to_documentation"]
)
def test_actual_native_fr2_uses_same_labels_and_discrete_weights_without_new_calls(
    tmp_path, monkeypatch, direction
):
    result, nli = execute(tmp_path, monkeypatch, direction=direction)
    assert len({c[0] for c in nli.calls}) == 1
    assert result["counts"]["nli_requests"] == len(nli.calls) == 4
    fr = result["factreasoner"]
    assert fr["direction"] == (
        "forward" if direction == "documentation_to_code" else "reverse"
    )
    assert fr["nli_runtime"]["execution_mode"] == "cache_only_frozen_ledger"
    assert (
        fr["nli_runtime"]["direct_llm_calls"]
        == fr["nli_runtime"]["factreasoner_build_llm_calls"]
        == 0
    )
    assert {
        r["probability"] for r in fr["network_relations"] if r["label"] == "entailment"
    } == {0.75}
    assert len(fr["marginals"]) > 0
    assert result["traversal"]["summary"]["denominator_count"] == 2
    assert [v["scope"]["id"] for v in result["traversal"]["visits"]] == [
        "root",
        "A2" if direction == "documentation_to_code" else "D2",
    ]
    assert len(result["nli_ledger"]) == 4


@pytest.mark.skipif(
    not MERLIN.is_file(), reason="Local native Merlin runtime unavailable"
)
@pytest.mark.parametrize(
    "ambiguity", ["substantial_ambiguity", "closely_competing_relations"]
)
def test_high_ambiguity_is_visible_but_has_no_signed_native_edge(
    tmp_path, monkeypatch, ambiguity
):
    result, _ = execute(tmp_path, monkeypatch, result=raw("entailment", ambiguity))
    assert result["counts"]["withheld_pairs"] == 4
    assert result["traversal"]["summary"]["supported_count"] == 0
    assert all(
        r["status"] == "withheld" and r["judgment"]["relation"] == "entailment"
        for r in result["nli_ledger"]
    )
    assert not [
        r
        for r in result["factreasoner"]["network_relations"]
        if r["label"] in {"entailment", "contradiction"}
    ]


@pytest.mark.parametrize(
    "failure",
    [
        {"_factreasoner_fallback": True, "label": "neutral", "probability": 1.0},
        RuntimeError("failure"),
    ],
)
def test_failed_nli_never_becomes_neutral_or_an_fr2_edge(
    tmp_path, monkeypatch, failure
):
    result, nli = execute(tmp_path, monkeypatch, result=failure)
    assert result["factreasoner"] is None
    assert result["counts"]["fallback_pairs"] == 4
    assert all(r["judgment"] is None for r in result["nli_ledger"])
    assert result["traversal"]["summary"]["unresolved_ids"] == ["A1", "A2"]
    from fact_reasoner.core.nli_cache import NLIVerdictCache

    assert len(NLIVerdictCache(str(tmp_path / "cache"))) == 0
    assert (
        len(nli.calls) == 4
    )  # Revisiting a failed pair is not another provider request.


@pytest.mark.parametrize(
    "field,value",
    [
        ("relation_ambiguity", "invented"),
        ("probability", 0.999),
        ("label", "contradiction"),
        ("basis", ""),
    ],
)
def test_invalid_parsed_values_are_recorded_without_entering_native_cache(
    tmp_path, monkeypatch, field, value
):
    result, _ = execute(tmp_path, monkeypatch, result={**raw(), field: value})
    assert result["counts"]["fallback_pairs"] == 4 and result["factreasoner"] is None


def test_request_budget_aborts_before_backend_or_cache_creation(tmp_path):
    plan, view = inputs()
    nli = Extractor()
    with pytest.raises(ValueError, match="exceeds ceiling"):
        asyncio.run(
            run_recursive_factreasoner(
                plan,
                view,
                backend=None,
                merlin_path=MERLIN,
                nli_cache_dir=tmp_path / "cache",
                extractor=nli,
                max_judgments=3,
            )
        )
    assert nli.calls == [] and not (tmp_path / "cache").exists()


def test_view_statement_changes_are_rejected_before_nli(tmp_path):
    plan, view = inputs()
    nli = Extractor()
    plan = replace(
        plan,
        targets=(replace(plan.targets[0], text="Changed meaning"), plan.targets[1]),
    )
    with pytest.raises(ValueError, match="differ"):
        asyncio.run(
            run_recursive_factreasoner(
                plan,
                view,
                backend=None,
                merlin_path=MERLIN,
                nli_cache_dir=tmp_path / "cache",
                extractor=nli,
            )
        )
    assert nli.calls == []


def test_async_and_sync_controllers_have_identical_decisions():
    plan, _ = inputs()

    def compare(r):
        return [
            Judgment(
                r.scope_id + ":" + t.id,
                r.direction,
                t.id,
                (r.evidence[0].id,),
                "entailment",
                "Fixture.",
            )
            for t in r.targets
        ]

    async def acompare(r):
        return compare(r)

    sync = run_traversal(plan, compare)
    asynchronous = asyncio.run(run_traversal_async(plan, acompare))
    assert sync.summary == asynchronous.summary and sync.metrics == asynchronous.metrics
    assert [(v.path, v.decision) for v in sync.visits] == [
        (v.path, v.decision) for v in asynchronous.visits
    ]


def test_async_cancellation_does_not_become_a_missing_judgment():
    plan, _ = inputs()

    async def compare(r):
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(run_traversal_async(plan, compare))


def test_cache_only_native_mode_refuses_missing_judgments_without_provider_calls(
    tmp_path, monkeypatch
):
    _, view = inputs()
    nli = Extractor()
    monkeypatch.setattr(fr2, "build_project_nli_extractor", lambda *a, **kw: nli)
    with pytest.raises(RuntimeError, match="Cache-only"):
        asyncio.run(
            fr2.run_factreasoner_v2(
                view,
                backend=None,
                merlin_path=MERLIN,
                nli_cache_dir=tmp_path / "empty",
                cache_only=True,
            )
        )
    assert nli.calls == []


def test_pair_eligibility_is_preserved_inside_union_packets(tmp_path, monkeypatch):
    from fact_reasoner.skill_consistency.comprehensiveness import RelationPair
    from fact_reasoner.skill_consistency.traversal_factreasoner import TraversedView

    plan, view = inputs()
    view = TraversedView(
        "fixture",
        "question",
        view.atoms,
        view.contexts,
        pairing_mode="adaptive",
        traversed_forward=(RelationPair("D1", "A1", "context_atom"),),
    )
    nli = Extractor({"_factreasoner_fallback": True})
    result = asyncio.run(
        run_recursive_factreasoner(
            plan,
            view,
            backend=None,
            merlin_path=MERLIN,
            nli_cache_dir=tmp_path / "cache",
            extractor=nli,
            max_judgments=1,
        )
    )
    assert result["counts"]["visited_pairs"] == result["counts"]["nli_requests"] == 1
    assert result["traversal"]["summary"]["denominator_count"] == 2
    assert result["traversal"]["summary"]["unresolved_ids"] == ["A1", "A2"]


def test_progress_preserves_attempt_counts_when_cancelled(tmp_path):
    plan, view = inputs()
    events = []

    class Cancelled(Extractor):
        async def run_batch(self, p, h):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            run_recursive_factreasoner(
                plan,
                view,
                backend=None,
                merlin_path=MERLIN,
                nli_cache_dir=tmp_path / "cache",
                extractor=Cancelled(),
                progress=events.append,
            )
        )
    assert events[-1]["stage"] == "nli" and events[-1]["nli_requests"] == 2


def test_existing_cache_cannot_leak_results_across_repeats(tmp_path):
    plan, view = inputs()
    cache = tmp_path / "cache"
    cache.mkdir()
    nli = Extractor()
    with pytest.raises(FileExistsError, match="fresh NLI cache"):
        asyncio.run(
            run_recursive_factreasoner(
                plan,
                view,
                backend=None,
                merlin_path=MERLIN,
                nli_cache_dir=cache,
                extractor=nli,
            )
        )
    assert nli.calls == []


@pytest.mark.skipif(
    not MERLIN.is_file(), reason="Local native Merlin runtime unavailable"
)
@pytest.mark.parametrize(
    "label,level,weight",
    [
        ("entailment", "unambiguous", 0.9),
        ("entailment", "moderate_ambiguity", 0.6),
        ("contradiction", "moderate_ambiguity", 0.6),
    ],
)
def test_remaining_discrete_weights_and_opposition_reach_native_solver(
    tmp_path, monkeypatch, label, level, weight
):
    result, _ = execute(tmp_path, monkeypatch, result=raw(label, level))
    edges = result["factreasoner"]["network_relations"]
    assert len(edges) == 4
    assert {(e["label"], e["probability"]) for e in edges} == {(label, weight)}
    assert all(
        e["source_id"].startswith("D") and e["target_id"].startswith("A") for e in edges
    )
