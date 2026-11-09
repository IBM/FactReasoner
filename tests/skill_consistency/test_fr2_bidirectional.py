from __future__ import annotations

import asyncio
import inspect
import json

from fact_reasoner.skill_consistency.comprehensiveness import (
    FactAtom,
    FactRelation,
    RelationPair,
    interpret_fr2_support,
    required_fr2_relation_pairs,
    required_reverse_fr2_relation_pairs,
    score_requirement_comprehensiveness,
)
from fact_reasoner.skill_consistency.candidate_pairs import (
    DEFAULT_PAIR_SELECTOR_POLICY,
    PAIR_SELECTOR_ID,
    calibrate_pair_selector,
    evaluate_candidate_recall,
    pairing_report,
    select_candidate_relation_pairs,
    selector_sensitivity_report,
)
from fact_reasoner.skill_consistency.models import SemanticUnit
from fact_reasoner.skill_consistency.relation_policy import apply_relation_policy
from fact_reasoner.skill_consistency.analysis_views import AnalysisView
from fact_reasoner.skill_consistency.factreasoner_fr2 import (
    _run_cached_relation_pairs,
    _sanitize_native_score,
    factreasoner_payload,
    run_factreasoner_v2,
    validate_fr2_view,
)


def test_forward_and_reverse_fr2_have_only_declared_cross_relations() -> None:
    atoms = [FactAtom("A1", "Code sends a request.", "atom")]
    contexts = [
        FactAtom("C1", "Use the weather service.", "context"),
        FactAtom("C2", "The service is wttr.in.", "context"),
    ]

    forward = required_fr2_relation_pairs(atoms, contexts)
    reverse = required_reverse_fr2_relation_pairs(atoms, contexts)

    assert {item.kind for item in forward} == {"context_atom"}
    assert {item.kind for item in reverse} == {"atom_context"}
    assert len(forward) == len(atoms) * len(contexts)
    assert len(reverse) == len(atoms) * len(contexts)
    assert not any(
        item.premise_id.startswith("A") and item.hypothesis_id.startswith("A")
        for item in (*forward, *reverse)
    )


def test_adaptive_pairing_is_variable_deterministic_and_auditable() -> None:
    evidence = [
        FactAtom(
            "C1",
            "Send an HTTP request to https://weather.example/{city}.",
            "context",
            capability_types=("network_request",),
            entities=("https://weather.example/{city}",),
        ),
        FactAtom(
            "C2",
            "Return a concise weather forecast.",
            "context",
            capability_types=("weather_forecast",),
        ),
        FactAtom("C3", "Read a local settings file.", "context"),
        FactAtom("C4", "Execute a helper process.", "context"),
    ]
    claims = [
        FactAtom(
            "A1",
            "Send an HTTP request to https://weather.example/London.",
            "atom",
            capability_types=("network_request",),
            entities=("https://weather.example/London",),
        ),
        FactAtom(
            "A2",
            "Return a concise weather forecast.",
            "atom",
            capability_types=("weather_forecast",),
        ),
        FactAtom("A3", "Rotate a cryptographic signing key.", "atom"),
    ]

    first = select_candidate_relation_pairs(
        evidence,
        claims,
        kind="context_atom",
        false_discovery_rate=0.05,
    )
    second = select_candidate_relation_pairs(
        evidence,
        claims,
        kind="context_atom",
        false_discovery_rate=0.05,
    )

    assert first == second
    assert len(first) == 2
    assert {item.hypothesis_id for item in first} == {"A1", "A2"}
    assert first[0].premise_id == "C1"
    assert any(item.premise_id == "C2" and item.hypothesis_id == "A2" for item in first)
    assert all(item.selection_method.startswith(PAIR_SELECTOR_ID) for item in first)
    assert all(item.selection_rank == 1 for item in first)
    assert all(item.selection_reasons for item in first)

    report = pairing_report(
        first,
        possible_pair_count=12,
        claim_nodes=claims,
        mode="adaptive",
        false_discovery_rate=0.05,
    )
    assert report["selected_pairs"] == 2
    assert report["pruned_pairs"] == 10
    assert report["claims_without_candidates"] == ["A3"]
    assert report["target_retrieval_coverage"] == 2 / 3
    assert report["forced_fallback_pair_count"] == 0
    assert report["selector_policy"]["policy_sha256"]
    assert len(report["selected_pair_provenance"]) == 2


def test_capability_mismatch_does_not_hard_filter_candidate_pair() -> None:
    pairs = select_candidate_relation_pairs(
        [
            FactAtom(
                "C1",
                "Perform the declared operation.",
                "context",
                capability_types=("filesystem_read",),
                entities=("/tmp/settings.json",),
            )
        ],
        [
            FactAtom(
                "A1",
                "Perform the declared operation.",
                "atom",
                capability_types=("network_request",),
                entities=("/tmp/settings.json",),
            )
        ],
        kind="context_atom",
        false_discovery_rate=0.05,
    )

    assert [(item.premise_id, item.hypothesis_id) for item in pairs] == [("C1", "A1")]


def test_candidate_mode_payload_contains_only_selected_contexts_per_claim() -> None:
    view = AnalysisView(
        id="candidate",
        question="question",
        atoms=(
            FactAtom(
                "A1",
                "Send an HTTP request.",
                "atom",
                capability_types=("network_request",),
            ),
            FactAtom("A2", "Read a local file.", "atom"),
        ),
        contexts=(
            FactAtom(
                "C1",
                "Use an HTTP service.",
                "context",
                capability_types=("network_request",),
            ),
            FactAtom("C2", "Read the repository file.", "context"),
            FactAtom("C3", "Execute a process.", "context"),
        ),
        pairing_mode="adaptive",
        lexical_false_discovery_rate=0.05,
    )

    payload = factreasoner_payload(view, direction="forward")
    selected_by_claim = {
        item.hypothesis_id: [item.premise_id] for item in view.forward_fr2_pairs
    }

    assert len(view.forward_fr2_pairs) == 1
    assert all(
        atom["contexts"] == selected_by_claim[atom["id"]] for atom in payload["atoms"]
    )
    assert [item["id"] for item in payload["atoms"]] == ["A1"]
    assert payload["unassessed_claim_ids"] == ["A2"]
    assert payload["pairing"]["possible_pairs"] == 6
    assert payload["pairing"]["selected_pairs"] == 1
    assert "contexts_per_atom_only=True" in inspect.getsource(run_factreasoner_v2)


def test_candidate_recall_is_not_presented_as_semantic_accuracy() -> None:
    candidates = [
        RelationPair("C1", "A1", "context_atom"),
        RelationPair("C2", "A1", "context_atom"),
    ]
    exhaustive = [
        FactRelation("C1", "A1", "entailment"),
        FactRelation("C2", "A1", "neutral"),
        FactRelation("C3", "A1", "contradiction"),
    ]

    result = evaluate_candidate_recall(candidates, exhaustive)

    assert result["human_gold"] is False
    assert result["decisive_relation_count"] == 2
    assert result["retained_decisive_relation_count"] == 1
    assert result["decisive_relation_recall"] == 0.5
    assert result["by_label"]["entailment"]["recall"] == 1.0
    assert result["by_label"]["contradiction"]["recall"] == 0.0


def test_adaptive_selector_sensitivity_and_calibration_are_auditable() -> None:
    evidence = (
        FactAtom(
            "C1",
            "Send a request to the weather endpoint.",
            "context",
            capability_types=("network_request",),
        ),
        FactAtom("C2", "Read a repository file.", "context"),
    )
    claims = (
        FactAtom(
            "A1",
            "Send an HTTP request.",
            "atom",
            capability_types=("network_request",),
        ),
        FactAtom("A2", "Rotate a signing key.", "atom"),
    )

    sensitivity = selector_sensitivity_report(
        evidence,
        claims,
        kind="context_atom",
        false_discovery_rates=(0.01, 0.05),
    )
    assert [item["lexical_false_discovery_rate"] for item in sensitivity] == [
        0.01,
        0.05,
    ]
    assert all(item["selected_pairs"] >= 1 for item in sensitivity)
    assert DEFAULT_PAIR_SELECTOR_POLICY.forced_fallback is False
    assert DEFAULT_PAIR_SELECTOR_POLICY.policy_sha256

    calibration = calibrate_pair_selector(
        evidence,
        claims,
        (FactRelation("C1", "A1", "entailment"),),
        kind="context_atom",
        false_discovery_rates=(0.01, 0.05),
        minimum_decisive_recall=1.0,
    )
    assert calibration["status"] == "selected"
    assert calibration["human_gold"] is False
    assert calibration["semantic_accuracy_estimate"] is False
    assert calibration["chosen_policy"]["policy_sha256"]


def test_factreasoner_payload_preserves_frozen_nodes_and_context_schema() -> None:
    view = AnalysisView(
        id="example",
        question="question",
        atoms=(
            FactAtom(
                "A1",
                "Frozen code behavior.",
                "atom",
                observation_ids=("O1",),
                excerpt="call()",
            ),
        ),
        contexts=(
            FactAtom(
                "C1",
                "Exact requirement.",
                "context",
                component="instruction",
                source_file="spec.md",
                excerpt="Exact requirement.",
            ),
        ),
    )

    payload = factreasoner_payload(view, direction="forward")

    atom_model_input = json.loads(payload["atoms"][0]["text"])
    context_model_input = json.loads(payload["contexts"][0]["text"])
    assert atom_model_input == {
        "entities": [],
        "statement": "Frozen code behavior.",
        "statement_type": "grounded_code_atom",
    }
    assert context_model_input == {
        "entities": [],
        "statement": "Exact requirement.",
        "statement_type": "documentation_requirement",
    }
    assert payload["atoms"][0]["original"] == "Frozen code behavior."
    assert payload["atoms"][0]["normalized_statement"] == "Frozen code behavior."
    assert payload["atoms"][0]["contexts"] == ["C1"]
    assert payload["atoms"][0]["observation_ids"] == ["O1"]
    assert {"id", "title", "text", "snippet", "link"} <= set(payload["contexts"][0])
    assert payload["contexts"][0]["snippet"] == ""
    assert payload["contexts"][0]["excerpt"] == "Exact requirement."
    assert "call()" not in payload["atoms"][0]["text"]
    assert "Exact requirement." not in atom_model_input["statement"]
    assert payload["input"] == "question"
    assert payload["output"] == "Frozen code behavior."

    reverse = factreasoner_payload(view, direction="reverse")
    assert [item["id"] for item in reverse["atoms"]] == ["C1"]
    assert [item["id"] for item in reverse["contexts"]] == ["A1"]
    assert reverse["atoms"][0]["project_kind"] == "context"
    assert reverse["contexts"][0]["project_kind"] == "atom"
    assert reverse["atoms"][0]["contexts"] == ["A1"]


def test_cached_pair_runner_keeps_remote_calls_on_one_event_loop() -> None:
    class FakeCache:
        def __init__(self) -> None:
            self.values: dict[str, dict[str, object]] = {}

        def make_key(
            self,
            model_id: str,
            method: str,
            premise: str,
            hypothesis: str,
        ) -> str:
            return "|".join((model_id, method, premise, hypothesis))

        def get_many(self, keys: list[str]) -> dict[str, dict[str, object]]:
            return {key: self.values[key] for key in keys if key in self.values}

        def put_many(self, items: object) -> None:
            for key, value in items:  # type: ignore[union-attr]
                if value is not None:
                    self.values[key] = value

    class FakeBackend:
        model_id = "test-model"

    class FakeExtractor:
        method = "logprobs"
        backend = FakeBackend()

        def __init__(self) -> None:
            self.loop_ids: list[int] = []
            self.calls = 0

        async def run_batch(
            self, premises: list[str], hypotheses: list[str]
        ) -> list[dict[str, object]]:
            self.loop_ids.append(id(asyncio.get_running_loop()))
            self.calls += len(premises)
            return [
                {"label": "entailment", "probability": 0.9}
                for _ in zip(premises, hypotheses, strict=True)
            ]

    pairs = [
        RelationPair("C1", "A1", "context_atom"),
        RelationPair("C2", "A1", "context_atom"),
        RelationPair("A1", "C1", "atom_context"),
    ]
    texts = {"A1": "atom", "C1": "first", "C2": "second"}
    cache = FakeCache()
    extractor = FakeExtractor()

    async def exercise() -> None:
        first, first_stats = await _run_cached_relation_pairs(
            pairs,
            text_by_id=texts,
            extractor=extractor,
            cache=cache,
            batch_size=2,
        )
        second, second_stats = await _run_cached_relation_pairs(
            pairs,
            text_by_id=texts,
            extractor=extractor,
            cache=cache,
            batch_size=2,
        )
        assert first == second
        assert first_stats["direct_llm_calls"] == 3
        assert second_stats["direct_llm_calls"] == 0
        assert second_stats["preexisting_cache_hits"] == 3

    asyncio.run(exercise())

    assert extractor.calls == 3
    assert len(set(extractor.loop_ids)) == 1


def test_fr2_view_rejects_duplicate_context_text_before_runtime() -> None:
    view = AnalysisView(
        id="duplicates",
        question="question",
        atoms=(FactAtom("A1", "Behavior", "atom"),),
        contexts=(
            FactAtom("C1", "Same requirement", "context"),
            FactAtom("C2", " same   requirement ", "context"),
        ),
    )

    try:
        validate_fr2_view(view, direction="forward")
    except ValueError as exc:
        assert "duplicate evidence contexts" in str(exc)
    else:
        raise AssertionError("duplicate context text was accepted")


def test_fr2_view_rejects_duplicate_atom_text_before_runtime() -> None:
    view = AnalysisView(
        id="duplicates",
        question="question",
        atoms=(
            FactAtom("A1", "Same behavior", "atom"),
            FactAtom("A2", " same   behavior ", "atom"),
        ),
        contexts=(FactAtom("C1", "Requirement", "context"),),
    )

    try:
        validate_fr2_view(view, direction="forward")
    except ValueError as exc:
        assert "duplicate claims" in str(exc)
    else:
        raise AssertionError("duplicate atom text was accepted")


def test_neutral_is_undetermined_and_behavior_aggregation_is_strict() -> None:
    nodes = [
        FactAtom(
            "A1",
            "Code sends a request.",
            "atom",
            claim_id="a",
            unit_id="B1",
            grounding_status="grounded",
        ),
        FactAtom(
            "A2",
            "Code uses a timeout.",
            "atom",
            claim_id="b",
            unit_id="B1",
            grounding_status="grounded",
        ),
        FactAtom("C1", "Use the weather service.", "context"),
    ]
    relations = [
        FactRelation("C1", "A1", "entailment", 0.9),
        FactRelation("C1", "A2", "neutral", 0.8),
    ]
    unit = SemanticUnit(
        id="B1",
        unit_kind="behavior",
        statement="Request weather with a timeout.",
        claim_ids=("a", "b"),
        component="code",
    )

    result = interpret_fr2_support(nodes, relations, behavior_units=[unit])

    assert [item.status for item in result.atom_assessments] == [
        "supported",
        "undetermined",
    ]
    assert result.behavior_assessments[0].status == "partial"
    assert result.atom_explicit_support_rate == 0.5
    assert result.behavior_strict_support_rate == 0.0
    assert result.behavior_fractional_support_rate == 0.5
    assert result.behavior_assessments[0].support_fraction == 0.5
    assert result.grounded_behavior_count == 1
    assert result.unassessed_behavior_count == 0


def test_conflicting_explicit_url_cannot_count_as_support() -> None:
    nodes = [
        FactAtom(
            "A1",
            "Code sends data to B.",
            "atom",
            grounding_status="grounded",
            capability_types=("network_request",),
            entities=("https://b.example",),
        ),
        FactAtom(
            "C1",
            "Use service A.",
            "context",
            capability_types=("network_request",),
            entities=("https://a.example",),
        ),
    ]

    result = interpret_fr2_support(
        nodes, [FactRelation("C1", "A1", "entailment", 0.99)]
    )

    assessment = result.atom_assessments[0]
    assert assessment.status == "undetermined"
    assert "conflicting_explicit_url" in assessment.support_review_flags
    assert "entailment_downgraded_by_entity_guard" in assessment.support_review_flags


def test_requirement_comprehensiveness_aggregates_constituent_contexts() -> None:
    nodes = [
        FactAtom("B1", "Joint implementation evidence.", "atom"),
        FactAtom("C1", "Fetch the data.", "context", claim_id="r1"),
        FactAtom("C2", "Format the output.", "context", claim_id="r2"),
        FactAtom("C3", "Do not modify files.", "context", claim_id="r3"),
    ]
    relations = [
        FactRelation("B1", "C1", "entailment"),
        FactRelation("B1", "C2", "neutral"),
        FactRelation("B1", "C3", "entailment"),
    ]
    requirements = [
        SemanticUnit(
            id="REQ1",
            unit_kind="requirement",
            statement="Fetch and format.",
            claim_ids=("r1", "r2"),
            component="instruction",
        ),
        SemanticUnit(
            id="REQ2",
            unit_kind="requirement",
            statement="Do not modify files.",
            claim_ids=("r3",),
            component="instruction",
        ),
    ]

    result = score_requirement_comprehensiveness(
        nodes,
        relations,
        requirements,
        posterior_by_context_id={"C1": 0.92, "C2": 0.44, "C3": 0.97},
    )

    assert result.strict_score == 0.5
    assert result.fractional_score == 0.75
    assert [item.status for item in result.assessments] == ["partial", "covered"]
    assert result.assessments[0].posterior_min_probability == 0.44
    assert result.assessments[0].posterior_mean_probability == 0.68


def test_ungrounded_code_premise_cannot_cover_a_requirement() -> None:
    nodes = [
        FactAtom(
            "A1",
            "Code sends a request.",
            "atom",
            grounding_status="uncertain",
        ),
        FactAtom("C1", "Send a request.", "context", claim_id="r1"),
    ]
    requirement = SemanticUnit(
        id="REQ1",
        unit_kind="requirement",
        statement="Send a request.",
        claim_ids=("r1",),
        component="instruction",
    )

    result = score_requirement_comprehensiveness(
        nodes,
        [FactRelation("A1", "C1", "entailment")],
        [requirement],
    )

    assert result.strict_score == 0.0
    assert result.assessments[0].status == "unassessed"
    assert result.assessments[0].evidence_atom_ids == ()


def test_missing_candidate_relation_is_unassessed_not_neutral() -> None:
    nodes = [
        FactAtom("A1", "Code sends a request.", "atom", grounding_status="grounded"),
        FactAtom("C1", "Send a request.", "context", claim_id="r1"),
    ]
    requirement = SemanticUnit(
        id="REQ1",
        unit_kind="requirement",
        statement="Send a request.",
        claim_ids=("r1",),
        component="instruction",
    )

    forward = interpret_fr2_support(nodes, [], pairing_mode="adaptive")
    reverse = score_requirement_comprehensiveness(
        nodes,
        [],
        [requirement],
        pairing_mode="adaptive",
    )

    assert forward.atom_assessments[0].status == "unassessed"
    assert forward.atom_assessments[0].evaluated_relation_count == 0
    assert reverse.assessments[0].status == "unassessed"
    assert reverse.assessments[0].evaluated_parent_relation_count == 0
    assert reverse.strict_parent_score == 0.0


def test_relation_policy_neutralizes_read_only_false_contradiction() -> None:
    nodes = [
        FactAtom(
            "C1",
            "Do not make changes; just report findings.",
            "context",
            capability_types=("filesystem_write",),
            polarity="negative",
        ),
        FactAtom(
            "A1",
            "Bundled code reads surrounding source and reports a finding.",
            "atom",
            capability_types=("filesystem_read", "repository_inspection"),
            polarity="positive",
        ),
    ]

    result = apply_relation_policy(
        nodes,
        [FactRelation("C1", "A1", "contradiction", 0.99)],
        direction="forward",
    )

    guarded = result.relations[0]
    assert guarded.label == "neutral"
    assert guarded.probability is None
    assert guarded.raw_label == "contradiction"
    assert guarded.raw_probability == 0.99
    assert guarded.policy_action == "neutralized"
    assert "read_only_activity_is_compatible_with_no_modification" in (
        guarded.policy_reasons
    )


def test_relation_policy_preserves_explicit_same_scope_contradiction() -> None:
    nodes = [
        FactAtom(
            "C1",
            "The skill must never make an outbound network request.",
            "context",
            capability_types=("network_request",),
            polarity="negative",
        ),
        FactAtom(
            "A1",
            "Bundled code makes an outbound network request.",
            "atom",
            capability_types=("network_request",),
            polarity="positive",
        ),
    ]

    result = apply_relation_policy(
        nodes,
        [FactRelation("C1", "A1", "contradiction", 0.93)],
        direction="forward",
    )

    assert result.adjusted_count == 0
    assert result.relations[0] == FactRelation("C1", "A1", "contradiction", 0.93)


def test_capability_type_mismatch_is_review_only() -> None:
    nodes = [
        FactAtom(
            "C1",
            "The skill produces the requested report.",
            "context",
            capability_types=("output_generation",),
        ),
        FactAtom(
            "A1",
            "Bundled code returns the requested report.",
            "atom",
            capability_types=("model_instruction",),
        ),
    ]

    result = apply_relation_policy(
        nodes,
        [FactRelation("C1", "A1", "entailment", 0.91)],
        direction="forward",
    )

    relation = result.relations[0]
    assert relation.label == "entailment"
    assert relation.probability == 0.91
    assert relation.policy_action == "unchanged"
    assert relation.review_flags == ("no_shared_capability_type",)
    assert result.review_flag_counts == (("no_shared_capability_type", 1),)


def test_dynamic_and_concrete_urls_are_comparable() -> None:
    nodes = [
        FactAtom(
            "C1",
            "Use https://wttr.in/London?format=3.",
            "context",
            entities=("https://wttr.in/London?format=3",),
        ),
        FactAtom(
            "A1",
            "Code requests https://wttr.in/{location}?format=3.",
            "atom",
            grounding_status="grounded",
            entities=("https://wttr.in/{location}?format=3",),
        ),
    ]

    result = interpret_fr2_support(
        nodes,
        [FactRelation("C1", "A1", "entailment", 0.9)],
    )

    assert result.atom_assessments[0].status == "supported"
    assert (
        "conflicting_explicit_url"
        not in result.atom_assessments[0].support_review_flags
    )


def test_parent_and_member_coverage_are_separate_and_role_stratified() -> None:
    requirement = SemanticUnit(
        id="REQ1",
        unit_kind="requirement",
        statement="Fetch and format.",
        claim_ids=("r1", "r2"),
        component="instruction",
        documentation_role="normative_requirement",
    )
    nodes = [
        FactAtom("B1", "Joint code evidence.", "atom", grounding_status="grounded"),
        FactAtom(
            "R_REQ1",
            "Joint documentation requirement: fetch and format.",
            "context",
            claim_ids=("r1", "r2"),
            unit_id="REQ1",
            granularity="requirement_unit",
        ),
        FactAtom(
            "RM_1",
            "Requirement member: fetch.",
            "context",
            claim_ids=("r1",),
            granularity="requirement_member",
        ),
        FactAtom(
            "RM_2",
            "Requirement member: format.",
            "context",
            claim_ids=("r2",),
            granularity="requirement_member",
        ),
    ]
    relations = [
        FactRelation("B1", "R_REQ1", "neutral"),
        FactRelation("B1", "RM_1", "entailment"),
        FactRelation("B1", "RM_2", "neutral"),
    ]

    result = score_requirement_comprehensiveness(
        nodes,
        relations,
        [requirement],
        posterior_by_context_id={"R_REQ1": 0.3, "RM_1": 0.9, "RM_2": 0.4},
    )

    assessment = result.assessments[0]
    assert assessment.status == "partial"
    assert assessment.parent_status == "not_evidenced"
    assert assessment.member_coverage_fraction == 0.5
    assert result.strict_parent_score == 0.0
    assert result.member_fractional_score == 0.5
    assert (
        result.by_documentation_role[0]["documentation_role"] == "normative_requirement"
    )
    assert assessment.native_fr2_diagnostics["used_for_guarded_state"] is False


def test_mixed_reverse_evidence_is_not_counted_as_coverage() -> None:
    nodes = [
        FactAtom("A1", "Supporting implementation.", "atom"),
        FactAtom("A2", "Conflicting implementation.", "atom"),
        FactAtom("C1", "Complete requirement.", "context", claim_id="r1"),
    ]
    requirement = SemanticUnit(
        id="REQ1",
        unit_kind="requirement",
        statement="Complete requirement.",
        claim_ids=("r1",),
        component="instruction",
    )

    result = score_requirement_comprehensiveness(
        nodes,
        [
            FactRelation("A1", "C1", "entailment", 0.9),
            FactRelation("A2", "C1", "contradiction", 0.8),
        ],
        [requirement],
    )

    assessment = result.assessments[0]
    assert assessment.status == "mixed"
    assert assessment.covered_context_ids == ()
    assert assessment.mixed_context_ids == ("C1",)
    assert assessment.coverage_fraction == 0.0
    assert result.strict_score == 0.0
    assert result.fractional_score == 0.0


def test_native_score_drops_fake_gold_metrics_when_references_are_absent() -> None:
    sanitized = _sanitize_native_score(
        {
            "factuality_score": 0.5,
            "references": {"A1": None},
            "gold_factuality_score": 0.0,
            "gold_true_atoms": 0,
            "true_positive": 0,
            "true_negative": 1,
            "false_positive": 1,
            "false_negative": 0,
            "avg_brier": 0.5,
        }
    )

    assert sanitized["factuality_score"] == 0.5
    assert sanitized["reference_metrics_available"] is False
    assert "gold_factuality_score" not in sanitized
    assert "true_negative" not in sanitized
    assert "avg_brier" not in sanitized
