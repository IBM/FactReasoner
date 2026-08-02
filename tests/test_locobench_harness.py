# Copyright 2023-present the International Business Machines.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the LoCoBench generation harness (offline).

Nothing here touches a network, an LLM or the Merlin binary: the harness's ``--dry-run``
generator is the fixture, which is the point of having built it. The classes cover
the checks the Phase-2 plan calls out by name:

* topic coverage is enforced rather than assumed,
* the builder assertion actually fires on gold that contradicts the taxonomy,
* resume is real (a second run does zero work),
* gate failures are recoverable and recorded with reasons,
* Defect 1 is encoded in the emitted expectations,
* the prompts still match Phase 1, so the two cannot drift silently,
* the published cost model stays derived from the real ladders, and
* THE LIVE PATH. ``build_llm`` shipped broken precisely because the dry run substitutes
  at the ``LLM`` seam, so no test ever built a live callable. Those tests monkeypatch
  ``build_backend`` and ``ainstruct``, so they need no credentials.
"""

import asyncio
import json
import os

import pytest

from fact_reasoner.locobench import (
    mock,
    parse,
    perturb,
    pipeline,
    prompts,
    topics,
    validate,
)
from fact_reasoner.locobench.cli import _build_generators, _interleave, _slots
from fact_reasoner.locobench.config import (
    Capabilities,
    GenConfig,
    ModelRef,
    load_config,
    load_models,
)
from fact_reasoner.locobench.pipeline import (
    _atoms_payload,
    _Caller,
    _check_probes,
    _check_retry_note,
    _retry_note,
    _temperature_for,
    build_llm,
    generate_family,
    make_mock_llm,
    which_prompt,
)
from fact_reasoner.locobench.schema import SchemaError, validate_item
from fact_reasoner.locobench.store import FamilyState, Store


def _dry_cfg(**kw):
    """A dry-run config with sane defaults for a small test corpus."""
    kw.setdefault("dry_run", True)
    kw.setdefault("n_families", 4)
    return GenConfig(**kw)


def _valid_item():
    """A minimal schema-valid item."""
    return {
        "id": "locobench-f001-r1",
        "name": "t",
        "source": "generated:P4/mock",
        "response": "AeroParts supplied blades. No one was harmed. Three people died.",
        "num_atoms": 3,
        "notes": "",
        "atoms": [
            {"id": "a0", "text": "AeroParts supplied blades.", "role": "claim"},
            {"id": "a1", "text": "No one was harmed.", "role": "claim"},
            {"id": "a2", "text": "Three people died.", "role": "claim"},
        ],
        "relations": [
            {
                "source_id": "a1",
                "target_id": "a2",
                "level2_sense": "Alternative",
                "level1_coupling": "exclusive",
                "directed": False,
                "exhaustive": True,
                "validity": "valid",
            }
        ],
        "non_relations": [{"source_id": "a0", "target_id": "a2"}],
        "expected": {
            "family_id": "f001",
            "family": "CONFLICT",
            "rung_index": 1,
            "readout_directions": {
                "mean_marginal": "increase",
                "consistency": "increase",
                "log_partition": "increase",
            },
        },
        "meta": {
            "canonical_topic": "Civil Engineering",
            "domain": "Engineering & Technology",
        },
    }


class TestTopicCoverage:
    """The 36-topic grid and the three-per-topic floor."""

    def test_thirty_six_topics_over_four_domains(self):
        assert len(topics.TOPICS) == 36
        assert set(topics.TOPICS.values()) == set(topics.DOMAINS)

    def test_domain_counts_match_lorefact_table_9(self):
        counts = {d: len(topics.topics_in(d)) for d in topics.DOMAINS}
        assert counts == {
            "Natural Sciences": 11,
            "Humanities": 10,
            "Engineering & Technology": 8,
            "Social Sciences": 7,
        }

    def test_allocate_120_gives_every_topic_at_least_three(self):
        alloc = topics.allocate(120)
        assert sum(alloc.values()) == 120
        assert len(alloc) == 36
        assert min(alloc.values()) >= topics.MIN_FAMILIES_PER_TOPIC

    def test_surplus_goes_to_the_adjudicated_topics(self):
        alloc = topics.allocate(120)
        extra = {t for t, n in alloc.items() if n > topics.MIN_FAMILIES_PER_TOPIC}
        assert extra <= set(topics.SURPLUS_TOPICS)

    def test_allocate_below_the_floor_is_an_error(self):
        # A best-effort allocation would silently drop topics, so this refuses instead.
        with pytest.raises(ValueError, match="below the 3-per-topic floor"):
            topics.allocate(50)

    def test_coverage_report_flags_a_short_corpus(self):
        rep = topics.coverage_report({"Law": 5, "Medicine": 3})
        assert not rep["meets_floor"]
        assert rep["n_topics_below_floor"] == 34

    def test_lorefact_aliases_resolve(self):
        assert topics.canonicalize("Animals") == "Zoology"
        assert topics.domain_of("Weather") == "Natural Sciences"
        with pytest.raises(ValueError, match="Unknown topic"):
            topics.canonicalize("Underwater Basket Weaving")

    def test_slots_cover_all_topics_and_the_right_family_mix(self):
        slots = _slots(_dry_cfg(n_families=120, dry_run=True))
        assert len({t for _, t, _ in slots}) == 36
        types = [f for _, _, f in slots]
        assert {
            t: types.count(t) for t in perturb.FAMILY_TYPES
        } == perturb.FAMILY_COUNTS

    def test_slots_are_deterministic(self):
        a = _slots(_dry_cfg(n_families=120))
        b = _slots(_dry_cfg(n_families=120))
        assert a == b

    def test_limited_slots_still_vary(self):
        # The development loop must exercise more than one topic and one ladder, else a
        # `--limit 4` dry run would test a quarter of the code.
        slots = _slots(_dry_cfg(n_families=120, limit=4))
        assert len({t for _, t, _ in slots}) == 4
        assert len({f for _, _, f in slots}) == 4

    def test_interleave_preserves_totals(self):
        vals = ["a"] * 3 + ["b"] * 2
        out = _interleave(vals)
        assert sorted(out) == sorted(vals)
        assert out[:2] == ["a", "b"]


class TestBuilderAssertion:
    """Gold may not contradict ``lcs.taxonomy.COMPILE``."""

    def test_a_valid_item_passes(self):
        validate_item(_valid_item())

    def test_wrong_coupling_for_sense_is_rejected(self):
        item = _valid_item()
        item["relations"][0]["level1_coupling"] = "contradiction"
        with pytest.raises(SchemaError, match="COMPILE"):
            validate_item(item)

    def test_wrong_directedness_is_rejected(self):
        item = _valid_item()
        item["relations"][0]["directed"] = True  # Alternative is undirected
        with pytest.raises(SchemaError, match="directed"):
            validate_item(item)

    def test_exhaustive_on_a_non_conflict_edge_is_rejected(self):
        item = _valid_item()
        item["relations"] = [
            {
                "source_id": "a0",
                "target_id": "a1",
                "level2_sense": "Cause-Effect",
                "level1_coupling": "entailment",
                "exhaustive": True,
            }
        ]
        with pytest.raises(SchemaError, match="exhaustive"):
            validate_item(item)

    def test_non_canonical_topic_is_rejected(self):
        # Defect 2: a free-text framing in `canonical_topic` would make the coverage
        # constraint unverifiable, so it is refused.
        item = _valid_item()
        item["meta"]["canonical_topic"] = "aviation component failure investigation"
        with pytest.raises(SchemaError, match="not one of"):
            validate_item(item)

    def test_atom_ids_must_be_contiguous(self):
        item = _valid_item()
        item["atoms"][2]["id"] = "a9"
        item["relations"] = []
        item["non_relations"] = []
        with pytest.raises(SchemaError, match="contiguous"):
            validate_item(item)

    def test_a_pair_cannot_be_both_relation_and_non_relation(self):
        item = _valid_item()
        item["non_relations"] = [{"source_id": "a1", "target_id": "a2"}]
        with pytest.raises(SchemaError, match="both a relation and a non-relation"):
            validate_item(item)


class TestDefectOneEncoded:
    """``log_partition`` is invariant at rung 1->2, not increasing."""

    def test_conflict_rung_two_expects_invariance(self):
        dirs = perturb.readout_directions("CONFLICT", 2)
        assert dirs["log_partition"] == "invariant"

    def test_conflict_rung_two_expects_the_consistency_inversion(self):
        # Phase 1's Finding 1, asserted as a positive prediction rather than exempted.
        dirs = perturb.readout_directions("CONFLICT", 2)
        assert dirs["consistency"] == "decrease"

    def test_mean_marginal_is_monotone_across_the_whole_chain(self):
        exp = perturb.expectations_for("CONFLICT")
        assert all(v["mean_marginal"] == "increase" for v in exp.values())

    def test_rung_zero_has_no_directions(self):
        assert perturb.readout_directions("CONFLICT", 0) is None

    def test_control_family_is_flat_everywhere(self):
        exp = perturb.expectations_for("CONTROL")
        assert all(d == "invariant" for pair in exp.values() for d in pair.values())

    def test_the_invariance_reaches_the_generated_item(self):
        res = generate_family("f001", "Civil Engineering", "CONFLICT", _dry_cfg())
        assert res.admitted
        rung2 = next(i for i in res.items if i["expected"]["rung_index"] == 2)
        assert rung2["expected"]["readout_directions"]["log_partition"] == "invariant"

    def test_constraints_split_c1_from_c2(self):
        cons = {c["id"]: c for c in perturb.ordering_constraints("CONFLICT")}
        c2_pairs = {(p["readout"], tuple(p["pair"])) for p in cons["c2"]["pairs"]}
        assert ("log_partition", (1, 2)) in c2_pairs
        assert ("consistency", (1, 2)) in c2_pairs
        c1_pairs = {(p["readout"], tuple(p["pair"])) for p in cons["c1"]["pairs"]}
        assert ("log_partition", (1, 2)) not in c1_pairs


class TestCallBudget:
    """The cost model in Phase 2 Section 7 must stay derived from the real ladders.

    The tables in the document quote these exact numbers, so a ladder edit that changes
    the budget has to fail here rather than silently invalidate the published cost.
    """

    def test_p5_calls_are_not_uniform_across_family_types(self):
        # The trap the hand-derived table fell into: one call per non-base rung would
        # give a flat 4, but a rung may compose several calls.
        per_family = {f: perturb.p5_calls_for(f) for f in perturb.FAMILY_TYPES}
        assert per_family == {"CONFLICT": 7, "CHAIN": 7, "ORDER": 4, "CONTROL": 4}

    def test_documented_corpus_budget(self):
        b = perturb.call_budget(perturb.family_type_slots(120))
        assert b["total"] == 7800
        assert b["generation"] == 1560
        assert b["committee"] == 6240

    def test_documented_per_prompt_budget(self):
        b = perturb.call_budget(perturb.family_type_slots(120))
        assert {k: b[k] for k in ("P1", "P2", "P3", "P4", "P5")} == {
            "P1": 120,
            "P2": 120,
            "P3": 120,
            "P4": 120,
            "P5": 720,
        }
        assert {k: b[k] for k in ("V1", "V2", "V3", "V4")} == {
            "V1": 2520,
            "V2": 1440,
            "V3": 120,
            "V4": 2520,
        }

    def test_v2_is_per_family_not_per_item(self):
        # 3 conflict edges x 4 voters x 120 families -- NOT multiplied by the 5 rungs.
        b = perturb.call_budget(perturb.family_type_slots(120), n_voters=4)
        assert b["V2"] == 3 * 4 * 120

    def test_committee_dominates_generation(self):
        b = perturb.call_budget(perturb.family_type_slots(120))
        assert b["committee"] > 3 * b["generation"]

    def test_closing_the_v3_scope_gap_costs_eighteen_percent(self):
        shipped = perturb.call_budget(perturb.family_type_slots(120))
        complete = perturb.call_budget(
            perturb.family_type_slots(120), inline_responses=5
        )
        assert complete["total"] == 9240
        assert round(complete["total"] / shipped["total"] - 1, 2) == 0.18

    def test_budget_accepts_a_mapping_and_agrees_with_a_list(self):
        from_list = perturb.call_budget(perturb.family_type_slots(120))
        from_map = perturb.call_budget(perturb.FAMILY_COUNTS)
        assert from_list == from_map

    def test_empty_budget_is_zero(self):
        b = perturb.call_budget([])
        assert b["total"] == 0 and b["committee"] == 0

    def test_voter_count_scales_only_the_committee(self):
        two = perturb.call_budget(perturb.FAMILY_COUNTS, n_voters=2)
        four = perturb.call_budget(perturb.FAMILY_COUNTS, n_voters=4)
        assert two["generation"] == four["generation"]
        assert two["committee"] * 2 == four["committee"]


class TestPromptDrift:
    """The harness's prompts must still match Phase 1 Appendix A."""

    @pytest.mark.parametrize("pid", sorted(prompts.PROMPT_SPEC))
    def test_instruction_count_and_placeholders(self, pid):
        want_n, want_ph = prompts.PROMPT_SPEC[pid]
        assert prompts.instruction_count(pid) == want_n
        assert prompts.placeholders(pid) == tuple(sorted(want_ph))

    def test_all_nine_prompts_present(self):
        assert set(prompts.PROMPTS) == set(prompts.GENERATION_PROMPTS) | set(
            prompts.VALIDATION_PROMPTS
        )
        assert len(prompts.PROMPTS) == 9

    def test_fill_requires_every_placeholder(self):
        with pytest.raises(ValueError, match="missing value"):
            prompts.fill("P3", question="q")

    def test_fill_rejects_an_unknown_placeholder(self):
        with pytest.raises(ValueError, match="unexpected value"):
            prompts.fill("P1", topic="t", bogus="x")

    def test_filled_prompt_has_no_leftover_placeholder(self):
        out = prompts.fill("P5", response="r", plan="{}", operator="drop_relation(r1)")
        assert "_PLACEHOLDER]" not in out

    def test_mock_dispatch_probes_are_unambiguous(self):
        # If a Phase-1 prompt edit made a probe match two prompts, the dry run would
        # silently route one stage's call to another's mock.
        _check_probes()
        for pid in prompts.PROMPTS:
            rendered = prompts.PROMPTS[pid]
            assert which_prompt(rendered) == pid


class TestParsers:
    """Every parser returns ``(value, error)`` and never raises."""

    def test_question_round_trip(self):
        q, err = parse.parse_question(mock.mock_question("Law"))
        assert err is None and q.endswith("?")

    def test_claims_round_trip(self):
        claims, err = parse.parse_claims(mock.mock_claims("q"))
        assert err is None
        tags = [c["tag"] for c in claims]
        for tag in ("alt-pair-1", "disj-pair-1", "equiv-pair-1", "holding"):
            assert tag in tags

    def test_plan_round_trip_and_gates(self):
        plan, err = parse.parse_plan(mock.mock_plan("q", "c"))
        assert err is None
        assert validate.gate_plan(plan).passed

    def test_plan_with_a_contradicting_coupling_is_rejected(self):
        plan, _ = parse.parse_plan(mock.mock_plan("q", "c"))
        plan["relations"][0]["coupling"] = "contradiction"  # sense is Cause-Effect
        out, err = parse.parse_plan(json.dumps(plan))
        assert out is None and "COMPILE" in err

    @pytest.mark.parametrize(
        ("fn", "bad"),
        [
            ("parse_question", "no brackets here"),
            ("parse_claims", "not a list"),
            ("parse_plan", "definitely not json"),
            ("parse_response", ""),
            ("parse_perturbation", "no code block"),
            ("parse_verdict", ""),
            ("parse_audit", "{}"),
            ("parse_coverage", "nope"),
        ],
    )
    def test_malformed_input_returns_an_error_and_does_not_raise(self, fn, bad):
        value, err = getattr(parse, fn)(bad)
        assert value is None
        assert isinstance(err, str) and err

    def test_response_below_the_word_floor_is_rejected(self):
        out, err = parse.parse_response("too short", min_words=500)
        assert out is None and "below the 500-word floor" in err


class TestGates:
    """The thresholds, and the reasons a gate gives when it rejects."""

    def test_all_thresholds_live_in_one_table(self):
        for key in ("v1_coupling", "v1_sense", "v3_min_score", "v4_coverage", "window"):
            assert key in validate.THRESHOLDS

    def test_v1_full_recovery_passes_and_partial_fails(self):
        planned = [
            {"source_pos": 1, "target_pos": 2, "sense": "Cause-Effect"},
            {"source_pos": 3, "target_pos": 4, "sense": "Alternative"},
        ]
        full = [
            {
                "source": 1,
                "target": 2,
                "sense": "Cause-Effect",
                "coupling": "entailment",
            },
            {"source": 3, "target": 4, "sense": "Alternative", "coupling": "exclusive"},
        ]
        assert validate.gate_recovery(planned, full).passed
        v = validate.gate_recovery(planned, full[:1])
        assert not v.passed and "recovery too low" in v.reason()

    def test_v3_leakage_fails_with_a_reason(self):
        v = validate.gate_audit(
            {
                "fluency": 5,
                "formality": 5,
                "organization": 5,
                "leakage": ["the relation plan"],
                "hedging": [],
            }
        )
        assert not v.passed and "leakage" in v.reason()

    def test_v4_missing_atom_fails(self):
        v = validate.gate_coverage(
            [{"index": 0, "status": "asserted"}, {"index": 1, "status": "missing"}], 2
        )
        assert not v.passed and "not asserted" in v.reason()

    def test_length_drift_gate(self):
        assert validate.gate_length_drift("w " * 100, "w " * 110).passed
        assert not validate.gate_length_drift("w " * 100, "w " * 200).passed

    def test_generator_is_excluded_from_its_own_committee(self):
        panel = [ModelRef(name=n, model_id=n) for n in ("a-1", "b-1", "c-1", "d-1")]
        voters = validate.committee_for(panel, "b-1")
        assert [m.name for m in voters] == ["a-1", "c-1", "d-1"]

    def test_majority_and_unanimity(self):
        assert validate.majority(["A", "A", "B"])[2] is True
        assert validate.majority(["A", "B"])[2] is False
        assert validate.unanimous(["A", "A"])
        assert not validate.unanimous(["A", "B"])

    def test_agreement_flags_facets_below_their_floor(self):
        rep = validate.agreement_report(
            coupling=[["entailment", "exclusive"], ["exclusive", "entailment"]],
            sense=[["Cause-Effect", "Alternative"], ["Alternative", "Cause-Effect"]],
        )
        assert "coupling" in rep["low_agreement"]

    def test_perfect_agreement_flags_nothing(self):
        rep = validate.agreement_report(
            coupling=[["entailment"] * 3, ["exclusive"] * 3],
            sense=[["Cause-Effect"] * 3, ["Alternative"] * 3],
        )
        assert rep["low_agreement"] == []
        assert rep["kappa_coupling"] == pytest.approx(1.0)

    def test_stratified_sample_prioritizes_the_scarce_facets(self):
        rare = {
            "id": "rare",
            "relations": [{"level1_coupling": "exclusive"}],
        }
        common = {"id": "common", "relations": [{"level1_coupling": "entailment"}]}
        picked = validate.stratified_sample([common, rare], n=1)
        assert picked == ["rare"]


class TestConfig:
    """Everything checkable before the first call is checked at load time."""

    def test_dry_run_needs_no_models(self):
        _dry_cfg().validate()

    def test_live_run_requires_a_generator(self):
        cfg = GenConfig(dry_run=False, generators=[], committee=[])
        with pytest.raises(ValueError, match="at least one generator"):
            cfg.validate()

    def test_committee_too_small_is_refused_up_front(self):
        # With one generator excluded per item, three models leave two voters and no
        # majority. Better to fail now than at item 400.
        cfg = GenConfig(
            dry_run=False,
            generators=[ModelRef(name="g-1", model_id="g")],
            committee=[ModelRef(name=f"m-{i}", model_id="m") for i in range(3)],
        )
        with pytest.raises(ValueError, match="at least 4 are needed"):
            cfg.validate()

    def test_committee_needs_three_families(self):
        cfg = GenConfig(
            dry_run=False,
            generators=[ModelRef(name="g-1", model_id="g")],
            committee=[
                ModelRef(name="a-1", model_id="a", family="x"),
                ModelRef(name="a-2", model_id="a", family="x"),
                ModelRef(name="b-1", model_id="b", family="y"),
                ModelRef(name="b-2", model_id="b", family="y"),
            ],
        )
        with pytest.raises(ValueError, match="at least 3 distinct families"):
            cfg.validate()

    def test_mln_formulation_is_refused(self):
        with pytest.raises(ValueError, match="out of scope"):
            GenConfig(formulation="mln").validate()

    def test_unknown_topic_is_refused(self):
        with pytest.raises(ValueError, match="not one of the 36"):
            GenConfig(only_topics=["Basket Weaving"]).validate()

    def test_unknown_config_key_is_an_error(self):
        with pytest.raises(ValueError, match="Unknown config key"):
            GenConfig.from_dict({"n_familes": 10})  # deliberate typo

    def test_round_trip_through_dict(self):
        cfg = GenConfig(n_families=8, generators=[ModelRef(name="g-1", model_id="g")])
        again = GenConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
        assert again.n_families == 8
        assert again.generators[0].name == "g-1"

    def test_load_config_from_json(self, tmp_path):
        p = tmp_path / "c.json"
        p.write_text(json.dumps({"n_families": 9, "dry_run": True}))
        assert load_config(str(p)).n_families == 9

    def test_load_config_rejects_a_non_object(self, tmp_path):
        p = tmp_path / "c.json"
        p.write_text('"just a string"')
        with pytest.raises(ValueError, match="must contain a JSON/YAML object"):
            load_config(str(p))

    def test_load_config_gives_a_list_its_own_message(self, tmp_path):
        # A list is a model inventory, which is a plausible mistake worth naming --
        # see TestModelInventory.
        p = tmp_path / "c.json"
        p.write_text("[1, 2]")
        with pytest.raises(ValueError, match="model inventory"):
            load_config(str(p))

    def test_model_ref_parse_forms(self):
        assert ModelRef.parse("granite:rits").model_id == "granite"
        assert ModelRef.parse("g:granite-4:rits").model_id == "granite-4"
        assert ModelRef.parse("g:m:vllm:http://x").base_url == "http://x"
        with pytest.raises(ValueError, match="Cannot parse model spec"):
            ModelRef.parse("nocolon")


class TestPipelineAndResume:
    """The stage machine, and the property that makes the harness safe to re-run."""

    @pytest.mark.parametrize("family", list(perturb.FAMILY_TYPES))
    def test_every_ladder_admits_five_items(self, family):
        res = generate_family("f001", "Law", family, _dry_cfg())
        assert res.admitted, res.verdict.reason()
        assert len(res.items) == 5
        assert [i["expected"]["rung_index"] for i in res.items] == [0, 1, 2, 3, 4]

    def test_admitted_items_are_schema_valid(self):
        res = generate_family("f001", "Law", "CONFLICT", _dry_cfg())
        for item in res.items:
            validate_item(item)

    def test_manifest_carries_the_ordering_constraints(self):
        res = generate_family("f001", "Law", "CONFLICT", _dry_cfg())
        ids = {c["id"] for c in res.manifest["ordering_constraints"]}
        assert {"c1", "c2", "c3"} <= ids

    def test_window_admission_is_recorded(self):
        res = generate_family("f001", "Law", "CONFLICT", _dry_cfg())
        for r in res.items[0]["relations"]:
            assert r["window_admission"] in (
                "window",
                "gate",
                "discourse_promoted",
                "out_of_window",
            )

    def test_generation_is_deterministic(self):
        a = generate_family("f001", "Law", "CONFLICT", _dry_cfg())
        b = generate_family("f001", "Law", "CONFLICT", _dry_cfg())
        assert [i["response"] for i in a.items] == [i["response"] for i in b.items]

    def test_resume_does_zero_work_on_a_finished_corpus(self, tmp_path):
        store = Store(str(tmp_path))
        slots = [("f001", "Law", "CONFLICT")]
        todo, _ = store.plan_work(slots, max_attempts=3)
        assert todo == slots

        res = generate_family("f001", "Law", "CONFLICT", _dry_cfg())
        store.append_items(res.items)
        store.put(
            FamilyState(
                "f001",
                "Law",
                "CONFLICT",
                stage="admitted",
                item_ids=[i["id"] for i in res.items],
            )
        )

        again = Store(str(tmp_path))
        todo2, summary = again.plan_work(slots, max_attempts=3)
        assert todo2 == []
        assert summary["done"] == 1
        assert "nothing to do" in again.banner(summary, 1)

    def test_appending_the_same_items_twice_is_idempotent(self, tmp_path):
        store = Store(str(tmp_path))
        res = generate_family("f001", "Law", "CONFLICT", _dry_cfg())
        store.append_items(res.items)
        store.append_items(res.items)
        assert len(store.existing_item_ids()) == 5

    def test_a_rejected_family_is_retried_then_exhausted(self, tmp_path):
        store = Store(str(tmp_path))
        slots = [("f001", "Law", "CONFLICT")]
        store.put(
            FamilyState(
                "f001",
                "Law",
                "CONFLICT",
                stage="respond",
                attempts=1,
                rejected_reason="V3: leakage",
            )
        )
        todo, summary = store.plan_work(slots, max_attempts=3)
        assert todo == slots and summary["retry"] == 1

        store.put(
            FamilyState(
                "f001",
                "Law",
                "CONFLICT",
                stage="respond",
                attempts=3,
                rejected_reason="V3: leakage",
            )
        )
        todo2, summary2 = store.plan_work(slots, max_attempts=3)
        assert todo2 == [] and summary2["exhausted"] == 1

    def test_rejection_records_the_reason(self, tmp_path):
        store = Store(str(tmp_path))
        store.reject(
            "f001",
            {"passed": False, "gates": [], "reason": "V3: leakage"},
            stage="respond",
        )
        assert store.rejected_ids() == ["f001"]
        payload = json.loads((tmp_path / "rejected" / "f001.json").read_text())
        assert payload["reason"] == "V3: leakage"
        assert payload["stage"] == "respond"
        store.clear_rejection("f001")
        assert store.rejected_ids() == []

    def test_a_leaky_response_is_rejected_at_the_v3_gate(self):
        # Drive the failure path the way the harness would see it in the wild.
        cfg = _dry_cfg()
        holder = {"force_leak": True}
        llm = make_mock_llm(cfg, plan_holder=holder)
        res = generate_family("f001", "Law", "CONFLICT", cfg, llm=llm)
        assert not res.admitted
        assert res.stage == "respond"
        assert "leakage" in res.verdict.reason()

    def test_a_missing_atom_is_rejected_at_the_v4_gate(self):
        cfg = _dry_cfg()
        holder = {"force_missing": 2}
        llm = make_mock_llm(cfg, plan_holder=holder)
        res = generate_family("f001", "Law", "CONFLICT", cfg, llm=llm)
        assert not res.admitted
        assert "not asserted" in res.verdict.reason()

    def test_a_dead_llm_rejects_at_the_first_stage_without_raising(self):
        def dead(_rendered):
            raise RuntimeError("backend down")

        res = generate_family("f001", "Law", "CONFLICT", _dry_cfg(), llm=dead)
        assert not res.admitted
        assert res.stage == "plan"
        assert "backend down" in res.verdict.reason()

    def test_resume_reuses_the_stored_plan(self):
        first = generate_family("f001", "Law", "CONFLICT", _dry_cfg())
        assert first.admitted
        second = generate_family(
            "f001", "Law", "CONFLICT", _dry_cfg(), resume_from=first.artifacts
        )
        assert second.admitted
        # P1/P2/P3 are not called again when their artefacts are supplied.
        assert "P1" not in second.calls
        assert "P3" not in second.calls


# =====================================================================================
# The live path. These are the tests whose absence let `build_llm` ship broken.
# =====================================================================================


class _FakeResult:
    """Stands in for Mellea's ``SamplingResult``."""

    def __init__(self, text, success=True):
        self.result = text
        self.success = success


def _fake_live(monkeypatch, canned, *, success=True, record=None):
    """Point ``build_llm`` at a fake backend and a recording ``ainstruct``.

    Args:
        monkeypatch: pytest's fixture.
        canned: ``{prompt_id: text}`` the fake model returns.
        success: What the fake ``SamplingResult`` reports.
        record: Optional dict; receives the last call's kwargs and a ``prompts`` list.

    Returns:
        Nothing; patches in place.
    """
    import mellea.stdlib.functional as mfuncs

    import fact_reasoner.backends as backends

    monkeypatch.setattr(backends, "build_backend", lambda *a, **k: object())

    async def fake_ainstruct(desc, **kw):
        if record is not None:
            record.update(kw)
            record.setdefault("prompts", []).append(desc)
            record.setdefault("loops", []).append(id(asyncio.get_running_loop()))
        pid = which_prompt(desc)
        return _FakeResult(canned.get(pid, "unparseable"), success=success)

    monkeypatch.setattr(mfuncs, "ainstruct", fake_ainstruct)


# What each prompt is contractually required to return. P1 is bracketed, P2--P5 are
# fenced, V1/V4 are BARE JSON lists and V2 is a BARE single letter -- and it is exactly
# those last three that the old tuple-repr bug corrupted, because they have no fence for
# the extractor to anchor on.
_CANNED = {
    "P1": "[What caused the actuator to fail during the test flight?]",
    # 1-based integer atom numbers -- the keys of the mapping V1 is handed. This fixture
    # previously used string "a0"/"a1" ids, a third convention that would silently score as
    # total recovery failure; the parser now rejects it outright.
    "V1": '[{"source": 1, "target": 2, "sense": "Restatement", '
    '"coupling": "equivalence"}]',
    "V2": "A",
    "V3": '{"fluency": 5, "formality": 4, "organization": 5}',
    "V4": '[{"index": 0, "status": "asserted"}]',
}


class TestLivePathReturnsText:
    """``build_llm`` must return the model's text, not a repr of a tuple."""

    @pytest.mark.parametrize("pid", sorted(_CANNED))
    def test_output_is_text_and_parses(self, monkeypatch, pid):
        _fake_live(monkeypatch, _CANNED)
        llm = build_llm(ModelRef("m", "m", "rits"), _dry_cfg())
        out = llm(prompts.PROMPTS[pid])
        # The regression itself: `str(out)` on the (thunk, context) tuple embedded the
        # text inside a repr, which is why fenced prompts survived and bare ones did not.
        assert "ModelOutputThunk" not in out
        assert "SimpleContext" not in out
        assert out == _CANNED[pid]
        assert parse.PARSERS[pid](out)[1] is None

    def test_the_old_tuple_repr_would_have_failed_the_bare_prompts(self):
        # Pins WHY this matters: a repr-wrapped payload breaks exactly V1/V2/V4.
        for pid in ("V1", "V2", "V4"):
            wrapped = (
                f"(ModelOutputThunk({_CANNED[pid]}), <SimpleContext object at 0x1>)"
            )
            assert parse.PARSERS[pid](wrapped)[1] is not None

    def test_a_failed_sampling_result_raises_so_caller_retries(self, monkeypatch):
        _fake_live(monkeypatch, _CANNED, success=False)
        llm = build_llm(ModelRef("m", "m", "rits"), _dry_cfg())
        with pytest.raises(RuntimeError, match="did not satisfy"):
            llm(prompts.PROMPTS["P1"])

    def test_caller_treats_that_raise_as_a_retryable_attempt(self, monkeypatch):
        _fake_live(monkeypatch, _CANNED, success=False)
        llm = build_llm(ModelRef("m", "m", "rits"), _dry_cfg())
        caller = _Caller(llm, attempts=2)
        value, err = caller.ask("P1", parse.parse_question, topic="Law")
        assert value is None
        assert caller.counts["P1"] == 2
        assert "did not satisfy" in err


class TestLivePathCallShape:
    """The Mellea call must use the repo's canonical, validated shape."""

    def test_requests_sampling_results_with_a_parser_requirement(self, monkeypatch):
        rec = {}
        _fake_live(monkeypatch, _CANNED, record=rec)
        build_llm(ModelRef("m", "m", "rits"), _dry_cfg())(prompts.PROMPTS["P1"])
        # Without this the return value is a tuple -- the original bug.
        assert rec["return_sampling_results"] is True
        # The prompt's own parser is the rejection-sampling predicate, which is what
        # substitutes for server-side schema enforcement on backends that lack it.
        assert rec["requirements"]
        assert rec["strategy"] is not None

    def test_an_output_ceiling_is_always_requested(self, monkeypatch):
        """Without it the gateway truncates P4 and the failure is silently misattributed.

        P4 asks for 550-650 words. The IBM gateway grants 4096 completion tokens by
        default and then returns ``finish_reason: "length"`` with the prose stopped
        mid-sentence and no closing fence -- which the parser rejects for being under the
        500-word floor, surfacing as ``P4: SamplingFailed``. That is indistinguishable
        from a model that will not follow instructions, and it killed three of four
        families on one live run. Measured on one prompt: default -> 4096 tokens,
        ``length``, 1114 chars; max 16000 -> ``stop``, 4779 tokens, 4033 chars.
        """
        from mellea.backends.model_options import ModelOption

        rec = {}
        _fake_live(monkeypatch, _CANNED, record=rec)
        build_llm(ModelRef("m", "m", "rits"), _dry_cfg())(prompts.PROMPTS["P4"])
        opts = rec["model_options"]
        assert opts.get(ModelOption.MAX_NEW_TOKENS), (
            "every live call must request an output ceiling; the backend default "
            "truncates P4 mid-sentence and the parser blames the model"
        )
        assert opts[ModelOption.MAX_NEW_TOKENS] > 4096

    def test_a_model_may_override_the_output_ceiling(self, monkeypatch):
        from mellea.backends.model_options import ModelOption

        rec = {}
        _fake_live(monkeypatch, _CANNED, record=rec)
        model = ModelRef("m", "m", "rits", model_options={"max_new_tokens": 999})
        build_llm(model, _dry_cfg())(prompts.PROMPTS["P4"])
        assert rec["model_options"][ModelOption.MAX_NEW_TOKENS] == 999

    def test_the_requirement_names_the_prompt_it_guards(self, monkeypatch):
        rec = {}
        _fake_live(monkeypatch, _CANNED, record=rec)
        build_llm(ModelRef("m", "m", "rits"), _dry_cfg())(prompts.PROMPTS["V2"])
        req = rec["requirements"][0]
        assert req.validation_fn is not None
        # The requirement is built from the prompt the harness recovered, which is what
        # makes the predicate the *right* parser rather than an arbitrary one. Exercising
        # `validation_fn` itself would mean constructing a Mellea Context, so the
        # predicate's behaviour is covered directly below instead.
        assert "V2" in str(req.description)

    @pytest.mark.parametrize(
        "pid,good,bad",
        [
            # "banana" would NOT do here: parse_verdict reads the first character, so it
            # resolves to a valid "B". A rejected verdict has to start with a non-letter
            # or a letter outside A-D.
            ("V2", "A", "zebra"),
            ("P1", "[What caused the actuator to fail in flight?]", "no brackets"),
            ("V4", '[{"index": 0, "status": "asserted"}]', "{}"),
        ],
    )
    def test_the_predicate_accepts_only_what_the_parser_accepts(self, pid, good, bad):
        # The parser IS the acceptance criterion, so rejection sampling and the pipeline's
        # own gate can never disagree.
        parser = parse.PARSERS[pid]
        assert parser(good)[1] is None
        assert parser(bad)[1] is not None

    def test_all_four_backend_arguments_are_forwarded(self, monkeypatch):
        seen = {}
        import mellea.stdlib.functional as mfuncs

        import fact_reasoner.backends as backends

        def spy(kind, **kw):
            seen["kind"] = kind
            seen.update(kw)
            return object()

        monkeypatch.setattr(backends, "build_backend", spy)

        async def ok(desc, **kw):
            return _FakeResult(_CANNED["P1"])

        monkeypatch.setattr(mfuncs, "ainstruct", ok)
        model = ModelRef(
            "m", "mid", "rits", "http://x", api_key="k", model_options={"a": 1}
        )
        build_llm(model, _dry_cfg())
        assert seen["kind"] == "rits"
        assert seen["model_id"] == "mid"
        assert seen["base_url"] == "http://x"
        # Dropped before this change, so a per-model credential was silently ignored.
        assert seen["api_key"] == "k"
        assert seen["model_options"] == {"a": 1}

    def test_one_event_loop_is_reused_across_calls(self, monkeypatch):
        rec = {}
        _fake_live(monkeypatch, _CANNED, record=rec)
        llm = build_llm(ModelRef("m", "m", "rits"), _dry_cfg())
        for _ in range(3):
            llm(prompts.PROMPTS["P1"])
        # Was one fresh loop per prompt, ~7,800 times over a full run.
        assert len(set(rec["loops"])) == 1

    def test_calling_from_inside_a_running_loop_says_so_clearly(self, monkeypatch):
        _fake_live(monkeypatch, _CANNED)
        llm = build_llm(ModelRef("m", "m", "rits"), _dry_cfg())

        async def inner():
            with pytest.raises(RuntimeError, match="running event loop"):
                llm(prompts.PROMPTS["P1"])

        asyncio.run(inner())


class TestRetriesCanSucceed:
    """A retry has to differ from the attempt that failed."""

    def test_temperature_climbs_then_cycles(self):
        cfg = _dry_cfg(retry_temperatures=[0.0, 0.3, 0.7])
        m = ModelRef("m", "m", "rits")
        assert [_temperature_for(m, cfg, i) for i in range(4)] == [0.0, 0.3, 0.7, 0.0]

    def test_attempt_zero_sends_no_temperature_by_default(self):
        # Measured, not stylistic: openai/gpt-oss-120b-a100 on RITS answers P2 at its
        # default temperature and at 0.3, but at 0.0 it returns successfully while
        # emitting no bulleted claims -- so pinning 0.0 for reproducibility made a capable
        # model look incapable. None means "send no temperature at all".
        m = ModelRef("m", "m", "rits")
        assert _temperature_for(m, _dry_cfg(), 0) is None
        assert _temperature_for(m, _dry_cfg(), 1) == 0.3

    def test_a_none_entry_is_omitted_from_model_options(self, monkeypatch):
        rec = {}
        _fake_live(monkeypatch, _CANNED, record=rec)
        llm = build_llm(ModelRef("m", "m", "rits"), _dry_cfg())
        llm(prompts.PROMPTS["P1"], attempt=0)
        # Absent, not present-and-zero: 0.0 is a different (and sometimes unusable)
        # setting from "unset".
        from mellea.backends.model_options import ModelOption

        assert ModelOption.TEMPERATURE not in (rec.get("model_options") or {})
        llm(prompts.PROMPTS["P1"], attempt=1)
        assert rec["model_options"][ModelOption.TEMPERATURE] == 0.3

    def test_a_none_entry_passes_validation(self):
        _dry_cfg(retry_temperatures=[None, 0.5]).validate()

    def test_claude_clamps_but_rits_does_not(self):
        cfg = _dry_cfg(retry_temperatures=[1.8])
        claude = ModelRef(
            "c", "claude-opus-5", "openai", "https://api.anthropic.com/v1/"
        )
        # Anthropic's compatibility endpoint rejects temperature > 1.0.
        assert _temperature_for(claude, cfg, 0) == 1.0
        assert _temperature_for(ModelRef("g", "g", "rits"), cfg, 0) == 1.8

    def test_the_parse_error_is_fed_back_on_retry(self):
        seen = []

        def bad(rendered, *, attempt=0):
            seen.append(rendered)
            return "no brackets here"

        _Caller(bad, attempts=3).ask("P1", parse.parse_question, topic="Law")
        assert len(seen) == 3
        assert seen[0] != seen[1]  # the note is appended after the first failure
        assert "could not be read" in seen[1]
        assert "no bracketed question" in seen[1]

    def test_attempt_index_is_passed_through(self):
        seen = []

        def bad(rendered, *, attempt=0):
            seen.append(attempt)
            return "junk"

        _Caller(bad, attempts=3).ask("P1", parse.parse_question, topic="Law")
        assert seen == [0, 1, 2]

    def test_a_one_argument_callable_is_still_a_valid_llm(self):
        # The seam's value is that a bare function can stand in for a backend; requiring
        # the kwarg would make the harness harder to test, so arity is probed.
        seen = []

        def legacy(rendered):
            seen.append(rendered)
            return "junk"

        _Caller(legacy, attempts=2).ask("P1", parse.parse_question, topic="Law")
        assert len(seen) == 2
        assert "could not be read" in seen[1]

    def test_kwargs_callables_are_detected(self):
        def star(rendered, **kw):
            return "junk"

        assert _Caller._probe_attempt_kwarg(star) is True

    def test_success_short_circuits_the_remaining_attempts(self):
        calls = {"n": 0}

        def flaky(rendered, *, attempt=0):
            calls["n"] += 1
            return "junk" if attempt == 0 else _CANNED["P1"]

        value, err = _Caller(flaky, attempts=3).ask(
            "P1", parse.parse_question, topic="Law"
        )
        assert err is None and value
        assert calls["n"] == 2


class TestRetryNoteIsSafe:
    """The note is appended to a rendered prompt, so it must not disturb dispatch."""

    def test_the_note_contains_no_dispatch_probe(self):
        _check_retry_note()  # raises if it does

    @pytest.mark.parametrize("pid", sorted(prompts.PROMPTS))
    def test_dispatch_survives_the_appended_note(self, pid):
        assert which_prompt(prompts.PROMPTS[pid] + _retry_note("some reason")) == pid

    def test_p5s_parent_response_slice_survives_the_note(self):
        # The mock reconstructs the parent by slicing the rendered prompt; a suffix lands
        # after the RELATION PLAN marker, so the slice must still yield the parent.
        parent = "The actuator failed because of fatigue cracking."
        rendered = (
            f"pick one PERTURBATION to apply\nRESPONSE:\n{parent}\nRELATION PLAN:\n{{}}"
        )

        def sliced(text):
            return (
                text.rsplit("RESPONSE:", 1)[-1].rsplit("RELATION PLAN:", 1)[0].strip()
            )

        assert sliced(rendered) == parent
        assert sliced(rendered + _retry_note("bad json")) == parent


class TestCapabilities:
    """Capabilities are derived from the endpoint, never configured."""

    def test_claude_over_the_compat_endpoint(self):
        cap = ModelRef(
            "c", "claude-opus-5", "openai", "https://api.anthropic.com/v1/"
        ).capabilities()
        assert cap == Capabilities(
            schema_enforced=False, temperature_range=(0.0, 1.0), supports_seed=False
        )

    def test_rits_enforces_schemas(self):
        assert ModelRef("g", "g", "rits").capabilities().schema_enforced is True

    def test_a_lookalike_host_is_not_treated_as_anthropic(self):
        cap = ModelRef(
            "e", "x", "openai", "https://api.anthropic.com.evil.org/v1/"
        ).capabilities()
        assert cap.temperature_range == (0.0, 2.0)
        assert cap.supports_seed is True

    def test_openai_base_url_env_is_honoured(self, monkeypatch):
        # build_backend falls back to this variable, so capabilities must too or a run
        # would be reported as schema-enforced when it is not.
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.anthropic.com/v1/")
        assert ModelRef("c", "gpt", "openai").capabilities().supports_seed is False


class TestCredentialsAreNotSerialized:
    """The config is written to the corpus directory as provenance."""

    def test_api_key_is_absent_from_the_serialized_config(self):
        cfg = GenConfig(
            generators=[ModelRef("m", "m", "rits", api_key="sk-SECRET")],
            committee=[ModelRef("c", "c", "rits", api_key="sk-ALSO-SECRET")],
            auditor=ModelRef("a", "a", "rits", api_key="sk-THIRD"),
        )
        blob = json.dumps(cfg.to_dict())
        assert "api_key" not in blob
        assert "SECRET" not in blob
        assert "sk-" not in blob

    def test_the_rest_of_the_model_survives(self):
        d = ModelRef("m", "mid", "rits", "http://x", api_key="k").to_dict()
        assert d["name"] == "m" and d["model_id"] == "mid"
        assert d["backend"] == "rits" and d["base_url"] == "http://x"

    def test_a_config_round_trip_still_works(self):
        cfg = GenConfig(generators=[ModelRef("m", "m", "rits")])
        assert GenConfig.from_dict(cfg.to_dict()).generators[0].name == "m"


class TestBackendValidation:
    """A typo must be caught at load time, not hours into a run."""

    def test_unknown_kind_is_rejected_even_on_a_dry_run(self):
        cfg = _dry_cfg(generators=[ModelRef("x", "m", "rist")])
        with pytest.raises(ValueError, match="not one of"):
            cfg.validate()

    def test_claude_on_openai_without_an_endpoint_is_rejected(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        cfg = GenConfig(
            generators=[ModelRef("c", "claude-opus-5", "openai")],
            committee=[ModelRef(n, n, "rits") for n in ("a-1", "b-2", "c-3", "d-4")],
        )
        with pytest.raises(ValueError, match="compatibility endpoint"):
            cfg.validate()

    def test_the_documented_two_backend_run_validates(self):
        cfg = GenConfig(
            generators=[
                ModelRef(
                    "claude", "claude-opus-5", "openai", "https://api.anthropic.com/v1/"
                ),
                ModelRef("granite", "granite-4-1-30b", "rits"),
            ],
            committee=[ModelRef(n, n, "rits") for n in ("a-1", "b-2", "c-3", "d-4")],
        )
        cfg.validate()

    def test_an_empty_retry_ladder_is_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            _dry_cfg(retry_temperatures=[]).validate()


class TestV3AuditorExclusion:
    """R3: the model that ran P3/P4 must not audit its own prose.

    `resolved_auditor` existed, was documented as "the single model that runs V3", and had
    zero callers -- so V3 ran on the generator's own caller. Measured consequence: on
    identical prose with the identical narrowed prompt, a self-auditing deepseek flagged 5
    coordinations the prompt names verbatim as exempt, where a distinct auditor flagged 1
    real span naming a sense. A weak self-auditor rejects items for compliance.
    """

    def _cfg(self, **kw):
        return GenConfig(
            generators=[ModelRef("gen", "vendor/model-a", "rits")],
            committee=[
                # A legitimate config shape: the same underlying model listed under a
                # different label for the agreement statistics.
                ModelRef("alias-of-gen", "vendor/model-a", "rits"),
                ModelRef("other", "vendor/model-b", "rits"),
            ],
            **kw,
        )

    def test_exclusion_is_by_model_id_not_by_name(self):
        cfg = self._cfg()
        gen = cfg.generators[0]
        aud = cfg.resolved_auditor(gen)
        assert aud is not None
        # A by-name check would have returned "alias-of-gen" and self-audited in silence.
        assert aud.name == "other"
        assert aud.model_id != gen.model_id

    def test_a_configured_auditor_matching_the_generator_is_refused(self):
        cfg = self._cfg(auditor=ModelRef("explicit", "vendor/model-a", "rits"))
        # Same model id as the generator, so it is not an independent judgment.
        assert cfg.resolved_auditor(cfg.generators[0]) is None

    def test_no_eligible_committee_member_yields_none(self):
        cfg = GenConfig(
            generators=[ModelRef("gen", "vendor/model-a", "rits")],
            committee=[ModelRef("alias", "vendor/model-a", "rits")],
        )
        # None rather than a raise: the caller decides to warn and degrade, because a
        # self-audit is a weaker result and not an invalid one.
        assert cfg.resolved_auditor(cfg.generators[0]) is None

    def test_v3_runs_on_the_auditor_when_one_is_given(self, monkeypatch):
        seen = []

        def _mk(tag):
            def _llm(rendered, *, attempt=0):
                seen.append((tag, which_prompt(rendered)))
                return _CANNED[which_prompt(rendered)]

            return _llm

        v = pipeline._validate_response(
            _Caller(_mk("gen"), attempts=1),
            "prose",
            {"atoms": [], "relations": []},
            [],
            auditors=[("aud", _Caller(_mk("aud"), attempts=1))],
        )
        assert v is not None
        by_prompt = {pid: tag for tag, pid in seen}
        assert by_prompt.get("V3") == "aud", "V3 must run on the auditor"
        assert by_prompt.get("V1") == "gen", "V1 stays on the generator's caller"
        assert by_prompt.get("V4") == "gen"

    def test_every_panel_member_audits(self):
        seen = []

        def _mk(tag):
            def _llm(rendered, *, attempt=0):
                pid = which_prompt(rendered)
                if pid == "V3":
                    seen.append(tag)
                return _CANNED[pid]

            return _llm

        pipeline._validate_response(
            _Caller(_mk("gen"), attempts=1),
            "prose",
            {"atoms": [], "relations": []},
            [],
            auditors=[
                ("a", _Caller(_mk("a"), attempts=1)),
                ("b", _Caller(_mk("b"), attempts=1)),
                ("c", _Caller(_mk("c"), attempts=1)),
            ],
        )
        assert sorted(seen) == ["a", "b", "c"]

    def test_the_panel_dedupes_by_model_id(self):
        # Two labels for one model would let it vote twice and defeat the majority.
        cfg = GenConfig(
            generators=[ModelRef("gen", "vendor/model-a", "rits")],
            committee=[
                ModelRef("dup1", "vendor/model-b", "rits"),
                ModelRef("dup2", "vendor/model-b", "rits"),
                ModelRef("other", "vendor/model-c", "rits"),
            ],
        )
        ids = [m.model_id for m in cfg.eligible_auditors(cfg.generators[0])]
        assert ids == ["vendor/model-b", "vendor/model-c"]

    def test_an_explicit_auditor_overrides_the_panel(self):
        cfg = GenConfig(
            generators=[ModelRef("gen", "vendor/model-a", "rits")],
            committee=[ModelRef("x", "vendor/model-b", "rits")],
            auditor=ModelRef("chosen", "vendor/model-z", "rits"),
        )
        panel = cfg.eligible_auditors(cfg.generators[0])
        assert [m.name for m in panel] == ["chosen"]

    def test_v3_falls_back_to_the_generator_when_no_auditor(self, monkeypatch):
        seen = []

        def _llm(rendered, *, attempt=0):
            seen.append(which_prompt(rendered))
            return _CANNED[which_prompt(rendered)]

        pipeline._validate_response(
            _Caller(_llm, attempts=1), "prose", {"atoms": [], "relations": []}, []
        )
        # Degrades rather than crashing: still audits, just not independently.
        assert "V3" in seen


class TestV3MajorityVote:
    """A single V3 rater must not decide admission; its judgments diverge too much.

    Measured on one response, identical prose and identical prompt: opus-5 found 0 leakage
    spans, sonnet-4-6 found **5**, opus-4-8 found 0, opus-4-7 found 0. The harness had been
    resolving one auditor and picking whichever model sat first in committee order, so
    admission turned on that accident -- and the arbitrary pick was the lone outlier. All
    four agreed on one hedging span, which is what a true positive looks like here.
    """

    def _a(self, **kw):
        base = {
            "fluency": 5,
            "formality": 5,
            "organization": 4,
            "leakage": [],
            "hedging": [],
            "artifacts": [],
        }
        base.update(kw)
        return base

    def test_a_lone_dissenter_does_not_reject(self):
        clean = self._a()
        outlier = self._a(leakage=["at least one of two diagnostics"])
        v = validate.gate_audit_panel(
            [("opus-5", clean), ("sonnet-4-6", outlier), ("opus-4-8", clean)]
        )
        spans = next(g for g in v.results if g.gate == "V3.spans")
        assert spans.passed
        # The dissent is still recorded -- suppressing it would hide a real disagreement.
        assert spans.observed["n_flagging_per_kind"]["leakage"] == 1
        assert "sonnet-4-6" in spans.observed["spans"]

    def test_a_majority_still_rejects(self):
        leaky = self._a(leakage=["naming the sense Concession"])
        v = validate.gate_audit_panel([("a", leaky), ("b", leaky), ("c", self._a())])
        assert not next(g for g in v.results if g.gate == "V3.spans").passed

    def test_span_kinds_are_voted_separately(self):
        # The measured shape: leakage split 1-of-3, hedging unanimous. Pooling them would
        # let the agreed hedge carry the disputed leakage into the rejection reason.
        hedged = self._a(hedging=["might seem"])
        both = self._a(leakage=["at least one of"], hedging=["might seem"])
        v = validate.gate_audit_panel([("a", hedged), ("b", both), ("c", hedged)])
        spans = next(g for g in v.results if g.gate == "V3.spans")
        assert not spans.passed, (
            "the unanimous hedge is a true positive and must reject"
        )
        per_kind = spans.observed["n_flagging_per_kind"]
        assert per_kind == {"leakage": 1, "hedging": 3}
        # Only hedging is named as the REJECTING facet; leakage still appears in the
        # per-auditor vote dump, which is the transparency the votes exist for.
        rejecting = spans.detail.split(":")[0]
        assert "hedging" in rejecting
        assert "leakage" not in rejecting

    def test_scores_are_voted_too(self):
        low = self._a(organization=3)
        # One low score out of three does not reject -- which is the boundary-flapping
        # case: repeat audits of one family scored organization [3, 3, 4].
        assert validate.gate_audit_panel(
            [("a", low), ("b", self._a()), ("c", self._a())]
        ).passed
        assert not validate.gate_audit_panel(
            [("a", low), ("b", low), ("c", self._a())]
        ).passed

    def test_a_single_auditor_behaves_like_the_old_gate(self):
        assert validate.gate_audit_panel([("solo", self._a())]).passed
        assert not validate.gate_audit_panel(
            [("solo", self._a(leakage=["the relation plan"]))]
        ).passed

    def test_no_audits_is_a_harness_failure_not_a_verdict(self):
        v = validate.gate_audit_panel([])
        assert not v.passed
        assert "no audit output" in v.results[0].detail

    def test_the_rule_is_declared_in_thresholds(self):
        assert validate.THRESHOLDS["v3_rule"] == "majority"


class TestP4BackReferencing:
    """P4 must not narrate its own earlier sentences as plan steps.

    Measured live: a Claude family was rejected for leakage on "the superpositional
    ordering just described", "the general downweighting provision just stated" and "the
    outlier flag just reported". Those are the writer walking the plan item by item, which
    instruction 6 already forbade in the abstract -- the auditor was right, so the fix is
    in P4 rather than in V3's exemption list.
    """

    def test_instruction_six_names_the_pattern_concretely(self):
        p4 = prompts.PROMPTS["P4"]
        assert "just reported" in p4
        assert "just stated" in p4
        assert "never" in p4

    def test_the_guidance_is_phrased_as_a_substitution(self):
        # Prohibitions measurably cost output length (see
        # TestV3LeakageSemantics.test_p4_does_not_widen_the_hedge_scope), so this names an
        # acceptable alternative rather than adding another "do NOT" clause.
        p4 = prompts.PROMPTS["P4"]
        assert "Restate a subject to carry an argument forward" in p4

    def test_the_instruction_count_is_unchanged(self):
        import re

        assert len(re.findall(r"(?m)^\d+\. ", prompts.PROMPTS["P4"])) == 8


class TestGeneratorBuildFailures:
    """R3: the rotation carries the no-single-author claim, so a gap aborts the run."""

    def _cfg(self):
        return GenConfig(
            generators=[
                ModelRef(
                    "claude", "claude-opus-5", "openai", "https://api.anthropic.com/v1/"
                ),
                ModelRef("granite", "granite-4-1-30b", "rits"),
            ]
        )

    def test_every_failure_is_reported_not_just_the_first(self, monkeypatch, capsys):
        import fact_reasoner.backends as backends

        def dead(kind, **kw):
            raise RuntimeError(f"no creds for {kind}")

        monkeypatch.setattr(backends, "build_backend", dead)
        with pytest.raises(SystemExit) as e:
            _build_generators(self._cfg())
        msg = str(e.value)
        assert "claude" in msg and "granite" in msg
        assert "2 of 2" in msg

    def test_one_bad_generator_still_aborts(self, monkeypatch):
        import fact_reasoner.backends as backends

        def half(kind, **kw):
            if kind == "rits":
                raise RuntimeError("RITS_API_KEY unset")
            return object()

        monkeypatch.setattr(backends, "build_backend", half)
        with pytest.raises(SystemExit, match="1 of 2"):
            _build_generators(self._cfg())

    def test_duplicate_generator_names_are_rejected(self):
        cfg = GenConfig(
            generators=[ModelRef("same", "a", "rits"), ModelRef("same", "b", "rits")]
        )
        with pytest.raises(SystemExit, match="duplicate generator name"):
            _build_generators(cfg)

    def test_the_capability_posture_is_printed_before_any_work(
        self, monkeypatch, capsys
    ):
        import fact_reasoner.backends as backends

        monkeypatch.setattr(backends, "build_backend", lambda *a, **k: object())
        _build_generators(self._cfg())
        out = capsys.readouterr().out
        assert "schema_enforced=False" in out  # claude
        assert "schema_enforced=True" in out  # rits
        assert "rejection-samples against the real parsers" in out


@pytest.mark.skipif(
    not os.getenv("LOCOBENCH_LIVE"),
    reason="live smoke test; set LOCOBENCH_LIVE=1 and configure credentials",
)
class TestLiveSmoke:
    """Opt-in live checks. The thing to run once before committing to a long run.

    Verified 2026-08-01 against RITS (``llama-3-3-70b-instruct``,
    ``openai/gpt-oss-120b-a100``) and Claude via an OpenAI-protocol gateway
    (``aws/claude-opus-5``).

    Two things the default model choice encodes. ``granite-3-3-8b-instruct`` cannot satisfy
    P2 -- it tags claims correctly but never emits the mandated alt/disj/equiv pairs -- so a
    smaller generator fails at the plan stage for reasons about the model, not the harness.
    And ``gpt-oss-120b-a100`` *can*, but takes ~135 s per P2 attempt, which multiplies
    against the rejection-sampling loop; hence a 70B default rather than a reasoning model.

    Override with ``LOCOBENCH_LIVE_MODEL``. For a gateway whose credential is not in
    ``OPENAI_API_KEY``, also set ``LOCOBENCH_LIVE_KEY`` (argv would leak it).
    """

    def _model(self):
        spec = os.environ.get("LOCOBENCH_LIVE_MODEL", "llama-3-3-70b-instruct:rits")
        model = ModelRef.parse(spec if ":" in spec else f"{spec}:rits")
        key = os.environ.get("LOCOBENCH_LIVE_KEY")
        if key:
            model = ModelRef(
                name=model.name,
                model_id=model.model_id,
                backend=model.backend,
                base_url=model.base_url,
                api_key=key,
            )
        return model

    def test_the_live_callable_returns_parseable_text(self):
        # The narrowest possible live assertion, and the one that pins the original bug:
        # real text out of a real endpoint, not a repr of a (thunk, context) tuple.
        model = self._model()
        cfg = GenConfig(n_families=1, generators=[model], dry_run=False)
        llm = build_llm(model, cfg)
        out = llm(prompts.fill("P1", topic="Aerospace Engineering"))
        assert "ModelOutputThunk" not in out
        assert parse.parse_question(out)[1] is None, out[:200]

    def test_a_family_runs_end_to_end(self):
        model = self._model()
        cfg = GenConfig(n_families=1, limit=1, generators=[model], dry_run=False)
        llm = build_llm(model, cfg)
        res = generate_family(
            "f001", "Law", "CONFLICT", cfg, llm=llm, generator=model.name
        )
        # Deliberately NOT `assert res.admitted`: a live model can fail a *semantic* gate
        # (the 55/45 validity split, the window constraint) while every prompt round-trips
        # perfectly, and that is the harness working. What must hold is that the pipeline
        # reached the model and got usable output for each prompt it issued -- checked per
        # prompt, because P1--P5 masked the old tuple bug by surviving it.
        assert res.calls, "no prompt was ever issued"
        assert "P1" in res.calls
        if res.admitted:
            for pid in ("P2", "P3", "P4", "V1", "V3", "V4"):
                assert pid in res.calls, f"{pid} was never called"
        else:
            # A rejection must name a gate, not an unparseable payload.
            reason = res.verdict.reason()
            assert "ModelOutputThunk" not in reason, reason


class TestModelInventory:
    """A shared model list (``configs/rits_models.json``) is not a run config."""

    def _inventory(self, tmp_path):
        p = tmp_path / "models.json"
        p.write_text(
            json.dumps(
                [
                    {"name": "a-one", "model_id": "m1", "backend": "rits"},
                    {
                        "name": "b-two",
                        "model_id": "openai/m2",
                        "backend": "rits",
                        "base_url": "https://example.invalid/m2",
                    },
                ]
            )
        )
        return p

    def test_load_models_reads_a_bare_list(self, tmp_path):
        models = load_models(str(self._inventory(tmp_path)))
        assert [m.name for m in models] == ["a-one", "b-two"]
        assert models[1].base_url == "https://example.invalid/m2"

    def test_the_shipped_inventory_loads(self):
        # The real file, so a field added to it without updating ModelRef fails here.
        models = load_models("configs/rits_models.json")
        assert models
        for m in models:
            assert m.backend in ("rits", "ollama", "vllm", "openai")
            # Every entry gives an explicit endpoint, so model_id must be the raw served
            # id -- build_backend's custom-endpoint branch does not consult the catalog.
            assert m.base_url

    def test_a_list_passed_to_load_config_names_the_fix(self, tmp_path):
        with pytest.raises(ValueError, match="model inventory"):
            load_config(str(self._inventory(tmp_path)))

    def test_unknown_fields_are_rejected(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps([{"name": "x", "model_id": "m", "bogus": 1}]))
        with pytest.raises(ValueError, match="unknown field"):
            load_models(str(p))

    def test_duplicate_names_are_rejected(self, tmp_path):
        p = tmp_path / "dupe.json"
        p.write_text(
            json.dumps(
                [
                    {"name": "same", "model_id": "a", "backend": "rits"},
                    {"name": "same", "model_id": "b", "backend": "rits"},
                ]
            )
        )
        with pytest.raises(ValueError, match="duplicate model name"):
            load_models(str(p))

    def test_a_config_can_select_inventory_models_by_name(self, tmp_path):
        inv = self._inventory(tmp_path)
        cfg_path = tmp_path / "run.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "models_file": str(inv),
                    "generators": ["b-two"],
                    "n_families": 4,
                    "dry_run": True,
                }
            )
        )
        cfg = load_config(str(cfg_path))
        assert [m.name for m in cfg.generators] == ["b-two"]
        assert cfg.generators[0].model_id == "openai/m2"
        cfg.validate()

    def test_a_misspelled_selection_lists_the_alternatives(self, tmp_path):
        inv = self._inventory(tmp_path)
        cfg_path = tmp_path / "run.json"
        cfg_path.write_text(
            json.dumps({"models_file": str(inv), "generators": ["b-twoo"]})
        )
        with pytest.raises(ValueError, match="not in.*available"):
            load_config(str(cfg_path))

    def test_inline_models_still_work_alongside_a_models_file(self, tmp_path):
        inv = self._inventory(tmp_path)
        cfg_path = tmp_path / "run.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "models_file": str(inv),
                    "generators": [
                        "a-one",
                        {"name": "inline", "model_id": "m", "backend": "rits"},
                    ],
                    "dry_run": True,
                }
            )
        )
        cfg = load_config(str(cfg_path))
        assert [m.name for m in cfg.generators] == ["a-one", "inline"]


class TestDatasetName:
    """The item id prefix, which is what distinguishes merged corpora."""

    def test_default_prefix_is_unchanged(self):
        res = generate_family("f001", "Law", "CONFLICT", _dry_cfg())
        assert res.admitted
        assert res.items[0]["id"] == "locobench-f001-r0"

    def test_the_name_tags_every_item_id(self):
        cfg = _dry_cfg(dataset_name="locobench-deepseek-v3.2")
        res = generate_family("f001", "Law", "CONFLICT", cfg)
        assert res.admitted
        assert [i["id"] for i in res.items] == [
            f"locobench-deepseek-v3.2-f001-r{i}" for i in range(5)
        ]

    def test_the_manifest_agrees_with_the_item_ids(self):
        # A mismatch would silently break the family->item join.
        cfg = _dry_cfg(dataset_name="ds-x")
        res = generate_family("f001", "Law", "CONFLICT", cfg)
        assert res.manifest["dataset"] == "ds-x"
        assert [r["item_id"] for r in res.manifest["rungs"]] == [
            i["id"] for i in res.items
        ]

    def test_the_generator_is_still_recorded_separately(self):
        # dataset_name is a label; provenance must not depend on it being set correctly.
        cfg = _dry_cfg(dataset_name="ds-x")
        res = generate_family("f001", "Law", "CONFLICT", cfg, generator="deepseek-v3.2")
        assert res.items[0]["source"] == "generated:P4/deepseek-v3.2"
        assert res.manifest["generator"] == "deepseek-v3.2"

    @pytest.mark.parametrize("bad", ["", "has space", "slash/es", "quote'", "..#"])
    def test_unsafe_names_are_rejected(self, bad):
        with pytest.raises(ValueError, match="dataset_name"):
            _dry_cfg(dataset_name=bad).validate()

    def test_dots_and_hyphens_are_allowed(self):
        # 'locobench-deepseek-v3.2' is the motivating case.
        _dry_cfg(dataset_name="locobench-deepseek-v3.2").validate()

    def test_items_still_validate_against_the_schema(self):
        cfg = _dry_cfg(dataset_name="locobench-deepseek-v3.2")
        res = generate_family("f001", "Law", "CONFLICT", cfg)
        for item in res.items:
            validate_item(item)


class TestP3Gates:
    """Negative-path coverage for the three plan gates.

    Before this class every gate had only the mock's positive path, so a gate could be
    loosened or broken outright with the suite still green. Two must keep biting; one is
    deliberately an observation.
    """

    def _plan(self):
        plan, err = parse.parse_plan(mock.mock_plan("q", "c", 0))
        assert err is None
        return plan

    def _no_shared_entities(self, plan, *positions):
        """Strip capitalized tokens so `_shares_entity` cannot admit the pair."""
        for p in positions:
            plan["atoms"][p - 1]["text"] = f"a lowercase claim numbered {p}"
        return plan

    def test_the_mock_plan_is_still_the_positive_witness(self):
        assert validate.gate_plan(self._plan()).passed

    def test_a_far_edge_is_recorded_but_does_not_block(self):
        # Phase 1 R2 gives instruction 4 the weakest role ("biases generation"); the
        # authoritative check runs on realized text at build time. Blocking here was the
        # single largest cause of plan-stage rejection on live runs.
        plan = self._no_shared_entities(self._plan(), 1, 16)
        plan["relations"][0]["source_pos"] = 1
        plan["relations"][0]["target_pos"] = 16
        v = validate.gate_plan(plan)
        window = next(r for r in v.results if r.gate == "plan.window")
        assert window.passed, "the window gate must not block"
        assert window.observed == [(1, 16, 15)], "but it must still report the far edge"
        assert v.passed, "and the plan as a whole must be admissible"

    def test_the_validity_split_still_blocks(self):
        plan = self._plan()
        for r in plan["relations"]:
            r["validity"] = "valid"
        v = validate.gate_plan(plan)
        assert not v.passed
        assert "validity_split" in v.reason()

    def test_the_required_senses_still_block(self):
        plan = self._plan()
        for r in plan["relations"]:
            if r["sense"] == "Restatement":
                r["sense"] = "Evidence"
        v = validate.gate_plan(plan)
        assert not v.passed
        assert "Restatement" in v.reason()

    def test_window_threshold_key_survives(self):
        # `annotate_window_admission` is now its only consumer, and
        # test_all_thresholds_live_in_one_table asserts the key exists.
        assert validate.THRESHOLDS["window"] == 4


class TestP3PromptTeachesAConformingAnswer:
    """The worked example in instruction 10 must itself pass, or it teaches failure."""

    def test_the_example_parses_and_passes_every_gate(self):
        plan, err = parse.parse_plan(prompts.PROMPTS["P3"])
        assert err is None, err
        v = validate.gate_plan(plan)
        assert v.passed, v.reason()

    def test_the_example_matches_what_the_instructions_ask_for(self):
        plan, _ = parse.parse_plan(prompts.PROMPTS["P3"])
        rels = plan["relations"]
        n_valid = sum(1 for r in rels if r["validity"] == "valid")
        # Instruction 3 says exactly 10; instruction 8 says exactly 6 valid / 4 invalid.
        assert len(rels) == 10
        assert n_valid == 6
        assert n_valid / len(rels) == pytest.approx(0.60)

    def test_the_example_positions_are_contiguous_from_one(self):
        # The rule instruction 2 now spells out, and the likeliest cause of the parser
        # failures it is meant to prevent.
        plan, _ = parse.parse_plan(prompts.PROMPTS["P3"])
        positions = [a["pos"] for a in plan["atoms"]]
        assert positions == list(range(1, len(positions) + 1))

    def test_the_example_covers_every_required_sense(self):
        plan, _ = parse.parse_plan(prompts.PROMPTS["P3"])
        senses = {r["sense"] for r in plan["relations"]}
        from fact_reasoner.locobench.taxonomy_bridge import (
            REQUIRED_EITHER,
            REQUIRED_SENSES,
        )

        assert set(REQUIRED_SENSES) <= senses
        assert senses & set(REQUIRED_EITHER)

    def test_the_example_non_relations_are_disjoint_from_the_relations(self):
        plan, _ = parse.parse_plan(prompts.PROMPTS["P3"])
        rels = {(r["source_pos"], r["target_pos"]) for r in plan["relations"]}
        for nr in plan["non_relations"]:
            pair = (nr["source_pos"], nr["target_pos"])
            assert pair not in rels and pair[::-1] not in rels

    def test_the_drift_guards_still_hold(self):
        # The couplings that would silently break the whole dry-run suite.
        assert prompts.instruction_count("P3") == 10
        assert prompts.placeholders("P3") == (
            "CLAIMS_PLACEHOLDER",
            "QUESTION_PLACEHOLDER",
        )
        _check_probes()
        assert which_prompt(prompts.PROMPTS["P3"]) == "P3"


class TestGateFailuresAreRepairable:
    """A gate complaint is retryable feedback, and a near miss is worth keeping."""

    def test_a_semantic_check_failure_consumes_attempts(self):
        # Previously the gate ran outside the retry loop, so a parseable-but-rejected
        # plan burned zero attempts.
        calls = []

        def llm(rendered, *, attempt=0):
            calls.append(attempt)
            return mock.mock_plan("q", "c", 0)

        caller = _Caller(llm, attempts=3)
        value, err = caller.ask(
            "P3",
            parse.parse_plan,
            check=lambda _p: "always unhappy",
            question="q",
            claims="- a [correct]",
        )
        assert calls == [0, 1, 2], "the check must be retried, not evaluated once"
        assert err == "always unhappy"
        assert value is not None, "the near miss must still be returned, not discarded"

    def test_the_check_reason_is_fed_back_to_the_model(self):
        seen = []

        def llm(rendered, *, attempt=0):
            seen.append(rendered)
            return mock.mock_plan("q", "c", 0)

        _Caller(llm, attempts=2).ask(
            "P3",
            parse.parse_plan,
            check=lambda _p: "only 4 of 10 relations are valid",
            question="q",
            claims="- a [correct]",
        )
        assert "only 4 of 10 relations are valid" in seen[1]
        assert "could not be read" in seen[1]

    def test_a_passing_check_short_circuits(self):
        calls = []

        def llm(rendered, *, attempt=0):
            calls.append(attempt)
            return mock.mock_plan("q", "c", 0)

        value, err = _Caller(llm, attempts=3).ask(
            "P3",
            parse.parse_plan,
            check=lambda _p: None,
            question="q",
            claims="- a [correct]",
        )
        assert err is None and value is not None
        assert calls == [0]

    def test_callers_without_a_check_are_unaffected(self):
        # Every other prompt passes no `check` and must behave exactly as before.
        value, err = _Caller(lambda r, **kw: "junk", attempts=2).ask(
            "P1", parse.parse_question, topic="Law"
        )
        assert value is None and err

    def test_a_rejected_plan_is_persisted_for_offline_analysis(self):
        # The artefact that makes a gate change evaluable without spending API calls.
        cfg = _dry_cfg()

        def llm(rendered, *, attempt=0):
            pid = which_prompt(rendered)
            if pid == "P3":
                plan, _ = parse.parse_plan(mock.mock_plan("q", "c", 0))
                for r in plan["relations"]:  # force a validity-split failure
                    r["validity"] = "valid"
                return "```json\n" + json.dumps(plan) + "\n```"
            return make_mock_llm(cfg)(rendered)

        res = generate_family("f001", "Law", "CONFLICT", cfg, llm=llm)
        assert not res.admitted
        assert res.stage == "plan"
        assert "rejected_plan" in res.artifacts, "the near miss must be kept"
        assert res.artifacts["rejected_plan"]["relations"]


class TestRespondStageArtifacts:
    """A respond-stage rejection must keep the prose the validators judged.

    Without this the verdict names a number -- "recovery 0.00", "259 words below the
    500-word floor" -- with no text behind it, so the failure cannot be diagnosed and
    invites guessing at the cause.
    """

    def test_a_p4_parser_failure_keeps_the_raw_completion(self):
        cfg = _dry_cfg()
        base = make_mock_llm(cfg)

        def short_p4(rendered, *, attempt=0):
            if which_prompt(rendered) == "P4":
                return "```\ntoo short\n```"
            return base(rendered)

        res = generate_family("f001", "Law", "CONFLICT", cfg, llm=short_p4)
        assert not res.admitted
        assert res.stage == "respond"
        # `parse_response` returns None on failure, so the prose is reachable only via the
        # caller's raw record.
        assert "rejected_response_raw" in res.artifacts
        assert "too short" in res.artifacts["rejected_response_raw"]

    def test_a_validator_failure_keeps_the_response(self):
        cfg = _dry_cfg()
        holder: dict = {}
        base = make_mock_llm(cfg, plan_holder=holder)

        def bad_v1(rendered, *, attempt=0):
            if which_prompt(rendered) == "V1":
                return "[]"  # recovered nothing -> coupling recall 0.00
            return base(rendered)

        res = generate_family("f001", "Law", "CONFLICT", cfg, llm=bad_v1)
        assert not res.admitted
        assert res.stage == "respond"
        assert "recovery too low" in res.verdict.reason()
        kept = res.artifacts.get("rejected_response")
        assert kept, "the prose V1 judged must be persisted"
        assert len(kept.split()) > 100

    def test_the_rejected_response_does_not_resume_as_the_response(self):
        # Stored under `rejected_response`, not `response`, so a retry regenerates rather
        # than resuming from prose that already failed.
        cfg = _dry_cfg()
        base = make_mock_llm(cfg)

        def bad_v1(rendered, *, attempt=0):
            return "[]" if which_prompt(rendered) == "V1" else base(rendered)

        res = generate_family("f001", "Law", "CONFLICT", cfg, llm=bad_v1)
        assert "response" not in res.artifacts

    def test_the_caller_records_the_last_raw_completion_per_prompt(self):
        caller = _Caller(lambda r, **kw: "unparseable junk", attempts=2)
        caller.ask("P1", parse.parse_question, topic="Law")
        assert caller.last_raw["P1"] == "unparseable junk"

    def test_an_admitted_family_is_unaffected(self):
        res = generate_family("f001", "Law", "CONFLICT", _dry_cfg())
        assert res.admitted
        assert "rejected_response" not in res.artifacts
        assert "rejected_response_raw" not in res.artifacts


class TestV1IndexContract:
    """V1's endpoints are the 1-based keys of the atoms mapping it is handed.

    Nothing guarded this before: the prompt named no base, the payload was an unlabelled
    array whose natural reading is 0-based, the parser type-checked nothing, the gate
    normalized nothing, and the mock echoed plan positions so every dry run scored 1.00.
    A live model duly returned 0-based indices and scored 0.08/0.00 -- indistinguishable
    in the verdict from recovering nothing, while the same output re-indexed scored
    0.50/0.50.
    """

    _PLANNED = [
        {"source_pos": 1, "target_pos": 2, "sense": "Cause-Effect"},
        {"source_pos": 3, "target_pos": 4, "sense": "Alternative"},
    ]

    def _rec(self, pairs, senses=("Cause-Effect", "Alternative")):
        from fact_reasoner.locobench.taxonomy_bridge import coupling_for_sense

        return [
            {
                "source": s,
                "target": t,
                "sense": sense,
                "coupling": coupling_for_sense(sense),
            }
            for (s, t), sense in zip(pairs, senses)
        ]

    def test_the_payload_is_a_mapping_keyed_from_one(self):
        from fact_reasoner.locobench.pipeline import _validate_response

        plan, _ = parse.parse_plan(mock.mock_plan("q", "c", 0))
        atom_texts = [a["text"] for a in sorted(plan["atoms"], key=lambda x: x["pos"])]
        seen: dict = {}

        def spy(rendered, *, attempt=0):
            pid = which_prompt(rendered)
            if pid == "V1":
                seen["payload"] = _atoms_payload(rendered)
            return "[]" if pid in ("V1", "V4") else '{"fluency": 5}'

        _validate_response(_Caller(spy, attempts=1), "prose", plan, atom_texts)
        payload = seen["payload"]
        assert payload, "V1 must receive a labelled atoms mapping, not a bare array"
        # Keys are 1..N and align with plan positions, which is what the gate compares.
        assert sorted(payload, key=int) == [str(i + 1) for i in range(len(atom_texts))]
        assert payload["1"] == atom_texts[0]

    def test_one_based_recovery_passes(self):
        v = validate.gate_recovery(
            self._PLANNED, self._rec([(1, 2), (3, 4)]), n_atoms=5
        )
        assert v.passed

    def test_zero_based_recovery_is_rejected_with_the_evidence(self):
        v = validate.gate_recovery(
            self._PLANNED, self._rec([(0, 1), (2, 3)]), n_atoms=5
        )
        assert not v.passed
        reason = v.reason()
        assert "0-based" in reason
        assert "n_atoms=5" in reason
        # Deliberately not auto-shifted: a silent correction would mask a prompt
        # regression indefinitely.
        assert "mask a prompt regression" in reason

    def test_the_observed_payload_carries_the_pairs(self):
        v = validate.gate_recovery(self._PLANNED, self._rec([(1, 2)]), n_atoms=5)
        obs = v.results[0].observed
        # Without these, "recovered nothing" and "matched the wrong key space" are the
        # same number in the persisted record.
        assert obs["matched_pairs"] == [(1, 2)]
        assert (1, 2, "Cause-Effect") in obs["planned_pairs"]
        assert obs["recovered_pairs"]

    def test_the_parser_rejects_non_integer_endpoints(self):
        out, err = parse.parse_recovery(
            '[{"source": "a0", "target": "a1", "sense": "Restatement", '
            '"coupling": "equivalence"}]'
        )
        assert out is None
        assert "atom number" in err

    def test_the_parser_rejects_a_zero_endpoint(self):
        out, err = parse.parse_recovery(
            '[{"source": 0, "target": 1, "sense": "Evidence", '
            '"coupling": "entailment"}]'
        )
        assert out is None
        assert "numbered from 1" in err

    def test_the_parser_coerces_a_numeric_string(self):
        out, err = parse.parse_recovery(
            '[{"source": "1", "target": "2", "sense": "Evidence", '
            '"coupling": "entailment"}]'
        )
        assert err is None
        assert (out[0]["source"], out[0]["target"]) == (1, 2)

    def test_the_mock_resolves_endpoints_through_the_payload(self):
        # The dry run must exercise the contract, not bypass it.
        plan, _ = parse.parse_plan(mock.mock_plan("q", "c", 0))
        atoms = [a["text"] for a in sorted(plan["atoms"], key=lambda x: x["pos"])]
        payload = {str(i + 1): t for i, t in enumerate(atoms)}
        rendered = prompts.fill("V1", response="prose", atoms=json.dumps(payload))
        out = json.loads(mock.mock_recovery(rendered, payload, plan))
        assert out
        for r in out:
            assert isinstance(r["source"], int) and r["source"] >= 1
            assert isinstance(r["target"], int) and r["target"] >= 1
        assert validate.gate_recovery(plan["relations"], out, n_atoms=len(atoms)).passed


class TestRecoveryDirection:
    """`wrong_direction` is a first-class error kind, so the match must see direction."""

    def test_a_reversed_directed_sense_no_longer_counts(self):
        planned = [{"source_pos": 1, "target_pos": 2, "sense": "Cause-Effect"}]
        reversed_rec = [
            {
                "source": 2,
                "target": 1,
                "sense": "Cause-Effect",
                "coupling": "entailment",
            }
        ]
        v = validate.gate_recovery(planned, reversed_rec, n_atoms=5)
        assert not v.passed
        assert v.results[0].observed["matched_pairs"] == []

    def test_a_reversed_undirected_sense_still_counts(self):
        # Alternative is symmetric, so order carries no information.
        planned = [{"source_pos": 1, "target_pos": 2, "sense": "Alternative"}]
        rec = [
            {"source": 2, "target": 1, "sense": "Alternative", "coupling": "exclusive"}
        ]
        v = validate.gate_recovery(planned, rec, n_atoms=5)
        assert v.passed


class TestRecoveryCouplingIsCompiled:
    """Coupling recall compiles the recovered sense; it never reads the model's field.

    Measured live: 4 of 7 relations recovered by claude-opus-5 carried a coupling that
    disagreed with `coupling_for_sense(its own sense)` -- Restatement labelled
    "entailment" rather than "equivalence", Concession likewise. Because the gate compared
    that free-choice string against a value DERIVED from the plan's sense, a perfect
    recovery scored sense 1.00 and coupling 0.00. On the real run data this was the sole
    remaining blocker: Claude f001 matched 10/11 planned pairs with an identical sense in
    every case, yet scored coupling 8/11 = 0.727 against a 0.80 threshold.
    """

    def test_a_non_canonical_coupling_label_no_longer_fails_a_perfect_recovery(self):
        planned = [{"source_pos": 1, "target_pos": 2, "sense": "Concession"}]
        # Right pair, right sense -- but the model's own coupling string is not the
        # canonical compilation of Concession ("contradiction").
        rec = [
            {"source": 1, "target": 2, "sense": "Concession", "coupling": "entailment"}
        ]
        v = validate.gate_recovery(planned, rec, n_atoms=5)
        assert v.passed
        obs = v.results[0].observed
        assert obs["coupling"] == 1.0
        assert obs["sense"] == 1.0

    def test_coupling_is_a_genuine_coarsening_of_sense(self):
        # Evidence and Cause-Effect both compile to `entailment`, so confusing them
        # misses the sense but still earns coupling credit. This asymmetry is why the
        # coupling threshold (0.80) is HIGHER than the sense threshold (0.70), and what
        # licenses grading lossily-migrated seed edges on coupling alone.
        planned = [{"source_pos": 1, "target_pos": 2, "sense": "Evidence"}]
        rec = [
            {
                "source": 1,
                "target": 2,
                "sense": "Cause-Effect",
                "coupling": "entailment",
            }
        ]
        obs = validate.gate_recovery(planned, rec, n_atoms=5).results[0].observed
        assert obs["coupling"] == 1.0
        assert obs["sense"] == 0.0

    def test_a_different_coupling_family_earns_no_credit(self):
        # Restatement -> equivalence vs Concession -> contradiction: a real miss, and the
        # coarsening must not paper over it.
        planned = [{"source_pos": 1, "target_pos": 2, "sense": "Concession"}]
        rec = [
            {
                "source": 1,
                "target": 2,
                "sense": "Effect-Cause",
                "coupling": "contradiction",
            }
        ]
        obs = validate.gate_recovery(planned, rec, n_atoms=5).results[0].observed
        assert obs["coupling"] == 0.0
        assert obs["sense"] == 0.0

    def test_an_unknown_recovered_sense_does_not_raise(self):
        # `coupling_for_sense` raises on an unknown sense, and `_pair_key` falls back to
        # directed on the same input -- so a bogus sense reaches the comparison and would
        # take down the whole gate if the compile were unguarded. `gate_recovery` is a
        # public export called directly, so the parser is not a sufficient guard.
        planned = [{"source_pos": 1, "target_pos": 2, "sense": "Cause-Effect"}]
        rec = [{"source": 1, "target": 2, "sense": "Bogus", "coupling": "entailment"}]
        obs = validate.gate_recovery(planned, rec, n_atoms=5).results[0].observed
        assert obs["coupling"] == 0.0
        assert obs["sense"] == 0.0

    def test_the_prompt_states_the_compilation_rule(self):
        v1 = prompts.PROMPTS["V1"]
        assert "not an independent choice" in v1
        # The full map, not just the abstract rule: the observed disagreements were on
        # exactly the senses a model cannot guess (Restatement, Concession).
        assert '"equivalence"' in v1
        assert '"co_necessity"' in v1


class TestV4MergedSemantics:
    """`merged` means an atom was absorbed, not that a connective joined two atoms.

    P4 instruction 3 *mandates* "either X or Y" for Alternative and "although X, Y" for
    Concession, and those senses are required in every plan -- so the old definition
    rejected every family for obeying the prompt. Measured: 12 of 14 non-asserted flags
    across four live families landed on a coordination-sense endpoint.
    """

    def test_the_prompt_states_that_a_connective_asserts_both(self):
        v4 = prompts.PROMPTS["V4"]
        assert "either X or Y" in v4
        assert "NOT merged" in v4

    def test_p4_no_longer_forbids_what_p4_requires(self):
        p4 = prompts.PROMPTS["P4"]
        # The old wording was "must not merge two atoms into one clause", which
        # contradicted instruction 3's mandated realizations.
        assert "must not merge two atoms into one clause" not in p4
        assert "instruction 3" in p4

    def test_a_merged_atom_still_fails_the_coverage_gate(self):
        # The narrowing must not turn the gate off: genuine absorption still rejects.
        v = validate.gate_coverage(
            [{"index": 1, "status": "asserted"}, {"index": 2, "status": "merged"}], 2
        )
        assert not v.passed
        assert "not asserted" in v.reason()

    def test_full_coverage_passes(self):
        v = validate.gate_coverage(
            [{"index": 1, "status": "asserted"}, {"index": 2, "status": "asserted"}], 2
        )
        assert v.passed

    def test_the_coverage_threshold_is_unchanged(self):
        # The fix is definitional; the committed Phase-1 bar stays at 100%.
        assert validate.THRESHOLDS["v4_coverage"] == 1.00


class TestV3LeakageSemantics:
    """Leakage means naming the apparatus, not using the connectives it requires.

    The third instance of one structural bug: P4 instruction 3 *mandates* "either X or Y",
    "at least one of", "the two cannot both be true, and one of them must hold", and an
    explicit disclaimer that an ordering carries no inferential force -- and V3's leakage
    clause scored every one of them as evidence a plan existed. Measured live: all 8
    leakage spans on claude f001 were instruction-3 compliance, and the two Claude
    families were rejected with 16 and 9 spans.

    The decisive measurement was a cross-audit: deepseek's own response scored 5/5/5 with
    leakage 0 under its own auditor and 3/4/4 with 6 leakage spans under a Claude auditor.
    Same prose, opposite verdict -- so this is a definition problem, not a prose problem.
    """

    def test_the_mandated_constructions_are_exempted(self):
        v3 = prompts.PROMPTS["V3"]
        assert "NARROWLY" in v3
        for mandated in (
            "either X or Y",
            "at least one of X or Y",
            "the two cannot both be true, and one of them must hold",
            "although X, Y",
        ):
            assert mandated in v3, f"V3 must name {mandated!r} as non-leakage"

    def test_subject_matter_vocabulary_is_exempted(self):
        # The prose used "claim", "outcome" and "reading" about archaeology, not about the
        # annotation apparatus. The old clause flagged them anyway.
        v3 = prompts.PROMPTS["V3"]
        assert "ordinary scholarly vocabulary" in v3
        for word in ('"claim"', '"outcome"', '"reading"'):
            assert word in v3

    def test_naming_the_apparatus_is_still_leakage(self):
        # The narrowing must not gut the gate.
        v = validate.gate_audit(
            {
                "fluency": 5,
                "formality": 5,
                "organization": 5,
                "leakage": ["the relation plan"],
                "hedging": [],
            }
        )
        assert not v.passed

    def test_hedging_was_deliberately_not_narrowed(self):
        # deepseek's prose contained "and possibly both" -- a TRUE positive that only the
        # stricter auditor caught. A hedge is detectable from register alone, which is the
        # exact risk P4 instruction 7 exists to prevent, so this clause stays broad.
        v3 = prompts.PROMPTS["V3"]
        for word in ("possibly", "allegedly", "it is claimed"):
            assert word in v3

    def test_p4_warns_about_every_word_v3_checks(self):
        # The writer was being graded on a list it had never been given: V3 checked
        # "possibly" and "it is claimed"; P4 instruction 7 named neither. Parity asserted so
        # the two cannot drift apart again.
        p4, v3 = prompts.PROMPTS["P4"], prompts.PROMPTS["V3"]
        hedges = (
            "assume",
            "might",
            "possibly",
            "allegedly",
            "supposedly",
            "reportedly",
            "it is claimed",
        )
        checked = {w for w in hedges if w in v3}
        warned = {w for w in hedges if w in p4}
        assert checked <= warned, (
            f"V3 checks but P4 never warns about: {checked - warned}"
        )

    def test_p4_does_not_widen_the_hedge_scope(self):
        """Aligning the word lists is safe; widening the SCOPE collapses output length.

        Measured on one family with everything else fixed, deterministic across repeats:
        581 words under the original wording, 637 with the word list widened, **308** once
        "ANYWHERE in the response" was added, and **144** once a further sentence banned a
        specific phrasing. Each extra prohibition bought a shorter answer until the prose
        fell under instruction 6's 500-word floor and P4 failed outright -- three of four
        families died there. The residual mismatch (V3 may flag a hedge instruction 7 did
        not strictly forbid) is deliberate and preferable.
        """
        p4 = prompts.PROMPTS["P4"]
        assert "ANYWHERE in the response" not in p4
        assert 'never\n   "and possibly both"' not in p4
        # The scoping phrase that keeps the clause short must survive.
        assert "around planned-invalid relations" in p4

    def test_the_gate_records_span_text_not_just_counts(self):
        # "leakage: 16" is undiagnosable -- it reads as leaky prose when every span was a
        # mandated coordination. The V1 index bug hid behind the same count-only record.
        v = validate.gate_audit(
            {
                "fluency": 5,
                "formality": 5,
                "organization": 5,
                "leakage": ["naming the sense Concession", "x" * 400],
                "hedging": [],
                "artifacts": ["an enumerated list"],
            }
        )
        spans = v.results[1].observed["spans"]["leakage"]
        assert spans[0] == "naming the sense Concession"
        assert len(spans[1]) == 120, "span text must be truncated, not persisted whole"

    def test_artifacts_are_recorded_but_never_gated(self):
        # Previously dropped on the floor entirely: `observed` was built by iterating
        # v3_empty_spans, which does not include artifacts.
        v = validate.gate_audit(
            {
                "fluency": 5,
                "formality": 5,
                "organization": 5,
                "leakage": [],
                "hedging": [],
                "artifacts": ["heading: Introduction", "bulleted list"],
            }
        )
        art = next(g for g in v.results if g.gate == "V3.artifacts")
        assert art.passed, "artifacts are a quality signal, not an admission criterion"
        assert art.observed["count"] == 2
        assert v.passed

    def test_the_score_floor_is_unchanged(self):
        # This change narrows a definition; it does not lower a committed Phase-1 bar.
        assert validate.THRESHOLDS["v3_min_score"] == 4
        assert validate.THRESHOLDS["v3_empty_spans"] == ("leakage", "hedging")


class TestPromptTexParity:
    """The tex holds P3/P4/V1/V3/V4 verbatim; nothing guarded that until now."""

    @pytest.mark.parametrize("pid", ["P3", "P4", "V1", "V3", "V4"])
    def test_the_prompt_matches_the_phase_one_document(self, pid):
        tex = open(
            "docs/ideation/coherence/benchmark/locobench_phase1.tex", encoding="utf-8"
        ).read()
        assert prompts.PROMPTS[pid].strip() in tex, (
            f"{pid} has drifted from locobench_phase1.tex; update the verbatim block"
        )
