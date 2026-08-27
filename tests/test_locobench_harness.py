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

    def test_two_relations_over_one_pair_are_rejected(self):
        # This branch had NO coverage, which is why the mismatch with `parse_plan` reached a
        # live run: a plan carrying a duplicate pair cleared every gate and died here.
        item = _valid_item()
        dup = dict(item["relations"][0])
        item["relations"] = item["relations"] + [dup]
        with pytest.raises(SchemaError, match="duplicate relation"):
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
        # Phase 1 quoted 7800/1560/6240 with V2 in the committee. V2 is gone -- its
        # `exclusive`/`co_necessity` labels are derived from the sense by COMPILE rather than
        # adjudicated by a model -- so the committee drops by exactly its 1440 calls.
        b = perturb.call_budget(perturb.family_type_slots(120))
        assert b["total"] == 6360
        assert b["generation"] == 1560
        assert b["committee"] == 4800

    def test_documented_per_prompt_budget(self):
        b = perturb.call_budget(perturb.family_type_slots(120))
        assert {k: b[k] for k in ("P1", "P2", "P3", "P4", "P5")} == {
            "P1": 120,
            "P2": 120,
            "P3": 120,
            "P4": 120,
            "P5": 720,
        }
        assert {k: b[k] for k in ("V1", "V3", "V4")} == {
            "V1": 2520,
            "V3": 120,
            "V4": 2520,
        }

    def test_the_budget_no_longer_carries_a_v2_term(self):
        # The prompt, its parser, its mock and this budget line were all removed together;
        # a stray V2 term here would mean the deletion was partial.
        b = perturb.call_budget(perturb.family_type_slots(120))
        assert "V2" not in b

    def test_committee_dominates_generation(self):
        b = perturb.call_budget(perturb.family_type_slots(120))
        assert b["committee"] > 3 * b["generation"]

    def test_closing_the_v3_scope_gap_costs_twenty_three_percent(self):
        # The relative cost of auditing all five rungs rather than the base response rose
        # from 18% to 23% when V2's 1440 fixed calls left the denominator.
        shipped = perturb.call_budget(perturb.family_type_slots(120))
        complete = perturb.call_budget(
            perturb.family_type_slots(120), inline_responses=5
        )
        assert complete["total"] == 7800
        assert round(complete["total"] / shipped["total"] - 1, 2) == 0.23

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

    def test_all_eight_prompts_present(self):
        assert set(prompts.PROMPTS) == set(prompts.GENERATION_PROMPTS) | set(
            prompts.VALIDATION_PROMPTS
        )
        # Eight, not Phase 1's nine: V2 was removed with the rest of the exhaustiveness
        # adjudicator, since COMPILE derives `exclusive`/`co_necessity` from the sense.
        assert len(prompts.PROMPTS) == 8
        assert "V2" not in prompts.PROMPTS

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


class TestHedgesAreRejectionSampled:
    """P4 and V3 disagree on hedge SCOPE, and the parser closes the gap for free.

    P4 instruction 7 warns about the hedge words only "around planned-invalid relations";
    V3 flags them anywhere. Measured on gpt-oss-120b r5 f001, whose ONLY remaining rejection
    was `"possibly for body painting"` -- ordinary descriptive prose, so P4 permitted exactly
    what V3 rejects, and both auditors correctly flagged it as a true positive.

    Widening P4's scope is guarded against by `test_p4_does_not_widen_the_hedge_scope`,
    because added prohibitions measurably suppressed output (581 -> 308 -> 144 words).
    Checking in the parser needs no prompt change and costs nothing: `build_llm` installs the
    parser as the Mellea rejection-sampling requirement, so the hedge is re-sampled inside
    the same P4 call rather than surfacing as a V3 rejection after the whole audit panel ran.
    """

    def _prose(self, extra=""):
        return "```\n" + " ".join(["word"] * 600) + " " + extra + "\n```"

    @pytest.mark.parametrize("word", list(parse.HEDGE_WORDS))
    def test_each_hedge_word_is_rejected(self, word):
        out, err = parse.parse_response(self._prose(f"The finding {word} holds."))
        assert out is None
        assert word in err

    def test_clean_prose_of_sufficient_length_passes(self):
        out, err = parse.parse_response(self._prose("The finding holds."))
        assert err is None and out

    def test_the_mandated_disjunction_phrasing_is_not_a_hedge(self):
        # P4 instruction 3 REQUIRES "perhaps both" for Disjunction. Rejecting it here would
        # make the two prompts unsatisfiable together -- the eighth instance of that bug.
        out, err = parse.parse_response(
            self._prose("At least one of X or Y holds, perhaps both.")
        )
        assert err is None, err
        assert out

    @pytest.mark.parametrize(
        "word", ["assumption", "mighty", "possibilities", "reported", "supposed"]
    )
    def test_words_merely_containing_a_hedge_are_not_flagged(self, word):
        # Word-bounded: "reported" is not "reportedly", and "the assumption of" is ordinary
        # scholarly prose. An unbounded scan would reject compliant text.
        out, err = parse.parse_response(self._prose(f"The {word} of the site is clear."))
        assert err is None, f"{word!r} was wrongly treated as a hedge: {err}"

    def test_the_structural_complaints_are_reported_first(self):
        # A short hedged response must report its LENGTH, not the hedge: fixing the hedge
        # would not make it admissible, so the more fundamental complaint has to win.
        out, err = parse.parse_response("```\nIt might hold.\n```")
        assert out is None
        assert "word floor" in err

    def test_the_canonical_list_matches_what_the_prompts_recite(self):
        # The three prompt strings that recite these words had no constant behind them.
        v3, p4 = prompts.PROMPTS["V3"], prompts.PROMPTS["P4"]
        for word in parse.HEDGE_WORDS:
            assert f'"{word}"' in v3, f"V3 does not check {word!r}"
            assert word in p4, f"P4 does not warn about {word!r}"


class TestResponseMustBeBareProse:
    """P4's code block holds prose; every other prompt's holds JSON.

    Measured on a live deepseek-v3.2 run: P4 returned ```json {"response": "..."} ```.
    Nothing rejected it -- the block is taken verbatim, and `ignore_language=True` strips
    the ```json tag but not the object inside -- so a long-enough wrapped answer would
    have entered the corpus with a literal `{"response": "` prefix and `\\n` escapes as
    response text. It only surfaced because that answer was ALSO under the word floor.
    """

    def _wrapped(self, n_words):
        body = " ".join(["word"] * n_words)
        return '```json\n{"response": "' + body + '"}\n```'

    def test_a_json_wrapped_block_is_rejected(self):
        out, err = parse.parse_response(self._wrapped(600))
        assert out is None
        assert "not a JSON object" in err

    def test_the_format_complaint_wins_over_the_word_floor(self):
        # The wrapper is checked FIRST. With the order reversed, a short wrapped answer
        # reports only its length and the format defect stays invisible -- which is
        # precisely how this bug survived a live run.
        out, err = parse.parse_response(self._wrapped(5))
        assert out is None
        assert "not a JSON object" in err
        assert "word floor" not in err

    def test_a_json_array_is_also_rejected(self):
        out, err = parse.parse_response("```\n[1, 2, 3]\n```")
        assert out is None and "not a JSON object" in err

    def test_bare_prose_of_sufficient_length_still_passes(self):
        prose = " ".join(["word"] * 600)
        out, err = parse.parse_response(f"```\n{prose}\n```")
        assert err is None
        assert out.startswith("word")

    def test_the_real_captured_failure_is_now_diagnosed_as_a_format_error(self):
        # The exact payload from data/locobench-deepseek-v3.2 f002, reduced to its shape.
        raw = (
            '```json\n{\n  "response": "When an archaeologist encounters an artifact '
            'assemblage lacking definitive textual evidence, the attribution is guided '
            'by a multifaceted analytical framework."\n}\n```'
        )
        out, err = parse.parse_response(raw)
        assert out is None and "not a JSON object" in err


class TestGates:
    """The thresholds, and the reasons a gate gives when it rejects."""

    def test_all_thresholds_live_in_one_table(self):
        for key in (
            "v1_coupling",
            "v1_sense",
            "v3_min_score",
            "v4_coverage",
            "v4_gating_statuses",
            "min_incorrect_atoms",
            "window",
        ):
            assert key in validate.THRESHOLDS

    def test_thresholds_for_deleted_machinery_are_gone(self):
        # V2 and the V5 kappa statistics were removed; a stale threshold would advertise a
        # gate that nothing enforces, which is exactly how `v1_rule` sat unused for so long.
        for key in ("v2_exclusive", "kappa_coupling", "kappa_sense", "kappa_exhaustive"):
            assert key not in validate.THRESHOLDS

    def test_v1_coupling_stays_below_the_unanimity_cliff(self):
        # With planted errors excluded the denominator is 6, so anything above 0.834 demands
        # 6-of-6 -- a fresh unsatisfiable gate of the species this change removed.
        assert validate.THRESHOLDS["v1_coupling"] <= 0.834

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
        assert not v.passed and "not present" in v.reason()

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

    def test_aliases_of_one_model_are_not_a_committee(self):
        # Three entries, one underlying model: they dedupe to a single voter, which cannot
        # form a majority. A raw `len(committee)` check called this a committee of 3.
        cfg = GenConfig(
            dry_run=False,
            generators=[ModelRef(name="g-1", model_id="g")],
            committee=[ModelRef(name=f"m-{i}", model_id="m") for i in range(3)],
        )
        with pytest.raises(ValueError, match="eligible voter"):
            cfg.validate()

    def test_a_committee_that_is_mostly_the_generator_is_refused(self):
        # Four entries but three are the generator under other labels, so R3 leaves one
        # voter. This is the case a size-only check admitted, and it is the self-validation
        # bias the exclusion exists to prevent.
        cfg = GenConfig(
            dry_run=False,
            generators=[ModelRef(name="g-1", model_id="vendor/g")],
            committee=[
                ModelRef(name="alias-1", model_id="vendor/g", family="p"),
                ModelRef(name="alias-2", model_id="vendor/g", family="q"),
                ModelRef(name="alias-3", model_id="vendor/g", family="r"),
                ModelRef(name="other", model_id="vendor/h", family="s"),
            ],
        )
        with pytest.raises(ValueError, match="eligible voter"):
            cfg.validate()

    def test_three_distinct_models_are_enough(self):
        # The frontier committee shape: 3 models, none sharing the generator's id, so all
        # three vote and a 2-1 majority is available. The old `>= MIN + 1` size check
        # refused this outright even though it satisfies the property the rule is about.
        cfg = GenConfig(
            dry_run=False,
            generators=[ModelRef(name="claude", model_id="aws/claude-opus-5",
                                 family="claude")],
            committee=[
                ModelRef(name="gpt", model_id="azure/gpt-5.6-terra", family="openai"),
                ModelRef(name="gemini", model_id="gcp/gemini-3.1-pro", family="gemini"),
                ModelRef(name="sonnet", model_id="aws/claude-sonnet-5",
                         family="anthropic"),
            ],
        )
        cfg.validate()
        assert len(cfg.eligible_auditors(cfg.generators[0])) == 3

    def test_committee_needs_three_families(self):
        # Four DISTINCT models, so the eligible-voter floor is satisfied and this isolates
        # the family rule: they span only two families, which makes agreement an artefact of
        # shared training rather than independent judgment.
        cfg = GenConfig(
            dry_run=False,
            generators=[ModelRef(name="g-1", model_id="g")],
            committee=[
                ModelRef(name="a-1", model_id="a1", family="x"),
                ModelRef(name="a-2", model_id="a2", family="x"),
                ModelRef(name="b-1", model_id="b1", family="y"),
                ModelRef(name="b-2", model_id="b2", family="y"),
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
        assert "not present" in res.verdict.reason()

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
# fenced, and V1/V4 are BARE JSON lists -- and it is exactly those two that the old
# tuple-repr bug corrupted, because they have no fence for the extractor to anchor on.
_CANNED = {
    "P1": "[What caused the actuator to fail during the test flight?]",
    # 1-based integer atom numbers -- the keys of the mapping V1 is handed. This fixture
    # previously used string "a0"/"a1" ids, a third convention that would silently score as
    # total recovery failure; the parser now rejects it outright.
    "V1": '[{"source": 1, "target": 2, "sense": "Restatement", '
    '"coupling": "equivalence"}]',
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
        # Pins WHY this matters: a repr-wrapped payload breaks exactly V1/V4.
        for pid in ("V1", "V4"):
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
        build_llm(ModelRef("m", "m", "rits"), _dry_cfg())(prompts.PROMPTS["V3"])
        req = rec["requirements"][0]
        assert req.validation_fn is not None
        # The requirement is built from the prompt the harness recovered, which is what
        # makes the predicate the *right* parser rather than an arbitrary one. Exercising
        # `validation_fn` itself would mean constructing a Mellea Context, so the
        # predicate's behaviour is covered directly below instead.
        assert "V3" in str(req.description)

    @pytest.mark.parametrize(
        "pid,good,bad",
        [
            ("V3", '{"fluency": 5, "formality": 4, "organization": 5}', "{}"),
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

    @pytest.mark.parametrize("pid", sorted(prompts.PROMPTS))
    def test_dispatch_survives_the_semantic_note(self, pid):
        # There are two notes now, and `_check_retry_note` covers both -- but dispatch is
        # what actually breaks if a note embeds a probe, so assert it for both.
        note = _retry_note("a gate complaint", semantic=True)
        assert which_prompt(prompts.PROMPTS[pid] + note) == pid

    def test_the_two_notes_give_opposite_advice(self):
        parse_note = _retry_note("bad json", semantic=False)
        gate_note = _retry_note("V4: 1 atom(s) not asserted", semantic=True)
        assert "could not be read" in parse_note
        assert "could not be read" not in gate_note
        assert "Keep the same output format" in gate_note
        # Both must still carry the reason -- that is the whole point of the feedback.
        assert "bad json" in parse_note
        assert "V4: 1 atom(s) not asserted" in gate_note

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

    def test_all_three_validators_run_on_the_auditor_not_the_author(self, monkeypatch):
        # V1 and V4 used to run on the generator's own caller while only V3 was moved to the
        # panel, so the corpus's recall figures were self-reported -- the exact R3
        # self-generation bias `validate.py`'s header names first. All three now run on the
        # committee, and the generator's caller is only a fallback when no panel exists.
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
        assert by_prompt.get("V1") == "aud", "V1 must not grade its own author's prose"
        assert by_prompt.get("V3") == "aud"
        assert by_prompt.get("V4") == "aud"
        assert "gen" not in {tag for tag, _ in seen}, (
            "the generator must not appear as a rater when a panel is configured"
        )

    def test_the_generator_is_the_fallback_when_no_panel_exists(self):
        # Degrading to a self-audit is a weaker result, not an invalid one, so the dry run
        # and single-model configs must keep working.
        seen = []

        def _llm(rendered, *, attempt=0):
            seen.append(which_prompt(rendered))
            return _CANNED[which_prompt(rendered)]

        pipeline._validate_response(
            _Caller(_llm, attempts=1), "prose", {"atoms": [], "relations": []}, []
        )
        assert {"V1", "V3", "V4"} <= set(seen)

    def test_v1_takes_the_best_rater_not_the_first(self):
        # Recoverability is a claim about whether a careful reader CAN recover the plan, so
        # one competent reader succeeding settles it. Ordering must not decide admission --
        # that accident is what `gate_audit_panel` was built to remove from V3, and V1 is
        # graded across raters for the same reason.
        planned = [
            {"source_pos": 1, "target_pos": 2, "sense": "Restatement", "validity": "valid"}
        ]
        good = '[{"source": 1, "target": 2, "sense": "Restatement", "coupling": "equivalence"}]'

        def _mk(payload):
            def _llm(rendered, *, attempt=0):
                pid = which_prompt(rendered)
                return payload if pid == "V1" else _CANNED[pid]

            return _Caller(_llm, attempts=1)

        v = pipeline._validate_response(
            _Caller(lambda r, **kw: _CANNED[which_prompt(r)], attempts=1),
            "prose",
            {"atoms": [], "relations": planned},
            ["a", "b"],
            # The weak rater is FIRST, so a first-wins rule would reject.
            auditors=[("weak", _mk("[]")), ("strong", _mk(good))],
        )
        v1 = next(g for g in v.results if g.gate == "V1")
        assert v1.passed
        raters = next(g for g in v.results if g.gate == "V1.raters")
        assert raters.observed["reported"] == "strong"
        # Both rates are recorded, so a persistently weak panel member is diagnosable.
        assert set(raters.observed["rates"]) == {"weak", "strong"}

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


class TestFactualityReachesTheCorpus:
    """The response must contain false claims, not only true ones.

    P2 generates 4 `[incorrect]` claims, but P3's atom schema was `{pos, text}` and
    `parse_plan` required only those two keys, so the tag was dropped at the P2->P3 boundary
    and `_atoms_from_plan` defaulted every atom to `factual: True`. Measured on the shipped
    corpus: **all 170 atoms across both admitted families were `factual: true`** -- the
    benchmark's false content never reached it. No test caught this because `mock.py` emits
    `factual` itself, which made the dry run more faithful than the live path.
    """

    _CLAIMS = [
        {"text": "The regulator opened an investigation.", "tag": "correct"},
        {"text": "The component was certified in 1998.", "tag": "incorrect"},
        {"text": "The fleet returned to service in a week.", "tag": "incorrect"},
        {"text": "No one was harmed.", "tag": "alt-pair-1"},
        {"text": "Three people were injured.", "tag": "alt-pair-2"},
        {"text": "The tribunal held the supplier liable.", "tag": "holding"},
    ]

    def _plan(self, texts):
        return {"atoms": [{"pos": i + 1, "text": t} for i, t in enumerate(texts)]}

    def test_an_incorrect_claim_becomes_a_false_atom(self):
        plan = self._plan([c["text"] for c in self._CLAIMS])
        atoms = pipeline._atoms_from_plan(plan, self._CLAIMS)
        by_text = {a["text"]: a for a in atoms}
        assert by_text["The component was certified in 1998."]["factual"] is False
        assert by_text["The fleet returned to service in a week."]["factual"] is False
        assert by_text["The regulator opened an investigation."]["factual"] is True

    def test_the_exhaustive_pair_is_flagged_as_imprecise(self):
        # Exactly one of an alt-pair IS false and P2 does not say which, so labelling both
        # True is the honest default but must be marked so Phase 3 can exclude them.
        atoms = pipeline._atoms_from_plan(
            self._plan([c["text"] for c in self._CLAIMS]), self._CLAIMS
        )
        alt = [a for a in atoms if a["text"] in ("No one was harmed.", "Three people were injured.")]
        assert len(alt) == 2
        assert all("exhaustive_pair" in a.get("factual_note", "") for a in alt)

    def test_a_paraphrased_atom_falls_back_and_is_counted(self):
        # P3 instruction 2 requires verbatim reuse. If it paraphrases anyway, the tag cannot
        # be matched -- so the miss is COUNTED rather than silently defaulting to all-true.
        plan = self._plan(["Something P2 never said."])
        atoms = pipeline._atoms_from_plan(plan, self._CLAIMS)
        assert atoms[0]["factual"] is True  # the old default
        assert atoms[0]["factual_unmatched"] == 1

    def test_without_claims_the_old_behaviour_is_preserved(self):
        plan = {"atoms": [{"pos": 1, "text": "x", "factual": False}]}
        assert pipeline._atoms_from_plan(plan)[0]["factual"] is False

    def test_trailing_punctuation_does_not_defeat_the_match(self):
        plan = self._plan(["The component was certified in 1998"])  # no full stop
        assert pipeline._atoms_from_plan(plan, self._CLAIMS)[0]["factual"] is False

    def test_the_gate_rejects_a_plan_with_too_few_false_claims(self):
        plan = self._plan(["The regulator opened an investigation."])
        v = validate.gate_plan(plan, self._CLAIMS)
        gate = next(g for g in v.results if g.gate == "plan.factuality")
        assert not gate.passed
        assert gate.observed["n_incorrect_selected"] == 0
        # The reason must tell P3 what to DO, since it is fed back as a retry note.
        assert "choose more of the [incorrect] claims" in gate.detail

    def test_the_gate_accepts_the_quota(self):
        plan = self._plan([c["text"] for c in self._CLAIMS])
        gate = next(
            g for g in validate.gate_plan(plan, self._CLAIMS).results
            if g.gate == "plan.factuality"
        )
        assert gate.passed
        assert gate.observed["n_incorrect_selected"] == 2

    def test_the_gate_is_skipped_when_no_claims_are_supplied(self):
        plan = self._plan(["The regulator opened an investigation."])
        gates = [g.gate for g in validate.gate_plan(plan).results]
        assert "plan.factuality" not in gates

    def test_the_quota_is_two(self):
        assert validate.THRESHOLDS["min_incorrect_atoms"] == 2

    def test_a_non_boolean_factual_field_is_rejected(self):
        # "false" is a truthy string, so accepting it would mark a false claim as true.
        atoms = [{"pos": i + 1, "text": f"claim {i}"} for i in range(14)]
        atoms[3]["factual"] = "false"
        plan, err = parse.parse_plan(
            json.dumps({"atoms": atoms, "relations": [], "non_relations": []})
        )
        assert plan is None
        assert "must be true or false" in err

    def test_an_admitted_dry_run_family_carries_false_atoms(self):
        # The end-to-end acceptance test for the requirement: a generated corpus must contain
        # incorrect claims, which is exactly what the shipped one did not.
        res = generate_family("f001", "Law", "CONFLICT", _dry_cfg())
        assert res.admitted, res.verdict.reason()
        for item in res.items:
            n_false = sum(1 for a in item["atoms"] if a["factual"] is False)
            assert n_false >= validate.THRESHOLDS["min_incorrect_atoms"], (
                f"{item['id']} carries only {n_false} false atom(s)"
            )


class TestV3ClosedListsAreClosedInCode:
    """V3's prompt promises closed lists; the code, not the auditor, is the guarantor.

    Measured live on one response, auditors reported spans the prompt names VERBATIM as not
    reportable: "at least one of these was true:" and "as indicated by the fact that" as
    leakage, and "perhaps both" as hedging -- the last being a phrase P4 instruction 3
    *mandates* for Disjunction. One auditor also flagged "One or both of these factors may
    hold" as hedging, which contains no word on the closed list at all. Two of three agreed
    on some of these, so the majority rule did not save the family.

    Filtering happens before the vote for that reason: two auditors making the same
    prompt-exempt mistake would otherwise constitute a "majority".
    """

    def _a(self, **kw):
        base = {"fluency": 5, "formality": 5, "organization": 5,
                "leakage": [], "hedging": [], "artifacts": []}
        base.update(kw)
        return base

    def test_hedging_must_contain_a_word_from_the_closed_list(self):
        # "perhaps both" and "may hold" are not on HEDGE_WORDS, and `parse_response` has
        # already rejected any prose containing one that is.
        out = validate._filter_spans(
            self._a(hedging=["perhaps both", "One or both of these factors may hold"])
        )
        assert out["hedging"] == []
        assert out["_raw_counts"]["hedging"] == 2

    def test_a_real_hedge_word_survives(self):
        out = validate._filter_spans(self._a(hedging=["we assume the alloy held"]))
        assert out["hedging"] == ["we assume the alloy held"]

    @pytest.mark.parametrize(
        "span",
        [
            "at least one of these was true",
            "as indicated by",
            "perhaps both",
            "either this or that",
            "one or both",
            "one of these accounts must be wrong",
            "subsequently",
            "earlier than",
            "which postdates",
            "although",
        ],
    )
    def test_p4_mandated_phrasings_are_never_leakage(self, span):
        # Each of these is required by P4 instruction 3 to realize a coupling. A writer
        # cannot express joint truth without them, so flagging them punishes obedience.
        assert validate._filter_spans(self._a(leakage=[span]))["leakage"] == []

    @pytest.mark.parametrize(
        "span",
        [
            "the relation plan",
            "naming the sense Concession",
            "the earlier statement that they were manufactured after 6500 BCE",
            "the plan's strength band",
        ],
    )
    def test_real_leakage_still_reports(self, span):
        # The last one CONTAINS the exempt word "after"; a naive substring test in the wrong
        # direction would suppress it, which would be a true positive lost.
        assert validate._filter_spans(self._a(leakage=[span]))["leakage"] == [span]

    def test_a_span_absent_from_the_response_is_dropped(self):
        # V3's prompt: "if you cannot copy it out of the text, it is not there and must not
        # be reported." An invented quote cannot evidence anything.
        out = validate._filter_spans(
            self._a(leakage=["the annotation plan"]), "Prose that says nothing of the sort."
        )
        assert out["leakage"] == []

    def test_a_verbatim_span_is_kept_when_the_response_is_given(self):
        prose = "The tribunal reviewed the annotation plan before ruling."
        out = validate._filter_spans(self._a(leakage=["the annotation plan"]), prose)
        assert out["leakage"] == ["the annotation plan"]

    def test_two_auditors_making_the_same_exempt_mistake_do_not_reject(self):
        # The live failure mode: a "majority" built entirely out of false positives.
        bad = self._a(leakage=["at least one of these was true"], hedging=["perhaps both"])
        v = validate.gate_audit_panel([("a", bad), ("b", bad), ("c", self._a())])
        spans = next(g for g in v.results if g.gate == "V3.spans")
        assert spans.passed
        # The raw reports are preserved: a systematically-wrong auditor is a finding.
        assert spans.observed["raw_counts"]["a"]["leakage"] == 1
        assert spans.observed["n_flagging_per_kind"] == {"leakage": 0, "hedging": 0}


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
        # The leakage span must be REAL leakage -- a bare "at least one of" is a connective
        # P4 mandates, so `_filter_spans` drops it before the vote and the split would be
        # 0-of-3 rather than 1-of-3.
        hedged = self._a(hedging=["might seem"])
        both = self._a(leakage=["the plan's strength band"], hedging=["might seem"])
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


class TestP4PreservesAtomContent:
    """Quantifiers are content, not surface wording -- the V4 `altered` failure mode.

    Measured live: deepseek realized the atom "All tools were crafted by a single,
    innovative culture." as "...indicates they were crafted by a single, innovative
    culture", dropping "All". V4 correctly flagged it `altered`, coverage came to 0.938
    against a threshold of 1.00, and the family was lost. Instruction 2 permitted
    "adjust surface wording for fluency" without ever saying a quantifier is not wording.
    """

    def test_instruction_two_names_quantifiers_as_content(self):
        p4 = prompts.PROMPTS["P4"]
        assert "are CONTENT" in p4
        for word in ("Quantifiers", "determiners", "polarity", "modals"):
            assert word in p4

    def test_it_gives_the_measured_counter_example(self):
        # A worked substitution, not an abstract rule: the observed failure was on exactly
        # this shape, and the abstract rule ("must not change what is asserted") was
        # already present and insufficient.
        p4 = prompts.PROMPTS["P4"]
        assert '"All tools were crafted by a single' in p4
        assert "dropping" in p4 and "All" in p4

    def test_the_word_floor_is_stated_as_hard(self):
        # deepseek wrote 318 and 516 words against a 500 floor (Claude: 637, 659), so the
        # floor is restated as a rejection rather than advice ("below 500 is too short").
        p4 = prompts.PROMPTS["P4"]
        assert "500-word floor is a hard" in p4
        assert "rejects a shorter answer" in p4

    def test_instruction_eight_asks_for_prose_not_json(self):
        p4 = prompts.PROMPTS["P4"]
        assert "not a JSON object" in p4

    def test_no_new_prohibition_was_added(self):
        # Prohibitions measurably collapse P4's output length (581 -> 308 -> 144 words in
        # a recorded A/B), which is why all three edits above are phrased positively. The
        # bound is an upper one, not an equality: going DOWN is the direction that
        # measurement favours, and the Precedence rewrite removed one (see
        # TestP4RealizesOrderingRelations).
        import re

        assert len(re.findall(r"Do NOT", prompts.PROMPTS["P4"])) <= 6
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

    @pytest.mark.parametrize(
        "path",
        ["configs/locobench_gptoss.json"],
    )
    def test_the_shipped_run_configs_are_valid(self, path):
        # The real files: a rename in the inventory, or a committee that stops spanning
        # three families, breaks a live run at startup. Cheaper to catch here.
        cfg = load_config(path)
        cfg.validate()  # raises on a bad committee, generator or merlin path
        assert len(cfg.generators) == 1
        gen = cfg.generators[0]
        # Every RITS entry carries an explicit endpoint, so model_id is the raw served id.
        assert gen.base_url and gen.model_id
        # R3: the generator may not audit its own prose, and V3 votes, so a panel is
        # needed rather than a single rater.
        panel = cfg.eligible_auditors(gen)
        assert len(panel) >= 3
        assert all(m.model_id != gen.model_id for m in panel)

    def test_the_gptoss_config_does_not_pin_temperature_zero(self):
        # Measured: openai/gpt-oss-120b-a100 returns success=True at temperature 0.0 while
        # emitting no bulleted claims, so P2's parser rejects every attempt and a capable
        # model looks incapable. The ladder must start at None ("send no temperature").
        cfg = load_config("configs/locobench_gptoss.json")
        assert cfg.retry_temperatures[0] is None
        assert 0.0 not in [t for t in cfg.retry_temperatures if t is not None]

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
        # A gate complaint must NOT be framed as an unreadable output. The output parsed
        # fine; the content is wrong. Telling the model to fix the format instead is
        # actively misleading -- a live deepseek family had one atom's quantifier dropped,
        # was flagged by V4 on all three attempts, and never repaired it because every
        # retry asked for a reformat.
        assert "could not be read" not in seen[1]
        assert "did not meet a quality requirement" in seen[1]
        assert "Keep the same output format" in seen[1]

    def test_a_parse_failure_still_asks_for_a_format_fix(self):
        """The other branch keeps the original wording -- the advice is kind-specific."""
        seen = []

        def llm(rendered, *, attempt=0):
            seen.append(rendered)
            return "not a plan at all"

        _Caller(llm, attempts=2).ask(
            "P3", parse.parse_plan, question="q", claims="- a [correct]"
        )
        assert "could not be read" in seen[1]
        assert "did not meet a quality requirement" not in seen[1]

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


class TestPlantedErrorsAreNotGraded:
    """V1 scores RECOVERABILITY, so the deliberately-broken relations are not in it.

    This is the regression test for an unsatisfiable gate. P3 mandates that exactly 4 of
    10 relations be `validity: "invalid"` -- planted errors with an `error_kind` -- and
    `gate_recovery` used to divide by `len(planned)`, putting all 10 in the denominator
    against `v1_coupling = 0.80`. The ceiling was therefore 6/10 = 0.60 and no writer and
    no reader could ever pass: an invalid relation is broken ON PURPOSE, so its absence
    from the recovery is the intended outcome rather than a defect.

    Measured live before the fix: five frontier models independently scored one admitted
    family at ALL-10 0.80/0.80 but VALID-6 1.00/1.00 -- seven raters converging on the
    same shortfall while recovering every sound relation perfectly, which is the
    arithmetic fingerprint of the bug rather than of weak prose. The three historical
    "passes" only passed because the invalid quota was under-filled or because the error
    injection had silently failed and the broken edges were credited as correct.
    """

    def _spec_plan(self):
        """A plan shaped exactly as P3 instruction 8 mandates: 6 valid, 4 invalid."""
        valid = [
            {"source_pos": 1, "target_pos": 2, "sense": "Cause-Effect"},
            {"source_pos": 2, "target_pos": 3, "sense": "Evidence"},
            {"source_pos": 3, "target_pos": 4, "sense": "Restatement"},
            {"source_pos": 5, "target_pos": 6, "sense": "Alternative"},
            {"source_pos": 7, "target_pos": 8, "sense": "Disjunction"},
            {"source_pos": 9, "target_pos": 10, "sense": "Concession"},
        ]
        for r in valid:
            r["validity"] = "valid"
        invalid = [
            {"source_pos": 11, "target_pos": 12, "sense": "Precedence",
             "validity": "invalid", "error_kind": "wrong_sense"},
            {"source_pos": 12, "target_pos": 13, "sense": "Cause-Effect",
             "validity": "invalid", "error_kind": "wrong_direction"},
            {"source_pos": 13, "target_pos": 14, "sense": "Contrast",
             "validity": "invalid", "error_kind": "spurious"},
            {"source_pos": 4, "target_pos": 6, "sense": "Instantiation",
             "validity": "invalid", "error_kind": "false_endpoint"},
        ]
        return valid + invalid

    def _recover(self, planned):
        from fact_reasoner.locobench.taxonomy_bridge import coupling_for_sense

        return [
            {
                "source": p["source_pos"],
                "target": p["target_pos"],
                "sense": p["sense"],
                "coupling": coupling_for_sense(p["sense"]),
            }
            for p in planned
        ]

    def test_a_spec_conforming_plan_can_reach_a_perfect_score(self):
        # The satisfiability test. If this ever fails, the gate has become unpassable
        # again for a plan that obeys P3 exactly -- which is the whole bug class.
        planned = self._spec_plan()
        valid_only = [p for p in planned if p["validity"] == "valid"]
        v = validate.gate_recovery(planned, self._recover(valid_only), n_atoms=14)
        obs = v.results[0].observed
        assert (obs["coupling"], obs["sense"]) == (1.0, 1.0)
        assert v.passed

    def test_invalid_relations_are_excluded_from_the_denominator(self):
        planned = self._spec_plan()
        valid_only = [p for p in planned if p["validity"] == "valid"]
        obs = validate.gate_recovery(
            planned, self._recover(valid_only), n_atoms=14
        ).results[0].observed
        # 6 valid recovered out of 6 graded, NOT 6 out of 10.
        assert obs["n_graded"] == 6
        assert obs["n_planted"] == 4

    def test_recovering_nothing_still_fails(self):
        # The fix must not turn the gate off: an empty recovery is still a rejection.
        v = validate.gate_recovery(self._spec_plan(), [], n_atoms=14)
        assert not v.passed
        assert "recovery too low" in v.reason()

    def test_relations_with_no_validity_key_are_all_graded(self):
        # Back-compatibility: `validity` defaults to graded, so the many call sites and
        # tests that omit it -- and any plan predating the field -- behave as before.
        planned = [
            {"source_pos": 1, "target_pos": 2, "sense": "Cause-Effect"},
            {"source_pos": 3, "target_pos": 4, "sense": "Alternative"},
        ]
        obs = validate.gate_recovery(
            planned, self._recover(planned[:1]), n_atoms=5
        ).results[0].observed
        assert obs["n_graded"] == 2
        assert obs["coupling"] == 0.5

    def test_an_all_invalid_plan_passes_instead_of_dividing_by_zero(self):
        planned = [
            {"source_pos": 1, "target_pos": 2, "sense": "Cause-Effect",
             "validity": "invalid", "error_kind": "spurious"},
        ]
        v = validate.gate_recovery(planned, [], n_atoms=5)
        assert v.passed
        assert "no valid planned relations" in v.reason() or v.results[0].detail

    def test_planted_realization_is_recorded_but_never_gates(self):
        planned = self._spec_plan()
        # Every relation recovered as planned, including all four planted errors: the
        # writer realized the broken edges faithfully, which is a MEASUREMENT and must
        # not be an admission criterion in either direction.
        v = validate.gate_recovery(planned, self._recover(planned), n_atoms=14)
        planted = [r for r in v.results if r.gate == "V1.planted"]
        assert len(planted) == 1
        assert planted[0].passed  # observation only
        obs = planted[0].observed
        assert obs["n_invalid"] == 4
        # wrong_direction is recovered with the endpoints as planned here, so it counts as
        # realized; the per-kind breakdown is what makes error injection auditable.
        assert obs["by_error_kind"]["spurious"]["recovered_as_planned"] == 1
        assert set(obs["by_error_kind"]) == {
            "wrong_sense", "wrong_direction", "spurious", "false_endpoint"
        }
        assert v.passed


class TestRecoveryDirection:
    """`wrong_direction` is a first-class error kind, so the match must see direction."""

    def test_a_reversed_directed_sense_earns_no_credit(self):
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
        obs = v.results[0].observed
        # The credit is what matters, and it is zero on BOTH rates. The pair is now
        # *matched* on its endpoints and reported as reversed, rather than vanishing --
        # direction moved out of the identity key and into the grading, so that "recovered
        # backwards" and "not recovered at all" are finally distinguishable.
        assert obs["coupling"] == 0.0
        assert obs["sense"] == 0.0
        assert obs["reversed_pairs"] == [(1, 2, "Cause-Effect")]

    def test_a_reversal_is_reported_separately_from_an_absence(self):
        planned = [
            {"source_pos": 1, "target_pos": 2, "sense": "Evidence"},
            {"source_pos": 3, "target_pos": 4, "sense": "Evidence"},
        ]
        # (1,2) recovered backwards; (3,4) not recovered at all.
        rec = [{"source": 2, "target": 1, "sense": "Evidence", "coupling": "entailment"}]
        obs = validate.gate_recovery(planned, rec, n_atoms=6).results[0].observed
        assert obs["reversed_pairs"] == [(1, 2, "Evidence")]
        assert obs["matched_pairs"] == [(1, 2)]  # (3,4) is absent, not reversed
        assert obs["coupling"] == 0.0

    def test_a_cross_directedness_sense_over_the_same_pair_is_matchable(self):
        """The measured bug: same endpoints, adjacent sense, previously scored zero twice.

        gpt-oss-120b recovered a planned ``Concession(7, 12)`` as ``Contrast(7, 12)``.
        Both compile to ``contradiction``, so this is a sense miss that should still earn
        coupling credit -- coupling is the coarser label, which is why its threshold is the
        higher one. Instead the pair was unmatchable, because ``Concession`` is directed and
        ``Contrast`` is not, so the two keys could never collide and the relation counted
        as "not recovered".
        """
        planned = [{"source_pos": 7, "target_pos": 12, "sense": "Concession"}]
        rec = [
            {
                "source": 7,
                "target": 12,
                "sense": "Contrast",
                "coupling": "contradiction",
            }
        ]
        obs = validate.gate_recovery(planned, rec, n_atoms=20).results[0].observed
        assert obs["matched_pairs"] == [(7, 12)]
        assert obs["coupling"] == 1.0, "same coupling class, so credit is earned"
        assert obs["sense"] == 0.0, "but the sense is still wrong"

    def test_the_pair_key_is_sense_independent(self):
        # The property that makes the above work: identity is the unordered atom pair, so
        # no two senses over the same endpoints can land in different key spaces.
        assert validate._pair_key(1, 2, "Concession") == validate._pair_key(
            1, 2, "Contrast"
        )
        assert validate._pair_key(1, 2, "Concession") == validate._pair_key(
            2, 1, "Alternative"
        )
        assert validate._pair_key(1, 2) != validate._pair_key(1, 3)

    def test_two_relations_over_one_pair_are_graded_independently(self):
        """A dict keyed by pair dropped all but the last, then graded it against both.

        Exposed by the pair-identity fix and pre-existing: a live plan carried 10-11 as
        both `Alternative` and `Concession`. Because the two senses differ in directedness,
        the old sense-derived key happened to keep them apart -- so a plan whose two
        same-pair relations were BOTH recovered exactly would have scored 0.5, not 1.0, the
        moment the key stopped depending on the sense.
        """
        planned = [
            {"source_pos": 10, "target_pos": 11, "sense": "Alternative"},
            {"source_pos": 10, "target_pos": 11, "sense": "Concession"},
        ]
        rec = [
            {"source": 10, "target": 11, "sense": "Alternative"},
            {"source": 10, "target": 11, "sense": "Concession"},
        ]
        obs = validate.gate_recovery(planned, rec).results[0].observed
        assert obs["sense"] == 1.0
        assert obs["coupling"] == 1.0

    def test_one_recovery_cannot_satisfy_two_planned_relations(self):
        # The other half of the same property: consumption is one-to-one, so a single
        # recovery must not be credited twice.
        planned = [
            {"source_pos": 10, "target_pos": 11, "sense": "Alternative"},
            {"source_pos": 10, "target_pos": 11, "sense": "Concession"},
        ]
        rec = [{"source": 10, "target": 11, "sense": "Alternative"}]
        obs = validate.gate_recovery(planned, rec).results[0].observed
        assert obs["sense"] == 0.5

    def test_a_contended_recovery_goes_to_the_relation_it_explains(self):
        """Assignment is competitive, so it must be best-first rather than plan-order.

        Measured on a live plan holding BOTH `(1,2) Precedence` and `(2,1) Cause-Effect`
        against one recovered `(2,1) Evidence`. In plan order `Precedence` claims it and is
        scored reversed, while `Cause-Effect` -- which the recovery matches in direction and
        in coupling class (both entailment) -- is left with nothing and scored absent. One
        defect in the prose, two relations penalized.
        """
        planned = [
            {"source_pos": 1, "target_pos": 2, "sense": "Precedence"},
            {"source_pos": 2, "target_pos": 1, "sense": "Cause-Effect"},
        ]
        rec = [{"source": 2, "target": 1, "sense": "Evidence"}]
        obs = validate.gate_recovery(planned, rec).results[0].observed
        # Cause-Effect(2,1) takes it: same direction, and Evidence compiles to entailment
        # exactly as Cause-Effect does, so coupling credit is earned.
        assert obs["coupling"] == 0.5
        # And no spurious reversal is reported against Precedence.
        assert obs["reversed_pairs"] == []

    def test_plan_order_does_not_change_the_rates(self):
        # The assignment must be order-independent, or the same prose scores differently
        # depending on how P3 happened to list its relations.
        a = {"source_pos": 1, "target_pos": 2, "sense": "Precedence"}
        b = {"source_pos": 2, "target_pos": 1, "sense": "Cause-Effect"}
        rec = [{"source": 2, "target": 1, "sense": "Evidence"}]
        first = validate.gate_recovery([a, b], rec).results[0].observed
        second = validate.gate_recovery([b, a], rec).results[0].observed
        assert (first["coupling"], first["sense"]) == (
            second["coupling"],
            second["sense"],
        )

    def test_a_self_loop_is_not_collapsed(self):
        # The original frozenset key collapsed self-loops; the replacement must not
        # reintroduce that, so (1,1) stays distinct from any other pair.
        assert validate._pair_key(1, 1) != validate._pair_key(1, 2)
        obs = (
            validate.gate_recovery(
                [{"source_pos": 1, "target_pos": 1, "sense": "Evidence"}],
                [{"source": 1, "target": 1, "sense": "Evidence"}],
            )
            .results[0]
            .observed
        )
        assert obs["sense"] == 1.0

    def test_both_directions_offered_does_not_overwrite(self):
        # The other defect the frozenset key had: V1 emitting both (i,j) and (j,i) silently
        # overwrote one. They are now both candidates, and the correct one is chosen.
        planned = [{"source_pos": 1, "target_pos": 2, "sense": "Evidence"}]
        rec = [
            {"source": 2, "target": 1, "sense": "Evidence"},
            {"source": 1, "target": 2, "sense": "Evidence"},
        ]
        obs = validate.gate_recovery(planned, rec).results[0].observed
        assert obs["sense"] == 1.0
        assert obs["reversed_pairs"] == []

    def test_an_unknown_planned_sense_is_treated_as_directed(self):
        # Conservative default, matching the old key's fallback: a bogus sense must not buy
        # direction-blind credit.
        planned = [{"source_pos": 1, "target_pos": 2, "sense": "NotASense"}]
        rec = [{"source": 2, "target": 1, "sense": "NotASense", "coupling": "x"}]
        obs = validate.gate_recovery(planned, rec, n_atoms=5).results[0].observed
        assert obs["reversed_pairs"] == [(1, 2, "NotASense")]
        assert obs["sense"] == 0.0

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
        # Whitespace-normalized: the prompt wraps at 80 columns, so the phrase can straddle
        # a line break without the meaning changing.
        assert "instruction 3" in " ".join(p4.split())

    def test_a_merged_atom_still_fails_the_coverage_gate(self):
        # The narrowing must not turn the gate off: genuine absorption still rejects.
        v = validate.gate_coverage(
            [{"index": 1, "status": "asserted"}, {"index": 2, "status": "merged"}], 2
        )
        assert not v.passed
        assert "not present" in v.reason()

    def test_full_coverage_passes(self):
        v = validate.gate_coverage(
            [{"index": 1, "status": "asserted"}, {"index": 2, "status": "asserted"}], 2
        )
        assert v.passed

    def test_the_coverage_threshold_is_unchanged(self):
        # The fix is definitional; the committed Phase-1 bar stays at 100%.
        assert validate.THRESHOLDS["v4_coverage"] == 1.00


class TestV4AlteredIsRecordedNotGated:
    """`altered` is a wording judgment, so it must not cost a family on its own.

    P4 instruction 2 explicitly licenses surface rewording ("You may adjust surface wording
    for fluency"), and `altered` means "asserts something related but changes the content" --
    the judgment most sensitive to exactly that licence. Measured across 11 live rejections, 5
    were V4 and 3 of those were a SINGLE `altered` atom at coverage 0.938-0.944. `missing` and
    `merged` are structural losses and still reject.
    """

    def test_an_altered_atom_does_not_reject(self):
        v = validate.gate_coverage(
            [{"index": 1, "status": "asserted"}, {"index": 2, "status": "altered"}], 2
        )
        assert v.passed
        # Recorded, though: a rising altered rate is a P4 fidelity signal worth having.
        assert v.results[0].observed["n_altered"] == 1

    def test_a_missing_atom_still_rejects(self):
        v = validate.gate_coverage(
            [{"index": 1, "status": "asserted"}, {"index": 2, "status": "missing"}], 2
        )
        assert not v.passed

    def test_only_missing_and_merged_gate(self):
        assert validate.THRESHOLDS["v4_gating_statuses"] == ("missing", "merged")


class TestV4PanelVote:
    """V4 was the last single-rater gate, over a 14-16 way conjunction.

    Measured on identical prose, seven raters: six returned all 17 atoms asserted, one
    returned three defects (altered/missing/merged) and one returned a single altered. Under
    a single-rater gate, drawing either of the last two rejects a family the other six judged
    perfect. Voting per ATOM keeps "every atom present" absolute while denying any one rater
    the power to declare an atom absent.
    """

    def _cov(self, statuses):
        return [{"index": i + 1, "status": s} for i, s in enumerate(statuses)]

    def test_a_lone_missing_vote_does_not_reject(self):
        clean = self._cov(["asserted"] * 3)
        outlier = self._cov(["asserted", "missing", "asserted"])
        v = validate.gate_coverage_panel(
            [("a", clean), ("b", outlier), ("c", clean)], 3
        )
        assert v.passed
        # The dissent is still visible -- outvoting is not the same as discarding.
        assert v.results[0].observed["lost_votes"] == {"2": ["b"]}

    def test_a_majority_missing_vote_rejects(self):
        bad = self._cov(["asserted", "missing", "asserted"])
        v = validate.gate_coverage_panel(
            [("a", bad), ("b", bad), ("c", self._cov(["asserted"] * 3))], 3
        )
        assert not v.passed
        assert "majority" in v.results[0].detail

    def test_altered_never_counts_however_many_raters_agree(self):
        alt = self._cov(["asserted", "altered", "asserted"])
        v = validate.gate_coverage_panel([("a", alt), ("b", alt), ("c", alt)], 3)
        assert v.passed
        assert v.results[0].observed["altered_votes"] == {"2": ["a", "b", "c"]}

    def test_a_single_rater_panel_matches_the_solo_gate(self):
        # The degenerate case must not change behaviour: `gate_coverage` is still used on
        # the no-panel path, so the two must agree.
        cov = self._cov(["asserted", "missing"])
        panel = validate.gate_coverage_panel([("only", cov)], 2)
        solo = validate.gate_coverage(cov, 2)
        assert panel.passed == solo.passed is False

    def test_no_raters_is_a_harness_failure_not_a_verdict(self):
        v = validate.gate_coverage_panel([], 3)
        assert not v.passed
        assert "no coverage output" in v.reason()

    def test_mixed_index_types_do_not_raise(self):
        # Found on the first live run: `parse_coverage` accepts both `3` and `"3"`, and real
        # raters disagree about which they emit -- one returned ints, another numeric strings.
        # Sorting the mixed keys raised TypeError and killed the whole run mid-family.
        v = validate.gate_coverage_panel(
            [
                ("ints", [{"index": 1, "status": "asserted"},
                          {"index": 2, "status": "missing"}]),
                ("strs", [{"index": "1", "status": "asserted"},
                          {"index": "2", "status": "asserted"}]),
            ],
            2,
        )
        assert v.results[0].observed["lost_votes"] == {"2": ["ints"]}

    def test_a_string_and_an_int_index_are_the_same_atom(self):
        # The subtler half of the same bug: keying `2` apart from `"2"` splits one atom's
        # votes across two buckets, so two raters agreeing it is missing each count once
        # against a quorum of two and the majority silently never forms.
        v = validate.gate_coverage_panel(
            [
                ("a", [{"index": 1, "status": "asserted"}, {"index": 2, "status": "missing"}]),
                ("b", [{"index": "1", "status": "asserted"}, {"index": "2", "status": "missing"}]),
            ],
            2,
        )
        assert not v.passed, "two raters agreed atom 2 is missing; that is a majority"
        assert v.results[0].observed["lost_votes"] == {"2": ["a", "b"]}

    def test_a_repeated_vote_from_one_rater_is_not_two_votes(self):
        v = validate.gate_coverage_panel(
            [
                ("a", [{"index": 2, "status": "missing"}, {"index": 2, "status": "merged"}]),
                ("b", [{"index": 1, "status": "asserted"}]),
            ],
            2,
        )
        assert v.passed, "one rater listing an atom twice must not constitute a majority"


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

    def test_the_mandated_evidential_markers_are_exempted(self):
        """P4's Evidence markers were mandated but never exempted from leakage.

        Measured on gpt-oss r4 f001: deepseek's auditor flagged "as indicated by the fact
        that" as leakage. P4 instruction 3 mandates "as indicated by" for Evidence, and the
        exemption list covered the coordination senses but not the evidential ones.
        """
        v3 = prompts.PROMPTS["V3"]
        for mandated in ("as indicated by", "as shown by", "perhaps both"):
            assert mandated in v3, f"V3 must name {mandated!r} as non-leakage"

    def test_the_hedge_list_is_declared_closed_and_exempts_required_uncertainty(self):
        """Two auditors flagged "perhaps both" as hedging, which P4 mandates verbatim.

        The hedging clause had NO exemption list at all -- unlike the leakage clause -- so a
        construction the writer is required to use had nowhere to be excused. And one auditor
        reported "assume" as a hedge span when that word does not occur anywhere in the
        prose, so the clause also needed the quote-verbatim requirement.
        """
        v3 = prompts.PROMPTS["V3"]
        assert "list\n  is CLOSED" in v3 or "is CLOSED" in v3
        assert "perhaps both" in v3
        assert "not hedging" in v3

    def test_every_span_must_be_quotable(self):
        # Two of the seven flagged spans in the measured run were FABRICATED -- not present
        # in the response at all ("assume", and a paraphrased "either this or the earlier
        # statement..."). A quote-verbatim requirement is the only lever the prompt has.
        assert "QUOTED VERBATIM" in prompts.PROMPTS["V3"]

    def test_the_hedge_words_still_match_p4s_warning_list(self):
        # The pre-existing parity property: V3 may not check a word P4 never warned about.
        v3, p4 = prompts.PROMPTS["V3"], prompts.PROMPTS["P4"]
        for word in (
            "assume",
            "might",
            "possibly",
            "allegedly",
            "supposedly",
            "reportedly",
            "it is claimed",
        ):
            assert f'"{word}"' in v3, f"V3 dropped {word!r}"
            assert word in p4, f"P4 never warns about {word!r} but V3 checks it"

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


class TestResponseGateIsRetryable:
    """A V3/V4 rejection used to be terminal, which is why runs ended 0/2.

    `generate_family` returned as soon as `_validate_response` failed, and `max_attempts`
    covered only *parse* failures -- so one leakage span cost a whole family with no
    recovery, on prose that was otherwise good. V3 is measurably stochastic: repeat audits
    of a single family returned leakage [0, 3, 0] and organization [3, 3, 4] against a
    floor of 4. The gate is unchanged; the response is now re-asked, exactly as `_plan_ok`
    already did for P3.
    """

    def _flaky_v3(self, cfg, *, leak_until):
        """The dry-run mock, but V3 reports leakage until the given P4 attempt."""
        holder = {}
        base = make_mock_llm(cfg, plan_holder=holder)
        state = {"p4": 0}

        def llm(rendered, *, attempt=0):
            pid = which_prompt(rendered)
            if pid == "P4":
                state["p4"] += 1
            if pid == "V3" and state["p4"] <= leak_until:
                # Quoted from the prose under audit, because `validate._filter_spans` drops
                # a span it cannot find in the response -- V3's prompt requires verbatim
                # quotation. A fixed string absent from the mock's text would be filtered
                # and this fixture would stop simulating leakage at all.
                body = rendered.split('check_response(response="', 1)[-1]
                words = [w for w in body.replace('"', " ").split() if w.isalpha()]
                return json.dumps(
                    {
                        "fluency": 5,
                        "formality": 5,
                        "organization": 5,
                        "leakage": [" ".join(words[:4])],
                        "hedging": [],
                        "artifacts": [],
                    }
                )
            return base(rendered, attempt=attempt)

        return llm, state

    def test_a_leaking_response_is_re_asked(self):
        cfg = _dry_cfg(max_attempts=3)
        llm, state = self._flaky_v3(cfg, leak_until=99)  # never recovers
        res = generate_family("f001", "Archaeology", "CONFLICT", cfg, llm=llm)
        assert not res.admitted
        assert res.stage == "respond"
        # The whole point: the gate failure consumed attempts instead of being discarded.
        assert state["p4"] == 3, f"P4 asked {state['p4']}x, expected max_attempts=3"
        # And the prose the validators judged is still persisted for diagnosis.
        assert res.artifacts.get("rejected_response")

    def test_a_family_that_recovers_on_a_later_attempt_is_admitted(self):
        # The property the retry exists for: a single unlucky flag is survivable.
        cfg = _dry_cfg(max_attempts=3)
        llm, state = self._flaky_v3(cfg, leak_until=1)  # clean from attempt 2
        res = generate_family("f001", "Archaeology", "CONFLICT", cfg, llm=llm)
        assert res.admitted, res.verdict.reason()
        assert state["p4"] == 2

    def test_an_admitted_family_validates_exactly_once(self):
        # Guards against recomputing the verdict after `ask` returns, which would silently
        # double every validator call (1 V1 + N V3 + 1 V4 per evaluation).
        cfg = _dry_cfg(max_attempts=3)
        seen = []
        holder = {}
        base = make_mock_llm(cfg, plan_holder=holder)

        def llm(rendered, *, attempt=0):
            pid = which_prompt(rendered)
            if pid in ("V1", "V3", "V4"):
                seen.append(pid)
            return base(rendered, attempt=attempt)

        res = generate_family("f001", "Archaeology", "CONFLICT", cfg, llm=llm)
        assert res.admitted, res.verdict.reason()
        assert seen.count("V1") == 1
        assert seen.count("V3") == 1
        assert seen.count("V4") == 1


class TestAtomsAreWorldStatesNotFindings:
    """The leakage P4 was rejected for originated in P2's own exemplar.

    Claude f001 was rejected on unanimous 3/3 leakage, the top span being
    "the analysis's flagging of traits incompatible with a modern human affected by
    disease". That is a near-verbatim nominalization of the PLAN's atom 13, "The carpal
    morphology analysis flagged traits incompatible with a modern human affected by
    disease" -- and P4 instruction 2 requires preserving each atom's factual content, so
    P4 had no world-object to name. It complied and leaked.

    The atom shape came from P2 instruction 3(d), which taught the disjunctive pair as
    "two independent checks, at least one of which flagged a defect" and exemplified it
    with "The vibration analysis flagged the defect." / "The metallurgical assay flagged
    the defect." Both live families obeyed.
    """

    def test_p2_no_longer_teaches_detection_procedure_pairs(self):
        p2 = prompts.PROMPTS["P2"]
        assert "flagged the defect" not in p2
        assert "at least one of which flagged" not in p2
        assert "STATE OF THE WORLD" in p2

    def test_the_disjunctive_exemplars_are_world_states(self):
        p2 = prompts.PROMPTS["P2"]
        # Still a genuine disjunction (either may hold, both may hold), but neither atom
        # is an act of detection, so P4 can refer to it without meta-vocabulary.
        assert "[disj-pair-1]" in p2 and "[disj-pair-2]" in p2
        # Only the exemplar CLAIM lines, not the tag-vocabulary line in instruction 4.
        disj = [
            ln
            for ln in p2.split("\n")
            if ln.lstrip().startswith("- ") and "[disj-pair-" in ln
        ]
        assert len(disj) == 2
        for ln in disj:
            assert "analysis" not in ln.lower()
            assert "assay" not in ln.lower()
            assert "flagged" not in ln.lower()

    def test_p4_names_the_subject_matter_not_the_finding_status(self):
        p4 = prompts.PROMPTS["P4"]
        assert "not after its status as a finding" in p4
        # The exact flagged phrasings, as the counter-examples. Whitespace-normalized
        # because the prompt wraps them across lines.
        flat = " ".join(p4.split())
        assert "the analysis's flagging of primitive traits" in flat
        assert "the two claims in circulation" in flat

    def test_summary_phrases_over_both_endpoints_stay_allowed(self):
        # "the two independent flags" was flagged by one auditor, but a plural summary head
        # is the only compact way to scope one connective over two atoms -- banning it would
        # break instruction 3's required "at least one of X or Y".
        p4 = prompts.PROMPTS["P4"]
        assert "A summary phrase covering both endpoints of one" in p4
        assert "at least one of these must hold" in p4

    def test_disjunction_has_a_legal_paraphrase(self):
        # Both hedge rejections realized a Disjunction, across two generators and two runs:
        # "and possibly both" (claude) and "might involve" (deepseek). Instruction 3 offered
        # only "one or both", a bare quantifier fragment that does not attach to a clause,
        # so fluent prose paraphrased it onto a banned word. Widening the ban was measured
        # to swap "possibly" for "might"; supplying a legal alternative is the lever.
        p4 = prompts.PROMPTS["P4"]
        assert "perhaps both" in p4
        for pid in ("P2", "P4", "V3"):
            assert "perhaps" not in prompts.PROMPTS[pid].replace("perhaps both", ""), (
                f"{pid} must not ban the paraphrase it now recommends"
            )

    def test_no_new_prohibitions_were_added(self):
        # Measured: added prohibitions suppress output length (581 -> 308 -> 144 words,
        # deterministic across repeats), and 144 fell under the 500-word floor. Both edits
        # are substitutions, so the count must not grow -- an upper bound, since removing
        # one is the direction that measurement favours.
        import re

        assert len(re.findall(r"Do NOT", prompts.PROMPTS["P4"])) <= 6
        assert len(re.findall(r"(?m)^\d+\. ", prompts.PROMPTS["P4"])) == 8


class TestOnePairOnePlannedRelation:
    """The gold schema allows one relation per atom pair; the parser did not enforce it.

    A stage-contract mismatch, not a model error. `schema.validate_item` rejects a duplicate
    outright -- the Markov network builds one factor per pair, so two edges over the same
    pair have no unambiguous factor table -- while `parse_plan` accepted it and P3 never
    stated the rule. Measured live: a gpt-oss plan carried two relations over one pair,
    cleared EVERY gate (plan, V1, V3, V4 and all seven P5 perturbations) and then died at
    serialization with `duplicate relation a13->a14`, spending the whole family's work.
    Enforcing it at the plan stage makes it a retryable parse complaint instead.
    """

    def _plan_with_duplicate(self, second_sense="Contrast"):
        raw = mock.mock_plan("q", "c", 0)
        plan, err = parse.parse_plan(raw)
        assert err is None
        dup = dict(plan["relations"][0])
        dup["sense"] = second_sense
        dup["validity"] = "valid"
        dup["error_kind"] = None
        dup["level1_coupling"] = None
        return {**plan, "relations": plan["relations"] + [dup]}

    def test_a_duplicate_pair_is_rejected_at_the_plan_stage(self):
        txt = "```json\n" + json.dumps(self._plan_with_duplicate()) + "\n```"
        value, err = parse.parse_plan(txt)
        assert value is None
        assert "two relations" in err
        assert "at most one relation" in err

    def test_the_check_is_direction_insensitive(self):
        # (i,j) and (j,i) collide on the same factor, so order must not evade the check.
        raw = mock.mock_plan("q", "c", 0)
        plan, _ = parse.parse_plan(raw)
        first = plan["relations"][0]
        flipped = dict(first)
        flipped["source_pos"], flipped["target_pos"] = (
            first["target_pos"],
            first["source_pos"],
        )
        flipped["sense"] = "Contrast"
        flipped["validity"] = "valid"
        flipped["error_kind"] = None
        txt = "```json\n" + json.dumps(
            {**plan, "relations": plan["relations"] + [flipped]}
        ) + "\n```"
        value, err = parse.parse_plan(txt)
        assert value is None and "two relations" in err

    def test_the_error_names_both_senses_so_the_retry_is_actionable(self):
        txt = "```json\n" + json.dumps(self._plan_with_duplicate("Concession")) + "\n```"
        _, err = parse.parse_plan(txt)
        assert "Concession" in err

    def test_p3_states_the_rule(self):
        p3 = prompts.PROMPTS["P3"]
        assert "must join a DIFFERENT pair of claims" in p3
        assert "that pair is used up" in p3

    def test_the_parser_and_the_schema_agree_on_this_rule(self):
        """The invariant that was missing: both stages must reject the same plan.

        Six defects in this harness have now had the same shape -- one stage mandates or
        permits what another forbids. Asserting agreement directly is cheaper than
        rediscovering it from a live rejection.
        """
        item = _valid_item()
        item["relations"] = item["relations"] + [dict(item["relations"][0])]
        with pytest.raises(SchemaError, match="duplicate relation"):
            validate_item(item)
        txt = "```json\n" + json.dumps(self._plan_with_duplicate()) + "\n```"
        value, err = parse.parse_plan(txt)
        assert value is None and err, "the parser must reject what the schema rejects"

    def test_the_atom_range_stays_inside_p4s_length_capability(self):
        """P4 compresses rather than expands as the atom count rises, so the ceiling matters.

        Measured on gpt-oss-120b across three runs: 16 atoms -> 581 and 596 words
        (~36 words/atom), 17 -> 516 and 543 (~31), 18 -> **293** (~16), which is under the
        500-word floor and killed the family at `P4: SamplingFailed` with a structurally
        perfect plan. The range's top was 18, i.e. it admitted a regime where P4 reliably
        fails, and nothing in the benchmark needs it -- the corpus property that matters is
        the relation graph, not the atom count.
        """
        assert parse.N_CLAIMS_RANGE == (14, 16)
        # The two copies of this range desynchronize silently: `THRESHOLDS["n_claims"]` is
        # read by nothing, so only a test couples them.
        assert validate.THRESHOLDS["n_claims"] == parse.N_CLAIMS_RANGE

    def test_p3_asks_for_less_than_the_parser_accepts(self):
        # The deliberate asymmetry the repo already uses for relations (prompt says 10,
        # parser takes 8-12): the model gets the easy centre plus slack, and the worked
        # example's own count stays a valid independent witness.
        p3 = prompts.PROMPTS["P3"]
        lo, hi = parse.N_CLAIMS_RANGE
        assert f"Select {lo}-{hi} claims" in p3
        assert "Aim for 15" in p3
        assert lo < 15 < hi or lo <= 15 <= hi

    def test_the_rule_is_satisfiable_within_the_declared_ranges(self):
        # Uniqueness must not make a conforming plan impossible: the pair budget has to
        # cover the largest legal relation + non-relation count from the fewest legal atoms.
        n_atoms = parse.N_CLAIMS_RANGE[0]
        need = parse.N_RELATIONS_RANGE[1] + parse.N_NON_RELATIONS_RANGE[1]
        assert n_atoms * (n_atoms - 1) // 2 >= need

    def test_the_required_senses_draw_on_disjoint_claim_pairs(self):
        # Alternative/Disjunction/Restatement are all mandatory, so if they shared a tagged
        # pair the uniqueness rule would contradict instruction 5. P2 emits three disjoint
        # pair families, so they cannot collide.
        claims, err = parse.parse_claims(mock.mock_claims("q", 0))
        assert err is None
        tags = {c["tag"] for c in claims}
        for family in ("alt-pair-1", "alt-pair-2", "disj-pair-1", "equiv-pair-1"):
            assert family in tags

    def test_the_mock_plan_and_p3s_worked_example_still_conform(self):
        # Both are independent witnesses; if either carried a duplicate pair the new check
        # would make the whole dry-run suite unrunnable.
        plan, err = parse.parse_plan(mock.mock_plan("q", "c", 0))
        assert err is None, err
        pairs = [
            (min(r["source_pos"], r["target_pos"]), max(r["source_pos"], r["target_pos"]))
            for r in plan["relations"]
        ]
        assert len(pairs) == len(set(pairs))


class TestP4RealizesOrderingRelations:
    """Precedence was the one sense whose guidance was mostly a prohibition, and P4 dropped it.

    Measured on gpt-oss-120b f001: a planned ``Precedence(1, 2)`` produced prose with **zero**
    ordering cues -- both atoms asserted in separate paragraphs with no connective, exactly as
    if they were a planned NON-relation. V1 then recovered none of the 2 planned
    Precedence/Succession relations across both families, which reads as a V1 failure but was
    really P4 never writing the relation down.

    The cause is the shape of the instruction, not the model: every other sense in
    instruction 3 lists surface cues, while Precedence listed cues *and then* a "Do NOT write
    'and therefore'" clause. Given that added prohibitions measurably suppress output here,
    writing no connective at all is the safest reading -- and produces an unrealizable
    relation. Rewritten as a positive requirement plus a contrastive example.
    """

    def test_the_ordering_guidance_requires_one_joined_sentence(self):
        p4 = prompts.PROMPTS["P4"]
        assert "state the sequence EXPLICITLY in one sentence" in p4
        assert "which predates" in p4

    def test_it_says_separate_sentences_leave_it_unrealized(self):
        # The actual observed failure mode, named so the writer recognizes it.
        assert "separate sentences leaves this relation" in prompts.PROMPTS["P4"]

    def test_a_bare_or_is_named_as_insufficient(self):
        """Alternative and Disjunction are both mandatory and a bare "or" conflates them.

        Measured on gpt-oss r3 f001: the prose contained **zero** occurrences of "either",
        "at least one of", "one or both" and "perhaps both", realizing a planned Disjunction
        as "X, or they relied heavily on Y". V1 read that as Alternative -- correctly, since
        nothing in the sentence rules out both holding. The markers were already mandated;
        what was missing was the reason they cannot be skipped.
        """
        p4 = prompts.PROMPTS["P4"]
        assert 'A bare "X, or Y" is NOT enough' in p4
        assert "separating Alternative from Disjunction" in p4

    def test_the_prohibition_was_replaced_not_supplemented(self):
        # A substitution: the "Do NOT write 'and therefore'" clause became the positive
        # "not 'X and therefore Y'" contrast inside the example, so the count went DOWN.
        import re

        p4 = prompts.PROMPTS["P4"]
        assert len(re.findall(r"Do NOT", p4)) <= 6
        assert "and therefore" in p4, "the anti-pattern is still named, as a contrast"


class TestV1PrefersTheSpecificSense:
    """V1 over-labelled `Evidence` and never emitted Precedence or Instantiation.

    Measured across both gpt-oss-120b families: **14 recovered `Evidence` against 3 planned**,
    zero `Precedence` against 2 planned, zero `Instantiation` against 1, and 13 of 24
    recovered relations were pairs the plan never related at all. Five senses compile to
    `entailment` and `Evidence` is the broadest, so it was the default for anything
    inferential; meanwhile Precedence/Succession were described only by what they must not
    imply, and Instantiation had no guidance at all.
    """

    def test_it_warns_that_evidence_is_the_broad_default(self):
        v1 = prompts.PROMPTS["V1"]
        assert "MOST SPECIFIC sense" in v1
        assert "broadest" in v1

    def test_the_entailment_senses_get_distinguishing_cues(self):
        v1 = prompts.PROMPTS["V1"]
        for cue in ("led to", "resulted from", "provided that", "for example"):
            assert cue in v1, f"V1 gives no surface cue for {cue!r}"

    def test_ordering_senses_are_required_not_merely_permitted(self):
        v1 = prompts.PROMPTS["V1"]
        assert "These ARE relations and must be" in v1
        for cue in ("subsequently", "earlier", "later"):
            assert cue in v1

    def test_it_discourages_inflating_the_relation_count(self):
        # 12 recovered for a 10-relation plan, most of them spurious.
        assert "Emitting many relations does not make" in prompts.PROMPTS["V1"]

    def test_concession_has_its_own_cues(self):
        """V1 had no cue for Concession at all, and duly missed one P4 wrote correctly.

        Measured on gpt-oss r3 f001: P4 realized the planned `Concession(14,15)` as
        "...+2 per mil, indicating a purely terrestrial diet, despite the limitation of
        microwear" -- `despite` is a marker P4 mandates -- and V1 did not recover it. The
        Contrast rule mentioned "without either being conceded", implying Concession exists,
        while never saying what it looks like.
        """
        v1 = prompts.PROMPTS["V1"]
        for cue in ("although", "despite", "even though"):
            assert cue in v1, f"V1 gives no Concession cue {cue!r}"
        # The subordinate-clause case is the one that was missed.
        assert "subordinate clause" in v1

    def test_the_compilation_map_still_agrees_with_the_taxonomy(self):
        # The added cues name senses; the coupling map must still be the authority and must
        # still be stated correctly for every sense V1 may emit.
        from fact_reasoner.locobench.taxonomy_bridge import coupling_for_sense

        v1 = prompts.PROMPTS["V1"]
        for sense in ("Cause-Effect", "Evidence", "Condition", "Instantiation"):
            assert coupling_for_sense(sense) == "entailment"
            assert sense in v1
        assert coupling_for_sense("Precedence") == "none"


class TestPromptTexParity:
    """The tex holds the prompts verbatim; nothing guarded that until now.

    Widened from five to all of them after a P2 edit: every prompt was already
    byte-identical, but only P3/P4/V1/V3/V4 were checked, so editing P2 would have silently
    desynced the document from the code.

    Parametrized over `prompts.PROMPTS` rather than a hard-coded list, so removing a prompt
    (V2) or adding one cannot leave a stale entry behind. The tex is the Phase 1 design
    record and still documents V2; that is deliberate -- it records what was specified,
    while this test only asserts that prompts the harness SHIPS match it.
    """

    @pytest.mark.parametrize("pid", sorted(prompts.PROMPTS))
    def test_the_prompt_matches_the_phase_one_document(self, pid):
        tex = open(
            "docs/ideation/coherence/benchmark/locobench_phase1.tex", encoding="utf-8"
        ).read()
        assert prompts.PROMPTS[pid].strip() in tex, (
            f"{pid} has drifted from locobench_phase1.tex; update the verbatim block"
        )


class TestDefectTwoPerRungRelations:
    """A rung's gold relations describe that rung, not the base plan (Defect 2).

    Before the fix, `pipeline` copied the base plan's relation list into all five items of
    a family: every rung's TEXT was perturbed but its LABELS were not. The whole family
    then scored identically on every readout, and the C1/C3 strict-increase constraints
    could not hold by construction.
    """

    @staticmethod
    def _base_relations():
        """A small edge set with the shapes the CONFLICT ladder needs."""
        return [
            {
                "id": "r000", "source_id": "a0", "target_id": "a1",
                "level2_sense": "Cause-Effect", "level1_coupling": "entailment",
                "directed": True, "ordering_only": False,
                "intended_strength_band": "strong", "strength_range": [0.85, 1.0],
                "validity": "valid", "error_kind": None, "is_concession": False,
                "is_resolved_concession": False, "resolver_atom_id": None,
            },
            {
                "id": "r001", "source_id": "a2", "target_id": "a3",
                "level2_sense": "Alternative", "level1_coupling": "exclusive",
                "directed": False, "ordering_only": False,
                "intended_strength_band": "strong", "strength_range": [0.85, 1.0],
                "validity": "valid", "error_kind": None, "is_concession": False,
                "is_resolved_concession": False, "resolver_atom_id": None,
                "exhaustive": True,
            },
            {
                "id": "r002", "source_id": "a4", "target_id": "a5",
                "level2_sense": "Contrast", "level1_coupling": "contradiction",
                "directed": False, "ordering_only": False,
                "intended_strength_band": "weak", "strength_range": [0.35, 0.59],
                "validity": "invalid", "error_kind": "false_endpoint",
                "is_concession": False, "is_resolved_concession": False,
                "resolver_atom_id": None, "exhaustive": False,
            },
            {
                "id": "r003", "source_id": "a6", "target_id": "a7",
                "level2_sense": "Precedence", "level1_coupling": "none",
                "directed": True, "ordering_only": True,
                "intended_strength_band": "strong", "strength_range": [0.85, 1.0],
                "validity": "valid", "error_kind": None, "is_concession": False,
                "is_resolved_concession": False, "resolver_atom_id": None,
            },
        ]

    # -- the target chooser ---------------------------------------------------

    def test_successive_drops_target_different_edges(self):
        """The old code passed the literal `r000` to every call, so a two-drop rung
        asked twice for the same edge and the second call had nothing to do."""
        targets = perturb.plan_targets("CONFLICT", self._base_relations())
        coherent = targets[4]  # add_resolution, drop_relation, drop_relation
        assert len(coherent) == 3
        named = [t for t in coherent if t]
        assert len(named) == len(set(named)), (
            f"a rung must not target the same edge twice: {coherent}"
        )

    def test_no_call_targets_the_hardcoded_r000_by_default(self):
        """r000 is an entailment here; a conflict-oriented call must not pick it."""
        targets = perturb.plan_targets("CONFLICT", self._base_relations())
        assert "r000" not in targets[4]

    def test_drop_prefers_the_invalid_conflict(self):
        """The planted-invalid conflict is the one a "fix" should remove first."""
        targets = perturb.plan_targets("CONFLICT", self._base_relations())
        # rung 3 = add_resolution + drop_relation.
        assert targets[3][1] == "r002"

    def test_resolution_targets_a_valid_conflict(self):
        """Resolving a planted error would dress up a mistake, not settle a tension."""
        rels = self._base_relations()
        targets = perturb.plan_targets("CONFLICT", rels)
        resolved_id = targets[2][0]
        edge = next(e for e in rels if e["id"] == resolved_id)
        assert edge["validity"] == "valid"
        assert edge["level1_coupling"] in ("contradiction", "exclusive")

    # -- a perturbation must actually change the text --------------------------

    def test_gate_text_changed_rejects_an_unchanged_response(self):
        """A P5 call that returns the parent prose did not happen.

        Measured on f013: `add_resolution` set the resolution flag on the gold edge but
        returned the base response verbatim, so rungs 1 and 2 shipped identical text with
        differing labels. The adjacency gate passed it -- it compares edge SIGNATURES, and
        those did differ -- leaving a `c1` strict-increase assertion that a gold-arm readout
        can satisfy and a mined arm never can. Corpus ceiling 201/202.
        """
        same = validate.gate_text_changed("a b c", "a b c", operator="add_resolution(r9)")
        assert not same.passed
        assert "unchanged" in same.detail
        assert validate.gate_text_changed("a b c", "a b d", operator="x").passed

    def test_gate_text_changed_ignores_reflowing(self):
        """A response that only had its whitespace redone is still not an edit."""
        assert not validate.gate_text_changed("a  b\nc", "a b c", operator="x").passed

    def test_every_call_is_a_text_edit_so_none_is_exempt(self):
        """The gate applies to EVERY call, unlike the adjacency gate.

        `shuffle_order` and `ordering_only` are exempt from the edge-effect check because
        they are factor-invariant by design -- but both still rewrite the prose (one
        reorders sentences, the other swaps a connective), so neither may return it
        unchanged. A text-side exemption list would silently readmit the f013 defect.
        """
        for call in perturb.ALL_CALLS:
            assert not validate.gate_text_changed(
                "x y z", "x y z", operator=f"{call}(r0)"
            ).passed, f"{call} must not be allowed to leave the prose unchanged"

    # -- P3 must plan enough droppable conflict edges --------------------------

    def test_p3_demands_four_conflict_edges_with_at_most_one_resolved(self):
        """A CONFLICT ladder's deepest rung needs `add_resolution` plus TWO DISTINCT
        `drop_relation` targets, and the resolution consumes one edge from the same pool.

        Only three senses compile to a conflict coupling (Alternative -> exclusive,
        Concession and Contrast -> contradiction), so instruction 5's one-of-each mandate
        yields exactly three -- and if the Concession is resolved, only TWO are droppable
        and rung 4 collapses onto rung 3. Measured live: droppable==3 admitted 4/4 and
        droppable==2 was rejected 4/4 across 8 distinct topics.
        """
        head = prompts.PROMPTS["P3"].split("```json")[0]
        assert "AT LEAST FOUR" in head
        assert "AT MOST" in head and "resolved Concession" in head

    def test_parse_plan_rejects_too_few_droppable_conflicts(self):
        """The prompt alone did not work, so the PARSER enforces the pool.

        Instruction 5 was extended to demand >= 4 conflict edges with <= 1 resolved, and a
        live model ignored it on six consecutive families across five topics -- each time
        planning the same three-edge shape (Alternative + resolved Concession + Contrast),
        droppable 2, rejected at `P5.rung4.edge_effect` only AFTER the respond stage had
        been paid for. As a parser error the complaint reaches `_Caller.ask` and re-plans
        inside the same call.
        """
        import json as _json
        import re as _re

        block = _re.search(r"```json(.*?)```", prompts.PROMPTS["P3"], _re.S)
        good = _json.loads(block.group(1))

        # The worked example is admissible as shipped.
        parsed, err = parse.parse_plan("```json" + _json.dumps(good) + "```")
        assert parsed is not None, err

        # Downgrade its fourth conflict edge and the plan must be refused.
        bad = _json.loads(_json.dumps(good))
        for r in bad["relations"]:
            if r["sense"] == "Contrast" and r["source_pos"] == 4:
                r["sense"] = "Instantiation"
        parsed, err = parse.parse_plan("```json" + _json.dumps(bad) + "```")
        assert parsed is None
        assert "conflicting relation" in err and "unresolved" in err

    def test_p3_worked_example_is_itself_admissible(self):
        """The example in instruction 10 must satisfy the rule instruction 5 states.

        It previously carried exactly the failing shape -- 3 conflict edges, one of them a
        resolved Concession, so droppable==2 -- which taught the model the pattern that the
        adjacency gate then rejected.
        """
        import json as _json
        import re as _re

        from fact_reasoner.locobench.taxonomy_bridge import coupling_for_sense

        block = _re.search(r"```json(.*?)```", prompts.PROMPTS["P3"], _re.S)
        plan = _json.loads(block.group(1))
        conflicts = [
            r
            for r in plan["relations"]
            if coupling_for_sense(r["sense"]) in ("contradiction", "exclusive")
        ]
        resolved = [r for r in conflicts if r.get("resolved")]
        assert len(conflicts) >= 4, (
            f"the example plans only {len(conflicts)} conflict edges; a CONFLICT ladder "
            "needs 4 so that 3 survive add_resolution"
        )
        assert len(resolved) <= 1
        assert len(conflicts) - len(resolved) >= 3
        # and it must still honour instruction 8's exact split
        assert sum(1 for r in plan["relations"] if r["validity"] == "valid") == 6
        assert sum(1 for r in plan["relations"] if r["validity"] == "invalid") == 4

    # -- the ORDER shuffle levels ---------------------------------------------

    def test_each_order_shuffle_rung_gets_a_distinct_level(self):
        """The P5 prompt's signature is `shuffle_order(level)`, and the three ORDER
        shuffle rungs differ ONLY by that level.

        Before the fix they fell through to the generic branch and were all handed the
        first unused edge id, so every rung rendered as `shuffle_order(r000)`. A capable
        model given the same prompt three times returned byte-identical prose, which made
        the ladder's strict-increase C1 pairs (0,1) and (1,2) unsatisfiable by
        construction -- measured live as a 9/15 ceiling on f004.
        """
        targets = perturb.plan_targets("ORDER", self._base_relations())
        lad = perturb.ladder_for("ORDER")
        levels = {
            r.name: targets[r.index][0]
            for r in lad.rungs
            if r.calls == ("shuffle_order",)
        }
        assert levels == {
            "shuffle_full": "full",
            "shuffle_block": "block",
            "shuffle_adjacent": "adjacent",
        }, f"each shuffle rung needs its own level, got {levels}"

    def test_a_shuffle_level_is_never_an_edge_id(self):
        """A level in the target slot must not be mistaken for an edge."""
        rels = self._base_relations()
        targets = perturb.plan_targets("ORDER", rels)
        ids = {str(e["id"]) for e in rels}
        lad = perturb.ladder_for("ORDER")
        for r in lad.rungs:
            if r.calls != ("shuffle_order",):
                continue
            assert targets[r.index][0] not in ids

    def test_shuffle_level_leaves_the_gold_edge_set_untouched(self):
        """`shuffle_order` is edge-invariant BY DESIGN, so passing a level rather than an
        edge id must not change the labels -- only the prompt."""
        rels = self._base_relations()
        before = [dict(e) for e in rels]
        out, _nons, log = perturb.apply_calls(rels, ("shuffle_order",), targets=["full"])
        assert {e["id"] for e in out} == {e["id"] for e in before}
        assert [e["level2_sense"] for e in out] == [e["level2_sense"] for e in before]
        assert log[0]["effect"] == "reordered"

    def test_ordering_only_still_targets_a_real_edge(self):
        """The shuffle fix must not disturb the OTHER edge-invariant call, which really
        does need an edge (it flips that edge's Precedence/Succession sense)."""
        rels = self._base_relations()
        for fam in ("ORDER", "CONTROL"):
            targets = perturb.plan_targets(fam, rels)
            lad = perturb.ladder_for(fam)
            ids = {str(e["id"]) for e in rels}
            for r in lad.rungs:
                if "ordering_only" not in r.calls:
                    continue
                got = targets[r.index][r.calls.index("ordering_only")]
                assert got == "" or got in ids, (
                    f"{fam} rung {r.name} ordering_only target {got!r} is not an edge id"
                )

    # -- the edge-set transforms ---------------------------------------------

    def test_drop_relation_removes_exactly_one_edge(self):
        rels = self._base_relations()
        out, _nons, log = perturb.apply_calls(rels, ("drop_relation",), targets=["r002"])
        assert len(out) == len(rels) - 1
        assert "r002" not in {e["id"] for e in out}
        assert log[0]["effect"] == "removed"

    def test_add_resolution_only_sets_the_flag(self):
        """It must NOT retype the edge: turning an exclusive into a contradiction would
        change which factor table the MRF builds, making the rung differ from its parent
        by a coupling change rather than by a resolution."""
        rels = self._base_relations()
        out, _nons, log = perturb.apply_calls(
            rels, ("add_resolution",), targets=["r001"]
        )
        edge = next(e for e in out if e["id"] == "r001")
        assert edge["is_resolved_concession"] is True
        assert edge["level2_sense"] == "Alternative"      # unchanged
        assert edge["level1_coupling"] == "exclusive"     # unchanged
        assert log[0]["effect"] == "resolved"

    def test_apply_calls_does_not_mutate_its_input(self):
        rels = self._base_relations()
        before = json.dumps(rels, sort_keys=True)
        perturb.apply_calls(rels, ("drop_relation",), targets=["r002"])
        assert json.dumps(rels, sort_keys=True) == before

    def test_spurious_relation_adds_an_invalid_edge_and_frees_the_non_relation(self):
        """An added edge on a declared non-relation pair would leave the item asserting
        the pair both is and is not related, which `validate_item` rejects."""
        rels = self._base_relations()
        nons = [{"source_id": "a0", "target_id": "a5", "position_distance": 5}]
        out, out_nons, log = perturb.apply_calls(
            rels, ("spurious_relation",), non_relations=nons
        )
        assert len(out) == len(rels) + 1
        added = out[-1]
        assert (added["source_id"], added["target_id"]) == ("a0", "a5")
        assert added["validity"] == "invalid"
        assert out_nons == []          # the pair is no longer a declared non-relation
        assert log[0]["effect"] == "added"

    def test_wrong_sense_relabels_and_marks_the_edge_invalid(self):
        out, _nons, log = perturb.apply_calls(
            self._base_relations(), ("wrong_sense",), targets=["r000"]
        )
        edge = next(e for e in out if e["id"] == "r000")
        assert edge["level2_sense"] != "Cause-Effect"
        assert edge["validity"] == "invalid"
        assert edge["error_kind"] == "wrong_sense"
        assert log[0]["effect"] == "relabeled"

    def test_every_transform_keeps_gold_consistent_with_the_taxonomy(self):
        """The builder assertion in `schema.py` must have nothing to catch: a retyped
        edge's coupling and derived flags are re-derived from its new sense."""
        from fact_reasoner.locobench.taxonomy_bridge import (
            coupling_for_sense,
            is_directed,
        )

        for call in perturb.ALL_CALLS:
            rels = self._base_relations()
            targets = ["r001"] if call != "spurious_relation" else [""]
            out, _nons, _log = perturb.apply_calls(
                rels,
                (call,),
                targets=targets,
                non_relations=[
                    {"source_id": "a0", "target_id": "a5", "position_distance": 5}
                ],
            )
            for edge in out:
                sense = edge["level2_sense"]
                assert edge["level1_coupling"] == coupling_for_sense(sense), (
                    f"{call} left {edge['id']} with a coupling COMPILE disagrees with"
                )
                assert bool(edge["directed"]) == is_directed(sense)

    def test_ordering_only_flip_keeps_the_factor_set_unchanged(self):
        """Precedence and Succession both compile to Level-1 `none`, so an ORDER or
        CONTROL rung built on this edit is factor-invariant by design."""
        out, _nons, log = perturb.apply_calls(
            self._base_relations(), ("ordering_only",), targets=["r003"]
        )
        edge = next(e for e in out if e["id"] == "r003")
        assert edge["level2_sense"] == "Succession"
        assert edge["level1_coupling"] == "none"
        assert log[0]["effect"] == "reordered"

    def test_unknown_call_raises(self):
        with pytest.raises(ValueError, match="Unknown perturbation call"):
            perturb.apply_calls(self._base_relations(), ("teleport",))

    def test_edge_invariant_calls_are_named_not_implicit(self):
        assert set(perturb.EDGE_INVARIANT_CALLS) == {"shuffle_order", "ordering_only"}

    # -- end to end through the pipeline -------------------------------------

    @pytest.mark.parametrize("family", ["CONFLICT", "CHAIN"])
    def test_a_conflict_or_chain_ladder_gets_five_distinct_edge_sets(
        self, family, tmp_path
    ):
        """The regression itself: five rungs, five different gold relation sets."""
        cfg = GenConfig(
            dataset_name="t", out_dir=str(tmp_path), dry_run=True
        )
        res = pipeline.generate_family("f001", "Anthropology", family, cfg)
        assert res.admitted, res.verdict.reason()
        sigs = {
            pipeline._edge_set_signature(it["relations"]) for it in res.items
        }
        assert len(sigs) == 5, (
            f"{family}: {len(sigs)} distinct edge sets across 5 rungs; before the fix "
            "this was 1"
        )

    @pytest.mark.parametrize("family", ["ORDER", "CONTROL"])
    def test_a_factor_invariant_ladder_still_admits(self, family, tmp_path):
        """ORDER's shuffle rungs and CONTROL's edits are meaning-preserving and add no
        factor, which is exactly what those ladders test. The per-rung edge-effect gate
        must exempt them rather than reject the families that check the invariance."""
        cfg = GenConfig(dataset_name="t", out_dir=str(tmp_path), dry_run=True)
        res = pipeline.generate_family("f001", "Anthropology", family, cfg)
        assert res.admitted, res.verdict.reason()

    def test_each_rung_records_the_edges_it_targeted(self, tmp_path):
        cfg = GenConfig(dataset_name="t", out_dir=str(tmp_path), dry_run=True)
        res = pipeline.generate_family("f001", "Anthropology", "CONFLICT", cfg)
        assert res.admitted, res.verdict.reason()
        for item in res.items:
            pert = item["expected"]["perturbation"]
            assert "targets" in pert and "edge_effects" in pert
            assert len(pert["edge_effects"]) == len(pert["calls"])

    def test_the_dry_run_plan_carries_enough_conflicts_for_the_deepest_rung(self):
        """CONFLICT's `coherent` rung composes add_resolution + 2x drop_relation, so the
        plan needs three unresolved conflict edges. With two, rung 4's edge set collapsed
        onto rung 3's -- Defect 2 one rung at a time -- and the fixture silently stopped
        exercising the deepest rung."""
        plan = json.loads(
            mock.mock_plan("q", "claims").split("```json")[1].split("```")[0]
        )
        from fact_reasoner.locobench.taxonomy_bridge import coupling_for_sense

        unresolved = [
            r
            for r in plan["relations"]
            if coupling_for_sense(r["sense"]) in ("contradiction", "exclusive")
            and not r.get("resolved")
        ]
        assert len(unresolved) >= 3, (
            f"only {len(unresolved)} unresolved conflict edges in the dry-run plan; "
            "the CONFLICT ladder's deepest rung needs 3"
        )

    def test_the_edge_effect_gate_fires_when_a_rung_matches_its_parent(self):
        """The gate's own logic, exercised directly: a rung claiming an edge-set change
        whose relations equal its parent's must be caught."""
        rels = self._base_relations()
        # `drop_relation` with no eligible target is a no-op, so the edge set is unchanged
        # while the call still claims a change.
        out, _nons, log = perturb.apply_calls(
            rels, ("drop_relation",), targets=[""]
        )
        assert log[0]["effect"] == "noop"
        assert pipeline._edge_set_signature(out) == pipeline._edge_set_signature(rels)
        assert any(
            call not in perturb.EDGE_INVARIANT_CALLS for call in ("drop_relation",)
        ), "drop_relation must not be exempt from the edge-effect gate"

    def test_the_gate_compares_adjacent_rungs_not_parentage(self):
        """Every CONFLICT rung's `parent` is the base, so a parent-only comparison passes
        a rung that duplicates the rung immediately BELOW it -- and the C1 constraints are
        stated over adjacent pairs. A plan with only two unresolved conflicts cannot feed
        the `coherent` rung's third call, so rungs 3 and 4 collide and must be rejected."""
        # Two unresolved conflicts only: one for add_resolution, one for the first drop.
        rels = [
            e
            for e in self._base_relations()
            if e["level1_coupling"] != "contradiction"
        ] + [
            {
                "id": "r010", "source_id": "a8", "target_id": "a9",
                "level2_sense": "Contrast", "level1_coupling": "contradiction",
                "directed": False, "ordering_only": False,
                "intended_strength_band": "weak", "strength_range": [0.35, 0.59],
                "validity": "invalid", "error_kind": "spurious",
                "is_concession": False, "is_resolved_concession": False,
                "resolver_atom_id": None, "exhaustive": False,
            },
        ]
        unresolved = [
            e
            for e in rels
            if e["level1_coupling"] in ("contradiction", "exclusive")
            and not e["is_resolved_concession"]
        ]
        assert len(unresolved) == 2, "fixture must have exactly two to force the collision"

        lad = perturb.ladder_for("CONFLICT")
        targets = perturb.plan_targets("CONFLICT", rels)
        sigs = {}
        for rung in lad.rungs:
            new = (
                rels
                if rung.is_base
                else perturb.apply_calls(
                    rels, rung.calls, targets=targets[rung.index]
                )[0]
            )
            sigs[rung.index] = pipeline._edge_set_signature(new)

        # Rungs 3 and 4 collide...
        assert sigs[3] == sigs[4]
        # ...even though rung 4 differs from its declared parent, the base.
        parent = next(r.parent for r in lad.rungs if r.index == 4)
        assert parent == 1
        assert sigs[4] != sigs[parent], (
            "this is precisely why a parent-only comparison misses the collision"
        )
