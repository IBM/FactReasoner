# coding=utf-8
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

"""Offline tests for scripts/gen_incoherent_responses.py (no LLM, no Merlin).

Covers the parts that can silently produce a useless dataset: the ConflictBank record
contract (the atom id is index-dependent, NOT always ``a0``), the prompt's hard
anti-hedging instructions, and response cleanup.
"""

import importlib.util
import json
import os

import pytest

_SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts",
    "gen_incoherent_responses.py",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("gen_incoherent", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load_module()


def _record(idx: int) -> dict:
    """A minimal ConflictBank record whose atom id is index-dependent."""
    aid = f"a{idx}"
    return {
        "input": f"Tell me about subject {idx}.",
        "output": "",
        "topic": f"Subject {idx}",
        "atoms": [
            {
                "id": aid,
                "text": f"Subject {idx} is a real thing.",
                "label": "S",
                "contexts": [f"c_{aid}_0", f"c_{aid}_1"],
            }
        ],
        "contexts": [
            {"id": f"c_{aid}_0", "title": "t", "text": "Supporting evidence."},
            {"id": f"c_{aid}_1", "title": "t", "text": "Fabricated counter-story."},
        ],
    }


def _write(tmp_path, rows):
    p = tmp_path / "cb.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(p)


class TestLoadRecords:
    def test_atom_id_is_not_hardcoded_to_a0(self, gen, tmp_path):
        """The regression this guards: record 7's atom is a7, not a0."""
        path = _write(tmp_path, [_record(i) for i in range(8)])
        recs = gen.load_records(path)
        assert [r["atom_id"] for r in recs] == [f"a{i}" for i in range(8)]
        last = recs[-1]
        assert last["context_supporting"] == "c_a7_0"
        assert last["context_conflicting"] == "c_a7_1"

    def test_context_order_follows_the_atom(self, gen, tmp_path):
        """Supporting/conflicting order comes from the atom's own list, not sorting."""
        row = _record(0)
        row["contexts"].reverse()  # storage order differs from the atom's order
        recs = gen.load_records(_write(tmp_path, [row]))
        assert recs[0]["context_supporting"] == "c_a0_0"
        assert recs[0]["support_text"] == "Supporting evidence."
        assert recs[0]["conflict_text"] == "Fabricated counter-story."

    def test_ids_are_stable_and_sequential(self, gen, tmp_path):
        recs = gen.load_records(_write(tmp_path, [_record(i) for i in range(3)]))
        assert [r["id"] for r in recs] == ["cb-000", "cb-001", "cb-002"]

    @pytest.mark.parametrize(
        "mutate,msg",
        [
            (lambda r: r["atoms"].append(r["atoms"][0]), "exactly 1 atom"),
            (lambda r: r["atoms"][0]["contexts"].pop(), "expected 2"),
            (lambda r: r["contexts"].pop(), "absent from the record"),
        ],
    )
    def test_malformed_records_fail_loudly(self, gen, tmp_path, mutate, msg):
        """A bad input must stop the run, not yield junk responses."""
        row = _record(0)
        mutate(row)
        with pytest.raises(SystemExit, match=msg):
            gen.load_records(_write(tmp_path, [row]))

    def test_empty_file_is_an_error(self, gen, tmp_path):
        with pytest.raises(SystemExit, match="no records"):
            gen.load_records(_write(tmp_path, []))


class TestPrompt:
    def test_carries_claim_and_both_contexts(self, gen, tmp_path):
        rec = gen.load_records(_write(tmp_path, [_record(0)]))[0]
        prompt = gen.build_prompt(rec)
        assert rec["claim"] in prompt
        assert "Supporting evidence." in prompt
        assert "Fabricated counter-story." in prompt

    def test_forbids_hedging_and_resolution(self, gen, tmp_path):
        """Hedged prose reads as a correct report about a disputed claim, not as
        incoherence, so the ban has to be explicit in the prompt."""
        prompt = gen.build_prompt(gen.load_records(_write(tmp_path, [_record(0)]))[0])
        assert "reportedly" in prompt
        assert "Do NOT hedge" in prompt
        assert "Do NOT resolve the contradiction" in prompt

    def test_truncates_long_contexts(self, gen, tmp_path):
        row = _record(0)
        row["contexts"][0]["text"] = "x" * 50_000
        rec = gen.load_records(_write(tmp_path, [row]))[0]
        assert len(gen.build_prompt(rec, max_context_chars=100)) < 5_000


class TestInventedSubtlePattern:
    """The v2 pattern: plant a derivable inconsistency and never name it.

    The two constraints are in tension and both were measured: naming the conflict makes
    an LLM judge catch it (v1 scored judge_direct 0.25), while smoothing it away makes the
    relation miner extract zero conflict edges (a smooth probe scored mean_marginal 0.634
    with 0 conflicts -- fooling the metric too).
    """

    def test_prompt_bans_the_judge_tells(self, gen, tmp_path):
        rec = gen.load_records(_write(tmp_path, [_record(0)]))[0]
        prompt = gen.build_prompt_invented(rec, conflict_type="numeric")
        for banned in ("mutually exclusive", "cannot both", "contradicts", "therefore"):
            assert banned in prompt, f"prompt must explicitly ban {banned!r}"
        assert "NEVER signal" in prompt

    def test_prompt_demands_two_explicit_assertions(self, gen, tmp_path):
        """Without this the miner finds no conflict edge -- the measured failure."""
        rec = gen.load_records(_write(tmp_path, [_record(0)]))[0]
        prompt = gen.build_prompt_invented(rec, conflict_type="numeric")
        assert "TWO SEPARATE, PLAINLY STATED sentences" in prompt
        assert "EXACTLY ONE inconsistency" in prompt

    def test_prompt_does_not_leak_the_fabricated_context(self, gen, tmp_path):
        """v2 invents its own conflict; the fabricated context is what a judge catches."""
        rec = gen.load_records(_write(tmp_path, [_record(0)]))[0]
        prompt = gen.build_prompt_invented(rec, conflict_type="numeric")
        assert rec["support_text"] in prompt
        assert rec["conflict_text"] not in prompt

    @pytest.mark.parametrize(
        "kind,marker",
        [
            ("numeric", "NUMERIC"),
            ("chronology", "CHRONOLOGICAL"),
            ("quantifier", "QUANTIFIER"),
        ],
    )
    def test_each_conflict_type_has_its_own_instruction(
        self, gen, tmp_path, kind, marker
    ):
        rec = gen.load_records(_write(tmp_path, [_record(0)]))[0]
        prompt = gen.build_prompt_invented(rec, conflict_type=kind)
        assert marker in prompt
        # and the other two kinds' instructions are absent
        others = {"NUMERIC", "CHRONOLOGICAL", "QUANTIFIER"} - {marker}
        assert not (others & set(prompt.split()))

    def test_auto_uses_only_the_representable_conflict_type(self, gen, tmp_path):
        """`auto` must not emit numeric/chronology.

        Measured: a numeric conflict is split across THREE atoms ("356 teams" /
        "32 conferences" / "10 per conference"), so no pair contradicts and a pairwise
        MRF cannot represent it -- verified with all_pairs, which compares every pair and
        still mined 0 conflict edges. Chronology returns NO RELATION even on the isolated
        pair. Rotating over them would silently yield passages with no conflict at all.
        """
        recs = gen.load_records(_write(tmp_path, [_record(i) for i in range(6)]))
        kinds = [gen.pick_conflict_type(r, "auto") for r in recs]
        assert set(kinds) == {"quantifier"}
        # An explicit request still wins, so the ablation stays available.
        assert gen.pick_conflict_type(recs[0], "numeric") == "numeric"
        assert gen.pick_conflict_type(recs[0], "chronology") == "chronology"

    def test_quantifier_leads_the_type_list(self, gen):
        """The working type is the default, so a bare --conflict-type choice is safe."""
        assert gen.CONFLICT_TYPES[0] == "quantifier"


class TestControlPattern:
    """The matched coherent control. Without it the incoherent numbers have no
    reference -- LCS has no absolute zero."""

    def test_control_uses_only_the_supporting_context(self, gen, tmp_path):
        rec = gen.load_records(_write(tmp_path, [_record(0)]))[0]
        prompt = gen.build_prompt_control(rec)
        assert rec["support_text"] in prompt
        assert rec["conflict_text"] not in prompt

    def test_control_demands_consistency(self, gen, tmp_path):
        rec = gen.load_records(_write(tmp_path, [_record(0)]))[0]
        prompt = gen.build_prompt_control(rec)
        assert "Do not contradict" in prompt

    def test_control_is_selectable(self, gen):
        assert gen.parse_args(["--pattern", "control"]).pattern == "control"


class TestTellDetection:
    """`find_tells` is the v2 quality gate: a leaked tell means the prompt failed."""

    @pytest.mark.parametrize(
        "text,hit",
        [
            ("The two statements are mutually exclusive.", True),
            ("She cannot both play and not play.", True),
            ("Therefore she does not play.", True),
            ("This is logically inconsistent.", True),
            ("She played 31 games. The season had 30 games.", False),
        ],
    )
    def test_find_tells(self, gen, text, hit):
        assert bool(gen.find_tells(text)) is hit


class TestHedgeDetection:
    @pytest.mark.parametrize(
        "text,hit",
        [
            ("It is reportedly the case.", True),
            ("Some sources say otherwise.", True),
            ("She allegedly joined the team.", True),
            ("She plays in the league. She does not play in the league.", False),
        ],
    )
    def test_find_hedges(self, gen, text, hit):
        assert bool(gen.find_hedges(text)) is hit


class TestCleanResponse:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("  A passage.  ", "A passage."),
            ("Passage:\nA passage.", "A passage."),
            ("```\nA passage.\n```", "A passage."),
            ("A passage.", "A passage."),
        ],
    )
    def test_strips_wrappers(self, gen, raw, expected):
        assert gen._clean_response(raw) == expected

    def test_empty_stays_empty(self, gen):
        """main() turns this into a recorded per-item error rather than a bad row."""
        assert gen._clean_response("") == ""
        assert gen._clean_response(None) == ""


class TestCLI:
    def test_scoring_without_merlin_is_refused(self, gen, monkeypatch):
        """Fail before spending any API calls, not partway through."""
        monkeypatch.delenv("MERLIN_PATH", raising=False)
        with pytest.raises(SystemExit, match="Merlin"):
            gen.main(["--score", "--limit", "1"])

    def test_frontier_and_rits_are_mutually_exclusive(self, gen):
        with pytest.raises(SystemExit, match="not both"):
            gen.main(["--frontier", "claude", "--rits-model", "gpt-oss-120b-a100"])

    def test_unknown_rits_model_lists_the_valid_ones(self, gen):
        with pytest.raises(SystemExit, match="Unknown --rits-model"):
            gen.main(["--rits-model", "no-such-model"])

    def test_frontier_defaults(self, gen):
        args = gen.parse_args(["--frontier", "claude"])
        assert gen._apply_frontier(args) is True
        assert args.backend == "openai"
        assert args.model_id == gen.FRONTIER_MODELS["claude"]
        assert args.base_url == gen.GATEWAY_BASE_URL

    def test_pattern_default_is_invented_subtle(self, gen):
        """v1's pattern stays available but is no longer the default: it scores
        judge_direct 0.25, i.e. a judge catches it as easily as LCS does."""
        args = gen.parse_args([])
        assert args.pattern == "invented-subtle"
        assert args.conflict_type == "auto"
        assert gen.PATTERNS == (
            "invented-subtle",
            "assert-both-then-negate",
            "control",
        )

    def test_v1_pattern_is_still_selectable(self, gen):
        args = gen.parse_args(["--pattern", "assert-both-then-negate"])
        assert args.pattern == "assert-both-then-negate"

    def test_mining_defaults_avoid_the_degenerate_config(self, gen):
        """all_pairs+logprobs saturates every weight and drives mm to exactly 0.0 on
        this material; the defaults must not be those."""
        args = gen.parse_args([])
        assert args.pair_policy == "windowed"
        assert args.strength_method == "verbalized"
