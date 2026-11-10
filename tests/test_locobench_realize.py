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

"""Rung realization: span gate, V5 verdict, target screening, gated rungs.

These pin the behaviour the dev-set repair depends on. The motivating measurement: a
blind side-by-side reading found drop_relation realized in 0/18 dev and 31/113 release
transitions, its conflict sentence surviving verbatim in 17/18 dev drops.
"""

import json
import os
import random

import pytest

from fact_reasoner.locobench import parse, prompts, realize, validate
from fact_reasoner.locobench.perturb import (
    apply_calls,
    eligible_targets,
    ladder_for,
    plan_targets,
)

CONFLICT = ("conflict", False)
RESOLVED = ("conflict", True)
NONE = ("none", None)


def _reading(lo, hi):
    blk = lambda r: {"relation": r[0], "resolved": r[1], "evidence": ""}  # noqa: E731
    return {"text_1": blk(lo), "text_2": blk(hi), "differs": lo != hi, "change": ""}


# ---------------------------------------------------------------------------
# Span gate.
# ---------------------------------------------------------------------------


class TestSpanGate:
    PARENT = "Bone loses collagen. By contrast, buried bone stays plastic. Done."

    def test_a_drop_whose_span_survives_is_rejected(self):
        # The measured dev failure: the conflict sentence left verbatim.
        out = "Bone loses collagen. By contrast, buried bone stays plastic. Done!"
        g = validate.gate_spans(
            self.PARENT, out, {"spans_removed": ["By contrast,"]}, operator="drop_relation"
        )
        assert not g.passed and "survive verbatim" in g.detail

    def test_a_drop_that_reports_nothing_is_rejected(self):
        out = "Bone loses collagen. Buried bone stays plastic. Done."
        g = validate.gate_spans(self.PARENT, out, {}, operator="drop_relation")
        assert not g.passed and "no spans_removed" in g.detail

    def test_a_phantom_removal_is_rejected(self):
        out = "Bone loses collagen. Buried bone stays plastic. Done."
        g = validate.gate_spans(
            self.PARENT, out, {"spans_removed": ["Whereas,"]}, operator="drop_relation"
        )
        assert not g.passed and "never in the parent" in g.detail

    def test_a_clean_drop_passes(self):
        out = "Bone loses collagen. Buried bone stays plastic. Done."
        g = validate.gate_spans(
            self.PARENT, out, {"spans_removed": ["By contrast,"]}, operator="drop_relation"
        )
        assert g.passed

    def test_whitespace_and_case_do_not_hide_a_survivor(self):
        out = "Bone loses collagen. BY   CONTRAST, buried bone stays plastic. Done."
        g = validate.gate_spans(
            self.PARENT, out, {"spans_removed": ["by contrast,"]}, operator="drop_relation"
        )
        assert not g.passed

    def test_an_add_must_contain_its_added_span(self):
        g = validate.gate_spans(
            self.PARENT, self.PARENT + " New.", {"spans_added": ["Absent."]},
            operator="add_resolution",
        )
        assert not g.passed and "absent from the output" in g.detail

    def test_a_non_editing_operator_passes_vacuously(self):
        g = validate.gate_spans(self.PARENT, "x", {}, operator="shuffle_order")
        assert g.passed


# ---------------------------------------------------------------------------
# V5 verdict logic.
# ---------------------------------------------------------------------------


class TestRealizationVerdict:
    def test_expected_reading_by_coupling(self):
        assert validate.expected_reading(None) == NONE
        assert validate.expected_reading({"level1_coupling": "none"}) == NONE
        assert validate.expected_reading({"level1_coupling": "entailment"}) == ("support", None)
        assert validate.expected_reading(
            {"level1_coupling": "exclusive", "is_resolved_concession": True}
        ) == RESOLVED

    def test_a_drop_is_realized_only_when_the_upper_text_reads_none(self):
        assert validate.realized(CONFLICT, NONE, _reading(CONFLICT, NONE))
        assert not validate.realized(CONFLICT, NONE, _reading(CONFLICT, CONFLICT))

    def test_a_resolution_is_judged_on_the_flag(self):
        assert validate.realized(CONFLICT, RESOLVED, _reading(CONFLICT, RESOLVED))
        # Already resolved in the lower text: the measured 4/8 dev resolution failure.
        assert not validate.realized(CONFLICT, RESOLVED, _reading(RESOLVED, RESOLVED))

    def test_flipped_order_is_undone(self):
        # The rater saw upper as text_1; the verdict must read it back in rung order.
        assert validate.realized(CONFLICT, NONE, _reading(NONE, CONFLICT), flipped=True)
        assert not validate.realized(CONFLICT, NONE, _reading(NONE, CONFLICT))

    def test_majority_decides(self):
        good, bad = (_reading(CONFLICT, NONE), False), (_reading(CONFLICT, CONFLICT), False)
        t = lambda rs: [{"edge": "a-b", "lower": CONFLICT, "upper": NONE, "readings": rs}]  # noqa: E731
        assert validate.gate_realization(t([good, good, bad]), []).passed
        assert not validate.gate_realization(t([good, bad, bad]), []).passed
        # A tie is not a majority.
        assert not validate.gate_realization(t([good, bad]), []).passed
        # No readings at all is not a pass.
        assert not validate.gate_realization(t([]), []).passed

    def test_spill_onto_an_untouched_edge_fails_the_rung(self):
        ok = [{"edge": "a-b", "lower": CONFLICT, "upper": NONE,
               "readings": [(_reading(CONFLICT, NONE), False)] * 3}]
        spilled = [{"edge": "c-d", "lower": ("support", None), "upper": ("support", None),
                    "readings": [(_reading(("support", None), NONE), False)] * 3}]
        g = validate.gate_realization(ok, spilled)
        assert not g.passed and "untouched" in g.detail and "c-d" in g.detail

    def test_the_failure_detail_says_what_the_raters_saw(self):
        t = [{"edge": "a-b", "lower": CONFLICT, "upper": NONE,
              "readings": [(_reading(CONFLICT, CONFLICT), False)] * 3}]
        g = validate.gate_realization(t, [])
        assert "a-b" in g.detail and "conflict" in g.detail


class TestV5Parsers:
    def test_parse_realization_normalizes(self):
        txt = json.dumps({
            "text_1": {"relation": "Conflict", "resolved": False, "evidence": "e"},
            "text_2": {"relation": "none", "resolved": True, "evidence": ""},
            "differs": True, "change": "dropped",
        })
        r, err = parse.parse_realization(txt)
        assert err is None
        assert r["text_1"]["relation"] == "conflict" and r["text_1"]["resolved"] is False
        # `resolved` is meaningless for a non-conflict and is dropped.
        assert r["text_2"]["resolved"] is None

    def test_parse_realization_rejects_an_unknown_relation(self):
        txt = json.dumps({"text_1": {"relation": "maybe"}, "text_2": {"relation": "none"}})
        assert parse.parse_realization(txt)[0] is None

    def test_parse_compatibility(self):
        assert parse.parse_compatibility('{"compatible": false, "why": "x"}')[0] == {
            "compatible": False, "why": "x"}
        assert parse.parse_compatibility('{"compatible": "no"}')[0] is None

    def test_v5_prompts_fill(self):
        out = prompts.fill("V5", text_a="A", text_b="B", claim_x="X", claim_y="Y")
        assert 'text_1="A"' in out and 'y="Y"' in out
        assert "jointly_assertable(" in prompts.fill("V5c", claim_x="X", claim_y="Y")


# ---------------------------------------------------------------------------
# Target screening.
# ---------------------------------------------------------------------------


def _conflict_relations():
    def e(i, s, t, sense, coupling, validity="valid", band="strong", resolved=False):
        return {"id": f"r{i:03d}", "source_id": s, "target_id": t, "level2_sense": sense,
                "level1_coupling": coupling, "validity": validity,
                "intended_strength_band": band, "is_resolved_concession": resolved,
                "is_concession": sense == "Concession"}
    return [
        e(0, "a0", "a1", "Contrast", "contradiction", validity="invalid"),
        e(1, "a2", "a3", "Contrast", "contradiction"),
        e(2, "a4", "a5", "Alternative", "exclusive"),
        e(3, "a6", "a7", "Concession", "contradiction"),
        e(4, "a8", "a9", "Cause-Effect", "entailment"),
    ]


class TestTargetExclusion:
    def test_an_excluded_drop_target_is_skipped(self):
        # Drops live on CHAIN's `coherent` rung now; CONFLICT is resolution-only.
        rels = _conflict_relations()
        first_drop = plan_targets("CHAIN", rels)[4][0]
        excl = plan_targets("CHAIN", rels, exclude={"drop_relation": {first_drop}})
        assert excl[4][0] and excl[4][0] != first_drop

    def test_exhausting_targets_leaves_an_empty_slot(self):
        rels = _conflict_relations()
        every = {str(e["id"]) for e in rels}
        assert plan_targets("CHAIN", rels, exclude={"drop_relation": every})[4] == [""]
        t = plan_targets("CONFLICT", rels, exclude={"add_resolution": every})
        assert t[4] == ["", ""]

    def test_eligible_targets_lead_with_the_planned_target(self):
        rels = _conflict_relations()
        t, el = plan_targets("CONFLICT", rels), eligible_targets(rels)
        assert el["drop_relation"][0] == plan_targets("CHAIN", rels)[4][0]
        assert el["add_resolution"][:2] == t[4]
        assert el["break_chain"] == ["r004"]


class _Rater:
    """A `_Caller`-like stub answering V5 and V5c from fixed tables."""

    def __init__(self, compat=True, base_resolved=False):
        self.compat, self.base_resolved = compat, base_resolved

    def ask(self, pid, parser, **values):
        if pid == "V5c":
            return {"compatible": self.compat, "why": ""}, None
        if pid == "V5r":
            res = "The panel held both apply." if self.base_resolved else ""
            return parser(json.dumps({"tension_sentence": "X whereas Y.",
                                      "resolving_sentence": res, "resolved": True}))
        r = RESOLVED if self.base_resolved else CONFLICT
        return parser(json.dumps(_reading(r, r)))


class TestScreenTargets:
    def test_content_incompatible_pairs_are_excluded_from_drops(self):
        rels = _conflict_relations()
        atoms = {f"a{i}": f"claim {i}" for i in range(10)}
        panel = realize.RaterPanel([("r1", _Rater(compat=False))] * 3)
        excl, log = realize.screen_targets("text", rels, atoms, panel)
        assert excl["drop_relation"] == {"r000", "r001", "r002", "r003"}
        assert "r004" not in excl["drop_relation"]  # entailments are never screened

    def test_already_resolved_conflicts_are_excluded_from_resolution(self):
        rels = _conflict_relations()
        atoms = {f"a{i}": f"claim {i}" for i in range(10)}
        panel = realize.RaterPanel([("r1", _Rater(base_resolved=True))] * 3)
        excl, _ = realize.screen_targets("text", rels, atoms, panel)
        # Only VALID, unresolved conflicts are resolution candidates, so only they are read.
        assert excl["add_resolution"] == {"r001", "r002", "r003"}


# ---------------------------------------------------------------------------
# Gated rungs, end to end on the mock.
# ---------------------------------------------------------------------------


class _P5:
    """P5 stub: fails the first `bad` attempts on a target, then edits faithfully."""

    def __init__(self, text, bad=0, bad_targets=()):
        self.text, self.bad, self.bad_targets, self.n = text, bad, set(bad_targets), {}

    def ask(self, pid, parser, *, response, plan, operator):
        tgt = operator.split("(", 1)[1].split(")", 1)[0]
        self.n[tgt] = self.n.get(tgt, 0) + 1
        if tgt in self.bad_targets or self.n[tgt] <= self.bad:
            # The dev failure mode: a tiny edit, the conflict span still present.
            out, diff = response.replace("Done.", "Done!"), {"spans_removed": [f"[{tgt}]"]}
        else:
            out = response.replace(f"[{tgt}]", "")
            diff = {"spans_removed": [f"[{tgt}]"]}
        return parser("```\n" + out + "\n```\n```json\n"
                      + json.dumps({"operator": "x", **diff}) + "\n```")


def _text(rels):
    return " ".join(f"Claim [{e['id']}] here." for e in rels) + " Done."


def _oracle_panel():
    p = realize.RaterPanel([("r", None)] * 3, rng=random.Random(1))
    p.oracle = lambda lo, hi: _reading(lo, hi)
    return p


class TestRealizeRung:
    def _run(self, p5, targets, alternates):
        rels = _conflict_relations()
        atoms = {f"a{i}": f"claim {i}" for i in range(10)}
        return realize.realize_rung(
            p5, _oracle_panel(), calls=("drop_relation",), targets=targets,
            alternates=alternates, operator_for=lambda c, t: f"{c}({t})",
            parent_text=_text(rels), parent_relations=rels, parent_non_relations=[],
            plan_json="{}", atom_text=atoms, generator="g",
        )

    def test_a_clean_drop_is_admitted_and_labels_follow(self):
        out, err, _ = self._run(_P5(_text(_conflict_relations())), ["r001"], {})
        assert err is None and "[r001]" not in out.text
        assert all(e["id"] != "r001" for e in out.relations)

    def test_a_bad_drop_is_retried_with_the_reason(self):
        p5 = _P5(_text(_conflict_relations()), bad=2)
        out, err, gates = self._run(p5, ["r001"], {})
        assert err is None and p5.n["r001"] == 3
        assert any(g.gate == "P5.spans" and not g.passed for g in gates)

    def test_an_unrealizable_target_is_retargeted_once(self):
        p5 = _P5(_text(_conflict_relations()), bad_targets={"r001"})
        out, err, _ = self._run(p5, ["r001"], {"drop_relation": ["r001", "r002", "r003"]})
        assert err is None and out.targets == ["r002"]
        # Retargeted labels: r002 dropped, r001 kept.
        ids = {e["id"] for e in out.relations}
        assert "r001" in ids and "r002" not in ids

    def test_two_unrealizable_targets_reject_the_rung(self):
        p5 = _P5(_text(_conflict_relations()), bad_targets={"r001", "r002"})
        out, err, _ = self._run(p5, ["r001"], {"drop_relation": ["r001", "r002", "r003"]})
        assert out is None and "drop_relation(r001)" in err


class TestCallByCallLabelsComposeWithSinglePass:
    """Gated rungs build labels one call at a time; that must equal the old single pass."""

    @pytest.mark.parametrize("family", ("CONFLICT", "CHAIN", "ORDER", "CONTROL"))
    def test_compose(self, family):
        from fact_reasoner.locobench.pipeline import _edge_set_signature

        rels = _conflict_relations()
        tg = plan_targets(family, rels)
        for rung in ladder_for(family).rungs:
            if rung.is_base:
                continue
            once, _, _ = apply_calls(rels, rung.calls, targets=tg[rung.index])
            step = rels
            for c, t in zip(rung.calls, tg[rung.index]):
                step, _, _ = apply_calls(step, (c,), targets=[t])
            assert _edge_set_signature(once) == _edge_set_signature(step)


@pytest.mark.skipif(
    not os.path.exists("data/locobench-opus-5-5/items.jsonl"), reason="release not present"
)
def test_call_by_call_labels_compose_on_the_release():
    from fact_reasoner.locobench.pipeline import _edge_set_signature

    items = [json.loads(line) for line in open("data/locobench-opus-5-5/items.jsonl")]
    fams = json.load(open("data/locobench-opus-5-5/families.json"))["families"]
    n = 0
    for f in fams:
        b = next(r["index"] for r in f["rungs"] if r["is_base"])
        base = next(i for i in items if i["expected"]["family_id"] == f["family_id"]
                    and i["expected"]["rung_index"] == b)
        tg = plan_targets(f["family"], base["relations"])
        for rung in ladder_for(f["family"]).rungs:
            if rung.is_base:
                continue
            once, _, _ = apply_calls(base["relations"], rung.calls, targets=tg[rung.index],
                                     non_relations=base["non_relations"])
            step, nons = base["relations"], base["non_relations"]
            for c, t in zip(rung.calls, tg[rung.index]):
                step, nons, _ = apply_calls(step, (c,), targets=[t], non_relations=nons)
            assert _edge_set_signature(once) == _edge_set_signature(step)
            n += 1
    assert n == 400


# ---------------------------------------------------------------------------
# The repair CLI, end to end on the mock.
# ---------------------------------------------------------------------------


def test_repair_cli_round_trip_on_the_mock(tmp_path):
    """Generate a mock dataset, repair one family, copy the rest through unchanged."""
    import subprocess
    import sys

    src, out = tmp_path / "src", tmp_path / "out"
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({
        "dataset_name": "lb-test", "out_dir": str(src), "n_families": 4,
        "generators": [{"name": "gen", "model_id": "mock/gen", "backend": "openai",
                        "family": "g"}],
        "committee": [
            {"name": f"c{i}", "model_id": f"mock/c{i}", "backend": "openai",
             "family": f"v{i}"} for i in range(3)
        ],
    }))
    env = {**os.environ, "PYTHONPATH": "src"}
    gen = subprocess.run(
        [sys.executable, "-m", "fact_reasoner.locobench.cli", "--config", str(cfg),
         "--out", str(src), "--dry-run", "--n-families", "4", "--limit", "4"],
        capture_output=True, text=True, env=env,
    )
    assert gen.returncode == 0, gen.stderr[-2000:]
    src_items = [json.loads(x) for x in (src / "items.jsonl").read_text().splitlines()]
    assert len(src_items) == 20

    rep = subprocess.run(
        [sys.executable, "-m", "fact_reasoner.locobench.rung_repair", "--src", str(src),
         "--out", str(out), "--config", str(cfg), "--families", "f001", "--dry-run"],
        capture_output=True, text=True, env=env,
    )
    assert rep.returncode == 0, rep.stdout[-2000:] + rep.stderr[-2000:]
    out_items = {json.loads(x)["id"]: json.loads(x)
                 for x in (out / "items.jsonl").read_text().splitlines()}
    src_by_id = {i["id"]: i for i in src_items}
    assert set(out_items) == set(src_by_id)  # ids stable
    log = json.loads((out / "repair_log.json").read_text())
    assert [x["action"] for x in log].count("repaired") == 1
    for iid, it in out_items.items():
        fid = it["expected"]["family_id"]
        if fid != "f001":
            assert it == src_by_id[iid]  # copied through byte-identical
        elif it["expected"]["rung_index"] == next(
            r["index"] for r in json.loads((out / "families.json").read_text())["families"][0]["rungs"]
            if r["is_base"]
        ):
            # The repaired family's BASE rung keeps its text and labels.
            assert it["response"] == src_by_id[iid]["response"]
            assert it["relations"] == src_by_id[iid]["relations"]
    conf = json.loads((out / "config.json").read_text())
    assert conf["repaired_from"] == str(src) and conf["repair"]["families"] == ["f001"]


class TestPilotFixes:
    def test_a_failed_realization_quotes_the_surviving_span(self):
        # Pilot f005: the link sentence was removed but a summary sentence restated it.
        survivor = "The inference therefore runs from colour to grain size."
        r = _reading(("support", None), ("support", None))
        r["text_2"]["evidence"] = survivor
        t = [{"edge": "a11-a12", "lower": ("support", None), "upper": NONE,
              "readings": [(r, False)] * 3}]
        g = validate.gate_realization(t, [])
        assert not g.passed and survivor in g.detail and "Remove or rewrite" in g.detail

    def test_the_quote_is_read_from_the_edited_side_when_flipped(self):
        r = _reading(("support", None), ("support", None))
        r["text_1"]["evidence"], r["text_2"]["evidence"] = "EDITED", "PARENT"
        t = [{"edge": "x", "lower": ("support", None), "upper": NONE, "readings": [(r, True)]}]
        g = validate.gate_realization(t, [])
        assert "EDITED" in g.detail and "PARENT" not in g.detail

    def test_unfillable_calls_are_found_before_any_p5_call(self):
        calls = {2: ("add_resolution",), 3: ("add_resolution", "drop_relation"),
                 0: ("spurious_relation",)}
        targets = {2: [""], 3: ["", "r002"], 0: [""]}
        assert realize.unfillable_calls(calls, targets) == [
            "rung2:add_resolution", "rung3:add_resolution"]

    def test_screen_records_evidence_per_edge(self):
        rels = _conflict_relations()
        atoms = {f"a{i}": f"claim {i}" for i in range(10)}
        _, log = realize.screen_targets(
            "text", rels, atoms, realize.RaterPanel([("r1", _Rater())] * 3))
        assert set(log["evidence"]) == {"r000", "r001", "r002", "r003"}
        assert log["evidence"]["r001"]["resolved_votes"] == [False] * 3


class _V5cByPair:
    """A rater whose V5c verdict is incompatible for the named claim pairs."""

    def __init__(self, head_on_pairs):
        self.head_on = {frozenset(p) for p in head_on_pairs}

    def ask(self, pid, parser, **v):
        if pid == "V5c":
            return {"compatible": frozenset((v["claim_x"], v["claim_y"])) not in self.head_on,
                    "why": ""}, None
        raise AssertionError(f"unexpected {pid}")


class TestTextDrawnConflicts:
    ATOMS = {f"a{i}": f"claim {i}" for i in range(10)}

    def _panel(self, *pairs):
        return realize.RaterPanel([("r", _V5cByPair(pairs))] * 3)

    def test_a_plan_with_droppable_conflicts_passes(self):
        reason, log = realize.plan_droppability(
            "CONFLICT", _conflict_relations(), self.ATOMS, self._panel())
        assert reason is None and log["head_on"] == []

    def test_a_plan_of_head_on_conflicts_is_sent_back_with_the_pairs_named(self):
        # Pilot 2: "no crater" vs "a crater" -- every conflict head-on.
        pairs = [("claim 0", "claim 1"), ("claim 2", "claim 3"), ("claim 4", "claim 5")]
        # CHAIN's coherent rung drops one conflict; with every conflict head-on, none can go.
        pairs = pairs + [("claim 6", "claim 7")]
        reason, log = realize.plan_droppability(
            "CHAIN", _conflict_relations(), self.ATOMS, self._panel(*pairs))
        assert reason is not None and "HEAD-ON" in reason and '"claim 2" vs "claim 3"' in reason
        assert set(log["head_on"]) == {"r000", "r001", "r002", "r003"}

    def test_conflict_allows_one_head_on_pair_among_the_reserved_conflicts(self):
        # The two reserved conflicts (r003 Concession first, then r001) are screened; P4
        # must leave them open, and it adjudicates flat contradictions. One head-on pair
        # is allowed, two are not.
        rels = _conflict_relations()
        reason, log = realize.plan_droppability("CONFLICT", rels, self.ATOMS, self._panel())
        assert reason is None and log["head_on"] == []
        assert set(log["checked"]) == {"r001", "r003"}
        # A single head-on reserved pair is now refused (MAX_HEAD_ON_RESERVED = 0).
        reason, log = realize.plan_droppability(
            "CONFLICT", rels, self.ATOMS, self._panel(("claim 2", "claim 3")))
        assert reason is not None and log["head_on"] == ["r001"]
        reason, log = realize.plan_droppability(
            "CONFLICT", rels, self.ATOMS,
            self._panel(("claim 2", "claim 3"), ("claim 6", "claim 7")))
        assert reason is not None and "HEAD-ON" in reason and "TEXT-DRAWN" in reason

    def test_a_head_on_reserved_pair_sends_the_plan_back(self):
        # P4 adjudicates flat contradictions it is told to leave open (f008), so none of
        # the conflicts reserved for resolution may be head-on.
        reason, _ = realize.plan_droppability(
            "CONFLICT", _conflict_relations(), self.ATOMS, self._panel(("claim 2", "claim 3")))
        assert reason is not None and "None of the" in reason

    def test_ladders_that_never_drop_skip_the_screen(self):
        for fam in ("ORDER", "CONTROL"):
            reason, log = realize.plan_droppability(
                fam, _conflict_relations(), self.ATOMS,
                self._panel(("claim 0", "claim 1"), ("claim 2", "claim 3")))
            assert reason is None and log["checked"] == []

    def test_head_on_conflicts_are_preferred_as_resolution_targets(self):
        rels = _conflict_relations()
        plain = plan_targets("CONFLICT", rels)[3][0]
        pref = plan_targets("CONFLICT", rels, prefer_resolve={"r002"})[3][0]
        assert plain == "r003" and pref == "r002"

    def test_text_drawn_conflicts_are_asked_where_a_rung_needs_them(self):
        # CHAIN drops one; CONFLICT must leave its reserved conflicts open in prose.
        assert "TEXT-DRAWN" in prompts.LADDER_REQUIREMENTS["CHAIN"]
        assert "TEXT-DRAWN" in prompts.LADDER_REQUIREMENTS["CONFLICT"]
        assert "Neither may be a HEAD-ON pair" in prompts.LADDER_REQUIREMENTS["CONFLICT"]
        assert "TEXT-DRAWN" not in prompts.LADDER_REQUIREMENTS["ORDER"]


@pytest.mark.parametrize("family", ("CONFLICT", "CHAIN", "ORDER", "CONTROL"))
def test_a_dry_run_with_a_mock_panel_admits_every_ladder(family):
    """Plan gate, V5 screen and gated rungs all run when a committee is present."""
    from fact_reasoner.locobench.pipeline import generate_family, make_mock_llm
    from fact_reasoner.locobench.config import GenConfig

    cfg = GenConfig(dry_run=True, n_families=1, out_dir="/tmp/unused")
    shared: dict = {}
    panel = [(f"a{i}", make_mock_llm(cfg, plan_holder=shared)) for i in range(3)]
    r = generate_family("f001", "Law", family, cfg, auditor_llms=panel, plan_holder=shared)
    assert r.admitted, r.verdict.reason()
    gates = {g.gate for g in r.verdict.results}
    assert "V5.screen" in gates


class TestReservedResolution:
    ATOMS = {f"a{i}": f"claim {i}" for i in range(10)}

    def test_a_plan_with_no_valid_open_conflict_is_sent_back(self):
        # Pilot 3 f001: the only valid open conflict was resolved or head-on; the others
        # were planted-invalid, which a resolution may not target.
        rels = _conflict_relations()
        for e in rels:
            if e["id"] in ("r001", "r002"):
                e["validity"] = "invalid"
            if e["id"] == "r003":
                e["is_resolved_concession"] = True
        panel = realize.RaterPanel([("r", _V5cByPair([]))] * 3)
        reason, _ = realize.plan_droppability("CONFLICT", rels, self.ATOMS, panel)
        assert reason is not None and "VALID and UNRESOLVED" in reason

    def test_keep_open_targets_are_the_ladders_resolution_targets(self):
        # Both conflicts the ladder will settle must be left open by P4.
        t = {str(k): v for k, v in plan_targets("CONFLICT", _conflict_relations()).items()}
        assert realize.keep_open_targets("CONFLICT", t) == {"r003", "r001"}
        tc = {str(k): v for k, v in plan_targets("CHAIN", _conflict_relations()).items()}
        assert realize.keep_open_targets("CHAIN", tc) == set()

    def test_p4_is_told_to_leave_a_keep_open_relation_standing(self):
        assert '"keep_open": true' in prompts.PROMPTS["P4"]


def test_p4_receives_keep_open_on_a_plan_copy_and_the_stored_plan_is_clean():
    from fact_reasoner.locobench.config import GenConfig
    from fact_reasoner.locobench.pipeline import generate_family, make_mock_llm

    cfg = GenConfig(dry_run=True, n_families=1, out_dir="/tmp/unused")
    shared: dict = {}
    mock = make_mock_llm(cfg, plan_holder=shared)
    seen = []

    def spy(rendered, **kw):
        if "realizes the plan exactly" in rendered:
            seen.append(rendered)
        return mock(rendered, **kw)

    panel = [(f"a{i}", make_mock_llm(cfg, plan_holder=shared)) for i in range(3)]
    r = generate_family("f001", "Law", "CONFLICT", cfg, llm=spy, auditor_llms=panel,
                        plan_holder=shared)
    assert r.admitted, r.verdict.reason()
    assert seen and '"keep_open": true' in seen[-1]
    assert all("keep_open" not in rel for rel in r.artifacts["plan"]["relations"])


class TestUndeclaredEdits:
    PARENT = ("Bone loses collagen. By contrast, buried bone stays plastic. "
              "The panel later ruled the matter settled. Imaging was then applied.")

    def test_a_rewrite_of_an_unrelated_sentence_is_rejected(self):
        # Pilot 4: a drop also rewrote another rung's resolution sentence.
        out = ("Bone loses collagen. Buried bone stays plastic. "
               "A review later found the point open. Imaging was then applied.")
        g = validate.gate_spans(self.PARENT, out, {"spans_removed": ["By contrast,"]},
                                operator="drop_relation")
        assert not g.passed and "did not declare" in g.detail
        assert "panel later ruled" in g.detail.lower()

    def test_the_rewritten_remainder_of_a_declared_sentence_passes(self):
        out = ("Bone loses collagen. Buried bone stays plastic. "
               "The panel later ruled the matter settled. Imaging was then applied.")
        g = validate.gate_spans(self.PARENT, out, {"spans_removed": ["By contrast,"]},
                                operator="drop_relation")
        assert g.passed, g.detail

    def test_an_added_sentence_must_be_declared(self):
        out = self.PARENT + " Moreover, the tribunal held both readings valid."
        g = validate.gate_spans(self.PARENT, out, {"spans_added": []}, operator="ordering_only")
        assert not g.passed
        g = validate.gate_spans(self.PARENT, out,
                                {"spans_added": ["the tribunal held both readings valid"]},
                                operator="ordering_only")
        assert g.passed, g.detail

    def test_a_shuffle_is_exempt(self):
        out = "Imaging was then applied. " + self.PARENT.rsplit(" Imaging", 1)[0]
        g = validate.gate_spans(self.PARENT, out, {}, operator="shuffle_order")
        assert g.passed

    def test_moved_sentences_are_not_edits(self):
        from fact_reasoner.locobench.validate import _undeclared_edits
        s = self.PARENT.split(". ")
        moved = ". ".join([s[1], s[0]] + s[2:])
        assert _undeclared_edits(self.PARENT, moved, [], []) == []


def test_every_untouched_edge_is_read():
    before = _conflict_relations()
    sample = realize._untouched_sample(before, before, set(), random.Random(0))
    assert len(sample) == len(before)


class TestPhase2Fixes:
    def test_rewrapped_lines_are_not_edits(self):
        from fact_reasoner.locobench.validate import _undeclared_edits
        parent = "First claim holds here. Second claim" + chr(10) + "holds too. Third."
        wrapped = "First claim holds" + chr(10) + "here. Second claim holds too. Third."
        assert _undeclared_edits(parent, wrapped, [], []) == []

    def test_an_unrealized_addition_asks_for_a_connective_not_a_removal(self):
        r = _reading(NONE, NONE)
        r["text_2"]["evidence"] = "Resolution began with governance."
        t = [{"edge": "a1-a2", "lower": NONE, "upper": CONFLICT, "readings": [(r, False)] * 3}]
        g = validate.gate_realization(t, [])
        assert not g.passed and "add one connective" in g.detail
        assert "Remove or rewrite" not in g.detail

    def test_an_unrealized_resolution_asks_for_both_claims(self):
        r = _reading(CONFLICT, CONFLICT)
        t = [{"edge": "x", "lower": CONFLICT, "upper": RESOLVED, "readings": [(r, False)] * 3}]
        assert "name both claims" in validate.gate_realization(t, []).detail


class TestResolutionOnlyConflictLadder:
    """The CONFLICT ladder settles conflicts and never drops them (2026-11-08)."""

    def test_no_conflict_rung_drops(self):
        for r in ladder_for("CONFLICT").rungs:
            assert "drop_relation" not in r.calls

    def test_each_rung_adds_exactly_one_edit_to_its_parent(self):
        lad = ladder_for("CONFLICT")
        assert lad.incremental
        assert [r.calls.count("add_resolution") for r in lad.rungs] == [0, 0, 0, 1, 2]
        assert [r.calls.count("spurious_relation") for r in lad.rungs] == [2, 1, 0, 0, 0]
        assert all(len(lad.increment(r)) == 1 for r in lad.rungs if not r.is_base)

    def test_an_incremental_rung_must_extend_its_parent_by_one_call(self):
        from fact_reasoner.locobench.perturb import Ladder, Rung

        with pytest.raises(ValueError, match="extend its parent"):
            Ladder("X", (Rung(0, "a", ("add_resolution", "add_resolution"), parent=1),
                         Rung(1, "base"), Rung(2, "b", ("add_resolution",), parent=1),
                         Rung(3, "c", ("add_resolution",), parent=1),
                         Rung(4, "d", ("add_resolution",), parent=1)), incremental=True)

    def test_c3_declares_the_consistency_endpoint_a_decrease_and_the_evaluator_reads_it(self):
        from fact_reasoner.locobench.perturb import ordering_constraints
        from fact_reasoner.locoeval.runner import evaluate_constraints

        fam = {"ordering_constraints": ordering_constraints("CONFLICT")}
        # Scores that follow variant B's measured gold pattern: everything rises through
        # the two spurious steps; consistency then falls at each resolution, and
        # log_partition falls at the first, below its rung-0 value at no point.
        mm = [0.50, 0.52, 0.55, 0.57, 0.60]
        cs = [0.50, 0.54, 0.58, 0.52, 0.46]
        lp = [0.30, 0.34, 0.40, 0.36, 0.38]
        sbr = {i: {"mean_marginal": mm[i], "consistency": cs[i], "log_partition": lp[i]}
               for i in range(5)}
        checks = evaluate_constraints(fam, sbr)
        assert checks and all(c["passed"] for c in checks), [c for c in checks if not c["passed"]]
        c3 = [c for c in checks if c["constraint_class"] == "C3"]
        assert {(c["readout"], c["expected"]) for c in c3} == {
            ("mean_marginal", "increase"), ("log_partition", "increase"),
            ("consistency", "decrease")}

    def test_an_old_manifest_without_decrease_readouts_evaluates_unchanged(self):
        from fact_reasoner.locoeval.runner import evaluate_constraints

        old = {"ordering_constraints": [{"id": "c3", "class": "C3", "strict": True,
                                         "readouts": ["mean_marginal", "consistency"],
                                         "required": [[0, 4]]}]}
        sbr = {0: {"mean_marginal": 0.1, "consistency": 0.1},
               4: {"mean_marginal": 0.2, "consistency": 0.2}}
        checks = evaluate_constraints(old, sbr)
        assert [c["expected"] for c in checks] == ["increase", "increase"]
        assert all(c["passed"] for c in checks)


class TestFreeRetries:
    """`_Caller.ask` grants extra attempts only for failures the caller vouches for."""

    def _caller(self, reasons):
        from fact_reasoner.locobench.pipeline import _Caller

        calls = {"n": 0}

        def llm(prompt, attempt=0):
            calls["n"] += 1
            return "```\n" + " ".join(["word"] * 560) + "\n```"

        tag = {"free": False}
        seq = iter(reasons)

        def check(_value):
            kind = next(seq, None)
            tag["free"] = kind == "keep_open"
            return None if kind is None else kind

        return _Caller(llm, attempts=3), check, (lambda: tag["free"]), calls

    def _ask(self, caller, check, is_free, free):
        from fact_reasoner.locobench import parse

        return caller.ask("P4", parse.parse_response, check=check, question="q", plan="{}",
                          free_retries=free, is_free=is_free)

    def test_ordinary_failures_keep_the_base_budget(self):
        c, check, free, calls = self._caller(["leak"] * 9)
        self._ask(c, check, free, free=2)
        assert calls["n"] == 3

    def test_keep_open_failures_get_up_to_two_extra(self):
        c, check, free, calls = self._caller(["keep_open"] * 9)
        self._ask(c, check, free, free=2)
        assert calls["n"] == 5

    def test_a_late_success_after_free_retries_is_returned(self):
        c, check, free, calls = self._caller(["keep_open", "keep_open", "leak", "leak"])
        value, err = self._ask(c, check, free, free=2)
        assert err is None and calls["n"] == 5

    def test_no_free_retries_without_the_grant(self):
        c, check, free, calls = self._caller(["keep_open"] * 9)
        self._ask(c, check, free, free=0)
        assert calls["n"] == 3


def test_spurious_pairs_and_retargeting_render_the_same_pair():
    from fact_reasoner.locobench.perturb import spurious_pairs

    rels = _conflict_relations()
    nons = [{"source_id": "a0", "target_id": "a9"}, {"source_id": "a2", "target_id": "a8"},
            {"source_id": "a0", "target_id": "a1"}]  # last pair already related -> skipped
    assert spurious_pairs(rels, nons) == ["a0|a9", "a2|a8"]
    out, _, log = apply_calls(rels, ("spurious_relation",), targets=["a2|a8"], non_relations=nons)
    added = [e for e in out if e.get("error_kind") == "spurious"]
    assert len(added) == 1 and (added[0]["source_id"], added[0]["target_id"]) == ("a2", "a8")


class TestSpuriousLinkFixes:
    def test_spurious_pairs_rank_the_most_related_pair_first(self):
        from fact_reasoner.locobench.perturb import spurious_pairs

        rels = _conflict_relations()
        txt = {f"a{i}": f"unrelated claim number {i}" for i in range(10)}
        txt["a2"] = "The artifact committee found the compiler flags differed."
        txt["a8"] = "The compiler flags were identical across the two builds."
        nons = [{"source_id": "a0", "target_id": "a9"}, {"source_id": "a2", "target_id": "a8"}]
        assert spurious_pairs(rels, nons) == ["a0|a9", "a2|a8"]  # unranked: plan order
        assert spurious_pairs(rels, nons, txt)[0] == "a2|a8"

    def test_the_spurious_edit_may_add_a_bridging_sentence_but_no_new_claim(self):
        p5 = prompts.PROMPTS["P5"]
        assert "ONE short sentence that names the subject both atoms bear on" in p5
        assert "add no new factual claim" in p5


def test_an_incremental_rung_never_falls_back_to_a_target_its_parent_used():
    p5 = _P5(_text(_conflict_relations()), bad_targets={"r001"})
    rels = _conflict_relations()
    out, err, _ = realize.realize_rung(
        p5, _oracle_panel(), calls=("drop_relation",), targets=["r001"],
        alternates={"drop_relation": ["r001", "r002", "r003"]},
        operator_for=lambda c, t: f"{c}({t})", parent_text=_text(rels),
        parent_relations=rels, parent_non_relations=[], plan_json="{}",
        atom_text={f"a{i}": f"claim {i}" for i in range(10)}, generator="g",
        already_used=["r002"],
    )
    assert err is None and out.targets == ["r003"]  # skipped the parent's r002
