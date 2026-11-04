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

"""Tests for the validation study's importer and analysis.

The central test simulates a whole study: it builds Label Studio exports from the real
task files, answered by annotators whose behaviour is fixed in advance, runs the
importer and the analysis, and checks that every hypothesis recovers what was planted.
A mis-joined claim pair, an off-by-one in the Repeater index, or a unit counted twice
would each move a recovered rate away from its planted value.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE = os.path.join(_ROOT, "docs", "human_study_release")
DATA = os.path.join(_ROOT, "data", "locobench-opus-5-5")


def _load(name: str):
    path = os.path.join(_ROOT, "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA, "items.jsonl")), reason="the release corpus is not present"
)

imp = _load("import_human_study_release")
ana = _load("analyze_human_study_release")
build = _load("build_human_study_release")

ROW = {"contradiction": (False, True), "exclusive": (False, False), "co_necessity": (True, False),
       "entailment": (True, True), "equivalence": (True, True), "none": (True, True)}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A fresh build into a temp dir, so the test never depends on a stale checkout."""
    out = tmp_path_factory.mktemp("release")
    key = tmp_path_factory.mktemp("key")
    import sys
    argv = sys.argv
    sys.argv = ["build", "--out-dir", str(out), "--key-dir", str(key), "--data-dir", DATA]
    try:
        build.main()
    finally:
        sys.argv = argv
    return str(out), os.path.join(str(key), "answer_key.jsonl")


# -- a simulated annotator -------------------------------------------------------------


def _answer_core(task, key, rng, *, accuracy, fail_attention=False):
    """Answer one core task. With probability `accuracy` each question gets the gold
    answer; otherwise a fixed wrong one. Returns a Label Studio result list."""
    k = key[task["task_id"]]
    res = []

    def ch(name, to, val):
        res.append({"from_name": name, "to_name": to, "type": "choices", "value": {"choices": [val]}})

    pk = {p["pair_id"]: p for p in k["pairs"]}
    for i, row in enumerate(task["pairs"]):
        p = pk[row["pair_id"]]
        if p["kind"] == "attention_check":
            linked, bt, bf = (not fail_attention), True, True
        elif p["kind"] == "non":
            linked, (bt, bf) = False, (True, True)
        else:
            linked, (bt, bf) = True, ROW[p["level1_coupling"]]
        if p["kind"] != "attention_check" and rng.random() > accuracy:
            linked = not linked
        ch(f"linked_{i}", f"pair_{i}", "Linked" if linked else "Not linked")
        ch(f"both_true_{i}", f"pair_{i}", "Both could be true" if bt else "Not both true")
        ch(f"both_false_{i}", f"pair_{i}", "Both could be false" if bf else "Not both false")
    for i, _ in enumerate(task["claims"]):
        ok = rng.random() <= accuracy
        ch(f"asserted_{i}", f"claim_{i}", "Asserted as written" if ok else "Changed or contradicted")
    vk = {v["version_id"]: v for v in k["versions"]}
    for i, row in enumerate(task["versions"]):
        v = vk[row["version_id"]]
        if "attention_check" in v:
            coh, coll = ("Less coherent", "Nothing else changed") if fail_attention else ("About as coherent", "Nothing else changed")
        else:
            if k["ladder"] == "CONTROL":
                coh = "About as coherent"
            else:
                coh = "Less coherent" if v["rungs"][0] < v["parent"] else "More coherent"
            if rng.random() > accuracy:
                coh = "About as coherent" if coh != "About as coherent" else "Less coherent"
            coll = "Nothing else changed"
        ch(f"coherence_{i}", f"version_{i}", coh)
        ch(f"collateral_{i}", f"version_{i}", coll)
    return res


def _answer_instruction(task, key, rng, *, equal_under):
    k = key[task["task_id"]]
    p_equal = equal_under[k["instruction"]]
    choice = "About equal" if rng.random() < p_equal else ("Response A" if k["base_side"] == "A" else "Response B")
    return [{"from_name": "choice", "to_name": "response_a", "type": "choices", "value": {"choices": [choice]}},
            {"from_name": "confidence", "to_name": "response_a", "type": "choices", "value": {"choices": ["medium"]}}]


def _export(path, tasks, results):
    """A Label Studio JSON export: one task per entry, its data, one annotation."""
    with open(path, "w") as f:
        json.dump([{"id": n + 1, "data": t, "annotations": [{"id": n + 1, "completed_by": 1, "result": r,
                                                            "was_cancelled": False}]}
                   for n, (t, r) in enumerate(zip(tasks, results))], f)


def _simulate(release, key_path, tmp, *, accuracy=1.0, equal_under=None, fail_attention=()):
    key = imp.load_key(key_path)
    rng = random.Random(7)
    exports = []
    for n in range(1, 8):
        who = f"C{n}"
        tasks = json.load(open(os.path.join(release, "core", f"tasks_{who}.json")))
        res = [_answer_core(t, key, rng, accuracy=accuracy, fail_attention=who in fail_attention) for t in tasks]
        p = os.path.join(tmp, f"core_{who}.json")
        _export(p, tasks, res)
        exports.append(p)
    eq = equal_under or {"consistency": 0.0, "argument": 0.0}
    for n in range(1, 4):
        who = f"I{n}"
        tasks = json.load(open(os.path.join(release, "instruction", f"tasks_{who}.json")))
        res = [_answer_instruction(t, key, rng, equal_under=eq) for t in tasks]
        p = os.path.join(tmp, f"instr_{who}.json")
        _export(p, tasks, res)
        exports.append(p)
    return imp.convert(exports, key)


# -- the importer ----------------------------------------------------------------------


class TestImporter:
    def test_every_judgement_is_imported_and_keyed(self, built, tmp_path):
        release, key = built
        rows = _simulate(release, key, str(tmp_path))
        kinds = {k: sum(1 for r in rows if r["kind"] == k) for k in ("pair", "claim", "version", "instruction")}
        # 22 families x 3 raters x 15 pairs, plus one attention-check pair per core annotator.
        assert kinds["pair"] == 22 * 3 * 15 + 7
        assert kinds["claim"] == sum(1 for r in rows if r["kind"] == "claim")
        assert kinds["instruction"] == 3 * 21
        assert all(r["annotator"] in {f"C{i}" for i in range(1, 8)} | {f"I{i}" for i in range(1, 4)} for r in rows)

    def test_three_annotators_per_core_unit(self, built, tmp_path):
        release, key = built
        rows = [r for r in _simulate(release, key, str(tmp_path)) if r["kind"] == "pair" and not r["attention_check"]]
        per_unit = {}
        for r in rows:
            per_unit.setdefault(r["unit"], set()).add(r["annotator"])
        assert set(map(len, per_unit.values())) == {3}

    def test_a_unit_is_one_thing_not_one_task_index(self, built, tmp_path):
        # Each annotator's copy of a family is shuffled independently, so a task's own
        # "p01" names a different claim pair for each annotator. Keying units on that id
        # silently pooled three unrelated pairs into one "unit" and scored perfect
        # annotators at 64% on A1. A unit must hold exactly one distinct thing.
        release, key = built
        rows = _simulate(release, key, str(tmp_path))
        for kind, ident in (("pair", lambda r: (r["gold_kind"], r.get("gold_coupling"), r.get("validity"))),
                            ("claim", lambda r: r["atom_id"]),
                            ("version", lambda r: tuple(sorted(r.get("rungs") or ())))):
            seen = {}
            for r in rows:
                if r["kind"] == kind and not r.get("attention_check"):
                    seen.setdefault(r["unit"], set()).add(ident(r))
            assert all(len(v) == 1 for v in seen.values()), kind

    def test_an_unanswered_question_stops_the_import(self, built, tmp_path):
        release, key = built
        k = imp.load_key(key)
        task = json.load(open(os.path.join(release, "core", "tasks_C1.json")))[0]
        res = [r for r in _answer_core(task, k, random.Random(0), accuracy=1.0) if r["from_name"] != "linked_0"]
        p = os.path.join(str(tmp_path), "gap.json")
        _export(p, [task], [res])
        with pytest.raises(SystemExit, match="no answer"):
            imp.convert([p], k)

    def test_an_answer_the_config_never_offered_stops_the_import(self, built, tmp_path):
        release, key = built
        k = imp.load_key(key)
        task = json.load(open(os.path.join(release, "core", "tasks_C1.json")))[0]
        res = _answer_core(task, k, random.Random(0), accuracy=1.0)
        res[0]["value"]["choices"] = ["Probably linked"]
        p = os.path.join(str(tmp_path), "odd.json")
        _export(p, [task], [res])
        with pytest.raises(SystemExit, match="not an option"):
            imp.convert([p], k)

    def test_one_export_with_two_annotators_is_refused(self, built, tmp_path):
        release, key = built
        k = imp.load_key(key)
        a = json.load(open(os.path.join(release, "core", "tasks_C1.json")))[0]
        b = json.load(open(os.path.join(release, "core", "tasks_C2.json")))[0]
        rng = random.Random(0)
        p = os.path.join(str(tmp_path), "mixed.json")
        _export(p, [a, b], [_answer_core(a, k, rng, accuracy=1.0), _answer_core(b, k, rng, accuracy=1.0)])
        with pytest.raises(SystemExit, match="several annotators"):
            imp.convert([p], k)

    def test_a_task_from_another_build_is_refused(self, built, tmp_path):
        release, key = built
        k = imp.load_key(key)
        task = dict(json.load(open(os.path.join(release, "core", "tasks_C1.json")))[0])
        res = _answer_core(task, k, random.Random(0), accuracy=1.0)
        task["task_id"] = "C1-999"
        p = os.path.join(str(tmp_path), "stale.json")
        _export(p, [task], [res])
        with pytest.raises(SystemExit, match="not in the answer key"):
            imp.convert([p], k)

    def test_names_split_at_the_last_underscore(self):
        assert imp.split_name("both_false_12") == ("both_false", 12)
        assert imp.split_name("collateral_span_0") == ("collateral_span", 0)
        assert imp.split_name("choice") == ("choice", None)


# -- the analysis recovers what was planted -------------------------------------------


class TestAnalysisRecoversPlantedAnswers:
    def test_perfect_annotators_pass_everything(self, built, tmp_path):
        release, key = built
        res = ana.analyse(_simulate(release, key, str(tmp_path), accuracy=1.0,
                                    equal_under={"consistency": 0.6, "argument": 0.0}))
        for h in ("A1", "A2", "A4", "B"):
            assert res[h]["rate"] == 1.0 and res[h]["passed"], h
        assert res["H1"]["rate"] == 1.0 and res["H1"]["passed"]
        assert res["H3"]["rate"] == 1.0
        # Planted relations are asserted in the prose, so careful readers link them at
        # their surface coupling: every planted error reads as accepted. A3 reports that;
        # it is not a test the labels can fail.
        assert all(r["rate"] == 1.0 for r in res["A3"].values())
        assert res["H4"]["passed"] and res["H4"]["equal_rate"]["argument"] == 0.0

    def test_unit_counts_match_the_paper(self, built, tmp_path):
        release, key = built
        res = ana.analyse(_simulate(release, key, str(tmp_path)))
        assert res["A1"]["units"] == 132
        assert res["A2"]["units"] == 110
        assert sum(r["units"] for r in res["A3"].values()) == 88
        assert res["A4"]["units"] == 45
        assert res["B"]["units"] == 78
        assert res["H1"]["units"] == 68
        assert res["H3"]["units"] == 10
        assert sum(res["H4"]["n"].values()) == 63

    def test_no_instruction_effect_is_not_a_pass(self, built, tmp_path):
        release, key = built
        res = ana.analyse(_simulate(release, key, str(tmp_path), equal_under={"consistency": 0.3, "argument": 0.3}))
        assert not res["H4"]["passed"]

    def test_a_planted_error_rate_is_recovered_within_its_interval(self, built, tmp_path):
        release, key = built
        # 80% per-judgement accuracy on "linked": a majority of 3 is right with
        # p = 0.8^3 + 3 * 0.8^2 * 0.2 = 0.896, so A2's majority rate should sit near it.
        res = ana.analyse(_simulate(release, key, str(tmp_path), accuracy=0.8))
        lo, hi = res["A2"]["ci95"]
        assert lo <= 0.896 <= hi
        assert not res["A1"]["passed"] or res["A1"]["rate"] >= 0.90

    def test_an_annotator_who_fails_attention_is_excluded(self, built, tmp_path):
        release, key = built
        rows = _simulate(release, key, str(tmp_path), fail_attention=("C3",))
        att = ana.attention(rows)
        assert not all(att["C3"].values())
        assert all(all(v.values()) for a, v in att.items() if a != "C3")

    def test_coverage_names_the_families_an_exclusion_leaves_short(self, built, tmp_path):
        release, key = built
        rows = _simulate(release, key, str(tmp_path))
        assert ana.analyse(rows)["coverage"]["below_three"] == {}
        held = {r["family_id"] for r in rows if r["annotator"] == "C3" and r["kind"] == "pair"}
        short = ana.analyse([r for r in rows if r["annotator"] != "C3"])["coverage"]["below_three"]
        assert set(short) == held and all(len(a) == 2 for a in short.values())


# -- the statistics --------------------------------------------------------------------


class TestStatistics:
    def test_wilson_matches_a_known_value(self):
        lo, hi = ana.wilson(81, 100)
        assert abs(lo - 0.7222) < 1e-3 and abs(hi - 0.8749) < 1e-3

    def test_wilson_is_undefined_on_no_units(self):
        assert ana.wilson(0, 0) is None

    def test_two_proportion_direction(self):
        assert ana.two_proportion_z(15, 30, 3, 30) < 0.01
        assert ana.two_proportion_z(3, 30, 15, 30) > 0.99

    def test_truth_rows_identify_each_conflict_coupling(self):
        rows = {v: k for k, v in ana.CONFLICT_ROW.items()}
        assert rows["contradiction"] == (False, True)
        assert rows["exclusive"] == (False, False)
        assert rows["co_necessity"] == (True, False)
