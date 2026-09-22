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

"""Tests for the human-study export and analysis.

Two things are worth testing here and nothing else is: that the export cannot leak the
declared ordering to an annotator, and that the agreement statistic is the one it claims
to be. Everything else in these scripts is presentation.
"""

from __future__ import annotations

import collections
import importlib.util
import itertools
import json
import os
import random

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name: str):
    """Import one of the study scripts by path (they live in scripts/, not a package)."""
    path = os.path.join(_ROOT, "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DATA_DIR = os.path.join(_ROOT, "data", "locobench-claude-5-test")
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, "items.jsonl")),
    reason="the ladder corpus is not present",
)


class TestExportCannotLeakTheAnswer:
    """The corpus states the intended ordering in six places; none may reach a screen."""

    @pytest.fixture(scope="class")
    def built(self):
        mod = _load("export_human_study")
        screens, key = mod.build_screens(DATA_DIR, seed=0)
        return mod, screens, key

    def test_screens_carry_only_the_four_visible_fields(self, built):
        _mod, screens, _key = built
        allowed = {
            "screen_id",
            "framing",
            "response_a",
            "response_b",
            "ask_noticed_reordering",
        }
        for scr in screens:
            assert set(scr) == allowed, set(scr) - allowed

    def test_no_rung_label_or_gold_field_survives(self, built):
        """`expected.rung_name` is literally 'worse'/'coherent' and `name` repeats it."""
        mod, screens, _key = built
        mod.assert_no_leaks(screens)  # raises SystemExit on a leak
        for scr in screens:
            blob = json.dumps(scr)
            for bad in ("rung_name", "rung_index", "\"notes\"", "\"factual\"",
                        "\"expected\"", "\"relations\""):
                assert bad not in blob, (scr["screen_id"], bad)

    def test_opaque_screen_ids_do_not_encode_the_rung(self, built):
        """The corpus item id ends -r0..-r4, monotone in the ladder for CONFLICT/CHAIN."""
        _mod, screens, _key = built
        for scr in screens:
            assert scr["screen_id"].startswith("s")
            assert "locobench" not in scr["screen_id"]
            assert "-r" not in scr["screen_id"]

    def test_no_screen_compares_a_text_with_itself(self, built):
        """CONTROL's ladder is symmetric, so its (0,4) pair is byte-identical.

        Such a screen would measure only whether an annotator notices identical text,
        which is why CONTROL uses (0,2)/(1,2) instead. Regression guard: reverting to the
        generic PAIRS for CONTROL reintroduces a wasted screen.
        """
        _mod, screens, _key = built
        for scr in screens:
            assert scr["response_a"] != scr["response_b"], scr["screen_id"]

    def test_higher_side_is_not_always_the_same_side(self, built):
        """If position encoded the answer, an annotator could score without reading."""
        _mod, _screens, key = built
        sides = {k["higher_side"] for k in key}
        assert sides == {"A", "B"}

    def test_consecutive_screens_are_never_the_same_family(self, built):
        """Two screens from one family in a row would reveal the shared claim set."""
        _mod, _screens, key = built
        for a, b in zip(key, key[1:]):
            assert a["family_id"] != b["family_id"]

    def test_invariance_screens_declare_equal(self, built):
        _mod, _screens, key = built
        for k in key:
            if k["kind"] == "invariance":
                assert k["declared_answer"] == "equal"
            else:
                assert k["declared_answer"] in ("A", "B")

    def test_the_run_is_reproducible_under_a_seed(self, built):
        mod, screens, _key = built
        again, _ = mod.build_screens(DATA_DIR, seed=0)
        assert [s["screen_id"] for s in screens] == [s["screen_id"] for s in again]
        assert [s["response_a"] for s in screens] == [s["response_a"] for s in again]


class TestKrippendorffAlpha:
    """The agreement number will be quoted in a paper, so verify it against known values."""

    @pytest.fixture(scope="class")
    def alpha(self):
        return _load("analyze_human_study").krippendorff_alpha_nominal

    def test_perfect_agreement_is_one(self, alpha):
        units = [["A"] * 3, ["B"] * 3, ["A"] * 3, ["B"] * 3]
        assert alpha(units) == pytest.approx(1.0)

    def test_hand_computed_case(self, alpha):
        """Two units, two raters, one agreeing and one split.

        Coincidence matrix: unit 1 contributes 2 to (A,A); unit 2 contributes 1 each to
        (A,B) and (B,A). n=4, so observed disagreement is 2/4 = 0.5. Marginals are A=3,
        B=1, giving expected disagreement (3*1 + 1*3)/(4*3) = 0.5. Alpha = 1 - 0.5/0.5 = 0.
        """
        assert alpha([["A", "A"], ["A", "B"]]) == pytest.approx(0.0)

    def test_systematic_disagreement_is_negative(self, alpha):
        units = [["A", "B", "A"], ["B", "A", "B"]] * 2
        assert alpha(units) < 0

    def test_random_labels_centre_near_zero(self, alpha):
        rng = random.Random(0)
        vals = []
        for _ in range(200):
            units = [[rng.choice("ABC") for _ in range(3)] for _ in range(14)]
            a = alpha(units)
            if a is not None:
                vals.append(a)
        assert abs(sum(vals) / len(vals)) < 0.05

    def test_undefined_cases_return_none(self, alpha):
        assert alpha([["A", "A"], ["A", "A"]]) is None  # no variation: denominator 0
        assert alpha([["A", "A"]]) is None  # too few units
        assert alpha([]) is None

    def test_a_unit_with_one_rating_is_skipped_not_crashed(self, alpha):
        """Missing ratings are expected: an annotator may skip a screen."""
        a = alpha([["A", "A", "A"], ["B"], ["B", "B", "B"]])
        assert a == pytest.approx(1.0)

    def test_more_categories_do_not_break_the_marginals(self, alpha):
        units = [["A", "A"], ["B", "B"], ["equal", "equal"]]
        assert alpha(units) == pytest.approx(1.0)


class TestMajority:
    @pytest.fixture(scope="class")
    def majority(self):
        return _load("analyze_human_study").majority

    def test_strict_majority(self, majority):
        assert majority(["A", "A", "B"]) == "A"
        assert majority(["equal", "equal", "equal"]) == "equal"

    def test_three_way_split_has_no_majority(self, majority):
        """Forcing a verdict here would manufacture ground truth the study lacks."""
        assert majority(["A", "B", "equal"]) is None

    def test_even_tie_has_no_majority(self, majority):
        assert majority(["A", "B"]) is None

    def test_empty_is_none(self, majority):
        assert majority([]) is None


def test_alpha_and_permutations_agree_on_a_tiny_case():
    """Cross-check the coincidence construction against a brute-force pair count."""
    alpha = _load("analyze_human_study").krippendorff_alpha_nominal
    units = [["A", "A", "B"], ["B", "B", "B"], ["A", "B", "A"]]
    # Observed disagreement, counted directly over ordered pairs within units.
    num = den = 0.0
    for u in units:
        w = 1.0 / (len(u) - 1)
        for i, j in itertools.permutations(range(len(u)), 2):
            den += w
            if u[i] != u[j]:
                num += w
    observed = num / den
    # Reconstruct alpha from the same marginals the function uses.
    counts = {}
    for u in units:
        w = 1.0 / (len(u) - 1)
        for i, j in itertools.permutations(range(len(u)), 2):
            counts[u[i]] = counts.get(u[i], 0.0) + w
    n = sum(counts.values())
    expected = sum(
        counts[a] * counts[b] for a in counts for b in counts if a != b
    ) / (n * (n - 1))
    assert alpha(units) == pytest.approx(1.0 - observed / expected)


class TestLabelStudioPackaging:
    """The import payload is the file an annotator's browser actually loads."""

    @pytest.fixture(scope="class")
    def built(self, tmp_path_factory):
        exp = _load("export_human_study")
        ls = _load("export_label_studio")
        d = tmp_path_factory.mktemp("ls")
        screens, key = exp.build_screens(DATA_DIR, seed=0)
        with open(os.path.join(d, "screens.jsonl"), "w") as f:
            for scr in screens:
                f.write(json.dumps(scr) + "\n")
        with open(os.path.join(d, "answer_key.jsonl"), "w") as f:
            for row in key:
                f.write(json.dumps(row) + "\n")
        return ls, str(d), ls.build_tasks(str(d)), key

    def test_tasks_nest_under_data(self, built):
        """Label Studio resolves $vars against the `data` key and nothing else."""
        _ls, _d, tasks, _key = built
        for t in tasks:
            assert set(t) == {"data"}
            assert set(t["data"]) == {
                "screen_id", "framing", "response_a", "response_b"
            }

    def test_no_answer_key_field_reaches_a_task(self, built):
        """Label Studio round-trips any `data` field into the browser."""
        ls, _d, tasks, _key = built
        ls.assert_no_leaks(tasks)
        for t in tasks:
            blob = json.dumps(t)
            for bad in ("declared_answer", "higher_side", "rung", "family_id",
                        "ladder", "kind"):
                assert f'"{bad}"' not in blob

    def test_tasks_carry_no_predictions(self, built):
        """A prediction would pre-fill the very judgement the study measures."""
        _ls, _d, tasks, _key = built
        for t in tasks:
            assert "predictions" not in t and "annotations" not in t

    def test_the_reordering_flag_is_not_carried_through(self, built):
        """It would tell an attentive annotator which screens are invariance probes."""
        _ls, _d, tasks, _key = built
        for t in tasks:
            assert "ask_noticed_reordering" not in t["data"]

    def test_labeling_config_is_well_formed_xml(self, built):
        """A malformed config fails silently in the UI, so parse it here."""
        import xml.etree.ElementTree as ET

        ls, _d, _tasks, _key = built
        root = ET.fromstring(ls.build_config())
        assert root.tag == "View"

    def test_every_toname_points_at_a_declared_object_tag(self, built):
        import xml.etree.ElementTree as ET

        ls, _d, _tasks, _key = built
        root = ET.fromstring(ls.build_config())
        objects = {
            el.attrib["name"]
            for el in root.iter()
            if el.tag in ("Text", "HyperText", "Image", "Audio") and "name" in el.attrib
        }
        for el in root.iter():
            target = el.attrib.get("toName")
            if target:
                assert target in objects, (el.tag, target)

    def test_config_variables_are_all_supplied_by_the_tasks(self, built):
        import re

        ls, _d, tasks, _key = built
        wanted = set(re.findall(r'value="\$(\w+)"', ls.build_config()))
        assert wanted <= set(tasks[0]["data"]), wanted - set(tasks[0]["data"])

    def test_forced_choice_and_confidence_are_required(self, built):
        """A blank forced choice cannot be aggregated; a blank confidence loses signal."""
        import xml.etree.ElementTree as ET

        ls, _d, _tasks, _key = built
        root = ET.fromstring(ls.build_config())
        req = {
            el.attrib["name"]
            for el in root.iter("Choices")
            if el.attrib.get("required") == "true"
        }
        assert {"choice", "confidence"} <= req

    def test_the_four_controls_have_distinct_from_names(self, built):
        """The importer tells the questions apart by `from_name`."""
        import xml.etree.ElementTree as ET

        ls, _d, _tasks, _key = built
        root = ET.fromstring(ls.build_config())
        names = [
            el.attrib["name"]
            for el in root.iter()
            if el.tag in ("Choices", "TextArea")
        ]
        assert sorted(names) == ["choice", "confidence", "noticed_reordering", "why"]


class TestLabelStudioImport:
    """Converting an export back into the flat per-annotator files."""

    @pytest.fixture(scope="class")
    def imp(self):
        return _load("import_label_studio")

    def _export(self, choice_label="Response A is more coherent", **kw):
        ann = {
            "completed_by": kw.get("completed_by", 7),
            "was_cancelled": kw.get("was_cancelled", False),
            "lead_time": 300.0,
            "result": [
                {"from_name": "choice", "to_name": "response_a", "type": "choices",
                 "value": {"choices": [choice_label]}},
                {"from_name": "confidence", "to_name": "response_a", "type": "choices",
                 "value": {"choices": ["high"]}},
                {"from_name": "why", "to_name": "response_a", "type": "textarea",
                 "value": {"text": ["it contradicts itself"]}},
            ],
        }
        return [{"id": 1, "data": {"screen_id": "s001"}, "annotations": [ann]}]

    def test_exported_display_value_maps_to_the_short_form(self, imp, tmp_path):
        """Label Studio exports a Choice's `value`, NOT its `alias`.

        The tag docs are explicit that "value will be used in the exported result", so an
        importer keyed on the alias silently rejects every real export. Regression guard.
        """
        path = tmp_path / "e.json"
        path.write_text(json.dumps(self._export()))
        out = imp.convert(str(path), str(tmp_path))
        assert out["A1"][0]["choice"] == "A"

    def test_short_forms_still_import(self, imp, tmp_path):
        path = tmp_path / "e.json"
        path.write_text(json.dumps(self._export(choice_label="equal")))
        out = imp.convert(str(path), str(tmp_path))
        assert out["A1"][0]["choice"] == "equal"

    def test_an_unknown_choice_fails_loudly(self, imp, tmp_path):
        path = tmp_path / "e.json"
        path.write_text(json.dumps(self._export(choice_label="Somewhat better")))
        with pytest.raises(SystemExit, match="not one of"):
            imp.convert(str(path), str(tmp_path))

    def test_cancelled_annotations_are_dropped_not_counted(self, imp, tmp_path):
        """A skipped screen is a MISSING rating, which alpha already handles by weight."""
        path = tmp_path / "e.json"
        path.write_text(json.dumps(self._export(was_cancelled=True)))
        with pytest.raises(SystemExit, match="no completed annotations"):
            imp.convert(str(path), str(tmp_path))

    def test_completed_by_accepts_both_int_and_object(self, imp, tmp_path):
        """Label Studio serializes it either way depending on version."""
        path = tmp_path / "e.json"
        path.write_text(
            json.dumps(self._export(completed_by={"id": 7, "email": "a@b.c"}))
        )
        out = imp.convert(str(path), str(tmp_path))
        assert list(out) == ["A1"]
        # The email must not survive into a results file.
        assert "a@b.c" not in json.dumps(out)

    def test_json_min_export_is_rejected_with_a_useful_message(self, imp, tmp_path):
        """JSON-MIN drops `from_name`, making the four controls indistinguishable."""
        path = tmp_path / "e.json"
        path.write_text(json.dumps([{"choice": "A", "confidence": "high"}]))
        with pytest.raises(SystemExit, match="JSON-MIN"):
            imp.convert(str(path), str(tmp_path))

    def test_a_screen_id_outside_the_answer_key_is_refused(self, imp, tmp_path):
        """Guards against pairing an export with the wrong study directory."""
        (tmp_path / "answer_key.jsonl").write_text(
            json.dumps({"screen_id": "s999"}) + "\n"
        )
        path = tmp_path / "e.json"
        path.write_text(json.dumps(self._export()))
        with pytest.raises(SystemExit, match="not in"):
            imp.convert(str(path), str(tmp_path))


class TestOneExportPerAnnotator:
    """Per-project exports all say ``completed_by: 1`` while being different people.

    This is the shape the LoCoBench study actually arrived in: four annotators, each
    labelling in their own Label Studio project, so each export numbers its users from
    scratch. Trusting ``completed_by`` there merges all four into one annotator holding
    one rating per screen -- at which point Krippendorff's alpha is not low, it is
    *undefined*, and the study looks like it produced no agreement data. So the merge must
    be impossible to do by accident.
    """

    @pytest.fixture(scope="class")
    def imp(self):
        return _load("import_label_studio")

    def _one_project_export(self, choice: str, screens=("s001", "s002")):
        """An export from a project whose only user is id 1, as Label Studio emits it."""
        tasks = []
        for i, sid in enumerate(screens, 1):
            tasks.append({
                "id": i,
                "data": {"screen_id": sid},
                "annotations": [{
                    "completed_by": 1,
                    "was_cancelled": False,
                    "lead_time": 120.0,
                    "result": [{
                        "from_name": "choice", "to_name": "response_a",
                        "type": "choices", "value": {"choices": [choice]},
                    }],
                }],
            })
        return tasks

    def test_force_label_keeps_same_id_exports_distinct(self, imp, tmp_path):
        """``force_label`` overrides ``completed_by`` so each file is its own annotator."""
        out = {}
        for i, choice in enumerate(["A", "B", "equal", "A"], 1):
            path = tmp_path / f"ann_{i}.json"
            path.write_text(json.dumps(self._one_project_export(choice)))
            got = imp.convert(str(path), str(tmp_path), force_label=f"A{i}")
            assert list(got) == [f"A{i}"], (
                f"file {i} was attributed to {list(got)}, not A{i}"
            )
            out.update(got)
        assert sorted(out) == ["A1", "A2", "A3", "A4"]
        # Each annotator rated each screen exactly once: the 4x2 design is intact.
        for who, rows in out.items():
            assert sorted(r["screen_id"] for r in rows) == ["s001", "s002"]

    def test_trusting_completed_by_would_have_merged_them(self, imp, tmp_path):
        """The failure this guards against, asserted so the guard cannot be removed.

        Without ``force_label`` every file resolves to ``A1``, and a caller merging the
        results gets one annotator with two ratings per screen instead of four with one.
        """
        merged: dict[str, list[dict]] = {}
        shared: dict[str, str] = {}
        for i, choice in enumerate(["A", "B"], 1):
            path = tmp_path / f"ann_{i}.json"
            path.write_text(json.dumps(self._one_project_export(choice)))
            got = imp.convert(str(path), str(tmp_path), labels=shared)
            assert list(got) == ["A1"], "per-project exports all report completed_by: 1"
            for who, rows in got.items():
                merged.setdefault(who, []).extend(rows)
        assert list(merged) == ["A1"]
        dup = [s for s, n in collections.Counter(
            r["screen_id"] for r in merged["A1"]).items() if n > 1]
        assert dup == ["s001", "s002"], (
            "the merge should double-rate every screen under one label; this is what "
            "main()'s duplicate check refuses"
        )

    def test_shared_labels_keep_numbering_across_files(self, imp, tmp_path):
        """With a genuine multi-user project set, ids still map to stable labels."""
        shared: dict[str, str] = {}
        p1 = tmp_path / "a.json"
        p1.write_text(json.dumps([{
            "id": 1, "data": {"screen_id": "s001"},
            "annotations": [{
                "completed_by": 11, "was_cancelled": False,
                "result": [{"from_name": "choice", "to_name": "response_a",
                            "type": "choices", "value": {"choices": ["A"]}}],
            }],
        }]))
        p2 = tmp_path / "b.json"
        p2.write_text(json.dumps([{
            "id": 2, "data": {"screen_id": "s001"},
            "annotations": [{
                "completed_by": 22, "was_cancelled": False,
                "result": [{"from_name": "choice", "to_name": "response_a",
                            "type": "choices", "value": {"choices": ["B"]}}],
            }],
        }]))
        a = imp.convert(str(p1), str(tmp_path), labels=shared)
        b = imp.convert(str(p2), str(tmp_path), labels=shared)
        assert list(a) == ["A1"] and list(b) == ["A2"], (
            "a second file's new user must get A2, not restart at A1"
        )


class TestStatisticalHelpers:
    """The numbers quoted in the paper come from these three functions."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load("analyze_human_study")

    def test_binom_sf_against_hand_values(self, mod):
        assert mod.binom_sf(0, 5, 0.5) == pytest.approx(1.0)
        assert mod.binom_sf(5, 5, 0.5) == pytest.approx(1 / 32)
        assert mod.binom_sf(3, 3, 1 / 3) == pytest.approx(1 / 27)
        # The study's own figure, checked independently.
        assert mod.binom_sf(8, 10, 1 / 3) == pytest.approx(0.00340395, abs=1e-8)

    def test_fisher_matches_a_known_table(self, mod):
        """The study's ORDER-vs-CONTROL table, verified against SciPy.

        SciPy gives 0.006993006993 for [[0,8],[6,2]]. Pinned because an earlier
        hand-rolled version of this function mishandled the fourth cell and returned the
        one-sided tail (0.0035), which would have been published as a two-sided p.
        """
        assert mod.fisher_exact_two_sided(0, 8, 6, 2) == pytest.approx(
            0.006993006993, abs=1e-9
        )
        # A table with no association sits at p = 1.
        assert mod.fisher_exact_two_sided(2, 2, 2, 2) == pytest.approx(1.0)

    def test_verdict_maps_score_difference_onto_the_randomized_side(self, mod):
        """The export randomizes which side holds the higher rung; the map must follow it."""
        assert mod.verdict_from_scores(0.10, 0.20, "B") == "B"
        assert mod.verdict_from_scores(0.10, 0.20, "A") == "A"
        # A lower score on the higher rung means the measure prefers the OTHER side.
        assert mod.verdict_from_scores(0.20, 0.10, "B") == "A"
        assert mod.verdict_from_scores(0.20, 0.10, "A") == "B"

    def test_verdict_calls_a_sub_tolerance_difference_equal(self, mod):
        """Float noise must not be scored as a preference, and "equal" is a real answer."""
        assert mod.verdict_from_scores(0.5, 0.5 + 1e-9, "A") == "equal"
        assert mod.verdict_from_scores(0.5, 0.5, "A") == "equal"
        # Just past the tolerance is a preference. (0.5 + 1e-6 is not exactly 1e-6 away
        # in binary floating point, so the boundary itself is probed with an exact pair.)
        assert mod.verdict_from_scores(0.0, 1e-6, "A") == "equal"
        assert mod.verdict_from_scores(0.5, 0.5 + 2e-6, "A") == "A"

    def test_verdict_abstains_on_a_missing_score(self, mod):
        assert mod.verdict_from_scores(None, 0.5, "A") is None
        assert mod.verdict_from_scores(0.5, None, "A") is None

    def test_tie_tolerance_matches_the_ladder_scorer(self, mod):
        """A human-referenced column and a ladder column must use one tolerance."""
        assert mod.TIE_TOLERANCE == 1e-6


class TestInvarianceSplitIsNotPooled:
    """The ORDER/CONTROL split is the study's main finding; guard the arithmetic."""

    @pytest.fixture(scope="class")
    def mod(self):
        return _load("analyze_human_study")

    def test_the_observed_split_is_significant_and_directional(self, mod):
        """ORDER 0/8 equal against CONTROL 6/8 equal.

        Pinned as a regression on the finding itself: if a future data or code change
        makes these two ladders look alike, the paper's claim that one invariance
        requirement is human-validated and the other is not has stopped being true.
        """
        p = mod.fisher_exact_two_sided(0, 8, 6, 2)
        assert p < 0.05
        assert p == pytest.approx(0.006993, abs=1e-6)

    def test_pooling_would_have_hidden_it(self, mod):
        """Pooled, the four invariance screens read as a single unremarkable fraction."""
        pooled_equal, pooled_n = 0 + 6, 8 + 8
        assert pooled_equal / pooled_n == pytest.approx(0.375)
        # Split, the two halves are 0.0 and 0.75 -- nothing near the pooled 0.375.
        assert 0 / 8 == 0.0
        assert 6 / 8 == 0.75
