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

# Offline tests for the LoCoBench gold-relation LCS evaluation.
#
# No LLM and no Merlin subprocess: scoring goes through the brute-force oracle in
# `experiments.mock`, which is exact for these small networks.

import copy
import json
import os
import re
from unittest.mock import MagicMock

import pytest

from fact_reasoner.experiments.mock import (
    MAX_BRUTEFORCE_VARS,
    brute_force_run_merlin,
    dry_run_patches,
)
from fact_reasoner.lcs import candidate_pairs as cp
from fact_reasoner.lcs.lcs_scorer import LCS_METHODS, LCSScorer
from fact_reasoner.locoeval import cli
from fact_reasoner.locoeval import gold_graph as gg
from fact_reasoner.locoeval import mined_graph as mg
from fact_reasoner.locoeval import models as lm
from fact_reasoner.locoeval import report as rp
from fact_reasoner.locoeval import runner as rn


# ---------------------------------------------------------------------------
# Fixtures: a small hand-built item exercising every branch.
# ---------------------------------------------------------------------------


def _relation(rid, src, trg, sense, coupling, band, rng, **kw):
    rel = {
        "id": rid,
        "source_id": src,
        "target_id": trg,
        "level2_sense": sense,
        "level1_coupling": coupling,
        "directed": True,
        "ordering_only": False,
        "intended_strength_band": band,
        "strength_range": list(rng),
        "validity": "valid",
        "error_kind": None,
        "is_concession": False,
        "is_resolved_concession": False,
        "resolver_atom_id": None,
        "position_distance": 1,
    }
    rel.update(kw)
    return rel


@pytest.fixture
def item():
    """A 5-atom item with one edge of each interesting shape."""
    return {
        "id": "t-f001-r1",
        "name": "Test -- base",
        "response": "A response that mentions all of the atoms below.",
        "num_atoms": 5,
        "atoms": [
            {"id": "a0", "text": "Atom zero.", "factual": True, "pos": 1},
            {"id": "a1", "text": "Atom one.", "factual": False, "pos": 2},
            {"id": "a2", "text": "Atom two.", "factual": True, "pos": 3},
            {"id": "a3", "text": "Atom three.", "factual": True, "pos": 4},
            {"id": "a4", "text": "The tribunal ultimately held for atom four.",
             "factual": True, "pos": 5},
        ],
        "relations": [
            # Ordering-only: no factor.
            _relation("r0", "a0", "a2", "Precedence", "none", "strong", (0.85, 1.0),
                      ordering_only=True),
            # A planted-invalid contradiction.
            _relation("r1", "a1", "a0", "Contrast", "contradiction", "weak",
                      (0.35, 0.59), directed=False, validity="invalid",
                      error_kind="false_endpoint"),
            # A valid entailment.
            _relation("r2", "a2", "a3", "Evidence", "entailment", "strong",
                      (0.85, 1.0)),
            # A resolved concession, resolver stated.
            _relation("r3", "a3", "a2", "Concession", "contradiction", "moderate",
                      (0.6, 0.84), is_concession=True, is_resolved_concession=True,
                      resolver_atom_id="a4"),
            # A symmetric equivalence.
            _relation("r4", "a3", "a4", "Restatement", "equivalence", "strong",
                      (0.85, 1.0), directed=False),
        ],
        "non_relations": [{"source_id": "a0", "target_id": "a4",
                          "position_distance": 4}],
        "expected": {
            "family_id": "f001",
            "family": "CONFLICT",
            "rung_index": 1,
            "rung_name": "base",
            "perturbation": {"calls": [], "parent_rung": None},
        },
        "meta": {"canonical_topic": "Testing", "domain": "Meta",
                 "framing": "Does it work?", "split": "test",
                 "word_count": 9, "generator": "hand"},
        "notes": "hand-built",
    }


@pytest.fixture(autouse=True)
def _offline_merlin(monkeypatch):
    """Route the scorer's inference through the exact brute-force oracle.

    Also lowers the consistency support term's aux-var batching cap to the
    oracle's 2^n limit; production keeps the high default.
    """
    monkeypatch.setattr(
        "fact_reasoner.lcs.lcs_scorer.run_merlin", brute_force_run_merlin
    )
    monkeypatch.setattr(
        "fact_reasoner.lcs.lcs_scorer.DEFAULT_MAX_NETWORK_VARS",
        MAX_BRUTEFORCE_VARS,
    )


# ---------------------------------------------------------------------------
# Priors and band midpoints.
# ---------------------------------------------------------------------------


def test_atom_priors_from_factual_flag(item):
    priors = gg.atom_priors(item)
    assert priors == {"a0": 0.9, "a1": 0.1, "a2": 0.9, "a3": 0.9, "a4": 0.9}


def test_atom_missing_factual_is_treated_as_not_factual():
    item = {"atoms": [{"id": "a0", "text": "No flag."}]}
    assert gg.atom_priors(item) == {"a0": gg.PRIOR_NOT_FACTUAL}


@pytest.mark.parametrize(
    "rng,expected",
    [((0.85, 1.0), 0.925), ((0.6, 0.84), 0.72), ((0.35, 0.59), 0.47)],
)
def test_band_probability_is_range_midpoint(rng, expected):
    assert gg.band_probability({"strength_range": list(rng)}) == pytest.approx(expected)


def test_band_probability_falls_back_to_canonical_band():
    # No strength_range: the band's canonical range is used.
    assert gg.band_probability(
        {"intended_strength_band": "moderate"}
    ) == pytest.approx(0.72)


def test_band_probability_unknown_band_is_uninformative():
    assert gg.band_probability({"intended_strength_band": "bogus"}) == 0.5
    assert gg.band_probability({}) == 0.5


# ---------------------------------------------------------------------------
# Relation conversion.
# ---------------------------------------------------------------------------


def test_ordering_only_relations_produce_no_factor(item):
    rels, stats = gg.gold_relations(item)
    assert stats["gold_total"] == 5
    assert stats["dropped_ordering_only"] == 1
    assert stats["relations_kept"] == 4
    # The Precedence edge a0->a2 must not appear.
    assert ("a0", "a2") not in {(r.source_id, r.target_id) for r in rels}


def test_concession_discount_uses_gold_resolver(item):
    rels, stats = gg.gold_relations(item)
    assert stats["concessions_discounted"] == 1
    conc = next(r for r in rels if r.concession_resolved)
    # moderate midpoint 0.72, discounted by lambda=0.45.
    assert conc.probability == pytest.approx(0.72 * 0.55)
    assert conc.resolving_atom_id == "a4"
    # The raw strength is preserved: only the factor probability is softened.
    assert conc.strength == pytest.approx(0.72)


def test_concession_discount_can_be_disabled(item):
    rels, _ = gg.gold_relations(item, concession_discount=0.0)
    conc = next(r for r in rels if r.concession_resolved)
    assert conc.probability == pytest.approx(0.72)


def test_gold_is_a_label_so_type_confidence_is_one(item):
    rels, _ = gg.gold_relations(item)
    assert all(r.type_confidence == 1.0 for r in rels)


def test_include_invalid_false_drops_exactly_the_invalid_edges(item):
    rels, stats = gg.gold_relations(item, include_invalid=False)
    assert stats["dropped_invalid"] == 1
    assert stats["relations_kept"] == 3
    assert ("a1", "a0") not in {(r.source_id, r.target_id) for r in rels}


def test_sense_coupling_mismatch_raises(item):
    bad = copy.deepcopy(item)
    # Restatement compiles to equivalence, not contradiction.
    bad["relations"][4]["level1_coupling"] = "contradiction"
    with pytest.raises(gg.GoldGraphError, match="compiles to coupling"):
        gg.gold_relations(bad)


def test_unknown_sense_raises(item):
    bad = copy.deepcopy(item)
    bad["relations"][2]["level2_sense"] = "Elaboration"
    with pytest.raises(gg.GoldGraphError, match="Unknown sense"):
        gg.gold_relations(bad)


def test_resolver_outside_the_item_raises(item):
    bad = copy.deepcopy(item)
    bad["relations"][3]["resolver_atom_id"] = "a99"
    with pytest.raises(gg.GoldGraphError, match="not an atom of this item"):
        gg.gold_relations(bad)


def test_edge_to_unknown_atom_raises(item):
    bad = copy.deepcopy(item)
    bad["relations"][2]["target_id"] = "a99"
    with pytest.raises(gg.GoldGraphError, match="unknown atom"):
        gg.build_gold_result(bad)


# ---------------------------------------------------------------------------
# The assembled MiningResult.
# ---------------------------------------------------------------------------


def test_build_gold_result_shape(item):
    res = gg.build_gold_result(item)
    assert set(res.atoms) == {"a0", "a1", "a2", "a3", "a4"}
    assert len(res.relations) == 4
    # Fact-graph nodes carry the 0.9/0.1 priors.
    probs = {n.id: n.probability for n in res.fact_graph.get_nodes()}
    assert probs == {"a0": 0.9, "a1": 0.1, "a2": 0.9, "a3": 0.9, "a4": 0.9}
    # One unary factor per atom plus one pairwise factor per relation.
    assert len(res.markov_network.factors) == 5 + 4
    assert res.config["relation_source"] == "gold"
    assert res.config["prior_source"] == "per_atom"
    # `prior` must stay a float: LCSScorer reads float(config["prior"]).
    assert isinstance(res.config["prior"], float)


def test_scorer_resolves_the_gold_priors(item):
    res = gg.build_gold_result(item)
    scorer = LCSScorer("unused-merlin-path")
    assert scorer._node_priors(res) == {
        "a0": 0.9, "a1": 0.1, "a2": 0.9, "a3": 0.9, "a4": 0.9
    }


def test_mining_result_serializes(item):
    res = gg.build_gold_result(item)
    payload = json.loads(json.dumps(res.to_json()))
    assert set(payload["atoms"]) == set(res.atoms)
    assert len(payload["relations"]) == 4


def test_all_four_readouts_score(item):
    res = gg.build_gold_result(item)
    out = LCSScorer("unused-merlin-path").score_all(
        res, node_priors=res.config["node_priors"]
    )
    for method in ("mean_marginal", "consistency", "reified", "log_partition"):
        assert out[method] is not None
        assert 0.0 <= out[method] <= 1.0
    assert out["log_z"] is not None


def test_dropping_invalid_conflict_edges_raises_consistency(item):
    """The planted-invalid contradiction is what makes the item inconsistent."""
    scorer = LCSScorer("unused-merlin-path")

    def consistency(include_invalid):
        res = gg.build_gold_result(item, include_invalid=include_invalid)
        return scorer.score_all(
            res, methods=("consistency",), node_priors=res.config["node_priors"]
        )["consistency"]

    assert consistency(False) > consistency(True)


# ---------------------------------------------------------------------------
# Ladder constraint evaluation.
# ---------------------------------------------------------------------------


_FAMILY = {
    "family_id": "f001",
    "ordering_constraints": [
        {
            "id": "c1",
            "class": "C1",
            "strict": True,
            "pairs": [
                {"readout": "mean_marginal", "pair": [0, 1]},
                {"readout": "mean_marginal", "pair": [1, 2]},
            ],
        },
        {
            "id": "c2",
            "class": "C2",
            "strict": False,
            "pairs": [{"readout": "consistency", "pair": [1, 2],
                       "expect": "decrease"}],
        },
        {
            "id": "c3",
            "class": "C3",
            "strict": True,
            "readouts": ["mean_marginal"],
            "required": [[0, 2]],
        },
    ],
}


def test_monotone_scores_satisfy_the_increase_constraints():
    scores = {
        0: {"mean_marginal": 0.40, "consistency": 0.30},
        1: {"mean_marginal": 0.50, "consistency": 0.30},
        2: {"mean_marginal": 0.60, "consistency": 0.20},
    }
    checks = rn.evaluate_constraints(_FAMILY, scores)
    assert all(c["passed"] for c in checks)
    summary = rn.summarize_constraints(checks)
    assert summary["passed"] == summary["total"] == 4
    assert summary["by_class"]["C1"] == {"total": 2, "passed": 2}


def test_flat_scores_fail_increase_and_pass_invariance():
    """The arithmetic signature of the duplication bug: everything ties."""
    flat = {i: {"mean_marginal": 0.5, "consistency": 0.3} for i in range(3)}
    checks = rn.evaluate_constraints(_FAMILY, flat)
    by_class = {c["constraint_class"]: [] for c in checks}
    for c in checks:
        by_class[c["constraint_class"]].append(c)
    # C1 (strict increase) and C3 (endpoint separation) cannot hold when tied.
    assert not any(c["passed"] for c in by_class["C1"])
    assert not any(c["passed"] for c in by_class["C3"])
    # The C2 assertion here expects a decrease, so a tie fails it too.
    assert not any(c["passed"] for c in by_class["C2"])
    assert all(c["observed"] == "invariant" for c in checks)


def test_invariant_expectation_passes_on_a_tie():
    fam = {
        "family_id": "f",
        "ordering_constraints": [
            {"id": "c2", "class": "C2", "strict": False,
             "pairs": [{"readout": "log_partition", "pair": [1, 2],
                        "expect": "invariant"}]}
        ],
    }
    flat = {1: {"log_partition": 0.25}, 2: {"log_partition": 0.25}}
    checks = rn.evaluate_constraints(fam, flat)
    assert len(checks) == 1
    assert checks[0]["passed"] is True
    assert checks[0]["observed"] == "invariant"


def test_missing_score_is_reported_unknown_not_passed():
    checks = rn.evaluate_constraints(_FAMILY, {0: {"mean_marginal": 0.4}})
    unknown = [c for c in checks if c["observed"] == "unknown"]
    assert unknown
    assert not any(c["passed"] for c in unknown)


def test_c3_invariant_shape_is_handled():
    """A control family asserts its endpoints are EQUAL, via `invariant`."""
    fam = {
        "family_id": "flat",
        "ordering_constraints": [
            {"id": "c3-flat", "class": "C3", "strict": False,
             "readouts": ["mean_marginal"], "invariant": [[0, 4]]}
        ],
    }
    checks = rn.evaluate_constraints(
        fam, {0: {"mean_marginal": 0.5}, 4: {"mean_marginal": 0.5}}
    )
    assert len(checks) == 1
    assert checks[0]["expected"] == "invariant"
    assert checks[0]["passed"] is True


# ---------------------------------------------------------------------------
# Loading, and the duplication detector.
# ---------------------------------------------------------------------------


def _write_dataset(tmp_path, items, families=None):
    os.makedirs(tmp_path, exist_ok=True)
    with open(tmp_path / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    if families is not None:
        with open(tmp_path / "families.json", "w") as f:
            json.dump({"manifest_version": "1.0", "families": families}, f)
    return str(tmp_path)


def test_load_items_filters_and_orders(tmp_path, item):
    second = copy.deepcopy(item)
    second["id"] = "t-f001-r2"
    d = _write_dataset(tmp_path, [item, second])
    assert [i["id"] for i in rn.load_items(d)] == ["t-f001-r1", "t-f001-r2"]
    assert [i["id"] for i in rn.load_items(d, ["t-f001-r2"])] == ["t-f001-r2"]


def test_load_items_rejects_unknown_id(tmp_path, item):
    d = _write_dataset(tmp_path, [item])
    with pytest.raises(ValueError, match="Unknown item id"):
        rn.load_items(d, ["nope"])


def test_load_families_absent_is_empty(tmp_path, item):
    d = _write_dataset(tmp_path, [item])
    assert rn.load_families(d) == {}


def test_relations_identical_detects_duplication(item):
    a = copy.deepcopy(item)
    b = copy.deepcopy(item)
    b["id"] = "t-f001-r2"
    b["response"] = "A different response text entirely."
    # Same relations, different text: exactly the duplication bug's signature.
    assert rn._relations_identical([a, b]) is True

    b["relations"] = b["relations"][:-1]
    assert rn._relations_identical([a, b]) is False


# ---------------------------------------------------------------------------
# End-to-end sweep + report rendering (offline).
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset(tmp_path, item):
    """Two rungs of one family, sharing gold relations (as the corpus does)."""
    r1 = copy.deepcopy(item)
    r2 = copy.deepcopy(item)
    r2["id"] = "t-f001-r2"
    r2["name"] = "Test -- concession_resolved"
    r2["response"] = "A perturbed response, different text, same gold relations."
    r2["expected"] = dict(item["expected"], rung_index=2,
                          rung_name="concession_resolved")
    families = [
        {
            "family_id": "f001",
            "family": "CONFLICT",
            "canonical_topic": "Testing",
            "rungs": [
                {"index": 1, "name": "base", "item_id": "t-f001-r1"},
                {"index": 2, "name": "concession_resolved",
                 "item_id": "t-f001-r2"},
            ],
            "ordering_constraints": [
                {"id": "c1", "class": "C1", "strict": True,
                 "pairs": [{"readout": "mean_marginal", "pair": [1, 2]}]}
            ],
        }
    ]
    return _write_dataset(tmp_path / "data", [r1, r2], families), tmp_path


def test_sweep_scores_every_item_and_arm(dataset, tmp_path):
    data_dir, root = dataset
    out_dir = str(root / "out")
    results = rn.GoldEvalRunner(
        data_dir=data_dir, output_dir=out_dir, merlin_path="unused"
    ).run()

    assert len(results["records"]) == 4  # 2 items x 2 arms
    assert not any("error" in r for r in results["records"])
    for rec in results["records"]:
        assert rec["lcs"]["mean_marginal"] is not None
        assert rec["num_relations"] == (4 if rec["arm"] == "gold" else 3)

    # Artefacts on disk.
    assert os.path.exists(os.path.join(out_dir, "results.json"))
    assert len(os.listdir(os.path.join(out_dir, "records"))) == 4
    assert len(os.listdir(os.path.join(out_dir, "by_item"))) == 2


def test_sweep_flags_identical_gold_relations(dataset, tmp_path):
    data_dir, root = dataset
    results = rn.GoldEvalRunner(
        data_dir=data_dir, output_dir=str(root / "out2"), merlin_path="unused"
    ).run()
    fam = results["families"][0]
    assert fam["gold_relations_identical_across_rungs"] is True
    assert fam["distinct_responses"] == 2
    # And the strict-increase constraint therefore cannot hold.
    checks = fam["arms"]["gold"]["checks"]
    assert checks and not any(c["passed"] for c in checks)
    assert all(c["observed"] == "invariant" for c in checks)


def test_dataset_summary_counts(dataset, tmp_path):
    data_dir, root = dataset
    results = rn.GoldEvalRunner(
        data_dir=data_dir, output_dir=str(root / "out3"), merlin_path="unused"
    ).run()
    ds = results["dataset"]
    assert ds["num_items"] == 2
    assert ds["num_atoms"] == 10
    assert ds["num_atoms_factual"] == 8
    assert ds["num_atoms_not_factual"] == 2
    assert ds["couplings"]["none"] == 2  # one Precedence per item
    assert ds["validity"]["invalid"] == 2
    assert ds["error_kinds"]["false_endpoint"] == 2


def test_bad_arm_is_rejected():
    with pytest.raises(ValueError, match="Unknown arm"):
        rn.GoldEvalRunner(
            data_dir=".", output_dir=".", merlin_path="m", arms=("gold", "bogus")
        )


def test_report_renders_valid_tex(dataset, tmp_path):
    data_dir, root = dataset
    out_dir = str(root / "out4")
    results = rn.GoldEvalRunner(
        data_dir=data_dir, output_dir=out_dir, merlin_path="unused"
    ).run()
    path = rp.write_report(results, out_dir)
    tex = open(path).read()

    # Balanced document and the expected sections.
    assert tex.count(r"\begin{document}") == 1
    assert tex.count(r"\end{document}") == 1
    assert tex.count(r"\begin{tabular}") == tex.count(r"\end{tabular}")
    assert tex.count(r"\begin{table}") == tex.count(r"\end{table}")
    assert tex.count(r"\begin{tikzpicture}") == tex.count(r"\end{tikzpicture}")
    for section in ("Setup", "Dataset", "LCS scores",
                    "Ladder ordering constraints", "Worked examples",
                    "Findings", "Threats to validity"):
        assert f"\\section{{{section}}}" in tex

    # The duplication finding is stated, not buried.
    assert "identical gold relation set" in tex
    # A worked example, its graph, and its specific relations are all present.
    assert r"\subsection{Test -- base" in tex
    assert r"\begin{tikzpicture}" in tex
    assert "The specific gold relations" in tex
    # The modelling choices are on the page.
    assert "0.9" in tex and "0.1" in tex


def test_report_switches_narrative_when_relations_vary(dataset, tmp_path):
    """The duplication finding must not be asserted about a dataset that fixed it.

    The report's ladder narrative is driven by `gold_relations_identical_across_rungs`,
    so a regenerated corpus gets the "every rung carries its own relations" reading and a
    pre-fix corpus gets the defect reading. Getting this backwards would have the report
    stating something false about the data in front of it.
    """
    import copy as _copy

    data_dir, root = dataset
    out_dir = str(root / "out-vary")
    results = rn.GoldEvalRunner(
        data_dir=data_dir, output_dir=out_dir, merlin_path="unused"
    ).run()

    # As generated by the pre-fix pipeline: relations repeat across rungs.
    assert results["families"][0]["gold_relations_identical_across_rungs"] is True
    tex_dup = open(rp.write_report(results, out_dir)).read()
    assert "identical gold relation set" in tex_dup
    assert "has since been fixed" in tex_dup

    # Now the post-fix shape: per-rung relations, constraints satisfied.
    varying = _copy.deepcopy(results)
    for fam in varying["families"]:
        fam["gold_relations_identical_across_rungs"] = False
        for arm in fam["arms"].values():
            for check in arm["checks"]:
                check["passed"] = True
                check["observed"] = check["expected"]
    tex_vary = open(
        rp.write_report(varying, out_dir, filename="report-vary.tex")
    ).read()
    assert "Every rung carries its own gold relations" in tex_vary
    assert "identical gold relation set" not in tex_vary
    assert "has since been fixed" not in tex_vary
    # And the threats section flips from "corpus measurement" to the real limitation.
    assert "tests the readouts, not the miner" in tex_vary
    assert "is a corpus measurement" not in tex_vary


def test_report_escapes_latex_specials(dataset, tmp_path):
    data_dir, root = dataset
    out_dir = str(root / "out5")
    # Inject characters that would break a LaTeX build if unescaped.
    items = rn.load_items(data_dir)
    items[0]["response"] = "100% of a_b & c #d $e {f} ~g ^h"
    with open(os.path.join(data_dir, "items.jsonl"), "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")

    results = rn.GoldEvalRunner(
        data_dir=data_dir, output_dir=out_dir, merlin_path="unused"
    ).run()
    tex = open(rp.write_report(results, out_dir)).read()
    assert r"100\%" in tex
    assert r"a\_b" in tex
    assert r"\&" in tex
    assert r"\#d" in tex


def test_report_honours_explicit_example_ids(dataset, tmp_path):
    data_dir, root = dataset
    out_dir = str(root / "out6")
    results = rn.GoldEvalRunner(
        data_dir=data_dir, output_dir=out_dir, merlin_path="unused"
    ).run()
    tex = open(
        rp.write_report(results, out_dir, example_ids=["t-f001-r2"])
    ).read()
    assert "t-f001-r2" in tex
    assert r"\subsection{Test -- concession_resolved" not in tex  # escaped form
    assert "concession\\_resolved" in tex


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def test_cli_end_to_end_without_pdf(dataset, tmp_path):
    from fact_reasoner.locoeval import cli

    data_dir, root = dataset
    out_dir = str(root / "cli-out")
    # A truthy merlin path is required by the scorer; inference is stubbed.
    fake_merlin = root / "merlin"
    fake_merlin.write_text("#!/bin/sh\n")

    rc = cli.main(
        [
            "--data-dir", data_dir,
            "--out-dir", out_dir,
            "--merlin-path", str(fake_merlin),
            "--no-pdf",
        ]
    )
    assert rc == 0
    assert os.path.exists(os.path.join(out_dir, "results.json"))
    assert os.path.exists(os.path.join(out_dir, "report.tex"))
    assert not os.path.exists(os.path.join(out_dir, "report.pdf"))


def test_cli_report_only_reuses_results(dataset, tmp_path):
    from fact_reasoner.locoeval import cli

    data_dir, root = dataset
    out_dir = str(root / "cli-out2")
    fake_merlin = root / "merlin2"
    fake_merlin.write_text("#!/bin/sh\n")
    cli.main(
        ["--data-dir", data_dir, "--out-dir", out_dir,
         "--merlin-path", str(fake_merlin), "--no-report"]
    )
    assert not os.path.exists(os.path.join(out_dir, "report.tex"))

    # Now render from the saved results, with no scoring and no merlin at all.
    assert cli.main(["--out-dir", out_dir, "--report-only", "--no-pdf"]) == 0
    assert os.path.exists(os.path.join(out_dir, "report.tex"))


def test_cli_requires_merlin_path_when_scoring(dataset, tmp_path, monkeypatch):
    from fact_reasoner.locoeval import cli

    data_dir, root = dataset
    monkeypatch.delenv("MERLIN_PATH", raising=False)
    with pytest.raises(SystemExit):
        cli.main(["--data-dir", data_dir, "--out-dir", str(root / "x")])


def test_cli_rejects_missing_merlin_executable(dataset, tmp_path):
    from fact_reasoner.locoeval import cli

    data_dir, root = dataset
    with pytest.raises(SystemExit):
        cli.main(
            ["--data-dir", data_dir, "--out-dir", str(root / "y"),
             "--merlin-path", str(root / "nope")]
        )


def test_cli_report_only_without_results_errors(tmp_path):
    from fact_reasoner.locoeval import cli

    with pytest.raises(SystemExit):
        cli.main(["--out-dir", str(tmp_path / "empty"), "--report-only"])


def test_relation_graph_skips_unknown_couplings():
    """A record with an unrecognized coupling must not emit a broken edge."""
    record = {
        "num_atoms": 3,
        "relations": [
            {"source": "a0", "target": "a1", "type": "entailment",
             "probability": 0.9},
            {"source": "a0", "target": "a2", "type": "bogus", "probability": 0.9},
        ],
    }
    pic = rp._relation_graph(record, {"a0": 0.9, "a1": 0.1, "a2": 0.9})
    assert pic.count(r"\draw") == 1
    # The non-factual atom is tinted differently.
    assert "red!18" in pic


def test_relation_graph_handles_no_atoms():
    assert "no atoms" in rp._relation_graph({"num_atoms": 0, "relations": []})


# ---------------------------------------------------------------------------
# Mined arms: naming, validation and the model inventory.
# ---------------------------------------------------------------------------


def test_parse_arm_returns_none_for_gold_arms():
    assert mg.parse_arm("gold") is None
    assert mg.parse_arm("gold_valid") is None


def test_parse_arm_extracts_model_and_policy():
    spec = mg.parse_arm("mined:llama-3.3-70b-instruct:windowed")
    assert spec == mg.MinedArm(model="llama-3.3-70b-instruct",
                               pair_policy="windowed")
    # The arm name round-trips, so records and report labels agree.
    assert spec.arm == "mined:llama-3.3-70b-instruct:windowed"


def test_parse_arm_rejects_unknown_pair_policy():
    # RelationMiner does not validate pair_policy; catching it at parse time is
    # what keeps a typo from costing a whole cell of tokens.
    with pytest.raises(mg.MinedArmError, match="Unknown pair policy"):
        mg.parse_arm("mined:some-model:sliding")


@pytest.mark.parametrize("arm", ["mined:onlytwo", "mined:a:b:c", "mined::windowed"])
def test_parse_arm_rejects_malformed_arms(arm):
    with pytest.raises(mg.MinedArmError):
        mg.parse_arm(arm)


def test_format_arm_matches_parse_arm():
    arm = mg.format_arm("m1", "all_pairs")
    assert mg.parse_arm(arm) == mg.MinedArm(model="m1", pair_policy="all_pairs")


def test_count_call_exceptions_tallies_by_type():
    outs = [object(), ValueError("x"), RuntimeError("y"), ValueError("z"), None]
    total, kinds = mg.count_call_exceptions(outs)
    assert total == 3
    assert kinds == {"ValueError": 2, "RuntimeError": 1}


def test_load_model_specs_reads_the_repo_inventory():
    specs = lm.load_model_specs("configs/rits_models.json")
    spec = lm.resolve_model("llama-3.3-70b-instruct", specs)
    assert spec.model_id == "meta-llama/llama-3-3-70b-instruct"
    assert spec.backend == "rits"
    # A RITS endpoint must be explicit: the catalog cannot see RITS without
    # mellea_ibm installed, and a friendly-name lookup can 404.
    assert spec.base_url and spec.base_url.endswith("llama-3-3-70b-instruct")
    assert spec.has_logprobs is True


def test_resolve_model_rejects_an_unknown_name():
    specs = lm.load_model_specs("configs/rits_models.json")
    with pytest.raises(ValueError, match="Unknown model"):
        lm.resolve_model("gpt-oss-120b", specs)  # served name is -a100


def test_load_model_specs_rejects_duplicate_names(tmp_path):
    p = tmp_path / "dup.json"
    p.write_text(json.dumps([
        {"name": "m", "model_id": "x", "backend": "rits"},
        {"name": "m", "model_id": "y", "backend": "rits"},
    ]))
    with pytest.raises(ValueError, match="duplicate model name"):
        lm.load_model_specs(str(p))


# ---------------------------------------------------------------------------
# Mined vs gold: edge-level agreement.
# ---------------------------------------------------------------------------


def test_compare_to_gold_is_perfect_on_gold_itself(item):
    """Feeding gold's own relations back in must recover everything.

    The identity test: any bug in key construction (direction handling, coupling
    spelling, ordering-only filtering) shows up here as a shortfall.
    """
    result = gg.build_gold_result(item)
    comp = mg.compare_to_gold(item, result.relations)
    assert comp["gold_edges_total"] == 5
    assert comp["gold_edges_scorable"] == 4  # the Precedence produces no factor
    for level in ("pair", "coupling", "sense"):
        assert comp[level]["precision"] == 1.0
        assert comp[level]["recall"] == 1.0
        assert comp[level]["fn"] == 0


def test_compare_to_gold_matches_undirected_edges_written_either_way(item):
    """An undirected coupling is one edge regardless of which way it is written.

    This is what keeps a forward-only pair policy from being charged for the
    ordering of a symmetric relation it did find.
    """
    result = gg.build_gold_result(item)
    flipped = []
    for rel in result.relations:
        if not rel.directed:
            rel = copy.copy(rel)
            rel.source_id, rel.target_id = rel.target_id, rel.source_id
        flipped.append(rel)
    comp = mg.compare_to_gold(item, flipped)
    assert comp["coupling"]["recall"] == 1.0
    assert comp["coupling"]["fp"] == 0


def test_compare_to_gold_requires_direction_for_directed_couplings(item):
    """A reversed entailment is a different claim, so it must not match."""
    result = gg.build_gold_result(item)
    reversed_rels = []
    for rel in result.relations:
        if rel.directed:
            rel = copy.copy(rel)
            rel.source_id, rel.target_id = rel.target_id, rel.source_id
        reversed_rels.append(rel)
    comp = mg.compare_to_gold(item, reversed_rels)
    # Pair-level still matches (same two atoms); coupling-level does not.
    assert comp["pair"]["recall"] == 1.0
    assert comp["coupling"]["recall"] < 1.0


def test_compare_to_gold_excludes_ordering_only_from_the_denominator(item):
    comp = mg.compare_to_gold(item, [])
    assert comp["gold_edges_scorable"] == 4
    assert comp["coupling"]["fn"] == 4  # not 5: the Precedence is not scorable
    assert comp["coupling"]["recall"] == 0.0


def test_compare_to_gold_counts_non_relation_violations(item):
    """An edge on a pair the item declares UNrelated is a measurable error."""
    result = gg.build_gold_result(item)
    intruder = copy.copy(result.relations[0])
    intruder.source_id, intruder.target_id = "a0", "a4"  # the declared non-relation
    comp = mg.compare_to_gold(item, [*result.relations, intruder])
    assert comp["non_relation_pairs"] == 1
    assert comp["non_relation_violations"] == 1
    assert comp["non_relation_violation_rate"] == 1.0


def test_compare_to_gold_stratifies_recall(item):
    comp = mg.compare_to_gold(item, gg.build_gold_result(item).relations)
    # a1->a0 and a3->a2 run backward in atom order; a2->a3 and a3->a4 forward.
    assert comp["recall_by_direction"]["backward"]["total"] == 2
    assert comp["recall_by_direction"]["forward"]["total"] == 2
    assert comp["recall_by_validity"]["invalid"]["total"] == 1
    assert comp["recall_by_coupling"]["entailment"]["total"] == 1
    assert all(
        cell["recall"] == 1.0
        for cell in comp["recall_by_coupling"].values()
    )


def test_count_duplicate_unordered_pairs(item):
    """all_pairs can label a pair in both directions; factors are not deduped."""
    rels = gg.build_gold_result(item).relations
    assert mg.count_duplicate_unordered_pairs(rels) == 1  # a2<->a3 twice in gold
    assert mg.count_duplicate_unordered_pairs([]) == 0


def test_aggregate_comparisons_micro_averages(item):
    comp = mg.compare_to_gold(item, gg.build_gold_result(item).relations)
    agg = mg.aggregate_comparisons([comp, comp])
    assert agg["num_items"] == 2
    assert agg["coupling"]["tp"] == 2 * comp["coupling"]["tp"]
    assert agg["coupling"]["recall"] == 1.0
    assert agg["gold_edges_scorable"] == 8
    # Stratified cells are summed, then divided once (micro, not a mean of means).
    assert agg["recall_by_direction"]["backward"]["total"] == 4


def test_aggregate_comparisons_handles_no_blocks():
    assert mg.aggregate_comparisons([])["num_items"] == 0


# ---------------------------------------------------------------------------
# Pair-policy reach. No LLM and no Merlin: selection is pure.
# ---------------------------------------------------------------------------


def _sixteen_atoms():
    from fact_reasoner.core.base import Atom

    return {f"a{i}": Atom(id=f"a{i}", text=f"Sentence number {i} about topic {i}.")
            for i in range(16)}


def test_all_pairs_and_windowed_differ_in_reach():
    """Pins the pair arithmetic the cost estimate and the report prose rest on.

    `windowed` is NOT simply the order window. Response-grounded refinement both
    promotes out-of-window forward pairs the prose links and demotes in-window
    pairs it does not, so the selected count is `num_window_pairs + promoted -
    demoted` and is bounded by the forward pairs, never by the window alone.
    """
    atoms = _sixteen_atoms()
    response = " ".join(a.text for a in atoms.values())
    all_pairs, acov = cp.select(atoms, response=response, policy="all_pairs")
    windowed, wcov = cp.select(atoms, response=response, policy="windowed", window=4)

    assert len(all_pairs) == 16 * 15 == 240  # every ordered pair, both directions
    assert acov["discourse_anchored"] is False  # all_pairs skips refinement

    assert wcov["num_window_pairs"] == 54  # sum_i min(4, 15-i)
    assert wcov["discourse_anchored"] is True
    assert len(windowed) == 54 + wcov["num_promoted"] - wcov["num_demoted"]
    assert len(windowed) <= wcov["forward_pairs_possible"] == 120
    # Whatever refinement does, windowed stays strictly cheaper than all_pairs.
    assert len(windowed) < len(all_pairs)


def test_all_pairs_visits_each_unordered_pair_twice():
    """Why all_pairs scores a denser MRF: two factors per pair are possible."""
    atoms = _sixteen_atoms()
    response = " ".join(a.text for a in atoms.values())
    pairs, _ = cp.select(atoms, response=response, policy="all_pairs")
    unordered = {tuple(sorted(p)) for p in pairs}
    assert len(pairs) == 2 * len(unordered)


def test_windowed_is_forward_only():
    """The structural reason a backward DIRECTED gold edge is unreachable.

    Discourse refinement can push a pair past the window radius, but never
    reverses one: every selected pair still runs source-before-target. So a
    directed gold edge written backward in atom order cannot be recovered under
    this policy no matter what the prose says -- a property of the policy, not a
    miner failure, and the reason recall is reported split by direction.
    """
    atoms = _sixteen_atoms()
    response = " ".join(a.text for a in atoms.values())
    pairs, _ = cp.select(atoms, response=response, policy="windowed", window=4)
    idx = lambda aid: int(aid[1:])  # noqa: E731
    assert pairs
    assert all(idx(t) - idx(s) > 0 for s, t in pairs)


def test_all_pairs_reaches_backward_pairs_that_windowed_cannot():
    """The contrast that makes the policy comparison legible."""
    atoms = _sixteen_atoms()
    response = " ".join(a.text for a in atoms.values())
    all_pairs, _ = cp.select(atoms, response=response, policy="all_pairs")
    windowed, _ = cp.select(atoms, response=response, policy="windowed", window=4)
    idx = lambda aid: int(aid[1:])  # noqa: E731
    assert any(idx(t) - idx(s) < 0 for s, t in all_pairs)
    assert not any(idx(t) - idx(s) < 0 for s, t in windowed)


# ---------------------------------------------------------------------------
# The mined cell in the runner (mock LLM + brute-force Merlin).
# ---------------------------------------------------------------------------


MINED_ARM = "mined:m1:windowed"


@pytest.fixture
def mined_specs():
    return {"m1": lm.ModelSpec(name="m1", model_id="m1", backend="rits")}


@pytest.fixture
def mock_llm():
    """Stub only the LLM; the autouse fixture already routes Merlin to the oracle."""
    with dry_run_patches(patch_merlin=False):
        yield


def _mined_runner(dataset, out, specs, **kw):
    data_dir, root = dataset
    return rn.GoldEvalRunner(
        data_dir=data_dir,
        output_dir=str(root / out),
        merlin_path="unused",
        model_specs=specs,
        backend_factory=lambda spec: MagicMock(name=spec.name),
        **kw,
    )


def test_runner_rejects_a_mined_arm_without_an_inventory():
    with pytest.raises(ValueError, match="not in the model inventory"):
        rn.GoldEvalRunner(
            data_dir=".", output_dir=".", merlin_path="m", arms=(MINED_ARM,)
        )


def test_runner_rejects_a_mined_arm_with_a_bad_policy(mined_specs):
    with pytest.raises(mg.MinedArmError, match="Unknown pair policy"):
        rn.GoldEvalRunner(
            data_dir=".", output_dir=".", merlin_path="m",
            arms=("mined:m1:sliding",), model_specs=mined_specs,
        )


def test_bad_gold_arm_is_still_rejected(mined_specs):
    """The mined-arm parser must not have widened the gold vocabulary."""
    with pytest.raises(ValueError, match="Unknown arm"):
        rn.GoldEvalRunner(
            data_dir=".", output_dir=".", merlin_path="m",
            arms=("gold", "bogus"), model_specs=mined_specs,
        )


def test_record_filename_slugifies_a_mined_arm():
    name = rn.GoldEvalRunner._record_filename(
        "it/1", "mined:llama-3.3-70b-instruct:windowed"
    )
    assert name == "it_1__mined_llama-3_3-70b-instruct_windowed.json"
    assert ":" not in name and "/" not in name


def test_mined_cell_scores_all_readouts_with_fixed_priors(
    dataset, mined_specs, mock_llm
):
    runner = _mined_runner(
        dataset, "mined1", mined_specs, arms=("gold", MINED_ARM), max_concurrency=4
    )
    results = runner.run()
    mined = [r for r in results["records"] if r["arm"] == MINED_ARM]
    assert len(mined) == 2  # two items in the dataset fixture
    for rec in mined:
        assert "error" not in rec, rec.get("error")
        assert rec["relation_source"] == "mined"
        assert rec["model"] == "m1"
        assert rec["pair_policy"] == "windowed"
        # Every readout computed, and the priors are the item's own 0.9/0.1 with
        # nothing defaulted to the scorer's uniform 0.5.
        for method in LCS_METHODS:
            assert rec["lcs"][method] is not None
        assert set(rec["node_priors"].values()) <= {0.9, 0.1}
        # "auto" is resolved inside the miner, so the record must show the real one.
        assert rec["strength_method"] in (
            "surrogate_logprobs", "surrogate_sampled", "verbalized"
        )
        assert rec["comparison"]["gold_edges_scorable"] > 0


def test_gold_cells_carry_no_comparison_block(dataset, mined_specs, mock_llm):
    runner = _mined_runner(dataset, "mined2", mined_specs, arms=("gold", MINED_ARM))
    results = runner.run()
    for rec in results["records"]:
        if rec["arm"] == "gold":
            assert "comparison" not in rec
            assert rec["pair_policy"] == "gold"
            assert rec["model"] is None


def test_mining_summary_micro_averages_per_arm(dataset, mined_specs, mock_llm):
    runner = _mined_runner(dataset, "mined3", mined_specs, arms=("gold", MINED_ARM))
    results = runner.run()
    summary = results["mining"]
    assert set(summary) == {MINED_ARM}
    block = summary[MINED_ARM]
    assert block["num_items"] == 2
    assert block["model"] == "m1"
    assert block["pair_policy"] == "windowed"
    assert block["num_call_exceptions"] == 0
    assert "coupling" in block and "recall_by_direction" in block


def test_mined_cell_fails_when_the_atom_set_changes(item, monkeypatch, mock_llm):
    """A dropped atom would silently send its prior to the scorer's 0.5 default.

    Patches the MINER (not the function under test) so the real guard runs.
    """
    import asyncio

    from fact_reasoner.lcs.relation_miner import RelationMiner

    real_mine = RelationMiner.amine_from_atoms

    async def _losing_an_atom(self, atoms, response, **kw):
        result = await real_mine(self, atoms, response, **kw)
        result.atoms.pop("a0")  # as a duplicate-id collapse would
        return result

    monkeypatch.setattr(RelationMiner, "amine_from_atoms", _losing_an_atom)
    with pytest.raises(mg.MinedArmError, match="atom set changed"):
        asyncio.run(
            mg.abuild_mined_result(
                item, backend=MagicMock(), pair_policy="windowed",
                nli_method="logprobs",
            )
        )


def test_mined_cell_requires_a_response(mined_specs, mock_llm, tmp_path):
    """Mining is always response-grounded, so an empty response must be an error."""
    import asyncio
    bad = {"id": "x", "response": "   ", "atoms": [{"id": "a0", "text": "t",
                                                    "factual": True}]}
    with pytest.raises(mg.MinedArmError, match="response-grounded"):
        asyncio.run(
            mg.abuild_mined_result(bad, backend=MagicMock(),
                                   pair_policy="windowed", nli_method="logprobs")
        )


def test_mined_cell_refuses_a_high_call_error_rate(
    dataset, mined_specs, monkeypatch, mock_llm
):
    """A throttled endpoint must fail the cell, not quietly report a sparse graph."""
    real = mg.abuild_mined_result

    async def _with_errors(itm, **kw):
        result = await real(itm, **kw)
        cov = dict(result.coverage or {})
        cov["llm_calls"] = 100
        cov["llm_call_errors"] = 50
        cov["llm_call_errors_by_type"] = {"TimeoutError": 50}
        result.coverage = cov
        return result

    monkeypatch.setattr(rn, "abuild_mined_result", _with_errors)
    runner = _mined_runner(dataset, "mined4", mined_specs, arms=(MINED_ARM,))
    results = runner.run()
    for rec in results["records"]:
        assert "error" in rec
        assert "LLM calls failed" in rec["error"]


def test_call_error_rate_below_the_ceiling_is_recorded_not_fatal(
    dataset, mined_specs, monkeypatch, mock_llm
):
    real = mg.abuild_mined_result

    async def _one_error(itm, **kw):
        result = await real(itm, **kw)
        cov = dict(result.coverage or {})
        cov["llm_calls"] = 1000
        cov["llm_call_errors"] = 1  # 0.1%, under the 2% default
        result.coverage = cov
        return result

    monkeypatch.setattr(rn, "abuild_mined_result", _one_error)
    runner = _mined_runner(dataset, "mined5", mined_specs, arms=(MINED_ARM,))
    results = runner.run()
    for rec in results["records"]:
        assert "error" not in rec
        assert rec["num_call_exceptions"] == 1
        assert rec["call_error_rate"] == pytest.approx(0.001)


# ---------------------------------------------------------------------------
# Resume.
# ---------------------------------------------------------------------------


def test_resume_reuses_a_completed_record(dataset, mined_specs, mock_llm):
    runner = _mined_runner(dataset, "res1", mined_specs, arms=(MINED_ARM,))
    runner.run()

    again = _mined_runner(dataset, "res1", mined_specs, arms=(MINED_ARM,), resume=True)
    calls = []
    orig = again._run_cell
    again._run_cell = lambda i, a: (calls.append((i["id"], a)), orig(i, a))[1]
    results = again.run()
    assert calls == []  # nothing re-run
    assert len(results["records"]) == 2


def test_resume_reruns_a_failed_record(dataset, mined_specs, mock_llm):
    data_dir, root = dataset
    out = root / "res2"
    (out / "records").mkdir(parents=True)
    fname = rn.GoldEvalRunner._record_filename("t-f001-r0", "gold")
    (out / "records" / fname).write_text(json.dumps({"error": "boom"}))

    runner = _mined_runner(dataset, "res2", mined_specs, arms=("gold",), resume=True)
    results = runner.run()
    assert all("error" not in r for r in results["records"])


def test_resume_discards_a_stale_fingerprint(dataset, mined_specs, mock_llm, capsys):
    runner = _mined_runner(dataset, "res3", mined_specs, arms=(MINED_ARM,), window=4)
    runner.run()
    # Same cells, different mining configuration: the cache must not be trusted.
    again = _mined_runner(
        dataset, "res3", mined_specs, arms=(MINED_ARM,), window=6, resume=True
    )
    again.run()
    assert "discarding" in capsys.readouterr().out


def test_run_fingerprint_tracks_the_mining_knobs(mined_specs):
    def fp(**kw):
        return rn.GoldEvalRunner(
            data_dir=".", output_dir=".", merlin_path="m",
            model_specs=mined_specs, **kw,
        )._run_fingerprint()

    base = fp()
    assert fp() == base  # deterministic
    assert fp(window=6) != base
    assert fp(strength_method="verbalized") != base
    assert fp(ibound=8) != base


# ---------------------------------------------------------------------------
# Report: mined arms, generalized deltas, new tables.
# ---------------------------------------------------------------------------


def _mined_results(dataset, out, specs, arms, **kw):
    runner = _mined_runner(dataset, out, specs, arms=arms, **kw)
    results = runner.run()
    return results, runner.output_dir


def test_arm_label_describes_a_mined_arm():
    assert rp._arm_label("gold") == "all gold edges"
    label = rp._arm_label("mined:llama-3.3-70b-instruct:all_pairs")
    assert "llama-3.3-70b-instruct" in label and "all pairs" in label
    # A name that does not parse must not raise from inside the report.
    assert rp._arm_label("something-else") == "something-else"


def test_report_renders_every_arm(dataset, mined_specs, mock_llm):
    arms = ("gold", "gold_valid", MINED_ARM)
    results, out_dir = _mined_results(dataset, "rep1", mined_specs, arms)
    tex = open(rp.write_report(results, out_dir)).read()

    assert tex.count(r"\begin{document}") == 1
    assert tex.count(r"\end{document}") == 1
    # One scores table and one ladder table per arm, mined included.
    for arm in arms:
        assert r"\label{tab:scores-" + rp._key(arm) + "}" in tex
    assert r"\label{tab:mining-pr}" in tex
    assert r"\label{tab:policy-lcs}" in tex
    assert r"\label{tab:ladder-by-arm}" in tex
    # A raw colon must never reach a LaTeX label.
    for label in re.findall(r"\\label\{([^}]*)\}", tex):
        assert ":" not in label.split(":", 1)[1] if ":" in label else True


def test_report_arm_delta_generalizes_to_mined_arms(dataset, mined_specs, mock_llm):
    results, out_dir = _mined_results(
        dataset, "rep2", mined_specs, ("gold", "gold_valid", MINED_ARM)
    )
    tex = open(rp.write_report(results, out_dir)).read()
    # The original gold-vs-gold_valid delta survives...
    assert r"\label{tab:arm-delta-gold-gold-valid}" in tex
    # ...and the mined arm gets its own, with mined-specific prose.
    assert r"\label{tab:arm-delta-gold-" + rp._key(MINED_ARM) + "}" in tex
    assert "mined-versus-labelled comparison" in tex


def test_report_states_the_policy_asymmetry(dataset, mined_specs, mock_llm):
    """The interpretive caveat must be in the report, not just in the plan."""
    results, out_dir = _mined_results(dataset, "rep3", mined_specs, ("gold", MINED_ARM))
    tex = open(rp.write_report(results, out_dir)).read()
    assert "do not have the same reach" in tex
    assert "unreachable" in tex
    assert "do not build equally dense networks" in tex


def test_report_survives_a_mined_only_run(dataset, mined_specs, mock_llm):
    """No gold arm: sections that describe gold must degrade, not crash."""
    results, out_dir = _mined_results(dataset, "rep4", mined_specs, (MINED_ARM,))
    tex = open(rp.write_report(results, out_dir)).read()
    assert tex.count(r"\end{document}") == 1
    assert r"\label{tab:mining-pr}" in tex


def test_report_survives_when_every_mined_cell_failed(
    dataset, mined_specs, monkeypatch, mock_llm
):
    async def _boom(*a, **k):
        raise RuntimeError("no endpoint")

    monkeypatch.setattr(rn, "abuild_mined_result", _boom)
    results, out_dir = _mined_results(dataset, "rep5", mined_specs, ("gold", MINED_ARM))
    tex = open(rp.write_report(results, out_dir)).read()
    assert tex.count(r"\end{document}") == 1
    assert "cells failed" in tex


def test_new_tables_have_matching_column_counts(dataset, mined_specs, mock_llm):
    """Guards the hardcoded `tabular` specs against a column being added."""
    results, out_dir = _mined_results(
        dataset, "rep6", mined_specs, ("gold", "gold_valid", MINED_ARM)
    )
    tex = open(rp.write_report(results, out_dir)).read()
    checked = 0
    for m in re.finditer(
        r"\\label\{(tab:(?:mining|policy|ladder-by|arm-delta|scores)[^}]*)\}"
        r".*?\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}",
        tex,
        re.S,
    ):
        label, spec, body = m.group(1), m.group(2), m.group(3)
        ncols = len(re.findall(r"[lrc]|p\{[^}]*\}", spec))
        header = next(
            line for line in body.split("\n")
            if line.strip() and not line.strip().startswith("\\")
        )
        assert header.count("&") + 1 == ncols, f"{label}: {spec}"
        checked += 1
    assert checked >= 6


def test_baseline_repro_section_reports_a_clean_match(dataset, mined_specs, mock_llm):
    results, out_dir = _mined_results(dataset, "rep7", mined_specs, ("gold",))
    text = rp._baseline_repro_section(results, results)  # identical by construction
    assert "reproduce to within" in text


def test_baseline_repro_section_flags_a_drift(dataset, mined_specs, mock_llm):
    results, out_dir = _mined_results(dataset, "rep8", mined_specs, ("gold",))
    drifted = copy.deepcopy(results)
    drifted["records"][0]["lcs"]["mean_marginal"] += 0.01
    text = rp._baseline_repro_section(results, drifted)
    assert "differ by more than the tolerance" in text


def test_baseline_repro_section_handles_disjoint_runs(dataset, mined_specs, mock_llm):
    results, _ = _mined_results(dataset, "rep9", mined_specs, ("gold",))
    other = {"records": [{"item_id": "nope", "arm": "gold", "lcs": {"reified": 0.5}}]}
    assert "no reproduction check" in rp._baseline_repro_section(results, other)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def test_cli_expands_models_and_policies_into_arms():
    parser = cli.build_parser()
    args = parser.parse_args(
        ["--arms", "gold", "--models", "m1,m2", "--pair-policies", "windowed,all_pairs"]
    )
    arms = cli._expand_arms(args, parser)
    assert arms == [
        "gold",
        "mined:m1:windowed", "mined:m1:all_pairs",
        "mined:m2:windowed", "mined:m2:all_pairs",
    ]


def test_cli_defaults_to_windowed_when_only_models_given():
    parser = cli.build_parser()
    args = parser.parse_args(["--models", "m1"])
    assert cli._expand_arms(args, parser) == [
        "gold", "gold_valid", "mined:m1:windowed"
    ]


def test_cli_dedupes_arms():
    parser = cli.build_parser()
    args = parser.parse_args(
        ["--arms", "gold,mined:m1:windowed", "--models", "m1",
         "--pair-policies", "windowed"]
    )
    assert cli._expand_arms(args, parser) == ["gold", "mined:m1:windowed"]


def test_cli_rejects_pair_policies_without_models():
    parser = cli.build_parser()
    args = parser.parse_args(["--pair-policies", "windowed"])
    with pytest.raises(SystemExit):
        cli._expand_arms(args, parser)


def test_cli_defaults_do_not_target_the_gold_baseline():
    """The published gold run must not be a default output directory."""
    args = cli.build_parser().parse_args([])
    assert args.out_dir != "results/locobench_claude_5_fixed_lcs"
    assert args.data_dir == "data/locobench-claude-5-test"
    assert os.path.isdir(args.data_dir)  # and it actually exists


def test_cli_rejects_mined_arms_without_a_rits_key(monkeypatch, tmp_path):
    # The CLI loads the project-root .env before validating, so on a developer
    # machine that file would put the key straight back and make this
    # precondition unreachable. Opt out of loading for this test: the point is
    # the guard, not the credential source.
    monkeypatch.setenv("FACT_REASONER_NO_DOTENV", "1")
    monkeypatch.delenv("RITS_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        cli.main([
            "--models", "llama-3.3-70b-instruct",
            "--merlin-path", str(tmp_path),
            "--out-dir", str(tmp_path / "o"),
        ])


def test_cli_rejects_an_unknown_model(monkeypatch):
    with pytest.raises(SystemExit):
        cli.main(["--models", "gpt-oss-120b", "--estimate-only"])


def test_cli_estimate_only_exits_zero_and_prints_counts(capsys):
    rc = cli.main([
        "--models", "llama-3.3-70b-instruct",
        "--pair-policies", "windowed,all_pairs",
        "--estimate-only",
    ])
    out = capsys.readouterr().out
    assert rc == 0
    # Pair counts scale with the dataset, so assert the RELATION between the policies
    # rather than two constants: an earlier version pinned "264 pairs"/"2400 pairs" and
    # broke the moment the corpus grew from 10 to 70 items, which said nothing about the
    # estimator. `all_pairs` visits every ordered pair and `windowed` only a forward
    # window, so all_pairs must dominate by a wide margin.
    counts = [int(m) for m in re.findall(r"(\d+) pairs", out)]
    assert len(counts) == 2, out
    windowed, all_pairs = counts
    assert 0 < windowed < all_pairs
    assert all_pairs > 5 * windowed


def test_cli_estimate_only_is_silent_about_llm_for_gold_arms(capsys):
    rc = cli.main(["--estimate-only"])
    assert rc == 0
    assert "make no LLM calls" in capsys.readouterr().out


def test_findings_flag_saturated_type_confidence(dataset, mined_specs, mock_llm):
    """A pinned P(tau|a_i,a_j) means the factor weight is only the strength."""
    results, out_dir = _mined_results(dataset, "sat1", mined_specs, ("gold", MINED_ARM))
    for rec in results["records"]:
        for rel in rec.get("relations") or []:
            rel["type_confidence"] = 1.0
        rec["relation_source"] = "mined" if rec["arm"] == MINED_ARM else "gold"
    text = rp._findings_section(results)
    assert "type confidence is saturated" in text


def test_findings_omit_the_saturation_bullet_when_confidence_varies(
    dataset, mined_specs, mock_llm
):
    results, out_dir = _mined_results(dataset, "sat2", mined_specs, ("gold", MINED_ARM))
    for rec in results["records"]:
        for i, rel in enumerate(rec.get("relations") or []):
            rel["type_confidence"] = 0.4 + 0.01 * i
    assert "type confidence is saturated" not in rp._findings_section(results)


def test_gate_admitted_gold_edges_are_inside_the_window():
    """Pins the fact the report's stratification caption now asserts.

    `window_admission == "gate"` does NOT mean out-of-window in this corpus: every
    such edge sits within the radius-4 window, so a miss is the response-anchored
    refinement declining the link, not a candidate-selection shortfall. If a future
    corpus changes that, this fails and the caption must be revisited.
    """
    path = "data/locobench-claude-5-test/items.jsonl"
    if not os.path.exists(path):
        pytest.skip("dataset not present")
    items = [json.loads(line) for line in open(path)]
    gate_edges = [
        rel
        for item in items
        for rel in item.get("relations", [])
        if rel.get("window_admission") == "gate"
    ]
    assert gate_edges, "expected some gate-admitted edges in this corpus"
    for rel in gate_edges:
        # ABSOLUTE distance: `schema.annotate_window_admission` assigns the `gate` verdict
        # on `abs(pos[src] - pos[trg]) <= window`, so a BACKWARD edge inside the window is
        # legitimately gate-admitted. An earlier version asserted `0 < dist <= 4`, which
        # held only because the 2-family corpus happened to contain no backward gate edge;
        # the 14-family one has three (distances -3, -3, -2).
        dist = abs(mg._atom_index(rel["target_id"]) - mg._atom_index(rel["source_id"]))
        assert 0 < dist <= 4, f"{rel['id']}: distance {dist} is outside the window"
    # An earlier version also asserted every gate-admitted edge is a planted error. That
    # held on the 2-family corpus by accident, not by construction: on the 14-family one
    # 33 of 61 are `valid`. `window_admission` records how the candidate selector would
    # SEE the edge, which is independent of whether the edge is a planted error, so the
    # two were never linked. What must hold is that the label is one of the two legal
    # values -- a gate-admitted edge with a junk validity would be a real defect.
    assert {rel["validity"] for rel in gate_edges} <= {"valid", "invalid"}


def test_fingerprint_is_scoped_to_what_each_arm_reads(mined_specs):
    """Changing --gate must not invalidate arms that never read it.

    A coarse run-level fingerprint would discard cached windowed / all_pairs / gold
    cells when a `gated` arm is added, silently re-spending thousands of LLM calls
    on a knob those arms never saw.
    """
    specs = {"m1": lm.ModelSpec(name="m1", model_id="m1", backend="rits")}
    arms = ("gold", "mined:m1:windowed", "mined:m1:gated", "mined:m1:all_pairs")

    def fps(**kw):
        r = rn.GoldEvalRunner(
            data_dir=".", output_dir=".", merlin_path="m",
            arms=arms, model_specs=specs, **kw,
        )
        return {a: r._run_fingerprint(a) for a in arms}

    base, changed = fps(gate="none"), fps(gate="embedding")
    assert changed["mined:m1:gated"] != base["mined:m1:gated"]
    for arm in ("gold", "mined:m1:windowed", "mined:m1:all_pairs"):
        assert changed[arm] == base[arm], f"{arm} invalidated by an unrelated knob"


def test_window_only_invalidates_the_policies_that_use_it(mined_specs):
    specs = {"m1": lm.ModelSpec(name="m1", model_id="m1", backend="rits")}
    arms = ("gold", "mined:m1:windowed", "mined:m1:gated", "mined:m1:all_pairs")

    def fps(**kw):
        r = rn.GoldEvalRunner(
            data_dir=".", output_dir=".", merlin_path="m",
            arms=arms, model_specs=specs, **kw,
        )
        return {a: r._run_fingerprint(a) for a in arms}

    base, changed = fps(window=4), fps(window=6)
    assert changed["mined:m1:windowed"] != base["mined:m1:windowed"]
    assert changed["mined:m1:gated"] != base["mined:m1:gated"]
    # all_pairs ignores the window entirely, and gold reads no mining knob.
    assert changed["mined:m1:all_pairs"] == base["mined:m1:all_pairs"]
    assert changed["gold"] == base["gold"]


def test_gold_fingerprint_ignores_every_mining_knob(mined_specs):
    specs = {"m1": lm.ModelSpec(name="m1", model_id="m1", backend="rits")}

    def fp(**kw):
        return rn.GoldEvalRunner(
            data_dir=".", output_dir=".", merlin_path="m",
            arms=("gold",), model_specs=specs, **kw,
        )._run_fingerprint("gold")

    base = fp()
    for kw in ({"window": 9}, {"gate": "entity"}, {"strength_method": "verbalized"},
               {"nli_method": "simbauq"}, {"strength_samples": 32}):
        assert fp(**kw) == base, f"gold invalidated by {kw}"
    # But a SCORING knob must still invalidate it.
    assert fp(ibound=10) != base
    assert fp(reified_prior=0.7) != base


# ---------------------------------------------------------------------------
# Event loop lifecycle.
# ---------------------------------------------------------------------------


def test_all_cells_share_one_event_loop(dataset, mined_specs, mock_llm):
    """One loop per sweep, not one per cell.

    Mellea caches its async client per event loop in an LRU of capacity 2, so a
    fresh loop per cell means a fresh client per cell and the third cell evicts the
    first without awaiting its close -- whose connection pool is then finalized
    against a dead loop. Sharing one loop is what prevents that.
    """
    seen = []
    runner = _mined_runner(dataset, "loop1", mined_specs, arms=(MINED_ARM,))
    real = runner._run_async

    def spy(coro):
        out = real(coro)
        seen.append(id(runner._loop))
        return out

    runner._run_async = spy
    runner.run()
    assert len(seen) == 2  # two items in the fixture
    assert len(set(seen)) == 1, "a new loop was created per cell"
    # And the loop is released once the sweep returns.
    assert runner._loop is None


def test_run_closes_the_loop_even_when_a_cell_raises(
    dataset, mined_specs, monkeypatch, mock_llm
):
    async def _boom(*a, **k):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr(rn, "abuild_mined_result", _boom)
    runner = _mined_runner(dataset, "loop2", mined_specs, arms=(MINED_ARM,))
    results = runner.run()
    assert all("error" in r for r in results["records"])
    assert runner._loop is None  # released despite every cell failing


def test_close_is_idempotent_and_safe_before_any_run(dataset, mined_specs):
    runner = _mined_runner(dataset, "loop3", mined_specs, arms=("gold",))
    runner.close()  # never ran; must not raise
    runner.close()
    assert runner._loop is None


def test_close_shuts_async_clients_down_on_their_own_loop(
    dataset, mined_specs, mock_llm
):
    """A client must be closed while its loop is alive, else it strands connections."""
    import asyncio

    closed_on = []

    class _Client:
        async def close(self):
            # Recording the loop is the whole point: closing on a different (or
            # already-closed) loop is what strands the connection pool.
            closed_on.append(id(asyncio.get_running_loop()))

    class _Cache:
        def __init__(self):
            self.cache = {1: _Client()}

    runner = _mined_runner(dataset, "loop4", mined_specs, arms=(MINED_ARM,))
    runner.run()  # populates and then releases the loop
    # Re-arm with a fake client on a fresh loop and close again.
    runner._loop = None
    backend = MagicMock()
    backend._client_cache = _Cache()
    runner._backends = {"m1": backend}
    runner._run_async(_noop())
    loop_id = id(runner._loop)
    runner.close()
    assert closed_on == [loop_id], "client was not closed on its own live loop"
    assert not backend._client_cache.cache  # cache cleared


async def _noop():
    return None


def test_runner_is_a_context_manager(dataset, mined_specs, mock_llm):
    with _mined_runner(dataset, "loop5", mined_specs, arms=("gold",)) as runner:
        runner._run_async(_noop())
        assert runner._loop is not None
    assert runner._loop is None


# ---------------------------------------------------------------------------
# The bidirectional arm and the new mining knobs.
# ---------------------------------------------------------------------------


def test_parse_arm_accepts_bidirectional():
    spec = mg.parse_arm("mined:m1:bidirectional")
    assert spec == mg.MinedArm(model="m1", pair_policy="bidirectional")


def test_parse_arm_still_rejects_unknown_policies():
    with pytest.raises(mg.MinedArmError, match="Unknown pair policy"):
        mg.parse_arm("mined:m1:sideways")


def test_fingerprint_defaults_are_backward_compatible(mined_specs):
    """Adding knobs must not invalidate records mined before they existed.

    The knobs enter the hash only when set away from the historical default, so a
    cached arm keeps its fingerprint and `--resume` still serves it. Without this,
    merely adding a parameter would silently re-spend thousands of LLM calls.
    """
    arms = ("gold", MINED_ARM)

    def fp(**kw):
        r = rn.GoldEvalRunner(
            data_dir=".", output_dir=".", merlin_path="m",
            arms=arms, model_specs=mined_specs, **kw,
        )
        return {a: r._run_fingerprint(a) for a in arms}

    base = fp()
    assert fp(sense_menu="full", reconcile="ratchet", discourse=None) == base


def test_fingerprint_changes_for_each_new_knob(mined_specs):
    """The stale-resume trap: a changed strategy must not reuse a cached record."""
    arms = ("gold", MINED_ARM)

    def fp(**kw):
        r = rn.GoldEvalRunner(
            data_dir=".", output_dir=".", merlin_path="m",
            arms=arms, model_specs=mined_specs, **kw,
        )
        return {a: r._run_fingerprint(a) for a in arms}

    base = fp()
    for kw in ({"sense_menu": "gold9"}, {"reconcile": "strict"},
               {"discourse": False}):
        changed = fp(**kw)
        assert changed[MINED_ARM] != base[MINED_ARM], kw
        # A gold arm reads no mining knob, so it must be untouched.
        assert changed["gold"] == base["gold"], kw


def test_max_distance_only_invalidates_the_bidirectional_arm(mined_specs):
    specs = {"m1": lm.ModelSpec(name="m1", model_id="m1", backend="rits")}
    arms = ("gold", "mined:m1:windowed", "mined:m1:bidirectional")

    def fp(**kw):
        r = rn.GoldEvalRunner(
            data_dir=".", output_dir=".", merlin_path="m",
            arms=arms, model_specs=specs, **kw,
        )
        return {a: r._run_fingerprint(a) for a in arms}

    base, changed = fp(max_distance=1), fp(max_distance=2)
    assert changed["mined:m1:bidirectional"] != base["mined:m1:bidirectional"]
    assert changed["mined:m1:windowed"] == base["mined:m1:windowed"]
    assert changed["gold"] == base["gold"]


def test_a_backward_directed_gold_edge_needs_the_reversed_arc():
    """Pins the finding that motivated the bidirectional policy.

    A directed gold relation running backward in atom order is NOT matched by the
    forward arc, and IS matched by the reversed one. So a forward-only candidate
    pool cannot recover it however well the model labels it.
    """
    from fact_reasoner.lcs.relation_miner import MinedRelation

    item = {
        "id": "t-bd",
        "relations": [
            {
                "id": "r0", "source_id": "a3", "target_id": "a1",  # backward
                "level2_sense": "Evidence", "level1_coupling": "entailment",
                "directed": True, "validity": "valid",
                "window_admission": "window",
            }
        ],
        "non_relations": [],
    }

    def rel(src, trg):
        return MinedRelation(
            source_id=src, target_id=trg, level2_sense="Evidence",
            level1_type="entailment", probability=0.9, type_confidence=1.0,
            strength=0.9, directed=True,
        )

    forward_only = mg.compare_to_gold(item, [rel("a1", "a3")])
    assert forward_only["coupling"]["recall"] == 0.0

    reversed_arc = mg.compare_to_gold(item, [rel("a3", "a1")])
    assert reversed_arc["coupling"]["recall"] == 1.0


def test_mined_record_carries_the_new_provenance(dataset, mined_specs, mock_llm):
    arm = "mined:m1:bidirectional"
    runner = _mined_runner(
        dataset, "bidi1", mined_specs, arms=(arm,),
        max_distance=1, sense_menu="gold9", reconcile="strict",
    )
    results = runner.run()
    for rec in results["records"]:
        assert "error" not in rec, rec.get("error")
        assert rec["pair_policy"] == "bidirectional"
        assert rec["max_distance"] == 1
        assert rec["sense_menu"] == "gold9"
        assert rec["reconcile"] == "strict"
        assert rec["discourse"] is False  # off by default for this policy
        assert rec["num_inadmissible_sense"] is not None


def test_cli_exposes_the_new_flags():
    args = cli.build_parser().parse_args(
        ["--max-distance", "2", "--sense-menu", "gold9", "--reconcile", "strict",
         "--no-discourse"]
    )
    assert args.max_distance == 2
    assert args.sense_menu == "gold9"
    assert args.reconcile == "strict"
    assert args.discourse is False


def test_cli_discourse_defaults_to_policy_choice():
    assert cli.build_parser().parse_args([]).discourse is None
    assert cli.build_parser().parse_args(["--discourse"]).discourse is True


def test_cli_rejects_unknown_menu_or_reconcile():
    for flag, value in (("--sense-menu", "gold10"), ("--reconcile", "lenient")):
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args([flag, value])


# ---------------------------------------------------------------------------
# Named mining strategies travel WITH the arm.
# ---------------------------------------------------------------------------


def test_strategy_for_resolves_a_named_variant():
    v2 = mg.strategy_for("v2")
    assert v2["sense_menu"] == "gold9"
    assert v2["prompt_variant"] == "v2"
    assert v2["require_evidence"] is True
    assert mg.strategy_for(None) is None


def test_strategy_for_rejects_an_unknown_variant():
    with pytest.raises(mg.MinedArmError, match="Unknown mining strategy"):
        mg.strategy_for("v99")


def test_two_strategies_do_not_collide_on_one_arm_name():
    """Regression: a global --prompt-variant cannot express a two-strategy sweep.

    Both strategies previously produced the arm `mined:m:bidirectional`, so the
    second overwrote the first's records and the report compared a strategy against
    itself. The variant label is what keeps them distinct.
    """
    plain = mg.format_arm("m", "bidirectional")
    tagged = mg.format_arm("m", "bidirectional", "v2")
    assert plain != tagged
    assert mg.parse_arm(plain).variant is None
    assert mg.parse_arm(tagged).variant == "v2"


def test_fingerprint_follows_the_arms_own_strategy(mined_specs):
    """Two arms in one run must hash differently when their strategies differ.

    Otherwise `--resume` serves a record mined under the other arm's settings --
    which is exactly how a v2 arm came to hold v1 numbers.
    """
    specs = {"m1": lm.ModelSpec(name="m1", model_id="m1", backend="rits")}
    arms = ("mined:m1:bidirectional", "mined:m1:bidirectional:v2")
    r = rn.GoldEvalRunner(
        data_dir=".", output_dir=".", merlin_path="m",
        arms=arms, model_specs=specs, max_distance=1,
    )
    assert r._run_fingerprint(arms[0]) != r._run_fingerprint(arms[1])


def test_arm_strategy_overrides_the_run_level_flags(dataset, mined_specs, mock_llm):
    """The arm's own strategy wins, so one sweep can hold both."""
    arm_v2 = "mined:m1:bidirectional:v2"
    runner = _mined_runner(
        dataset, "strat1", mined_specs, arms=(arm_v2,),
        max_distance=1,
        # Deliberately the v1 defaults at run level; the arm must still be v2.
        sense_menu="full", prompt_variant="v1", require_evidence=False,
    )
    results = runner.run()
    for rec in results["records"]:
        assert rec["strategy"] == "v2"
        assert rec["sense_menu"] == "gold9"
        assert rec["prompt_variant"] == "v2"
        assert rec["require_evidence"] is True


def test_short_arm_keeps_the_variant_visible():
    assert rp._short_arm("mined:llama-3.3-70b-instruct:bidirectional:v2") == (
        "bidirectional:v2"
    )
    assert rp._short_arm("mined:m:windowed") == "windowed"
    assert rp._short_arm("gold") == "gold"
