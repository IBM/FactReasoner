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

import pytest

from fact_reasoner.experiments.mock import brute_force_run_merlin
from fact_reasoner.lcs.lcs_scorer import LCSScorer
from fact_reasoner.locoeval import gold_graph as gg
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
    """Route the scorer's inference through the exact brute-force oracle."""
    monkeypatch.setattr(
        "fact_reasoner.lcs.lcs_scorer.run_merlin", brute_force_run_merlin
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
