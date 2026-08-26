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

"""Pins the exact values the coherence paper's five-claim worked pair reports.

The paper (``docs/iclr2027/coherence``, Sec. 9.1) quotes LCS readouts for two
responses over one five-claim set -- three claims true (prior 0.9), two false
(prior 0.1). Its Reproducibility statement promises those numbers are pinned by
regression tests; this is that test.

Everything is offline and exact: the relation graphs come from the fixture
``data/lcs/example-7-coherence-pair.json`` and inference is the 2^5 = 32-world
brute-force oracle, not Merlin.
"""

import json
import os

import pytest

from fact_reasoner.core.base import Atom
from fact_reasoner.factors import build_markov_network
from fact_reasoner.lcs import lcs_scorer as lcs_scorer_mod
from fact_reasoner.lcs.lcs_scorer import LCSScorer
from fact_reasoner.lcs.relation_miner import MinedRelation, MiningResult, RelationMiner

from fact_reasoner.experiments.mock import (
    MAX_BRUTEFORCE_VARS,
    brute_force_marginals,
    brute_force_run_merlin,
)

FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "lcs",
    "example-7-coherence-pair.json",
)


@pytest.fixture(scope="module")
def fixture_data():
    with open(FIXTURE) as f:
        return json.load(f)


def _patch_fake_merlin(monkeypatch):
    """Route the scorer's inference at the exact oracle (n=5, so it is exact)."""
    monkeypatch.setattr(lcs_scorer_mod, "run_merlin", brute_force_run_merlin)
    monkeypatch.setattr(
        lcs_scorer_mod, "DEFAULT_MAX_NETWORK_VARS", MAX_BRUTEFORCE_VARS
    )


def _build(fixture_data, key):
    """Build (result, priors) for response ``key`` straight from the fixture."""
    priors = {a["id"]: float(a["prior"]) for a in fixture_data["atoms"]}
    atoms = {a["id"]: Atom(id=a["id"], text=a["text"]) for a in fixture_data["atoms"]}
    entry = fixture_data["responses"][key]
    mined = [
        MinedRelation(
            source_id=r["source"],
            target_id=r["target"],
            level2_sense=r["sense"],
            level1_type=r["coupling"],
            probability=float(r["p"]),
            type_confidence=1.0,
            strength=float(r["p"]),
        )
        for r in entry["gold_relations"]
    ]
    miner = object.__new__(RelationMiner)  # bypass __init__ (no backend needed)
    miner.prior = 0.5
    fg = miner._build_fact_graph(atoms, mined)
    mn = build_markov_network(fg, use_priors=True, node_priors=priors)
    result = MiningResult(
        atoms=atoms,
        relations=mined,
        fact_graph=fg,
        markov_network=mn,
        coverage={},
        config={"prior": 0.5},
    )
    return result, priors


class TestWorkedPairFixture:
    def test_shape_is_three_true_two_false(self, fixture_data):
        atoms = fixture_data["atoms"]
        assert len(atoms) == 5
        assert sum(1 for a in atoms if a["truth"]) == 3
        assert sum(1 for a in atoms if not a["truth"]) == 2
        # The paper's stated prior convention: 0.9 for true claims, 0.1 for false.
        for a in atoms:
            assert a["prior"] == pytest.approx(0.9 if a["truth"] else 0.1)

    def test_both_responses_share_one_claim_set(self, fixture_data):
        """The whole point of the pair: factuality is held fixed, only wiring moves."""
        ids = {a["id"] for a in fixture_data["atoms"]}
        for key in ("A", "B"):
            touched = set()
            for r in fixture_data["responses"][key]["gold_relations"]:
                touched.add(r["source"])
                touched.add(r["target"])
            assert touched <= ids
        # ... and the arrangements genuinely differ.
        def edges(k):
            return {
                (r["source"], r["target"], r["coupling"])
                for r in fixture_data["responses"][k]["gold_relations"]
            }
        assert edges("A") != edges("B")


class TestWorkedPairValues:
    """The numbers Sec. 9.1 prints. Changing these changes the paper."""

    def test_response_a_is_coherent(self, fixture_data, monkeypatch):
        _patch_fake_merlin(monkeypatch)
        result, priors = _build(fixture_data, "A")
        s = LCSScorer("/fake/merlin").score_all(result, node_priors=priors)
        assert s["mean_marginal"] == pytest.approx(0.5905, abs=1e-3)
        assert s["consistency"] == pytest.approx(0.9625, abs=1e-3)
        assert s["log_z"] == pytest.approx(-0.9524, abs=1e-3)
        assert s["log_partition"] == pytest.approx(0.1935, abs=1e-3)
        # Every claim lands on the correct side of its own prior.
        assert s["num_below_prior"] == 2
        assert s["marginals"]["a2"] == pytest.approx(0.9920, abs=1e-3)  # paper's a3

    def test_response_b_is_incoherent(self, fixture_data, monkeypatch):
        _patch_fake_merlin(monkeypatch)
        result, priors = _build(fixture_data, "B")
        s = LCSScorer("/fake/merlin").score_all(result, node_priors=priors)
        assert s["mean_marginal"] == pytest.approx(0.4938, abs=1e-3)
        assert s["consistency"] == pytest.approx(0.4136, abs=1e-3)
        assert s["log_z"] == pytest.approx(-3.7802, abs=1e-3)
        assert s["num_below_prior"] == 3

    def test_a_scores_above_b_on_both_headline_readouts(
        self, fixture_data, monkeypatch
    ):
        _patch_fake_merlin(monkeypatch)
        scores = {}
        for key in ("A", "B"):
            result, priors = _build(fixture_data, key)
            scores[key] = LCSScorer("/fake/merlin").score_all(
                result, node_priors=priors
            )
        a, b = scores["A"], scores["B"]
        assert a["mean_marginal"] > b["mean_marginal"]
        assert a["consistency"] > b["consistency"]
        # log Z is comparable across responses (raw, unnormalized) and agrees.
        assert a["log_z"] > b["log_z"]

    def test_true_claim_is_dragged_below_its_prior_in_b(
        self, fixture_data, monkeypatch
    ):
        """The per-claim diagnostic the paper leans on: a TRUE claim suppressed.

        In B the true a3 (fixture id a2) is the target of a contradiction from the
        chain and of an entailment from a FALSE premise, so its marginal falls far
        below its own 0.9 prior -- which no aggregate score would reveal.
        """
        _patch_fake_merlin(monkeypatch)
        result, priors = _build(fixture_data, "B")
        s = LCSScorer("/fake/merlin").score_all(result, node_priors=priors)
        q = s["marginals"]["a2"]  # the paper's a3
        assert q == pytest.approx(0.5126, abs=1e-3)
        assert q < priors["a2"]

    def test_log_partition_is_within_response_only(self, fixture_data, monkeypatch):
        """LCS_lp must NOT be read across responses -- its references differ.

        Zmax/Zmin are recomputed per response, so the normalized value inverts the
        pair. The paper says so explicitly; this pins the fact so nobody "fixes"
        the inversion by quietly changing the readout.
        """
        _patch_fake_merlin(monkeypatch)
        out = {}
        for key in ("A", "B"):
            result, priors = _build(fixture_data, key)
            out[key] = LCSScorer("/fake/merlin").score_all(result, node_priors=priors)
        # Different skeletons => different ceilings, hence incomparable.
        assert out["A"]["log_z_max"] != pytest.approx(out["B"]["log_z_max"], abs=1e-6)
        assert out["A"]["log_partition"] < out["B"]["log_partition"]


class TestWorkedPairOracleAgreement:
    """The two computation paths must agree; the paper quotes values from both."""

    @pytest.mark.parametrize("key", ["A", "B"])
    def test_scorer_matches_direct_enumeration(self, fixture_data, monkeypatch, key):
        _patch_fake_merlin(monkeypatch)
        result, priors = _build(fixture_data, key)
        s = LCSScorer("/fake/merlin").score_all(result, node_priors=priors)

        marginals, log_z, _log_max = brute_force_marginals(
            result.markov_network, priors
        )
        mm = sum(marginals.values()) / len(marginals)
        assert s["mean_marginal"] == pytest.approx(mm, abs=1e-9)
        assert s["log_z"] == pytest.approx(log_z, abs=1e-9)
        for aid, q in marginals.items():
            assert s["marginals"][aid] == pytest.approx(q, abs=1e-9)

    @pytest.mark.parametrize("key", ["A", "B"])
    def test_fixture_records_what_the_code_computes(
        self, fixture_data, monkeypatch, key
    ):
        """The fixture's `expected` block must not drift from the code."""
        _patch_fake_merlin(monkeypatch)
        result, priors = _build(fixture_data, key)
        s = LCSScorer("/fake/merlin").score_all(result, node_priors=priors)
        for field, want in fixture_data["responses"][key]["expected"].items():
            assert s[field] == pytest.approx(float(want), abs=5e-4), field
