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

"""Offline unit tests for the LCS relation miner (no LLM, no Merlin).

These cover the deterministic parts of the coherence pipeline:
  * the Level-2 -> Level-1 compile map (deep-dive Table 2);
  * the with-priors pairwise factor tables (deep-dive Table 1);
  * the candidate-pair selection policies;
  * the FactGraph/MarkovNetwork construction, validated against the exact
    AeroParts numbers from the deep-dive (LCS 0.587 -> 0.620) using a brute-force
    2^n oracle instead of Merlin;
  * MinedRelation / MiningResult round-trip and the LCSScorer readout (with a
    monkeypatched Merlin helper).
"""

import itertools
import math
import re

import pytest

from fact_reasoner.core.base import Atom
from fact_reasoner.fact_graph import Edge, FactGraph, Node
from fact_reasoner.factors import (
    build_markov_network,
    edge_factor_values,
    pairwise_prior,
)
from fact_reasoner.lcs import candidate_pairs as cp
from fact_reasoner.lcs.taxonomy import (
    LEVEL1_CONECESSITY,
    LEVEL1_CONTRADICTION,
    LEVEL1_ENTAILMENT,
    LEVEL1_EQUIVALENCE,
    LEVEL1_EXCLUSIVE,
    LEVEL1_NONE,
    Level2Sense,
    compile_sense,
    coupling_from_string,
)
from fact_reasoner.lcs.relation_miner import (
    STRENGTH_METHODS,
    MinedRelation,
    MiningResult,
    RelationMiner,
)
from fact_reasoner.lcs import lcs_scorer as lcs_scorer_mod
from fact_reasoner.lcs.lcs_scorer import LCSScorer
from fact_reasoner.lcs.strength import (
    IdentityCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    affirm_fraction,
    surrogate_probability_from_logprobs,
)

# The brute-force MRF oracle and synthetic-logprobs helper are shared with the
# experiment harness's offline mode; import the canonical copies from there.
from fact_reasoner.experiments.mock import (
    brute_force_marginals as _brute_force_marginals,
    brute_force_run_merlin as _brute_force_run_merlin,
    yesno_logprob_meta as _yesno_logprob_meta,
)


# ---------------------------------------------------------------------------
# AeroParts fixture (deep-dive Section 5 / Tables 3-5).
# ---------------------------------------------------------------------------

# (source, target, level1_type, p). Contradictions listed last.
AEROPARTS_BASE = [
    ("a1", "a2", "entailment", 0.90),
    ("a2", "a3", "entailment", 0.85),
    ("a3", "a4", "entailment", 0.70),
    ("a4", "a5", "entailment", 0.80),
    ("a5", "a6", "entailment", 0.75),
    ("a4", "a7", "entailment", 0.65),
    ("a9", "a4", "entailment", 0.72),
    ("a6", "a8", "equivalence", 0.88),
    ("a1", "a14", "entailment", 0.78),
    ("a14", "a15", "entailment", 0.70),
    ("a5", "a16", "entailment", 0.60),
    ("a13", "a12", "entailment", 0.85),
    ("a7", "a10", "contradiction", 0.93),  # unresolved casualty conflict
    ("a11", "a12", "contradiction", 0.80),  # concession, NOT yet discounted (base row)
]

# The "concession resolved" variant: a11 != a12 discounted 0.80 -> 0.55 (Eq. 2).
AEROPARTS_CONCESSION = [
    (s, t, ty, (0.55 if (s, t) == ("a11", "a12") else p))
    for (s, t, ty, p) in AEROPARTS_BASE
]

# The REVISED 5-coupling AeroParts (Level-1 3->5). The casualty (a7/a10) and blame
# (a11/a12) conflicts are EXHAUSTIVE alternatives, so they are `exclusive`, not
# `contradiction` (revised deep-dive Section 5 / Tables 3-5): base LCS 0.607.
AEROPARTS_BASE_5 = [
    (s, t, ("exclusive" if (s, t) in (("a7", "a10"), ("a11", "a12")) else ty), p)
    for (s, t, ty, p) in AEROPARTS_BASE
]

AEROPARTS_CONCESSION_5 = [
    (s, t, ty, (0.55 if (s, t) == ("a11", "a12") else p))
    for (s, t, ty, p) in AEROPARTS_BASE_5
]

AEROPARTS_IDS = [f"a{i}" for i in range(1, 17)]


def _aeroparts_graph(relations, prior=0.5):
    fg = FactGraph()
    for a in AEROPARTS_IDS:
        fg.add_node(Node(id=a, type="atom", probability=prior))
    for s, t, ty, p in relations:
        fg.add_edge(Edge(source=s, target=t, type=ty, probability=p, link="atom_atom"))
    return fg


def _aeroparts_lcs(relations, prior=0.5):
    fg = _aeroparts_graph(relations, prior)
    priors = {a: prior for a in AEROPARTS_IDS}
    mn = build_markov_network(fg, use_priors=True, node_priors=priors)
    marginals, log_z, _log_max = _brute_force_marginals(mn, priors)
    lcs = sum(marginals.values()) / len(marginals)
    return lcs, log_z, marginals


# ---------------------------------------------------------------------------
# Taxonomy (compile map).
# ---------------------------------------------------------------------------


class TestTaxonomy:
    def test_sense_parsing_is_tolerant(self):
        assert Level2Sense.from_string("Cause-Effect") is Level2Sense.CAUSE_EFFECT
        assert Level2Sense.from_string("cause_effect") is Level2Sense.CAUSE_EFFECT
        assert Level2Sense.from_string("cause effect") is Level2Sense.CAUSE_EFFECT
        assert Level2Sense.from_string("nonsense") is Level2Sense.NONE
        assert Level2Sense.from_string("") is Level2Sense.NONE

    @pytest.mark.parametrize(
        "sense,expected_level1",
        [
            (Level2Sense.CAUSE_EFFECT, LEVEL1_ENTAILMENT),
            (Level2Sense.EFFECT_CAUSE, LEVEL1_ENTAILMENT),
            (Level2Sense.EVIDENCE, LEVEL1_ENTAILMENT),
            (Level2Sense.CONDITION, LEVEL1_ENTAILMENT),
            (Level2Sense.INSTANTIATION, LEVEL1_ENTAILMENT),
            (Level2Sense.RESTATEMENT, LEVEL1_EQUIVALENCE),
            (Level2Sense.CONTRAST, LEVEL1_CONTRADICTION),
            (Level2Sense.CONCESSION, LEVEL1_CONTRADICTION),
            (Level2Sense.ALTERNATIVE, LEVEL1_EXCLUSIVE),
            (Level2Sense.DISJUNCTION, LEVEL1_CONECESSITY),
            (Level2Sense.PRECEDENCE, LEVEL1_NONE),
            (Level2Sense.SUCCESSION, LEVEL1_NONE),
            (Level2Sense.NONE, LEVEL1_NONE),
        ],
    )
    def test_compile_map_matches_table2(self, sense, expected_level1):
        level1, _strength, _spec = compile_sense(sense, 0.7)
        assert level1 == expected_level1

    def test_restatement_has_strength_prior(self):
        # Restatement starts near 0.90 when no estimate is supplied.
        level1, strength, spec = compile_sense(Level2Sense.RESTATEMENT)
        assert level1 == LEVEL1_EQUIVALENCE
        assert strength == pytest.approx(0.90)
        assert spec.directed is False

    def test_concession_is_flagged(self):
        _l, _s, spec = compile_sense(Level2Sense.CONCESSION, 0.8)
        assert spec.is_concession is True

    def test_ordering_only_senses_produce_no_edge(self):
        level1, strength, spec = compile_sense(Level2Sense.PRECEDENCE, 0.9)
        assert level1 == LEVEL1_NONE
        assert strength is None
        assert spec.ordering_only is True

    def test_alternative_and_disjunction_are_symmetric(self):
        _l, _s, spec_alt = compile_sense(Level2Sense.ALTERNATIVE, 0.9)
        _l2, _s2, spec_dis = compile_sense(Level2Sense.DISJUNCTION, 0.9)
        assert spec_alt.directed is False
        assert spec_dis.directed is False

    def test_coupling_from_string(self):
        assert coupling_from_string("[entailment]") == LEVEL1_ENTAILMENT
        assert coupling_from_string("contradiction") == LEVEL1_CONTRADICTION
        assert coupling_from_string("equivalence") == LEVEL1_EQUIVALENCE
        assert coupling_from_string("[coupling=exclusive]") == LEVEL1_EXCLUSIVE
        assert coupling_from_string("exactly one") == LEVEL1_EXCLUSIVE
        assert coupling_from_string("co_necessity") == LEVEL1_CONECESSITY
        assert coupling_from_string("at least one") == LEVEL1_CONECESSITY
        assert coupling_from_string("disjunction") == LEVEL1_CONECESSITY
        assert coupling_from_string("neutral") == LEVEL1_NONE
        assert coupling_from_string("independent") == LEVEL1_NONE
        assert coupling_from_string("") == LEVEL1_NONE


# ---------------------------------------------------------------------------
# Factor tables (deep-dive Table 1, with-priors).
# ---------------------------------------------------------------------------


class _E:
    def __init__(self, type, link, probability):
        self.type = type
        self.link = link
        self.probability = probability


class TestFactorTables:
    def test_entailment_with_priors(self):
        # [1-pi_s, pi_s, 1-p, p] with pi_s = 0.5 for atom_atom.
        vals = edge_factor_values(_E("entailment", "atom_atom", 0.7), use_priors=True)
        assert vals == pytest.approx([0.5, 0.5, 0.3, 0.7])

    def test_contradiction_with_priors(self):
        # [1-pi_s, pi_s, p, 1-p]
        vals = edge_factor_values(
            _E("contradiction", "atom_atom", 0.93), use_priors=True
        )
        assert vals == pytest.approx([0.5, 0.5, 0.93, 0.07])

    def test_equivalence(self):
        # [p, 1-p, 1-p, p] (symmetric; priors-independent).
        vals = edge_factor_values(_E("equivalence", "atom_atom", 0.88), use_priors=True)
        assert vals == pytest.approx([0.88, 0.12, 0.12, 0.88])

    def test_exclusive(self):
        # exactly-one: [1-p, p, p, 1-p] (penalizes (0,0) and (1,1)); same in both
        # variants. Revised deep-dive Table 1.
        for up in (True, False):
            vals = edge_factor_values(_E("exclusive", "atom_atom", 0.93), use_priors=up)
            assert vals == pytest.approx([0.07, 0.93, 0.93, 0.07])

    def test_co_necessity_with_priors(self):
        # at-least-one: [1-p, pi_s, pi_s, p] (penalizes only (0,0)).
        vals = edge_factor_values(_E("co_necessity", "atom_atom", 0.9), use_priors=True)
        assert vals == pytest.approx([0.1, 0.5, 0.5, 0.9])

    def test_co_necessity_no_priors(self):
        vals = edge_factor_values(_E("co_necessity", "atom_atom", 0.9), use_priors=False)
        assert vals == pytest.approx([0.1, 0.9, 0.9, 0.9])

    def test_no_priors_entailment(self):
        vals = edge_factor_values(_E("entailment", "atom_atom", 0.7), use_priors=False)
        assert vals == pytest.approx([0.7, 0.7, 0.3, 0.7])

    def test_pairwise_prior(self):
        assert pairwise_prior("atom_atom") == 0.5
        assert pairwise_prior("context_atom") == 0.5
        assert pairwise_prior("context_context") == 0.9
        with pytest.raises(ValueError):
            pairwise_prior("bogus")


# ---------------------------------------------------------------------------
# Candidate-pair policies.
# ---------------------------------------------------------------------------


def _atoms(texts):
    return {f"a{i}": Atom(id=f"a{i}", text=t) for i, t in enumerate(texts)}


def _resp(atoms):
    """A synthetic response = the atom texts joined (for response-required select)."""
    return " ".join(a.text for a in atoms.values())


class TestCandidatePairs:
    def test_all_pairs_is_all_ordered(self):
        atoms = _atoms(["x", "y", "z"])
        pairs, cov = cp.select(atoms, response=_resp(atoms), policy="all_pairs")
        assert len(pairs) == 3 * 2  # n(n-1) ordered
        assert cov["pairs_pruned"] == 0
        assert cov["discourse_anchored"] is False  # all_pairs takes every pair

    def test_windowed_respects_radius(self):
        # Distinct content per atom (no shared-entity promotion) so the window
        # bookkeeping is clean; anchoring may promote/demote by sentence adjacency.
        atoms = _atoms([
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
        ])
        pairs, cov = cp.select(
            atoms, response=_resp(atoms), policy="windowed", window=2,
        )
        # forward pairs only (source before target).
        for s, t in pairs:
            assert int(t[1:]) > int(s[1:])
        # The window universe is |j-i| in [1,2]; selected pairs are that window
        # minus demotions plus discourse promotions.
        assert cov["num_window_pairs"] == sum(
            1 for i in range(6) for j in range(i + 1, min(i + 3, 6))
        )
        assert cov["forward_pairs_possible"] == 6 * 5 // 2
        assert cov["discourse_anchored"] is True
        assert (
            cov["pairs_selected"]
            == cov["num_window_pairs"] - cov["num_demoted"] + cov["num_promoted"]
        )

    def test_gated_adds_callbacks(self):
        # a0 and a5 share the salient token "reactor"; window=1 excludes them,
        # the entity gate should re-admit the long-range pair.
        atoms = _atoms(
            [
                "the reactor overheated badly",
                "a manager filed a report",
                "the weather was cold",
                "lunch was served late",
                "the meeting adjourned early",
                "the reactor was later inspected",
            ]
        )
        pairs, cov = cp.select(
            atoms, response=_resp(atoms), policy="gated", window=1,
            gate="entity", gate_threshold=0.05,
        )
        assert cov["num_callback_pairs"] >= 1
        assert ("a0", "a5") in pairs

    def test_unknown_policy_raises(self):
        atoms = _atoms(["x"])
        with pytest.raises(ValueError):
            cp.select(atoms, response=_resp(atoms), policy="bogus")

    def test_response_is_required(self):
        """Selection is always response-anchored: missing/empty response raises."""
        atoms = _atoms([f"s{i}" for i in range(4)])
        with pytest.raises(TypeError):
            cp.select(atoms, policy="windowed", window=2)  # no response kwarg
        with pytest.raises(ValueError):
            cp.select(atoms, response="   ", policy="windowed", window=2)

    def test_response_promotes_long_range_callback(self):
        """A response linking a far-apart entity promotes the out-of-window pair."""
        texts = [
            "The reactor overheated badly on Monday.",
            "A manager filed an unrelated expense report.",
            "The weather was cold that week.",
            "Lunch was served late in the cafeteria.",
            "The quarterly meeting adjourned early.",
            "The reactor was later inspected for overheating damage.",
        ]
        atoms = _atoms(texts)
        response = " ".join(texts)
        # window=1 excludes (a0,a5); the shared "reactor/overheat" content makes the
        # response relate them, so discourse anchoring promotes the callback.
        pairs, cov = cp.select(
            atoms, policy="windowed", window=1, response=response,
            discourse_gate_threshold=0.05,
        )
        assert cov["discourse_anchored"] is True
        assert cov["num_promoted"] >= 1
        assert ("a0", "a5") in pairs

    def test_response_demotes_unrelated_in_window_pair(self):
        """An in-window pair the response does not relate is demoted (dropped)."""
        texts = [
            "The reactor overheated badly.",
            "Separately, the cafeteria introduced a new dessert menu.",
            "The reactor was inspected for overheating.",
        ]
        atoms = _atoms(texts)
        response = " ".join(texts)
        pairs, cov = cp.select(
            atoms, policy="windowed", window=2, response=response,
            discourse_gate_threshold=0.05, discourse_sentence_span=0,
        )
        # The dessert atom (a1) shares no content with the reactor atoms, so the
        # (a0,a1) and (a1,a2) in-window pairs are demoted; the reactor callback
        # (a0,a2) survives on shared content. The raw window over 3 atoms (w=2)
        # is {a0a1, a0a2, a1a2} = 3 pairs; demotion drops it below that.
        assert cov["num_demoted"] >= 1
        assert ("a0", "a2") in pairs
        assert cov["num_window_pairs"] == 3
        assert len(pairs) < cov["num_window_pairs"]


# ---------------------------------------------------------------------------
# Network construction validated against the exact AeroParts numbers.
# ---------------------------------------------------------------------------


class TestAeroPartsBehaviour:
    def test_base_lcs_matches_deepdive(self):
        lcs, log_z, _ = _aeroparts_lcs(AEROPARTS_BASE)
        # Deep-dive Table 4: base LCS 0.587, log Z -9.75.
        assert lcs == pytest.approx(0.587, abs=1e-3)
        assert log_z == pytest.approx(-9.75, abs=0.05)

    def test_coherent_rewrite_raises_lcs(self):
        coherent = [r for r in AEROPARTS_BASE if r[2] != "contradiction"]
        lcs, log_z, _ = _aeroparts_lcs(coherent)
        # Deep-dive Table 4/5: coherent rewrite LCS 0.620, log Z -8.25.
        assert lcs == pytest.approx(0.620, abs=1e-3)
        assert log_z == pytest.approx(-8.25, abs=0.05)

    def test_concession_discount_raises_lcs(self):
        # Discounting the resolved concession (0.80 -> 0.55) lifts the LCS from
        # the base 0.587 to 0.601 (deep-dive Table 4 concession row).
        base_lcs, _z, _m = _aeroparts_lcs(AEROPARTS_BASE)
        conc_lcs, _z2, _m2 = _aeroparts_lcs(AEROPARTS_CONCESSION)
        assert conc_lcs == pytest.approx(0.601, abs=1e-3)
        assert conc_lcs > base_lcs

    def test_lcs_is_monotone_in_contradictions(self):
        base_lcs, base_z, _ = _aeroparts_lcs(AEROPARTS_BASE)
        coherent = [r for r in AEROPARTS_BASE if r[2] != "contradiction"]
        coh_lcs, coh_z, _ = _aeroparts_lcs(coherent)
        # Removing contradictions must not decrease the LCS (R3 monotonicity).
        assert coh_lcs > base_lcs
        assert coh_z > base_z

    def test_contradiction_drags_endpoint_below_prior(self):
        # a10 (loser of the unresolved a7 != a10 contradiction) collapses toward
        # 0.5 and drops below its 0.5 prior in the base; recovers when removed.
        _lcs, _z, marg_base = _aeroparts_lcs(AEROPARTS_BASE)
        coherent = [r for r in AEROPARTS_BASE if r[2] != "contradiction"]
        _lcs2, _z2, marg_coh = _aeroparts_lcs(coherent)
        assert marg_base["a10"] < marg_coh["a10"]


class TestAeroPartsBehaviour5Couplings:
    """The revised 5-coupling model: a7/a10 & a11/a12 as EXCLUSIVE (deep-dive 0.607)."""

    _CONFLICT = ("contradiction", "exclusive")

    def test_base_lcs_matches_revised_deepdive(self):
        lcs, log_z, _ = _aeroparts_lcs(AEROPARTS_BASE_5)
        # Revised deep-dive Tables 3-4: base LCS 0.607, log Z -9.64.
        assert lcs == pytest.approx(0.607, abs=1e-3)
        assert log_z == pytest.approx(-9.64, abs=0.05)

    def test_coherent_rewrite_raises_lcs(self):
        coherent = [r for r in AEROPARTS_BASE_5 if r[2] not in self._CONFLICT]
        lcs, log_z, _ = _aeroparts_lcs(coherent)
        # Revised deep-dive: coherent rewrite LCS 0.620, log Z -8.25 (unchanged).
        assert lcs == pytest.approx(0.620, abs=1e-3)
        assert log_z == pytest.approx(-8.25, abs=0.05)

    def test_concession_discount_raises_lcs(self):
        # Discounting the resolved concession exclusive (0.80 -> 0.55) lifts the
        # base 0.607 to 0.612 (revised deep-dive concession row).
        base_lcs, _z, _m = _aeroparts_lcs(AEROPARTS_BASE_5)
        conc_lcs, _z2, _m2 = _aeroparts_lcs(AEROPARTS_CONCESSION_5)
        assert conc_lcs == pytest.approx(0.612, abs=1e-3)
        assert conc_lcs > base_lcs

    def test_lcs_is_monotone_in_conflicts(self):
        base_lcs, base_z, _ = _aeroparts_lcs(AEROPARTS_BASE_5)
        coherent = [r for r in AEROPARTS_BASE_5 if r[2] not in self._CONFLICT]
        coh_lcs, coh_z, _ = _aeroparts_lcs(coherent)
        assert coh_lcs > base_lcs
        assert coh_z > base_z

    def test_exclusive_lifts_loser_above_contradiction(self):
        # Exclusive forbids "neither", so the casualty loser a10 sits higher under
        # exclusive (~0.40) than under a bare contradiction (~0.24): rejecting a10
        # no longer drives it near zero.
        _l, _z, marg_contra = _aeroparts_lcs(AEROPARTS_BASE)
        _l2, _z2, marg_excl = _aeroparts_lcs(AEROPARTS_BASE_5)
        assert marg_excl["a10"] > marg_contra["a10"]
        assert marg_excl["a10"] == pytest.approx(0.404, abs=1e-2)


# ---------------------------------------------------------------------------
# MinedRelation / MiningResult round-trip + LCSScorer readout.
# ---------------------------------------------------------------------------


def _mined_from_tuples(relations):
    out = []
    for s, t, ty, p in relations:
        out.append(
            MinedRelation(
                source_id=s,
                target_id=t,
                level2_sense="Cause-Effect" if ty == "entailment" else ty,
                level1_type=ty,
                probability=p,
                type_confidence=1.0,
                strength=p,
            )
        )
    return out


def _aeroparts_result(relations, prior=0.5):
    atoms = {a: Atom(id=a, text=f"atom {a}") for a in AEROPARTS_IDS}
    mined = _mined_from_tuples(relations)
    miner = object.__new__(RelationMiner)  # bypass __init__ (no backend needed)
    miner.prior = prior
    fg = miner._build_fact_graph(atoms, mined)
    priors = {a: prior for a in AEROPARTS_IDS}
    mn = build_markov_network(fg, use_priors=True, node_priors=priors)
    return MiningResult(
        atoms=atoms,
        relations=mined,
        fact_graph=fg,
        markov_network=mn,
        coverage={"policy": "all_pairs", "pairs_scored": len(relations)},
        config={"prior": prior},
    )


class TestMiningResult:
    def test_json_round_trip(self):
        result = _aeroparts_result(AEROPARTS_BASE)
        data = result.to_json()
        assert set(data) >= {"atoms", "relations", "fact_graph", "coverage", "config"}
        assert len(data["relations"]) == len(AEROPARTS_BASE)
        # FactGraph serializes to its own JSON form (nodes + edges).
        assert len(data["fact_graph"]["edges"]) == len(AEROPARTS_BASE)
        assert len(data["fact_graph"]["nodes"]) == len(AEROPARTS_IDS)

    def test_describe_runs(self):
        result = _aeroparts_result(AEROPARTS_BASE)
        text = result.describe()
        assert "Relations" in text and "Coverage" in text

    def test_build_fact_graph_uses_atom_atom_link(self):
        result = _aeroparts_result(AEROPARTS_BASE)
        for edge in result.fact_graph.get_edges():
            assert edge.link == "atom_atom"


def _patch_fake_merlin(monkeypatch):
    """Replace the scorer's Merlin helper with the exact brute-force oracle.

    Enumerates every variable in whatever network it is handed (base, U-chain,
    reified R, or contradiction-free), so all four scoring methods -- including
    the MAP-based log Zmin floor -- route through the same exact oracle instead of
    the real Merlin executable.
    """
    monkeypatch.setattr(lcs_scorer_mod, "run_merlin", _brute_force_run_merlin)


class TestLCSScorer:
    def test_default_is_mean_marginal(self, monkeypatch):
        """Default method reproduces the deep-dive base mean-marginal (Eq. 4)."""
        _patch_fake_merlin(monkeypatch)
        result = _aeroparts_result(AEROPARTS_BASE)
        scores = LCSScorer("/fake/merlin").score(result)
        assert scores["method"] == "mean_marginal"
        assert scores["lcs"] == pytest.approx(0.587, abs=1e-3)
        assert scores["mean_marginal"] == pytest.approx(0.587, abs=1e-3)
        assert scores["log_z"] == pytest.approx(-9.75, abs=0.05)
        assert scores["num_atoms"] == 16
        # a10 (contradiction loser) is dragged below its 0.5 prior.
        assert scores["num_below_prior"] >= 1
        # Alternatives not computed unless requested.
        assert scores["consistency"] is None
        assert scores["reified"] is None
        assert scores["log_partition"] is None

    def test_consistency_matches_deepdive(self, monkeypatch):
        """(b) consistency probability = 0.813 on the AeroParts base (Table 3)."""
        _patch_fake_merlin(monkeypatch)
        result = _aeroparts_result(AEROPARTS_BASE)
        scores = LCSScorer("/fake/merlin").score(result, method="consistency")
        assert scores["method"] == "consistency"
        assert scores["consistency"] == pytest.approx(0.813, abs=1e-3)
        assert scores["lcs"] == scores["consistency"]

    def test_consistency_is_one_without_contradictions(self, monkeypatch):
        _patch_fake_merlin(monkeypatch)
        coherent = [r for r in AEROPARTS_BASE if r[2] != "contradiction"]
        result = _aeroparts_result(coherent)
        scores = LCSScorer("/fake/merlin").score(result, method="consistency")
        assert scores["consistency"] == pytest.approx(1.0, abs=1e-9)

    def test_reified_matches_deepdive(self, monkeypatch):
        """(c) reified P(R=1) = 0.150 on the AeroParts base, rho=0.5 (Table 3)."""
        _patch_fake_merlin(monkeypatch)
        result = _aeroparts_result(AEROPARTS_BASE)
        scores = LCSScorer("/fake/merlin").score(result, method="reified")
        assert scores["method"] == "reified"
        assert scores["reified"] == pytest.approx(0.150, abs=2e-3)
        assert scores["lcs"] == scores["reified"]

    def test_reified_subnet_matches_figure5(self, monkeypatch):
        """(c) reified P(R=1) = 0.459 on the 3-atom subnet of Figure 5.

        Subnet: a4 -> a7 (entailment .65); a7 != a10 (contradiction .93); rho=0.5.
        """
        _patch_fake_merlin(monkeypatch)
        atoms = {a: Atom(id=a, text=f"atom {a}") for a in ("a4", "a7", "a10")}
        rels = [
            MinedRelation("a4", "a7", "Cause-Effect", "entailment", 0.65, 1.0, 0.65),
            MinedRelation("a7", "a10", "Contrast", "contradiction", 0.93, 1.0, 0.93),
        ]
        miner = object.__new__(RelationMiner)
        miner.prior = 0.5
        fg = miner._build_fact_graph(atoms, rels)
        mn = build_markov_network(fg, use_priors=True, node_priors={a: 0.5 for a in atoms})
        result = MiningResult(atoms=atoms, relations=rels, fact_graph=fg,
                              markov_network=mn, coverage={}, config={"prior": 0.5})
        scores = LCSScorer("/fake/merlin").score(result, method="reified")
        assert scores["reified"] == pytest.approx(0.459, abs=2e-3)

    def test_log_partition(self, monkeypatch):
        """(d) normalized log-partition: graded in [0,1] via the MAP-world floor.

        Zmax = contradictions removed (ceiling), Zmin = the base network's MAP
        world mass (a provable lower bound: Z is a sum of nonneg terms, so
        log max_x mass(x) <= log Z). The base sits strictly between, grading it
        well inside (0,1).
        """
        _patch_fake_merlin(monkeypatch)
        result = _aeroparts_result(AEROPARTS_BASE)
        scores = LCSScorer("/fake/merlin").score(result, method="log_partition")
        assert scores["method"] == "log_partition"
        # log Z (base) and log Z_max (contradictions removed) reproduce Table 3.
        assert scores["log_z"] == pytest.approx(-9.75, abs=0.05)
        assert scores["log_z_max"] == pytest.approx(-8.25, abs=0.05)
        # Zmin is the MAP world mass; provably below the base log Z.
        assert scores["log_z_min"] == pytest.approx(-15.16, abs=0.05)
        # Valid ordering (guaranteed) and a graded, non-degenerate score.
        assert scores["log_z_min"] <= scores["log_z"] <= scores["log_z_max"]
        assert scores["log_partition"] == pytest.approx(0.7831, abs=1e-3)
        assert 0.0 < scores["log_partition"] < 1.0

    def test_log_partition_no_contradictions_is_one(self, monkeypatch):
        _patch_fake_merlin(monkeypatch)
        coherent = [r for r in AEROPARTS_BASE if r[2] != "contradiction"]
        result = _aeroparts_result(coherent)
        scores = LCSScorer("/fake/merlin").score(result, method="log_partition")
        # No contradictions: base log Z sits at the ceiling (Z == Zmax), while the
        # MAP floor is strictly below, so the score is exactly 1.0.
        assert scores["log_z"] == pytest.approx(scores["log_z_max"], abs=1e-6)
        assert scores["log_z_min"] < scores["log_z"]
        assert scores["log_partition"] == pytest.approx(1.0, abs=1e-9)

    def test_reified_prior_is_configurable(self, monkeypatch):
        _patch_fake_merlin(monkeypatch)
        result = _aeroparts_result(AEROPARTS_BASE)
        s_low = LCSScorer("/fake/merlin").score(result, method="reified", reified_prior=0.2)
        s_high = LCSScorer("/fake/merlin").score(result, method="reified", reified_prior=0.8)
        # A higher Bernoulli prior on R yields a higher P(R=1).
        assert s_high["reified"] > s_low["reified"]

    def test_unknown_method_raises(self, monkeypatch):
        _patch_fake_merlin(monkeypatch)
        result = _aeroparts_result(AEROPARTS_BASE)
        with pytest.raises(ValueError):
            LCSScorer("/fake/merlin").score(result, method="bogus")

    def test_empty_result(self, monkeypatch):
        _patch_fake_merlin(monkeypatch)
        empty = MiningResult(
            atoms={},
            relations=[],
            fact_graph=FactGraph(),
            markov_network=build_markov_network(FactGraph()),
            coverage={},
            config={"prior": 0.5},
        )
        scores = LCSScorer("/fake/merlin").score(empty)
        assert scores["lcs"] == 0.0
        assert scores["num_atoms"] == 0


# ---------------------------------------------------------------------------
# End-to-end miner flow with a mocked LLM (no real backend).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Conditional-strength UQ: surrogate-token reader + calibrators + method wiring.
# ---------------------------------------------------------------------------


class TestConditionalStrength:
    def test_surrogate_reader_renormalizes(self):
        # First token top_logprobs: Yes at -0.1, No at -2.3 -> P(yes) ~ 0.90.
        lps = [{
            "token": "Yes", "logprob": -0.1,
            "top_logprobs": [{"token": "Yes", "logprob": -0.1},
                             {"token": "No", "logprob": -2.3}],
        }]
        p = surrogate_probability_from_logprobs(lps)
        assert p == pytest.approx(0.9002495, abs=1e-3)

    def test_surrogate_reader_is_whitespace_case_tolerant(self):
        lps = [{
            "token": " No", "logprob": -0.05,
            "top_logprobs": [{"token": " NO", "logprob": -0.05},
                             {"token": " yes", "logprob": -3.0}],
        }]
        assert surrogate_probability_from_logprobs(lps) == pytest.approx(0.05, abs=1e-2)

    def test_surrogate_reader_none_when_absent(self):
        lps = [{"token": "maybe", "logprob": -0.1,
                "top_logprobs": [{"token": "maybe", "logprob": -0.1}]}]
        assert surrogate_probability_from_logprobs(lps) is None
        assert surrogate_probability_from_logprobs([]) is None

    def test_affirm_fraction(self):
        assert affirm_fraction(["Yes", "yes.", "No", "YES"]) == pytest.approx(0.75)
        assert affirm_fraction(["maybe", "unsure"]) is None

    def test_identity_calibrator_is_noop(self):
        cal = IdentityCalibrator()
        for p in (0.0, 0.3, 0.5, 0.9, 1.0):
            assert cal.transform(p) == p

    def test_temperature_softens_and_sharpens(self):
        assert TemperatureCalibrator(2.0).transform(0.9) < 0.9   # toward 0.5
        assert TemperatureCalibrator(0.5).transform(0.9) > 0.9   # sharpen
        assert TemperatureCalibrator(1.0).transform(0.73) == pytest.approx(0.73, abs=1e-6)

    def test_temperature_fit_recovers_sharpening(self):
        # Raw probs are under-confident relative to the {0,1} labels -> T < 1.
        raw = [0.6, 0.6, 0.4, 0.4]
        labels = [1, 1, 0, 0]
        T = TemperatureCalibrator.fit(raw, labels).temperature
        assert T < 1.0

    def test_platt_calibrator(self):
        # a=1, b=0 is the identity in logit space.
        assert PlattCalibrator(1.0, 0.0).transform(0.7) == pytest.approx(0.7, abs=1e-6)

    def test_auto_strength_method_resolution(self):
        from unittest.mock import MagicMock
        be = MagicMock(); be.model_id = "mock"
        lp = RelationMiner(be, nli_method="logprobs").strength_method
        sb = RelationMiner(be, nli_method="simbauq").strength_method
        assert lp == "surrogate_logprobs" and lp in STRENGTH_METHODS
        assert sb == "surrogate_sampled" and sb in STRENGTH_METHODS
        assert RelationMiner(be, nli_method="logprobs",
                             strength_method="verbalized").strength_method == "verbalized"

    def test_unknown_strength_method_raises(self):
        from unittest.mock import MagicMock
        be = MagicMock(); be.model_id = "mock"
        with pytest.raises(ValueError):
            RelationMiner(be, strength_method="bogus")


class _Thunk:
    def __init__(self, text, meta=None):
        self._text = text
        self._meta = meta or {}

    def __str__(self):
        return self._text


class _Sample:
    def __init__(self, text, meta=None):
        self.success = True
        self.result = _Thunk(text, meta)


def _is_surrogate_prompt(prompt) -> bool:
    """Heuristic: the surrogate strength prompt asks for a Yes/No first word."""
    return "Yes or No" in str(prompt)


def _is_strength_prompt(prompt) -> bool:
    """Heuristic: the verbalized strength prompt asks for [p=0.NN]."""
    return "[p=0.NN]" in str(prompt)


class TestMinerEndToEnd:
    def _fake_ainstruct_factory(self, surrogate_p_yes=0.8):
        """Build a fake ainstruct: Prompt A senses + surrogate/verbalized strength."""
        async def fake_ainstruct(prompt, **kw):
            uv = kw["user_variables"]
            if _is_surrogate_prompt(prompt):
                word = "Yes" if surrogate_p_yes >= 0.5 else "No"
                return _Sample(word, meta=_yesno_logprob_meta(surrogate_p_yes))
            if _is_strength_prompt(prompt):
                return _Sample("Fairly likely. [p=0.70]")
            b = uv.get("atom_b", "")
            if "fired" in b:
                return _Sample("[sense=Cause-Effect] [coupling=entailment]")
            if "harmed" in b or "died" in b:
                return _Sample("[sense=Contrast] [coupling=contradiction]")
            return _Sample("[sense=None] [coupling=none]")

        return fake_ainstruct

    def test_mine_from_atoms_with_mocked_llm(self, monkeypatch):
        """The full mine flow: Prompt A -> compile -> surrogate strength -> MRF."""
        import mellea.stdlib.functional as mfuncs
        from unittest.mock import MagicMock

        monkeypatch.setattr(mfuncs, "ainstruct", self._fake_ainstruct_factory(0.8))

        backend = MagicMock()
        backend.model_id = "mock"
        # Default strength for nli_method="logprobs" is surrogate_logprobs.
        miner = RelationMiner(backend, nli_method="logprobs", pair_policy="all_pairs")
        assert miner.strength_method == "surrogate_logprobs"
        atoms = [
            "The stock fell 15 percent",
            "The CEO was fired",
            "No one was harmed",
            "Three people died",
        ]
        result = miner.mine_from_atoms(atoms, " ".join(atoms))

        # 4 atoms -> 12 ordered pairs; the "None" couplings are dropped.
        assert result.coverage["pairs_scored"] == 12
        assert result.coverage["dropped_none"] >= 1
        assert result.relations
        for rel in result.relations:
            assert rel.level1_type in (LEVEL1_ENTAILMENT, LEVEL1_CONTRADICTION,
                                       LEVEL1_EQUIVALENCE)
            assert 0.0 <= rel.probability <= 1.0
            # Surrogate strength read from the fake logprobs = 0.8.
            assert rel.strength == pytest.approx(0.8, abs=1e-6)
            assert rel.strength_raw == pytest.approx(0.8, abs=1e-6)
        assert len(result.markov_network.factors) == 4 + len(result.relations)
        assert result.markov_network.to_uai().splitlines()[0] == "MARKOV"

    def test_verbalized_strength_still_parses(self, monkeypatch):
        """The verbalized baseline still reads [p=0.NN] at face value."""
        import mellea.stdlib.functional as mfuncs
        from unittest.mock import MagicMock

        monkeypatch.setattr(mfuncs, "ainstruct", self._fake_ainstruct_factory())
        backend = MagicMock()
        backend.model_id = "mock"
        miner = RelationMiner(backend, nli_method="logprobs",
                              strength_method="verbalized", pair_policy="all_pairs")
        atoms = ["The CEO was fired", "The stock fell"]
        result = miner.mine_from_atoms(atoms, " ".join(atoms))
        assert result.relations
        for rel in result.relations:
            assert rel.strength == pytest.approx(0.70, abs=1e-6)

    def test_surrogate_sampled_affirm_fraction(self, monkeypatch):
        """surrogate_sampled: strength = affirm fraction over N Yes/No samples."""
        import mellea.stdlib.functional as mfuncs
        from unittest.mock import MagicMock

        # 3 of every 4 surrogate samples say Yes -> strength 0.75.
        state = {"i": 0}

        async def fake_ainstruct(prompt, **kw):
            if _is_surrogate_prompt(prompt):
                state["i"] += 1
                return _Sample("Yes" if state["i"] % 4 != 0 else "No")
            uv = kw["user_variables"]
            b = uv.get("atom_b", "")
            if "fired" in b:
                return _Sample("[sense=Cause-Effect] [coupling=entailment]")
            return _Sample("[sense=None] [coupling=none]")

        monkeypatch.setattr(mfuncs, "ainstruct", fake_ainstruct)
        backend = MagicMock()
        backend.model_id = "mock"
        miner = RelationMiner(backend, nli_method="simbauq",
                              strength_method="surrogate_sampled", strength_samples=4,
                              pair_policy="all_pairs")
        assert miner.strength_method == "surrogate_sampled"
        atoms = ["The CEO was fired", "The stock fell"]
        result = miner.mine_from_atoms(atoms, " ".join(atoms))
        assert result.relations
        for rel in result.relations:
            assert rel.strength == pytest.approx(0.75, abs=1e-6)

    def test_calibrator_is_applied(self, monkeypatch):
        """A supplied calibrator transforms the raw strength before the factor."""
        import mellea.stdlib.functional as mfuncs
        from unittest.mock import MagicMock
        from fact_reasoner.lcs import TemperatureCalibrator

        monkeypatch.setattr(mfuncs, "ainstruct", self._fake_ainstruct_factory(0.9))
        backend = MagicMock()
        backend.model_id = "mock"
        cal = TemperatureCalibrator(2.0)  # softens 0.9 toward 0.5
        miner = RelationMiner(backend, nli_method="logprobs",
                              strength_calibrator=cal, pair_policy="all_pairs")
        atoms = ["The CEO was fired", "The stock fell"]
        result = miner.mine_from_atoms(atoms, " ".join(atoms))
        assert result.relations
        for rel in result.relations:
            assert rel.strength_raw == pytest.approx(0.9, abs=1e-6)
            # Calibrated strength is the softened value, strictly below the raw 0.9.
            assert rel.strength == pytest.approx(cal.transform(0.9), abs=1e-6)
            assert rel.strength < rel.strength_raw

    def test_response_grounding_prunes_unasserted_edges(self, monkeypatch):
        """The grounded Prompt A drops pairs the response does not relate.

        The fake Prompt A (which always receives the response now) answers
        ``none`` for a pair whose atoms share no content (the response draws no
        connection), so those edges never enter the graph.
        """
        import mellea.stdlib.functional as mfuncs
        from unittest.mock import MagicMock

        def _content(s):
            return {w for w in re.findall(r"[A-Za-z]+", s.lower()) if len(w) > 3}

        async def fake_ainstruct(prompt, **kw):
            uv = kw["user_variables"]
            if _is_surrogate_prompt(prompt):
                return _Sample("Yes", meta=_yesno_logprob_meta(0.8))
            if _is_strength_prompt(prompt):
                return _Sample("Likely. [p=0.70]")
            a, b = uv.get("atom_a", ""), uv.get("atom_b", "")
            # The response variable is always present; answer "none" when the two
            # atoms share no content (the response does not relate them).
            assert "response" in uv, "Prompt A must be response-grounded"
            if not (_content(a) & _content(b)):
                return _Sample("[sense=None] [coupling=none]")
            return _Sample("[sense=Cause-Effect] [coupling=entailment]")

        monkeypatch.setattr(mfuncs, "ainstruct", fake_ainstruct)
        backend = MagicMock()
        backend.model_id = "mock"
        atoms = [
            "The reactor overheated during the test.",
            "The cafeteria served pasta for lunch.",
            "The reactor was shut down after overheating.",
        ]
        response = " ".join(atoms)
        # all_pairs so every pair is scored; grounding does the pruning.
        miner = RelationMiner(backend, nli_method="logprobs", pair_policy="all_pairs")
        result = miner.mine_from_atoms(atoms, response)

        # The reactor<->cafeteria pairs (no shared content) are dropped as none;
        # only the two reactor atoms (a0<->a2) relate, in both directions.
        assert result.coverage["dropped_none"] >= 1
        for rel in result.relations:
            ids = {rel.source_id, rel.target_id}
            assert ids == {"a0", "a2"}

    def test_mine_from_atoms_requires_response(self, monkeypatch):
        """Mining is always response-grounded: no/empty response raises."""
        import mellea.stdlib.functional as mfuncs
        from unittest.mock import MagicMock

        monkeypatch.setattr(mfuncs, "ainstruct", self._fake_ainstruct_factory(0.8))
        backend = MagicMock()
        backend.model_id = "mock"
        miner = RelationMiner(backend, nli_method="logprobs", pair_policy="all_pairs")
        atoms = ["The stock fell 15 percent", "The CEO was fired"]
        with pytest.raises(TypeError):
            miner.mine_from_atoms(atoms)  # response is a required positional arg
        with pytest.raises(ValueError):
            miner.mine_from_atoms(atoms, "   ")  # empty response
        with pytest.raises(ValueError):
            miner.mine_from_response("")  # empty raw response
