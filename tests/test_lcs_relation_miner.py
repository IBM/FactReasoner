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
    LEVEL1_CONTRADICTION,
    LEVEL1_ENTAILMENT,
    LEVEL1_EQUIVALENCE,
    LEVEL1_NONE,
    Level2Sense,
    compile_sense,
    coupling_from_string,
)
from fact_reasoner.lcs.relation_miner import MinedRelation, MiningResult, RelationMiner
from fact_reasoner.lcs import lcs_scorer as lcs_scorer_mod
from fact_reasoner.lcs.lcs_scorer import LCSScorer


# ---------------------------------------------------------------------------
# Brute-force MRF oracle (exact, replaces Merlin for tests).
# ---------------------------------------------------------------------------


def _brute_force_marginals(network, node_priors):
    """Exact marginals and log Z of a binary pairwise Markov network.

    Enumerates all 2^n worlds. Only feasible for small n (the diagnostic
    examples), which is exactly what the deep-dive uses as a validation oracle.
    """
    var_names = list(node_priors.keys())
    idx = {v: i for i, v in enumerate(var_names)}
    n = len(var_names)

    z = 0.0
    ones = [0.0] * n
    for world in itertools.product([0, 1], repeat=n):
        w = 1.0
        for variables, _cards, values in network.factors:
            k = 0
            for v in variables:
                k = k * 2 + world[idx[v]]
            w *= values[k]
        z += w
        for i, bit in enumerate(world):
            if bit == 1:
                ones[i] += w
    marginals = {var_names[i]: ones[i] / z for i in range(n)}
    return marginals, math.log(z)


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
    marginals, log_z = _brute_force_marginals(mn, priors)
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

    def test_coupling_from_string(self):
        assert coupling_from_string("[entailment]") == LEVEL1_ENTAILMENT
        assert coupling_from_string("contradiction") == LEVEL1_CONTRADICTION
        assert coupling_from_string("equivalence") == LEVEL1_EQUIVALENCE
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


class TestCandidatePairs:
    def test_all_pairs_is_all_ordered(self):
        atoms = _atoms(["x", "y", "z"])
        pairs, cov = cp.select(atoms, policy="all_pairs")
        assert len(pairs) == 3 * 2  # n(n-1) ordered
        assert cov["pairs_pruned"] == 0

    def test_windowed_respects_radius(self):
        atoms = _atoms([f"s{i}" for i in range(6)])
        pairs, cov = cp.select(atoms, policy="windowed", window=2)
        # forward pairs only, |j-i| in [1,2]
        for s, t in pairs:
            si = int(s[1:])
            ti = int(t[1:])
            assert 0 < ti - si <= 2
        assert cov["num_window_pairs"] == len(pairs)
        assert cov["forward_pairs_possible"] == 6 * 5 // 2

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
            atoms, policy="gated", window=1, gate="entity", gate_threshold=0.05
        )
        assert cov["num_callback_pairs"] >= 1
        assert ("a0", "a5") in pairs

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError):
            cp.select(_atoms(["x"]), policy="bogus")


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


class TestLCSScorer:
    def test_score_reproduces_aeroparts(self, monkeypatch):
        """LCSScorer.score against a monkeypatched (brute-force) Merlin."""
        result = _aeroparts_result(AEROPARTS_BASE)
        priors = {a: 0.5 for a in AEROPARTS_IDS}

        def fake_run_merlin(network, merlin_path, *, task="MAR", ibound=6,
                            query_variables=None, verbose=False):
            marginals, log_z = _brute_force_marginals(network, priors)
            if task == "MAR":
                names = query_variables or list(marginals)
                return {
                    "task": "MAR",
                    "marginals": [
                        {"variable": v, "probabilities": [1 - marginals[v], marginals[v]]}
                        for v in names
                    ],
                    "all_marginals": [],
                }
            return {"task": "PR", "log_z": log_z}

        monkeypatch.setattr(lcs_scorer_mod, "run_merlin", fake_run_merlin)

        scorer = LCSScorer("/fake/merlin")
        scores = scorer.score(result)
        assert scores["lcs"] == pytest.approx(0.587, abs=1e-3)
        assert scores["log_z"] == pytest.approx(-9.75, abs=0.05)
        assert scores["num_atoms"] == 16
        # a10 (contradiction loser) is dragged below its 0.5 prior.
        assert scores["num_below_prior"] >= 1

    def test_empty_result(self, monkeypatch):
        empty = MiningResult(
            atoms={},
            relations=[],
            fact_graph=FactGraph(),
            markov_network=build_markov_network(FactGraph()),
            coverage={},
            config={"prior": 0.5},
        )
        scorer = LCSScorer("/fake/merlin")
        scores = scorer.score(empty)
        assert scores["lcs"] == 0.0
        assert scores["num_atoms"] == 0


# ---------------------------------------------------------------------------
# End-to-end miner flow with a mocked LLM (no real backend).
# ---------------------------------------------------------------------------


class _Thunk:
    def __init__(self, text):
        self._text = text
        self._meta = {}

    def __str__(self):
        return self._text


class _Sample:
    def __init__(self, text):
        self.success = True
        self.result = _Thunk(text)


class TestMinerEndToEnd:
    def test_mine_from_atoms_with_mocked_llm(self, monkeypatch):
        """The full mine flow: Prompt A -> compile -> Prompt B -> MRF."""
        import mellea.stdlib.functional as mfuncs
        from unittest.mock import MagicMock

        async def fake_ainstruct(prompt, **kw):
            uv = kw["user_variables"]
            if "coupling" in uv:  # Prompt B (strength)
                return _Sample("Fairly likely. [p=0.70]")
            b = uv.get("atom_b", "")
            if "fired" in b:
                return _Sample("[sense=Cause-Effect] [coupling=entailment]")
            if "harmed" in b or "died" in b:
                return _Sample("[sense=Contrast] [coupling=contradiction]")
            return _Sample("[sense=None] [coupling=none]")

        monkeypatch.setattr(mfuncs, "ainstruct", fake_ainstruct)

        backend = MagicMock()
        backend.model_id = "mock"
        # SIMBA-UQ method avoids the logprobs requirement (the mock has none),
        # exercising the confidence fallback path deterministically.
        miner = RelationMiner(
            backend, nli_method="logprobs", pair_policy="all_pairs"
        )
        atoms = [
            "The stock fell 15 percent",
            "The CEO was fired",
            "No one was harmed",
            "Three people died",
        ]
        result = miner.mine_from_atoms(atoms)

        # 4 atoms -> 12 ordered pairs; the "None" couplings are dropped.
        assert result.coverage["pairs_scored"] == 12
        assert result.coverage["dropped_none"] >= 1
        # Every kept relation has an edge-producing coupling and p in [0, 1].
        assert result.relations
        for rel in result.relations:
            assert rel.level1_type in (LEVEL1_ENTAILMENT, LEVEL1_CONTRADICTION,
                                       LEVEL1_EQUIVALENCE)
            assert 0.0 <= rel.probability <= 1.0
        # MRF has one unary factor per atom plus one pairwise per relation.
        assert len(result.markov_network.factors) == 4 + len(result.relations)
        assert result.markov_network.to_uai().splitlines()[0] == "MARKOV"
