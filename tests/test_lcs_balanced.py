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

"""The balanced readout (f): support rewards plus conflict penalties.

(f) is ``sigmoid(log E_pi[prod (1 - w_r V_r) * prod (1 + u_r C_r)] / n)`` with the
expectation under the priors alone. Its defining property is monotonicity in every
relation in the direction coherence requires: removing or softening a conflict and
adding or strengthening a support never lower it. Everything here runs against the
exact brute-force oracle.
"""

import itertools
import math
import random

import pytest

from fact_reasoner.experiments.mock import (
    MAX_BRUTEFORCE_VARS,
    brute_force_run_merlin as _brute_force_run_merlin,
)
from fact_reasoner.fact_graph import Edge, FactGraph, Node
from fact_reasoner.lcs import lcs_scorer as lcs_scorer_mod
from fact_reasoner.lcs.lcs_scorer import (
    ALL_LCS_METHODS,
    LCS_METHODS,
    LCSScorer,
    _credited,
    _reward_factor,
)
from fact_reasoner.lcs.relation_miner import MiningResult
from tests.test_lcs_relation_miner import AEROPARTS_BASE_5

CONFLICTS = ("contradiction", "exclusive")
SUPPORTS = ("entailment", "equivalence", "co_necessity")


@pytest.fixture
def scorer(monkeypatch):
    """An LCSScorer whose Merlin helper is the exact brute-force oracle."""
    monkeypatch.setattr(lcs_scorer_mod, "run_merlin", _brute_force_run_merlin)
    monkeypatch.setattr(
        lcs_scorer_mod, "DEFAULT_MAX_NETWORK_VARS", MAX_BRUTEFORCE_VARS
    )
    return LCSScorer("/fake/merlin")


def _result(relations, priors):
    fg = FactGraph()
    for aid, pi in priors.items():
        fg.add_node(Node(id=aid, type="atom", probability=pi))
    for s, t, ty, p in relations:
        fg.add_edge(Edge(source=s, target=t, type=ty, probability=p, link="atom_atom"))
    return MiningResult(
        atoms={aid: None for aid in priors},
        relations=[],
        fact_graph=fg,
        markov_network=None,
        config={"prior": 0.5},
    )


def _balanced(scorer, relations, priors):
    return scorer.score(_result(relations, priors), method="balanced")["balanced"]


def _reference(relations, priors):
    """Independent closed-form: enumerate worlds, multiply prior and factors."""
    ids = list(priors)
    total = 0.0
    for world in itertools.product((0, 1), repeat=len(ids)):
        a = dict(zip(ids, world))
        w = 1.0
        for aid in ids:
            w *= priors[aid] if a[aid] else 1.0 - priors[aid]
        for s, t, ty, p in relations:
            x, y = a[s], a[t]
            if ty == "contradiction" and x and y:
                w *= 1.0 - min(1.0 - 1e-6, p)
            elif ty == "exclusive" and x == y:
                w *= 1.0 - min(1.0 - 1e-6, p)
            elif ty in ("entailment", "equivalence") and x and y:
                w *= 1.0 + p
            elif ty == "co_necessity" and x != y:
                w *= 1.0 + p
        total += w
    c = 2.0  # the default display scale
    return 1.0 / (1.0 + math.exp(-c * math.log(total) / len(ids)))


UNIFORM6 = {f"a{i}": 0.5 for i in range(6)}
CHAIN = [(f"a{i}", f"a{i + 1}", "entailment", 0.9) for i in range(5)]
CONTRA = [("a0", "a5", "contradiction", 0.9)]


class TestRegistration:
    def test_not_in_default_methods(self):
        assert "balanced" not in LCS_METHODS
        assert "balanced" in ALL_LCS_METHODS

    def test_score_reports_log_z(self, scorer):
        s = scorer.score(_result(CHAIN, UNIFORM6), method="balanced")
        assert s["lcs"] == s["balanced"]
        assert s["log_z_balanced"] is not None


class TestFactors:
    def test_credit_worlds(self):
        assert [_credited("entailment", s, t) for s in (0, 1) for t in (0, 1)] == [
            False, False, False, True,
        ]
        assert [_credited("co_necessity", s, t) for s in (0, 1) for t in (0, 1)] == [
            False, True, True, False,
        ]

    def test_reward_table(self):
        assert _reward_factor("equivalence", 0.5) == pytest.approx([1, 1, 1, 1.5])

    def test_conflict_rejected_as_support(self):
        with pytest.raises(ValueError):
            _credited("contradiction", 1, 1)


class TestValues:
    def test_bag_scores_one_half(self, scorer):
        assert _balanced(scorer, [], UNIFORM6) == 0.5

    def test_support_lifts_conflict_lowers(self, scorer):
        chain = _balanced(scorer, CHAIN, UNIFORM6)
        contra = _balanced(scorer, CONTRA, UNIFORM6)
        both = _balanced(scorer, CHAIN + CONTRA, UNIFORM6)
        assert contra < 0.5 < chain
        assert contra < both < chain

    @pytest.mark.parametrize("relations", [CHAIN, CONTRA, CHAIN + CONTRA])
    def test_matches_reference(self, scorer, relations):
        assert _balanced(scorer, relations, UNIFORM6) == pytest.approx(
            _reference(relations, UNIFORM6), abs=1e-9
        )

    def test_matches_reference_skewed_priors(self, scorer):
        priors = {f"a{i}": 0.1 + 0.15 * i for i in range(6)}
        rels = CHAIN + CONTRA + [("a1", "a4", "co_necessity", 0.7),
                                 ("a2", "a3", "exclusive", 0.8)]
        assert _balanced(scorer, rels, priors) == pytest.approx(
            _reference(rels, priors), abs=1e-9
        )

    def test_aeroparts_family_strictly_ordered(self, scorer):
        priors = {f"a{i}": 0.5 for i in range(1, 17)}
        b = AEROPARTS_BASE_5
        family = [
            b + [("a3", "a12", "contradiction", 0.85)],
            b,
            [(s, t, ty, (0.55 if (s, t) == ("a11", "a12") else p)) for s, t, ty, p in b],
            [r for r in b if (r[0], r[1]) != ("a7", "a10")],
            [r for r in b if r[2] != "exclusive"],
        ]
        values = [_balanced(scorer, r, priors) for r in family]
        assert all(lo < hi for lo, hi in zip(values, values[1:])), values

    def test_padding_is_bounded_by_entailment_padding(self, scorer):
        """co_necessity pads are worth no more than entailment pads."""
        pairs = [(a, b) for a in UNIFORM6 for b in UNIFORM6 if a < b]
        con = _balanced(scorer, [(a, b, "co_necessity", 0.99) for a, b in pairs]
                        + CONTRA, UNIFORM6)
        ent = _balanced(scorer, [(a, b, "entailment", 0.99) for a, b in pairs]
                        + CONTRA, UNIFORM6)
        assert con <= ent + 0.06


class TestMonotonicity:
    """Every edit in the coherence-improving direction is non-decreasing."""

    def test_random_networks(self, scorer):
        rng = random.Random(0)
        types = CONFLICTS + SUPPORTS
        for _ in range(120):
            n = rng.randint(2, 6)
            priors = {f"a{i}": rng.uniform(0.05, 0.95) for i in range(n)}
            ids = list(priors)
            rels = []
            for _ in range(rng.randint(1, 6)):
                s, t = rng.sample(ids, 2)
                rels.append((s, t, rng.choice(types), rng.uniform(0.0, 1.0)))
            k = rng.randrange(len(rels))
            s, t, ty, p = rels[k]
            base = _balanced(scorer, rels, priors)
            removed = _balanced(scorer, rels[:k] + rels[k + 1 :], priors)
            weaker = _balanced(
                scorer, rels[:k] + [(s, t, ty, rng.uniform(0, p))] + rels[k + 1 :], priors
            )
            if ty in CONFLICTS:
                assert removed >= base - 1e-12 and weaker >= base - 1e-12
            else:
                assert removed <= base + 1e-12 and weaker <= base + 1e-12
            a, b = rng.sample(ids, 2)
            added = _balanced(
                scorer, rels + [(a, b, rng.choice(SUPPORTS), rng.random())], priors
            )
            assert added >= base - 1e-12


class TestDisplayScale:
    """The scale c only stretches the score; it never changes an ordering."""

    def test_default_scale(self):
        assert lcs_scorer_mod.DEFAULT_BALANCED_SCALE == 2.0

    def test_one_bit_per_claim_reads_point_eight(self, scorer):
        """One maximal support per claim -- one bit each -- reads 0.8 (odds 4:1) at c=2."""
        # A 3-cycle of equivalences: three claims, three relations, so one relation
        # per claim. With priors near 1 the all-true world carries all the mass and
        # credits every relation, so E = 2^3 and log E / n = log 2 -- one bit each.
        priors = {f"a{i}": 1 - 1e-9 for i in range(3)}
        rels = [("a0", "a1", "equivalence", 1.0), ("a1", "a2", "equivalence", 1.0),
                ("a2", "a0", "equivalence", 1.0)]
        assert _balanced(scorer, rels, priors) == pytest.approx(0.8, abs=1e-4)

    def test_scale_preserves_order(self, monkeypatch):
        monkeypatch.setattr(lcs_scorer_mod, "run_merlin", _brute_force_run_merlin)
        variants = [CONTRA, [], CHAIN + CONTRA, CHAIN]
        orders = []
        for c in (0.5, 1.0, 2.0, 8.0):
            s = LCSScorer("/fake/merlin", balanced_scale=c)
            vals = [
                s.score(_result(r, UNIFORM6), method="balanced")["balanced"]
                for r in variants
            ]
            orders.append(sorted(range(len(vals)), key=vals.__getitem__))
            assert vals[1] == 0.5
        assert all(o == orders[0] for o in orders)

    def test_scale_must_be_positive(self):
        with pytest.raises(ValueError):
            LCSScorer("/fake/merlin", balanced_scale=0.0)
