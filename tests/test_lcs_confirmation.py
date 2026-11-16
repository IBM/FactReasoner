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

"""The confirmation readout (g): the committed world under a sign-definite MRF.

``Lambda = log P_R(x*) - log pi(x*)`` and ``confirmation = sigmoid(Lambda / n)``,
with ``x*`` the committed world (verdict polarity by default: ``x*_i = 1`` iff
``pi_i >= 1/2``). Its defining property (docs/arr2027/coherence, Thm. 1) is strict
monotonicity under every coherence-improving edit -- add/strengthen a support,
delete/weaken a conflict, both relative to ``x*`` -- for any priors and any other
relations. Everything here runs against
the exact brute-force oracle and an independent enumeration.
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
from fact_reasoner.lcs.lcs_scorer import ALL_LCS_METHODS, LCS_METHODS, LCSScorer
from fact_reasoner.lcs.relation_miner import MiningResult

CONFLICTS = ("contradiction", "exclusive")
SUPPORTS = ("entailment", "equivalence", "co_necessity")

# Violated sets V(tau) over (a_s, a_t).
VIOLATED = {
    "entailment": {(1, 0)},
    "equivalence": {(0, 1), (1, 0)},
    "co_necessity": {(0, 0)},
    "contradiction": {(1, 1)},
    "exclusive": {(0, 0), (1, 1)},
}


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


def _lam(scorer, relations, priors, polarity=None):
    out = scorer.score(
        _result(relations, priors), method="confirmation", polarity=polarity
    )
    return out["log_confirmation"], out["confirmation"]


def _verdict(priors):
    return {a: 0 if pi < 0.5 else 1 for a, pi in priors.items()}


def _is_support(rel, x):
    s, t, ty, _ = rel
    return (x[s], x[t]) not in VIOLATED[ty]


def _reference_lambda(relations, priors, x=None, mrf_priors=None):
    """Independent enumeration of log P_R(x*) - log pi(x*).

    x* defaults to the verdict of ``priors``; the MRF's own priors default to
    uniform (the scorer's ``confirmation_priors="uniform"``).
    """
    if x is None:
        x = _verdict(priors)
    if mrf_priors is None:
        mrf_priors = {a: 0.5 for a in priors}
    priors = mrf_priors
    ids = list(priors)
    idx = {a: i for i, a in enumerate(ids)}
    z = 0.0
    for w in itertools.product((0, 1), repeat=len(ids)):
        m = 1.0
        for a, pi in priors.items():
            m *= pi if w[idx[a]] else 1.0 - pi
        for s, t, ty, p in relations:
            if (w[idx[s]], w[idx[t]]) in VIOLATED[ty]:
                m *= 1.0 - p
        z += m
    committed = 1.0
    for s, t, ty, p in relations:
        if (x[s], x[t]) in VIOLATED[ty]:
            committed *= 1.0 - p
    return math.log(committed) - math.log(z)


def _random_case(rng, n):
    priors = {f"a{i}": rng.uniform(0.05, 0.95) for i in range(n)}
    rels = []
    for _ in range(rng.randint(0, 2 * n)):
        s, t = rng.sample(sorted(priors), 2)
        rels.append((s, t, rng.choice(CONFLICTS + SUPPORTS), rng.uniform(0.05, 0.95)))
    return priors, rels


def test_not_in_default_methods():
    assert "confirmation" in ALL_LCS_METHODS
    assert "confirmation" not in LCS_METHODS


def test_relation_free_is_neutral(scorer):
    lam, score = _lam(scorer, [], {"a0": 0.3, "a1": 0.8})
    assert lam == 0.0 and score == 0.5


def test_matches_independent_enumeration(scorer):
    rng = random.Random(7)
    for _ in range(40):
        priors, rels = _random_case(rng, rng.randint(2, 6))
        lam, score = _lam(scorer, rels, priors)
        ref = _reference_lambda(rels, priors)
        assert lam == pytest.approx(ref, abs=1e-9)
        assert score == pytest.approx(1 / (1 + math.exp(-ref / len(priors))))


def test_verdict_polarity_flips_signs(scorer):
    """Under verdict polarity a contradiction between a true and a false claim is a
    support, and an entailment from a true claim to a false one is a conflict."""
    priors = {"a0": 0.9, "a1": 0.1}
    assert _lam(scorer, [("a0", "a1", "contradiction", 0.8)], priors)[0] > 0
    assert _lam(scorer, [("a0", "a1", "entailment", 0.8)], priors)[0] < 0
    # The same relations read as asserted keep their x* = 1 signs.
    asserted = {"a0": 1, "a1": 1}
    assert _lam(scorer, [("a0", "a1", "contradiction", 0.8)], priors, asserted)[0] < 0
    assert _lam(scorer, [("a0", "a1", "entailment", 0.8)], priors, asserted)[0] > 0


def test_asserted_polarity_source(monkeypatch):
    monkeypatch.setattr(lcs_scorer_mod, "run_merlin", _brute_force_run_merlin)
    s = LCSScorer("/fake/merlin", polarity="asserted")
    priors = {"a0": 0.9, "a1": 0.1}
    rels = [("a0", "a1", "contradiction", 0.8)]
    out = s.score(_result(rels, priors), method="confirmation")
    assert out["committed_world"] == {"a0": 1, "a1": 1}
    assert out["log_confirmation"] == pytest.approx(
        _reference_lambda(rels, priors, {"a0": 1, "a1": 1})
    )
    with pytest.raises(ValueError):
        LCSScorer("/fake/merlin", polarity="bogus")


def test_sign_of_a_single_relation(scorer):
    priors = {"a0": 0.5, "a1": 0.5}
    for ty in SUPPORTS:
        assert _lam(scorer, [("a0", "a1", ty, 0.8)], priors)[0] > 0
    for ty in CONFLICTS:
        assert _lam(scorer, [("a0", "a1", ty, 0.8)], priors)[0] < 0


def test_strictly_monotone_under_every_improving_edit(scorer):
    """Thm. 1: each improving edit (signs relative to x*) raises Lambda, in any context."""
    rng = random.Random(11)
    checked = 0
    for _ in range(150):
        priors, rels = _random_case(rng, rng.randint(2, 6))
        x = _verdict(priors)
        before = _lam(scorer, rels, priors)[0]
        kind = rng.choice(("add_support", "strengthen", "delete_conflict", "weaken"))
        new = list(rels)
        if kind == "add_support":
            while True:
                s, t = rng.sample(sorted(priors), 2)
                cand = (s, t, rng.choice(CONFLICTS + SUPPORTS), rng.uniform(0.05, 0.95))
                if _is_support(cand, x):
                    break
            new.append(cand)
        else:
            want_support = kind == "strengthen"
            cands = [i for i, r in enumerate(rels) if _is_support(r, x) == want_support]
            if not cands:
                continue
            i = rng.choice(cands)
            s, t, ty, p = rels[i]
            if kind == "strengthen":
                new[i] = (s, t, ty, p + (0.99 - p) * rng.uniform(0.1, 1.0))
            elif kind == "weaken":
                new[i] = (s, t, ty, p * rng.uniform(0.0, 0.9))
            else:
                del new[i]
        after = _lam(scorer, new, priors)[0]
        assert after > before, (kind, rels, new)
        checked += 1
    assert checked > 100


def test_edit_identity(scorer):
    """Adding r changes Lambda by log psi_r(1) - log E_{P_R}[psi_r]."""
    priors = {"a0": 0.6, "a1": 0.4, "a2": 0.7}
    base = [("a0", "a1", "entailment", 0.7), ("a1", "a2", "contradiction", 0.6)]
    new = ("a0", "a2", "equivalence", 0.8)
    lam0 = _lam(scorer, base, priors)[0]
    lam1 = _lam(scorer, base + [new], priors)[0]
    # E_{P_R}[psi] = 1 - p * P_R(a0 != a2), by enumeration.
    ids = list(priors)
    num = den = 0.0
    for x in itertools.product((0, 1), repeat=3):
        w = {a: v for a, v in zip(ids, x)}
        m = 0.5 ** len(ids)  # the MRF's priors are uniform by default
        for s, t, ty, p in base:
            if (w[s], w[t]) in VIOLATED[ty]:
                m *= 1 - p
        den += m
        if w["a0"] != w["a2"]:
            num += m
    expected = -math.log(1 - 0.8 * num / den)
    assert lam1 - lam0 == pytest.approx(expected, abs=1e-9)


def test_redundant_support_saturates(scorer):
    """Repeating a support adds less each time, and Lambda <= -log pi(1)."""
    priors = {"a0": 0.5, "a1": 0.5}
    gains, prev = [], 0.0
    rels = []
    for _ in range(6):
        rels.append(("a0", "a1", "equivalence", 0.9))
        lam = _lam(scorer, rels, priors)[0]
        gains.append(lam - prev)
        prev = lam
    assert all(g > 0 for g in gains)
    assert all(a > b for a, b in zip(gains, gains[1:]))
    assert prev < -math.log(0.25)


def test_symmetric_reversal_and_ordering_only_are_invariant(scorer):
    priors = {"a0": 0.9, "a1": 0.1, "a2": 0.9}
    a = [("a0", "a1", "equivalence", 0.9), ("a1", "a2", "exclusive", 0.9)]
    b = [("a1", "a0", "equivalence", 0.9), ("a2", "a1", "exclusive", 0.9)]
    assert _lam(scorer, a, priors)[0] == pytest.approx(_lam(scorer, b, priors)[0])


def test_uniform_priors_do_not_collapse_under_certain_verdicts(monkeypatch):
    """Near-certain verdicts fix x* but, with the default uniform MRF priors, leave
    the score informative; using them as priors makes Lambda vanish."""
    monkeypatch.setattr(lcs_scorer_mod, "run_merlin", _brute_force_run_merlin)
    priors = {"a0": 1 - 1e-6, "a1": 1 - 1e-6, "a2": 1e-6}
    good = [("a0", "a1", "entailment", 0.9), ("a1", "a2", "contradiction", 0.9)]
    bad = [("a0", "a1", "contradiction", 0.9), ("a0", "a2", "entailment", 0.9)]
    uni = LCSScorer("/fake/merlin")
    lg = uni.score(_result(good, priors), method="confirmation")["log_confirmation"]
    lb = uni.score(_result(bad, priors), method="confirmation")["log_confirmation"]
    assert lg - lb > 1.0
    ver = LCSScorer("/fake/merlin", confirmation_priors="verdict")
    vg = ver.score(_result(good, priors), method="confirmation")["log_confirmation"]
    vb = ver.score(_result(bad, priors), method="confirmation")["log_confirmation"]
    assert abs(vg) < 1e-3 and abs(vb) < 1e-3  # O(eps): nothing left to confirm
    assert vg == pytest.approx(_reference_lambda(good, priors, mrf_priors=priors), abs=1e-9)
    with pytest.raises(ValueError):
        LCSScorer("/fake/merlin", confirmation_priors="bogus")
