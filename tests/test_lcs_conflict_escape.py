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

"""The conflict-escape readout (e): values, monotonicity, inference sharing.

(e) is ``E_{P0}[prod_r (1 - w_r [r violated])]`` with ``P0`` the conflict-free
network and ``w_r = max(0, 2 p_r - 1)``. Its defining property is that deleting
or softening a conflict can never lower it, for any priors and support
structure -- which the mean marginal and the normalized log-partition do not
satisfy. Everything here runs against the exact brute-force oracle.
"""

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
    _penalty_factor,
    _penalty_weight,
)
from fact_reasoner.lcs.relation_miner import MiningResult
from tests.test_lcs_relation_miner import (
    AEROPARTS_BASE,
    AEROPARTS_BASE_5,
    _aeroparts_result,
)

CONFLICTS = ("contradiction", "exclusive")


@pytest.fixture
def scorer(monkeypatch):
    """An LCSScorer whose Merlin helper is the exact brute-force oracle."""
    monkeypatch.setattr(lcs_scorer_mod, "run_merlin", _brute_force_run_merlin)
    monkeypatch.setattr(
        lcs_scorer_mod, "DEFAULT_MAX_NETWORK_VARS", MAX_BRUTEFORCE_VARS
    )
    return LCSScorer("/fake/merlin")


def _result(relations, priors):
    """A minimal MiningResult over atoms ``a1..an`` with the given priors."""
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


def _escape(scorer, relations, priors):
    return scorer.score(_result(relations, priors), method="conflict_escape")[
        "conflict_escape"
    ]


def _mean_marginal(scorer, relations, priors):
    return scorer.score(_result(relations, priors))["mean_marginal"]


# The AeroParts coherence-ordered family of Table tab:allcands
# (worse < base < concession < fixcasualty < coherent).
AEROPARTS_FAMILY = {
    "worse": AEROPARTS_BASE_5 + [("a3", "a12", "contradiction", 0.85)],
    "base": AEROPARTS_BASE_5,
    "concession": [
        (s, t, ty, (0.55 if (s, t) == ("a11", "a12") else p))
        for (s, t, ty, p) in AEROPARTS_BASE_5
    ],
    "fixcasualty": [r for r in AEROPARTS_BASE_5 if (r[0], r[1]) != ("a7", "a10")],
    "coherent": [r for r in AEROPARTS_BASE_5 if r[2] != "exclusive"],
}

# The published values: (e) on the family, rounded as in the paper's table.
AEROPARTS_ESCAPE = {
    "worse": 0.1809,
    "base": 0.3210,
    "concession": 0.3879,
    "fixcasualty": 0.6000,
    "coherent": 1.0000,
}


class TestRegistration:
    def test_not_in_default_methods(self):
        """Adding (e) must not change what the 'all readouts' pipelines compute."""
        assert "conflict_escape" not in LCS_METHODS
        assert "conflict_escape" in ALL_LCS_METHODS

    def test_score_accepts_it(self, scorer):
        s = scorer.score(_aeroparts_result(AEROPARTS_BASE_5), method="conflict_escape")
        assert s["lcs"] == s["conflict_escape"]
        assert s["log_z_pen"] is not None

    def test_other_readouts_leave_it_none(self, scorer):
        s = scorer.score_all(_aeroparts_result(AEROPARTS_BASE_5))
        assert s["conflict_escape"] is None
        assert s["log_z_pen"] is None


class TestFactors:
    @pytest.mark.parametrize(
        "p, w", [(-0.1, 0.0), (0.0, 0.0), (0.3, 0.3), (0.5, 0.5), (0.95, 0.95)]
    )
    def test_penalty_weight(self, p, w):
        assert _penalty_weight(p) == pytest.approx(w)

    def test_penalty_weight_is_capped_below_one(self):
        assert 0.0 < 1.0 - _penalty_weight(1.0) < 1e-5

    def test_penalty_tables(self):
        assert _penalty_factor("contradiction", 0.6) == pytest.approx([1, 1, 1, 0.4])
        assert _penalty_factor("exclusive", 0.6) == pytest.approx([0.4, 1, 1, 0.4])

    def test_non_conflict_rejected(self):
        with pytest.raises(ValueError):
            _penalty_factor("entailment", 0.6)


class TestValues:
    def test_lone_contradiction_closed_form(self, scorer):
        """One contradiction at uniform priors: 1 - w * P0(1,1) = 1 - p/4."""
        priors = {"a1": 0.5, "a2": 0.5}
        for p in (0.3, 0.6, 0.8, 0.95):
            got = _escape(scorer, [("a1", "a2", "contradiction", p)], priors)
            assert got == pytest.approx(1.0 - p / 4.0, abs=1e-9)

    def test_no_conflicts_scores_one(self, scorer):
        priors = {"a1": 0.5, "a2": 0.5}
        assert _escape(scorer, [], priors) == 1.0
        assert _escape(scorer, [("a1", "a2", "entailment", 0.9)], priors) == 1.0

    def test_weak_contradiction_is_charged(self, scorer):
        """No threshold: a weak conflict costs something, so resolving it registers.

        The old threshold map w = max(0, 2p - 1) charged nothing below p = 1/2, which
        left (e) tied on every LoCoBench pair resolving a gold "weak" conflict
        (p = 0.47).
        """
        priors = {"a1": 0.5, "a2": 0.5}
        weak = _escape(scorer, [("a1", "a2", "contradiction", 0.47)], priors)
        assert weak < 1.0
        assert _escape(scorer, [], priors) > weak

    @pytest.mark.parametrize("variant", list(AEROPARTS_FAMILY))
    def test_aeroparts_family(self, scorer, variant):
        got = _escape(
            scorer,
            AEROPARTS_FAMILY[variant],
            {f"a{i}": 0.5 for i in range(1, 17)},
        )
        assert got == pytest.approx(AEROPARTS_ESCAPE[variant], abs=1e-4)

    def test_aeroparts_family_is_strictly_ordered(self):
        values = [AEROPARTS_ESCAPE[v] for v in AEROPARTS_FAMILY]
        assert values == sorted(values) and len(set(values)) == len(values)

    def test_shares_ceiling_with_log_partition(self, scorer):
        """Asked together, (d) and (e) agree on log Z0 and give the same values."""
        res = _aeroparts_result(AEROPARTS_BASE_5)
        both = scorer.score_all(res, methods=("log_partition", "conflict_escape"))
        alone = scorer.score(res, method="conflict_escape")
        assert both["conflict_escape"] == pytest.approx(alone["conflict_escape"])
        assert both["log_z_max"] == pytest.approx(alone["log_z_max"])


class TestMonotonicity:
    """The property (a) and (d) lack: removing or softening a conflict never hurts."""

    def test_star_counterexample_to_mean_marginal(self, scorer):
        """Prop. prop-nonmono: the deletion that lowers (a) raises (e)."""
        priors = {f"a{i}": 0.5 for i in range(1, 6)}
        spokes = [("a2", f"a{j}", "contradiction", 0.95) for j in (3, 4, 5)]
        blocker = [("a1", "a2", "contradiction", 0.95)]
        assert _mean_marginal(scorer, spokes, priors) < _mean_marginal(
            scorer, spokes + blocker, priors
        )
        assert _escape(scorer, spokes, priors) > _escape(
            scorer, spokes + blocker, priors
        )

    def test_random_networks(self, scorer):
        rng = random.Random(0)
        checked = 0
        for _ in range(150):
            n = rng.randint(3, 6)
            uniform = rng.random() < 0.5
            priors = {
                f"a{i}": (0.5 if uniform else rng.uniform(0.1, 0.9))
                for i in range(1, n + 1)
            }
            relations = []
            for _ in range(rng.randint(2, 6)):
                s, t = rng.sample(range(1, n + 1), 2)
                relations.append(
                    (
                        f"a{s}",
                        f"a{t}",
                        rng.choice(("contradiction", "exclusive", "entailment")),
                        rng.uniform(0.0, 0.99),
                    )
                )
            idx = [i for i, r in enumerate(relations) if r[2] in CONFLICTS]
            if not idx:
                continue
            k = rng.choice(idx)
            s, t, ty, p = relations[k]
            base = _escape(scorer, relations, priors)
            removed = _escape(scorer, relations[:k] + relations[k + 1 :], priors)
            softened = _escape(
                scorer,
                relations[:k] + [(s, t, ty, rng.uniform(0.0, p))] + relations[k + 1 :],
                priors,
            )
            assert removed >= base - 1e-12
            assert softened >= base - 1e-12
            checked += 1
        assert checked > 100


class TestInferenceCount:
    def test_shares_the_conflict_free_pr(self, monkeypatch):
        """All readouts: 9 Merlin runs -- (e) and (f) each add one PR to the 7 of (a)-(d)."""
        tasks = []

        def counting(network, merlin_path, **kwargs):
            task = kwargs.get("task", "MAR")
            tasks.append(task)
            if task == "MAR":
                return {
                    "marginals": [
                        {"variable": v, "probabilities": [0.5, 0.5]}
                        for v in kwargs.get("query_variables") or network.nodes
                    ]
                }
            return {"log_z": -9.0}

        monkeypatch.setattr(lcs_scorer_mod, "run_merlin", counting)
        s = LCSScorer("/fake/merlin")
        # The 3-coupling base: with contradiction edges present, consistency runs
        # its conflict MAR too, so the count matches the 7 of test_lcs_scorer_priors.
        s.score_all(_aeroparts_result(AEROPARTS_BASE), methods=ALL_LCS_METHODS)
        assert len(tasks) == 9, tasks
        # base, conflict-free Z0, penalty Z_pen, balanced
        assert tasks.count("PR") == 4
        tasks.clear()
        s.score(_aeroparts_result(AEROPARTS_BASE_5), method="conflict_escape")
        assert tasks == ["MAR", "PR", "PR", "PR"], tasks
