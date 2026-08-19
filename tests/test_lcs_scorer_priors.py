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

"""Per-atom priors and shared inference in the LCS scorer (no LLM, no Merlin).

Two things are pinned down here.

**Uniform invariance.** Making the priors per-atom must not move the validated
AeroParts numbers. The miner writes the same value to the fact-graph nodes and to
``config["prior"]``, so with a uniform prior the resolved mapping is exactly what
the scorer used before; every readout must be unchanged.

**All four readouts honour the priors.** Three of them (``consistency``,
``reified``, ``log_partition``) REBUILD the network from the fact graph, so before
this they silently used a uniform prior even when the graph carried per-atom ones
-- and ``log_partition`` compared a real-prior ``log Z`` against a uniform-prior
ceiling. Each readout must now respond to skewed priors.

Plus: ``score_all`` must agree with per-method ``score`` value-for-value while
running the base MAR and base PR once instead of once per method.
"""

import json

import pytest

from fact_reasoner.core.base import Atom
from fact_reasoner.experiments.mock import (
    MAX_BRUTEFORCE_VARS,
    brute_force_run_merlin as _brute_force_run_merlin,
)
from fact_reasoner.fact_graph import FactGraph
from fact_reasoner.factors import build_markov_network
from fact_reasoner.lcs import lcs_scorer as lcs_scorer_mod
from fact_reasoner.lcs.lcs_scorer import LCS_METHODS, LCSScorer
from fact_reasoner.lcs.relation_miner import MiningResult, RelationMiner

# The deep-dive AeroParts fixture. Imported rather than copied so this file and
# test_lcs_relation_miner can never disagree about the validated numbers.
from tests.test_lcs_relation_miner import (  # noqa: E402
    AEROPARTS_BASE,
    AEROPARTS_IDS,
    _aeroparts_result,
)


def _result(relations=AEROPARTS_BASE, prior=0.5):
    """An AeroParts MiningResult built exactly as RelationMiner would build it."""
    return _aeroparts_result(relations, prior=prior)


@pytest.fixture
def scorer(monkeypatch):
    """An LCSScorer whose Merlin helper is the exact brute-force oracle.

    The oracle enumerates 2^n worlds and refuses more than
    ``MAX_BRUTEFORCE_VARS`` variables, so the scorer's aux-var batching cap is
    lowered to match; production keeps the high default (see ``LCSScorer``).
    """
    monkeypatch.setattr(lcs_scorer_mod, "run_merlin", _brute_force_run_merlin)
    monkeypatch.setattr(
        lcs_scorer_mod, "DEFAULT_MAX_NETWORK_VARS", MAX_BRUTEFORCE_VARS
    )
    return LCSScorer("/fake/merlin")


class TestUniformInvariance:
    """The validated numbers must not move when priors become per-atom."""

    def test_explicit_uniform_priors_match_implicit(self, scorer):
        implicit = scorer.score_all(_result(), methods=LCS_METHODS)
        explicit = scorer.score_all(
            _result(),
            methods=LCS_METHODS,
            node_priors={a: 0.5 for a in AEROPARTS_IDS},
        )
        for key in LCS_METHODS:
            assert implicit[key] == pytest.approx(explicit[key], abs=1e-12), key
        for key in ("log_z", "log_z_max", "log_z_min", "avg_norm_entropy"):
            assert implicit[key] == pytest.approx(explicit[key], abs=1e-12), key

    def test_deepdive_numbers_unchanged(self, scorer):
        """The published AeroParts values, read through the new code path."""
        s = scorer.score_all(_result(), methods=LCS_METHODS)
        assert s["mean_marginal"] == pytest.approx(0.587, abs=1e-3)
        # Two-term consistency: conflict 0.813 (contradiction-only) + support 0.495.
        assert s["consistency"] == pytest.approx(0.6539, abs=1e-3)
        assert s["consistency_conflict"] == pytest.approx(0.8128, abs=1e-3)
        assert s["consistency_support"] == pytest.approx(0.4951, abs=1e-3)
        assert s["reified"] == pytest.approx(0.150, abs=2e-3)
        assert s["log_z"] == pytest.approx(-9.75, abs=0.05)
        assert s["log_z_max"] == pytest.approx(-8.25, abs=0.05)
        assert s["log_z_min"] == pytest.approx(-15.16, abs=0.05)
        assert s["log_partition"] == pytest.approx(0.7831, abs=1e-3)

    def test_scalar_prior_kwarg_still_overrides(self, scorer):
        """`prior=` applies one value to every atom, as it always did."""
        s = scorer.score(_result(), prior=0.5)
        assert s["mean_marginal"] == pytest.approx(0.587, abs=1e-3)
        assert set(s["node_priors"].values()) == {0.5}


class TestPerAtomPriors:
    """Every readout must respond to per-atom priors."""

    @pytest.mark.parametrize("method", LCS_METHODS)
    def test_readout_responds_to_skewed_priors(self, scorer, method):
        uniform = scorer.score(_result(), method=method)[method]
        # Drag one contradiction endpoint's prior far down.
        skewed_priors = {a: 0.5 for a in AEROPARTS_IDS}
        skewed_priors["a10"] = 0.05
        skewed = scorer.score(
            _result(), method=method, node_priors=skewed_priors
        )[method]
        assert skewed != pytest.approx(uniform, abs=1e-9), (
            f"readout {method!r} ignored per-atom priors "
            f"(uniform={uniform!r}, skewed={skewed!r})"
        )

    def test_priors_from_fact_graph_nodes_are_honoured(self, scorer):
        """A prior baked onto the graph nodes reaches the rebuilt networks too."""
        result = _result()
        for node in result.fact_graph.get_nodes():
            if node.id == "a10":
                node.probability = 0.05
        resolved = scorer.score(result, method="consistency")
        assert resolved["node_priors"]["a10"] == pytest.approx(0.05)
        assert resolved["consistency"] != pytest.approx(
            scorer.score(_result(), method="consistency")["consistency"], abs=1e-9
        )

    def test_explicit_priors_beat_graph_nodes(self, scorer):
        result = _result()
        for node in result.fact_graph.get_nodes():
            if node.id == "a10":
                node.probability = 0.05
        s = scorer.score(result, node_priors={"a10": 0.9})
        assert s["node_priors"]["a10"] == pytest.approx(0.9)

    def test_num_below_prior_compares_each_atom_to_its_own_prior(self, scorer):
        """The count is per-atom: q_i < pi_i, not q_i < one shared threshold.

        Two atoms with the same marginal can differ on "was it dragged down",
        because they started from different priors. Fixing one atom's prior just
        above its own posterior must add exactly that atom to the count.
        """
        base = scorer.score(_result())
        uniform_count = base["num_below_prior"]

        # a10 is the loser of the a7 -> a10 contradiction (p=0.93). Raising ONLY
        # a10's prior to 0.95 lifts its own posterior (0.237 -> 0.855) but pushes
        # its opponent a7 down (0.612 -> below 0.5): a7 is now an atom the argument
        # drags below its own prior, so the count grows by exactly one.
        target = "a10"
        assert base["marginals"][target] < 0.5  # already down at a uniform prior

        priors = {a: 0.5 for a in AEROPARTS_IDS}
        priors[target] = 0.95
        skewed = scorer.score(_result(), node_priors=priors)
        assert skewed["node_priors"][target] == pytest.approx(0.95)
        # a10 stays below its OWN 0.95 while sitting far above the 0.5 that every
        # other atom is judged against -- a distinction one shared threshold cannot
        # express.
        assert 0.5 < skewed["marginals"][target] < 0.95
        assert skewed["num_below_prior"] == uniform_count + 1

        below_before = {a for a, q in base["marginals"].items() if q < 0.5}
        below_after = {a for a, q in skewed["marginals"].items() if q < priors[a]}
        assert below_after - below_before == {"a7"}

    def test_log_partition_stays_bracketed_and_in_unit_range(self, scorer):
        """The floor/base/ceiling ordering survives non-uniform priors."""
        priors = {a: 0.5 for a in AEROPARTS_IDS}
        priors.update({"a1": 0.95, "a7": 0.1, "a10": 0.05, "a12": 0.8})
        s = scorer.score(_result(), method="log_partition", node_priors=priors)
        assert s["log_z_min"] <= s["log_z"] <= s["log_z_max"]
        assert 0.0 <= s["log_partition"] <= 1.0


class TestScoreAllSharesInference:
    def test_values_match_per_method_scores(self, scorer):
        shared = scorer.score_all(_result(), methods=LCS_METHODS)
        for m in LCS_METHODS:
            per_method = scorer.score(_result(), method=m)[m]
            assert shared[m] == pytest.approx(per_method, abs=1e-12), m

    def test_invocation_count_is_irreducible(self, monkeypatch):
        """All four readouts: 7 Merlin runs, not the 13 of per-method scoring.

        Deliberately does NOT lower ``max_network_vars``, so the consistency
        support term takes its production shape -- one batch, one MAR. (Under the
        brute-force oracle's 20-variable cap it splits into several batched MARs;
        that is a property of the offline oracle, not of the readout.) The
        single-batch support network exceeds what the oracle will enumerate, so
        this counts calls against a stub: the assertion is about how many runs
        happen, and the values are pinned by the other tests in this file.
        """
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

        s.score_all(_result(), methods=LCS_METHODS)
        shared = list(tasks)
        tasks.clear()
        for m in LCS_METHODS:
            s.score(_result(), method=m)
        per_method = list(tasks)

        assert len(shared) == 7, shared
        # 4 methods x (base MAR + base PR) = 8, plus each method's own extras:
        # mean_marginal 0, consistency 2 (conflict + support), reified 1,
        # log_partition 2 (ceiling PR + base MAP).
        assert len(per_method) == 13, per_method
        # base MAR + base PR + conflict U-chain MAR + support MAR + reified MAR
        # + ceiling PR + base MAP.
        assert shared.count("MAR") == 4
        assert shared.count("PR") == 2
        assert shared.count("MAP") == 1

    def test_single_method_is_a_valid_projection(self, scorer):
        s = scorer.score(_result(), method="consistency")
        assert s["method"] == "consistency"
        assert s["lcs"] == s["consistency"]
        # Unrequested readouts stay None, as callers expect.
        assert s["reified"] is None
        assert s["log_partition"] is None

    def test_empty_methods_and_unknown_method_raise(self, scorer):
        with pytest.raises(ValueError):
            scorer.score_all(_result(), methods=())
        with pytest.raises(ValueError):
            scorer.score_all(_result(), methods=("bogus",))
        with pytest.raises(ValueError):
            scorer.score(_result(), method="bogus")

    def test_empty_result(self, scorer):
        empty = MiningResult(
            atoms={},
            relations=[],
            fact_graph=FactGraph(),
            markov_network=build_markov_network(FactGraph()),
            coverage={},
            config={"prior": 0.5},
        )
        s = scorer.score_all(empty, methods=LCS_METHODS)
        assert s["lcs"] == 0.0
        assert s["num_atoms"] == 0
        assert s["node_priors"] == {}


class TestMinerPriorPlumbing:
    """RelationMiner accepts per-atom priors without breaking its contract."""

    def _miner(self, prior):
        """A miner with the prior attributes __init__ would have set (no backend)."""
        miner = object.__new__(RelationMiner)
        if isinstance(prior, dict):
            miner.node_priors = {k: float(v) for k, v in prior.items()}
            miner.default_prior = 0.5
        else:
            miner.node_priors = None
            miner.default_prior = float(prior)
        miner.prior = miner.default_prior
        return miner

    def test_mapping_prior_reaches_the_fact_graph(self):
        atoms = {a: Atom(id=a, text=f"atom {a}") for a in ("a0", "a1", "a2")}
        miner = self._miner({"a0": 0.9, "a1": 0.1})
        fg = miner._build_fact_graph(atoms, [])
        got = {n.id: n.probability for n in fg.get_nodes()}
        assert got == {"a0": 0.9, "a1": 0.1, "a2": 0.5}  # a2 falls back

    def test_float_prior_applies_to_every_atom(self):
        atoms = {a: Atom(id=a, text=f"atom {a}") for a in ("a0", "a1")}
        miner = self._miner(0.3)
        fg = miner._build_fact_graph(atoms, [])
        assert {n.probability for n in fg.get_nodes()} == {0.3}

    def test_per_call_priors_win_over_the_miners_own(self):
        atoms = {a: Atom(id=a, text=f"atom {a}") for a in ("a0", "a1")}
        miner = self._miner({"a0": 0.9})
        fg = miner._build_fact_graph(atoms, [], {"a0": 0.2, "a1": 0.4})
        got = {n.id: n.probability for n in fg.get_nodes()}
        assert got == {"a0": 0.2, "a1": 0.4}

    def test_config_stays_json_serializable(self):
        """`config["prior"]` must remain a float (LCSScorer does float(...) on it)."""
        result = _result()
        assert isinstance(result.config["prior"], float)
        json.dumps(result.to_json())  # must not raise
