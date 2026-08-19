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

"""The two-stage coherence pipeline and the MLN scaffold (no LLM, no Merlin).

Covers the orchestration (priors -> mining -> readouts), the atom-reuse invariant
that keeps the response atomized once, and the parts of the MLN formulation that
are genuinely implemented -- above all that its three-clause expansion reproduces
the with-priors MRF factor tables exactly, which is the deep-dive's claim that the
MLN's pairwise fragment *is* the MRF.
"""

import itertools
import json
import math

import pytest

from fact_reasoner.core.base import Atom
from fact_reasoner.experiments.mock import (
    MAX_BRUTEFORCE_VARS,
    brute_force_marginals,
    brute_force_run_merlin as _brute_force_run_merlin,
)
from fact_reasoner.factors import build_markov_network, edge_factor_values
from fact_reasoner.fact_graph import Edge, FactGraph, Node
from fact_reasoner.lcs import lcs_scorer as lcs_scorer_mod
from fact_reasoner.lcs.lcs_scorer import LCS_METHODS, LCSScorer
from fact_reasoner.lcs.pipeline import (
    COHERENCE_FORMULATIONS,
    RULE_SCHEMA,
    CoherencePipeline,
    MLNCoherenceModel,
    MRFCoherenceModel,
    build_coherence_model,
    mln_weight,
    three_clause_weights,
)
from fact_reasoner.lcs.priors import AtomPriors

from tests.test_lcs_relation_miner import (  # noqa: E402
    AEROPARTS_BASE,
    AEROPARTS_IDS,
    _aeroparts_result,
)


@pytest.fixture
def fake_merlin(monkeypatch):
    """Route the scorer's Merlin helper to the exact brute-force oracle.

    Also lowers the aux-var batching cap to the oracle's own 2^n limit, so the
    consistency support term fits; production keeps the high default.
    """
    monkeypatch.setattr(lcs_scorer_mod, "run_merlin", _brute_force_run_merlin)
    monkeypatch.setattr(
        lcs_scorer_mod, "DEFAULT_MAX_NETWORK_VARS", MAX_BRUTEFORCE_VARS
    )


class _StubMiner:
    """A RelationMiner stand-in that records how it was called."""

    def __init__(self, result):
        self.result = result
        self.from_atoms_calls = 0
        self.from_response_calls = 0
        self.last_atoms = None

    def mine_from_atoms(self, atoms, response, *, node_priors=None):
        self.from_atoms_calls += 1
        self.last_atoms = atoms
        return self.result

    def mine_from_response(self, response, *, query=None, node_priors=None):
        self.from_response_calls += 1
        return self.result

    async def amine_from_atoms(self, atoms, response, *, node_priors=None):
        return self.mine_from_atoms(atoms, response)

    async def amine_from_response(self, response, *, query=None, node_priors=None):
        return self.mine_from_response(response)


class _StubProvider:
    def __init__(self, atom_priors):
        self.atom_priors = atom_priors
        self.calls = 0

    def priors_for(self, *, response, query=None, topic=None):
        self.calls += 1
        return self.atom_priors


def _pipeline(miner, provider=None, methods=("mean_marginal",), **kwargs):
    return CoherencePipeline(
        miner=miner,
        merlin_path="/fake/merlin",
        prior_provider=provider,
        methods=methods,
        **kwargs,
    )


class TestCoherencePipeline:
    def test_uniform_priors_reproduce_the_plain_scorer(self, fake_merlin):
        """No priors => the coherence-only number, unchanged."""
        result = _aeroparts_result(AEROPARTS_BASE)
        out = _pipeline(_StubMiner(result)).run("some response")
        assert out.lcs == pytest.approx(0.587, abs=1e-3)
        assert out.method == "mean_marginal"
        assert out.formulation == "mrf"
        assert set(out.priors.values()) == {0.5}

    def test_low_priors_drag_the_score_down(self, fake_merlin):
        result = _aeroparts_result(AEROPARTS_BASE)
        baseline = _pipeline(_StubMiner(result)).run("resp").lcs

        provider = _StubProvider(
            AtomPriors(
                priors={a: 0.1 for a in AEROPARTS_IDS},
                source="factreasoner",
            )
        )
        primed = _pipeline(_StubMiner(result), provider).run("resp")
        assert primed.lcs < baseline
        assert primed.prior_coverage["n_matched_by_id"] == len(AEROPARTS_IDS)
        assert primed.prior_coverage["coverage"] == pytest.approx(1.0)

    def test_high_priors_lift_the_score(self, fake_merlin):
        result = _aeroparts_result(AEROPARTS_BASE)
        baseline = _pipeline(_StubMiner(result)).run("resp").lcs
        provider = _StubProvider(
            AtomPriors(priors={a: 0.9 for a in AEROPARTS_IDS}, source="factreasoner")
        )
        assert _pipeline(_StubMiner(result), provider).run("resp").lcs > baseline

    def test_stage_one_atoms_are_reused_not_reatomized(self, fake_merlin):
        """The atomize-once invariant: mining takes stage 1's atoms directly."""
        result = _aeroparts_result(AEROPARTS_BASE)
        stage1_atoms = {a: Atom(id=a, text=f"atom {a}") for a in AEROPARTS_IDS}
        provider = _StubProvider(
            AtomPriors(
                priors={a: 0.8 for a in AEROPARTS_IDS},
                atoms=stage1_atoms,
                source="factreasoner",
            )
        )
        miner = _StubMiner(result)
        _pipeline(miner, provider).run("resp")

        assert miner.from_atoms_calls == 1
        assert miner.from_response_calls == 0  # never re-atomized
        assert miner.last_atoms is stage1_atoms

    def test_atomizes_when_the_provider_has_no_atoms(self, fake_merlin):
        miner = _StubMiner(_aeroparts_result(AEROPARTS_BASE))
        _pipeline(miner, _StubProvider(AtomPriors(priors={"a1": 0.9}))).run("resp")
        assert miner.from_response_calls == 1
        assert miner.from_atoms_calls == 0

    def test_textless_atoms_are_not_reused(self, fake_merlin):
        """from_fact_graph's blank atoms must not be mined in place of real ones."""
        blank = {a: Atom(id=a, text="") for a in AEROPARTS_IDS}
        provider = _StubProvider(
            AtomPriors(priors={a: 0.7 for a in AEROPARTS_IDS}, atoms=blank)
        )
        miner = _StubMiner(_aeroparts_result(AEROPARTS_BASE))
        _pipeline(miner, provider).run("resp")
        assert miner.from_response_calls == 1
        assert miner.from_atoms_calls == 0

    def test_multiple_readouts_are_all_reported(self, fake_merlin):
        out = _pipeline(
            _StubMiner(_aeroparts_result(AEROPARTS_BASE)), methods=LCS_METHODS
        ).run("resp")
        assert set(out.scores) == set(LCS_METHODS)
        assert out.scores["mean_marginal"] == pytest.approx(0.587, abs=1e-3)
        assert out.scores["consistency"] == pytest.approx(0.6539, abs=1e-3)
        assert out.lcs == out.scores[LCS_METHODS[0]]
        assert out.diagnostics["log_z"] == pytest.approx(-9.75, abs=0.05)

    def test_factuality_diagnostics_are_carried_through(self, fake_merlin):
        provider = _StubProvider(
            AtomPriors(
                priors={a: 0.6 for a in AEROPARTS_IDS},
                source="factreasoner",
                diagnostics={"factuality_score": 0.42, "elapsed_time": 3.0},
            )
        )
        out = _pipeline(_StubMiner(_aeroparts_result(AEROPARTS_BASE)), provider).run("r")
        assert out.factuality["factuality_score"] == 0.42

    def test_run_from_mining_scores_without_re_mining(self, fake_merlin):
        result = _aeroparts_result(AEROPARTS_BASE)
        miner = _StubMiner(result)
        pipeline = _pipeline(miner)
        out = pipeline.run_from_mining(result, priors={a: 0.2 for a in AEROPARTS_IDS})
        assert miner.from_atoms_calls == 0
        assert miner.from_response_calls == 0
        assert out.lcs < 0.587

    def test_run_from_mining_defaults_to_the_graphs_own_priors(self, fake_merlin):
        result = _aeroparts_result(AEROPARTS_BASE)
        out = _pipeline(_StubMiner(result)).run_from_mining(result)
        assert out.lcs == pytest.approx(0.587, abs=1e-3)

    def test_arun_matches_run(self, fake_merlin):
        """arun is the same computation, driven through asyncio.run."""
        import asyncio

        result = _aeroparts_result(AEROPARTS_BASE)
        sync = _pipeline(_StubMiner(result)).run("resp")
        got = asyncio.run(_pipeline(_StubMiner(result)).arun("resp"))
        assert got.lcs == pytest.approx(sync.lcs, abs=1e-12)

    def test_result_is_json_serializable(self, fake_merlin):
        out = _pipeline(
            _StubMiner(_aeroparts_result(AEROPARTS_BASE)), methods=LCS_METHODS
        ).run("resp")
        payload = out.to_json()
        json.dumps(payload)  # must not raise
        assert payload["mining"]["config"]["prior"] == 0.5
        assert payload["formulation"] == "mrf"

    def test_describe_runs(self, fake_merlin):
        out = _pipeline(_StubMiner(_aeroparts_result(AEROPARTS_BASE))).run("resp")
        text = out.describe()
        assert "LCS" in text and "priors" in text

    def test_timing_is_recorded_per_stage(self, fake_merlin):
        out = _pipeline(_StubMiner(_aeroparts_result(AEROPARTS_BASE))).run("resp")
        assert set(out.timing) >= {"priors", "mining", "scoring", "total"}

    def test_rejects_bad_methods(self):
        miner = _StubMiner(_aeroparts_result(AEROPARTS_BASE))
        with pytest.raises(ValueError):
            _pipeline(miner, methods=())
        with pytest.raises(ValueError):
            _pipeline(miner, methods=("bogus",))

    def test_accepts_a_bare_mapping_as_the_prior_source(self, fake_merlin):
        out = _pipeline(
            _StubMiner(_aeroparts_result(AEROPARTS_BASE)),
            {a: 0.15 for a in AEROPARTS_IDS},
        ).run("resp")
        assert out.lcs < 0.587
        assert out.prior_coverage["source"] == "precomputed"

    def test_low_coverage_policy_is_honoured(self, fake_merlin):
        provider = _StubProvider(AtomPriors(priors={"a1": 0.9}, source="precomputed"))
        with pytest.raises(ValueError, match="coverage"):
            _pipeline(
                _StubMiner(_aeroparts_result(AEROPARTS_BASE)),
                provider,
                on_low_coverage="raise",
            ).run("resp")


class TestModelSelector:
    def test_default_is_mrf(self):
        model = build_coherence_model(merlin_path="/fake/merlin")
        assert isinstance(model, MRFCoherenceModel)
        assert model.formulation == "mrf"

    def test_mln_constructs_but_does_not_score(self):
        model = build_coherence_model("mln")
        assert isinstance(model, MLNCoherenceModel)
        assert model.formulation == "mln"
        with pytest.raises(NotImplementedError, match="coherence_mln_deepdive"):
            model.score(_aeroparts_result(AEROPARTS_BASE))

    def test_mln_placeholder_methods_all_point_at_the_doc(self):
        model = build_coherence_model("mln")
        for call in (
            lambda: model.ground(_aeroparts_result(AEROPARTS_BASE)),
            lambda: model.learn_rule_weights([]),
        ):
            with pytest.raises(NotImplementedError, match="coherence_mln_deepdive"):
                call()

    def test_unknown_formulation_raises(self):
        with pytest.raises(ValueError, match="Unknown coherence formulation"):
            build_coherence_model("psl")

    def test_mrf_requires_a_merlin_path(self):
        with pytest.raises(ValueError, match="merlin_path"):
            build_coherence_model("mrf")

    def test_formulations_tuple(self):
        assert COHERENCE_FORMULATIONS == ("mrf", "mln")

    def test_pipeline_accepts_the_mln_selector(self, fake_merlin):
        """Wiring is testable even though scoring is not implemented."""
        pipeline = _pipeline(
            _StubMiner(_aeroparts_result(AEROPARTS_BASE)), formulation="mln"
        )
        assert pipeline.coherence_model.formulation == "mln"
        with pytest.raises(NotImplementedError):
            pipeline.run("resp")


class TestMLNWeights:
    """The closed-form pairwise fragment: real, and checked against the MRF."""

    def test_mln_weight_is_the_logit(self):
        assert mln_weight(0.9) == pytest.approx(2.1972, abs=1e-3)
        assert mln_weight(0.5) == pytest.approx(0.0, abs=1e-12)
        assert mln_weight(0.1) == pytest.approx(-2.1972, abs=1e-3)

    @pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.5])
    def test_mln_weight_rejects_degenerate_p(self, p):
        with pytest.raises(ValueError):
            mln_weight(p)

    def test_three_clause_table_matches_the_deepdive(self):
        """b, c, d for the three tabulated couplings (pi_s = 0.5)."""
        p = 0.8
        logit = math.log(p / (1 - p))

        b, c, d = three_clause_weights("entailment", p)
        assert (b, c, d) == pytest.approx(
            (math.log((1 - p) / 0.5), 0.0, logit), abs=1e-12
        )

        b, c, d = three_clause_weights("contradiction", p)
        assert (b, c, d) == pytest.approx((math.log(p / 0.5), 0.0, -logit), abs=1e-12)

        b, c, d = three_clause_weights("equivalence", p)
        shared = math.log((1 - p) / p)
        assert (b, c, d) == pytest.approx((shared, shared, 2 * logit), abs=1e-12)

    @pytest.mark.parametrize("level1_type", ["entailment", "contradiction", "equivalence"])
    @pytest.mark.parametrize("p", [0.55, 0.65, 0.8, 0.93])
    def test_expansion_reproduces_the_mrf_factor_table(self, level1_type, p):
        """ln psi = a + b*a_s + c*a_t + d*a_s*a_t must recover edge_factor_values.

        This is the deep-dive's "exact MLN = MRF" claim on the pairwise fragment,
        checked rather than asserted -- and it is what makes the MLN placeholder's
        Stage-0 story trustworthy.
        """
        edge = Edge(
            source="a0", target="a1", type=level1_type, probability=p, link="atom_atom"
        )
        expected = edge_factor_values(edge, use_priors=True)
        b, c, d = three_clause_weights(level1_type, p, pi_s=0.5)
        a = math.log(expected[0])  # the constant, read off the (0,0) cell

        got = [
            math.exp(a + b * s + c * t + d * s * t)
            for s, t in itertools.product((0, 1), (0, 1))
        ]
        assert got == pytest.approx(expected, abs=1e-12)

    def test_expansion_reproduces_the_mrf_marginals(self):
        """The same equality at the level a score actually reads: the marginals."""
        p_ent, p_con = 0.8, 0.93
        fg = FactGraph()
        for aid in ("a0", "a1", "a2"):
            fg.add_node(Node(id=aid, type="atom", probability=0.5))
        fg.add_edge(Edge("a0", "a1", "entailment", p_ent, "atom_atom"))
        fg.add_edge(Edge("a1", "a2", "contradiction", p_con, "atom_atom"))
        priors = {aid: 0.5 for aid in ("a0", "a1", "a2")}
        mrf = build_markov_network(fg, use_priors=True, node_priors=priors)
        mrf_marginals, mrf_log_z, _ = brute_force_marginals(mrf, priors)

        # Rebuild the same network from the MLN's three-clause weights.
        mln = build_markov_network(FactGraph(), use_priors=True)
        for aid in ("a0", "a1", "a2"):
            mln.add_node(aid)
            mln.add_factor([aid], [2], [0.5, 0.5])
        for src, trg, ty, p in (
            ("a0", "a1", "entailment", p_ent),
            ("a1", "a2", "contradiction", p_con),
        ):
            b, c, d = three_clause_weights(ty, p, pi_s=0.5)
            const = math.log(
                edge_factor_values(
                    Edge(src, trg, ty, p, "atom_atom"), use_priors=True
                )[0]
            )
            mln.add_edge(src, trg)
            mln.add_factor(
                [src, trg],
                [2, 2],
                [
                    math.exp(const + b * s + c * t + d * s * t)
                    for s, t in itertools.product((0, 1), (0, 1))
                ],
            )
        mln_marginals, mln_log_z, _ = brute_force_marginals(mln, priors)

        assert mln_log_z == pytest.approx(mrf_log_z, abs=1e-10)
        for aid in mrf_marginals:
            assert mln_marginals[aid] == pytest.approx(mrf_marginals[aid], abs=1e-10)

    def test_revised_couplings_are_explicitly_untabulated(self):
        """The deep-dive's table covers three couplings; say so, don't guess."""
        for level1_type in ("exclusive", "co_necessity"):
            with pytest.raises(NotImplementedError, match="deep-dive"):
                three_clause_weights(level1_type, 0.8)

    def test_unknown_coupling_raises(self):
        with pytest.raises(ValueError, match="Unknown Level-1 coupling"):
            three_clause_weights("nonsense", 0.8)

    def test_rule_schema_names_the_documented_rules(self):
        assert RULE_SCHEMA["query_predicate"] == "Holds(i)"
        assert "Resolves(h,i,j)" in RULE_SCHEMA["evidence_predicates"]
        assert set(RULE_SCHEMA["pairwise_rules"]) == {
            "entail", "contradict", "equiv"
        }
        # The three weights the deep-dive says need learning (not a closed form).
        assert RULE_SCHEMA["learned_weights"] == ("w_t", "w_r", "w_d")
        assert set(RULE_SCHEMA["beyond_pairwise_rules"]) == {"w_t", "w_r", "w_d"}


class TestMRFCoherenceModel:
    def test_delegates_to_score_all(self, fake_merlin):
        model = MRFCoherenceModel("/fake/merlin")
        scores = model.score(
            _aeroparts_result(AEROPARTS_BASE), methods=("mean_marginal", "consistency")
        )
        assert scores["mean_marginal"] == pytest.approx(0.587, abs=1e-3)
        assert scores["consistency"] == pytest.approx(0.6539, abs=1e-3)

    def test_accepts_a_prebuilt_scorer(self, fake_merlin):
        model = MRFCoherenceModel(scorer=LCSScorer("/fake/merlin"))
        assert model.score(_aeroparts_result(AEROPARTS_BASE))["mean_marginal"] > 0

    def test_passes_node_priors_through(self, fake_merlin):
        model = MRFCoherenceModel("/fake/merlin")
        low = model.score(
            _aeroparts_result(AEROPARTS_BASE),
            node_priors={a: 0.1 for a in AEROPARTS_IDS},
        )
        assert low["mean_marginal"] < 0.587
