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

"""Atom prior sources for the coherence MRF (no LLM, no Merlin, no retrieval).

The interesting case throughout is alignment. Both pipelines mint atom ids
``a0, a1, ...``, but ``remove_duplicated_atoms`` drops duplicates keeping the
first-seen key, so a factuality run's surviving ids can be sparse while a fresh
atomization's are dense -- meaning the same id can name DIFFERENT text in the two
stages. Matching on id alone would then attach a prior to the wrong atom, which is
worse than attaching none, so text leads and ids only fill gaps.
"""

import json

import pytest

from fact_reasoner.core.base import Atom
from fact_reasoner.lcs.priors import (
    NEUTRAL_PRIOR,
    AtomPriors,
    FactReasonerPriorProvider,
    PrecomputedPriorProvider,
    UniformPriorProvider,
    atom_priors_from_results,
    coerce_prior_provider,
)


def _atoms(**id_to_text):
    return {aid: Atom(id=aid, text=text) for aid, text in id_to_text.items()}


class TestAtomPriorsFromResults:
    def test_reads_factuality_score_per_atom(self):
        """The shape assessor.score() actually builds."""
        results = {
            "factuality_score_per_atom": [
                {"a0": {"score": 0.91, "support": "S"}},
                {"a1": {"score": 0.12, "support": "NS"}},
            ],
            "factuality_score": 0.5,
        }
        assert atom_priors_from_results(results) == {"a0": 0.91, "a1": 0.12}

    def test_reads_merlin_marginals(self):
        results = {
            "marginals": [
                {"variable": "a0", "probabilities": [0.09, 0.91]},
                {"variable": "a1", "probabilities": [0.88, 0.12]},
            ]
        }
        assert atom_priors_from_results(results) == {"a0": 0.91, "a1": 0.12}

    def test_prefers_per_atom_over_marginals_when_both_present(self):
        results = {
            "factuality_score_per_atom": [{"a0": {"score": 0.7, "support": "S"}}],
            "marginals": [{"variable": "a0", "probabilities": [0.9, 0.1]}],
        }
        assert atom_priors_from_results(results) == {"a0": 0.7}

    def test_reads_a_bare_mapping(self):
        assert atom_priors_from_results({"a0": 0.8, "a3": 0.2}) == {"a0": 0.8, "a3": 0.2}

    def test_ignores_scalar_metrics_that_are_not_atom_ids(self):
        """A results dict's own metrics must not be mistaken for atom priors."""
        results = {"factuality_score": 0.75, "num_atoms": 12, "avg_prob": 0.6}
        assert atom_priors_from_results(results) == {}

    def test_empty_and_unrelated_inputs(self):
        assert atom_priors_from_results({}) == {}
        assert atom_priors_from_results({"topic": "x", "query": "y"}) == {}


class TestResolve:
    def test_identity_short_circuit(self):
        atoms = _atoms(a0="alpha", a1="beta")
        ap = AtomPriors(priors={"a0": 0.9, "a1": 0.2}, atoms=atoms, source="factreasoner")
        node_priors, cov = ap.resolve(atoms)
        assert node_priors == {"a0": 0.9, "a1": 0.2}
        assert cov["alignment"] == "identity"
        assert cov["n_matched_by_id"] == 2
        assert cov["n_defaulted"] == 0
        assert cov["coverage"] == pytest.approx(1.0)

    def test_id_match_without_atom_text(self):
        ap = AtomPriors(priors={"a0": 0.9, "a1": 0.2}, source="precomputed")
        node_priors, cov = ap.resolve(_atoms(a0="alpha", a1="beta"))
        assert node_priors == {"a0": 0.9, "a1": 0.2}
        assert cov["alignment"] == "id"
        assert cov["n_matched_by_id"] == 2

    def test_text_wins_over_id_when_indices_shifted(self):
        """The remove_duplicated_atoms case: the same id names different text.

        Stage 1 kept a0/a1/a3 (a2 was a duplicate). Stage 2 re-atomized densely to
        a0/a1/a2. Matching by id would give stage-2's "gamma" the prior mined for
        stage-1's "beta-dup"; matching by text puts each prior on its own claim.
        """
        stage1 = _atoms(a0="alpha", a1="beta", a3="gamma")
        ap = AtomPriors(
            priors={"a0": 0.9, "a1": 0.5, "a3": 0.1},
            atoms=stage1,
            source="factreasoner",
        )
        stage2 = _atoms(a0="alpha", a1="beta", a2="gamma")
        node_priors, cov = ap.resolve(stage2)
        assert node_priors == {"a0": 0.9, "a1": 0.5, "a2": 0.1}
        assert cov["n_matched_by_text"] == 3
        assert cov["alignment"] == "text"

    def test_text_matching_tolerates_formatting_differences(self):
        ap = AtomPriors(
            priors={"a0": 0.8},
            atoms=_atoms(a0="The  stock fell 15%."),
            source="factreasoner",
        )
        node_priors, cov = ap.resolve(_atoms(a9="the stock fell 15%"))
        assert node_priors == {"a9": 0.8}
        assert cov["n_matched_by_text"] == 1

    def test_id_fills_gaps_the_text_pass_missed(self):
        ap = AtomPriors(
            priors={"a0": 0.9, "a1": 0.3},
            atoms=_atoms(a0="alpha", a1="beta"),
            source="factreasoner",
        )
        # a0 matches by text; a1's text was rewritten, so only its id can place it.
        node_priors, cov = ap.resolve(_atoms(a0="alpha", a1="beta rewritten"))
        assert node_priors == {"a0": 0.9, "a1": 0.3}
        assert cov["n_matched_by_text"] == 1
        assert cov["n_matched_by_id"] == 1
        assert cov["alignment"] == "text+id"

    def test_uncovered_atoms_take_the_neutral_default(self):
        ap = AtomPriors(priors={"a0": 0.9}, source="precomputed")
        node_priors, cov = ap.resolve(_atoms(a0="alpha", a1="beta", a2="gamma"))
        assert node_priors == {"a0": 0.9, "a1": NEUTRAL_PRIOR, "a2": NEUTRAL_PRIOR}
        assert cov["n_defaulted"] == 2
        assert cov["coverage"] == pytest.approx(1 / 3)
        assert cov["degraded"] is True

    def test_low_coverage_raise_policy(self):
        ap = AtomPriors(priors={"a0": 0.9}, source="precomputed")
        with pytest.raises(ValueError, match="coverage"):
            ap.resolve(
                _atoms(a0="alpha", a1="beta", a2="gamma"), on_low_coverage="raise"
            )

    def test_low_coverage_uniform_policy_discards_all_priors(self):
        """Better a clean coherence-only score than a half-primed mixture."""
        ap = AtomPriors(priors={"a0": 0.9}, source="precomputed")
        node_priors, cov = ap.resolve(
            _atoms(a0="alpha", a1="beta", a2="gamma"), on_low_coverage="uniform"
        )
        assert set(node_priors.values()) == {NEUTRAL_PRIOR}
        assert cov["alignment"] == "uniform"
        assert cov["n_defaulted"] == 3

    def test_no_priors_resolves_to_uniform(self):
        node_priors, cov = AtomPriors().resolve(_atoms(a0="alpha", a1="beta"))
        assert node_priors == {"a0": 0.5, "a1": 0.5}
        assert cov["alignment"] == "uniform"

    def test_no_atoms(self):
        node_priors, cov = AtomPriors(priors={"a0": 0.9}).resolve({})
        assert node_priors == {}
        assert cov["n_atoms"] == 0

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError, match="on_low_coverage"):
            AtomPriors().resolve(_atoms(a0="x"), on_low_coverage="bogus")

    def test_carries_provider_degradation_into_the_report(self):
        ap = AtomPriors(
            priors={},
            source="uniform",
            coverage={"degraded": True, "degraded_reason": "factreasoner_early_exit"},
        )
        _priors, cov = ap.resolve(_atoms(a0="alpha"))
        assert cov["degraded"] is True
        assert cov["degraded_reason"] == "factreasoner_early_exit"


class TestUniformProvider:
    def test_resolves_to_the_configured_prior(self):
        ap = UniformPriorProvider().priors_for(response="anything")
        node_priors, cov = ap.resolve(_atoms(a0="alpha", a1="beta"))
        assert node_priors == {"a0": 0.5, "a1": 0.5}
        assert cov["degraded"] is False
        assert ap.source == "uniform"

    def test_custom_prior(self):
        ap = UniformPriorProvider(0.7).priors_for(response="x")
        node_priors, _cov = ap.resolve(_atoms(a0="alpha"))
        assert node_priors == {"a0": 0.7}


class TestPrecomputedProvider:
    def test_from_a_mapping(self):
        ap = PrecomputedPriorProvider({"a0": 0.9, "a1": 0.2}).priors_for(response="x")
        assert ap.priors == {"a0": 0.9, "a1": 0.2}
        assert ap.source == "precomputed"

    def test_from_a_results_file_with_atom_text(self, tmp_path):
        path = tmp_path / "results.json"
        path.write_text(
            json.dumps(
                {
                    "factuality_score_per_atom": [
                        {"a0": {"score": 0.91, "support": "S"}},
                        {"a1": {"score": 0.12, "support": "NS"}},
                    ],
                    "atoms": [
                        {"id": "a0", "text": "alpha"},
                        {"id": "a1", "text": "beta"},
                    ],
                }
            )
        )
        ap = PrecomputedPriorProvider(str(path)).priors_for(response="x")
        assert ap.source == "file"
        assert ap.priors == {"a0": 0.91, "a1": 0.12}
        # The lifted text lets a differently-numbered atom set still align.
        node_priors, cov = ap.resolve(_atoms(a5="alpha", a6="beta"))
        assert node_priors == {"a5": 0.91, "a6": 0.12}
        assert cov["n_matched_by_text"] == 2

    def test_file_without_priors_raises(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"topic": "nothing here"}))
        with pytest.raises(ValueError, match="No atom priors"):
            PrecomputedPriorProvider(str(path))


class _FakePipeline:
    """Minimal stand-in for a FactReasoner instance."""

    def __init__(self, atoms=None, fact_graph=object(), early_exit=None, timing=None):
        self.atoms = atoms or {}
        self.fact_graph = fact_graph
        self.early_exit_evaluation = early_exit
        self.timing = timing or {}


class _FakeRunner:
    """Stand-in for FactualityRunner exposing the two pipeline-returning hooks."""

    def __init__(self, results, pipeline):
        self.results = results
        self.pipeline = pipeline
        self.calls = []

    def assess_with_pipeline(self, query, response, topic=None, output_file=None):
        self.calls.append(("assess", query, response, topic))
        return self.results, self.pipeline

    def assess_item_with_pipeline(self, item):
        self.calls.append(("item", item))
        return self.results, self.pipeline


class TestFactReasonerProvider:
    def _results(self):
        return {
            "factuality_score_per_atom": [
                {"a0": {"score": 0.93, "support": "S"}},
                {"a1": {"score": 0.10, "support": "NS"}},
            ],
            "factuality_score": 0.5,
            "num_atoms": 2,
            "elapsed_time": 1.25,
        }

    def test_assess_mode_returns_marginals_and_atoms(self):
        atoms = _atoms(a0="alpha", a1="beta")
        runner = _FakeRunner(self._results(), _FakePipeline(atoms=atoms))
        ap = FactReasonerPriorProvider(runner=runner).priors_for(
            response="resp", query="q", topic="t"
        )
        assert ap.source == "factreasoner"
        assert ap.priors == {"a0": 0.93, "a1": 0.10}
        assert ap.atoms is atoms  # reusable: the identity path
        assert ap.diagnostics["factuality_score"] == 0.5
        assert ap.diagnostics["elapsed_time"] == 1.25
        assert runner.calls == [("assess", "q", "resp", "t")]

    def test_file_item_mode_uses_the_item_hook(self):
        atoms = _atoms(a0="alpha")
        runner = _FakeRunner(self._results(), _FakePipeline(atoms=atoms))
        item = {"input": "q", "output": "resp", "atoms": [], "contexts": []}
        ap = FactReasonerPriorProvider(
            runner=runner, mode="file_item", item=item
        ).priors_for(response="resp")
        assert ap.priors == {"a0": 0.93, "a1": 0.10}
        assert runner.calls == [("item", item)]

    def test_early_exit_degrades_to_uniform_but_keeps_atoms(self):
        """An early-exited run has no graph; scoring it would assert. Degrade."""
        atoms = _atoms(a0="alpha", a1="beta")
        pipeline = _FakePipeline(
            atoms=atoms,
            fact_graph=None,
            early_exit={"continue_pipeline_execution": False, "risk": "high"},
        )
        runner = _FakeRunner(None, pipeline)
        ap = FactReasonerPriorProvider(runner=runner).priors_for(response="resp")

        assert ap.priors == {}
        assert ap.source == "uniform"
        assert ap.atoms is atoms  # the atomize-once saving survives
        assert ap.coverage["degraded"] is True
        assert ap.coverage["degraded_reason"] == "factreasoner_early_exit"
        assert ap.diagnostics["early_exit_evaluation"]["risk"] == "high"

        node_priors, cov = ap.resolve(atoms)
        assert set(node_priors.values()) == {NEUTRAL_PRIOR}
        assert cov["degraded"] is True

    def test_degraded_raise_policy(self):
        runner = _FakeRunner(None, _FakePipeline(atoms=_atoms(a0="x"), fact_graph=None))
        provider = FactReasonerPriorProvider(runner=runner, on_degraded="raise")
        with pytest.raises(RuntimeError, match="no atom marginals"):
            provider.priors_for(response="resp")

    def test_textless_atoms_are_not_offered_for_reuse(self):
        """from_fact_graph rebuilds atoms with empty text -- useless for alignment."""
        blank = {"a0": Atom(id="a0", text=""), "a1": Atom(id="a1", text="")}
        runner = _FakeRunner(self._results(), _FakePipeline(atoms=blank))
        ap = FactReasonerPriorProvider(runner=runner).priors_for(response="resp")
        assert ap.atoms is None
        # Ids still align, which is all this mode promises.
        node_priors, cov = ap.resolve(_atoms(a0="alpha", a1="beta"))
        assert node_priors == {"a0": 0.93, "a1": 0.10}
        assert cov["alignment"] == "id"

    def test_argument_validation(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            FactReasonerPriorProvider(runner=_FakeRunner({}, None), mode="bogus")
        with pytest.raises(ValueError, match="requires runner"):
            FactReasonerPriorProvider()
        with pytest.raises(ValueError, match="requires item"):
            FactReasonerPriorProvider(runner=_FakeRunner({}, None), mode="file_item")
        with pytest.raises(ValueError, match="requires fact_graph"):
            FactReasonerPriorProvider(
                pipeline=_FakePipeline(), mode="fact_graph"
            )


class TestCoerce:
    def test_none_is_uniform(self):
        ap = coerce_prior_provider(None).priors_for(response="x")
        assert ap.source == "uniform"
        assert ap.priors == {}

    def test_float(self):
        ap = coerce_prior_provider(0.8).priors_for(response="x")
        node_priors, _ = ap.resolve(_atoms(a0="alpha"))
        assert node_priors == {"a0": 0.8}

    def test_mapping(self):
        ap = coerce_prior_provider({"a0": 0.3}).priors_for(response="x")
        assert ap.priors == {"a0": 0.3}

    def test_atom_priors_passthrough(self):
        original = AtomPriors(priors={"a0": 0.4}, source="precomputed")
        assert coerce_prior_provider(original).priors_for(response="x") is original

    def test_provider_passthrough(self):
        provider = UniformPriorProvider(0.6)
        assert coerce_prior_provider(provider) is provider

    def test_rejects_nonsense(self):
        with pytest.raises(TypeError):
            coerce_prior_provider("not priors")
