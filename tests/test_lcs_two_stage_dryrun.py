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

"""End-to-end two-stage run, fully offline (no LLM, no Merlin, no retrieval).

Both stages are the real code: a real ``FactReasoner`` computes real posterior
marginals over atoms and contexts, those become the coherence MRF's per-atom
priors, and a real ``LCSScorer`` reads the score off it. Only the leaves are
stubbed -- the LLM (mock NLI extractor / mocked ``ainstruct``) and Merlin (the
exact brute-force oracle).

Note the brute-force oracle refuses networks above ``MAX_BRUTEFORCE_VARS`` (20)
variables, and a FACTUALITY network counts contexts as well as atoms. The fixture
here is deliberately tiny (2 atoms, 2 contexts); do not grow it.
"""

import asyncio
import json

import pytest

from fact_reasoner.assessor import FactReasoner
from fact_reasoner.experiments.mock import (
    MAX_BRUTEFORCE_VARS,
    brute_force_run_merlin,
    dry_run_patches,
)
from fact_reasoner.lcs.pipeline import CoherencePipeline
from fact_reasoner.lcs.priors import (
    FactReasonerPriorProvider,
    atom_priors_from_results,
)
from fact_reasoner.lcs.relation_miner import RelationMiner


class _StubAtomizer:
    """Returns a fixed atom set, so a run without stage-1 atoms is comparable.

    ``build_atoms`` only needs ``run(response)`` to return something whose
    ``.values()`` are the atom texts, in order.
    """

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def run(self, response):
        self.calls += 1
        return {str(i): text for i, text in enumerate(self.texts)}


def _verdicts_by_atom(nli, per_atom, default=None):
    """Drive the mock NLI extractor from the ATOM text (the hypothesis).

    The premise the pipeline builds is a formatted block ("Snippet/Summary of
    Text: ... Text: ..."), not the raw context string, so keying on it would pin
    this test to that formatting. The hypothesis is the atom text verbatim, and
    the atom is what we are differentiating factually anyway.

    Args:
        nli: The ``mock_nli_batch`` fixture.
        per_atom: ``{atom_text: verdict}``.
        default: Verdict for atoms not listed (defaults to neutral).
    """
    fallback = default or {"label": "neutral", "probability": 0.9}

    async def run_batch(premises, hypotheses):
        nli.calls.append((list(premises), list(hypotheses)))
        return [dict(per_atom.get(h, fallback)) for h in hypotheses]

    nli.run_batch = run_batch
    return nli


def _build_factuality(json_data, nli, summarizer):
    """A real FactReasoner over a pre-annotated item (no retrieval, no atomizing)."""
    pipeline = FactReasoner(
        nli_extractor=nli,
        context_summarizer=summarizer,
        merlin_path="/fake/merlin",
        use_priors=True,
    )
    pipeline.from_dict_with_contexts(json_data)
    asyncio.run(
        pipeline.build(
            has_atoms=True,
            has_contexts=True,
            revise_atoms=False,
            summarize_contexts=False,
        )
    )
    return pipeline


class _ItemRunner:
    """A FactualityRunner stand-in wrapping one real pre-built FactReasoner."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def assess_item_with_pipeline(self, item):
        results, _marginals = self.pipeline.score()
        return results, self.pipeline


class TestTwoStageDryRun:
    def test_factuality_posteriors_become_coherence_priors(
        self, sample_json_data, mock_nli_batch, mock_summarizer, monkeypatch
    ):
        """The whole point: stage 1's marginals are stage 2's unary priors."""
        # Make the two atoms differ factually: a0 supported, a1 contradicted.
        atoms = {a["id"]: a["text"] for a in sample_json_data["atoms"]}
        _verdicts_by_atom(
            mock_nli_batch,
            {
                atoms["a0"]: {"label": "entailment", "probability": 0.95},
                atoms["a1"]: {"label": "contradiction", "probability": 0.95},
            },
        )
        monkeypatch.setattr(
            "fact_reasoner.assessor._run_merlin_shared", brute_force_run_merlin
        )

        pipeline = _build_factuality(sample_json_data, mock_nli_batch, mock_summarizer)
        # The factuality network counts contexts too; stay under the oracle's cap.
        assert len(pipeline.markov_network.nodes) <= MAX_BRUTEFORCE_VARS

        provider = FactReasonerPriorProvider(
            runner=_ItemRunner(pipeline), mode="file_item", item=sample_json_data
        )
        atom_priors = provider.priors_for(response=sample_json_data["output"])

        # Stage 1 produced real, differentiated posteriors, and exported its atoms.
        assert atom_priors.source == "factreasoner"
        assert set(atom_priors.priors) == {"a0", "a1"}
        assert atom_priors.priors["a0"] > 0.5 > atom_priors.priors["a1"]
        assert atom_priors.atoms is pipeline.atoms

        # Stage 2: the coherence MRF, primed with those posteriors.
        with dry_run_patches():
            miner = RelationMiner(
                mock_nli_batch.backend, pair_policy="all_pairs", strength_method="verbalized"
            )
            coherence = CoherencePipeline(
                miner=miner,
                merlin_path="/fake/merlin",
                prior_provider=provider,
                methods=("mean_marginal", "consistency"),
            )
            out = coherence.run(sample_json_data["output"])

        assert out.priors["a0"] == pytest.approx(atom_priors.priors["a0"])
        assert out.priors["a1"] == pytest.approx(atom_priors.priors["a1"])
        assert out.prior_coverage["alignment"] == "identity"
        assert out.prior_coverage["coverage"] == pytest.approx(1.0)
        assert 0.0 <= out.lcs <= 1.0
        # Coherence moved the atoms off their priors: this is a joint model, not a
        # relabelling of stage 1.
        assert out.marginals != pytest.approx(out.priors, abs=1e-9)
        json.dumps(out.to_json())

    def test_uniform_and_primed_runs_differ(
        self, sample_json_data, mock_nli_batch, mock_summarizer, monkeypatch
    ):
        """The priors must actually change the score, not merely be carried along."""
        atoms = {a["id"]: a["text"] for a in sample_json_data["atoms"]}
        _verdicts_by_atom(
            mock_nli_batch,
            {
                atoms["a0"]: {"label": "contradiction", "probability": 0.97},
                atoms["a1"]: {"label": "contradiction", "probability": 0.97},
            },
        )
        monkeypatch.setattr(
            "fact_reasoner.assessor._run_merlin_shared", brute_force_run_merlin
        )
        pipeline = _build_factuality(sample_json_data, mock_nli_batch, mock_summarizer)
        provider = FactReasonerPriorProvider(
            runner=_ItemRunner(pipeline), mode="file_item", item=sample_json_data
        )

        # The uniform run has no stage-1 atoms to reuse, so it needs an atomizer;
        # stub it to the same atoms so the two runs differ ONLY in their priors.
        atomizer = _StubAtomizer([a["text"] for a in sample_json_data["atoms"]])

        with dry_run_patches():
            miner = RelationMiner(
                mock_nli_batch.backend,
                atomizer=atomizer,
                pair_policy="all_pairs",
                strength_method="verbalized",
            )
            uniform = CoherencePipeline(
                miner=miner, merlin_path="/fake/merlin"
            ).run(sample_json_data["output"])
            primed = CoherencePipeline(
                miner=miner, merlin_path="/fake/merlin", prior_provider=provider
            ).run(sample_json_data["output"])

        # Both atoms are factually contradicted, so priming must pull the score down.
        assert primed.lcs < uniform.lcs
        assert set(uniform.priors.values()) == {0.5}

    def test_dry_run_patches_can_cover_the_factuality_merlin(
        self, sample_json_data, mock_nli_batch, mock_summarizer
    ):
        """`patch_assessor_merlin=True` is what makes a full two-stage dry run work."""
        import fact_reasoner.assessor as assessor_mod

        original = assessor_mod._run_merlin_shared
        with dry_run_patches(patch_assessor_merlin=True):
            assert assessor_mod._run_merlin_shared is brute_force_run_merlin
            pipeline = _build_factuality(
                sample_json_data, mock_nli_batch, mock_summarizer
            )
            results, _marginals = pipeline.score()
            priors = atom_priors_from_results(results)
            assert set(priors) == {"a0", "a1"}
        # Restored afterwards, so other tests are unaffected.
        assert assessor_mod._run_merlin_shared is original

    def test_default_leaves_the_factuality_merlin_alone(self):
        """Off by default, so existing coherence-only dry runs are untouched."""
        import fact_reasoner.assessor as assessor_mod

        original = assessor_mod._run_merlin_shared
        with dry_run_patches():
            assert assessor_mod._run_merlin_shared is original

    def test_precomputed_priors_reproduce_a_live_two_stage_run(
        self, sample_json_data, mock_nli_batch, mock_summarizer, monkeypatch, tmp_path
    ):
        """A saved factuality run must score identically, with zero LLM calls."""
        monkeypatch.setattr(
            "fact_reasoner.assessor._run_merlin_shared", brute_force_run_merlin
        )
        pipeline = _build_factuality(sample_json_data, mock_nli_batch, mock_summarizer)
        results, _marginals = pipeline.score()

        path = tmp_path / "factuality.json"
        path.write_text(
            json.dumps(
                {
                    "factuality_score_per_atom": results["factuality_score_per_atom"],
                    "atoms": [
                        {"id": aid, "text": atom.get_text()}
                        for aid, atom in pipeline.atoms.items()
                    ],
                }
            )
        )

        from fact_reasoner.lcs.priors import PrecomputedPriorProvider

        with dry_run_patches():
            miner = RelationMiner(
                mock_nli_batch.backend, pair_policy="all_pairs",
                strength_method="verbalized",
            )
            live = CoherencePipeline(
                miner=miner,
                merlin_path="/fake/merlin",
                prior_provider=FactReasonerPriorProvider(
                    runner=_ItemRunner(pipeline), mode="file_item",
                    item=sample_json_data,
                ),
            ).run(sample_json_data["output"])
            replayed = CoherencePipeline(
                miner=miner,
                merlin_path="/fake/merlin",
                prior_provider=PrecomputedPriorProvider(str(path)),
            ).run(sample_json_data["output"])

        assert replayed.lcs == pytest.approx(live.lcs, abs=1e-12)
        assert replayed.prior_coverage["source"] == "file"
