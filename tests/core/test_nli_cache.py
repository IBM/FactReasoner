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

"""Tests for the cross-run NLI verdict cache."""

from fact_reasoner.core.base import Atom, Context
from fact_reasoner.core.nli_cache import (
    PROMPT_VERSION,
    NLIVerdictCache,
    extractor_identity,
)
from fact_reasoner.core.utils import build_relations

from .test_build_relations import make_atoms, make_contexts, relation_signature


class TestRoundTrip:
    def test_put_then_get(self, tmp_path):
        cache = NLIVerdictCache(str(tmp_path))
        key = cache.make_key("m1", "logprobs", "premise text", "hypothesis text")

        assert cache.get_many([key]) == {}

        cache.put_many([(key, {"label": "entailment", "probability": 0.87})])
        found = cache.get_many([key])

        assert found[key] == {"label": "entailment", "probability": 0.87}
        assert len(cache) == 1

    def test_get_many_omits_misses(self, tmp_path):
        cache = NLIVerdictCache(str(tmp_path))
        hit = cache.make_key("m", "logprobs", "a", "b")
        miss = cache.make_key("m", "logprobs", "c", "d")
        cache.put_many([(hit, {"label": "neutral", "probability": 0.5})])

        found = cache.get_many([hit, miss])

        assert hit in found
        assert miss not in found

    def test_empty_inputs_are_safe(self, tmp_path):
        cache = NLIVerdictCache(str(tmp_path))

        assert cache.get_many([]) == {}
        assert cache.put_many([]) == 0
        assert cache.put_many([("k", None)]) == 0

    def test_survives_reopen(self, tmp_path):
        """The point of the cache is that a *later run* pays nothing."""
        first = NLIVerdictCache(str(tmp_path))
        key = first.make_key("m", "logprobs", "p", "h")
        first.put_many([(key, {"label": "contradiction", "probability": 0.7})])

        second = NLIVerdictCache(str(tmp_path))

        assert second.get_many([key])[key]["label"] == "contradiction"

    def test_overwrites_existing_key(self, tmp_path):
        cache = NLIVerdictCache(str(tmp_path))
        key = cache.make_key("m", "logprobs", "p", "h")
        cache.put_many([(key, {"label": "neutral", "probability": 0.5})])
        cache.put_many([(key, {"label": "entailment", "probability": 0.9})])

        assert cache.get_many([key])[key]["label"] == "entailment"
        assert len(cache) == 1


class TestKeyDiscipline:
    def test_direction_matters(self, tmp_path):
        """NLI is directional, so the two orderings must never be aliased."""
        cache = NLIVerdictCache(str(tmp_path))

        forward = cache.make_key("m", "logprobs", "A", "B")
        reverse = cache.make_key("m", "logprobs", "B", "A")

        assert forward != reverse

    def test_model_and_method_matter(self, tmp_path):
        cache = NLIVerdictCache(str(tmp_path))
        base = cache.make_key("m1", "logprobs", "p", "h")

        assert cache.make_key("m2", "logprobs", "p", "h") != base
        # simbauq and logprobs produce different probabilities for the same text.
        assert cache.make_key("m1", "simbauq", "p", "h") != base

    def test_prompt_version_matters(self, tmp_path):
        """A prompt edit must invalidate stored verdicts."""
        old = NLIVerdictCache(str(tmp_path), prompt_version="nli-v1")
        new = NLIVerdictCache(str(tmp_path), prompt_version="nli-v2")

        assert old.make_key("m", "logprobs", "p", "h") != new.make_key(
            "m", "logprobs", "p", "h"
        )

    def test_separator_prevents_concatenation_collision(self, tmp_path):
        """('ab','c') must not hash the same as ('a','bc')."""
        cache = NLIVerdictCache(str(tmp_path))

        assert cache.make_key("m", "logprobs", "ab", "c") != cache.make_key(
            "m", "logprobs", "a", "bc"
        )

    def test_default_prompt_version_is_used(self, tmp_path):
        cache = NLIVerdictCache(str(tmp_path))
        assert cache.prompt_version == PROMPT_VERSION


class TestExtractorIdentity:
    def test_reads_model_id_from_backend(self):
        class Backend:
            model_id = "granite-4-1-30b"

        class Extractor:
            method = "logprobs"
            backend = Backend()

        assert extractor_identity(Extractor()) == ("granite-4-1-30b", "logprobs")

    def test_falls_back_to_direct_attributes(self):
        class Extractor:
            model_id = "direct-model"
            nli_method = "simbauq"

        assert extractor_identity(Extractor()) == ("direct-model", "simbauq")

    def test_missing_attributes_yield_empty_strings(self):
        class Extractor:
            pass

        assert extractor_identity(Extractor()) == ("", "")


class TestCacheInBuildRelations:
    def test_warm_cache_issues_no_llm_calls(self, tmp_path, mock_nli_batch):
        """A second identical run must cost nothing and return the same relations."""
        atoms = make_atoms(3)
        contexts = make_contexts(atoms, per_atom=2)
        mock_nli_batch.default_verdict = {"label": "entailment", "probability": 0.9}
        cache = NLIVerdictCache(str(tmp_path))

        cold_stats = {}
        cold = build_relations(
            atoms=atoms, contexts=contexts,
            rel_atom_context=True, rel_context_context=True,
            nli_extractor=mock_nli_batch, cache=cache, stats=cold_stats,
        )
        cold_calls = mock_nli_batch.total_calls()
        assert cold_calls > 0
        assert cold_stats["totals"]["cache_hits"] == 0

        mock_nli_batch.calls.clear()
        warm_stats = {}
        warm = build_relations(
            atoms=atoms, contexts=contexts,
            rel_atom_context=True, rel_context_context=True,
            nli_extractor=mock_nli_batch, cache=cache, stats=warm_stats,
        )

        assert mock_nli_batch.total_calls() == 0
        assert warm_stats["totals"]["llm_calls"] == 0
        assert warm_stats["totals"]["cache_hits"] == cold_calls
        # Score-neutral: identical relations, in identical order.
        assert relation_signature(warm) == relation_signature(cold)

    def test_partial_cache_only_computes_misses(self, tmp_path, mock_nli_batch):
        atoms = make_atoms(2)
        contexts = make_contexts(atoms, per_atom=1)
        cache = NLIVerdictCache(str(tmp_path))

        build_relations(
            atoms=atoms, contexts=contexts,
            rel_atom_context=True, rel_context_context=False,
            nli_extractor=mock_nli_batch, cache=cache,
        )
        first_calls = mock_nli_batch.total_calls()

        # Add an atom, so only the new pairs are unknown.
        atoms["a2"] = Atom(id="a2", text="A newly added atom asserting something.")
        new_context = Context(
            id="c_a2_0", atom=atoms["a2"],
            text="Evidence 0 concerning a2.",
            synthetic_summary="Summary 0 of a2.",
        )
        atoms["a2"].add_context(new_context)
        contexts["c_a2_0"] = new_context

        mock_nli_batch.calls.clear()
        stats = {}
        build_relations(
            atoms=atoms, contexts=contexts,
            rel_atom_context=True, rel_context_context=False,
            nli_extractor=mock_nli_batch, cache=cache, stats=stats,
        )

        total_pairs = len(atoms) * len(contexts)  # 3 * 3
        assert stats["totals"]["cache_hits"] == first_calls
        assert stats["totals"]["llm_calls"] == total_pairs - first_calls
        assert mock_nli_batch.total_calls() == total_pairs - first_calls

    def test_no_cache_is_the_default(self, mock_nli_batch):
        atoms = make_atoms(2)
        contexts = make_contexts(atoms, per_atom=1)
        stats = {}

        build_relations(
            atoms=atoms, contexts=contexts,
            rel_atom_context=True, rel_context_context=False,
            nli_extractor=mock_nli_batch, stats=stats,
        )

        assert stats["totals"]["cache_hits"] == 0
        assert stats["totals"]["llm_calls"] == len(atoms) * len(contexts)

    def test_cache_is_keyed_per_method(self, tmp_path, mock_nli_batch):
        """Switching the confidence method must not reuse the other's verdicts."""
        atoms = make_atoms(2)
        contexts = make_contexts(atoms, per_atom=1)
        cache = NLIVerdictCache(str(tmp_path))

        build_relations(
            atoms=atoms, contexts=contexts,
            rel_atom_context=True, rel_context_context=False,
            nli_extractor=mock_nli_batch, cache=cache,
        )

        mock_nli_batch.method = "simbauq"
        mock_nli_batch.calls.clear()
        stats = {}
        build_relations(
            atoms=atoms, contexts=contexts,
            rel_atom_context=True, rel_context_context=False,
            nli_extractor=mock_nli_batch, cache=cache, stats=stats,
        )

        assert stats["totals"]["cache_hits"] == 0
        assert mock_nli_batch.total_calls() == len(atoms) * len(contexts)
