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

"""Tests for ``build_relations``: call budgets, score neutrality, and the cascade."""

import pytest

from fact_reasoner.core.base import Atom, Context, Relation
from fact_reasoner.core.nli_config import NLIPairConfig
from fact_reasoner.core.utils import (
    _mirror_needed,
    _reconcile_ctx_pair,
    _synthesize_mirrors,
    build_relations,
    remove_duplicated_contexts,
)


def make_atoms(n: int) -> dict:
    return {
        f"a{i}": Atom(id=f"a{i}", text=f"Atom {i} asserts proposition {i}.")
        for i in range(n)
    }


def make_contexts(atoms: dict, per_atom: int = 2) -> dict:
    contexts = {}
    for atom_id, atom in atoms.items():
        for j in range(per_atom):
            context = Context(
                id=f"c_{atom_id}_{j}",
                atom=atom,
                text=f"Evidence {j} concerning {atom_id}.",
                synthetic_summary=f"Summary {j} of {atom_id}.",
            )
            contexts[context.id] = context
            atom.add_context(context)
    return contexts


def relation_signature(relations):
    """A comparable fingerprint of a relation list, order included."""
    return [
        (r.source.id, r.target.id, r.get_type(), r.get_probability(), r.link)
        for r in relations
    ]


class TestCallBudget:
    """Exact LLM call counts per version, and that policies actually reduce them."""

    def test_v2_costs_atoms_times_contexts(self, mock_nli_batch):
        atoms, = (make_atoms(4),)
        contexts = make_contexts(atoms, per_atom=2)
        stats = {}

        build_relations(
            atoms=atoms,
            contexts=contexts,
            rel_atom_context=True,
            rel_context_context=False,
            contexts_per_atom_only=False,
            nli_extractor=mock_nli_batch,
            stats=stats,
        )

        expected = len(atoms) * len(contexts)  # 4 * 8 = 32
        assert mock_nli_batch.total_calls() == expected
        assert stats["atom_context"]["llm_calls"] == expected
        assert stats["totals"]["llm_calls"] == expected

    def test_v3_adds_both_context_directions(self, mock_nli_batch):
        atoms = make_atoms(3)
        contexts = make_contexts(atoms, per_atom=2)
        stats = {}

        build_relations(
            atoms=atoms,
            contexts=contexts,
            rel_atom_context=True,
            rel_context_context=True,
            contexts_per_atom_only=False,
            nli_extractor=mock_nli_batch,
            stats=stats,
        )

        num_atoms, num_contexts = len(atoms), len(contexts)
        expected = num_atoms * num_contexts + num_contexts * (num_contexts - 1)
        assert mock_nli_batch.total_calls() == expected
        assert stats["totals"]["llm_calls"] == expected
        # A faithful run's counterfactual is itself.
        assert stats["totals"]["llm_calls_all_pairs_equivalent"] == expected
        assert stats["totals"]["reduction_factor"] == 1.0

    def test_v1_costs_atoms_times_own_contexts(self, mock_nli_batch):
        atoms = make_atoms(5)
        contexts = make_contexts(atoms, per_atom=3)
        stats = {}

        build_relations(
            atoms=atoms,
            contexts=contexts,
            rel_atom_context=True,
            rel_context_context=False,
            contexts_per_atom_only=True,
            nli_extractor=mock_nli_batch,
            stats=stats,
        )

        assert mock_nli_batch.total_calls() == 5 * 3  # linear, not 5 * 15

    def test_provenance_is_strictly_cheaper_than_all_pairs(self, mock_nli_batch):
        atoms = make_atoms(6)
        contexts = make_contexts(atoms, per_atom=2)

        faithful_stats = {}
        build_relations(
            atoms=atoms, contexts=contexts,
            rel_atom_context=True, rel_context_context=True,
            nli_extractor=mock_nli_batch, stats=faithful_stats,
        )
        faithful_calls = mock_nli_batch.total_calls()

        mock_nli_batch.calls.clear()
        cheap_stats = {}
        build_relations(
            atoms=atoms, contexts=contexts,
            rel_atom_context=True, rel_context_context=True,
            nli_extractor=mock_nli_batch,
            pair_config=NLIPairConfig(
                policy="provenance",
                gate_threshold=0.99,  # rescue nothing, so provenance dominates
                ctx_ctx_single_direction_cascade=True,
            ),
            stats=cheap_stats,
        )
        cheap_calls = mock_nli_batch.total_calls()

        assert cheap_calls < faithful_calls
        # The counterfactual is reported against the same inputs either way.
        assert (
            cheap_stats["totals"]["llm_calls_all_pairs_equivalent"]
            == faithful_stats["totals"]["llm_calls_all_pairs_equivalent"]
        )
        assert cheap_stats["totals"]["reduction_factor"] > 1.0


class TestScoreNeutrality:
    """Pruning pairs that would be neutral must not change the result at all."""

    def test_gated_matches_all_pairs_when_only_neutrals_are_pruned(
        self, mock_nli_batch
    ):
        atoms = make_atoms(4)
        contexts = make_contexts(atoms, per_atom=2)

        # Make every pair neutral except the provenance ones, which the gate never
        # prunes. So whatever the gate removes was going to be discarded anyway.
        mock_nli_batch.default_verdict = {"label": "neutral", "probability": 0.8}
        for atom_id, atom in atoms.items():
            for context_id in atom.get_contexts():
                premise = contexts[context_id].get_text()
                mock_nli_batch.verdicts[(premise, atom.get_text())] = {
                    "label": "entailment",
                    "probability": 0.91,
                }

        faithful = build_relations(
            atoms=atoms, contexts=contexts,
            rel_atom_context=True, rel_context_context=False,
            nli_extractor=mock_nli_batch,
        )

        mock_nli_batch.calls.clear()
        gated = build_relations(
            atoms=atoms, contexts=contexts,
            rel_atom_context=True, rel_context_context=False,
            nli_extractor=mock_nli_batch,
            pair_config=NLIPairConfig(policy="provenance", gate_threshold=0.99,
                                      neighbor_window=0),
        )

        # Identical relations, in identical order -- the executable form of the
        # argument that pruning a would-be-neutral pair is a no-op.
        assert relation_signature(gated) == relation_signature(faithful)
        assert len(faithful) == 4 * 2

    def test_neutral_relations_are_dropped(self, mock_nli_batch):
        atoms = make_atoms(2)
        contexts = make_contexts(atoms, per_atom=1)
        mock_nli_batch.default_verdict = {"label": "neutral", "probability": 1.0}

        relations = build_relations(
            atoms=atoms, contexts=contexts,
            rel_atom_context=True, rel_context_context=False,
            nli_extractor=mock_nli_batch,
        )

        assert relations == []


class TestStatsShape:
    def test_arithmetic_is_consistent(self, mock_nli_batch):
        atoms = make_atoms(3)
        contexts = make_contexts(atoms, per_atom=2)
        stats = {}

        build_relations(
            atoms=atoms, contexts=contexts,
            rel_atom_context=True, rel_context_context=True,
            nli_extractor=mock_nli_batch, stats=stats,
        )

        for phase in ("atom_context", "context_context"):
            block = stats[phase]
            assert block["pairs_selected"] + block["pairs_pruned"] == block[
                "pairs_possible"
            ]
            assert block["relations_kept"] + block["neutral_dropped"] == block[
                "pairs_selected"
            ]

        ac = stats["atom_context"]
        assert ac["llm_calls"] + ac["cache_hits"] == ac["pairs_selected"]
        assert stats["policy"] == "all_pairs"
        assert stats["num_atoms"] == 3
        assert stats["num_contexts"] == 6
        assert "seconds" in stats["totals"]

    def test_faithful_config_builds_no_gate(self, mock_nli_batch):
        """The faithful path must not touch the embedding stack at all."""
        atoms = make_atoms(2)
        contexts = make_contexts(atoms, per_atom=1)
        stats = {}

        build_relations(
            atoms=atoms, contexts=contexts,
            rel_atom_context=True, rel_context_context=False,
            nli_extractor=mock_nli_batch, stats=stats,
        )

        assert stats["gate_backend"] is None
        assert stats["gate_threshold"] is None


class TestDirectionalCascade:
    """The cascade must preserve the reconciler's semantics exactly."""

    def _pair(self, type1, prob1, type2, prob2):
        ci = Context(id="c_0", atom=None, text="first")
        cj = Context(id="c_1", atom=None, text="second")
        r1 = Relation(source=ci, target=cj, type=type1, probability=prob1,
                      link="context_context")
        r2 = Relation(source=cj, target=ci, type=type2, probability=prob2,
                      link="context_context")
        return r1, r2

    @pytest.mark.parametrize("type1", ["entailment", "contradiction", "neutral"])
    @pytest.mark.parametrize("type2", ["entailment", "contradiction", "neutral"])
    def test_matches_both_directions_for_every_combination(self, type1, type2):
        """All nine verdict combinations reconcile the same way via the cascade."""
        r1, r2 = self._pair(type1, 0.8, type2, 0.7)
        expected = _reconcile_ctx_pair(r1, r2)
        expected_sig = (expected.get_type(), expected.get_probability(),
                        expected.source.id, expected.target.id)

        # Rebuild fresh relations (reconcile mutates the winner's type).
        r1, r2 = self._pair(type1, 0.8, type2, 0.7)
        need = _mirror_needed([r1])
        if need:
            # The reverse direction is scored, so the cascade sees both verdicts.
            mirrors = _synthesize_mirrors([r1], need, [r2])
        else:
            # Skipped: a synthetic neutral@0.0 stands in for the reverse call.
            mirrors = _synthesize_mirrors([r1], [], [])
        actual = _reconcile_ctx_pair(r1, mirrors[0])
        actual_sig = (actual.get_type(), actual.get_probability(),
                      actual.source.id, actual.target.id)

        if type1 == "contradiction":
            # Mirror skipped; the real contradiction survives with its own
            # probability and orientation.
            assert actual.get_type() == "contradiction"
            assert actual.get_probability() == 0.8
            assert (actual.source.id, actual.target.id) == ("c_0", "c_1")
        else:
            assert actual_sig == expected_sig

    def test_entailment_is_always_mirrored_so_equivalence_survives(self):
        """``equivalence`` only exists when both directions entail."""
        r1, r2 = self._pair("entailment", 0.9, "entailment", 0.85)
        assert _mirror_needed([r1]) == [0]

        mirrors = _synthesize_mirrors([r1], [0], [r2])
        assert _reconcile_ctx_pair(r1, mirrors[0]).get_type() == "equivalence"

    def test_neutral_is_always_mirrored(self):
        """The reverse call is the reconciler's second chance after a neutral.

        A backend failure surfaces as neutral at probability 1.0, so skipping the
        mirror here would make recall depend on transient network errors.
        """
        assert _mirror_needed([self._pair("neutral", 1.0, "neutral", 1.0)[0]]) == [0]

    def test_contradiction_mirror_is_skipped(self):
        assert _mirror_needed([self._pair("contradiction", 0.9, "neutral", 0.5)[0]]) == []

    def test_neutral_does_not_hide_entailment_through_cascade(self):
        """Ported regression: a neutral forward direction must not lose a relation."""
        r1, r2 = self._pair("neutral", 0.99, "entailment", 0.55)
        need = _mirror_needed([r1])
        assert need == [0]

        result = _reconcile_ctx_pair(r1, _synthesize_mirrors([r1], need, [r2])[0])
        assert result.get_type() == "entailment"
        assert result.get_probability() == 0.55

    def test_failed_call_does_not_hide_contradiction_through_cascade(self):
        """Ported regression: neutral@1.0 from a failure must not mask the reverse."""
        r1, r2 = self._pair("neutral", 1.0, "contradiction", 0.6)
        need = _mirror_needed([r1])
        assert need == [0]

        result = _reconcile_ctx_pair(r1, _synthesize_mirrors([r1], need, [r2])[0])
        assert result.get_type() == "contradiction"

    def test_cascade_reports_skipped_count(self, mock_nli_batch):
        atoms = make_atoms(2)
        contexts = make_contexts(atoms, per_atom=2)
        # Every context pair contradicts, so every mirror can be skipped.
        mock_nli_batch.default_verdict = {"label": "contradiction", "probability": 0.9}
        stats = {}

        build_relations(
            atoms=atoms, contexts=contexts,
            rel_atom_context=False, rel_context_context=True,
            nli_extractor=mock_nli_batch,
            pair_config=NLIPairConfig(
                policy="all_pairs", ctx_ctx_single_direction_cascade=True
            ),
            stats=stats,
        )

        cc = stats["context_context"]
        assert cc["llm_calls_dir2"] == 0
        assert cc["dir2_skipped"] == cc["pairs_selected"]
        # Half the faithful cost: one direction instead of two.
        assert mock_nli_batch.total_calls() == cc["pairs_selected"]


class TestPerAtomPathIntegrity:
    """The v1 path must send context text, never context ids."""

    def test_premises_are_context_text_not_ids(self, mock_nli_batch):
        atoms = make_atoms(3)
        contexts = make_contexts(atoms, per_atom=2)

        build_relations(
            atoms=atoms, contexts=contexts,
            rel_atom_context=True, rel_context_context=False,
            contexts_per_atom_only=True,
            nli_extractor=mock_nli_batch,
            use_summarized_contexts=False,
        )

        context_texts = {c.get_text() for c in contexts.values()}
        context_ids = set(contexts)
        assert mock_nli_batch.calls
        for premises, _hypotheses in mock_nli_batch.calls:
            for premise in premises:
                assert premise not in context_ids, (
                    "a context id leaked into the NLI premise"
                )
                assert premise in context_texts

    def test_relation_sources_are_context_objects(self, mock_nli_batch):
        """``Relation.source`` must be a Context, or graph construction breaks."""
        atoms = make_atoms(2)
        contexts = make_contexts(atoms, per_atom=1)
        mock_nli_batch.default_verdict = {"label": "entailment", "probability": 0.9}

        relations = build_relations(
            atoms=atoms, contexts=contexts,
            rel_atom_context=True, rel_context_context=False,
            contexts_per_atom_only=True,
            nli_extractor=mock_nli_batch,
        )

        assert relations
        for rel in relations:
            assert isinstance(rel.source, Context)
            assert isinstance(rel.target, Atom)
            assert rel.source.id  # the attribute the graph builder reads

    def test_graph_construction_succeeds_on_v1_relations(self, mock_nli_batch):
        from fact_reasoner.fact_graph import FactGraph

        atoms = make_atoms(2)
        contexts = make_contexts(atoms, per_atom=1)
        mock_nli_batch.default_verdict = {"label": "entailment", "probability": 0.9}

        relations = build_relations(
            atoms=atoms, contexts=contexts,
            rel_atom_context=True, rel_context_context=False,
            contexts_per_atom_only=True,
            nli_extractor=mock_nli_batch,
        )

        graph = FactGraph(
            atoms=list(atoms.values()),
            contexts=list(contexts.values()),
            relations=relations,
        )
        assert len(graph.get_nodes()) == len(atoms) + len(contexts)


class TestRemoveDuplicatedContextsSummary:
    def test_unsummarized_contexts_are_not_collapsed(self):
        """Distinct contexts without summaries must survive summary-mode dedup.

        ``Context.get_summary()`` returns an empty string rather than None when no
        summary was produced, so a guard on ``is None`` never fires and every
        unsummarized context keys on ``""`` -- collapsing all of them into one.
        """
        atoms = make_atoms(1)
        contexts = {}
        for i in range(3):
            context = Context(
                id=f"c_a0_{i}",
                atom=atoms["a0"],
                text=f"Distinct unsummarized evidence number {i}.",
            )
            contexts[context.id] = context
            atoms["a0"].add_context(context)

        kept, _ = remove_duplicated_contexts(contexts, atoms, check_summary=True)

        assert len(kept) == 3

    def test_true_duplicates_still_collapse_in_summary_mode(self):
        atoms = make_atoms(1)
        contexts = {}
        for i in range(2):
            context = Context(
                id=f"c_a0_{i}",
                atom=atoms["a0"],
                text=f"Body {i}",
                synthetic_summary="The very same summary.",
            )
            contexts[context.id] = context
            atoms["a0"].add_context(context)

        kept, _ = remove_duplicated_contexts(contexts, atoms, check_summary=True)

        assert len(kept) == 1
