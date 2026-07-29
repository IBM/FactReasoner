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

"""Tests for NLI candidate-pair selection."""

from itertools import combinations

import pytest

from fact_reasoner.core.base import Atom, Context
from fact_reasoner.core.nli_config import NLIPairConfig
from fact_reasoner.core.nli_pairs import (
    _PairGate,
    build_gate,
    context_owners,
    dedup_contexts_near,
    is_query_context,
    prune_dangling_context_refs,
    select_atom_context_pairs,
    select_context_context_pairs,
)


def make_atoms(n: int) -> dict:
    """``n`` atoms with ids ``a0..a{n-1}`` and distinguishable text."""
    return {f"a{i}": Atom(id=f"a{i}", text=f"Atom {i} states fact number {i}.") for i in range(n)}


def make_contexts(atoms: dict, per_atom: int = 2, num_query: int = 0) -> dict:
    """Contexts retrieved per atom, plus optional query-level contexts.

    Mirrors the retriever: per-atom contexts get ``atom`` set and are registered on
    the atom, while query-level contexts have ``atom=None``.
    """
    contexts = {}
    for atom_id, atom in atoms.items():
        for j in range(per_atom):
            context_id = f"c_{atom_id}_{j}"
            context = Context(
                id=context_id,
                atom=atom,
                text=f"Context {j} about atom {atom_id} with supporting detail.",
                synthetic_summary=f"Summary {j} for {atom_id}.",
            )
            contexts[context_id] = context
            atom.add_context(context)
    for j in range(num_query):
        context_id = f"c_q_{j}"
        contexts[context_id] = Context(
            id=context_id,
            atom=None,
            text=f"Query level context {j} with general background.",
            synthetic_summary=f"Query summary {j}.",
        )
    return contexts


class TestAllPairsReproducibility:
    """The faithful policy must reproduce the original enumeration exactly.

    These assert *list* equality rather than set equality: iteration order is part
    of the contract, because reordering pairs can perturb sampling under
    temperature-based confidence methods that share a rate limiter.
    """

    def test_atom_context_matches_original_cross_product(self):
        atoms = make_atoms(4)
        contexts = make_contexts(atoms, per_atom=2)

        # Transcribed verbatim from the original build_relations enumeration:
        #   for _, atom in atoms.items():
        #       for _, context in contexts.items():
        #           atom_context_pairs.append((context, atom))
        expected = []
        for atom_id, _atom in atoms.items():
            for context_id, _context in contexts.items():
                expected.append((context_id, atom_id))

        pairs, coverage = select_atom_context_pairs(
            atoms, contexts, policy="all_pairs", contexts_per_atom_only=False
        )

        assert pairs == expected
        assert coverage["pairs_selected"] == len(atoms) * len(contexts)
        assert coverage["pairs_pruned"] == 0

    def test_context_context_matches_original_combinations(self):
        atoms = make_atoms(3)
        contexts = make_contexts(atoms, per_atom=2)

        # Original: combinations(sorted(contexts.keys()), 2)
        expected = list(combinations(sorted(contexts.keys()), 2))

        pairs, coverage = select_context_context_pairs(contexts, policy="all_pairs")

        assert pairs == expected
        num = len(contexts)
        assert coverage["pairs_possible"] == num * (num - 1) // 2
        assert coverage["pairs_pruned"] == 0

    def test_contexts_per_atom_only_yields_context_objects_not_ids(self):
        """Regression: the per-atom branch must resolve to real contexts.

        The original iterated ``atom.get_contexts()``, which is a dict, so it
        yielded context *id strings*. Those were then passed to the NLI call as the
        premise, meaning the model saw an id like ``c_a0_0`` instead of the context
        text.
        """
        atoms = make_atoms(3)
        contexts = make_contexts(atoms, per_atom=2)

        pairs, coverage = select_atom_context_pairs(
            atoms, contexts, policy="all_pairs", contexts_per_atom_only=True
        )

        # One pair per (atom, its own context).
        assert len(pairs) == 3 * 2
        for context_id, atom_id in pairs:
            assert context_id in contexts
            assert atom_id in atoms
            # The context id belongs to that atom.
            assert context_id in atoms[atom_id].get_contexts()

    def test_per_atom_skips_stale_context_references(self):
        """A context removed from the registry must not reach the NLI call."""
        atoms = make_atoms(2)
        contexts = make_contexts(atoms, per_atom=2)
        stale_id = "c_a0_0"
        del contexts[stale_id]  # removed globally, still on the atom

        pairs, _ = select_atom_context_pairs(
            atoms, contexts, policy="all_pairs", contexts_per_atom_only=True
        )

        assert all(context_id in contexts for context_id, _ in pairs)
        assert stale_id not in [c for c, _ in pairs]


class TestContextOwners:
    def test_recovers_all_owners_across_atoms(self):
        """A context listed by three atoms must report all three owners."""
        atoms = make_atoms(3)
        shared = Context(id="c_shared", atom=atoms["a0"], text="Shared evidence.")
        contexts = {"c_shared": shared}
        for atom in atoms.values():
            atom.add_context(shared)

        owners = context_owners(atoms, contexts)

        assert owners["c_shared"] == {"a0", "a1", "a2"}

    def test_unions_single_back_pointer(self):
        """The lossy ``Context.atom`` pointer is still honored as a fallback."""
        atoms = make_atoms(2)
        orphan = Context(id="c_x", atom=atoms["a1"], text="Only reachable via pointer.")
        contexts = {"c_x": orphan}  # no atom lists it

        owners = context_owners(atoms, contexts)

        assert owners["c_x"] == {"a1"}

    def test_ignores_ids_absent_from_contexts(self):
        atoms = make_atoms(1)
        ghost = Context(id="c_ghost", atom=atoms["a0"], text="Not registered.")
        atoms["a0"].add_context(ghost)

        owners = context_owners(atoms, {})

        assert owners == {}

    def test_query_context_has_no_owner(self):
        atoms = make_atoms(2)
        contexts = make_contexts(atoms, per_atom=1, num_query=1)

        owners = context_owners(atoms, contexts)

        assert owners["c_q_0"] == set()
        assert is_query_context("c_q_0")
        assert not is_query_context("c_a0_0")


class TestProvenancePolicy:
    def _gate_args(self, atoms, contexts, model="all-MiniLM-L6-v2"):
        gate, atom_ids, context_ids = build_gate(
            atoms, contexts, use_summary=False, embedding_model=model
        )
        return dict(
            gate=gate, gate_atom_ids=atom_ids, gate_context_ids=context_ids
        )

    def test_keeps_provenance_and_query_contexts(self):
        atoms = make_atoms(4)
        contexts = make_contexts(atoms, per_atom=2, num_query=1)

        pairs, coverage = select_atom_context_pairs(
            atoms,
            contexts,
            policy="provenance",
            gate_threshold=0.99,  # gate rescues nothing, isolating provenance
            neighbor_window=0,
            **self._gate_args(atoms, contexts),
        )

        pair_set = set(pairs)
        # Every context is compared against the atom that retrieved it.
        for atom_id, atom in atoms.items():
            for context_id in atom.get_contexts():
                assert (context_id, atom_id) in pair_set
        # The query-level context is compared against every atom.
        for atom_id in atoms:
            assert ("c_q_0", atom_id) in pair_set

        assert coverage["num_provenance"] == 4 * 2
        assert coverage["num_query_context"] == 4
        assert coverage["pairs_selected"] < coverage["pairs_possible"]

    def test_prunes_unrelated_distant_pairs(self):
        atoms = make_atoms(4)
        contexts = make_contexts(atoms, per_atom=1)

        pairs, coverage = select_atom_context_pairs(
            atoms,
            contexts,
            policy="provenance",
            gate_threshold=0.99,
            neighbor_window=0,
            **self._gate_args(atoms, contexts),
        )

        # a0's context must not be compared against the far atom a3.
        assert ("c_a0_0", "a3") not in set(pairs)
        assert coverage["pairs_pruned"] > 0

    def test_neighbor_window_admits_adjacent_atoms(self):
        atoms = make_atoms(4)
        contexts = make_contexts(atoms, per_atom=1)
        gate_args = self._gate_args(atoms, contexts)

        narrow, _ = select_atom_context_pairs(
            atoms, contexts, policy="provenance",
            gate_threshold=0.99, neighbor_window=0, **gate_args,
        )
        wide, coverage = select_atom_context_pairs(
            atoms, contexts, policy="provenance",
            gate_threshold=0.99, neighbor_window=1, **gate_args,
        )

        assert set(narrow) < set(wide)
        # a0's context now also reaches a1, but still not a3.
        assert ("c_a0_0", "a1") in set(wide)
        assert ("c_a0_0", "a3") not in set(wide)
        assert coverage["num_neighbor"] > 0

    def test_gate_threshold_zero_keeps_everything(self):
        """A permissive gate degenerates to the all_pairs set."""
        atoms = make_atoms(3)
        contexts = make_contexts(atoms, per_atom=2)
        gate_args = self._gate_args(atoms, contexts)

        pairs, coverage = select_atom_context_pairs(
            atoms, contexts, policy="gated", gate_threshold=0.0, **gate_args
        )

        assert coverage["pairs_selected"] == len(atoms) * len(contexts)
        assert len(pairs) == coverage["pairs_possible"]

    def test_gated_policy_requires_a_gate(self):
        atoms = make_atoms(2)
        contexts = make_contexts(atoms, per_atom=1)

        with pytest.raises(ValueError, match="requires a gate"):
            select_atom_context_pairs(atoms, contexts, policy="gated")
        with pytest.raises(ValueError, match="requires a gate"):
            select_context_context_pairs(contexts, policy="gated")

    def test_unknown_policy_rejected(self):
        atoms = make_atoms(1)
        contexts = make_contexts(atoms, per_atom=1)

        with pytest.raises(ValueError, match="Unknown NLI pair policy"):
            select_atom_context_pairs(atoms, contexts, policy="nonsense")
        with pytest.raises(ValueError, match="Unknown NLI pair policy"):
            select_context_context_pairs(contexts, policy="nonsense")


class TestGateFallback:
    def test_falls_back_to_jaccard_when_embedding_stack_unavailable(
        self, monkeypatch
    ):
        """Selection must still work when the embedding model cannot be loaded.

        sentence-transformers is a base dependency, so in practice this path is
        reached via a model-load failure (offline, corrupt cache) rather than a
        missing package -- the gate catches both. Simulating the import failure is
        the cheapest way to exercise it. The gate reports which backend it used.
        """
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("sentence_transformers"):
                raise ImportError("simulated: sentence-transformers not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        atoms = make_atoms(3)
        contexts = make_contexts(atoms, per_atom=2)
        gate, atom_ids, context_ids = build_gate(atoms, contexts)

        assert gate.backend == "jaccard"

        pairs, coverage = select_atom_context_pairs(
            atoms, contexts, policy="gated", gate_threshold=0.0,
            gate=gate, gate_atom_ids=atom_ids, gate_context_ids=context_ids,
        )
        assert len(pairs) > 0
        assert coverage["gate_backend"] == "jaccard"

    def test_similarity_is_symmetric_and_indexed_consistently(self):
        gate = _PairGate(["alpha beta gamma"], ["alpha beta delta", "totally other"])

        assert gate.context_context(0, 1) == gate.context_context(1, 0)
        # The atom shares more tokens with context 0 than with context 1.
        assert gate.atom_context(0, 0) > gate.atom_context(0, 1)


class TestDedupContextsNear:
    def test_merges_owners_onto_survivor(self):
        """Collapsing a duplicate must transfer its owners, not drop them.

        Exact-text dedup deletes the losing duplicate from its own atom's dict and
        leaves the survivor claiming a single atom, so an atom can lose evidence
        entirely. Near-dup dedup must be strictly better.
        """
        atoms = make_atoms(2)
        text = "Marie Curie won the Nobel Prize in Physics in 1903."
        c0 = Context(id="c_a0_0", atom=atoms["a0"], text=text, synthetic_summary=text)
        c1 = Context(id="c_a1_0", atom=atoms["a1"], text=text, synthetic_summary=text)
        atoms["a0"].add_context(c0)
        atoms["a1"].add_context(c1)
        contexts = {"c_a0_0": c0, "c_a1_0": c1}

        kept, atoms_out, coverage = dedup_contexts_near(
            contexts, atoms, threshold=0.9, use_summary=True
        )

        assert list(kept) == ["c_a0_0"]
        assert coverage["collapsed"] == 1
        assert coverage["contexts_before"] == 2
        assert coverage["contexts_after"] == 1
        # a1 keeps evidence: it now points at the survivor.
        assert "c_a0_0" in atoms_out["a1"].get_contexts()
        assert "c_a1_0" not in atoms_out["a1"].get_contexts()
        assert coverage["owners_merged"] == 1

    def test_keeps_distinct_contexts(self):
        atoms = make_atoms(2)
        c0 = Context(id="c_a0_0", atom=atoms["a0"],
                     text="Curie won the Nobel Prize in Physics.",
                     synthetic_summary="Curie won the Nobel Prize in Physics.")
        c1 = Context(id="c_a1_0", atom=atoms["a1"],
                     text="The Pacific Ocean is the largest ocean basin on Earth.",
                     synthetic_summary="The Pacific Ocean is the largest ocean basin.")
        atoms["a0"].add_context(c0)
        atoms["a1"].add_context(c1)
        contexts = {"c_a0_0": c0, "c_a1_0": c1}

        kept, _, coverage = dedup_contexts_near(contexts, atoms, threshold=0.92)

        assert len(kept) == 2
        assert coverage["collapsed"] == 0

    def test_single_context_is_a_noop(self):
        atoms = make_atoms(1)
        contexts = make_contexts(atoms, per_atom=1)

        kept, _, coverage = dedup_contexts_near(contexts, atoms)

        assert kept == contexts
        assert coverage["collapsed"] == 0

    def test_leaves_no_dangling_references(self):
        atoms = make_atoms(3)
        text = "Identical evidence text for all three atoms."
        contexts = {}
        for atom_id, atom in atoms.items():
            context = Context(id=f"c_{atom_id}_0", atom=atom, text=text,
                              synthetic_summary=text)
            contexts[context.id] = context
            atom.add_context(context)

        kept, atoms_out, _ = dedup_contexts_near(
            contexts, atoms, threshold=0.9, use_summary=True
        )

        for atom in atoms_out.values():
            for context_id in atom.get_contexts():
                assert context_id in kept


class TestPruneDanglingRefs:
    def test_removes_stale_and_counts(self):
        atoms = make_atoms(2)
        contexts = make_contexts(atoms, per_atom=2)
        del contexts["c_a0_0"]
        del contexts["c_a1_1"]

        removed = prune_dangling_context_refs(atoms, contexts)

        assert removed == 2
        for atom in atoms.values():
            for context_id in atom.get_contexts():
                assert context_id in contexts

    def test_noop_when_consistent(self):
        atoms = make_atoms(2)
        contexts = make_contexts(atoms, per_atom=2)

        assert prune_dangling_context_refs(atoms, contexts) == 0

    def test_cleans_shared_context_from_every_owner(self):
        """A context shared by several atoms must be cleared from all of them.

        Removal sites in the pipeline clear at most the single owning atom, so a
        context retrieved for three atoms leaves two stale references behind.
        """
        atoms = make_atoms(3)
        shared = Context(id="c_shared", atom=atoms["a0"], text="Shared evidence.")
        for atom in atoms.values():
            atom.add_context(shared)

        # The context is dropped globally but still listed by all three atoms.
        removed = prune_dangling_context_refs(atoms, {})

        assert removed == 3
        for atom in atoms.values():
            assert atom.get_contexts() == {}


class TestNLIPairConfig:
    def test_faithful_defaults(self):
        cfg = NLIPairConfig()
        assert cfg.policy == "all_pairs"
        assert cfg.is_faithful
        assert not cfg.needs_gate

    def test_gated_needs_gate_and_is_not_faithful(self):
        cfg = NLIPairConfig(policy="gated")
        assert cfg.needs_gate
        assert not cfg.is_faithful

    def test_cache_alone_stays_faithful(self):
        """Caching returns previously computed verdicts, so results are unchanged."""
        cfg = NLIPairConfig(cache_dir="/tmp/whatever")
        assert cfg.is_faithful

    def test_mode_presets_alias_the_internal_names(self):
        """`allpairs`/`fast` are the user-facing names for the same two configs."""
        from fact_reasoner.core.nli_config import NLI_MODES, get_pair_config

        assert tuple(NLI_MODES) == ("allpairs", "fast")
        assert get_pair_config("allpairs") is get_pair_config("faithful")
        assert get_pair_config("fast") is get_pair_config("provenance")

    def test_fast_preset_is_more_than_the_bare_policy(self):
        """The trap: `fast` must bundle dedup/cascade/merge, not just set policy."""
        from fact_reasoner.core.nli_config import get_pair_config

        fast = get_pair_config("fast")
        assert fast.policy == "provenance"
        assert fast.dedup_near_duplicates is True
        assert fast.ctx_ctx_single_direction_cascade is True
        assert fast.merge_phases is True
        assert not fast.is_faithful
        # A bare policy=provenance config sets none of those extras.
        bare = NLIPairConfig(policy="provenance")
        assert bare.dedup_near_duplicates is False
        assert bare.ctx_ctx_single_direction_cascade is False

    def test_allpairs_preset_is_faithful(self):
        from fact_reasoner.core.nli_config import get_pair_config

        cfg = get_pair_config("allpairs")
        assert cfg.is_faithful
        assert cfg.policy == "all_pairs"
        assert not cfg.needs_gate

    def test_validation(self):
        with pytest.raises(ValueError, match="Unknown NLI pair policy"):
            NLIPairConfig(policy="bogus")
        with pytest.raises(ValueError, match="gate_threshold"):
            NLIPairConfig(gate_threshold=1.5)
        with pytest.raises(ValueError, match="dedup_threshold"):
            NLIPairConfig(dedup_threshold=-0.1)
        with pytest.raises(ValueError, match="neighbor_window"):
            NLIPairConfig(neighbor_window=-1)
