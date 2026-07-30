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

"""Candidate (premise, hypothesis) selection for NLI relation extraction.

``build_relations`` sends every enumerated pair to an LLM NLI prompt: ``A * C``
for the atom-context phase and ``C * (C - 1)`` for the context-context phase
(both directions). Because contexts are retrieved *per atom*, ``C ~ A * k``, so
this is quadratic in the number of atoms for v2 and effectively cubic for v3.

Why prefiltering is safe
------------------------
Relations whose verdict is ``neutral`` are discarded by ``build_relations``, and
``FactGraph`` creates its nodes from the atoms/contexts dicts rather than from the
relation list, giving *every* node a unary factor. A node that ends up in no
pairwise factor therefore appears in exactly one normalized unary factor, which
factorizes out of the partition function and out of every atom marginal.

So pruning a pair that *would* have come back ``neutral`` is a bit-exact no-op on
the reported scores -- not approximately equal, identical. The recall risk of any
policy here is exactly ``P(pruned pair was non-neutral)``.

Note the asymmetry this implies: a false prune silently weakens an atom's
evidence, while a false keep only costs money. Thresholds should therefore be
tuned toward *keeping*.

Policies
--------
* ``"all_pairs"`` -- every enumerated pair. Faithful to the original
  implementation, including iteration order; reproduces published numbers.
* ``"gated"`` -- pairs surviving a cheap embedding/Jaccard similarity gate.
* ``"provenance"`` -- atom-context pairs restricted to the atoms that actually
  retrieved the context (plus query-level contexts, near-neighbor atoms, and a
  gate rescue for cross-atom evidence); context-context pairs gated.

This module deliberately mirrors :mod:`fact_reasoner.lcs.candidate_pairs` in
structure and coverage-dict conventions, and reuses its dependency-free helpers
rather than duplicating them.
"""

import re
from collections.abc import Sequence
from itertools import combinations

from .base import Atom, Context
from .nli_config import NLI_PAIR_POLICIES


def _embedding_gate_cls():
    """Import ``lcs.candidate_pairs._EmbeddingGate`` lazily.

    ``core`` is the lower layer and ``lcs`` imports back into ``core.utils``, so a
    module-level import here would be circular. Deferring it to first use keeps
    the dependency one-directional at import time while still reusing the single
    implementation of the sbert-with-Jaccard-fallback gate.
    """
    from fact_reasoner.lcs.candidate_pairs import _EmbeddingGate

    return _EmbeddingGate

#: Id prefix used by the retriever for query-level contexts, i.e. contexts
#: retrieved for the user query rather than for a specific atom. These have
#: ``Context.atom is None`` and are treated as bearing on every atom.
QUERY_CONTEXT_PREFIX = "c_q"


# ----------------------------------------------------------------------------
# Context ownership.
# ----------------------------------------------------------------------------


def context_owners(
    atoms: dict[str, Atom], contexts: dict[str, Context]
) -> dict[str, set[str]]:
    """Map each context id to the ids of *every* atom that retrieved it.

    ``Context.atom`` is a single pointer and is lossy in three ways: it is
    ``None`` for query-level contexts; context dedup deletes the losing
    duplicate's back-reference so a context retrieved for several atoms ends up
    claiming one; and rebuilding a pipeline from a dict lets the last atom to
    claim a context win.

    The authoritative relation is therefore the inverse index ``atom.contexts``,
    which this reads. ``Context.atom`` is unioned in as a fallback for contexts
    that no atom lists, so no ownership is lost either way.

    Args:
        atoms: The atoms, keyed by id.
        contexts: The contexts, keyed by id.

    Returns:
        ``{context_id: {atom_id, ...}}``, with an entry for every context in
        ``contexts`` (possibly an empty set, e.g. for query-level contexts).
    """
    owners: dict[str, set[str]] = {cid: set() for cid in contexts}

    # Primary source: the atom -> contexts inverse index.
    for atom_id, atom in atoms.items():
        for context_id in atom.get_contexts():  # dict -> id strings
            if context_id in owners:
                owners[context_id].add(atom_id)

    # Fallback: the single back-pointer, for contexts no atom lists.
    for context_id, context in contexts.items():
        atom = getattr(context, "atom", None)
        if atom is not None and getattr(atom, "id", None) in atoms:
            owners[context_id].add(atom.id)

    return owners


def is_query_context(context_id: str) -> bool:
    """Whether a context id denotes a query-level context (no owning atom)."""
    return context_id.startswith(QUERY_CONTEXT_PREFIX)


# ----------------------------------------------------------------------------
# The shared similarity gate.
# ----------------------------------------------------------------------------


class _PairGate:
    """Similarity gate over one shared embedding space of atoms and contexts.

    Atom and context texts are encoded together in a *single* pass, so one
    forward pass serves both the atom-context and context-context phases. Do not
    construct one gate per phase.

    Delegates to :class:`fact_reasoner.lcs.candidate_pairs._EmbeddingGate`, which
    uses sentence-transformers and transparently falls back to token Jaccard if the
    embedding model cannot be loaded, recording which in :attr:`backend`. That
    fallback is a degraded mode, not a supported configuration -- see the warning
    emitted below.
    """

    def __init__(
        self,
        atom_texts: Sequence[str],
        context_texts: Sequence[str],
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self._n_atoms = len(atom_texts)
        self._gate = _embedding_gate_cls()(
            list(atom_texts) + list(context_texts), model_name=model_name
        )
        self.backend = self._gate.backend
        if self.backend == "jaccard":
            # Loud on purpose. Measured on a 20-atom narrative, the lexical
            # fallback lost 22 of 72 non-neutral relations and shifted the
            # factuality score by 0.05, with no threshold reaching full recall --
            # whereas embeddings were lossless. Degrading quietly here would
            # silently weaken evidence rather than merely cost accuracy.
            print(
                "[NLI][WARNING] Similarity gate fell back to token Jaccard: the "
                "sentence-transformers embedding model could not be loaded. It is "
                "a base dependency, so this usually means an offline or corrupt "
                "model cache rather than a missing package. Lexical overlap misses "
                "semantically related pairs, so --nli-mode fast and the "
                "gated/provenance policies can drop real relations. Fix the model "
                "load, or use --nli-mode all_pairs."
            )

    def atom_context(self, atom_index: int, context_index: int) -> float:
        """Similarity between an atom and a context, by positional index."""
        return self._gate.similarity(atom_index, self._n_atoms + context_index)

    def context_context(self, index_i: int, index_j: int) -> float:
        """Similarity between two contexts, by positional index."""
        return self._gate.similarity(
            self._n_atoms + index_i, self._n_atoms + index_j
        )


def build_gate(
    atoms: dict[str, Atom],
    contexts: dict[str, Context],
    *,
    use_summary: bool = False,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> tuple["_PairGate", list[str], list[str]]:
    """Build the shared gate plus the id orders it is indexed by.

    The texts fed to the gate mirror what the NLI call itself would see, so the
    gate scores the same content the model would.

    Returns:
        ``(gate, atom_ids, context_ids)`` where the id lists give the positional
        index of each atom/context within the gate.
    """
    atom_ids = list(atoms.keys())
    context_ids = list(contexts.keys())
    atom_texts = [_pair_text(atoms[aid], use_summary) for aid in atom_ids]
    context_texts = [_pair_text(contexts[cid], use_summary) for cid in context_ids]
    gate = _PairGate(atom_texts, context_texts, model_name=embedding_model)
    return gate, atom_ids, context_ids


def _pair_text(obj, use_summary: bool) -> str:
    """The text an NLI call would use for this object.

    Mirrors ``predict_nli_relationships``: summaries when requested, text
    otherwise. Falls back to the text when a summary is empty, so a context that
    reached this point unsummarized still gets a meaningful similarity.
    """
    if use_summary:
        summary = obj.get_summary()
        if summary:
            return summary
    return obj.get_text()


def _atom_order_index(atom_ids: Sequence[str]) -> dict[str, int]:
    """Map atom ids to their source position.

    Atom ids are of the form ``a0, a1, ...`` which encode source order, so sort by
    the trailing integer when present and fall back to string order. Mirrors
    ``lcs.candidate_pairs._ordered_atoms``.
    """

    def key(atom_id: str):
        m = re.search(r"(\d+)$", atom_id)
        return (0, int(m.group(1))) if m else (1, atom_id)

    return {aid: i for i, aid in enumerate(sorted(atom_ids, key=key))}


# ----------------------------------------------------------------------------
# Atom-context pair selection.
# ----------------------------------------------------------------------------


def select_atom_context_pairs(
    atoms: dict[str, Atom],
    contexts: dict[str, Context],
    *,
    policy: str = "all_pairs",
    contexts_per_atom_only: bool = False,
    gate_threshold: float = 0.20,
    neighbor_window: int = 1,
    gate: _PairGate | None = None,
    gate_atom_ids: Sequence[str] | None = None,
    gate_context_ids: Sequence[str] | None = None,
) -> tuple[list[tuple[str, str]], dict[str, object]]:
    """Select ``(context_id, atom_id)`` pairs for the atom-context phase.

    Args:
        atoms: The atoms, keyed by id.
        contexts: The contexts, keyed by id.
        policy: One of :data:`NLI_PAIR_POLICIES`.
        contexts_per_atom_only: Compare each atom only against the contexts
            retrieved for it. Under ``"all_pairs"`` this is the v1 behavior.
        gate_threshold: Similarity at or above which a pair is admitted.
        neighbor_window: Under ``"provenance"``, how many atoms on either side of
            an owning atom are also compared against the context.
        gate: A shared :class:`_PairGate`; required for gated policies.
        gate_atom_ids: Atom id order the gate is indexed by.
        gate_context_ids: Context id order the gate is indexed by.

    Returns:
        ``(pairs, coverage)``. ``pairs`` holds ``(context_id, atom_id)`` tuples --
        source (premise) first, matching the ``Context -> Atom`` edge direction.

    Raises:
        ValueError: If ``policy`` is unknown, or a gated policy is requested
            without a gate.
    """
    if policy not in NLI_PAIR_POLICIES:
        raise ValueError(
            f"Unknown NLI pair policy: {policy!r} "
            f"(expected one of {list(NLI_PAIR_POLICIES)})."
        )

    num_atoms = len(atoms)
    num_contexts = len(contexts)
    # The all_pairs universe, i.e. what the original implementation would spend.
    pairs_possible = num_atoms * num_contexts

    coverage: dict[str, object] = {
        "policy": policy,
        "num_atoms": num_atoms,
        "num_contexts": num_contexts,
        "pairs_possible": pairs_possible,
        "contexts_per_atom_only": contexts_per_atom_only,
    }

    # ---- all_pairs: reproduce the original enumeration exactly, order included.
    if policy == "all_pairs":
        pairs: list[tuple[str, str]] = []
        if not contexts_per_atom_only:
            # Original: for atom: for context: append((context, atom)).
            for atom_id in atoms:
                for context_id in contexts:
                    pairs.append((context_id, atom_id))
        else:
            # Original: for atom: for context in atom.get_contexts().
            # Only ids present in `contexts` are kept, so a stale reference left
            # behind by context removal cannot reach the NLI call.
            for atom_id, atom in atoms.items():
                for context_id in atom.get_contexts():
                    if context_id in contexts:
                        pairs.append((context_id, atom_id))
        coverage.update(
            pairs_selected=len(pairs),
            pairs_pruned=max(0, pairs_possible - len(pairs)),
            gate_threshold=None,
        )
        return pairs, coverage

    # ---- gated / provenance: both need the shared gate.
    if gate is None or gate_atom_ids is None or gate_context_ids is None:
        raise ValueError(
            f"policy={policy!r} requires a gate; pass gate/gate_atom_ids/"
            "gate_context_ids (see build_gate)."
        )
    atom_pos = {aid: i for i, aid in enumerate(gate_atom_ids)}
    context_pos = {cid: i for i, cid in enumerate(gate_context_ids)}

    owners = context_owners(atoms, contexts)
    order = _atom_order_index(list(atoms.keys()))

    num_provenance = 0
    num_query_context = 0
    num_neighbor = 0
    num_gate_rescued = 0

    pairs = []
    seen: set[tuple[str, str]] = set()

    # Iterate atom-major, so the pair order matches the original enumeration for
    # whatever survives. This keeps result ordering comparable across policies.
    for atom_id in atoms:
        for context_id in contexts:
            pair = (context_id, atom_id)
            if pair in seen:
                continue

            reason = None
            if atom_id in owners.get(context_id, ()):
                # This context was retrieved FOR this atom: never gated away.
                reason = "provenance"
            elif is_query_context(context_id):
                # Query-level context: bears on every atom.
                reason = "query_context"
            elif policy == "provenance" and _within_neighbor_window(
                atom_id, owners.get(context_id, ()), order, neighbor_window
            ):
                reason = "neighbor"
            else:
                sim = gate.atom_context(atom_pos[atom_id], context_pos[context_id])
                if sim >= gate_threshold:
                    # Semantic rescue: evidence bearing on an atom that did not
                    # retrieve it. This is what the A*C cross product buys.
                    reason = "gate"

            if reason is None:
                continue
            seen.add(pair)
            pairs.append(pair)
            if reason == "provenance":
                num_provenance += 1
            elif reason == "query_context":
                num_query_context += 1
            elif reason == "neighbor":
                num_neighbor += 1
            else:
                num_gate_rescued += 1

    coverage.update(
        pairs_selected=len(pairs),
        pairs_pruned=max(0, pairs_possible - len(pairs)),
        gate_threshold=gate_threshold,
        gate_backend=gate.backend,
        neighbor_window=(neighbor_window if policy == "provenance" else None),
        num_provenance=num_provenance,
        num_query_context=num_query_context,
        num_neighbor=num_neighbor,
        num_gate_rescued=num_gate_rescued,
    )
    return pairs, coverage


def _within_neighbor_window(
    atom_id: str,
    owner_ids,
    order: dict[str, int],
    window: int,
) -> bool:
    """Whether ``atom_id`` is within ``window`` positions of an owning atom."""
    if window <= 0 or not owner_ids:
        return False
    pos = order.get(atom_id)
    if pos is None:
        return False
    return any(
        abs(pos - order[owner_id]) <= window
        for owner_id in owner_ids
        if owner_id in order
    )


# ----------------------------------------------------------------------------
# Context-context pair selection.
# ----------------------------------------------------------------------------


def select_context_context_pairs(
    contexts: dict[str, Context],
    *,
    policy: str = "all_pairs",
    gate_threshold: float = 0.20,
    gate: _PairGate | None = None,
    gate_context_ids: Sequence[str] | None = None,
) -> tuple[list[tuple[str, str]], dict[str, object]]:
    """Select unordered ``(context_i, context_j)`` pairs, ``i < j`` by sorted id.

    Matches ``combinations(sorted(contexts.keys()), 2)`` under ``"all_pairs"`` so
    the faithful path is order-identical to the original implementation. The
    caller scores each pair in one or both directions.

    Args:
        contexts: The contexts, keyed by id.
        policy: One of :data:`NLI_PAIR_POLICIES`. ``"provenance"`` gates
            context-context pairs the same way ``"gated"`` does -- provenance
            only constrains the atom-context phase.
        gate_threshold: Similarity at or above which a pair is admitted.
        gate: A shared :class:`_PairGate`; required for gated policies.
        gate_context_ids: Context id order the gate is indexed by.

    Returns:
        ``(pairs, coverage)``.

    Raises:
        ValueError: If ``policy`` is unknown, or a gated policy has no gate.
    """
    if policy not in NLI_PAIR_POLICIES:
        raise ValueError(
            f"Unknown NLI pair policy: {policy!r} "
            f"(expected one of {list(NLI_PAIR_POLICIES)})."
        )

    sorted_ids = sorted(contexts.keys())
    num_contexts = len(sorted_ids)
    # Unordered pairs; the original scored each in BOTH directions.
    pairs_possible = num_contexts * (num_contexts - 1) // 2

    coverage: dict[str, object] = {
        "policy": policy,
        "num_contexts": num_contexts,
        "pairs_possible": pairs_possible,
    }

    all_pairs = list(combinations(sorted_ids, 2))

    if policy == "all_pairs":
        coverage.update(
            pairs_selected=len(all_pairs),
            pairs_pruned=max(0, pairs_possible - len(all_pairs)),
            gate_threshold=None,
        )
        return all_pairs, coverage

    if gate is None or gate_context_ids is None:
        raise ValueError(
            f"policy={policy!r} requires a gate; pass gate/gate_context_ids "
            "(see build_gate)."
        )
    context_pos = {cid: i for i, cid in enumerate(gate_context_ids)}

    pairs = [
        (ci, cj)
        for ci, cj in all_pairs
        if gate.context_context(context_pos[ci], context_pos[cj]) >= gate_threshold
    ]

    coverage.update(
        pairs_selected=len(pairs),
        pairs_pruned=max(0, pairs_possible - len(pairs)),
        gate_threshold=gate_threshold,
        gate_backend=gate.backend,
    )
    return pairs, coverage


# ----------------------------------------------------------------------------
# Near-duplicate context dedup.
# ----------------------------------------------------------------------------


def dedup_contexts_near(
    contexts: dict[str, Context],
    atoms: dict[str, Atom],
    *,
    threshold: float = 0.92,
    use_summary: bool = True,
    embedding_model: str = "all-MiniLM-L6-v2",
    gate: _PairGate | None = None,
    gate_context_ids: Sequence[str] | None = None,
) -> tuple[dict[str, Context], dict[str, Atom], dict[str, object]]:
    """Collapse near-duplicate contexts, *merging* ownership onto the survivor.

    ``remove_duplicated_contexts`` matches exact text only, so retrieval across
    ``A * k`` queries leaves heavy near-duplicates (the same page with different
    boilerplate, mirrors, snippet-vs-fulltext variants) -- each of which is
    quadratically expensive downstream.

    Unlike that function, which deletes the losing duplicate from *its own*
    atom's context dict and leaves the survivor claiming a single atom, here every
    owning atom of a collapsed context is repointed at the survivor. The result is
    strictly more coverage than exact-text dedup, not less.

    Best run *after* summarization: summaries are short and synthetic, which makes
    near-duplicate detection both cheaper and more accurate than on raw pages.

    Args:
        contexts: The contexts, keyed by id.
        atoms: The atoms, keyed by id; their ``.contexts`` are updated in place.
        threshold: Similarity at or above which two contexts are duplicates.
        use_summary: Compare summaries rather than full text.
        embedding_model: Model for the similarity backend.
        gate: An existing gate to reuse; built on demand when omitted.
        gate_context_ids: Context id order ``gate`` is indexed by.

    Returns:
        ``(contexts, atoms, coverage)`` with a new contexts dict preserving the
        original iteration order of the survivors.
    """
    context_ids = list(contexts.keys())
    coverage: dict[str, object] = {
        "contexts_before": len(context_ids),
        "threshold": threshold,
    }

    if len(context_ids) < 2:
        coverage.update(
            contexts_after=len(context_ids), collapsed=0, owners_merged=0
        )
        return contexts, atoms, coverage

    if gate is None or gate_context_ids is None:
        texts = [_pair_text(contexts[cid], use_summary) for cid in context_ids]
        # Atom side unused here, so encode contexts only.
        gate = _PairGate([], texts, model_name=embedding_model)
        gate_context_ids = context_ids
    context_pos = {cid: i for i, cid in enumerate(gate_context_ids)}

    owners = context_owners(atoms, contexts)

    # Greedy agglomerative pass: first occurrence wins, later near-duplicates
    # collapse onto it.
    survivors: list[str] = []
    collapsed_into: dict[str, str] = {}
    for context_id in context_ids:
        match = None
        for survivor_id in survivors:
            sim = gate.context_context(
                context_pos[context_id], context_pos[survivor_id]
            )
            if sim >= threshold:
                match = survivor_id
                break
        if match is None:
            survivors.append(context_id)
        else:
            collapsed_into[context_id] = match

    # Repoint every owner of a collapsed context at its survivor, so no evidence
    # is lost. This is the part exact-text dedup gets wrong.
    owners_merged = 0
    for dup_id, survivor_id in collapsed_into.items():
        survivor = contexts[survivor_id]
        for atom_id in owners.get(dup_id, ()):
            atom = atoms.get(atom_id)
            if atom is None:
                continue
            atom.contexts.pop(dup_id, None)
            if survivor_id not in atom.contexts:
                atom.contexts[survivor_id] = survivor
                owners_merged += 1

    # Drop any reference to a collapsed context left anywhere else.
    for atom in atoms.values():
        for dup_id in collapsed_into:
            atom.contexts.pop(dup_id, None)

    kept = {cid: contexts[cid] for cid in survivors}
    coverage.update(
        contexts_after=len(kept),
        collapsed=len(collapsed_into),
        owners_merged=owners_merged,
        gate_backend=gate.backend,
    )
    return kept, atoms, coverage


def prune_dangling_context_refs(
    atoms: dict[str, Atom], contexts: dict[str, Context]
) -> int:
    """Drop ``atom.contexts`` entries whose context is no longer registered.

    Contexts are removed from the pipeline-level dict in several places without
    cleaning the atoms that reference them, which leaves ids that resolve to
    nothing. Selection skips such ids defensively, but pruning them keeps the
    atom's own view honest.

    Returns:
        The number of stale references removed.
    """
    removed = 0
    for atom in atoms.values():
        stale = [cid for cid in atom.contexts if cid not in contexts]
        for context_id in stale:
            del atom.contexts[context_id]
            removed += 1
    return removed
