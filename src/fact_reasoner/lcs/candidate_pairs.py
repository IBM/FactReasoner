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

# Candidate atom-pair selection for the relation miner.
#
# All-pairs relation mining is O(n^2), which the deep-dive (Section 4.5) and the
# research plan (Section 3.3) flag as the scaling bottleneck. This module offers
# three policies to choose which ordered pairs (a_i, a_j) get the expensive
# relation call, and always reports what was pruned so downstream coverage is
# explicit (a score must not silently assume full coverage):
#
#   * "all_pairs" -- every ordered pair. Faithful for the small diagnostic
#     examples; quadratic.
#   * "windowed"  -- only pairs within a sliding order window (radius `window`).
#     Near-linear; captures local discourse structure.
#   * "gated"     -- the window PLUS long-range "callback" pairs that survive a
#     cheap similarity/entity-overlap gate (an atom that echoes an entity from
#     far earlier). Near-linear with long-range recall.
#
# RESPONSE-ANCHORED gating. A fixed order-window is a blunt prefilter: it keeps
# spurious near pairs and drops genuine long-range callbacks. When the ORIGINAL
# RESPONSE is available, ``select(..., response=...)`` refines the window using
# discourse-adjacency signals derived from the response itself:
#   * sentence distance -- each atom is mapped to its originating sentence in the
#     response; atoms in adjacent/nearby sentences are discourse-adjacent.
#   * shared-entity / coreference -- a later atom that reintroduces an entity from
#     far earlier is a genuine callback worth mining beyond the window.
#   * connective cues -- an atom whose originating sentence opens with a discourse
#     connective ("therefore", "however", "because", "although", ...) signals an
#     intended link to what precedes it.
# The window stays the cheap prefilter; these signals PROMOTE out-of-window pairs
# and DEMOTE in-window pairs the response does not actually relate. All decisions
# are recorded in ``coverage`` so the report can quantify the effect. With
# ``response=None`` the behavior is exactly the pre-existing window/gate path.
#
# Pairs are ordered (source before target) by source position, matching the
# atom-id order (a0, a1, ...); direction is meaningful for the relation model.

import re
from typing import Dict, List, Optional, Tuple

from fact_reasoner.core.base import Atom

PAIR_POLICIES = ("all_pairs", "windowed", "gated")
GATE_METHODS = ("embedding", "entity", "none")

# Discourse connectives that, when they open an atom's originating sentence,
# signal an intended link to preceding content (used by response-anchored
# gating). Kept tiny and dependency-free, like ``_STOPWORDS``.
_CONNECTIVES = frozenset(
    """therefore thus hence consequently accordingly so because since as
    however but yet nevertheless nonetheless although though whereas while
    meanwhile then afterward afterwards subsequently later thereafter
    moreover furthermore additionally also besides instead conversely
    otherwise regardless despite""".split()
)

# Regex to split a response into sentences (dependency-free; splits on
# sentence-final punctuation followed by whitespace).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Lightweight stopword list for the entity-overlap gate; kept tiny and
# dependency-free (the goal is a cheap prune, not linguistic accuracy).
_STOPWORDS = frozenset(
    """a an the of to in on at for and or but is are was were be been being this
    that these those it its as by with from into over under after before during
    he she they them his her their we you i not no than then so such which who
    whom whose has have had do does did will would can could may might must""".split()
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _content_tokens(text: str) -> set:
    """Return the set of lower-cased content tokens (stopwords removed)."""
    return {
        t for t in (m.group(0).lower() for m in _TOKEN_RE.finditer(text or ""))
        if t not in _STOPWORDS and len(t) > 1
    }


def _ordered_atoms(atoms: Dict[str, Atom]) -> List[Atom]:
    """Return atoms in source order.

    Atom ids are of the form ``a0, a1, ...`` which encode source position, so we
    sort by the trailing integer when present, falling back to string order.
    """

    def key(item):
        atom_id = item[0]
        m = re.search(r"(\d+)$", atom_id)
        return (0, int(m.group(1))) if m else (1, atom_id)

    return [atom for _, atom in sorted(atoms.items(), key=key)]


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity of two token sets (0 if both empty)."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# ----------------------------------------------------------------------------
# Response-anchored discourse signals.
# ----------------------------------------------------------------------------


def _split_sentences(response: str) -> List[str]:
    """Split a response into sentences (dependency-free)."""
    if not response:
        return []
    return [s for s in _SENTENCE_SPLIT_RE.split(response.strip()) if s.strip()]


def _map_atoms_to_sentences(
    ordered: List[Atom], response: str
) -> List[Optional[int]]:
    """Map each atom (in source order) to the index of its originating sentence.

    An atom is aligned to the response sentence whose content-token set it most
    overlaps (Jaccard). Ties and no-overlap resolve to ``None`` (unaligned), which
    the caller treats as "no sentence-distance signal available for this atom".

    Args:
        ordered: Atoms in source order.
        response: The original response text the atoms were decomposed from.

    Returns:
        A list parallel to ``ordered`` of sentence indices (or ``None``).
    """
    sentences = _split_sentences(response)
    if not sentences:
        return [None] * len(ordered)
    sent_tokens = [_content_tokens(s) for s in sentences]
    mapping: List[Optional[int]] = []
    for atom in ordered:
        atok = _content_tokens(atom.text)
        best_idx: Optional[int] = None
        best_sim = 0.0
        for si, stok in enumerate(sent_tokens):
            sim = _jaccard(atok, stok)
            if sim > best_sim:
                best_sim = sim
                best_idx = si
        mapping.append(best_idx if best_sim > 0.0 else None)
    return mapping


def _opens_with_connective(text: str) -> bool:
    """Whether an atom's text begins with a discourse connective cue."""
    m = _TOKEN_RE.search(text or "")
    return bool(m) and m.group(0).lower() in _CONNECTIVES


class _EmbeddingGate:
    """Lazy embedding-similarity gate.

    Tries to use ``sentence-transformers`` for cosine similarity; if it is not
    installed, transparently falls back to token Jaccard so the miner never hard-
    depends on the embedding stack. The chosen backend is recorded in
    :attr:`backend` for the coverage report.
    """

    def __init__(self, texts: List[str], model_name: str = "all-MiniLM-L6-v2"):
        self.backend = "jaccard"
        self._vectors = None
        self._token_sets = [_content_tokens(t) for t in texts]
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            import numpy as np  # noqa: F401

            model = SentenceTransformer(model_name)
            emb = model.encode(texts, normalize_embeddings=True)
            self._vectors = emb
            self.backend = f"sbert:{model_name}"
        except Exception:
            # No sentence-transformers (or load failure): stay on Jaccard.
            self._vectors = None

    def similarity(self, i: int, j: int) -> float:
        if self._vectors is not None:
            import numpy as np

            return float(np.dot(self._vectors[i], self._vectors[j]))
        return _jaccard(self._token_sets[i], self._token_sets[j])


def select(
    atoms: Dict[str, Atom],
    *,
    response: str,
    policy: str = "windowed",
    window: int = 4,
    gate: str = "embedding",
    gate_threshold: float = 0.3,
    embedding_model: str = "all-MiniLM-L6-v2",
    discourse_gate_threshold: float = 0.2,
    discourse_sentence_span: int = 2,
) -> Tuple[List[Tuple[str, str]], Dict[str, object]]:
    """Select candidate ordered atom pairs for relation mining.

    Selection is always response-anchored: for windowed/gated policies the window
    is refined with discourse-adjacency signals derived from the response ---
    out-of-window pairs the response relates (shared entity, connective, near
    sentences) are PROMOTED, and in-window pairs the response does not relate are
    DEMOTED. (The ``all_pairs`` policy takes every ordered pair regardless.)

    Args:
        atoms: The atoms, keyed by id. Source order is taken from the ids.
        response: The original response the atoms were decomposed from (REQUIRED;
            used for the discourse-adjacency refinement).
        policy: One of ``PAIR_POLICIES``.
        window: Order-window radius (used by ``"windowed"`` and ``"gated"``): a
            pair ``(a_i, a_j)`` with ``0 < j - i <= window`` is inside the window.
        gate: The long-range gate for ``"gated"``: ``"embedding"`` (cosine
            similarity, falling back to Jaccard), ``"entity"`` (content-token
            Jaccard), or ``"none"`` (no long-range pairs, i.e. windowed only).
        gate_threshold: Similarity threshold above which an out-of-window pair is
            admitted as a long-range callback.
        embedding_model: Sentence-transformers model name for the embedding gate.
        discourse_gate_threshold: Content-token Jaccard above which an
            out-of-window pair counts as a shared-entity discourse callback.
        discourse_sentence_span: Sentence-distance radius within which two atoms
            count as sentence-adjacent for the discourse signal.

    Returns:
        A tuple ``(pairs, coverage)``:
          * ``pairs``: list of ordered ``(source_id, target_id)`` tuples, source
            before target in source order.
          * ``coverage``: a dict describing what was considered/scored/pruned,
            so callers can report coverage explicitly.

    Raises:
        ValueError: If ``response`` is empty, or ``policy`` / ``gate`` is unknown.
    """
    if not response or not str(response).strip():
        raise ValueError(
            "A non-empty response is required: candidate selection is always "
            "response-anchored."
        )
    if policy not in PAIR_POLICIES:
        raise ValueError(
            f"Unknown pair policy: {policy!r} (expected one of {list(PAIR_POLICIES)})."
        )
    if gate not in GATE_METHODS:
        raise ValueError(
            f"Unknown gate method: {gate!r} (expected one of {list(GATE_METHODS)})."
        )

    ordered = _ordered_atoms(atoms)
    n = len(ordered)
    ids = [a.id for a in ordered]
    total_ordered_pairs = n * (n - 1)  # all ordered i != j pairs (forward + back)

    coverage: Dict[str, object] = {
        "policy": policy,
        "num_atoms": n,
        "total_ordered_pairs": total_ordered_pairs,
    }

    # all_pairs: every ordered pair (both directions).
    if policy == "all_pairs":
        pairs = [
            (ids[i], ids[j]) for i in range(n) for j in range(n) if i != j
        ]
        coverage.update(
            pairs_selected=len(pairs),
            pairs_pruned=total_ordered_pairs - len(pairs),
            window=None,
            gate=None,
            discourse_anchored=False,  # all_pairs takes every pair, no refinement
        )
        return pairs, coverage

    # windowed / gated: forward window pairs (source before target).
    window_pairs: List[Tuple[str, str]] = []
    for i in range(n):
        for j in range(i + 1, min(i + window + 1, n)):
            window_pairs.append((ids[i], ids[j]))

    callback_pairs: List[Tuple[str, str]] = []
    gate_backend = None
    if policy == "gated" and gate != "none":
        texts = [a.text for a in ordered]
        if gate == "embedding":
            g = _EmbeddingGate(texts, model_name=embedding_model)
            sim = g.similarity
            gate_backend = g.backend
        else:  # entity
            token_sets = [_content_tokens(t) for t in texts]
            sim = lambda i, j: _jaccard(token_sets[i], token_sets[j])  # noqa: E731
            gate_backend = "entity:jaccard"

        # Long-range forward pairs beyond the window that survive the gate.
        for i in range(n):
            for j in range(i + window + 1, n):
                if sim(i, j) >= gate_threshold:
                    callback_pairs.append((ids[i], ids[j]))

    # Response-anchored refinement: promote out-of-window discourse callbacks and
    # demote in-window pairs the response does not actually relate. Always runs
    # (grounding is mandatory).
    discourse_promoted: List[Tuple[str, str]] = []
    discourse_demoted: set = set()
    index_of = {ids[i]: i for i in range(n)}
    token_sets = [_content_tokens(a.text) for a in ordered]
    sent_idx = _map_atoms_to_sentences(ordered, response)
    opens_conn = [_opens_with_connective(a.text) for a in ordered]

    def _discourse_adjacent(i: int, j: int) -> bool:
        """Does the response draw a link from atom i to atom j?"""
        # Shared-entity / coreference callback.
        if _jaccard(token_sets[i], token_sets[j]) >= discourse_gate_threshold:
            return True
        # Sentence adjacency in the response.
        si, sj = sent_idx[i], sent_idx[j]
        if si is not None and sj is not None and 0 <= sj - si <= discourse_sentence_span:
            return True
        # The target opens with a connective AND is a near neighbor: an
        # intended discourse link to preceding content.
        if opens_conn[j] and 0 < j - i <= window:
            return True
        return False

    # Promote out-of-window forward pairs the response relates.
    already = set(window_pairs) | set(callback_pairs)
    for i in range(n):
        for j in range(i + window + 1, n):
            pair = (ids[i], ids[j])
            if pair not in already and _discourse_adjacent(i, j):
                discourse_promoted.append(pair)

    # Demote in-window pairs the response does not relate.
    for (sid, tid) in window_pairs:
        i, j = index_of[sid], index_of[tid]
        if not _discourse_adjacent(i, j):
            discourse_demoted.add((sid, tid))

    discourse_stats = {
        "discourse_anchored": True,
        "num_sentences": len(_split_sentences(response)),
        "num_atoms_aligned": sum(1 for s in sent_idx if s is not None),
        "num_promoted": len(discourse_promoted),
        "num_demoted": len(discourse_demoted),
    }

    # Deduplicate while preserving order (window first, then callbacks, then
    # discourse-promoted), dropping any demoted in-window pairs.
    seen = set()
    pairs: List[Tuple[str, str]] = []
    for p in window_pairs + callback_pairs + discourse_promoted:
        if p in seen or p in discourse_demoted:
            continue
        seen.add(p)
        pairs.append(p)

    # Forward-only candidate universe for this policy (source before target).
    forward_universe = n * (n - 1) // 2
    coverage.update(
        window=window,
        gate=(gate if policy == "gated" else None),
        gate_backend=gate_backend,
        gate_threshold=(gate_threshold if policy == "gated" else None),
        num_window_pairs=len(window_pairs),
        num_callback_pairs=len(callback_pairs),
        pairs_selected=len(pairs),
        forward_pairs_possible=forward_universe,
        pairs_pruned=forward_universe - len(pairs),
    )
    coverage.update(discourse_stats)
    return pairs, coverage
