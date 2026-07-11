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
# Pairs are ordered (source before target) by source position, matching the
# atom-id order (a0, a1, ...); direction is meaningful for the relation model.

import re
from typing import Dict, List, Optional, Tuple

from fact_reasoner.core.base import Atom

PAIR_POLICIES = ("all_pairs", "windowed", "gated")
GATE_METHODS = ("embedding", "entity", "none")

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
    policy: str = "windowed",
    window: int = 4,
    gate: str = "embedding",
    gate_threshold: float = 0.3,
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Tuple[List[Tuple[str, str]], Dict[str, object]]:
    """Select candidate ordered atom pairs for relation mining.

    Args:
        atoms: The atoms, keyed by id. Source order is taken from the ids.
        policy: One of ``PAIR_POLICIES``.
        window: Order-window radius (used by ``"windowed"`` and ``"gated"``): a
            pair ``(a_i, a_j)`` with ``0 < j - i <= window`` is inside the window.
        gate: The long-range gate for ``"gated"``: ``"embedding"`` (cosine
            similarity, falling back to Jaccard), ``"entity"`` (content-token
            Jaccard), or ``"none"`` (no long-range pairs, i.e. windowed only).
        gate_threshold: Similarity threshold above which an out-of-window pair is
            admitted as a long-range callback.
        embedding_model: Sentence-transformers model name for the embedding gate.

    Returns:
        A tuple ``(pairs, coverage)``:
          * ``pairs``: list of ordered ``(source_id, target_id)`` tuples, source
            before target in source order.
          * ``coverage``: a dict describing what was considered/scored/pruned,
            so callers can report coverage explicitly.

    Raises:
        ValueError: If ``policy`` or ``gate`` is unknown.
    """
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

    # Deduplicate while preserving order (window first, then callbacks).
    seen = set()
    pairs: List[Tuple[str, str]] = []
    for p in window_pairs + callback_pairs:
        if p not in seen:
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
    return pairs, coverage
