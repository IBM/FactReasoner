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

"""Configuration for NLI relation-extraction cost control.

:func:`fact_reasoner.core.utils.build_relations` sends one LLM call per candidate
pair: ``A * C`` for the atom-context phase and ``C * (C - 1)`` for the
context-context phase (both directions). Since contexts are retrieved *per atom*
(``C ~ A * k`` for ``k = top_k``), that is quadratic in the number of atoms for v2
and effectively cubic for v3.

Every knob here defaults to today's behavior, so :data:`FAITHFUL` reproduces the
published numbers with no flags. The efficiency features are opt-in, selected as a
bundle by ``--nli-mode`` (see :data:`NLI_MODES`) or individually via the
``--nli-*`` flags, which override whichever mode was chosen.
"""

from dataclasses import dataclass
from typing import Optional

NLI_PAIR_POLICIES = ("all_pairs", "gated", "provenance")

#: Preset selectors for ``--nli-mode`` / ``FactualityRunner(nli_mode=...)``.
#:
#: A mode names a whole *preset*, whereas a :data:`NLI_PAIR_POLICIES` value names
#: one *mechanism* (the single ``policy`` field). The two vocabularies overlap on
#: purpose: ``"all_pairs"`` means the same thing in both (score every enumerated
#: pair), so ``--nli-mode all_pairs`` and ``--nli-pair-policy all_pairs`` read
#: consistently -- just as ``"gated"``/``"provenance"`` already appear in both.
#:
#: Where they differ is bundling. ``"fast"`` selects the provenance preset below,
#: which sets ``policy="provenance"`` **and** flips ``dedup_near_duplicates``,
#: ``ctx_ctx_single_direction_cascade`` and ``merge_phases`` -- strictly more than
#: the bare ``"provenance"`` policy does.
NLI_MODES = ("all_pairs", "fast")


@dataclass(frozen=True)
class NLIPairConfig:
    """Knobs controlling which NLI pairs are scored, and how they are cached.

    Attributes:
        policy: Candidate-pair policy, one of :data:`NLI_PAIR_POLICIES`.

            * ``"all_pairs"`` -- every enumerated pair, i.e. today's exact
              behavior (the default; reproduces published numbers).
            * ``"gated"`` -- pairs surviving a cheap embedding/Jaccard similarity
              gate.
            * ``"provenance"`` -- atom-context pairs restricted to the atoms that
              actually retrieved each context (plus query-level contexts,
              near-neighbor atoms, and a gate rescue for cross-atom evidence);
              context-context pairs gated.
        gate_threshold: Similarity at or above which a pair is admitted by the
            gate. Deliberately low: a false prune silently weakens an atom's
            evidence, while a false keep only costs money.

            The default of 0.20 is calibrated for the **embedding** backend. It was
            measured by replaying policies against recorded live llama-3.3-70b
            verdicts on a 20-atom narrative with 60 retrieved contexts (see
            ``scripts/e2e_nli_live.py --example``): with sbert
            ``all-MiniLM-L6-v2``, provenance holds recall at 1.000 through 0.20,
            slips to 0.985 at 0.30 and 0.833 at 0.40.

            **Make sure the embedding model actually loads before relying on any
            gated policy.** sentence-transformers is a base dependency, so the
            token-Jaccard fallback now signals a model *load* failure (offline,
            corrupt cache, incomplete install) rather than a missing package -- but
            the cost of landing on it is unchanged. On the same example that
            fallback lost 22 of 72 non-neutral relations (recall 0.694) and moved
            the factuality score by 0.05, and *no* threshold reached full recall --
            even 0.05 lost 6. It keeps the pipeline running; it is not a substitute
            for embeddings, because lexical overlap misses pairs that are
            semantically related through entities and events rather than shared
            vocabulary.

            Two further caveats. The saving is workload-dependent, not a constant:
            the same policy that prunes ~5x on unrelated subtopics prunes only
            ~1.2x on a narrative where every atom shares characters, because there
            the cross-product contains little genuine waste. And the model is not
            deterministic -- identical inputs yielded 72 and 66 non-neutral pairs
            across runs -- so a single run cannot establish recall; take the worst
            case over several.
        neighbor_window: For ``"provenance"``, how many atoms on either side of an
            owning atom also get compared against the context.
        dedup_near_duplicates: Collapse near-duplicate contexts before mining.
            Because both dominant cost terms are super-linear in ``C``, this has
            quadratic leverage.
        dedup_threshold: Similarity at or above which two contexts are considered
            near-duplicates.
        ctx_ctx_single_direction_cascade: Score one direction of each
            context-context pair, mirroring only where the second direction can
            change the reconciled outcome.
        merge_phases: Issue the phases as a single fan-out instead of separate
            ones. A latency win only -- the call count is unchanged.
        embedding_model: Sentence-transformers model for the gate. Falls back to
            token Jaccard when sentence-transformers is unavailable.
        cache_dir: Directory for the cross-run NLI verdict cache. ``None``
            disables caching.
    """

    policy: str = "all_pairs"
    gate_threshold: float = 0.20
    neighbor_window: int = 1
    dedup_near_duplicates: bool = False
    dedup_threshold: float = 0.92
    ctx_ctx_single_direction_cascade: bool = False
    merge_phases: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"
    cache_dir: str | None = None

    def __post_init__(self):
        if self.policy not in NLI_PAIR_POLICIES:
            raise ValueError(
                f"Unknown NLI pair policy: {self.policy!r} "
                f"(expected one of {list(NLI_PAIR_POLICIES)})."
            )
        if not 0.0 <= self.gate_threshold <= 1.0:
            raise ValueError(
                f"gate_threshold must be in [0, 1], got {self.gate_threshold!r}."
            )
        if not 0.0 <= self.dedup_threshold <= 1.0:
            raise ValueError(
                f"dedup_threshold must be in [0, 1], got {self.dedup_threshold!r}."
            )
        if self.neighbor_window < 0:
            raise ValueError(
                f"neighbor_window must be >= 0, got {self.neighbor_window!r}."
            )

    @property
    def is_faithful(self) -> bool:
        """Whether this config reproduces the original behavior exactly.

        The cache is excluded on purpose: it returns previously computed verdicts,
        so it never changes results, only cost.
        """
        return (
            self.policy == "all_pairs"
            and not self.dedup_near_duplicates
            and not self.ctx_ctx_single_direction_cascade
        )

    @property
    def needs_gate(self) -> bool:
        """Whether any enabled feature requires the embedding gate.

        When this is False the gate is never constructed, so the all-pairs path
        pays no embedding-model load cost at all.
        """
        return self.policy in ("gated", "provenance") or self.dedup_near_duplicates


#: The default configuration: every efficiency feature off, behavior identical to
#: the original implementation.
FAITHFUL = NLIPairConfig()

#: The cost-reducing preset: restrict atom-context pairs to the atoms that
#: actually retrieved each context, gate context-context pairs, collapse
#: near-duplicate contexts, and score one direction per context pair.
_PROVENANCE = NLIPairConfig(
    policy="provenance",
    dedup_near_duplicates=True,
    ctx_ctx_single_direction_cascade=True,
    merge_phases=True,
)

#: Presets addressable by name from ``--nli-mode`` and the runner's ``nli_mode``
#: argument. ``"all_pairs"``/``"fast"`` are the :data:`NLI_MODES` vocabulary;
#: ``"faithful"``/``"provenance"`` are retained aliases for the very same two
#: config objects, since they name the same thing from the implementation's point
#: of view (and ``is_faithful`` still reads naturally against the former).
NLI_PAIR_CONFIGS = {
    "all_pairs": FAITHFUL,
    "fast": _PROVENANCE,
    "gated": NLIPairConfig(policy="gated"),
    # Retained aliases.
    "faithful": FAITHFUL,
    "provenance": _PROVENANCE,
}


def get_pair_config(name: str) -> NLIPairConfig:
    """Look up a named preset from :data:`NLI_PAIR_CONFIGS`."""
    if name not in NLI_PAIR_CONFIGS:
        raise ValueError(
            f"Unknown NLI pair config: {name!r} "
            f"(expected one of {list(NLI_PAIR_CONFIGS)})."
        )
    return NLI_PAIR_CONFIGS[name]
