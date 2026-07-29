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

"""Configuration for NLI relation-extraction cost control.

:func:`fact_reasoner.core.utils.build_relations` sends one LLM call per candidate
pair: ``A * C`` for the atom-context phase and ``C * (C - 1)`` for the
context-context phase (both directions). Since contexts are retrieved *per atom*
(``C ~ A * k`` for ``k = top_k``), that is quadratic in the number of atoms for v2
and effectively cubic for v3.

Every knob here defaults to today's behavior, so :data:`FAITHFUL` reproduces the
published numbers with no flags. The efficiency features are opt-in.
"""

from dataclasses import dataclass
from typing import Optional

NLI_PAIR_POLICIES = ("all_pairs", "gated", "provenance")


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

            The default of 0.10 was chosen by replaying policies against recorded
            verdicts from two live llama-3.3-70b runs (see
            ``scripts/e2e_nli_live.py``). Worst case across both runs, the
            provenance policy holds recall at 1.000 up to 0.10 and slips to 0.923 at
            0.15, where a low-overlap entailment with token Jaccard 0.11 is pruned.

            Two caveats worth knowing before raising this. The model is not
            deterministic -- the same inputs yielded 13 non-neutral pairs in one run
            and 11 in another -- so a single run cannot establish recall; take the
            worst case over several. And the threshold interacts with the gate
            backend: without sentence-transformers the token-Jaccard fallback scores
            genuinely related text far lower than embeddings would, so a threshold
            tuned on one backend does not transfer to the other.
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
    gate_threshold: float = 0.10
    neighbor_window: int = 1
    dedup_near_duplicates: bool = False
    dedup_threshold: float = 0.92
    ctx_ctx_single_direction_cascade: bool = False
    merge_phases: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"
    cache_dir: Optional[str] = None

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

        When this is False the gate is never constructed, so the faithful path
        does not even import sentence-transformers.
        """
        return self.policy in ("gated", "provenance") or self.dedup_near_duplicates


#: The default configuration: every efficiency feature off, behavior identical to
#: the original implementation.
FAITHFUL = NLIPairConfig()

#: Presets addressable by name from the runner's version table and the CLI.
NLI_PAIR_CONFIGS = {
    "faithful": FAITHFUL,
    "gated": NLIPairConfig(policy="gated"),
    "provenance": NLIPairConfig(
        policy="provenance",
        dedup_near_duplicates=True,
        ctx_ctx_single_direction_cascade=True,
        merge_phases=True,
    ),
}


def get_pair_config(name: str) -> NLIPairConfig:
    """Look up a named preset from :data:`NLI_PAIR_CONFIGS`."""
    if name not in NLI_PAIR_CONFIGS:
        raise ValueError(
            f"Unknown NLI pair config: {name!r} "
            f"(expected one of {list(NLI_PAIR_CONFIGS)})."
        )
    return NLI_PAIR_CONFIGS[name]
