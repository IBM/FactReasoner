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

"""The discourse floor: coherence as measured by entity/lexical cohesion.

These wrap the three DiscoScore metrics that score a *single* document, plus a
local reimplementation used when DiscoScore is not installed. They are the
classical answer to "is this text coherent?", they predate the LLM era, and they
are the floor a probabilistic relation model has to clear.

Why only three of DiscoScore's metrics
--------------------------------------
DiscoScore (Zhao, Strube and Eger, EACL 2023) presents itself as a *reference-based*
metric, and every method on its scorer takes ``(sys, ref)``. Most are therefore
unusable here: assessing a standalone response's coherence gives us no reference to
compare against. But four of them ignore the reference argument entirely --
``scorer.py`` forwards only ``sys``:

===============  =================================  ==========
method           body                               usable
===============  =================================  ==========
``RC``           ``discourse.RC(sys)``              yes
``LC``           ``discourse.LC(sys)``              yes
``EntityGraph``  ``discourse.EntityGraph(sys)``     yes
``LexicalGraph`` needs external word vectors        skipped
``DS_Focus_*``   uses ``ref``                       no
``DS_SENT_*``    uses ``ref``                       no
===============  =================================  ==========

So ``RC``, ``LC`` and ``EntityGraph`` are genuine single-document coherence
measures, directly comparable to the LCS, and ``EntityGraph`` doubles as the
entity-grid baseline -- no separate reimplementation of Barzilay and Lapata needed.

What these metrics can and cannot see
-------------------------------------
All three reduce to **noun-lemma repetition across sentences**:

* ``RC`` -- repeated noun mentions over distinct nouns.
* ``LC`` -- nouns occurring in more than one sentence, over distinct nouns.
* ``EntityGraph`` -- mean sentence-adjacency weight, an edge of weight
  ``1/(j-i)`` wherever sentences *i* and *j* share a noun lemma.

None of them models entailment, relation type, or contradiction. A response can
contradict itself in every sentence and still score at the ceiling, so long as its
nouns recur. That is the cohesion/coherence distinction the LCS exists to draw, and
it is a property of these metrics' definitions rather than a claim about their
accuracy -- ``tests/test_coherence_baselines.py`` pins it with a noun-matched
contradictory pair.

Availability
------------
``disco_score`` is not on PyPI; it installs from
``git+https://github.com/AIPHES/DiscoScore.git`` and calls
``spacy_udpipe.download("en")`` at *import* time, which needs the network. Both
facts make it unfit for a hard dependency, so it is imported lazily and every
baseline here abstains -- ``BaselineScore.score is None`` with a recorded reason --
rather than raising when it is missing. Set ``allow_fallback`` to score with the
local noun-repetition implementation instead, which reproduces ``RC``/``LC``/
``EntityGraph`` closely enough to make the same argument without the dependency.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from typing import Any

from fact_reasoner.coherence_baselines.base import BaselineScore

#: Cache for the lazily-built DiscoScore scorer: ``None`` until first use, then
#: either the scorer or the import error that explains its absence.
_DISCO: dict[str, Any] = {}

#: Rough sentence splitter for the fallback path. DiscoScore uses NLTK's
#: ``sent_tokenize``; this is deliberately simpler, since the fallback exists to
#: keep the *argument* runnable without the dependency, not to reproduce
#: DiscoScore's numbers to the decimal.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

#: Tokens the fallback treats as nouns. Without a POS tagger we approximate by
#: taking alphabetic words of four or more characters that are not function
#: words. This over-counts, but only in ways that affect both members of a
#: compared pair equally, which is what the cohesion test needs.
_STOP = frozenset(
    ["about", "above", "after", "again", "against", "because", "before", "being", "below", "between", "both", "during", "each", "further", "having", "into", "more", "most", "other", "same", "some", "such", "than", "that", "their", "them", "then", "there", "these", "they", "this", "those", "through", "under", "until", "very", "were", "what", "when", "where", "which", "while", "with", "would", "your", "also", "from", "have", "here", "just", "like", "only", "over", "than", "they", "will", "your"]
)


def _load_disco(model_name: str, device: str):
    """Return a DiscoScore scorer, or None with the reason cached.

    The import is deferred to here on purpose: ``disco_score`` runs a model
    download at import time, so importing it eagerly would make this module
    unimportable offline.
    """
    key = f"{model_name}@{device}"
    if key in _DISCO:
        return _DISCO[key]
    try:
        from disco_score import DiscoScorer

        _DISCO[key] = DiscoScorer(device=device, model_name=model_name)
    except Exception as e:  # noqa: BLE001 - any failure means "unavailable"
        _DISCO[key] = None
        _DISCO[f"{key}:error"] = f"{type(e).__name__}: {e}"
    return _DISCO[key]


def _fallback_grid(response: str) -> tuple[dict[str, list[int]], int]:
    """A noun-lemma-by-sentence grid, approximated without a POS tagger.

    Returns:
        ``(grid, num_sentences)`` where ``grid`` maps a token to the indices of
        the sentences it appears in.
    """
    sents = [s for s in _SENT_SPLIT.split(response.strip()) if s.strip()]
    grid: dict[str, list[int]] = {}
    for i, sent in enumerate(sents):
        for word in re.findall(r"[A-Za-z]+", sent.lower()):
            if len(word) < 4 or word in _STOP:
                continue
            # Crude singular/plural folding, standing in for lemmatization.
            plural = word.endswith("s") and not word.endswith("ss")
            lemma = word[:-1] if plural else word
            grid.setdefault(lemma, [])
            if i not in grid[lemma]:
                grid[lemma].append(i)
    return grid, len(sents)


class _DiscourseBaseline:
    """Shared plumbing for the three intrinsic DiscoScore metrics.

    Args:
        allow_fallback: When True and ``disco_score`` is unavailable, score with
            the local noun-repetition implementation instead of abstaining.
        model_name: Passed to ``DiscoScorer``.
        device: Passed to ``DiscoScorer``; defaults to CPU so the baseline runs
            without a GPU.
    """

    #: Name of the ``DiscoScorer`` method this baseline calls.
    disco_method = ""

    def __init__(
        self,
        *,
        allow_fallback: bool = True,
        model_name: str = "bert-base-uncased",
        device: str = "cpu",
    ):
        self.allow_fallback = allow_fallback
        self.model_name = model_name
        self.device = device

    # -- fallback hook -------------------------------------------------------

    def _fallback_score(self, grid: dict[str, list[int]], n_sents: int) -> float:
        """Compute this metric from a noun grid. Overridden per metric."""
        raise NotImplementedError

    # -- entry point ---------------------------------------------------------

    def score(self, atoms: Sequence[str], response: str) -> BaselineScore:
        """Score one response's cohesion.

        Args:
            atoms: Unused. These metrics read the prose, not the claim list --
                which is itself informative: they cannot be affected by the
                decomposition at all, so they see nothing the atomizer found.
            response: The response text.

        Returns:
            The :class:`BaselineScore`, abstaining when the text has no usable
            noun grid or when DiscoScore is unavailable and fallback is off.
        """
        started = time.time()
        n_atoms = len([a for a in atoms if a and a.strip()])
        text = (response or "").strip()
        if not text:
            return BaselineScore(
                name=self.name,
                score=None,
                atoms_scored=n_atoms,
                diagnostics={"reason": "empty response"},
            )

        scorer = _load_disco(self.model_name, self.device)
        if scorer is not None:
            try:
                # The reference argument is ignored by these three metrics (see
                # the module docstring), so an empty list is passed to satisfy
                # the signature.
                raw = getattr(scorer, self.disco_method)(text.lower(), [])
                return self._finish(raw, n_atoms, "disco_score", started)
            except ZeroDivisionError:
                # RC and LC divide by the number of distinct nouns; EntityGraph
                # by the sentence count. Noun-free text hits this, and it means
                # "no cohesion signal", not "incoherent".
                return BaselineScore(
                    name=self.name,
                    score=None,
                    atoms_scored=n_atoms,
                    diagnostics={"reason": "empty entity grid (no nouns found)"},
                )
            except Exception as e:  # noqa: BLE001
                if not self.allow_fallback:
                    return BaselineScore(
                        name=self.name,
                        score=None,
                        atoms_scored=n_atoms,
                        diagnostics={"reason": f"disco_score failed: {e}"},
                    )

        if not self.allow_fallback:
            return BaselineScore(
                name=self.name,
                score=None,
                atoms_scored=n_atoms,
                diagnostics={
                    "reason": "disco_score unavailable",
                    "import_error": _DISCO.get(
                        f"{self.model_name}@{self.device}:error"
                    ),
                },
            )

        grid, n_sents = _fallback_grid(text)
        if not grid or n_sents == 0:
            return BaselineScore(
                name=self.name,
                score=None,
                atoms_scored=n_atoms,
                diagnostics={"reason": "empty entity grid (no nouns found)"},
            )
        return self._finish(
            self._fallback_score(grid, n_sents), n_atoms, "fallback", started
        )

    def _finish(
        self, raw: float, n_atoms: int, impl: str, started: float
    ) -> BaselineScore:
        """Clamp a raw metric value into ``[0, 1]`` and package it.

        ``RC`` is a ratio of mention counts to distinct nouns and so is not
        bounded above by one; the raw value is kept in diagnostics so the
        clamping is visible rather than silent.
        """
        return BaselineScore(
            name=self.name,
            score=max(0.0, min(1.0, float(raw))),
            atoms_scored=n_atoms,
            diagnostics={
                "raw": float(raw),
                "implementation": impl,
                "clamped": not (0.0 <= float(raw) <= 1.0),
                "seconds": round(time.time() - started, 3),
            },
        )


class DiscoScoreRC(_DiscourseBaseline):
    """Repeated-mention count over distinct nouns (DiscoScore ``RC``)."""

    name = "discourse_rc"
    disco_method = "RC"

    def _fallback_score(self, grid, n_sents):
        repeated = sum(len(v) for v in grid.values() if len(v) > 1)
        return repeated / len(grid)


class DiscoScoreLC(_DiscourseBaseline):
    """Fraction of nouns spanning more than one sentence (DiscoScore ``LC``)."""

    name = "discourse_lc"
    disco_method = "LC"

    def _fallback_score(self, grid, n_sents):
        spanning = len([v for v in grid.values() if len(v) > 1])
        return spanning / len(grid)


class EntityGraphCoherence(_DiscourseBaseline):
    """Mean sentence-adjacency weight over shared noun lemmas.

    This is the entity-grid baseline: DiscoScore's ``EntityGraph`` follows
    Barzilay and Lapata's grid, projected to a sentence graph whose edge weights
    are ``1/(j-i)``, then averaged over sentences.
    """

    name = "discourse_entity_graph"
    disco_method = "EntityGraph"

    def _fallback_score(self, grid, n_sents):
        total = 0.0
        for idxs in grid.values():
            uniq = sorted(set(idxs))
            for a_i in range(len(uniq)):
                for b_i in range(a_i + 1, len(uniq)):
                    total += 1.0 / (uniq[b_i] - uniq[a_i])
        return total / n_sents


#: The three intrinsic discourse baselines, ready to run.
DISCOURSE_BASELINES = (
    DiscoScoreRC(),
    DiscoScoreLC(),
    EntityGraphCoherence(),
)
