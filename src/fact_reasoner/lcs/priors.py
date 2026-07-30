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

# Where the coherence MRF's per-atom priors come from.
#
# The coherence MRF (``lcs.relation_miner`` / ``lcs.lcs_scorer``) puts one unary
# factor ``[1-pi_i, pi_i]`` on each atom. Coherence-only scoring uses a uniform
# ``pi_i = 0.5``, which says nothing about whether the atom is true. The
# factuality pipeline (``assessor.FactReasoner``) already computes exactly that:
# the posterior marginal ``q_i = P(a_i = 1 | contexts)``. Feeding those in as the
# priors is the two-stage model:
#
#   Stage 1  FactReasoner over atoms + retrieved contexts -> MAR -> q_i
#   Stage 2  coherence MRF over atoms alone, unary [1-q_i, q_i] + mined
#            atom<->atom relation factors -> MAR/PR/MAP -> LCS
#
# A :class:`PriorProvider` is stage 1 behind one small interface, so the
# coherence pipeline is indifferent to whether the priors came from a live
# factuality run, a cached results file, or nothing at all.
#
# ALIGNMENT. Both pipelines mint atom ids ``a0, a1, ...``, so ids usually line up.
# But ``core.utils.remove_duplicated_atoms`` drops duplicate atoms keeping the
# first-seen key, which makes the surviving id set sparse (``a0, a1, a3, ...``).
# Two independent atomizations of the same response can therefore disagree about
# which text lives at which index -- and matching on id alone would then attach a
# prior to the WRONG atom, which is worse than attaching none. So
# :meth:`AtomPriors.resolve` matches on normalized atom TEXT first and falls back
# to ids only for atoms the text pass missed. The main path avoids the question
# entirely: the provider hands its own atoms to the miner (see ``lcs.pipeline``),
# so the two stages share one atom set by construction.

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from fact_reasoner.core.base import Atom

# The neutral prior: an atom with no factual evidence either way. It is also the
# unary factor that leaves an atom's marginal to its coherence edges alone, which
# is why it is the right value for uncovered atoms.
NEUTRAL_PRIOR = 0.5

# What a provider names as the origin of its priors (recorded for provenance).
PRIOR_SOURCES = ("uniform", "factreasoner", "precomputed", "file")

# How a caller wants low prior coverage handled.
LOW_COVERAGE_POLICIES = ("warn", "raise", "uniform")

# Below this fraction of atoms actually covered by a prior, the resolution is
# flagged as degraded (and `on_low_coverage` decides what to do about it).
DEFAULT_MIN_COVERAGE = 0.5


def _normalize_text(text: str) -> str:
    """Normalize atom text for matching: casefold, collapse space, strip end punct.

    Deliberately conservative -- it should collapse incidental formatting
    differences between two atomizations of the same claim, not merge distinct
    claims.
    """
    collapsed = re.sub(r"\s+", " ", (text or "").strip()).casefold()
    return collapsed.rstrip(" .;,:!?")


def atom_priors_from_results(results: Mapping[str, Any]) -> dict[str, float]:
    """Extract ``{atom_id: P(atom = 1)}`` from a FactReasoner results dict.

    ``FactReasoner.score`` reports the per-atom posteriors in more than one shape,
    and callers hold results that came from different places (a live run, a
    ``jsonl`` evaluation row, a hand-built map). This accepts any of them, trying
    in order:

      1. ``"factuality_score_per_atom"`` -- a list of ``{atom_id: {"score", "support"}}``
         (``assessor.py`` builds this one).
      2. ``"marginals"`` -- a list of ``{"variable", "probabilities": [p0, p1]}``
         as returned by ``inference.run_merlin``.
      3. a bare ``{atom_id: float}`` mapping.

    Args:
        results: A FactReasoner results dict (or a bare id -> probability map).

    Returns:
        ``{atom_id: probability}``, empty when no recognizable shape is present.
    """
    if not results:
        return {}

    per_atom = results.get("factuality_score_per_atom")
    if isinstance(per_atom, list) and per_atom:
        out: dict[str, float] = {}
        for entry in per_atom:
            if not isinstance(entry, Mapping):
                continue
            for aid, payload in entry.items():
                if isinstance(payload, Mapping) and "score" in payload:
                    out[aid] = float(payload["score"])
                elif isinstance(payload, (int, float)):
                    out[aid] = float(payload)
        if out:
            return out

    marginals = results.get("marginals")
    if isinstance(marginals, list) and marginals:
        out = {}
        for m in marginals:
            if not isinstance(m, Mapping):
                continue
            var, probs = m.get("variable"), m.get("probabilities")
            if var is not None and isinstance(probs, (list, tuple)) and len(probs) > 1:
                out[str(var)] = float(probs[1])
        if out:
            return out

    # A bare {atom_id: float} map (e.g. hand-written, or a saved priors file).
    bare = {
        str(k): float(v)
        for k, v in results.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    # Guard against mistaking a results dict's scalar metrics (factuality_score,
    # num_atoms, ...) for atom priors: real atom ids look like "a0", "a_12".
    bare = {k: v for k, v in bare.items() if re.fullmatch(r"a_?\d+", k)}
    return bare


@dataclass(frozen=True)
class AtomPriors:
    """Per-atom priors from stage 1, plus what is known about their provenance.

    Attributes:
        priors: ``{atom_id: P(a_i = 1)}``. Empty means "no information", which
            resolves to a uniform ``default`` -- the coherence-only behaviour.
        atoms: The atoms stage 1 worked on, when it can supply them. Handing these
            straight to the miner is what lets the response be atomized once
            instead of twice, and makes id alignment exact.
        source: One of :data:`PRIOR_SOURCES`.
        default: The prior for atoms no entry covers (:data:`NEUTRAL_PRIOR`).
        coverage: How resolution went (filled in by :meth:`resolve`).
        diagnostics: Stage-1 side information worth reporting (factuality score,
            timings, an early-exit verdict, ...).
    """

    priors: dict[str, float] = field(default_factory=dict)
    atoms: dict[str, Atom] | None = None
    source: str = "uniform"
    default: float = NEUTRAL_PRIOR
    coverage: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def resolve(
        self,
        target_atoms: Mapping[str, Atom],
        *,
        on_low_coverage: str = "warn",
        min_coverage: float = DEFAULT_MIN_COVERAGE,
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Align these priors onto the atoms the coherence MRF will actually use.

        Matching, in order: an identity short-circuit when the atom dicts are the
        same object (the reuse path); then normalized-text matching; then ids for
        whatever text did not match; then the uniform ``default``. Text leads ids
        on purpose -- see the module docstring.

        Args:
            target_atoms: The atoms of the coherence MRF, keyed by id.
            on_low_coverage: What to do when coverage falls below
                ``min_coverage`` -- ``"warn"`` (default), ``"raise"``, or
                ``"uniform"`` (discard every prior, so the result is a clean
                coherence-only score rather than a half-primed mixture).
            min_coverage: The fraction of atoms that must carry a real prior.

        Returns:
            ``(node_priors, coverage)`` -- priors for every target atom, and a
            report with ``n_atoms``, ``n_priors``, ``n_matched_by_id``,
            ``n_matched_by_text``, ``n_defaulted``, ``coverage``, ``alignment``,
            ``degraded`` and (when degraded) ``degraded_reason``.

        Raises:
            ValueError: If ``on_low_coverage`` is unknown, or it is ``"raise"``
                and coverage is below ``min_coverage``.
        """
        if on_low_coverage not in LOW_COVERAGE_POLICIES:
            raise ValueError(
                f"Unknown on_low_coverage: {on_low_coverage!r} "
                f"(expected one of {list(LOW_COVERAGE_POLICIES)})."
            )

        n = len(target_atoms)
        report: dict[str, Any] = {
            "source": self.source,
            "n_atoms": n,
            "n_priors": len(self.priors),
            "n_matched_by_id": 0,
            "n_matched_by_text": 0,
            "n_defaulted": 0,
            "coverage": 0.0,
            "alignment": "none",
            "degraded": bool(self.coverage.get("degraded", False)),
        }
        if self.coverage.get("degraded_reason"):
            report["degraded_reason"] = self.coverage["degraded_reason"]
        if n == 0:
            return {}, report

        # No priors at all: uniform. This is the coherence-only path and it must
        # stay exactly equivalent to the pre-priors behaviour.
        if not self.priors:
            report["n_defaulted"] = n
            report["alignment"] = "uniform"
            return {aid: self.default for aid in target_atoms}, report

        resolved: dict[str, float] = {}

        # 1. Identity: stage 2 is mining the very atoms stage 1 produced.
        if self.atoms is not None and self.atoms is target_atoms:
            for aid in target_atoms:
                if aid in self.priors:
                    resolved[aid] = float(self.priors[aid])
                    report["n_matched_by_id"] += 1
            report["alignment"] = "identity"
        else:
            # 2. Text pass (leads, because a stale id can be actively wrong).
            by_text: dict[str, float] = {}
            if self.atoms:
                for aid, atom in self.atoms.items():
                    if aid not in self.priors:
                        continue
                    key = _normalize_text(getattr(atom, "text", "") or "")
                    if key:
                        # First writer wins; a duplicated text has one prior anyway.
                        by_text.setdefault(key, float(self.priors[aid]))
            for aid, atom in target_atoms.items():
                key = _normalize_text(getattr(atom, "text", "") or "")
                if key and key in by_text:
                    resolved[aid] = by_text[key]
                    report["n_matched_by_text"] += 1

            # 3. Id pass for whatever text could not place.
            for aid in target_atoms:
                if aid in resolved:
                    continue
                if aid in self.priors:
                    resolved[aid] = float(self.priors[aid])
                    report["n_matched_by_id"] += 1

            if report["n_matched_by_text"] and report["n_matched_by_id"]:
                report["alignment"] = "text+id"
            elif report["n_matched_by_text"]:
                report["alignment"] = "text"
            elif report["n_matched_by_id"]:
                report["alignment"] = "id"

        # 4. Uniform default for the rest.
        for aid in target_atoms:
            if aid not in resolved:
                resolved[aid] = self.default
                report["n_defaulted"] += 1

        matched = n - report["n_defaulted"]
        report["coverage"] = matched / n

        if report["coverage"] < min_coverage:
            reason = (
                f"only {matched}/{n} atoms carry a {self.source} prior "
                f"(coverage {report['coverage']:.2f} < {min_coverage:.2f})"
            )
            report["degraded"] = True
            report.setdefault("degraded_reason", reason)
            if on_low_coverage == "raise":
                raise ValueError(f"Insufficient atom prior coverage: {reason}.")
            if on_low_coverage == "uniform":
                report["alignment"] = "uniform"
                report["n_matched_by_id"] = 0
                report["n_matched_by_text"] = 0
                report["n_defaulted"] = n
                report["coverage"] = 0.0
                return {aid: self.default for aid in target_atoms}, report
            print(f"[AtomPriors] WARNING: {reason}; uncovered atoms use {self.default}.")

        return resolved, report


@runtime_checkable
class PriorProvider(Protocol):
    """Stage 1 of the two-stage model: supply per-atom priors for a response."""

    def priors_for(
        self,
        *,
        response: str,
        query: str | None = None,
        topic: str | None = None,
    ) -> AtomPriors:
        """Return the priors (and, where possible, the atoms) for a response."""
        ...


class UniformPriorProvider:
    """A flat prior for every atom -- i.e. coherence only, no factuality input.

    The default provider, and exactly equivalent to the behaviour before priors
    were configurable: an empty prior map that resolves to ``prior`` everywhere.
    """

    def __init__(self, prior: float = NEUTRAL_PRIOR):
        """Initialize the provider.

        Args:
            prior: The prior assigned to every atom.
        """
        self.prior = float(prior)

    def priors_for(
        self, *, response: str, query: str | None = None, topic: str | None = None
    ) -> AtomPriors:
        """Return empty priors, so every atom resolves to ``self.prior``."""
        return AtomPriors(priors={}, source="uniform", default=self.prior)


class PrecomputedPriorProvider:
    """Priors already computed elsewhere: a mapping, or a results file on disk.

    Costs nothing at run time, which makes it the way to reuse one factuality run
    across many coherence experiments (mine once per configuration, score with
    the same priors) and the way to score without any retrieval at all.
    """

    def __init__(
        self,
        priors: Mapping[str, float] | str | os.PathLike,
        *,
        atom_texts: Mapping[str, str] | None = None,
        default: float = NEUTRAL_PRIOR,
    ):
        """Initialize the provider.

        Args:
            priors: Either ``{atom_id: probability}`` directly, or a path to a JSON
                file holding a FactReasoner results dict (or a bare map) -- any
                shape :func:`atom_priors_from_results` understands. When the file
                also carries ``atoms`` (as ``FactReasoner.to_json`` writes), their
                text is used for text-based alignment.
            atom_texts: Optional ``{atom_id: text}`` enabling text alignment when
                the priors were passed as a bare mapping.
            default: The prior for atoms no entry covers.

        Raises:
            ValueError: If a file is given but holds no recognizable priors.
        """
        self.default = float(default)
        self._source = "precomputed"
        texts: dict[str, str] = dict(atom_texts or {})

        if isinstance(priors, Mapping):
            table = {str(k): float(v) for k, v in priors.items()}
        else:
            path = os.fspath(priors)
            with open(path) as f:
                payload = json.load(f)
            table = atom_priors_from_results(payload)
            if not table:
                raise ValueError(
                    f"No atom priors found in {path!r}: expected a FactReasoner "
                    'results dict ("factuality_score_per_atom" or "marginals") '
                    "or a bare {atom_id: probability} map."
                )
            # Lift atom text when the file has it, so alignment can use text.
            for entry in payload.get("atoms") or []:
                if isinstance(entry, Mapping) and entry.get("id"):
                    texts.setdefault(str(entry["id"]), str(entry.get("text", "")))
            self._source = "file"

        self.priors = table
        self.atom_texts = texts

    def priors_for(
        self, *, response: str, query: str | None = None, topic: str | None = None
    ) -> AtomPriors:
        """Return the stored priors (with atom text when it is known)."""
        atoms = (
            {aid: Atom(id=aid, text=text) for aid, text in self.atom_texts.items()}
            if self.atom_texts
            else None
        )
        return AtomPriors(
            priors=dict(self.priors),
            atoms=atoms,
            source=self._source,
            default=self.default,
        )


class FactReasonerPriorProvider:
    """Priors from a live FactReasoner run -- the posterior marginals themselves.

    Covers every way the factuality pipeline can be driven, selected by ``mode``:

      * ``"assess"`` -- a full run from a raw response: atomize, revise, retrieve
        contexts, score NLI relations, infer. Needs a :class:`FactualityRunner`.
        The runner's own axes (``pipeline_version``, ``nli_mode``, ``nli_method``,
        backend, caches, ...) all apply unchanged.
      * ``"file_item"`` -- a dataset item that already carries atoms and contexts
        (``FactReasoner.from_dict_with_contexts``): no retrieval, NLI only.
      * ``"fact_graph"`` -- an existing :class:`FactGraph`: inference only, zero
        LLM calls. Note that ``from_fact_graph`` reconstructs atoms with empty
        text, so this mode cannot supply atoms for reuse or text alignment; the
        priors are matched by id, and coverage reports it.

    In ``"assess"`` and ``"file_item"`` the provider keeps the atoms the factuality
    run produced and returns them, so the coherence stage mines those exact atoms
    rather than atomizing the response a second time.
    """

    def __init__(
        self,
        *,
        runner: Any = None,
        pipeline: Any = None,
        mode: str = "assess",
        item: Mapping[str, Any] | None = None,
        fact_graph: Any = None,
        default: float = NEUTRAL_PRIOR,
        on_degraded: str = "fallback",
    ):
        """Initialize the provider.

        Args:
            runner: A :class:`~fact_reasoner.runner.FactualityRunner` (required for
                ``mode="assess"`` and ``mode="file_item"``).
            pipeline: An already-built :class:`~fact_reasoner.assessor.FactReasoner`
                instance, used instead of ``runner`` for ``"file_item"`` /
                ``"fact_graph"``.
            mode: One of ``"assess"``, ``"file_item"``, ``"fact_graph"``.
            item: The dataset item for ``mode="file_item"``.
            fact_graph: The fact graph for ``mode="fact_graph"``.
            default: The prior for atoms the factuality run did not score.
            on_degraded: ``"fallback"`` (default) to fall back to uniform priors
                when the factuality run produces no marginals (e.g. an early
                exit), or ``"raise"`` to fail loudly.

        Raises:
            ValueError: On an unknown ``mode``/``on_degraded``, or a mode whose
                required argument is missing.
        """
        valid_modes = ("assess", "file_item", "fact_graph")
        if mode not in valid_modes:
            raise ValueError(
                f"Unknown mode: {mode!r} (expected one of {list(valid_modes)})."
            )
        if on_degraded not in ("fallback", "raise"):
            raise ValueError(
                f"Unknown on_degraded: {on_degraded!r} (expected 'fallback'/'raise')."
            )
        if mode == "assess" and runner is None:
            raise ValueError("mode='assess' requires runner=<FactualityRunner>.")
        if mode == "file_item":
            if item is None:
                raise ValueError("mode='file_item' requires item=<dataset item>.")
            if runner is None and pipeline is None:
                raise ValueError(
                    "mode='file_item' requires runner=<FactualityRunner> or "
                    "pipeline=<FactReasoner>."
                )
        if mode == "fact_graph":
            if fact_graph is None:
                raise ValueError("mode='fact_graph' requires fact_graph=<FactGraph>.")
            if pipeline is None:
                raise ValueError(
                    "mode='fact_graph' requires pipeline=<FactReasoner> (its "
                    "merlin_path drives the inference)."
                )

        self.runner = runner
        self.pipeline = pipeline
        self.mode = mode
        self.item = item
        self.fact_graph = fact_graph
        self.default = float(default)
        self.on_degraded = on_degraded

    # -- the three modes -----------------------------------------------------

    def priors_for(
        self, *, response: str, query: str | None = None, topic: str | None = None
    ) -> AtomPriors:
        """Run the factuality stage and return its posterior marginals as priors.

        Args:
            response: The response to assess (used by ``mode="assess"``).
            query: The query that produced it (``mode="assess"``).
            topic: Optional topic hint (``mode="assess"``).

        Returns:
            The priors, the factuality atoms (when the mode can supply them), and
            stage-1 diagnostics. Degrades to uniform priors -- keeping the atoms,
            so the reuse saving survives -- when the run yields no marginals.

        Raises:
            RuntimeError: If the run is degraded and ``on_degraded="raise"``.
        """
        if self.mode == "assess":
            results, pipeline = self.runner.assess_with_pipeline(
                query or "", response, topic=topic
            )
        elif self.mode == "file_item":
            if self.runner is not None:
                results, pipeline = self.runner.assess_item_with_pipeline(self.item)
            else:
                results, pipeline = self._run_pipeline_item(self.pipeline, self.item)
        else:  # fact_graph
            results, pipeline = self._run_fact_graph(self.pipeline, self.fact_graph)

        atoms = self._atoms_from_pipeline(pipeline)
        diagnostics = self._diagnostics(results, pipeline)

        priors = atom_priors_from_results(results or {})
        if not priors:
            return self._degraded(
                atoms,
                diagnostics,
                reason=self._degraded_reason(pipeline),
            )

        return AtomPriors(
            priors=priors,
            atoms=atoms,
            source="factreasoner",
            default=self.default,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _run_pipeline_item(pipeline: Any, item: Mapping[str, Any]):
        """Score one pre-annotated dataset item on a caller-supplied pipeline."""
        import asyncio

        pipeline.from_dict_with_contexts(dict(item))
        asyncio.run(
            pipeline.build(has_atoms=True, has_contexts=True, revise_atoms=False)
        )
        if getattr(pipeline, "fact_graph", None) is None:
            return None, pipeline
        results, _marginals = pipeline.score()
        return results, pipeline

    @staticmethod
    def _run_fact_graph(pipeline: Any, fact_graph: Any):
        """Infer on an existing fact graph (no LLM calls at all)."""
        pipeline.from_fact_graph(fact_graph)
        results, _marginals = pipeline.score()
        return results, pipeline

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _atoms_from_pipeline(pipeline: Any) -> dict[str, Atom] | None:
        """The factuality run's atoms, when they carry text worth reusing.

        ``from_fact_graph`` rebuilds atoms with empty text, so those are useless
        both for reuse and for text alignment; return None rather than pass on a
        set of blank atoms that would silently defeat text matching.
        """
        atoms = getattr(pipeline, "atoms", None)
        if not atoms:
            return None
        if not any((getattr(a, "text", "") or "").strip() for a in atoms.values()):
            return None
        return atoms

    @staticmethod
    def _diagnostics(results: Any, pipeline: Any) -> dict[str, Any]:
        """Stage-1 side information worth carrying into the coherence result."""
        diag: dict[str, Any] = {}
        if isinstance(results, Mapping):
            for key in (
                "factuality_score",
                "num_atoms",
                "num_contexts",
                "num_true_atoms",
                "num_false_atoms",
                "avg_norm_entropy",
                "elapsed_time",
                "nli_stats",
            ):
                if key in results:
                    diag[key] = results[key]
        early = getattr(pipeline, "early_exit_evaluation", None)
        if early is not None:
            diag["early_exit_evaluation"] = early
        timing = getattr(pipeline, "timing", None)
        if timing:
            diag["timing"] = dict(timing)
        return diag

    @staticmethod
    def _degraded_reason(pipeline: Any) -> str:
        """Why a factuality run produced no usable marginals."""
        if getattr(pipeline, "early_exit_evaluation", None) is not None and (
            getattr(pipeline, "fact_graph", None) is None
        ):
            return "factreasoner_early_exit"
        if getattr(pipeline, "fact_graph", None) is None:
            return "factreasoner_no_graph"
        return "factreasoner_no_marginals"

    def _degraded(
        self, atoms: dict[str, Atom] | None, diagnostics: dict[str, Any], *, reason: str
    ) -> AtomPriors:
        """Fall back to uniform priors, keeping the atoms and recording why.

        The atoms are still returned when available: they cost LLM calls to
        produce, and reusing them keeps the atomize-once saving even though the
        priors themselves are uninformative.
        """
        if self.on_degraded == "raise":
            raise RuntimeError(
                f"FactReasoner produced no atom marginals ({reason}); "
                "pass on_degraded='fallback' to score coherence with uniform priors."
            )
        print(
            f"[FactReasonerPriorProvider] WARNING: {reason}; "
            "falling back to uniform atom priors."
        )
        return AtomPriors(
            priors={},
            atoms=atoms,
            source="uniform",
            default=self.default,
            coverage={"degraded": True, "degraded_reason": reason},
            diagnostics=diagnostics,
        )


def coerce_prior_provider(
    priors: PriorProvider | AtomPriors | Mapping[str, float] | float | None,
) -> PriorProvider:
    """Turn any accepted ``priors=`` argument into a :class:`PriorProvider`.

    Accepts a provider (returned as-is), an :class:`AtomPriors` (wrapped), a
    ``{atom_id: probability}`` mapping, a float (uniform), or ``None``
    (uniform 0.5 -- coherence only).

    Args:
        priors: The user-supplied priors specification.

    Returns:
        A provider yielding those priors.

    Raises:
        TypeError: If ``priors`` is none of the accepted forms.
    """
    if priors is None:
        return UniformPriorProvider()
    if isinstance(priors, AtomPriors):
        return _FixedPriorProvider(priors)
    if isinstance(priors, PriorProvider) and hasattr(priors, "priors_for"):
        return priors
    if isinstance(priors, Mapping):
        return PrecomputedPriorProvider(priors)
    if isinstance(priors, (int, float)) and not isinstance(priors, bool):
        return UniformPriorProvider(float(priors))
    raise TypeError(
        f"Cannot interpret priors={priors!r} as atom priors: expected a "
        "PriorProvider, AtomPriors, {atom_id: probability} mapping, float, or None."
    )


class _FixedPriorProvider:
    """Wraps an already-resolved :class:`AtomPriors` as a provider."""

    def __init__(self, atom_priors: AtomPriors):
        self.atom_priors = atom_priors

    def priors_for(
        self, *, response: str, query: str | None = None, topic: str | None = None
    ) -> AtomPriors:
        return self.atom_priors
