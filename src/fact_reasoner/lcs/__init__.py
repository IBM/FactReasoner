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

"""Logical Coherence Score (LCS) components for FactReasoner.

This subpackage extends FactReasoner from factuality to *logical coherence*: it
mines inter-atom relations with an LLM, estimates their probabilities via the
same UQ backends the factuality pipeline uses (logprobs / SIMBA-UQ), builds the
Markov network (MRF) encoding, and reads a coherence score off it.

Quick start::

    from fact_reasoner import build_backend
    from fact_reasoner.core import Atomizer
    from fact_reasoner.lcs import RelationMiner, LCSScorer, mine_and_score

    backend = build_backend("rits", model_id="llama-3-3-70b-instruct")
    miner = RelationMiner(backend, atomizer=Atomizer(backend))

    # Mining is always response-grounded. From a raw response:
    result = miner.mine_from_response("The stock fell 15%. Consequently the CEO was fired.")
    result.describe()
    mn = result.markov_network            # the MRF (mn.to_uai() serializes it)

    lcs = LCSScorer(merlin_path).score(result)   # {"lcs": ..., "log_z": ...}

    # Several readouts at once share the base inference runs (6 Merlin calls, not 12):
    all_scores = LCSScorer(merlin_path).score_all(result)

    # From pre-extracted atoms, pass the response they came from:
    result = miner.mine_from_atoms(atom_texts, response)

    # or, one call end-to-end:
    lcs = mine_and_score(response, backend=backend, merlin_path=merlin_path,
                         atomizer=Atomizer(backend))

Two-stage scoring (factuality priors + coherence)
-------------------------------------------------

By default every atom starts from a flat 0.5 prior, so the LCS measures internal
coherence alone. Feeding in the factuality pipeline's posterior marginals instead
makes the coherence MRF start from how well each atom is *externally supported*::

    from fact_reasoner.lcs import CoherencePipeline, FactReasonerPriorProvider
    from fact_reasoner.runner import FactualityRunner

    runner = FactualityRunner(backend, merlin_path=merlin_path, nli_mode="fast")
    pipeline = CoherencePipeline(
        miner=RelationMiner(backend, atomizer=Atomizer(backend)),
        merlin_path=merlin_path,
        prior_provider=FactReasonerPriorProvider(runner=runner),
        methods=("mean_marginal", "consistency"),
    )
    out = pipeline.run(response, query=query)
    out.describe()          # out.priors are stage 1's posteriors

The provider hands its atoms to the miner, so the response is atomized once, not
once per stage. Priors can also come from a saved factuality run
(``PrecomputedPriorProvider("results.json")``) or a plain mapping -- both cost no
LLM calls -- and ``mine_and_score(..., priors=...)`` takes any of these forms.

The runner
----------

``CoherenceRunner`` is the layer above that pipeline: it owns the backend and the
shared components, builds the factuality stage itself, and adds a resumable
dataset sweep. Factuality priors are its default, so the two-stage model above is
what you get out of the box::

    from fact_reasoner.lcs import CoherenceRunner

    runner = CoherenceRunner.from_backend_kind("rits", merlin_path=merlin_path)
    runner.assess(query, response).describe()

    # A jsonl dataset of items that already carry atoms and contexts: nothing is
    # atomized or retrieved, results are written incrementally, and re-running
    # skips whatever finished.
    runner.assess_file("data.jsonl", "results/", dataset_name="demo",
                       model_id="llama-3-3-70b")

Pass ``prior_source="none"`` for coherence alone, or ``"file"`` with
``priors_file=`` to replay a saved factuality run. ``from_backend_kind`` builds any
backend ``build_backend`` supports (``"rits"`` by default); with ``"ollama"`` -- or
Claude via the OpenAI-compatibility endpoint -- pass ``nli_method="simbauq"``,
since neither returns usable logprobs.

``formulation="mln"`` selects the Markov-logic model of
``docs/ideation/coherence_mln_deepdive.pdf``; its closed-form pairwise fragment is
implemented (and verified to reproduce the MRF exactly), but scoring it raises
``NotImplementedError`` pending a MC-SAT / MaxWalkSAT engine.
"""

from typing import Any, Dict, List, Optional, Union

from fact_reasoner.core.base import Atom
from fact_reasoner.factors import (
    build_markov_network,
    edge_factor_values,
    pairwise_prior,
)
from fact_reasoner.lcs.lcs_scorer import LCS_METHODS, LCSScorer
from fact_reasoner.lcs.pipeline import (
    COHERENCE_FORMULATIONS,
    RULE_SCHEMA,
    CoherenceModel,
    CoherencePipeline,
    CoherenceResult,
    MLNCoherenceModel,
    MLNEngine,
    MRFCoherenceModel,
    build_coherence_model,
    mln_weight,
    three_clause_weights,
)
from fact_reasoner.lcs.priors import (
    NEUTRAL_PRIOR,
    AtomPriors,
    FactReasonerPriorProvider,
    PrecomputedPriorProvider,
    PriorProvider,
    UniformPriorProvider,
    atom_priors_from_results,
    coerce_prior_provider,
)
from fact_reasoner.lcs.relation_miner import (
    MinedRelation,
    MiningResult,
    RelationMiner,
)
from fact_reasoner.lcs.runner import (
    COHERENCE_PRIOR_SOURCES,
    DEFAULT_BACKEND_KIND,
    CoherenceRunner,
    atom_texts_from_item,
)
from fact_reasoner.lcs.strength import (
    IdentityCalibrator,
    PlattCalibrator,
    StrengthCalibrator,
    TemperatureCalibrator,
)
from fact_reasoner.lcs.taxonomy import (
    COMPILE,
    LEVEL1_CONECESSITY,
    LEVEL1_CONFLICT_COUPLINGS,
    LEVEL1_EDGE_COUPLINGS,
    LEVEL1_EXCLUSIVE,
    Level2Sense,
    SenseSpec,
    compile_sense,
    coupling_from_string,
)

__all__ = [
    "RelationMiner",
    "MinedRelation",
    "MiningResult",
    "LCSScorer",
    "LCS_METHODS",
    "mine_and_score",
    # The runner: one object over a single response or a jsonl dataset.
    "CoherenceRunner",
    "COHERENCE_PRIOR_SOURCES",
    "DEFAULT_BACKEND_KIND",
    "atom_texts_from_item",
    # Two-stage pipeline: factuality priors + a coherence model.
    "CoherencePipeline",
    "CoherenceResult",
    "CoherenceModel",
    "MRFCoherenceModel",
    "MLNCoherenceModel",
    "MLNEngine",
    "build_coherence_model",
    "COHERENCE_FORMULATIONS",
    "RULE_SCHEMA",
    "mln_weight",
    "three_clause_weights",
    # Atom priors.
    "AtomPriors",
    "PriorProvider",
    "UniformPriorProvider",
    "PrecomputedPriorProvider",
    "FactReasonerPriorProvider",
    "atom_priors_from_results",
    "coerce_prior_provider",
    "NEUTRAL_PRIOR",
    "StrengthCalibrator",
    "IdentityCalibrator",
    "TemperatureCalibrator",
    "PlattCalibrator",
    "Level2Sense",
    "SenseSpec",
    "COMPILE",
    "LEVEL1_EXCLUSIVE",
    "LEVEL1_CONECESSITY",
    "LEVEL1_EDGE_COUPLINGS",
    "LEVEL1_CONFLICT_COUPLINGS",
    "compile_sense",
    "coupling_from_string",
    "build_markov_network",
    "edge_factor_values",
    "pairwise_prior",
]


def mine_and_score(
    response_or_atoms: Union[str, List[str], List[Atom], Dict[str, Atom]],
    *,
    backend,
    merlin_path: str,
    atomizer=None,
    reviser=None,
    response: Optional[str] = None,
    priors=None,
    formulation: str = "mrf",
    scorer_kwargs: Optional[Dict[str, Any]] = None,
    **miner_kwargs,
) -> Dict[str, Any]:
    """Mine relations and compute the LCS in one call.

    A convenience wrapper around :class:`RelationMiner` + :class:`LCSScorer`.

    Mining is always response-grounded, so a response is always needed: pass a
    raw response string as ``response_or_atoms`` (it is atomized and grounded on
    itself), or pass pre-extracted atoms as ``response_or_atoms`` together with
    the ``response=`` they came from.

    Args:
        response_or_atoms: Either a raw response string (atomized via
            ``atomizer``) or a list/dict of atoms (mined directly, grounded in
            ``response``).
        backend: The Mellea backend.
        merlin_path: Path to the Merlin executable.
        atomizer: Required when ``response_or_atoms`` is a raw string.
        reviser: Optional decontextualizer for atoms from a response.
        response: The original response the atoms came from. REQUIRED when
            ``response_or_atoms`` is a list/dict of atoms (ignored for the raw
            string path, which already grounds on its own text).
        priors: Optional per-atom priors for the coherence MRF's unary factors --
            an :class:`AtomPriors`, a ``{atom_id: probability}`` mapping, a float,
            or a :class:`PriorProvider`. ``None`` (the default) keeps the uniform
            0.5 prior, i.e. coherence only. To prime the atoms with their
            factuality posteriors, pass a
            :class:`FactReasonerPriorProvider` -- or use
            :class:`CoherencePipeline`, which also reuses the factuality run's
            atoms instead of atomizing twice.
        formulation: ``"mrf"`` (default) or ``"mln"``; see
            :func:`build_coherence_model`.
        scorer_kwargs: Extra kwargs for :meth:`LCSScorer.score` (e.g.
            ``{"method": "reified"}`` to pick an alternative LCS readout).
        **miner_kwargs: Extra kwargs for :class:`RelationMiner` (e.g.
            ``nli_method``, ``strength_method`` (``"surrogate_logprobs"`` /
            ``"surrogate_sampled"`` / ``"verbalized"``), ``strength_calibrator``,
            ``pair_policy``, ``window``, ``gate``).

    Returns:
        A dict with the score fields from :meth:`LCSScorer.score` plus a
        ``"result"`` key holding the full :class:`MiningResult`.

    Raises:
        ValueError: If atoms are passed without a ``response``.
    """
    miner = RelationMiner(backend, atomizer=atomizer, reviser=reviser, **miner_kwargs)
    if isinstance(response_or_atoms, str):
        source_response = response_or_atoms
        result = miner.mine_from_response(response_or_atoms)
    else:
        if not response or not str(response).strip():
            raise ValueError(
                "mine_and_score with a list/dict of atoms requires response=... "
                "(mining is always response-grounded)."
            )
        source_response = response
        result = miner.mine_from_atoms(response_or_atoms, response=response)

    scorer_kwargs = dict(scorer_kwargs or {})
    node_priors = None
    if priors is not None:
        atom_priors = coerce_prior_provider(priors).priors_for(response=source_response)
        node_priors, _coverage = atom_priors.resolve(result.atoms)

    if formulation == "mrf":
        # Route through the scorer directly, so `scorer_kwargs` (method=, prior=,
        # reified_prior=) keep working exactly as before.
        scores = LCSScorer(merlin_path).score(
            result, node_priors=node_priors, **scorer_kwargs
        )
    else:
        model = build_coherence_model(formulation, merlin_path=merlin_path)
        method = scorer_kwargs.pop("method", "mean_marginal")
        scores = model.score(
            result, node_priors=node_priors, methods=(method,), **scorer_kwargs
        )
    scores["result"] = result
    return scores
