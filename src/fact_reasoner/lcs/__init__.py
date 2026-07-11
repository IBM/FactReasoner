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

    result = miner.mine_from_response("The stock fell 15%. Consequently the CEO was fired.")
    result.describe()
    mn = result.markov_network            # the MRF (mn.to_uai() serializes it)

    lcs = LCSScorer(merlin_path).score(result)   # {"lcs": ..., "log_z": ...}

    # or, one call end-to-end:
    lcs = mine_and_score(response, backend=backend, merlin_path=merlin_path,
                         atomizer=Atomizer(backend))
"""

from typing import Any, Dict, List, Optional, Union

from fact_reasoner.core.base import Atom
from fact_reasoner.factors import (
    build_markov_network,
    edge_factor_values,
    pairwise_prior,
)
from fact_reasoner.lcs.lcs_scorer import LCSScorer
from fact_reasoner.lcs.relation_miner import (
    MinedRelation,
    MiningResult,
    RelationMiner,
)
from fact_reasoner.lcs.strength import (
    IdentityCalibrator,
    PlattCalibrator,
    StrengthCalibrator,
    TemperatureCalibrator,
)
from fact_reasoner.lcs.taxonomy import (
    COMPILE,
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
    "mine_and_score",
    "StrengthCalibrator",
    "IdentityCalibrator",
    "TemperatureCalibrator",
    "PlattCalibrator",
    "Level2Sense",
    "SenseSpec",
    "COMPILE",
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
    scorer_kwargs: Optional[Dict[str, Any]] = None,
    **miner_kwargs,
) -> Dict[str, Any]:
    """Mine relations and compute the LCS in one call.

    A convenience wrapper around :class:`RelationMiner` + :class:`LCSScorer`.

    Args:
        response_or_atoms: Either a raw response string (atomized via
            ``atomizer``) or a list/dict of atoms (mined directly).
        backend: The Mellea backend.
        merlin_path: Path to the Merlin executable.
        atomizer: Required when ``response_or_atoms`` is a raw string.
        reviser: Optional decontextualizer for atoms from a response.
        scorer_kwargs: Extra kwargs for :meth:`LCSScorer.score` (e.g.
            ``{"method": "reified"}`` to pick an alternative LCS readout).
        **miner_kwargs: Extra kwargs for :class:`RelationMiner` (e.g.
            ``nli_method``, ``strength_method`` (``"surrogate_logprobs"`` /
            ``"surrogate_sampled"`` / ``"verbalized"``), ``strength_calibrator``,
            ``pair_policy``, ``window``, ``gate``).

    Returns:
        A dict with the score fields from :meth:`LCSScorer.score` plus a
        ``"result"`` key holding the full :class:`MiningResult`.
    """
    miner = RelationMiner(
        backend, atomizer=atomizer, reviser=reviser, **miner_kwargs
    )
    if isinstance(response_or_atoms, str):
        result = miner.mine_from_response(response_or_atoms)
    else:
        result = miner.mine_from_atoms(response_or_atoms)

    scorer = LCSScorer(merlin_path)
    scores = scorer.score(result, **(scorer_kwargs or {}))
    scores["result"] = result
    return scores
