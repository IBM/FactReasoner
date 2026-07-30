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

# Compute all LCS scores for one mined MRF.

from typing import Any, Dict, List, Optional

from fact_reasoner.lcs.lcs_scorer import LCS_METHODS, LCSScorer
from fact_reasoner.lcs.relation_miner import MiningResult


def score_all_lcs(
    result: MiningResult,
    scorer: LCSScorer,
    *,
    methods: Optional[List[str]] = None,
    reified_prior: float = 0.5,
    node_priors: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compute every LCS readout for a single mined MRF.

    A thin adapter over :meth:`LCSScorer.score_all`, which reads all the requested
    readouts off the same base network while running the shared base MAR and base
    PR only once (6 Merlin invocations for all four readouts rather than 12).

    Args:
        result: The mined :class:`MiningResult`.
        scorer: An :class:`LCSScorer` (real or dry-run monkeypatched).
        methods: LCS methods to compute (defaults to all of ``LCS_METHODS``).
        reified_prior: Bernoulli prior for the reified score.
        node_priors: Optional per-atom priors ``{atom_id: pi_i}`` (e.g. the
            factuality stage's posterior marginals).

    Returns:
        A dict with one key per LCS method (its scalar value) plus
        ``num_atoms``, ``num_below_prior``, ``avg_norm_entropy``, ``log_z``,
        ``log_z_max``, ``log_z_min`` (from the runs that compute them), and
        ``marginals``.
    """
    methods = methods or list(LCS_METHODS)
    scores = scorer.score_all(
        result,
        methods=methods,
        reified_prior=reified_prior,
        node_priors=node_priors,
    )

    out: Dict[str, Any] = {m: scores.get(m) for m in methods}
    for key in ("num_atoms", "num_below_prior", "avg_norm_entropy", "log_z",
                "log_z_max", "log_z_min"):
        val = scores.get(key)
        if val is not None:
            out[key] = val
    out["marginals"] = scores.get("marginals") or {}
    return out
