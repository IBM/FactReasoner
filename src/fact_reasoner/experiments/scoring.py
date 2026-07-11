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
) -> Dict[str, Any]:
    """Compute every LCS readout for a single mined MRF.

    All LCS scores read the same coherence MRF, so this calls
    ``scorer.score(result, method=m)`` once per method and collects the headline
    of each, plus the shared per-atom diagnostics (read once).

    Args:
        result: The mined :class:`MiningResult`.
        scorer: An :class:`LCSScorer` (real or dry-run monkeypatched).
        methods: LCS methods to compute (defaults to all of ``LCS_METHODS``).
        reified_prior: Bernoulli prior for the reified score.

    Returns:
        A dict with one key per LCS method (its scalar value) plus
        ``num_atoms``, ``num_below_prior``, ``avg_norm_entropy``, ``log_z``,
        ``log_z_max`` (from the runs that compute them), and ``marginals``.
    """
    methods = methods or list(LCS_METHODS)
    out: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {}
    marginals: Dict[str, float] = {}

    for m in methods:
        scores = scorer.score(result, method=m, reified_prior=reified_prior)
        out[m] = scores.get(m)
        # Diagnostics are identical across methods except log_z_max (only the
        # log_partition run computes it); keep the first non-None seen.
        for key in ("num_atoms", "num_below_prior", "avg_norm_entropy", "log_z", "log_z_max"):
            val = scores.get(key)
            if val is not None and diagnostics.get(key) is None:
                diagnostics[key] = val
        if not marginals and scores.get("marginals"):
            marginals = scores["marginals"]

    out.update(diagnostics)
    out["marginals"] = marginals
    return out
