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

# Logical Coherence Score (LCS) readout from a coherence MRF.
#
# The deep-dive (Section 7) selects the LCS as the mean posterior marginal support
# of the atoms under the coherence MRF (Eq. 4):
#
#     LCS(y) = (1/n) * sum_i P(a_i = 1)
#
# It is the MRF-native choice: monotone, in [0,1], constant-free, and already
# returned by Merlin (MAR task). Alongside the headline we report diagnostics
# (Section 7.3): per-atom marginals, the count of atoms dragged below their prior,
# the average normalized entropy, and log Z (PR task) as a contradiction-
# sensitivity gauge.
#
# This scorer only *reads* the MRF that ``RelationMiner`` built (via the shared
# Merlin helper); it does not define or duplicate the factuality scoring in
# ``assessor.py``.

import math
from typing import Any, Dict, Optional

from fact_reasoner.inference import run_merlin
from fact_reasoner.lcs.relation_miner import MiningResult, _atom_sort_key


def _binary_entropy(p: float) -> float:
    """Normalized binary entropy H_2(p) in [0, 1] (0 at p in {0,1}, 1 at 0.5)."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


class LCSScorer:
    """Compute the Logical Coherence Score from a mined coherence MRF."""

    def __init__(self, merlin_path: str, *, ibound: int = 6, verbose: bool = False):
        """Initialize the scorer.

        Args:
            merlin_path: Path to the Merlin executable.
            ibound: The i-bound for Merlin's weighted mini-bucket inference.
            verbose: Whether the Merlin helper prints its progress.
        """
        if not merlin_path:
            raise ValueError("merlin_path is required to run inference.")
        self.merlin_path = merlin_path
        self.ibound = ibound
        self.verbose = verbose

    def score(
        self,
        result: MiningResult,
        *,
        prior: Optional[float] = None,
        compute_log_z: bool = True,
    ) -> Dict[str, Any]:
        """Compute the LCS and diagnostics for a mining result.

        Args:
            result: The :class:`MiningResult` from ``RelationMiner``.
            prior: The atom prior ``pi`` used to count "atoms dragged below their
                prior". Defaults to the prior recorded in ``result.config``
                (falling back to 0.5).
            compute_log_z: Whether to also run Merlin's PR task for ``log Z`` (an
                extra inference run; deep-dive Section 8.3 diagnostic).

        Returns:
            A dict with:
              * ``"lcs"``: the mean posterior marginal support (Eq. 4).
              * ``"marginals"``: ``{atom_id: P(a_i=1)}``.
              * ``"num_atoms"``: number of atoms.
              * ``"num_below_prior"``: ``#{ q_i < prior }``.
              * ``"avg_norm_entropy"``: mean normalized binary entropy of the
                marginals (lower is more decisive).
              * ``"log_z"``: the log partition function, or None if not computed.
        """
        atoms = result.atoms
        n = len(atoms)
        if prior is None:
            prior = float(result.config.get("prior", 0.5))

        if n == 0:
            return {
                "lcs": 0.0,
                "marginals": {},
                "num_atoms": 0,
                "num_below_prior": 0,
                "avg_norm_entropy": 0.0,
                "log_z": None,
            }

        query_variables = sorted(atoms.keys(), key=_atom_sort_key)
        mar = run_merlin(
            result.markov_network,
            self.merlin_path,
            task="MAR",
            ibound=self.ibound,
            query_variables=query_variables,
            verbose=self.verbose,
        )

        marginals: Dict[str, float] = {}
        for m in mar["marginals"]:
            marginals[m["variable"]] = float(m["probabilities"][1])

        # Eq. 4: mean posterior marginal support.
        support = [marginals[aid] for aid in query_variables if aid in marginals]
        lcs = sum(support) / len(support) if support else 0.0

        num_below_prior = sum(1 for q in support if q < prior)
        avg_norm_entropy = (
            sum(_binary_entropy(q) for q in support) / len(support) if support else 0.0
        )

        log_z = None
        if compute_log_z:
            try:
                pr = run_merlin(
                    result.markov_network,
                    self.merlin_path,
                    task="PR",
                    ibound=self.ibound,
                    verbose=self.verbose,
                )
                log_z = pr["log_z"]
            except Exception as e:
                print(f"[LCSScorer] PR (log Z) task unavailable: {e}")

        return {
            "lcs": lcs,
            "marginals": marginals,
            "num_atoms": n,
            "num_below_prior": num_below_prior,
            "avg_norm_entropy": avg_norm_entropy,
            "log_z": log_z,
        }
