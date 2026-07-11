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

# Logical Coherence Score (LCS) readouts from a coherence MRF.
#
# The deep-dive (docs/ideation/coherence_mrf_deepdive.pdf, Sections 7-8) defines
# FOUR candidate scores over the coherence MRF that ``RelationMiner`` builds:
#
#   (a) mean_marginal  -- LCS = (1/n) sum_i P(a_i=1)               (Eq. 4, DEFAULT)
#   (b) consistency    -- P( no CONTRADICT edge jointly active )    (Eq. 5)
#   (c) reified        -- P(R=1) for an added coherence node R      (Eqs. 6-7)
#   (d) log_partition  -- normalized (log Z - log Zmin)/(log Zmax - log Zmin) (Eq. 8)
#
# (a) is the selected headline: MRF-native, monotone, constant-free, in [0,1], and
# read directly off Merlin's MAR marginals. (b)-(d) are alternative readouts that
# this scorer can compute on request via the ``method`` argument.
#
# All four are read off the SAME mined MRF (via the shared Merlin helper
# ``fact_reasoner.inference.run_merlin``); (b) and (c) add derived variables and
# (d) needs a second reference network, all built here from the fact graph reusing
# ``factors.build_markov_network``. This scorer only reads / augments the MRF; it
# does not define or duplicate the factuality scoring in ``assessor.py``.

import math
from typing import Any, Dict, List, Optional

from fact_reasoner.factors import build_markov_network
from fact_reasoner.fact_graph import Edge, FactGraph, Node
from fact_reasoner.inference import run_merlin
from fact_reasoner.markov_network import MarkovNetwork
from fact_reasoner.lcs.relation_miner import MiningResult, _atom_sort_key

# The four LCS readouts. ``mean_marginal`` is the default headline (deep-dive Eq. 4).
LCS_METHODS = ("mean_marginal", "consistency", "reified", "log_partition")

# Prefix for derived / auxiliary variables (consistency U-chain, reified R). It
# sorts after atom ids "a..." under Merlin's (cardinality, name) ordering, so the
# atom marginals keep their positions and the derived variable is addressable by
# its exact name.
_AUX = "z"


def _binary_entropy(p: float) -> float:
    """Normalized binary entropy H_2(p) in [0, 1] (0 at p in {0,1}, 1 at 0.5)."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


class LCSScorer:
    """Compute a Logical Coherence Score from a mined coherence MRF.

    The headline score is selected by ``score(..., method=...)``; the default is
    ``"mean_marginal"`` (deep-dive Eq. 4). The other three deep-dive candidates —
    ``"consistency"``, ``"reified"`` and ``"log_partition"`` — are available as
    alternatives.
    """

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

    # -- public API ----------------------------------------------------------

    def score(
        self,
        result: MiningResult,
        *,
        method: str = "mean_marginal",
        prior: Optional[float] = None,
        reified_prior: float = 0.5,
    ) -> Dict[str, Any]:
        """Compute the LCS and diagnostics for a mining result.

        Args:
            result: The :class:`MiningResult` from ``RelationMiner``.
            method: Which readout is the headline ``"lcs"`` value. One of
                ``LCS_METHODS`` (default ``"mean_marginal"``). The selected
                method's score is also stored under its own key; the other
                alternative keys are ``None`` unless that method was selected
                (they each need extra inference, so they are computed on demand).
            prior: The atom prior ``pi`` used to count "atoms dragged below their
                prior". Defaults to the prior recorded in ``result.config``
                (falling back to 0.5).
            reified_prior: The Bernoulli prior ``rho`` on the reified coherence
                node ``R`` (deep-dive default 0.5); used only by ``method="reified"``.

        Returns:
            A dict with:
              * ``"method"``: the selected method.
              * ``"lcs"``: the selected method's score (the headline).
              * ``"mean_marginal"``: Eq. 4 value (always computed).
              * ``"consistency"`` / ``"reified"`` / ``"log_partition"``: the
                alternative scores, populated when selected (else ``None``).
              * ``"marginals"``: ``{atom_id: P(a_i=1)}`` from the base network.
              * ``"num_atoms"``, ``"num_below_prior"``, ``"avg_norm_entropy"``.
              * ``"log_z"``: base-network log partition (always computed).
              * ``"log_z_max"``: contradiction-free log partition (only for
                ``method="log_partition"``, else ``None``).

        Raises:
            ValueError: If ``method`` is not one of ``LCS_METHODS``.
        """
        if method not in LCS_METHODS:
            raise ValueError(
                f"Unknown LCS method: {method!r} (expected one of {list(LCS_METHODS)})."
            )

        atoms = result.atoms
        n = len(atoms)
        if prior is None:
            prior = float(result.config.get("prior", 0.5))

        out: Dict[str, Any] = {
            "method": method,
            "lcs": 0.0,
            "mean_marginal": 0.0,
            "consistency": None,
            "reified": None,
            "log_partition": None,
            "marginals": {},
            "num_atoms": n,
            "num_below_prior": 0,
            "avg_norm_entropy": 0.0,
            "log_z": None,
            "log_z_max": None,
        }
        if n == 0:
            return out

        # Base MAR run: mean-marginal + per-atom diagnostics (always computed).
        marginals = self._marginals(result.markov_network, sorted(atoms, key=_atom_sort_key))
        out["marginals"] = marginals
        support = list(marginals.values())
        out["mean_marginal"] = sum(support) / len(support) if support else 0.0
        out["num_below_prior"] = sum(1 for q in support if q < prior)
        out["avg_norm_entropy"] = (
            sum(_binary_entropy(q) for q in support) / len(support) if support else 0.0
        )

        # Base log Z is a cheap PR run and a useful contradiction-sensitivity gauge.
        out["log_z"] = self._log_z(result.markov_network)

        # Compute the selected method's headline (mean-marginal already in hand).
        if method == "mean_marginal":
            out["lcs"] = out["mean_marginal"]
        elif method == "consistency":
            out["consistency"] = self._consistency_probability(result)
            out["lcs"] = out["consistency"]
        elif method == "reified":
            out["reified"] = self._reified_coherence(result, reified_prior)
            out["lcs"] = out["reified"]
        elif method == "log_partition":
            norm, log_z_max = self._normalized_log_partition(result, out["log_z"])
            out["log_partition"] = norm
            out["log_z_max"] = log_z_max
            out["lcs"] = norm

        return out

    # -- inference helpers ---------------------------------------------------

    def _marginals(
        self, network: MarkovNetwork, query_variables: List[str]
    ) -> Dict[str, float]:
        """Run MAR and return ``{variable: P(=1)}`` for the query variables."""
        mar = run_merlin(
            network,
            self.merlin_path,
            task="MAR",
            ibound=self.ibound,
            query_variables=query_variables,
            verbose=self.verbose,
        )
        return {m["variable"]: float(m["probabilities"][1]) for m in mar["marginals"]}

    def _log_z(self, network: MarkovNetwork) -> Optional[float]:
        """Run PR and return log Z, or None if the PR task is unavailable."""
        try:
            pr = run_merlin(
                network, self.merlin_path, task="PR", ibound=self.ibound,
                verbose=self.verbose,
            )
            return pr["log_z"]
        except Exception as e:
            print(f"[LCSScorer] PR (log Z) task unavailable: {e}")
            return None

    # -- (b) consistency probability -----------------------------------------

    def _consistency_probability(self, result: MiningResult) -> float:
        """P( no CONTRADICT edge is jointly active ) — deep-dive Eq. 5.

        Adds, on a copy of the base network, one AND aux-var per contradiction
        edge (``u_r = a_s AND a_t``) and a running-OR accumulator ``U = OR_r u_r``,
        then reads ``P(U=0)``. All aux factors are deterministic and at most
        ternary (no 2^k blow-up). Returns 1.0 when there are no contradiction edges.
        """
        contradictions = [
            r for r in result.relations if r.level1_type == "contradiction"
        ]
        if not contradictions:
            return 1.0

        network = self._base_network(result)

        # One AND aux var per contradiction edge: u_r = (s AND t).
        u_vars: List[str] = []
        for i, rel in enumerate(contradictions):
            u = f"{_AUX}u{i}"
            network.add_factor(
                [u, rel.source_id, rel.target_id],
                [2, 2, 2],
                _and_factor(),
            )
            u_vars.append(u)

        # Running-OR accumulator U = OR_r u_r, chained ternary factors.
        acc = u_vars[0]
        if len(u_vars) > 1:
            for i, u in enumerate(u_vars[1:], start=1):
                nxt = f"{_AUX}or{i}"
                network.add_factor([nxt, acc, u], [2, 2, 2], _or_factor())
                acc = nxt

        p_u = self._marginals(network, [acc])
        p_active = p_u.get(acc, 0.0)  # P(U=1) = some contradiction active
        return 1.0 - p_active

    # -- (c) reified coherence node ------------------------------------------

    def _reified_coherence(self, result: MiningResult, rho: float) -> float:
        """P(R=1) for the reified coherence node — deep-dive Eqs. 6-7.

        Adds a binary node ``R`` with Bernoulli prior ``rho`` and, per relation, a
        ternary noisy-AND vote factor ``h_r(R, a_s, a_t)`` that in the R=1 branch
        charges ``1 - p_r`` whenever the relation is violated, and is flat in the
        R=0 branch. Reads ``P(R=1)``.
        """
        if not result.relations:
            # No relations => R is decoupled; its marginal is just its prior.
            return rho

        network = self._base_network(result)
        node_R = f"{_AUX}R"
        # R's Bernoulli prior factor [1-rho, rho].
        network.add_factor([node_R], [2], [1.0 - rho, rho])
        # One vote factor per relation.
        for rel in result.relations:
            network.add_factor(
                [node_R, rel.source_id, rel.target_id],
                [2, 2, 2],
                _vote_factor(rel.level1_type, rel.probability),
            )
        p_r = self._marginals(network, [node_R])
        return p_r.get(node_R, rho)

    # -- (d) normalized log-partition ----------------------------------------

    def _normalized_log_partition(
        self, result: MiningResult, log_z: Optional[float]
    ) -> (Any):
        """(log Z - log Zmin)/(log Zmax - log Zmin) — deep-dive Eq. 8.

        ``Zmax`` is the same network with all CONTRADICT factors removed (the
        maximally-coherent reference); ``Zmin`` is the base network. With
        ``Zmin == Z`` the base normalizes to 0.0 by construction; both ``log Z``
        and ``log Z_max`` are returned so a caller can renormalize against another
        reference. Returns ``(normalized, log_z_max)``.
        """
        if log_z is None:
            return None, None

        cf_graph = _contradiction_free_graph(result.fact_graph)
        cf_network = build_markov_network(
            cf_graph, use_priors=True, node_priors=self._node_priors(result)
        )
        log_z_max = self._log_z(cf_network)
        if log_z_max is None:
            return None, None

        log_z_min = log_z  # deep-dive's stated Zmin choice: the base network.
        denom = log_z_max - log_z_min
        if abs(denom) < 1e-12:
            # No contradictions (or none bind): base is already maximally coherent.
            return 1.0, log_z_max
        return (log_z - log_z_min) / denom, log_z_max

    # -- network construction helpers ----------------------------------------

    def _node_priors(self, result: MiningResult) -> Dict[str, float]:
        """The per-atom priors used when (re)building a network from the graph."""
        prior = float(result.config.get("prior", 0.5))
        return {aid: prior for aid in result.atoms}

    def _base_network(self, result: MiningResult) -> MarkovNetwork:
        """Rebuild the base coherence MRF from the fact graph.

        Rebuilding (rather than mutating ``result.markov_network``) gives a fresh
        network the derived variables can be appended to without disturbing the
        mining result.
        """
        return build_markov_network(
            result.fact_graph, use_priors=True, node_priors=self._node_priors(result)
        )


# ----------------------------------------------------------------------------
# Deterministic / vote factor tables (row-major over the listed variable order).
# ----------------------------------------------------------------------------


def _and_factor() -> List[float]:
    """Deterministic ``u = (s AND t)`` over [u, s, t] (row-major, 8 values).

    Value 1.0 iff ``u == (s==1 and t==1)``, else 0.0.
    """
    vals = []
    for u in (0, 1):
        for s in (0, 1):
            for t in (0, 1):
                vals.append(1.0 if u == (1 if (s == 1 and t == 1) else 0) else 0.0)
    return vals


def _or_factor() -> List[float]:
    """Deterministic ``w = (a OR b)`` over [w, a, b] (row-major, 8 values)."""
    vals = []
    for w in (0, 1):
        for a in (0, 1):
            for b in (0, 1):
                vals.append(1.0 if w == (1 if (a == 1 or b == 1) else 0) else 0.0)
    return vals


def _vote_factor(level1_type: str, p: float) -> List[float]:
    """Reified vote factor ``h_r(R, s, t)`` over [R, s, t] (row-major, 8 values).

    R=0 branch is flat (1.0). R=1 branch is 1.0 when the relation is satisfied at
    (s, t) and ``1 - p`` when it is violated (deep-dive Eq. 6). Violation:
      * entailment  -> (s=1, t=0)
      * equivalence -> (s != t)
      * contradiction -> (s=1, t=1)
    """
    vals = []
    for R in (0, 1):
        for s in (0, 1):
            for t in (0, 1):
                if R == 0:
                    vals.append(1.0)
                else:
                    vals.append(1.0 if _satisfied(level1_type, s, t) else 1.0 - p)
    return vals


def _satisfied(level1_type: str, s: int, t: int) -> bool:
    """Whether relation ``level1_type`` is satisfied at atom states (s, t)."""
    if level1_type == "entailment":
        return not (s == 1 and t == 0)
    if level1_type == "equivalence":
        return s == t
    if level1_type == "contradiction":
        return not (s == 1 and t == 1)
    raise ValueError(f"Unknown relation type: {level1_type}")


def _contradiction_free_graph(fact_graph: FactGraph) -> FactGraph:
    """Return a copy of ``fact_graph`` with all contradiction edges removed."""
    cf = FactGraph()
    for node in fact_graph.get_nodes():
        cf.add_node(Node(id=node.id, type=node.type, probability=node.probability))
    for edge in fact_graph.get_edges():
        if edge.type == "contradiction":
            continue
        cf.add_edge(
            Edge(
                source=edge.source,
                target=edge.target,
                type=edge.type,
                probability=edge.probability,
                link=edge.link,
            )
        )
    return cf
