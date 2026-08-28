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

# Pairwise factor tables and Markov network construction from a fact graph.
#
# This is a pipeline-agnostic building block: it turns a FactGraph (nodes +
# typed, weighted edges) into a MarkovNetwork by adding one unary prior factor
# per node and one pairwise factor per edge. The factor tables are laid out
# row-major over ``[source, target]`` (value order (0,0),(0,1),(1,0),(1,1)),
# matching the UAI serialization in :mod:`fact_reasoner.markov_network`.
#
# The tables come in two variants:
#   * no-priors  -- the source being true is charged whenever the target is
#     uncertain. Correct for atom<->context factuality.
#   * with-priors -- the source's own probability is decoupled from the
#     implication. The default (``use_priors=True``).


from fact_reasoner.core.base import PRIOR_PROB_ATOM, PRIOR_PROB_CONTEXT
from fact_reasoner.fact_graph import FactGraph
from fact_reasoner.markov_network import MarkovNetwork

# Every table below uses BOTH ``p`` and ``1 - p``, so a probability of exactly 0.0 or
# exactly 1.0 writes a literal zero into it -- one cell for the directed couplings,
# two of four for `equivalence` / `exclusive`, which converts a soft preference into a
# hard logical constraint. Enough of those on overlapping variables and no world has
# any mass: Merlin then reports `logZ: -inf` with
# `"status":"false","message":"Inconsistent evidence or underflow"` and an empty
# `marginals` list, and the cell cannot be scored at all.
#
# Clamping here rather than at the estimator is deliberate: this is the single
# chokepoint every graph passes through on its way into the network, so no caller --
# present or future -- can reintroduce the problem, and the probabilities the miner
# RECORDS stay exactly what the model reported (the mining diagnostics would otherwise
# lie about what was extracted).
#
# 1e-6 is the established convention in this repo, not a new invention:
# `lcs/strength.py::_EPS` and `uncertainty/simbauq.py::_CLF_EPS` both use it two-sided,
# and `assessor.py` already floors a probability just before building factors. Gold
# arms are unaffected by construction -- `locoeval/gold_graph.band_probability` returns
# band midpoints 0.925 / 0.720 / 0.470 and atom priors are 0.9 / 0.1, all interior.
PROB_EPS = 1e-6


def _clamp_probability(prob: float) -> float:
    """Keep a factor probability strictly inside ``(0, 1)``.

    Args:
        prob: A probability that may sit on either boundary.

    Returns:
        ``prob`` confined to ``[PROB_EPS, 1 - PROB_EPS]``.
    """
    return max(PROB_EPS, min(1.0 - PROB_EPS, prob))


def pairwise_prior(link: str) -> float:
    """Return the source-node prior for a pairwise factor given its link type.

    Args:
        link: The edge link type ("context_atom", "context_context", or
            "atom_atom").

    Returns:
        The prior probability of the source node being true.

    Raises:
        ValueError: If ``link`` is not a known link type.
    """
    if link == "context_context":
        return PRIOR_PROB_CONTEXT
    elif link in ("context_atom", "atom_atom"):
        return PRIOR_PROB_ATOM
    else:
        raise ValueError(f"Unknown link type: {link}")


def edge_factor_values(edge, use_priors: bool = True) -> list[float]:
    """Compute the flattened pairwise factor table for a fact-graph edge.

    The table is laid out row-major over ``[source, target]``, i.e. the value
    order is (src=0,trg=0), (src=0,trg=1), (src=1,trg=0), (src=1,trg=1).

    Args:
        edge: A fact-graph edge with ``type``, ``link`` and ``probability``.
        use_priors: Whether to use the with-priors tables (decoupling the
            source's own probability from the implication). ``True`` by default.

    Returns:
        The four factor values for the pairwise factor.

    Raises:
        ValueError: If ``edge.type`` is not a known relation type.
    """
    prob = _clamp_probability(edge.probability)
    if edge.type == "entailment":  # source true implies target true
        if use_priors:
            src_prior = pairwise_prior(edge.link)
            return [1.0 - src_prior, src_prior, 1.0 - prob, prob]
        return [prob, prob, 1.0 - prob, prob]
    elif edge.type == "contradiction":  # source true implies target false
        if use_priors:
            src_prior = pairwise_prior(edge.link)
            return [1.0 - src_prior, src_prior, prob, 1.0 - prob]
        return [prob, prob, prob, 1.0 - prob]
    elif edge.type == "equivalence":  # source and target agree
        return [prob, 1.0 - prob, 1.0 - prob, prob]
    elif edge.type == "exclusive":  # exactly one holds: penalize (0,0) AND (1,1)
        # Symmetric; equivalence with the interaction sign flipped. Same in both
        # variants (no source-prior term -- it already pushes both endpoints).
        # Row-major (0,0),(0,1),(1,0),(1,1) = [1-p, p, p, 1-p].
        return [1.0 - prob, prob, prob, 1.0 - prob]
    elif edge.type == "co_necessity":  # at least one holds: penalize only (0,0)
        if use_priors:
            src_prior = pairwise_prior(edge.link)
            # [1-p, pi_s, pi_s, p]: only the both-false world is down-weighted;
            # the source keeps its own prior on the (0,*) / (1,*) split.
            return [1.0 - prob, src_prior, src_prior, prob]
        return [1.0 - prob, prob, prob, prob]
    else:
        raise ValueError(f"Unknown edge type: {edge.type}")


def build_markov_network(
    fact_graph: FactGraph,
    *,
    use_priors: bool = True,
    node_priors: dict[str, float] | None = None,
) -> MarkovNetwork:
    """Create the Markov network corresponding to a fact graph.

    Adds one unary prior factor ``[1-pi, pi]`` per node and one pairwise factor
    per edge (via :func:`edge_factor_values`).

    Args:
        fact_graph: The :class:`FactGraph` whose nodes and edges define the
            network.
        use_priors: Whether the pairwise factors use the with-priors tables.
        node_priors: Optional mapping from node id to prior probability, used to
            override a node's own ``probability`` (e.g. a uniform 0.5). When a
            node id is absent, the node's stored ``probability`` is used.

    Returns:
        A :class:`MarkovNetwork` encoding the fact graph.

    Raises:
        ValueError: If a node has an unknown type.
    """
    network = MarkovNetwork()

    # Unary prior factors for the atom/context variables.
    for node in fact_graph.get_nodes():
        if node.type not in ("atom", "context"):
            raise ValueError(f"Unknown node type: {node.type}")
        x = node.id
        prob = node.probability
        if node_priors is not None and x in node_priors:
            prob = node_priors[x]
        # Clamp AFTER the override, not before: an injected prior of exactly 0.0/1.0
        # would otherwise bypass the guard and zero out the variable's unary factor.
        prob = _clamp_probability(prob)
        network.add_node(x)
        network.add_factor([x], [2], [1.0 - prob, prob])

    # Pairwise factors for the edges.
    for edge in fact_graph.get_edges():
        x, y = edge.source, edge.target
        network.add_edge(x, y)
        values = edge_factor_values(edge, use_priors=use_priors)
        network.add_factor([x, y], [2, 2], values)

    return network
