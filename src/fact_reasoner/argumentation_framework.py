from dataclasses import dataclass
from typing import Literal, Protocol, overload
from math import prod

from src.fact_reasoner.fact_graph import Edge, FactGraph, Node


class AggregationFunction(Protocol):
    """
    Protocol for a modular argumentation semantics aggregation
    function. See https://doi.org/10.24963/kr.2024/56 for an
    introduction to modular argumentation semantics.
    """
    def __call__(
        self, attacker_strengths: list[float], supporter_strengths: list[float]
    ) -> float: ...


class InfluenceFunction(Protocol):
    """
    Protocol for a modular argumentation semantics influence
    function. See https://doi.org/10.24963/kr.2024/56 for an
    introduction to modular argumentation semantics.
    """
    def __call__(
        self, base_strength: float, aggregate: float, conservativeness: float = 1.0
    ) -> float: ...


@dataclass(frozen=True)
class ArgumentationSemantics:
    """
    A data class for defining argumentation semantics.
    
    Attributes:
        aggregation_fun: AggregationFunction
            The argumentation semantics aggregation function.
        influence_fun: InfluenceFunction
            The argumentation semantics influence function.
    """
    aggregation_fun: AggregationFunction
    influence_fun: InfluenceFunction


def sum_aggregation(
    attacker_strengths: list[float], supporter_strengths: list[float]
) -> float:
    """
    The sum aggregation function. Used in the Quadratic Energy
    semantics.

    Args:
        attacker_strengths: list[float]
            A list of attacker strengths.
        supporter_strengths: list[float]
            A list of supporter strengths.

    Returns:
        float:
            The resulting aggregate of attacker and supporter
            strengths.
    """
    return sum(supporter_strengths) - sum(attacker_strengths)


def product_aggregation(
    attacker_strengths: list[float], supporter_strengths: list[float]
) -> float:
    """
    The product aggregation function. Used in the DF-QuAD semantics.

    Args:
        attacker_strengths: list[float]
            A list of attacker strengths.
        supporter_strengths: list[float]
            A list of supporter strengths.

    Returns:
        float:
            The resulting aggregate of attacker and supporter
            strengths.
    """
    return prod([1 - a for a in attacker_strengths]) - prod(
        [1 - s for s in supporter_strengths]
    )


def linear_influence(
    base_strength: float, aggregate: float, conservativeness: float = 1.0
) -> float:
    """
    Linear influence function. Used in the DF-QuAD semantics.

    Args:
        base_strength: float
            The base strength of an argument.
        aggregate: float
            The aggregate strength of supporting/attacking arguments,
            computed using the aggregation function.
        conservativeness: float
            The conservativeness parameter. Higher conservativeness
            results in smaller argument strength updates.

    Returns:
        float:
            The updated argument strength.
    """
    return (
        base_strength
        + base_strength * min(0, aggregate) / conservativeness
        + (1 - base_strength) * max(0, aggregate) / conservativeness
    )


def qe_influence(
    base_strength: float, aggregate: float, conservativeness: float = 1.0
) -> float:
    """
    Quadratic Energy influence function. Used in the Quadratic Energy
    semantics.

    Args:
        base_strength: float
            The base strength of an argument.
        aggregate: float
            The aggregate strength of supporting/attacking arguments,
            computed using the aggregation function.
        conservativeness: float
            The conservativeness parameter. Higher conservativeness
            results in smaller argument strength updates.

    Returns:
        float:
            The updated argument strength.
    """
    aggregate /= conservativeness

    def h(agg):
        return max(0, agg) ** 2 / (1 + max(0, agg) ** 2)

    return (
        base_strength
        - base_strength * h(-aggregate)
        + (1 - base_strength) * h(aggregate)
    )


SUPPORTED_SEMANTICS: dict[str, ArgumentationSemantics] = {
    "qe": ArgumentationSemantics(
        aggregation_fun=sum_aggregation, influence_fun=qe_influence
    ),
    "dfquad": ArgumentationSemantics(
        aggregation_fun=product_aggregation, influence_fun=linear_influence
    ),
}


class ArgumentationFramework:
    """
    An argumentation framework based on a FactGraph.
    """

    def __init__(self, fact_graph: FactGraph):
        """
        Initializes an ArgumentationFramework based on a FactGraph.
        
        Args:
            fact_graph: FactGraph
                The underlying FactGraph object.
        """
        self.fact_graph = fact_graph

    @overload
    def get_attackers(
        self, node_id: str, return_edges: Literal[False] = False
    ) -> list[Node]: ...
    @overload
    def get_attackers(
        self, node_id: str, return_edges: Literal[True]
    ) -> list[tuple[Edge, Node]]: ...
    def get_attackers(
        self, node_id: str, return_edges: bool = False
    ) -> list[Node] | list[tuple[Edge, Node]]:
        """
        Retrieves attackers of the given node, optionally with the 
        corresponding edges.
        
        Args:
            node_id: str
                The ID of the node for which to retrieve attackers.
            return_edges: bool
                A boolean flag indicating whether to return the
                edges to the attackers.
                
        Returns:
            list[Node] | list[tuple[Edge, Node]]:
                A list of attacker nodes, or, when `return_edges` is True,
                a list of edge and attacker node tuples.
        """
        return self.fact_graph.get_parents(
            node_id, relation_types=["contradiction"], return_edges=return_edges
        )

    @overload
    def get_supporters(
        self, node_id: str, return_edges: Literal[False] = False
    ) -> list[Node]: ...
    @overload
    def get_supporters(
        self, node_id: str, return_edges: Literal[True]
    ) -> list[tuple[Edge, Node]]: ...
    def get_supporters(
        self, node_id: str, return_edges: bool = False
    ) -> list[Node] | list[tuple[Edge, Node]]:
        """
        Retrieves supporters of the given node, optionally with the
        corresponding edges.

        Args:
            node_id: str
                The ID of the node for which to retrieve supporters.
            return_edges: bool
                A boolean flag indicating whether to return the
                edges to the supporters.

        Returns:
            list[Node] | list[tuple[Edge, Node]]:
                A list of supporter nodes, or, when `return_edges` is True,
                a list of edge and supporter node tuples.
        """
        return self.fact_graph.get_parents(
            node_id, relation_types=["entailment"], return_edges=return_edges
        )

    def get_topological_sort(self) -> list[Node]:
        """
        Determines a topological sort of the nodes in the argumentation
        framework.
        
        Returns:
            list[Node]:
                A list of nodes in topological order.
        """
        entered_node_ids = []
        visited_node_ids = []

        def visit_node(node_id: str):
            if node_id in visited_node_ids:
                # Node already processed
                return
            if node_id in entered_node_ids:
                raise ValueError(
                    "Unable to determine topological sort — the fact graph has a cycle."
                )

            entered_node_ids.append(node_id)

            for child in self.fact_graph.get_children(node_id):
                visit_node(child.id)

            visited_node_ids.append(node_id)

        for node in self.fact_graph.nodes.keys():
            visit_node(node)

        return [self.fact_graph.nodes[nid] for nid in visited_node_ids[::-1]]

    def evaluate_strengths(
        self, semantics: Literal["qe", "dfquad"] = "qe", conservativeness: float = 1.0
    ) -> dict[str, float]:
        """
        Evaluates final strengths of the arguments in the argumentation
        framework using the specified semantics. For simplicity, this
        method currently doesn't support AFs with cycles.

        Args:
            semantics: Literal["qe", "dfquad"]
                The semantics to use for evaluating argument strengths.
            conservativeness: float
                The conservativeness value to use when evaluating argument strengths.

        Returns:
            dict[str, float]:
                A dictionary mapping node (variable) IDs to their final strenghts.
        """
        if semantics not in SUPPORTED_SEMANTICS:
            raise ValueError(
                f"Unsupported semantics {semantics}. "
                f"Supported values: {','.join(SUPPORTED_SEMANTICS.keys())}"
            )
        self.semantics = SUPPORTED_SEMANTICS[semantics]

        strengths: dict[str, float] = {}

        for node in self.get_topological_sort():
            attackers = self.get_attackers(node.id, return_edges=True)
            supporters = self.get_supporters(node.id, return_edges=True)

            aggregate = self.semantics.aggregation_fun(
                attacker_strengths=[
                    float(e.probability) * strengths[n.id] for (e, n) in attackers
                ],
                supporter_strengths=[
                    float(e.probability) * strengths[n.id] for (e, n) in supporters
                ],
            )
            strength = self.semantics.influence_fun(
                base_strength=float(node.probability),
                aggregate=aggregate,
                conservativeness=conservativeness,
            )

            strengths[node.id] = strength

        return strengths