# The entry-point script
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

# Factuality graph

import json
from statistics import mode
from typing import List, Literal, cast, overload, override
from tqdm import tqdm
import networkx as nx

from src.fact_reasoner.fact_components import Atom, Context

type EdgeType = Literal["equivalence", "entailment", "contradiction"]


class Node:
    def __init__(
        self,
        id: str,
        type: str,
        probability: float = 1.0,
        entity: Atom | Context | None = None,
    ):
        """
        Create a node in the graph.

        Args:
            id: str
                Unique ID of the node in the graph.
            type: str
                The node type: ["atom", "context"].
            probability: float
                The prior probability associated with the "atom" or "context".
            entity: Atom | Context | None
                The source entity associated with the node.
        """

        assert type in ["atom", "context"], f"Uknown node type: {type}."
        self.id = id
        self.type = type
        self.probability = probability
        self.entity = entity

    @override
    def __str__(self):
        return f"Node {self.id} ({self.type}): {self.probability}"

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return (
            self.id == other.id
            and self.type == other.type
            and self.probability == other.probability
        )

    @override
    def __hash__(self):
        return hash((self.id, self.type, self.probability))


class Edge:
    def __init__(
        self, source: str, target: str, type: str, probability: float, link: str
    ):
        """
        Create an edge in the graph.

        Args:
            source: str
                The `from` node ID in the graph.
            target: str
                The `to` node ID in the graph.
            type: str
                The NLI relation type represented by the edge. Allowed values are:
                ["entailment", "contradiction", "equivalence"].
            probability: float
                The probability value associated with the NLI relation type.
            link: str
                The type of link: context_atom, context_context, atom_atom, atom_context
        """

        assert type in ["entailment", "contradiction", "equivalence"], (
            f"Unknown relation type: {type}."
        )
        assert link in [
            "context_atom",
            "context_context",
            "atom_atom",
            "atom_context",
        ], f"Unknown link type: {link}"
        self.source = source
        self.target = target
        self.type = type
        self.probability = probability
        self.link = link

    def __str__(self):
        return f"[{self.source} -> {self.target} ({self.type}): {self.probability}]"


class FactGraph:
    """
    A graph representation of the atom-context relations.

    """

    def __init__(
        self,
        atoms: List | None = None,
        contexts: List | None = None,
        relations: List | None = None,
    ):
        """
        FactGraph constructor.

        Args:
            atoms: List
                The list of atoms in the answer.
            contexts: List
                The list of contexts for each of the atoms. Each context contains
                a reference to its corresponding atom (by construction).
            relations: List
                The list of relations between atoms and contexts.
        """

        # Initialize an empty graph
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []

        if atoms is not None:
            for atom in tqdm(atoms, desc="Atoms"):
                node = Node(
                    id=atom.id, type="atom", entity=atom, probability=atom.probability
                )
                self.add_node(node)
        if contexts is not None:
            for context in tqdm(contexts, desc="Contexts"):
                node = Node(
                    id=context.id,
                    type="context",
                    entity=context,
                    probability=context.probability,
                )
                self.add_node(node)

        if relations is not None:
            for rel in tqdm(relations, desc="Relations"):
                self.edges.append(
                    Edge(
                        source=rel.source.id,
                        target=rel.target.id,
                        type=rel.type,
                        probability=rel.probability,
                        link=rel.link,
                    )
                )

    def get_nodes(self) -> list:
        return list(self.nodes.values())

    def get_edges(self) -> list:
        return self.edges

    def add_node(self, node: Node):
        """
        Add a new node to the graph.

        Args:
            node: Node
                A new node to be added to the graph
        """

        self.nodes[node.id] = node

    def add_edge(self, edge: Edge):
        """
        Add a new edge to the graph.

        Args:
            edge: Edge
                A new edge to be added to the graph
        """

        self.edges.append(edge)

    @overload
    def get_children(
        self,
        node_id: str,
        return_edges: Literal[False] = False,
        relation_types: list[EdgeType] | None = None,
    ) -> list[Node]: ...
    @overload
    def get_children(
        self,
        node_id: str,
        return_edges: Literal[True],
        relation_types: list[EdgeType] | None = None,
    ) -> list[tuple[Edge, Node]]: ...
    def get_children(
        self,
        node_id: str,
        return_edges: bool = False,
        relation_types: list[EdgeType] | None = None,
    ) -> list[Node] | list[tuple[Edge, Node]]:
        """
        Retrieves children of the given node, optionally with the
        corresponding edges.

        Args:
            node_id: str
                The ID of the node for which to retrieve children.
            return_edges: bool
                A boolean flag indicating whether to return the
                edges to the node children.
            relation_types: list[EdgeType] | None
                A list of edge/relation types to consider.

        Returns:
            list[Node] | list[tuple[Edge, Node]]:
                A list of child nodes, or, when `return_edges` is True,
                a list of edge and child node tuples.
        """
        unidirectional_results = [
            ((e, self.nodes[e.target]) if return_edges else self.nodes[e.target])
            for e in self.edges
            if e.source == node_id
            and (relation_types is None or e.type in relation_types)
        ]
        bidirectional_results = [
            ((e, self.nodes[e.source]) if return_edges else self.nodes[e.source])
            for e in self.edges
            if e.target == node_id
            and (
                e.type in ["equivalence", "contradiction"]
                and (relation_types is None or e.type in relation_types)
            )
        ]
        return unidirectional_results + bidirectional_results  # type: ignore

    @overload
    def get_parents(
        self,
        node_id: str,
        return_edges: Literal[False] = False,
        relation_types: list[EdgeType] | None = None,
    ) -> list[Node]: ...
    @overload
    def get_parents(
        self,
        node_id: str,
        return_edges: Literal[True],
        relation_types: list[EdgeType] | None,
    ) -> list[tuple[Edge, Node]]: ...
    def get_parents(
        self,
        node_id: str,
        return_edges: bool = False,
        relation_types: list[EdgeType] | None = None,
    ) -> list[Node] | list[tuple[Edge, Node]]:
        """
        Retrieves parents of the given node, optionally with the
        corresponding edges.

        Args:
            node_id: str
                The ID of the node for which to retrieve parents.
            return_edges: bool
                A boolean flag indicating whether to return the
                edges to the node parents.
            relation_types: list[EdgeType] | None
                A list of edge/relation types to consider.

        Returns:
            list[Node] | list[tuple[Edge, Node]]:
                A list of parent nodes, or, when `return_edges` is True,
                a list of edge and parent node tuples.
        """
        unidirectional_results = [
            ((e, self.nodes[e.source]) if return_edges else self.nodes[e.source])
            for e in self.edges
            if e.target == node_id
            and (relation_types is None or e.type in relation_types)
        ]
        bidirectional_results = [
            ((e, self.nodes[e.target]) if return_edges else self.nodes[e.target])
            for e in self.edges
            if e.source == node_id
            and (
                e.type in ["equivalence", "contradiction"]
                and (relation_types is None or e.type in relation_types)
            )
        ]
        return unidirectional_results + bidirectional_results  # type: ignore

    def find_reachable_nodes(
        self, relation_types: list[EdgeType], descendants: bool = True
    ) -> dict[str, set[Node]]:
        """
        Finds sets of nodes reachable from each node in the fact graph
        by following child (descendant) or parent (ancestor) edges.

        Args:
            relation_types: list[EdgeType]
                The types of edges to be considered during traversal.
            descendants: bool
                If `True`, traverse the descendants from parents to childern,
                if `False`, traverse the ancestors from children to parents.

        Returns:
            dict[str, set[Node]]:
                A dictionary mapping each node ID to its ancestors/descendants
                according to the specified parameters, including the node itself.
        """
        results: dict[str, set[Node]] = {}
        for node_id, node in self.nodes.items():
            reachable_nodes = set([node])
            nodes_to_visit = [node_id]
            while nodes_to_visit:
                current_node = nodes_to_visit.pop()
                if descendants:
                    next_nodes = self.get_children(
                        current_node, relation_types=relation_types
                    )
                else:
                    next_nodes = self.get_parents(
                        current_node, relation_types=relation_types
                    )
                for next_node in next_nodes:
                    if next_node not in reachable_nodes:
                        reachable_nodes.add(next_node)
                        nodes_to_visit.append(next_node.id)
            results[node_id] = reachable_nodes
        return results

    @property
    def covered_context_ids(self) -> list[str]:
        """
        The list of context IDs covered in the response.

        Returns:
            list[str]:
                The list of context IDs covered in the response.
        """
        ancestors_dict = self.find_reachable_nodes(
            relation_types=["equivalence", "entailment"], descendants=False
        )
        return [
            nid
            for nid, ancestors in ancestors_dict.items()
            if any(a.type == "atom" for a in ancestors)
            and self.nodes[nid].type == "context"
        ]

    @property
    def uncovered_context_ids(self) -> list[str]:
        """
        The list of context IDs not covered in the response.

        Returns:
            list[str]:
                The list of context IDs not covered in the response.
        """
        ancestors_dict = self.find_reachable_nodes(
            relation_types=["equivalence", "entailment"], descendants=False
        )
        return [
            nid
            for nid, ancestors in ancestors_dict.items()
            if all(a.type == "context" for a in ancestors)
            and self.nodes[nid].type == "context"
        ]

    @property
    def context_equivalence_clusters(self) -> dict[str, list[str]]:
        """
        A dictionary mapping context equivalence cluster "prototype" IDs
        to list of node IDs in the given equivalence cluster. Equivalence
        clusters are computed as strongly connected components in the
        entailment and equivalence graph. Prototypes for each cluster
        are selected as textual modes of the cluster members.

        Returns:
            dict[str, list[str]]: The dictionary mapping equivalence cluster
                prototype IDs to the IDs of nodes in each cluster.
        """
        entailment_digraph = self.as_coverage_digraph(
            relation_types=["equivalence", "entailment"]
        )
        # Identify strongly connected components for determining the clusters
        condensation = nx.condensation(entailment_digraph)
        equivalence_clusters: dict[str, list[str]] = {}
        for component_node_id in condensation.nodes:
            component_members = cast(
                set[str], condensation.nodes.data()[component_node_id]["members"]
            )
            component_members = {
                m for m in component_members if self.nodes[m].type == "context"
            }

            # Skip the cluster if there are no contexts
            if not component_members:
                continue

            # Find the mode of the nodes in the component to serve as a prototype
            texts = [
                self.nodes[nid].entity.text  # type: ignore
                for nid in component_members
            ]
            text_mode = mode(texts)
            prototype_id = next(
                nid
                for nid in component_members
                if self.nodes[nid].entity.text == text_mode  # type: ignore
            )

            equivalence_clusters[prototype_id] = list(component_members)

        return equivalence_clusters

    @property
    def uncovered_context_basis(self) -> dict[str, list[str]]:
        """
        A dictionary mapping basis context IDs to the IDs of contexts
        they cover or would cover if included in the model response. Covering
        all the uncovered basis contexts in addition to any contexts already
        covered in the model output would be expected to result in 100$
        comprehensiveness. Note that each context might be covered by multiple
        basis nodes, but is only associated with the first such node in the
        topological order on the FactGraph condensation. This means that the
        lists of context IDs in the dictionary values partition
        uncovered_context_ids.

        For example, in a graph c1 → c2 → a1 → c3 (where → are entailment edges),
        the uncovered_context_prototypes should be {'c1': ['c1', 'c2']},
        as covering c1 would be sufficient for covering both c1 and c2
        (while c3 is already covered by answer atom a1).

        Returns:
            dict[str, list[str]]:
                A dictionary mapping the IDs of context prototypes to IDs
                of nodes that would be covered by these prototypes.
        """
        descendants_dict = self.find_reachable_nodes(
            relation_types=["equivalence", "entailment"], descendants=True
        )
        coverage_digraph = self.as_coverage_digraph(
            relation_types=["equivalence", "entailment"]
        )
        covered_context_ids = self.covered_context_ids
        uncovered_context_ids = self.uncovered_context_ids
        context_equivalence_clusters = self.context_equivalence_clusters
        # Identify strongly connected components for determining the clusters
        condensation = nx.condensation(coverage_digraph)
        currently_uncovered_ids = set(uncovered_context_ids)
        prototypes: dict[str, list[str]] = {}
        # Go over SCCs in topological order to obtain a minimal set of prototypes
        for component_node_id in nx.topological_sort(condensation):
            # Find the corresponding context equivalence cluster
            component_members = cast(
                set[str], condensation.nodes.data()[component_node_id]["members"]
            )
            matching_clusters = [
                (pid, cids)
                for pid, cids in context_equivalence_clusters.items()
                if any([m in cids for m in component_members])
            ]
            if not matching_clusters:
                # The current cluster only consists of answer atoms — skip
                continue
            prototype_id, cluster_node_ids = matching_clusters[0]

            # Safety checks
            assert len(matching_clusters) == 1, (
                f"Expected matching_clusters to be a singleton, but got {matching_clusters}."
            )
            assert all([nid in covered_context_ids for nid in cluster_node_ids]) or all(
                [nid in uncovered_context_ids for nid in cluster_node_ids]
            ), (
                f"Mismatch between equivalence cluster nodes coverage: {cluster_node_ids}"
            )

            if prototype_id not in currently_uncovered_ids:
                # All cluster members are already covered, so we continue
                # with another cluster.
                continue

            prototype_descendants = {n.id for n in descendants_dict[prototype_id]}
            newly_covered = list(
                currently_uncovered_ids.intersection(prototype_descendants)
            )
            prototypes[prototype_id] = newly_covered
            currently_uncovered_ids = currently_uncovered_ids - prototype_descendants

        return prototypes

    def get_coverage_path(self, context_id: str) -> tuple[list[Edge], list[str]]:
        """
        Computes a coverage path for the context with the specified ID.

        Args:
            context_id: str
                The ID of the context.

        Returns:
            tuple[list[Edge], list[str]]:
                A tuple containing a list of edges and node IDs on the
                coverage path to the given context.
        """
        if context_id not in self.covered_context_ids:
            raise ValueError("Cannot find coverage path, context is not covered.")
        candidate_paths: list[tuple[list[Edge], list[str]]] = [([], [context_id])]
        while True:
            candidate_edges, candidate_nodes = candidate_paths.pop(0)
            last_node = candidate_nodes[-1]
            next_parents = self.get_parents(
                last_node,
                relation_types=["entailment", "equivalence"],
                return_edges=True,
            )
            for edge, parent in next_parents:
                if parent.id not in candidate_nodes:
                    new_edges = list(candidate_edges)
                    new_nodes = list(candidate_nodes)
                    new_edges.append(edge)
                    new_nodes.append(parent.id)

                    if parent.type == "atom":
                        return new_edges, new_nodes

                    candidate_paths.append((new_edges, new_nodes))

    def show_coverage_path(self, context_id: str) -> str:
        """
        Produces a string representation of a coverage path for the context
        with the specified ID.

        Args:
            context_id: str
                The ID of the context.

        Returns:
            str:
                The string representation of a coverage path for the
                given contex.t
        """
        path_edges, path_nodes = self.get_coverage_path(context_id)
        explanation = context_id
        for edge, next_node in zip(path_edges, path_nodes[1:]):
            if edge.type == "equivalence":
                explanation += f" = {next_node}"
            elif edge.target == next_node and edge.type == "entailment":
                explanation += f" → {next_node}"
            elif edge.source == next_node and edge.type == "entailment":
                explanation += f" ← {next_node}"
        return explanation

    @staticmethod
    def from_json(
        json_str: str,
        atoms: list[Atom] | None = None,
        contexts: list[Context] | None = None,
    ) -> "FactGraph":
        """
        Create the FactGraph from a json string.

        Args:
            json_str: str
                The json string containing the graph.
            atoms: list[Atom] | None
                The atoms to populate in the `entity` fields of the nodes.
                Optional.
            contexts: list[Context] | None
                The contexts to populate in the `entity` fields of the nodes.
                Optional.
        """
        data = json.loads(json_str)
        assert "nodes" in data and "edges" in data, "Uknown graph format"

        fg = FactGraph()
        for node in tqdm(data["nodes"], desc="Nodes"):
            fg.add_node(
                Node(id=node["id"], type=node["type"], probability=node["probability"])
            )

        for edge in tqdm(data["edges"], desc="Edges"):
            fg.add_edge(
                Edge(
                    source=edge["source"],
                    target=edge["target"],
                    type=edge["type"],
                    probability=edge["probability"],
                    link=edge["link"],
                )
            )

        if atoms is not None:
            for atom in atoms:
                fg.nodes[atom.id].entity = atom
        if contexts is not None:
            for context in contexts:
                fg.nodes[context.id].entity = context

        return fg

    @staticmethod
    def from_json_file(json_file: str) -> "FactGraph":
        """
        Create the FactGraph from a json file.

        Args:
            json_file: str
                Path to the json file containing the graph.
        """

        with open(json_file, "r") as f:
            data = f.read()
            return FactGraph.from_json(data)

    def to_json(self):
        """
        Convert the graph to a JSON string.
        """
        nodes = [
            {"id": node.id, "type": node.type, "probability": node.probability}
            for node in self.nodes.values()
        ]
        edges = [
            {
                "source": edge.source,
                "target": edge.target,
                "type": edge.type,
                "probability": edge.probability,
                "link": edge.link,
            }
            for edge in self.edges
        ]
        data = {"nodes": nodes, "edges": edges}
        return json.dumps(data)

    def to_json_file(self, json_file: str):
        """
        Convert the graph to a JSON file.

        Args:
            json_file: str
                Path to the json file to save the graph in.
        """
        json_str = self.to_json()
        with open(json_file, "w") as f:
            f.write(json_str)

    def as_digraph(self):
        """
        Generate a networkx.DiGraph representation of the fact graph.
        """
        G = nx.DiGraph()
        for _, node in self.nodes.items():
            if node.type == "atom":
                G.add_node(node.id, color="green")
            else:
                G.add_node(node.id, color="orange")

        for edge in self.edges:
            if edge.type == "entailment":
                G.add_edge(
                    edge.source,
                    edge.target,
                    color="green",
                    label="{:.4g}".format(edge.probability),
                )
            elif edge.type == "contradiction":
                G.add_edge(
                    edge.source,
                    edge.target,
                    color="red",
                    label="{:.4g}".format(edge.probability),
                )
            elif edge.type == "equivalence":
                G.add_edge(
                    edge.source,
                    edge.target,
                    color="blue",
                    label="{:.4g}".format(edge.probability),
                )

        return G

    def as_coverage_digraph(
        self, relation_types: list[EdgeType] | None = None
    ) -> nx.DiGraph:
        """
        Generates a coverage-oriented networkx.DiGraph representation of the
        fact graph, representing equivalence and contradiction bidirectionality
        using two edges in different directions.

        Arguments:
            relation_types: list[EdgeType] | None
                Specifies which of the relation types should be included
                in the graph. Defaults to all relation types if None.

        Returns:
            nx.DiGraph:
                The generated DiGraph object.
        """
        if relation_types is None:
            relation_types = ["equivalence", "entailment", "contradiction"]

        G = nx.DiGraph()
        for _, node in self.nodes.items():
            if node.type == "atom":
                G.add_node(node.id, color="dodgerblue")
            elif node.id in self.covered_context_ids:
                G.add_node(node.id, color="forestgreen")
            else:
                G.add_node(node.id, color="crimson")

        EDGE_COLORS: dict[EdgeType, str] = {
            "equivalence": "forestgreen",
            "entailment": "forestgreen",
            "contradiction": "crimson",
        }

        for edge in self.edges:
            if edge.type not in relation_types:
                continue

            G.add_edge(
                edge.source,
                edge.target,
                color=EDGE_COLORS[edge.type],
                label=f"{edge.probability:.4f}",
            )

            if edge.type in ["equivalence", "contradiction"]:
                G.add_edge(
                    edge.target,
                    edge.source,
                    color=EDGE_COLORS[edge.type],
                    label=f"{edge.probability:.4f}",
                )

        return G

    def dump(self):
        print("Nodes:")
        for i, n in self.nodes.items():
            print(n)
        print("Edges:")
        for e in self.edges:
            print(e)
        print(f"Number of nodes: {len(self.nodes)}")
        print(f"Number of edges: {len(self.edges)}")


if __name__ == "__main__":
    file = "/home/radu/git/fm-factual/examples/graph.json"
    g = FactGraph()
    g.from_json_file(json_file=file)
    g.dump()
    print("Done.")
