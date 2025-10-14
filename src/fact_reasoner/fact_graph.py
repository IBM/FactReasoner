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
from typing import List

import networkx as nx
from tqdm import tqdm


class Node:
    def __init__(
            self, 
            id: str, 
            type: str,
            probability: float = 1.0
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
        """

        assert (type in ["atom", "context"]), \
            f"Uknown node type: {type}."
        self.id = id
        self.type = type
        self.probability = probability

    def __str__(self):
        return f"Node {self.id} ({self.type}): {self.probability}"
    
class Edge:
    def __init__(
            self, 
            source: str, 
            target: str, 
            type: str, 
            probability: float,
            link: str
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
                The type of link: context_atom, context_context, atom_atom
        """

        assert (type in ["entailment", "contradiction", "neutral", "equivalence"]), \
            f"Unknown relation type: {type}."
        assert (link in ["context_atom", "context_context", "atom_atom"]), \
            f"Unknown link type: {link}"
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
            atoms: List = None,
            contexts: List = None,
            relations: List = None,
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
        self.nodes = {}
        self.edges = []

        if atoms is not None:
            for atom in tqdm(atoms, desc="Atoms"):
                node = Node(
                    id=atom.id, 
                    type="atom", 
                    probability=atom.probability
                )
                self.add_node(node)
        if contexts is not None:
            for context in tqdm(contexts, desc="Contexts"):
                node = Node(
                    id=context.id, 
                    type="context", 
                    probability=context.probability
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
                        link=rel.link
                    )
                )
        
    def get_nodes(self) -> list:
        return list(self.nodes.values())
    
    def get_edges(self) -> list:
        return self.edges
    
    def add_node(
            self, 
            node: Node
    ):
        """
        Add a new node to the graph.

        Args:
            node: Node
                A new node to be added to the graph
        """

        self.nodes[node.id] = node

    def add_edge(
            self,
            edge: Edge
    ):
        """
        Add a new edge to the graph.

        Args:
            edge: Edge
                A new edge to be added to the graph
        """

        self.edges.append(edge)

    def from_json(
            self,
            json_file: str
    ):
        """
        Create the FactGraph from a json file.

        Args:
            json_file: str
                Path to the json file containing the graph.
        """

        with open(json_file, "r") as f:
            data = json.load(f)
            assert ("nodes" in data and "edges" in data), f"Uknown graph format"

            self.nodes = {}
            for node in tqdm(data["nodes"], desc="Nodes"):
                self.add_node(
                    Node(
                        id=node["id"], 
                        type=node["type"],
                        probability=node["probability"]
                    )
                )
            
            self.edges = []
            for edge in tqdm(data["edges"], desc="Edges"):
                self.edges.append(
                    Edge(
                        source=edge["from"],
                        target=edge["to"],
                        type=edge["relation"],
                        probability=edge["probability"],
                        link=edge["link"]
                    )
                )

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
                G.add_edge(edge.source, edge.target, color="green", label="{:.4g}".format(edge.probability))
            elif edge.type == "contradiction":
                G.add_edge(edge.source, edge.target, color="red", label="{:.4g}".format(edge.probability))
            elif edge.type == "equivalence":
                G.add_edge(edge.source, edge.target, color="blue", label="{:.4g}".format(edge.probability))

        return G

    def as_json_graph(self) -> dict:
        """
        Generate a full JSON-serializable representation of the FactGraph
        that can be round-tripped into `from_json`.

        Returns:
            dict with 'nodes' and 'edges'.
        """
        graph = {
            "nodes": [],
            "edges": []
        }

        for node in self.nodes.values():
            node_data = {
                "id": node.id,
                "type": node.type,
                "probability": node.probability,
            }
            graph["nodes"].append(node_data)

        for edge in self.edges:
            edge_data = {
                "from": edge.source,
                "to": edge.target,
                "relation": edge.type,
                "probability": edge.probability,
                "link": edge.link,
            }
            graph["edges"].append(edge_data)

        return graph


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
    g.from_json(json_file=file)
    g.dump()
    print(f"Done.")
    g.dump()
    print(f"Done.")
