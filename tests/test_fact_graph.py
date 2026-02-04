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

"""Unit tests for fact_reasoner.fact_graph module."""

import pytest
import json
import tempfile
import os

from src.fact_reasoner.fact_graph import Node, Edge, FactGraph
from src.fact_reasoner.core.utils import Atom, Context, Relation


class TestNode:
    """Tests for Node class."""

    def test_node_creation_atom(self):
        node = Node(id="a0", type="atom", probability=0.5)
        assert node.id == "a0"
        assert node.type == "atom"
        assert node.probability == 0.5

    def test_node_creation_context(self):
        node = Node(id="c0", type="context", probability=0.9)
        assert node.id == "c0"
        assert node.type == "context"
        assert node.probability == 0.9

    def test_node_default_probability(self):
        node = Node(id="a0", type="atom")
        assert node.probability == 1.0

    def test_node_invalid_type(self):
        with pytest.raises(AssertionError):
            Node(id="x0", type="invalid")

    def test_node_str(self):
        node = Node(id="a0", type="atom", probability=0.5)
        result = str(node)
        assert "a0" in result
        assert "atom" in result
        assert "0.5" in result


class TestEdge:
    """Tests for Edge class."""

    def test_edge_creation_entailment(self):
        edge = Edge(
            source="c0",
            target="a0",
            type="entailment",
            probability=0.9,
            link="context_atom"
        )
        assert edge.source == "c0"
        assert edge.target == "a0"
        assert edge.type == "entailment"
        assert edge.probability == 0.9
        assert edge.link == "context_atom"

    def test_edge_creation_contradiction(self):
        edge = Edge(
            source="c0",
            target="a0",
            type="contradiction",
            probability=0.85,
            link="context_atom"
        )
        assert edge.type == "contradiction"

    def test_edge_creation_equivalence(self):
        edge = Edge(
            source="c0",
            target="c1",
            type="equivalence",
            probability=0.95,
            link="context_context"
        )
        assert edge.type == "equivalence"
        assert edge.link == "context_context"

    def test_edge_invalid_type(self):
        with pytest.raises(AssertionError):
            Edge(
                source="c0",
                target="a0",
                type="neutral",  # neutral is not allowed in Edge
                probability=0.5,
                link="context_atom"
            )

    def test_edge_invalid_link(self):
        with pytest.raises(AssertionError):
            Edge(
                source="c0",
                target="a0",
                type="entailment",
                probability=0.5,
                link="invalid_link"
            )

    def test_edge_str(self):
        edge = Edge(
            source="c0",
            target="a0",
            type="entailment",
            probability=0.9,
            link="context_atom"
        )
        result = str(edge)
        assert "c0" in result
        assert "a0" in result
        assert "entailment" in result


class TestFactGraph:
    """Tests for FactGraph class."""

    def test_empty_graph(self):
        graph = FactGraph()
        assert len(graph.get_nodes()) == 0
        assert len(graph.get_edges()) == 0

    def test_graph_with_atoms(self):
        atoms = [
            Atom(id="a0", text="First atom"),
            Atom(id="a1", text="Second atom"),
        ]
        graph = FactGraph(atoms=atoms)
        nodes = graph.get_nodes()
        assert len(nodes) == 2
        assert all(n.type == "atom" for n in nodes)

    def test_graph_with_contexts(self):
        contexts = [
            Context(id="c0", atom=None, text="First context"),
            Context(id="c1", atom=None, text="Second context"),
        ]
        graph = FactGraph(contexts=contexts)
        nodes = graph.get_nodes()
        assert len(nodes) == 2
        assert all(n.type == "context" for n in nodes)

    def test_graph_with_relations(self):
        atom = Atom(id="a0", text="Atom")
        context = Context(id="c0", atom=atom, text="Context")
        relation = Relation(
            source=context,
            target=atom,
            type="entailment",
            probability=0.9,
            link="context_atom"
        )
        graph = FactGraph(atoms=[atom], contexts=[context], relations=[relation])

        assert len(graph.get_nodes()) == 2
        assert len(graph.get_edges()) == 1

        edge = graph.get_edges()[0]
        assert edge.source == "c0"
        assert edge.target == "a0"
        assert edge.type == "entailment"

    def test_add_node(self):
        graph = FactGraph()
        node = Node(id="a0", type="atom", probability=0.5)
        graph.add_node(node)
        assert len(graph.get_nodes()) == 1
        assert graph.nodes["a0"] == node

    def test_add_edge(self):
        graph = FactGraph()
        edge = Edge(
            source="c0",
            target="a0",
            type="entailment",
            probability=0.9,
            link="context_atom"
        )
        graph.add_edge(edge)
        assert len(graph.get_edges()) == 1

    def test_from_json(self):
        json_data = {
            "nodes": [
                {"id": "a0", "type": "atom", "probability": 0.5},
                {"id": "c0", "type": "context", "probability": 0.9},
            ],
            "edges": [
                {
                    "from": "c0",
                    "to": "a0",
                    "relation": "entailment",
                    "probability": 0.85,
                    "link": "context_atom"
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(json_data, f)
            temp_path = f.name

        try:
            graph = FactGraph()
            graph.from_json(temp_path)

            assert len(graph.get_nodes()) == 2
            assert len(graph.get_edges()) == 1

            edge = graph.get_edges()[0]
            assert edge.source == "c0"
            assert edge.target == "a0"
        finally:
            os.unlink(temp_path)

    def test_as_digraph(self):
        atom = Atom(id="a0", text="Atom")
        context = Context(id="c0", atom=atom, text="Context")
        relation = Relation(
            source=context,
            target=atom,
            type="entailment",
            probability=0.9,
            link="context_atom"
        )
        graph = FactGraph(atoms=[atom], contexts=[context], relations=[relation])

        digraph = graph.as_digraph()

        assert "a0" in digraph.nodes
        assert "c0" in digraph.nodes
        assert digraph.has_edge("c0", "a0")

    def test_as_digraph_colors(self):
        atom = Atom(id="a0", text="Atom")
        context = Context(id="c0", atom=atom, text="Context")
        graph = FactGraph(atoms=[atom], contexts=[context])

        digraph = graph.as_digraph()

        assert digraph.nodes["a0"]["color"] == "green"
        assert digraph.nodes["c0"]["color"] == "orange"

    def test_as_digraph_edge_colors(self):
        atom = Atom(id="a0", text="Atom")
        context = Context(id="c0", atom=atom, text="Context")

        # Test entailment edge
        rel_entail = Relation(
            source=context,
            target=atom,
            type="entailment",
            probability=0.9,
            link="context_atom"
        )
        graph = FactGraph(atoms=[atom], contexts=[context], relations=[rel_entail])
        digraph = graph.as_digraph()
        assert digraph.edges["c0", "a0"]["color"] == "green"

    def test_nodes_dict_access(self):
        atom = Atom(id="a0", text="Atom")
        graph = FactGraph(atoms=[atom])

        assert "a0" in graph.nodes
        assert graph.nodes["a0"].type == "atom"
