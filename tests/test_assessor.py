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

"""Unit tests for fact_reasoner.assessor module (FactReasoner class)."""

import pytest
import json
import tempfile
import os
from unittest.mock import MagicMock, patch

from src.fact_reasoner.assessor import FactReasoner
from src.fact_reasoner.fact_graph import FactGraph
from src.fact_reasoner.core.utils import Atom, Context, Relation


class TestFactReasonerInit:
    """Tests for FactReasoner initialization."""

    def test_init_requires_merlin_path(
        self, mock_atomizer, mock_reviser, mock_retriever, mock_nli_extractor, mock_summarizer
    ):
        with pytest.raises(AssertionError, match="Path to `merlin` cannot be None"):
            FactReasoner(
                atom_extractor=mock_atomizer,
                atom_reviser=mock_reviser,
                context_retriever=mock_retriever,
                context_summarizer=mock_summarizer,
                nli_extractor=mock_nli_extractor,
                merlin_path=None
            )

    def test_init_stores_components(
        self, mock_atomizer, mock_reviser, mock_retriever, mock_nli_extractor, mock_summarizer
    ):
        pipeline = FactReasoner(
            atom_extractor=mock_atomizer,
            atom_reviser=mock_reviser,
            context_retriever=mock_retriever,
            context_summarizer=mock_summarizer,
            nli_extractor=mock_nli_extractor,
            merlin_path="/path/to/merlin"
        )

        assert pipeline.atom_extractor == mock_atomizer
        assert pipeline.atom_reviser == mock_reviser
        assert pipeline.context_retriever == mock_retriever
        assert pipeline.context_summarizer == mock_summarizer
        assert pipeline.nli_extractor == mock_nli_extractor
        assert pipeline.merlin_path == "/path/to/merlin"

    def test_init_default_values(
        self, mock_atomizer, mock_reviser, mock_retriever, mock_nli_extractor, mock_summarizer
    ):
        pipeline = FactReasoner(
            atom_extractor=mock_atomizer,
            atom_reviser=mock_reviser,
            context_retriever=mock_retriever,
            context_summarizer=mock_summarizer,
            nli_extractor=mock_nli_extractor,
            merlin_path="/path/to/merlin"
        )

        assert pipeline.query is None
        assert pipeline.response is None
        assert pipeline.topic is None
        assert pipeline.use_priors is True
        assert pipeline.atoms == {}
        assert pipeline.contexts == {}
        assert pipeline.relations == []

    def test_init_use_priors_false(
        self, mock_atomizer, mock_reviser, mock_retriever, mock_nli_extractor, mock_summarizer
    ):
        pipeline = FactReasoner(
            atom_extractor=mock_atomizer,
            atom_reviser=mock_reviser,
            context_retriever=mock_retriever,
            context_summarizer=mock_summarizer,
            nli_extractor=mock_nli_extractor,
            merlin_path="/path/to/merlin",
            use_priors=False
        )

        assert pipeline.use_priors is False


class TestFactReasonerFromDictWithContexts:
    """Tests for FactReasoner.from_dict_with_contexts method."""

    def test_from_dict_loads_atoms(
        self, mock_atomizer, mock_reviser, mock_retriever, mock_nli_extractor,
        mock_summarizer, sample_json_data
    ):
        pipeline = FactReasoner(
            atom_extractor=mock_atomizer,
            atom_reviser=mock_reviser,
            context_retriever=mock_retriever,
            context_summarizer=mock_summarizer,
            nli_extractor=mock_nli_extractor,
            merlin_path="/path/to/merlin"
        )

        pipeline.from_dict_with_contexts(sample_json_data)

        assert len(pipeline.atoms) == 2
        assert "a0" in pipeline.atoms
        assert "a1" in pipeline.atoms
        assert pipeline.atoms["a0"].get_text() == "Albert Einstein was German-born."

    def test_from_dict_loads_contexts(
        self, mock_atomizer, mock_reviser, mock_retriever, mock_nli_extractor,
        mock_summarizer, sample_json_data
    ):
        pipeline = FactReasoner(
            atom_extractor=mock_atomizer,
            atom_reviser=mock_reviser,
            context_retriever=mock_retriever,
            context_summarizer=mock_summarizer,
            nli_extractor=mock_nli_extractor,
            merlin_path="/path/to/merlin"
        )

        pipeline.from_dict_with_contexts(sample_json_data)

        assert len(pipeline.contexts) == 2
        assert "c_a0_0" in pipeline.contexts
        assert "c_a1_0" in pipeline.contexts

    def test_from_dict_links_atoms_to_contexts(
        self, mock_atomizer, mock_reviser, mock_retriever, mock_nli_extractor,
        mock_summarizer, sample_json_data
    ):
        pipeline = FactReasoner(
            atom_extractor=mock_atomizer,
            atom_reviser=mock_reviser,
            context_retriever=mock_retriever,
            context_summarizer=mock_summarizer,
            nli_extractor=mock_nli_extractor,
            merlin_path="/path/to/merlin"
        )

        pipeline.from_dict_with_contexts(sample_json_data)

        atom_a0 = pipeline.atoms["a0"]
        contexts = atom_a0.get_contexts()
        assert "c_a0_0" in contexts

    def test_from_dict_loads_labels(
        self, mock_atomizer, mock_reviser, mock_retriever, mock_nli_extractor,
        mock_summarizer, sample_json_data
    ):
        pipeline = FactReasoner(
            atom_extractor=mock_atomizer,
            atom_reviser=mock_reviser,
            context_retriever=mock_retriever,
            context_summarizer=mock_summarizer,
            nli_extractor=mock_nli_extractor,
            merlin_path="/path/to/merlin"
        )

        pipeline.from_dict_with_contexts(sample_json_data)

        assert pipeline.labels_human is not None
        assert pipeline.labels_human["a0"] == "S"
        assert pipeline.labels_human["a1"] == "S"


class TestFactReasonerToJson:
    """Tests for FactReasoner.to_json method."""

    def test_to_json_returns_dict(
        self, mock_atomizer, mock_reviser, mock_retriever, mock_nli_extractor,
        mock_summarizer, sample_json_data
    ):
        pipeline = FactReasoner(
            atom_extractor=mock_atomizer,
            atom_reviser=mock_reviser,
            context_retriever=mock_retriever,
            context_summarizer=mock_summarizer,
            nli_extractor=mock_nli_extractor,
            merlin_path="/path/to/merlin"
        )

        pipeline.from_dict_with_contexts(sample_json_data)
        result = pipeline.to_json()

        assert isinstance(result, dict)
        assert "input" in result
        assert "output" in result
        assert "atoms" in result
        assert "contexts" in result

    def test_to_json_writes_file(
        self, mock_atomizer, mock_reviser, mock_retriever, mock_nli_extractor,
        mock_summarizer, sample_json_data
    ):
        pipeline = FactReasoner(
            atom_extractor=mock_atomizer,
            atom_reviser=mock_reviser,
            context_retriever=mock_retriever,
            context_summarizer=mock_summarizer,
            nli_extractor=mock_nli_extractor,
            merlin_path="/path/to/merlin"
        )

        pipeline.from_dict_with_contexts(sample_json_data)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            pipeline.to_json(json_file_path=temp_path)

            assert os.path.exists(temp_path)
            with open(temp_path, 'r') as f:
                saved_data = json.load(f)
            assert "atoms" in saved_data
            assert "contexts" in saved_data
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestFactReasonerFromFactGraph:
    """Tests for FactReasoner.from_fact_graph method."""

    def test_from_fact_graph_creates_atoms(
        self, mock_atomizer, mock_reviser, mock_retriever, mock_nli_extractor, mock_summarizer
    ):
        pipeline = FactReasoner(
            atom_extractor=mock_atomizer,
            atom_reviser=mock_reviser,
            context_retriever=mock_retriever,
            context_summarizer=mock_summarizer,
            nli_extractor=mock_nli_extractor,
            merlin_path="/path/to/merlin"
        )

        # Create a simple FactGraph
        atom = Atom(id="a0", text="Test atom")
        context = Context(id="c0", atom=atom, text="Test context")
        relation = Relation(
            source=context,
            target=atom,
            type="entailment",
            probability=0.9,
            link="context_atom"
        )
        fact_graph = FactGraph(atoms=[atom], contexts=[context], relations=[relation])

        pipeline.from_fact_graph(fact_graph)

        assert "a0" in pipeline.atoms
        assert "c0" in pipeline.contexts
        assert len(pipeline.relations) == 1


class TestFactReasonerBuildMarkovNetwork:
    """Tests for FactReasoner Markov Network building."""

    def test_build_markov_network_creates_nodes(
        self, mock_atomizer, mock_reviser, mock_retriever, mock_nli_extractor,
        mock_summarizer, sample_json_data
    ):
        pipeline = FactReasoner(
            atom_extractor=mock_atomizer,
            atom_reviser=mock_reviser,
            context_retriever=mock_retriever,
            context_summarizer=mock_summarizer,
            nli_extractor=mock_nli_extractor,
            merlin_path="/path/to/merlin"
        )

        # Create a simple setup
        atom = Atom(id="a0", text="Test atom")
        context = Context(id="c0", atom=atom, text="Test context")
        relation = Relation(
            source=context,
            target=atom,
            type="entailment",
            probability=0.9,
            link="context_atom"
        )
        fact_graph = FactGraph(atoms=[atom], contexts=[context], relations=[relation])

        pipeline.from_fact_graph(fact_graph)

        assert pipeline.markov_network is not None
        assert "a0" in pipeline.markov_network.nodes
        assert "c0" in pipeline.markov_network.nodes


class TestFactReasonerBuildFactGraph:
    """Tests for FactReasoner._build_fact_graph method."""

    def test_build_fact_graph_from_atoms_contexts_relations(
        self, mock_atomizer, mock_reviser, mock_retriever, mock_nli_extractor, mock_summarizer
    ):
        pipeline = FactReasoner(
            atom_extractor=mock_atomizer,
            atom_reviser=mock_reviser,
            context_retriever=mock_retriever,
            context_summarizer=mock_summarizer,
            nli_extractor=mock_nli_extractor,
            merlin_path="/path/to/merlin"
        )

        # Manually set up atoms, contexts, relations
        atom = Atom(id="a0", text="Test atom")
        context = Context(id="c0", atom=atom, text="Test context")
        relation = Relation(
            source=context,
            target=atom,
            type="entailment",
            probability=0.9,
            link="context_atom"
        )

        pipeline.atoms = {"a0": atom}
        pipeline.contexts = {"c0": context}
        pipeline.relations = [relation]

        pipeline._build_fact_graph()

        assert pipeline.fact_graph is not None
        assert len(pipeline.fact_graph.get_nodes()) == 2
        assert len(pipeline.fact_graph.get_edges()) == 1
