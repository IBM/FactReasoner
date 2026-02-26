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

"""Shared pytest fixtures for fact_reasoner tests."""

import pytest
from unittest.mock import MagicMock

from fact_reasoner.core.utils import Atom, Context, Relation


@pytest.fixture
def mock_backend():
    """Create a mock Mellea backend for testing."""
    backend = MagicMock()
    backend.model_id = "test-model-id"
    return backend


@pytest.fixture
def sample_atom():
    """Create a sample Atom for testing."""
    return Atom(id="a0", text="Albert Einstein was born in 1879.", label="S")


@pytest.fixture
def sample_context(sample_atom):
    """Create a sample Context for testing."""
    return Context(
        id="c0",
        atom=sample_atom,
        text="Albert Einstein was born on March 14, 1879, in Ulm, Germany.",
        title="Albert Einstein - Wikipedia",
        link="https://en.wikipedia.org/wiki/Albert_Einstein",
        snippet="German-born theoretical physicist"
    )


@pytest.fixture
def sample_relation(sample_context, sample_atom):
    """Create a sample Relation for testing."""
    return Relation(
        source=sample_context,
        target=sample_atom,
        type="entailment",
        probability=0.9,
        link="context_atom"
    )


@pytest.fixture
def sample_atoms_dict():
    """Create a sample dictionary of Atoms for testing."""
    return {
        "a0": Atom(id="a0", text="Einstein was German-born.", label="S"),
        "a1": Atom(id="a1", text="Einstein won the Nobel Prize.", label="S"),
        "a2": Atom(id="a2", text="Einstein invented the iPhone.", label="NS"),
    }


@pytest.fixture
def sample_contexts_dict(sample_atoms_dict):
    """Create a sample dictionary of Contexts for testing."""
    contexts = {
        "c_a0_0": Context(
            id="c_a0_0",
            atom=sample_atoms_dict["a0"],
            text="Einstein was born in Ulm, Germany.",
            title="Einstein Bio"
        ),
        "c_a0_1": Context(
            id="c_a0_1",
            atom=sample_atoms_dict["a0"],
            text="Einstein was a German physicist.",
            title="Einstein Wikipedia"
        ),
        "c_a1_0": Context(
            id="c_a1_0",
            atom=sample_atoms_dict["a1"],
            text="Einstein won the Nobel Prize in 1921.",
            title="Nobel Prize"
        ),
    }

    # Link contexts to atoms
    sample_atoms_dict["a0"].add_contexts([contexts["c_a0_0"], contexts["c_a0_1"]])
    sample_atoms_dict["a1"].add_contexts([contexts["c_a1_0"]])

    return contexts


@pytest.fixture
def sample_json_data():
    """Create sample JSON data for loading into assessors."""
    return {
        "input": "Tell me a biography of Albert Einstein.",
        "output": "Albert Einstein was a German-born theoretical physicist. He won the Nobel Prize in 1921.",
        "topic": "Albert Einstein",
        "atoms": [
            {
                "id": "a0",
                "text": "Albert Einstein was German-born.",
                "original": "Albert Einstein was a German-born theoretical physicist.",
                "label": "S",
                "contexts": ["c_a0_0"]
            },
            {
                "id": "a1",
                "text": "Albert Einstein won the Nobel Prize in 1921.",
                "original": "He won the Nobel Prize in 1921.",
                "label": "S",
                "contexts": ["c_a1_0"]
            }
        ],
        "contexts": [
            {
                "id": "c_a0_0",
                "title": "Albert Einstein - Wikipedia",
                "text": "Albert Einstein was born in Ulm, Germany, on March 14, 1879.",
                "snippet": "German-born theoretical physicist",
                "link": "https://en.wikipedia.org/wiki/Albert_Einstein"
            },
            {
                "id": "c_a1_0",
                "title": "Nobel Prize in Physics 1921",
                "text": "The Nobel Prize in Physics 1921 was awarded to Albert Einstein.",
                "snippet": "Nobel Prize winner",
                "link": "https://nobelprize.org/einstein"
            }
        ]
    }


@pytest.fixture
def mock_atomizer(mock_backend):
    """Create a mock Atomizer for testing."""
    from src.fact_reasoner.core.atomizer import Atomizer

    atomizer = MagicMock(spec=Atomizer)
    atomizer.backend = mock_backend
    atomizer.run.return_value = {
        "id1": "First atomic unit.",
        "id2": "Second atomic unit."
    }
    return atomizer


@pytest.fixture
def mock_reviser(mock_backend):
    """Create a mock Reviser for testing."""
    from src.fact_reasoner.core.reviser import Reviser

    reviser = MagicMock(spec=Reviser)
    reviser.backend = mock_backend
    reviser.run.return_value = [
        {"revised_unit": "Revised first unit.", "rationale": "No changes."},
        {"revised_unit": "Revised second unit.", "rationale": "Resolved pronoun."}
    ]
    return reviser


@pytest.fixture
def mock_retriever():
    """Create a mock ContextRetriever for testing."""
    from src.fact_reasoner.core.retriever import ContextRetriever

    retriever = MagicMock(spec=ContextRetriever)
    retriever.service_type = "wikipedia"
    retriever.top_k = 3
    retriever.query.return_value = [
        {
            "title": "Test Document",
            "text": "This is test content for retrieval.",
            "snippet": "Test snippet",
            "link": "https://example.com"
        }
    ]
    return retriever


@pytest.fixture
def mock_nli_extractor(mock_backend):
    """Create a mock NLIExtractor for testing."""
    from src.fact_reasoner.core.nli import NLIExtractor

    nli = MagicMock(spec=NLIExtractor)
    nli.backend = mock_backend
    nli.run.return_value = {"label": "entailment", "probability": 0.9}
    return nli


@pytest.fixture
def mock_summarizer(mock_backend):
    """Create a mock ContextSummarizer for testing."""
    from src.fact_reasoner.core.summarizer import ContextSummarizer

    summarizer = MagicMock(spec=ContextSummarizer)
    summarizer.backend = mock_backend
    return summarizer
