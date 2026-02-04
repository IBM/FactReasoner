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

"""Unit tests for fact_reasoner.baselines.veriscore module."""

import pytest
from unittest.mock import MagicMock, patch
from fact_reasoner.baselines.veriscore import VeriScore, INSTRUCTION_VERISCORE


class TestVeriScoreInit:
    """Tests for VeriScore initialization."""

    def test_veriscore_init(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        scorer = VeriScore(backend=mock_backend)
        assert scorer.backend == mock_backend
        assert scorer.binary_output is False  # VeriScore defaults to ternary
        assert scorer.query is None
        assert scorer.response is None
        assert scorer.topic is None

    def test_veriscore_init_with_components(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"
        mock_atomizer = MagicMock()
        mock_reviser = MagicMock()
        mock_retriever = MagicMock()

        scorer = VeriScore(
            backend=mock_backend,
            atom_extractor=mock_atomizer,
            atom_reviser=mock_reviser,
            context_retriever=mock_retriever
        )

        assert scorer.atom_extractor == mock_atomizer
        assert scorer.atom_reviser == mock_reviser
        assert scorer.context_retriever == mock_retriever


class TestVeriScoreInstruction:
    """Tests for VeriScore instruction template."""

    def test_instruction_contains_labels(self):
        assert "[Supported]" in INSTRUCTION_VERISCORE
        assert "[Contradicted]" in INSTRUCTION_VERISCORE
        assert "[Unverifiable]" in INSTRUCTION_VERISCORE

    def test_instruction_contains_placeholders(self):
        assert "{{knowledge_text}}" in INSTRUCTION_VERISCORE
        assert "{{atom_text}}" in INSTRUCTION_VERISCORE

    def test_instruction_contains_steps(self):
        assert "1. Summarize KNOWLEDGE Points" in INSTRUCTION_VERISCORE
        assert "2. Evaluate Evidence" in INSTRUCTION_VERISCORE
        assert "3. Restate the STATEMENT" in INSTRUCTION_VERISCORE
        assert "4. Final Answer" in INSTRUCTION_VERISCORE


class TestVeriScoreGetLabel:
    """Tests for VeriScore._get_label method."""

    def test_get_label_supported_binary(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        scorer = VeriScore(backend=mock_backend)
        scorer.binary_output = True

        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "The answer is [Supported]"

        result = scorer._get_label(mock_output)
        assert result == "S"

    def test_get_label_contradicted_binary(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        scorer = VeriScore(backend=mock_backend)
        scorer.binary_output = True

        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "[Contradicted]"

        result = scorer._get_label(mock_output)
        assert result == "NS"

    def test_get_label_unverifiable_binary(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        scorer = VeriScore(backend=mock_backend)
        scorer.binary_output = True

        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "[Unverifiable]"

        result = scorer._get_label(mock_output)
        assert result == "NS"

    def test_get_label_supported_ternary(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        scorer = VeriScore(backend=mock_backend)
        scorer.binary_output = False

        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "[Supported]"

        result = scorer._get_label(mock_output)
        assert result == "S"

    def test_get_label_contradicted_ternary(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        scorer = VeriScore(backend=mock_backend)
        scorer.binary_output = False

        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "[Contradicted]"

        result = scorer._get_label(mock_output)
        assert result == "C"

    def test_get_label_unverifiable_ternary(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        scorer = VeriScore(backend=mock_backend)
        scorer.binary_output = False

        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "[Unverifiable]"

        result = scorer._get_label(mock_output)
        assert result == "U"

    def test_get_label_empty_brackets(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        scorer = VeriScore(backend=mock_backend)
        scorer.binary_output = False

        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "No brackets here"

        result = scorer._get_label(mock_output)
        assert result == "U"  # Default for ternary when empty


class TestVeriScoreFromDict:
    """Tests for VeriScore.from_dict_with_contexts method."""

    def test_from_dict_loads_data(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        scorer = VeriScore(backend=mock_backend)

        data = {
            "input": "Tell me about Einstein",
            "output": "Einstein was a physicist.",
            "topic": "Einstein",
            "atoms": [
                {
                    "id": "a0",
                    "text": "Einstein was a physicist.",
                    "original": "Einstein was a physicist.",
                    "label": "S",
                    "contexts": ["c0"]
                }
            ],
            "contexts": [
                {
                    "id": "c0",
                    "title": "Einstein Wikipedia",
                    "text": "Albert Einstein was a theoretical physicist.",
                    "snippet": "German physicist",
                    "link": "https://example.com"
                }
            ]
        }

        scorer.from_dict_with_contexts(data)

        assert scorer.query == "Tell me about Einstein"
        assert scorer.response == "Einstein was a physicist."
        assert scorer.topic == "Einstein"
        assert len(scorer.atoms) == 1
        assert len(scorer.contexts) == 1

    def test_from_dict_links_atoms_contexts(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        scorer = VeriScore(backend=mock_backend)

        data = {
            "input": "Query",
            "output": "Response",
            "atoms": [
                {
                    "id": "a0",
                    "text": "Atom text",
                    "original": "Original",
                    "contexts": ["c0", "c1"]
                }
            ],
            "contexts": [
                {"id": "c0", "title": "Title 0", "text": "Text 0"},
                {"id": "c1", "title": "Title 1", "text": "Text 1"},
            ]
        }

        scorer.from_dict_with_contexts(data)

        atom = scorer.atoms["a0"]
        contexts = atom.get_contexts()
        assert len(contexts) == 2
        assert "c0" in contexts
        assert "c1" in contexts


class TestVeriScoreToJson:
    """Tests for VeriScore.to_json method."""

    def test_to_json_returns_dict(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        scorer = VeriScore(backend=mock_backend)
        scorer.query = "Test query"
        scorer.response = "Test response"
        scorer.topic = "Test topic"
        scorer.atoms = {}
        scorer.contexts = {}

        result = scorer.to_json()

        assert isinstance(result, dict)
        assert result["input"] == "Test query"
        assert result["output"] == "Test response"
        assert result["topic"] == "Test topic"
