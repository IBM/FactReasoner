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

"""Unit tests for fact_reasoner.baselines.factscore module."""

import pytest
from unittest.mock import MagicMock, patch
from fact_reasoner.baselines.factscore import (
    FactScore,
    INSTRUCTION_FACTSCORE,
    INSTRUCTION_FACTSCORE_NOTOPIC,
)


class TestFactScoreInit:
    """Tests for FactScore initialization."""

    def test_factscore_init(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        scorer = FactScore(backend=mock_backend)
        assert scorer.backend == mock_backend
        assert scorer.binary_output is True
        assert scorer.query is None
        assert scorer.response is None
        assert scorer.topic is None

    def test_factscore_init_with_components(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"
        mock_atomizer = MagicMock()
        mock_reviser = MagicMock()
        mock_retriever = MagicMock()

        scorer = FactScore(
            backend=mock_backend,
            atom_extractor=mock_atomizer,
            atom_reviser=mock_reviser,
            context_retriever=mock_retriever
        )

        assert scorer.atom_extractor == mock_atomizer
        assert scorer.atom_reviser == mock_reviser
        assert scorer.context_retriever == mock_retriever


class TestFactScoreInstructions:
    """Tests for FactScore instruction templates."""

    def test_instruction_with_topic_format(self):
        assert "{{topic_text}}" in INSTRUCTION_FACTSCORE
        assert "{{knowledge_text}}" in INSTRUCTION_FACTSCORE
        assert "{{atom_text}}" in INSTRUCTION_FACTSCORE
        assert "True or False" in INSTRUCTION_FACTSCORE

    def test_instruction_without_topic_format(self):
        assert "{{topic_text}}" not in INSTRUCTION_FACTSCORE_NOTOPIC
        assert "{{knowledge_text}}" in INSTRUCTION_FACTSCORE_NOTOPIC
        assert "{{atom_text}}" in INSTRUCTION_FACTSCORE_NOTOPIC


class TestFactScoreGetLabel:
    """Tests for FactScore._get_label method."""

    def test_get_label_true(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        scorer = FactScore(backend=mock_backend)

        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "True"

        result = scorer._get_label(mock_output)
        assert result == "S"

    def test_get_label_false(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        scorer = FactScore(backend=mock_backend)

        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "False"

        result = scorer._get_label(mock_output)
        assert result == "NS"

    def test_get_label_true_with_context(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        scorer = FactScore(backend=mock_backend)

        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "Based on the evidence, this is True."

        result = scorer._get_label(mock_output)
        assert result == "S"

    def test_get_label_false_with_context(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        scorer = FactScore(backend=mock_backend)

        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "The statement is False based on evidence."

        result = scorer._get_label(mock_output)
        assert result == "NS"

    def test_get_label_both_true_and_false(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        scorer = FactScore(backend=mock_backend)

        # When both appear, the one that appears later wins
        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "False is wrong, the answer is True"

        result = scorer._get_label(mock_output)
        assert result == "S"  # True appears after False

    def test_get_label_neither(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        scorer = FactScore(backend=mock_backend)

        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "The answer is yes"

        result = scorer._get_label(mock_output)
        # Falls back to heuristic - "not", "cannot", etc. not present
        assert result == "S"

    def test_get_label_with_not_keyword(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        scorer = FactScore(backend=mock_backend)

        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "This is not supported"

        result = scorer._get_label(mock_output)
        assert result == "NS"


class TestFactScoreFromDict:
    """Tests for FactScore.from_dict_with_contexts method."""

    def test_from_dict_loads_data(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        scorer = FactScore(backend=mock_backend)

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
        assert "a0" in scorer.atoms
        assert "c0" in scorer.contexts


class TestFactScoreToJson:
    """Tests for FactScore.to_json method."""

    def test_to_json_returns_dict(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        scorer = FactScore(backend=mock_backend)
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
        assert "atoms" in result
        assert "contexts" in result
