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

"""Unit tests for fact_reasoner.core.atomizer module."""

from typing import Any
import pytest
from unittest.mock import MagicMock, patch
from src.fact_reasoner.core.atomizer import Atomizer, INSTRUCTION_ATOMIZER


class TestAtomizerInit:
    """Tests for Atomizer initialization."""

    def test_atomizer_none_backend_raises(self):
        with pytest.raises(ValueError, match="Mellea backend is None"):
            Atomizer(backend=None)

    def test_atomizer_str(self):
        # Create a mock backend
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        atm = Atomizer(backend=mock_backend)
        assert str(atm) == "This is the atomizer"

    def test_atomizer_stores_backend(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        atm = Atomizer(backend=mock_backend)
        assert atm.backend == mock_backend


class TestAtomizerInstruction:
    """Tests for Atomizer instruction template."""

    def test_instruction_contains_examples(self):
        assert "Example 1:" in INSTRUCTION_ATOMIZER
        assert "Example 2:" in INSTRUCTION_ATOMIZER
        assert "Example 3:" in INSTRUCTION_ATOMIZER
        assert "Example 4:" in INSTRUCTION_ATOMIZER

    def test_instruction_contains_rules(self):
        assert "Rules:" in INSTRUCTION_ATOMIZER
        assert "atomic unit" in INSTRUCTION_ATOMIZER.lower()

    def test_instruction_contains_output_format(self):
        assert "```json" in INSTRUCTION_ATOMIZER
        assert "id1" in INSTRUCTION_ATOMIZER

    def test_instruction_contains_placeholder(self):
        assert "{{response}}" in INSTRUCTION_ATOMIZER


class TestAtomizerRun:
    """Tests for Atomizer.run method."""

    def test_run_returns_dict_on_success(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        mock_output = MagicMock()
        mock_output.success = True
        mock_output.__str__ = lambda self: '```json\n{"id1": "Test atom"}\n```'

        with patch('src.fact_reasoner.core.atomizer.mfuncs.instruct', return_value=mock_output):
            atm = Atomizer(backend=mock_backend)
            result = atm.run("Test response text")

            assert isinstance(result, dict)
            assert "id1" in result
            assert result["id1"] == "Test atom"

    def test_run_returns_empty_dict_on_failure(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        mock_output = MagicMock()
        mock_output.success = False

        with patch('src.fact_reasoner.core.atomizer.mfuncs.instruct', return_value=mock_output):
            atm = Atomizer(backend=mock_backend)
            result = atm.run("Test response text")

            assert result == {}

    def test_run_handles_multiple_atoms(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        mock_output = MagicMock()
        mock_output.success = True
        mock_output.__str__ = lambda self: '''```json
{
    "id1": "First atom",
    "id2": "Second atom",
    "id3": "Third atom"
}
```'''

        with patch('src.fact_reasoner.core.atomizer.mfuncs.instruct', return_value=mock_output):
            atm = Atomizer(backend=mock_backend)
            result = atm.run("Test response with multiple facts")

            assert len(result) == 3
            assert result["id1"] == "First atom"
            assert result["id2"] == "Second atom"
            assert result["id3"] == "Third atom"
