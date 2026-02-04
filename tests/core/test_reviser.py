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

"""Unit tests for fact_reasoner.core.reviser module."""

import pytest
from unittest.mock import MagicMock, patch
from src.fact_reasoner.core.reviser import Reviser, INSTRUCTION_REVISER


class TestReviserInit:
    """Tests for Reviser initialization."""

    def test_reviser_none_backend_raises(self):
        with pytest.raises(ValueError, match="Mellea session is None"):
            Reviser(backend=None)

    def test_reviser_stores_backend(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        reviser = Reviser(backend=mock_backend)
        assert reviser.backend == mock_backend


class TestReviserInstruction:
    """Tests for Reviser instruction template."""

    def test_instruction_contains_examples(self):
        assert "Example 1:" in INSTRUCTION_REVISER
        assert "Example 2:" in INSTRUCTION_REVISER
        assert "Example 3:" in INSTRUCTION_REVISER

    def test_instruction_mentions_vague_references(self):
        assert "Vague References:" in INSTRUCTION_REVISER
        assert "Pronouns" in INSTRUCTION_REVISER
        assert "Demonstrative pronouns" in INSTRUCTION_REVISER

    def test_instruction_contains_output_format(self):
        assert "```json" in INSTRUCTION_REVISER
        assert "revised_unit" in INSTRUCTION_REVISER
        assert "rationale" in INSTRUCTION_REVISER

    def test_instruction_contains_placeholders(self):
        assert "{{atomic_unit}}" in INSTRUCTION_REVISER
        assert "{{response}}" in INSTRUCTION_REVISER


class TestReviserRun:
    """Tests for Reviser.run method."""

    def test_run_returns_list(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        mock_output = MagicMock()
        mock_output.success = True
        mock_output.__str__ = lambda self: '''```json
{
    "revised_unit": "Albert Einstein was German-born.",
    "rationale": "No changes needed."
}
```'''

        with patch('src.fact_reasoner.core.reviser.mfuncs.instruct', return_value=mock_output):
            reviser = Reviser(backend=mock_backend)
            result = reviser.run(
                units=["Einstein was German-born."],
                response="Albert Einstein was a German-born physicist."
            )

            assert isinstance(result, list)
            assert len(result) == 1
            assert "revised_unit" in result[0]
            assert result[0]["revised_unit"] == "Albert Einstein was German-born."

    def test_run_multiple_units(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        mock_output = MagicMock()
        mock_output.success = True
        mock_output.__str__ = lambda self: '''```json
{
    "revised_unit": "Revised atom",
    "rationale": "Resolved reference."
}
```'''

        with patch('src.fact_reasoner.core.reviser.mfuncs.instruct', return_value=mock_output):
            reviser = Reviser(backend=mock_backend)
            result = reviser.run(
                units=["He was born in 1879.", "She won the prize."],
                response="Albert Einstein was born in 1879. Marie Curie won the prize."
            )

            assert len(result) == 2

    def test_run_handles_failure(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        mock_output = MagicMock()
        mock_output.success = False

        with patch('src.fact_reasoner.core.reviser.mfuncs.instruct', return_value=mock_output):
            reviser = Reviser(backend=mock_backend)
            result = reviser.run(
                units=["Test unit"],
                response="Test response"
            )

            # When all fail, result should be empty
            assert result == []

    def test_run_includes_original_text(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        mock_output = MagicMock()
        mock_output.success = True
        mock_output.__str__ = lambda self: '''```json
{
    "revised_unit": "Revised version",
    "rationale": "Changed pronoun."
}
```'''

        with patch('src.fact_reasoner.core.reviser.mfuncs.instruct', return_value=mock_output):
            reviser = Reviser(backend=mock_backend)
            result = reviser.run(
                units=["Original text"],
                response="Full response"
            )

            assert len(result) == 1
            assert "text" in result[0]
