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

"""Unit tests for fact_reasoner.core.query_builder module."""

import pytest
from unittest.mock import MagicMock, patch
from fact_reasoner.core.query_builder import QueryBuilder, INSTRUCTION_QUERY_BUILDER


class TestQueryBuilderInit:
    """Tests for QueryBuilder initialization."""

    def test_query_builder_none_backend_raises(self):
        with pytest.raises(ValueError, match="Mellea backend is None"):
            QueryBuilder(backend=None)

    def test_query_builder_stores_backend(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        qb = QueryBuilder(backend=mock_backend)
        assert qb.backend == mock_backend


class TestQueryBuilderInstruction:
    """Tests for QueryBuilder instruction template."""

    def test_instruction_contains_examples(self):
        assert "Example 1:" in INSTRUCTION_QUERY_BUILDER
        assert "Example 2:" in INSTRUCTION_QUERY_BUILDER
        assert "Example 3:" in INSTRUCTION_QUERY_BUILDER

    def test_instruction_contains_criteria(self):
        assert "QUERY CONSTRUCTION CRITERIA:" in INSTRUCTION_QUERY_BUILDER
        assert "factual accuracy" in INSTRUCTION_QUERY_BUILDER

    def test_instruction_contains_output_format(self):
        assert "```" in INSTRUCTION_QUERY_BUILDER
        assert "<your generated query here>" in INSTRUCTION_QUERY_BUILDER

    def test_instruction_contains_placeholder(self):
        assert "{{statement_text}}" in INSTRUCTION_QUERY_BUILDER

    def test_instruction_mentions_operators(self):
        assert "quotation marks" in INSTRUCTION_QUERY_BUILDER
        assert "Boolean operators" in INSTRUCTION_QUERY_BUILDER


class TestQueryBuilderRun:
    """Tests for QueryBuilder.run method."""

    def test_run_returns_cleaned_query_on_success(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        mock_output = MagicMock()
        mock_output.success = True
        mock_output.__str__ = lambda self: '```\n"Einstein" theory of relativity fact check\n```'

        with patch('src.fact_reasoner.core.query_builder.mfuncs.instruct', return_value=mock_output):
            qb = QueryBuilder(backend=mock_backend)
            result = qb.run("Einstein developed the theory of relativity.")

            assert isinstance(result, str)
            assert "Einstein" in result
            assert "```" not in result  # Should be stripped

    def test_run_returns_original_on_failure(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        mock_output = MagicMock()
        mock_output.success = False

        with patch('src.fact_reasoner.core.query_builder.mfuncs.instruct', return_value=mock_output):
            qb = QueryBuilder(backend=mock_backend)
            original_text = "Original statement text"
            result = qb.run(original_text)

            # On failure, should return the original text
            assert result == original_text

    def test_run_strips_markdown_fences(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        mock_output = MagicMock()
        mock_output.success = True
        mock_output.__str__ = lambda self: '```\ntest query\n```'

        with patch('src.fact_reasoner.core.query_builder.mfuncs.instruct', return_value=mock_output):
            qb = QueryBuilder(backend=mock_backend)
            result = qb.run("Test statement")

            assert "```" not in result
            assert result.strip() == "test query"

    def test_run_handles_complex_query(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        mock_output = MagicMock()
        mock_output.success = True
        mock_output.__str__ = lambda self: '''```
"Apple foldable iPhone 2026" rumor OR announcement site:apple.com
```'''

        with patch('src.fact_reasoner.core.query_builder.mfuncs.instruct', return_value=mock_output):
            qb = QueryBuilder(backend=mock_backend)
            result = qb.run("Apple will release a foldable iPhone in 2026")

            assert "Apple" in result
            assert "foldable" in result
            assert "OR" in result
