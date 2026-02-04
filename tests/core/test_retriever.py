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

"""Unit tests for fact_reasoner.core.retriever module."""

import pytest
from unittest.mock import patch, MagicMock

from src.fact_reasoner.core.retriever import (
    _clean_text,
    make_uniform,
    get_title,
    is_content_valid,
    CITATION_PATTERNS,
)


class TestCleanText:
    """Tests for _clean_text function."""

    def test_removes_bracket_citations(self):
        text = "Einstein developed relativity [1] in 1905 [23]."
        result = _clean_text(text)
        assert "[1]" not in result
        assert "[23]" not in result
        assert "Einstein" in result

    def test_removes_parenthesis_citations(self):
        text = "The theory was developed (1) by Einstein (23)."
        result = _clean_text(text)
        assert "(1)" not in result
        assert "(23)" not in result

    def test_removes_citation_needed(self):
        text = "This fact [citation needed] is important."
        result = _clean_text(text)
        assert "citation needed" not in result

    def test_removes_author_year_citations(self):
        text = "According to research [Smith 2020], this is true."
        result = _clean_text(text)
        assert "[Smith 2020]" not in result

    def test_collapses_whitespace(self):
        text = "Hello    world\n\ntest"
        result = _clean_text(text)
        assert "  " not in result
        assert "\n" not in result

    def test_strips_leading_trailing(self):
        text = "   hello world   "
        result = _clean_text(text)
        assert result == "hello world"


class TestMakeUniform:
    """Tests for make_uniform function."""

    def test_basic_text(self):
        text = "This is a test paragraph."
        result = make_uniform(text)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_long_text_split(self):
        # Create a long text
        text = "Word " * 500
        result = make_uniform(text)
        assert isinstance(result, str)


class TestGetTitle:
    """Tests for get_title function."""

    def test_extracts_title(self):
        text = "Title Here\nRest of the content"
        assert get_title(text) == "Title Here"

    def test_no_newline(self):
        text = "Single line text"
        # When no newline, find returns -1, so slice is text[:-1]
        result = get_title(text)
        assert result == "Single line tex"

    def test_empty_title(self):
        text = "\nContent after empty title"
        assert get_title(text) == ""


class TestIsContentValid:
    """Tests for is_content_valid function."""

    def test_valid_content(self):
        text = "Albert Einstein was a theoretical physicist who developed the theory of relativity."
        assert is_content_valid("http://example.com", text) is True

    def test_empty_content(self):
        assert is_content_valid("http://example.com", "") is False

    def test_none_content(self):
        assert is_content_valid("http://example.com", None) is False

    def test_invalid_non_string(self):
        assert is_content_valid("http://example.com", 123) is False

    def test_cookie_notice(self):
        text = "Cookies are used by this site for analytics."
        assert is_content_valid("http://example.com", text) is False

    def test_copyright_notice(self):
        text = "Copyright © 2024 All rights reserved."
        assert is_content_valid("http://example.com", text) is False

    def test_access_denied(self):
        text = "Access denied. You do not have permission."
        assert is_content_valid("http://example.com", text) is False

    def test_403_forbidden(self):
        text = "403 Forbidden - Access to this resource is denied."
        assert is_content_valid("http://example.com", text) is False

    def test_javascript_required(self):
        text = "You must have JavaScript enabled to view this page."
        assert is_content_valid("http://example.com", text) is False

    def test_captcha_verification(self):
        text = "To continue, please verify you are a human."
        assert is_content_valid("http://example.com", text) is False

    def test_high_replacement_char_ratio(self):
        # More than 10% replacement characters
        text = "Valid text " + "�" * 20
        assert is_content_valid("http://example.com", text) is False

    def test_low_replacement_char_ratio(self):
        # Less than 10% replacement characters
        text = "Valid text " * 50 + "�"
        assert is_content_valid("http://example.com", text) is True

    def test_short_valid_content(self):
        # Short content (< 50 chars) doesn't check replacement chars
        text = "Short valid content."
        assert is_content_valid("http://example.com", text) is True


class TestContextRetrieverInit:
    """Tests for ContextRetriever initialization."""

    def test_invalid_service_type(self):
        from src.fact_reasoner.core.retriever import ContextRetriever

        with pytest.raises(AssertionError):
            ContextRetriever(service_type="invalid_service")

    def test_wikipedia_service_type(self):
        from src.fact_reasoner.core.retriever import ContextRetriever

        retriever = ContextRetriever(service_type="wikipedia", top_k=3)
        assert retriever.service_type == "wikipedia"
        assert retriever.top_k == 3
        assert retriever.langchain_retriever is not None

    def test_google_service_type(self):
        from src.fact_reasoner.core.retriever import ContextRetriever
        import os

        with patch.dict(os.environ, {"SERPER_API_KEY": "test_key"}):
            retriever = ContextRetriever(
                service_type="google",
                top_k=5,
                cache_dir=None,
                fetch_text=True
            )
            assert retriever.service_type == "google"
            assert retriever.top_k == 5
            assert retriever.fetch_text is True
            assert retriever.google_retriever is not None

    def test_set_query_builder(self):
        from src.fact_reasoner.core.retriever import ContextRetriever

        retriever = ContextRetriever(service_type="wikipedia", top_k=3)
        mock_query_builder = MagicMock()
        retriever.set_query_builder(mock_query_builder)
        assert retriever.query_builder == mock_query_builder
