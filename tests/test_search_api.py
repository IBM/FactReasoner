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

"""Unit tests for fact_reasoner.search_api module."""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.fact_reasoner.search_api import SearchAPI


class TestSearchAPIInit:
    """Tests for SearchAPI initialization."""

    def test_init_without_cache(self):
        with patch.dict(os.environ, {"SERPER_API_KEY": "test_key"}):
            api = SearchAPI(cache_dir=None)
            assert api.do_caching is False
            assert api.serper_key == "test_key"

    def test_init_with_cache(self):
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name

        try:
            with patch.dict(os.environ, {"SERPER_API_KEY": "test_key"}):
                api = SearchAPI(cache_dir=temp_path)
                assert api.do_caching is True
                assert api.cache_dir == temp_path
                assert api.similarity_threshold == 90
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_init_custom_similarity_threshold(self):
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name

        try:
            with patch.dict(os.environ, {"SERPER_API_KEY": "test_key"}):
                api = SearchAPI(cache_dir=temp_path, similarity_threshold=80)
                assert api.similarity_threshold == 80
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestSearchAPIGetSnippets:
    """Tests for SearchAPI.get_snippets method."""

    def test_get_snippets_format(self):
        with patch.dict(os.environ, {"SERPER_API_KEY": "test_key"}):
            api = SearchAPI(cache_dir=None)

            mock_response = {
                "organic": [
                    {
                        "title": "Test Title",
                        "snippet": "Test snippet text",
                        "link": "https://example.com"
                    }
                ]
            }

            with patch.object(api, 'get_search_res', return_value=mock_response):
                results = api.get_snippets(["test query"])

                assert "test query" in results
                assert len(results["test query"]) == 1
                assert results["test query"][0]["title"] == "Test Title"
                assert results["test query"][0]["snippet"] == "Test snippet text"
                assert results["test query"][0]["link"] == "https://example.com"

    def test_get_snippets_empty_results(self):
        with patch.dict(os.environ, {"SERPER_API_KEY": "test_key"}):
            api = SearchAPI(cache_dir=None)

            mock_response = {"organic": []}

            with patch.object(api, 'get_search_res', return_value=mock_response):
                results = api.get_snippets(["test query"])

                assert "test query" in results
                assert len(results["test query"]) == 0

    def test_get_snippets_multiple_queries(self):
        with patch.dict(os.environ, {"SERPER_API_KEY": "test_key"}):
            api = SearchAPI(cache_dir=None)

            def mock_search(query):
                return {
                    "organic": [
                        {"title": f"Result for {query}", "snippet": "", "link": ""}
                    ]
                }

            with patch.object(api, 'get_search_res', side_effect=mock_search):
                results = api.get_snippets(["query1", "query2"])

                assert "query1" in results
                assert "query2" in results
                assert results["query1"][0]["title"] == "Result for query1"
                assert results["query2"][0]["title"] == "Result for query2"


class TestSearchAPICaching:
    """Tests for SearchAPI caching functionality."""

    def test_save_to_cache(self):
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name

        try:
            with patch.dict(os.environ, {"SERPER_API_KEY": "test_key"}):
                api = SearchAPI(cache_dir=temp_path)

                response = {
                    "searchParameters": {"q": "test"},
                    "organic": [
                        {"title": "Test", "snippet": "Snippet", "link": "http://test.com"}
                    ]
                }

                # Save to cache
                api._save_to_cache("test query", response)

                # Verify it was saved (no exception raised)
                assert True
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_save_empty_results_not_cached(self):
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name

        try:
            with patch.dict(os.environ, {"SERPER_API_KEY": "test_key"}):
                api = SearchAPI(cache_dir=temp_path)

                response = {"organic": []}

                # Save empty results (should not actually save)
                api._save_to_cache("test query", response)

                # Try to retrieve - should be None
                result = api._get_from_cache("test query")
                assert result is None
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestSearchAPIHelpers:
    """Tests for SearchAPI helper methods."""

    def test_get_snippets_handles_missing_fields(self):
        with patch.dict(os.environ, {"SERPER_API_KEY": "test_key"}):
            api = SearchAPI(cache_dir=None)

            mock_response = {
                "organic": [
                    {"title": "Title Only"},  # Missing snippet and link
                ]
            }

            with patch.object(api, 'get_search_res', return_value=mock_response):
                results = api.get_snippets(["test query"])

                assert results["test query"][0]["title"] == "Title Only"
                assert results["test query"][0]["snippet"] == ""
                assert results["test query"][0]["link"] == ""
