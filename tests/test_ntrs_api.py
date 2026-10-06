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

"""Unit tests for fact_reasoner.ntrs_api module."""

from unittest.mock import patch, MagicMock

from fact_reasoner.ntrs_api import NTRSAPI, NTRS_SEARCH_URL


class TestNTRSAPIInit:
    """Tests for NTRSAPI initialization."""

    def test_init(self):
        api = NTRSAPI()
        assert api.url == NTRS_SEARCH_URL


class TestNTRSAPIGetSnippets:
    """Tests for NTRSAPI.get_snippets method."""

    def test_get_snippets_format(self):
        api = NTRSAPI()

        mock_response = {
            "results": [
                {
                    "id": 20200000325,
                    "title": "Test Title",
                    "abstract": "Test abstract text",
                    "downloads": [
                        {"links": {"pdf": "/api/citations/20200000325/downloads/20200000325.pdf"}}
                    ],
                }
            ]
        }

        with patch.object(api, "get_search_res", return_value=mock_response):
            results = api.get_snippets(["test query"])

            assert "test query" in results
            assert len(results["test query"]) == 1
            assert results["test query"][0]["title"] == "Test Title"
            assert results["test query"][0]["snippet"] == "Test abstract text"
            assert results["test query"][0]["link"] == (
                "https://ntrs.nasa.gov/api/citations/20200000325/downloads/20200000325.pdf"
            )

    def test_get_snippets_skips_missing_abstract(self):
        api = NTRSAPI()

        mock_response = {
            "results": [
                {"id": 1, "title": "Has Abstract", "abstract": "Some text", "downloads": []},
                {"id": 2, "title": "No Abstract", "abstract": None, "downloads": []},
                {"id": 3, "title": "Empty Abstract", "abstract": "", "downloads": []},
            ]
        }

        with patch.object(api, "get_search_res", return_value=mock_response):
            results = api.get_snippets(["test query"])

            assert len(results["test query"]) == 1
            assert results["test query"][0]["title"] == "Has Abstract"

    def test_get_snippets_link_falls_back_to_citation_page(self):
        api = NTRSAPI()

        mock_response = {
            "results": [
                {"id": 12345, "title": "No PDF", "abstract": "Some text", "downloads": []}
            ]
        }

        with patch.object(api, "get_search_res", return_value=mock_response):
            results = api.get_snippets(["test query"])

            assert results["test query"][0]["link"] == "https://ntrs.nasa.gov/citations/12345"

    def test_get_snippets_empty_results(self):
        api = NTRSAPI()

        with patch.object(api, "get_search_res", return_value={"results": []}):
            results = api.get_snippets(["test query"])

            assert "test query" in results
            assert len(results["test query"]) == 0

    def test_get_snippets_multiple_queries(self):
        api = NTRSAPI()

        def mock_search(query, page_size=10):
            return {
                "results": [
                    {"id": 1, "title": f"Result for {query}", "abstract": "abstract", "downloads": []}
                ]
            }

        with patch.object(api, "get_search_res", side_effect=mock_search):
            results = api.get_snippets(["query1", "query2"])

            assert "query1" in results
            assert "query2" in results
            assert results["query1"][0]["title"] == "Result for query1"
            assert results["query2"][0]["title"] == "Result for query2"

    def test_get_snippets_handles_missing_title(self):
        api = NTRSAPI()

        mock_response = {
            "results": [{"id": 1, "abstract": "abstract only, no title", "downloads": []}]
        }

        with patch.object(api, "get_search_res", return_value=mock_response):
            results = api.get_snippets(["test query"])

            assert results["test query"][0]["title"] == ""


class TestNTRSAPIGetSearchRes:
    """Tests for NTRSAPI.get_search_res method."""

    def test_sends_expected_payload(self):
        api = NTRSAPI()

        with patch("fact_reasoner.ntrs_api.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.json.return_value = {"results": []}
            mock_post.return_value = mock_resp

            api.get_search_res("mars rover", page_size=25)

            mock_post.assert_called_once_with(
                NTRS_SEARCH_URL, json={"q": "mars rover", "page": {"size": 25}}
            )
            mock_resp.raise_for_status.assert_called_once()

    def test_raises_on_http_error(self):
        api = NTRSAPI()

        with patch("fact_reasoner.ntrs_api.requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = Exception("HTTP error")
            mock_post.return_value = mock_resp

            try:
                api.get_search_res("mars rover")
                assert False, "Expected an exception to be raised"
            except Exception as e:
                assert str(e) == "HTTP error"


class TestNTRSAPIPageSizeClamping:
    """Tests for the page_size computation in get_snippets."""

    def test_small_top_k_uses_minimum_page_size(self):
        api = NTRSAPI()

        with patch.object(api, "get_search_res", return_value={"results": []}) as mock_get:
            api.get_snippets(["q"], top_k=1)
            mock_get.assert_called_once_with("q", page_size=10)

    def test_large_top_k_is_capped_at_100(self):
        api = NTRSAPI()

        with patch.object(api, "get_search_res", return_value={"results": []}) as mock_get:
            api.get_snippets(["q"], top_k=500)
            mock_get.assert_called_once_with("q", page_size=100)

    def test_top_k_above_minimum_is_used_directly(self):
        api = NTRSAPI()

        with patch.object(api, "get_search_res", return_value={"results": []}) as mock_get:
            api.get_snippets(["q"], top_k=42)
            mock_get.assert_called_once_with("q", page_size=42)
