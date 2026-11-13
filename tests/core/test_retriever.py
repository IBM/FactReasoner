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

from fact_reasoner.core.retriever import (
    _clean_text,
    _domain_matches,
    make_uniform,
    get_title,
    is_content_valid,
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


class TestSourceRetrieverInit:
    """Tests for SourceRetriever initialization."""

    def test_invalid_service_type(self):
        from src.fact_reasoner.core.retriever import SourceRetriever

        with pytest.raises(AssertionError):
            SourceRetriever(service_type="invalid_service")

    def test_wikipedia_service_type(self):
        from src.fact_reasoner.core.retriever import SourceRetriever

        retriever = SourceRetriever(service_type="wikipedia", top_k=3)
        assert retriever.service_type == "wikipedia"
        assert retriever.top_k == 3
        assert retriever.langchain_retriever is not None

    def test_google_service_type(self):
        from src.fact_reasoner.core.retriever import SourceRetriever
        import os

        with patch.dict(os.environ, {"SERPER_API_KEY": "test_key"}):
            retriever = SourceRetriever(
                service_type="google", top_k=5, cache_dir=None, fetch_text=True
            )
            assert retriever.service_type == "google"
            assert retriever.top_k == 5
            assert retriever.fetch_text is True
            assert retriever.google_retriever is not None

    def test_ntrs_service_type(self):
        from src.fact_reasoner.core.retriever import SourceRetriever

        retriever = SourceRetriever(service_type="ntrs", top_k=5)
        assert retriever.service_type == "ntrs"
        assert retriever.top_k == 5
        assert retriever.ntrs_retriever is not None

    def test_set_query_builder(self):
        from src.fact_reasoner.core.retriever import SourceRetriever

        retriever = SourceRetriever(service_type="wikipedia", top_k=3)
        mock_query_builder = MagicMock()
        retriever.set_query_builder(mock_query_builder)
        assert retriever.query_builder == mock_query_builder

    def test_default_domain_filters_are_none(self):
        from src.fact_reasoner.core.retriever import SourceRetriever

        retriever = SourceRetriever(service_type="wikipedia", top_k=3)
        assert retriever.excluded_domains is None
        assert retriever.included_domains is None


class TestDomainMatches:
    """Tests for the _domain_matches helper."""

    def test_exact_match(self):
        assert _domain_matches("facebook.com", "facebook.com") is True

    def test_subdomain_matches(self):
        assert _domain_matches("www.facebook.com", "facebook.com") is True
        assert _domain_matches("m.facebook.com", "facebook.com") is True

    def test_unrelated_domain_does_not_match(self):
        assert _domain_matches("nasa.gov", "facebook.com") is False

    def test_lookalike_domain_does_not_match(self):
        # "notfacebook.com" is not a subdomain of "facebook.com"
        assert _domain_matches("notfacebook.com", "facebook.com") is False

    def test_bare_tld_matches_any_host_under_it(self):
        assert _domain_matches("www.nasa.gov", "gov") is True
        assert _domain_matches("nasa.gov", ".gov") is True  # leading dot allowed

    def test_case_insensitive(self):
        assert _domain_matches("WWW.Facebook.COM", "facebook.com") is True


class TestSourceRetrieverDomainFilter:
    """Tests for SourceRetriever's excluded_domains/included_domains handling."""

    def _make_hit(self, link):
        return {"title": "t", "snippet": "s", "link": link}

    def test_excluded_domains_drops_matching_hits(self):
        from src.fact_reasoner.core.retriever import SourceRetriever

        retriever = SourceRetriever(
            service_type="wikipedia", top_k=3, excluded_domains=["facebook.com"]
        )
        hits = [
            self._make_hit("https://www.facebook.com/groups/climate"),
            self._make_hit("https://www.nasa.gov/article"),
        ]
        filtered = retriever._filter_hits_by_domain(hits)
        assert [h["link"] for h in filtered] == ["https://www.nasa.gov/article"]

    def test_included_domains_keeps_only_matching_hits(self):
        from src.fact_reasoner.core.retriever import SourceRetriever

        retriever = SourceRetriever(
            service_type="wikipedia", top_k=3, included_domains=["gov", "edu"]
        )
        hits = [
            self._make_hit("https://www.facebook.com/groups/climate"),
            self._make_hit("https://www.nasa.gov/article"),
            self._make_hit("https://climate.mit.edu/article"),
        ]
        filtered = retriever._filter_hits_by_domain(hits)
        assert [h["link"] for h in filtered] == [
            "https://www.nasa.gov/article",
            "https://climate.mit.edu/article",
        ]

    def test_included_domains_takes_precedence_over_excluded(self):
        from src.fact_reasoner.core.retriever import SourceRetriever

        retriever = SourceRetriever(
            service_type="wikipedia",
            top_k=3,
            included_domains=["nasa.gov"],
            excluded_domains=["nasa.gov"],
        )
        hits = [self._make_hit("https://www.nasa.gov/article")]
        # included_domains says keep it, excluded_domains says drop it --
        # both conditions are checked, so it's dropped either way here since
        # a hit must pass the allowlist AND not match the blocklist.
        filtered = retriever._filter_hits_by_domain(hits)
        assert filtered == []

    def test_google_query_appends_exclusion_operators(self):
        from src.fact_reasoner.core.retriever import SourceRetriever
        import os

        with patch.dict(os.environ, {"SERPER_API_KEY": "test_key"}):
            retriever = SourceRetriever(
                service_type="google",
                top_k=1,
                cache_dir=None,
                excluded_domains=["facebook.com", "reddit.com"],
            )
            retriever.google_retriever = MagicMock()
            retriever.google_retriever.get_snippets.return_value = {
                "claim -site:facebook.com -site:reddit.com": [
                    {"title": "t", "snippet": "s", "link": "https://www.nasa.gov/x"}
                ]
            }
            retriever.query("claim")
            called_query = retriever.google_retriever.get_snippets.call_args[0][0][0]
            assert "-site:facebook.com" in called_query
            assert "-site:reddit.com" in called_query

    def test_google_query_builds_included_domains_or_clause(self):
        from src.fact_reasoner.core.retriever import SourceRetriever
        import os

        with patch.dict(os.environ, {"SERPER_API_KEY": "test_key"}):
            retriever = SourceRetriever(
                service_type="google",
                top_k=1,
                cache_dir=None,
                included_domains=["nasa.gov", "noaa.gov"],
            )
            retriever.google_retriever = MagicMock()
            retriever.google_retriever.get_snippets.return_value = {
                "claim (site:nasa.gov OR site:noaa.gov)": [
                    {"title": "t", "snippet": "s", "link": "https://www.nasa.gov/x"}
                ]
            }
            retriever.query("claim")
            called_query = retriever.google_retriever.get_snippets.call_args[0][0][0]
            assert "site:nasa.gov" in called_query
            assert "site:noaa.gov" in called_query
            assert "OR" in called_query

    def test_google_query_post_filters_hits_from_search_results(self):
        """Even if a hit slips through (e.g. a stale cache entry), the
        post-fetch filter should still drop it."""
        from src.fact_reasoner.core.retriever import SourceRetriever
        import os

        with patch.dict(os.environ, {"SERPER_API_KEY": "test_key"}):
            retriever = SourceRetriever(
                service_type="google",
                top_k=5,
                cache_dir=None,
                excluded_domains=["facebook.com"],
            )
            retriever.google_retriever = MagicMock()

            def fake_get_snippets(queries):
                return {
                    queries[0]: [
                        {
                            "title": "t",
                            "snippet": "s",
                            "link": "https://www.facebook.com/groups/x",
                        },
                        {
                            "title": "t2",
                            "snippet": "s2",
                            "link": "https://www.nasa.gov/x",
                        },
                    ]
                }

            retriever.google_retriever.get_snippets.side_effect = fake_get_snippets
            results = retriever.query("claim")
            links = [r["link"] for r in results]
            assert "https://www.facebook.com/groups/x" not in links
            assert "https://www.nasa.gov/x" in links


class TestSourceRetrieverNTRSQuery:
    """Tests for the ``ntrs`` branch of SourceRetriever.query."""

    def _retriever(self, top_k=2):
        from src.fact_reasoner.core.retriever import SourceRetriever

        return SourceRetriever(service_type="ntrs", top_k=top_k)

    def test_maps_abstract_to_text_and_snippet(self):
        retriever = self._retriever(top_k=2)

        hits = {
            "mars rover": [
                {
                    "title": "Rover Geology",
                    "snippet": "Abstract about rover geology.",
                    "link": "https://ntrs.nasa.gov/citations/1",
                }
            ]
        }

        with patch.object(retriever.ntrs_retriever, "get_snippets", return_value=hits):
            results = retriever.query("mars rover")

        assert len(results) == 1
        # NTRS exposes the abstract as the evidence text, so it fills both fields.
        assert results[0]["text"] == "Abstract about rover geology."
        assert results[0]["snippet"] == "Abstract about rover geology."
        assert results[0]["title"] == "Rover Geology"
        assert results[0]["link"] == "https://ntrs.nasa.gov/citations/1"

    def test_passes_top_k_and_truncates_to_it(self):
        retriever = self._retriever(top_k=2)

        hits = {
            "q": [
                {"title": f"T{i}", "snippet": f"A{i}", "link": f"L{i}"}
                for i in range(5)
            ]
        }

        with patch.object(
            retriever.ntrs_retriever, "get_snippets", return_value=hits
        ) as mock_get:
            results = retriever.query("q")

        mock_get.assert_called_once_with(["q"], top_k=2)
        assert [r["title"] for r in results] == ["T0", "T1"]

    def test_empty_text_returns_no_results(self):
        retriever = self._retriever()

        with patch.object(retriever.ntrs_retriever, "get_snippets") as mock_get:
            assert retriever.query("") == []
            mock_get.assert_not_called()

    def test_uses_query_builder_when_present(self):
        retriever = self._retriever(top_k=1)
        query_builder = MagicMock()
        query_builder.run.return_value = "built query"
        retriever.set_query_builder(query_builder)

        hits = {"built query": [{"title": "T", "snippet": "A", "link": "L"}]}

        with patch.object(
            retriever.ntrs_retriever, "get_snippets", return_value=hits
        ) as mock_get:
            results = retriever.query("raw atom text")

        query_builder.run.assert_called_once_with("raw atom text")
        mock_get.assert_called_once_with(["built query"], top_k=1)
        assert len(results) == 1
