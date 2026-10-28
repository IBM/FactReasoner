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

"""The Wikipedia source retriever: subject search, failed pages, rate limits.

Everything is offline: ``wikipedia.search`` / ``wikipedia.page`` and the HTTP
layer are replaced with fakes, so the tests pin the retriever's own behaviour.
"""

from types import SimpleNamespace

import pytest

import wikipedia
import wikipedia.wikipedia as wiki_mod

from fact_reasoner.core import retriever as R
from fact_reasoner.core.retriever import SourceRetriever, claim_subject


class TestClaimSubject:
    @pytest.mark.parametrize(
        "claim,subject",
        [
            ("Lanny Flaherty was born in Pensacola, Florida.", "Lanny Flaherty"),
            ("Lanny Flaherty's career began in the late 1970s.", "Lanny Flaherty"),
            ("The Eiffel Tower is in Paris.", "Eiffel Tower"),
            ("In 1903, Marie Curie won the Nobel Prize.", "Marie Curie"),
            ("The United States declared independence in 1776.", "United States"),
            ("Ludwig van Beethoven wrote nine symphonies.", "Ludwig van Beethoven"),
            ("The Bank of England sets rates.", "Bank of England"),
            ("Voyager 1 crossed the heliopause.", "Voyager 1"),
            ("Critics described Lanny Flaherty's voice as soft.", "Lanny Flaherty"),
            # No named subject: no extra search.
            ("It was completed in 1889.", ""),
            ("He was born in 1949.", ""),
            ("Critics often described him as a character actor.", ""),
            ("", ""),
        ],
    )
    def test_subject(self, claim, subject):
        assert claim_subject(claim) == subject


class _FakePage:
    def __init__(self, title):
        self.title = title
        self.summary = f"{title} summary."
        self.url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        self.content = f"{title}: the page text."


@pytest.fixture
def fake_wiki(monkeypatch):
    """Fake search / page loads; ``broken`` titles fail ``fails[title]`` times."""
    state = SimpleNamespace(
        results={}, broken={}, page_calls={},
    )

    def search(query, results=10, suggestion=False):
        return state.results.get(query, [])[:results]

    def page(title=None, auto_suggest=True, **_):
        state.page_calls[title] = state.page_calls.get(title, 0) + 1
        left = state.broken.get(title, 0)
        if left:
            state.broken[title] = left - 1
            raise ValueError("Expecting value: line 1 column 1 (char 0)")  # a non-JSON body
        return _FakePage(title)

    monkeypatch.setattr(wikipedia, "search", search)
    monkeypatch.setattr(wikipedia, "page", page)
    monkeypatch.setattr(R._ResilientWikipedia, "BACKOFF_S", 0.0)
    return state


def _titles(results):
    return [p["title"] for p in results]


class TestWikipediaQuery:
    CLAIM = "Lanny Flaherty is an American actor born on December 18, 1949."

    def test_subject_page_first(self, fake_wiki):
        # The full sentence only hits list pages; the subject finds its own page.
        fake_wiki.results[self.CLAIM] = ["2024 in film", "1942 in film", "February 16"]
        fake_wiki.results["Lanny Flaherty"] = ["Lanny Flaherty", "2024 in film", "Signs (film)"]
        r = SourceRetriever(service_type="wikipedia", top_k=3, fetch_text=True)
        assert _titles(r.query(self.CLAIM)) == ["Lanny Flaherty", "2024 in film", "1942 in film"]

    def test_claim_that_finds_nothing_gets_the_subject(self, fake_wiki):
        claim = "Lanny Flaherty was born in Pensacola, Florida."
        fake_wiki.results["Lanny Flaherty"] = ["Lanny Flaherty", "2024 in film"]
        r = SourceRetriever(service_type="wikipedia", top_k=3, fetch_text=True)
        assert _titles(r.query(claim)) == ["Lanny Flaherty", "2024 in film"]

    def test_no_subject_no_extra_search(self, fake_wiki):
        fake_wiki.results["It was completed in 1889."] = ["Eiffel Tower"]
        r = SourceRetriever(service_type="wikipedia", top_k=3, fetch_text=True)
        assert _titles(r.query("It was completed in 1889.")) == ["Eiffel Tower"]

    def test_a_failing_page_is_skipped_not_the_claim(self, fake_wiki):
        fake_wiki.results[self.CLAIM] = ["2024 in film", "1942 in film"]
        fake_wiki.results["Lanny Flaherty"] = ["Lanny Flaherty"]
        fake_wiki.broken["2024 in film"] = 99   # never loads
        r = SourceRetriever(service_type="wikipedia", top_k=3, fetch_text=True)
        assert _titles(r.query(self.CLAIM)) == ["Lanny Flaherty", "1942 in film"]

    def test_a_transient_failure_is_retried(self, fake_wiki):
        fake_wiki.results["Lanny Flaherty"] = ["Lanny Flaherty"]
        fake_wiki.broken["Lanny Flaherty"] = 1  # fails once, then loads
        r = SourceRetriever(service_type="wikipedia", top_k=3, fetch_text=True)
        assert _titles(r.query("Lanny Flaherty")) == ["Lanny Flaherty"]
        assert fake_wiki.page_calls["Lanny Flaherty"] == 2


class TestRateLimit:
    """Wikipedia's 429s are waited out and retried, not parsed as JSON."""

    def _responses(self, monkeypatch, codes):
        sent = []

        def get(url, params=None, headers=None, timeout=None):
            code = codes.pop(0)
            sent.append(url)
            body = {"query": {"search": []}}
            return SimpleNamespace(
                status_code=code,
                headers={"Retry-After": "0"} if code == 429 else {},
                json=lambda: body,
            )

        monkeypatch.setattr(R.requests, "get", get)
        monkeypatch.setattr(R, "WIKIPEDIA_MAX_RETRY_AFTER_S", 0.0)
        monkeypatch.setattr("time.sleep", lambda s: None)
        return sent

    def test_429_is_retried(self, monkeypatch):
        R._install_wikipedia_rate_limiting()
        sent = self._responses(monkeypatch, [429, 429, 200])
        assert wiki_mod._wiki_request({"list": "search", "srsearch": "x"}) == {"query": {"search": []}}
        assert len(sent) == 3
        assert all(u.startswith("https://") for u in sent)

    def test_persistent_429_raises_a_clear_error(self, monkeypatch):
        R._install_wikipedia_rate_limiting()
        self._responses(monkeypatch, [429] * (R.WIKIPEDIA_429_RETRIES + 1))
        with pytest.raises(R.requests.HTTPError, match="rate-limited"):
            wiki_mod._wiki_request({"list": "search", "srsearch": "x"})

    def test_installed_once(self):
        R._install_wikipedia_rate_limiting()
        first = wiki_mod._wiki_request
        R._install_wikipedia_rate_limiting()
        assert wiki_mod._wiki_request is first
