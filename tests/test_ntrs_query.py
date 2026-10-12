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

"""Unit tests for fact_reasoner.ntrs_query module."""

from fact_reasoner.ntrs_query import (
    _fold_to_ascii,
    strip_phrase_quotes,
    to_keyword_query,
)


class TestStripPhraseQuotes:
    """Tests for strip_phrase_quotes."""

    def test_removes_quotes_but_keeps_every_word(self):
        assert (
            strip_phrase_quotes('"liquid water" on Mars') == "liquid water on Mars"
        )

    def test_query_without_quotes_is_unchanged(self):
        assert strip_phrase_quotes("liquid water on Mars") == "liquid water on Mars"

    def test_collapses_whitespace_left_by_removed_quotes(self):
        assert strip_phrase_quotes('  "a"   "b"  ') == "a b"


class TestFoldToAscii:
    """Tests for _fold_to_ascii."""

    def test_folds_accented_letters_to_ascii_base(self):
        assert _fold_to_ascii("Séítah") == "Seitah"

    def test_plain_ascii_is_unchanged(self):
        assert _fold_to_ascii("Mars rover") == "Mars rover"


class TestToKeywordQuery:
    """Tests for to_keyword_query."""

    def test_anchors_outrank_content_words(self):
        # "landed" loses to the anchors and to the earlier content word, since
        # selection is by retrieval tier rather than token position.
        assert (
            to_keyword_query("the Apollo 11 mission landed on the Moon")
            == "Apollo 11 mission Moon"
        )

    def test_demoted_terms_only_fill_leftover_budget(self):
        # "NASA" is corpus-implicit and "detected" is a reporting verb, so both
        # are demoted; "NASA" fits the last slot and "detected" does not.
        assert (
            to_keyword_query("NASA detected liquid water on Mars")
            == "NASA liquid water Mars"
        )

    def test_interior_of_stays_inside_a_name(self):
        assert to_keyword_query("the Sea of Tranquility") == "Sea of Tranquility"

    def test_strips_possessives_and_folds_accents(self):
        assert to_keyword_query("Jupiter's moon Séítah") == "Jupiter moon Seitah"

    def test_drops_search_operators_and_booleans(self):
        assert (
            to_keyword_query("site:nasa.gov Perseverance OR Curiosity rover")
            == "Perseverance Curiosity rover"
        )

    def test_oversized_unit_is_truncated_to_the_budget(self):
        # The whole run is one atomic anchor unit of five words, which can never
        # fit a budget of three, so it keeps its first three words.
        assert (
            to_keyword_query("Apollo 11 Command Module Columbia", max_terms=3)
            == "Apollo 11 Command"
        )

    def test_deduplicates_repeated_units(self):
        assert to_keyword_query("Mars rover Mars rover") == "Mars rover"

    def test_unit_that_exceeds_remaining_budget_is_skipped(self):
        # "Apollo 11" needs two slots but only one remains after "Moon", so it is
        # skipped and the smaller later anchor "Mars" is taken instead.
        assert (
            to_keyword_query("Moon and Apollo 11 and Mars", max_terms=2)
            == "Moon Mars"
        )

    def test_never_exceeds_the_word_budget(self):
        query = "Perseverance rover collected samples in Jezero crater on Mars"
        for max_terms in (1, 2, 3, 4, 5):
            result = to_keyword_query(query, max_terms=max_terms)
            assert len(result.split()) <= max_terms

    def test_falls_back_to_original_query_when_nothing_survives(self):
        assert to_keyword_query("the and of") == "the and of"
