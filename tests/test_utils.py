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

"""Unit tests for fact_reasoner.utils module."""

import asyncio

from fact_reasoner.utils import (
    dotdict,
    strip_string,
    join_segments,
    extract_first_square_brackets,
    extract_last_square_brackets,
    extract_nli_label_and_span,
    extract_last_wrapped_response,
    extract_first_code_block,
    strip_code_fences,
    normalize_ws,
    validate_json_code_block,
    validate_markdown_code_block,
    punctuation_only_inside_quotes,
    batcher,
    run_throttled,
    gather_with_progress,
)


class TestDotdict:
    """Tests for dotdict class."""

    def test_dot_access(self):
        d = dotdict({"key": "value", "number": 42})
        assert d.key == "value"
        assert d.number == 42

    def test_dot_set(self):
        d = dotdict()
        d.key = "value"
        assert d["key"] == "value"

    def test_dot_delete(self):
        d = dotdict({"key": "value"})
        del d.key
        assert "key" not in d

    def test_missing_key_returns_none(self):
        d = dotdict({"key": "value"})
        assert d.missing is None


class TestStripString:
    """Tests for strip_string function."""

    def test_strips_spaces(self):
        assert strip_string("  hello  ") == "hello"

    def test_strips_newlines(self):
        assert strip_string("\n\nhello\n\n") == "hello"

    def test_strips_mixed(self):
        assert strip_string(" \n hello \n ") == "hello"

    def test_empty_string(self):
        assert strip_string("") == ""

    def test_preserves_internal_whitespace(self):
        assert strip_string("  hello world  ") == "hello world"


class TestJoinSegments:
    """Tests for join_segments function."""

    def test_join_strings(self):
        result = join_segments("hello", "world")
        assert result == "hello\n\n\nworld"

    def test_join_with_custom_separator(self):
        result = join_segments("hello", "world", separator=" | ")
        assert result == "hello | world"

    def test_join_list(self):
        result = join_segments(["a", "b", "c"], separator="-")
        assert result == "a-b-c"

    def test_join_mixed(self):
        result = join_segments("start", ["a", "b"], "end", separator="-")
        assert result == "start-a-b-end"


class TestExtractSquareBrackets:
    """Tests for extract_first_square_brackets and extract_last_square_brackets."""

    def test_extract_first_simple(self):
        assert extract_first_square_brackets("hello [world] foo") == "world"

    def test_extract_first_multiple(self):
        assert extract_first_square_brackets("[first] and [second]") == "first"

    def test_extract_first_empty(self):
        assert extract_first_square_brackets("no brackets here") == ""

    def test_extract_last_simple(self):
        assert extract_last_square_brackets("hello [world] foo") == "world"

    def test_extract_last_multiple(self):
        assert extract_last_square_brackets("[first] and [second]") == "second"

    def test_extract_last_empty(self):
        assert extract_last_square_brackets("no brackets here") == ""

    def test_extract_multiline(self):
        text = "[first]\n[second\nline]"
        assert extract_first_square_brackets(text) == "first"

    def test_extract_nli_labels(self):
        text = "The answer is [entailment]"
        assert extract_last_square_brackets(text) == "entailment"

    def test_extract_contradiction(self):
        text = "Based on the evidence, [contradiction]"
        assert extract_last_square_brackets(text) == "contradiction"


class TestExtractNliLabelAndSpan:
    """Tests for extract_nli_label_and_span (JSON + bracket auto-detection)."""

    def test_plain_json(self):
        label, span = extract_nli_label_and_span('{"label": "entailment"}')
        assert label == "entailment"
        assert '{"label": "entailment"}'[slice(*span)] == "entailment"

    def test_json_case_normalized_span_preserves_original(self):
        text = '{"label": "Contradiction"}'
        label, span = extract_nli_label_and_span(text)
        assert label == "contradiction"  # lower-cased label
        assert text[slice(*span)] == "Contradiction"  # span over original value

    def test_json_in_code_fence(self):
        label, _ = extract_nli_label_and_span('```json\n{"label": "neutral"}\n```')
        assert label == "neutral"

    def test_json_with_extra_keys(self):
        label, _ = extract_nli_label_and_span('{"reason": "supports", "label": "entailment"}')
        assert label == "entailment"

    def test_json_spacing(self):
        label, _ = extract_nli_label_and_span('{ "label" : "neutral" }')
        assert label == "neutral"

    def test_json_last_object_wins(self):
        label, _ = extract_nli_label_and_span(
            '{"label":"neutral"} ... actually {"label":"entailment"}'
        )
        assert label == "entailment"

    def test_json_preferred_over_preceding_bracket(self):
        # A citation bracket earlier must not shadow the JSON verdict.
        label, span = extract_nli_label_and_span('see [1]\n{"label": "contradiction"}')
        assert label == "contradiction"
        assert 'see [1]\n{"label": "contradiction"}'[slice(*span)] == "contradiction"

    def test_bracket_fallback(self):
        label, span = extract_nli_label_and_span("3. Final Answer:\n[entailment]")
        assert label == "entailment"
        assert "3. Final Answer:\n[entailment]"[slice(*span)] == "entailment"

    def test_bracket_trailing_punctuation(self):
        label, _ = extract_nli_label_and_span("[neutral.]")
        assert label == "neutral"  # normalized like extract_last_square_brackets

    def test_bare_word_fallback(self):
        label, span = extract_nli_label_and_span("the final answer is contradiction")
        assert label == "contradiction"
        assert "the final answer is contradiction"[slice(*span)] == "contradiction"

    def test_none_found(self):
        label, span = extract_nli_label_and_span("I cannot decide")
        assert label == ""
        assert span is None

    def test_empty(self):
        assert extract_nli_label_and_span("") == ("", None)


class TestExtractWrappedResponse:
    """Tests for extract_last_wrapped_response function."""

    def test_extract_wrapped(self):
        text = "###response###"
        assert extract_last_wrapped_response(text) == "response"

    def test_extract_last_wrapped(self):
        text = "###first### and ###second###"
        assert extract_last_wrapped_response(text) == "second"

    def test_no_wrapped(self):
        assert extract_last_wrapped_response("no wrapping here") == ""

    def test_multiline_wrapped(self):
        text = "###line1\nline2###"
        assert extract_last_wrapped_response(text) == "line1\nline2"


class TestExtractFirstCodeBlock:
    """Tests for extract_first_code_block function."""

    def test_extract_simple(self):
        text = "```\ncode here\n```"
        assert extract_first_code_block(text) == "code here"

    def test_extract_with_language(self):
        text = "```python\nprint('hello')\n```"
        result = extract_first_code_block(text, ignore_language=True)
        assert result == "print('hello')"

    def test_extract_with_language_not_ignored(self):
        text = '```json\n{"key": "value"}\n```'
        result = extract_first_code_block(text, ignore_language=False)
        assert "json" in result

    def test_no_code_block(self):
        assert extract_first_code_block("no code block") == ""

    def test_multiple_code_blocks(self):
        text = "```\nfirst\n```\n```\nsecond\n```"
        assert extract_first_code_block(text) == "first"


class TestStripCodeFences:
    """Tests for strip_code_fences function."""

    def test_strip_json_fences(self):
        text = '```json\n{"key": "value"}\n```'
        assert strip_code_fences(text) == '{"key": "value"}'

    def test_strip_plain_fences(self):
        text = "```\ncontent\n```"
        assert strip_code_fences(text) == "content"

    def test_no_fences(self):
        text = "no fences here"
        assert strip_code_fences(text) == "no fences here"

    def test_irregular_fences(self):
        text = '```json\n{"key": "value"}\n```'
        result = strip_code_fences(text)
        assert "key" in result
        assert "```" not in result


class TestNormalizeWs:
    """Tests for normalize_ws function."""

    def test_collapse_spaces(self):
        assert normalize_ws("hello    world") == "hello world"

    def test_collapse_newlines(self):
        assert normalize_ws("hello\n\nworld") == "hello world"

    def test_collapse_tabs(self):
        assert normalize_ws("hello\t\tworld") == "hello world"

    def test_mixed_whitespace(self):
        assert normalize_ws("hello \n\t world") == "hello world"

    def test_none_input(self):
        assert normalize_ws(None) is None

    def test_strips_leading_trailing(self):
        assert normalize_ws("  hello  ") == "hello"


class TestValidateJsonCodeBlock:
    """Tests for validate_json_code_block function."""

    def test_valid_json_with_fences(self):
        text = '```json\n{"key": "value"}\n```'
        assert validate_json_code_block(text) is True

    def test_valid_json_without_fences(self):
        text = '{"key": "value"}'
        assert validate_json_code_block(text) is True

    def test_invalid_json(self):
        text = '{"key": value}'  # missing quotes
        assert validate_json_code_block(text) is False

    def test_valid_json_with_required_keys(self):
        text = '{"name": "test", "value": 123}'
        assert validate_json_code_block(text, required_keys=["name", "value"]) is True

    def test_missing_required_keys(self):
        text = '{"name": "test"}'
        assert (
            validate_json_code_block(text, required_keys=["name", "missing"]) is False
        )

    def test_json_array(self):
        text = "[1, 2, 3]"
        assert validate_json_code_block(text) is True

    def test_json_array_no_required_keys(self):
        # Arrays don't have keys, so required_keys check should pass
        text = "[1, 2, 3]"
        assert validate_json_code_block(text, required_keys=["key"]) is True


class TestValidateMarkdownCodeBlock:
    """Tests for validate_markdown_code_block function."""

    def test_valid_code_block(self):
        text = "```\ncode\n```"
        assert validate_markdown_code_block(text) is True

    def test_valid_code_block_with_language(self):
        text = "```python\ncode\n```"
        assert validate_markdown_code_block(text) is True

    def test_invalid_no_fences(self):
        text = "no fences"
        assert validate_markdown_code_block(text) is False

    def test_invalid_only_start(self):
        text = "```\ncode"
        assert validate_markdown_code_block(text) is False


class TestPunctuationOnlyInsideQuotes:
    """Tests for punctuation_only_inside_quotes function."""

    def test_punctuation_inside_quotes(self):
        text = '"hello, world"'
        assert punctuation_only_inside_quotes(text) is True

    def test_punctuation_outside_quotes(self):
        text = "hello, world"
        assert punctuation_only_inside_quotes(text) is False

    def test_semicolon_outside(self):
        text = "hello; world"
        assert punctuation_only_inside_quotes(text) is False

    def test_mixed_inside_outside(self):
        text = '"hello, world"; more'
        assert punctuation_only_inside_quotes(text) is False

    def test_no_punctuation(self):
        text = "hello world"
        assert punctuation_only_inside_quotes(text) is True


class TestBatcher:
    """Tests for batcher function."""

    def test_basic_batching(self):
        items = [1, 2, 3, 4, 5, 6]
        batches = list(batcher(items, batch_size=2))
        assert batches == [[1, 2], [3, 4], [5, 6]]

    def test_leftover_batch(self):
        items = [1, 2, 3, 4, 5]
        batches = list(batcher(items, batch_size=2))
        assert batches == [[1, 2], [3, 4], [5]]

    def test_single_batch(self):
        items = [1, 2]
        batches = list(batcher(items, batch_size=10))
        assert batches == [[1, 2]]

    def test_empty_iterator(self):
        items = []
        batches = list(batcher(items, batch_size=2))
        assert batches == []

    def test_batch_size_one(self):
        items = [1, 2, 3]
        batches = list(batcher(items, batch_size=1))
        assert batches == [[1], [2], [3]]


class TestRunThrottled:
    """Tests for run_throttled (bounded-concurrency runner)."""

    def test_results_positionally_aligned(self):
        async def factory(item):
            return item * 2

        results = asyncio.run(run_throttled(factory, [1, 2, 3], max_concurrency=2))
        assert results == [2, 4, 6]

    def test_exception_captured_in_place(self):
        async def factory(item):
            if item == 2:
                raise ValueError("boom")
            return item

        results = asyncio.run(run_throttled(factory, [1, 2, 3], max_concurrency=3))
        assert results[0] == 1
        assert isinstance(results[1], ValueError)
        assert results[2] == 3

    def test_on_progress_called_once_per_item(self):
        calls = {"n": 0}

        async def factory(item):
            return item

        def on_progress():
            calls["n"] += 1

        asyncio.run(
            run_throttled(
                factory, [1, 2, 3, 4], max_concurrency=2, on_progress=on_progress
            )
        )
        assert calls["n"] == 4

    def test_on_progress_fires_even_on_failure(self):
        calls = {"n": 0}

        async def factory(item):
            raise RuntimeError("nope")

        def on_progress():
            calls["n"] += 1

        results = asyncio.run(
            run_throttled(
                factory, [1, 2], max_concurrency=2, on_progress=on_progress
            )
        )
        assert calls["n"] == 2  # progress advances for failures too
        assert all(isinstance(r, RuntimeError) for r in results)


class TestGatherWithProgress:
    """Tests for gather_with_progress (order-preserving async gather + ticks)."""

    def test_results_in_input_order_despite_completion_order(self):
        async def work(x, delay):
            await asyncio.sleep(delay)
            return x * 10

        async def run():
            # Finish out of order (index 1 first, then 2, then 0).
            return await gather_with_progress(
                [work(0, 0.03), work(1, 0.01), work(2, 0.02)]
            )

        assert asyncio.run(run()) == [0, 10, 20]

    def test_on_progress_called_once_per_coro(self):
        calls = {"n": 0}

        async def work(x):
            return x

        async def run():
            await gather_with_progress(
                [work(1), work(2), work(3)],
                on_progress=lambda: calls.__setitem__("n", calls["n"] + 1),
            )

        asyncio.run(run())
        assert calls["n"] == 3

    def test_empty_is_noop(self):
        assert asyncio.run(gather_with_progress([])) == []
