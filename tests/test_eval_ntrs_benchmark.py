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

"""Tests for the frozen NTRS benchmark evaluation runner."""

import json

import pytest

from fact_reasoner.benchmarks.eval_ntrs_benchmark import (
    evaluate,
    load_previous_results,
    summarize,
)


class FakePipeline:
    """Stands in for FactVerify/FactScore, recording how it was driven."""

    def __init__(self, score_result, calls):
        self._score_result = score_result
        self._calls = calls
        self.record = None
        self.build_kwargs = None

    def from_dict_with_contexts(self, data):
        self.record = data
        return True

    def build(self, **kwargs):
        self.build_kwargs = kwargs

    def score(self):
        self._calls.append(self)
        return dict(self._score_result)


def make_factory(calls, score_results=None):
    """Build a pipeline factory yielding one FakePipeline per record."""

    results = iter(score_results or [])

    def factory():
        try:
            result = next(results)
        except StopIteration:
            result = {"query": "unused", "num_atoms": 0}
        return FakePipeline(result, calls)

    return factory


def record(input_text, topic=None):
    return {"input": input_text, "topic": topic, "atoms": [], "contexts": []}


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def test_summarize_pools_confusion_matrix_across_responses():
    results = [
        {
            "true_positive": 3,
            "true_negative": 1,
            "false_positive": 1,
            "false_negative": 0,
            "num_atoms": 5,
            "num_contexts": 10,
            "factuality_score": 0.8,
        },
        {
            "true_positive": 2,
            "true_negative": 2,
            "false_positive": 0,
            "false_negative": 1,
            "num_atoms": 5,
            "num_contexts": 12,
            "factuality_score": 0.6,
        },
    ]

    summary = summarize(results)

    assert summary["num_responses"] == 2
    assert summary["num_atoms"] == 10
    assert summary["num_contexts"] == 22
    assert summary["true_positive"] == 5
    assert summary["true_negative"] == 3
    assert summary["false_positive"] == 1
    assert summary["false_negative"] == 1
    assert summary["precision"] == pytest.approx(5 / 6)
    assert summary["recall"] == pytest.approx(5 / 6)
    assert summary["f1"] == pytest.approx(5 / 6)
    # accuracy is (tp + tn) / num_atoms, not over the confusion matrix total.
    assert summary["accuracy"] == pytest.approx(0.8)
    assert summary["avg_factuality_score"] == pytest.approx(0.7)


def test_summarize_of_empty_results_is_all_zero():
    summary = summarize([])

    assert summary["num_responses"] == 0
    assert summary["precision"] == 0.0
    assert summary["recall"] == 0.0
    assert summary["f1"] == 0.0
    assert summary["accuracy"] == 0.0
    assert summary["avg_factuality_score"] == 0.0


def test_summarize_treats_missing_counts_as_zero():
    summary = summarize([{"factuality_score": 1.0}])

    assert summary["num_responses"] == 1
    assert summary["true_positive"] == 0
    assert summary["num_atoms"] == 0
    assert summary["accuracy"] == 0.0
    assert summary["avg_factuality_score"] == pytest.approx(1.0)


def test_summarize_ignores_responses_without_a_factuality_score():
    summary = summarize([{"factuality_score": 0.5}, {"num_atoms": 2}])

    # Averaged over the one response that reported a score.
    assert summary["avg_factuality_score"] == pytest.approx(0.5)


def test_summarize_zero_precision_and_recall_yields_zero_f1():
    summary = summarize(
        [{"true_positive": 0, "false_positive": 2, "false_negative": 3}]
    )

    assert summary["precision"] == 0.0
    assert summary["recall"] == 0.0
    assert summary["f1"] == 0.0


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_scores_every_record_and_tags_results(tmp_path):
    out = tmp_path / "eval.jsonl"
    calls = []
    results = evaluate(
        [record("a"), record("b")],
        make_factory(calls, [{"query": "a"}, {"query": "b"}]),
        str(out),
        model_id="gpt-4o-mini",
        service_type="ntrs",
    )

    assert len(results) == 2
    assert [r["query"] for r in results] == ["a", "b"]
    assert all(r["model_name"] == "gpt-4o-mini" for r in results)
    assert all(r["service_type"] == "ntrs" for r in results)
    assert len(calls) == 2


def test_evaluate_drives_build_with_frozen_atoms_and_contexts(tmp_path):
    calls = []
    evaluate(
        [record("a", topic="Apollo 11")],
        make_factory(calls, [{"query": "a"}]),
        str(tmp_path / "eval.jsonl"),
        model_id="m",
        service_type="ntrs",
    )

    assert calls[0].build_kwargs == {
        "topic": "Apollo 11",
        "has_atoms": True,
        "has_contexts": True,
        "revise_atoms": False,
    }


def test_evaluate_loads_each_record_into_the_pipeline(tmp_path):
    calls = []
    rec = record("a", topic="Voyager 1")
    evaluate(
        [rec],
        make_factory(calls, [{"query": "a"}]),
        str(tmp_path / "eval.jsonl"),
        model_id="m",
        service_type="ntrs",
    )

    assert calls[0].record is rec


def test_evaluate_writes_one_json_line_per_result(tmp_path):
    out = tmp_path / "eval.jsonl"
    evaluate(
        [record("a"), record("b")],
        make_factory([], [{"query": "a"}, {"query": "b"}]),
        str(out),
        model_id="m",
        service_type="google",
    )

    lines = out.read_text().splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["query"] for line in lines] == ["a", "b"]


def test_evaluate_skips_records_already_present_in_previous_results(tmp_path):
    calls = []
    results = evaluate(
        [record("a"), record("b")],
        make_factory(calls, [{"query": "b"}]),
        str(tmp_path / "eval.jsonl"),
        model_id="m",
        service_type="ntrs",
        evaluation_data=[{"query": "a", "num_atoms": 3}],
    )

    # Only "b" was scored; the resumed result for "a" is preserved.
    assert len(calls) == 1
    assert [r["query"] for r in results] == ["a", "b"]


def test_evaluate_resume_keys_on_query_not_input(tmp_path):
    """score() reports the response under "query", so resume must match on it."""

    calls = []
    evaluate(
        [record("a")],
        make_factory(calls, [{"query": "a"}]),
        str(tmp_path / "eval.jsonl"),
        model_id="m",
        service_type="ntrs",
        evaluation_data=[{"input": "a"}],
    )

    # An "input"-keyed prior result must not be mistaken for a completed one.
    assert len(calls) == 1


def test_evaluate_does_not_mutate_the_resumed_results_list(tmp_path):
    previous = [{"query": "a"}]
    evaluate(
        [record("b")],
        make_factory([], [{"query": "b"}]),
        str(tmp_path / "eval.jsonl"),
        model_id="m",
        service_type="ntrs",
        evaluation_data=previous,
    )

    assert previous == [{"query": "a"}]


def test_evaluate_with_no_records_returns_resumed_results(tmp_path):
    out = tmp_path / "eval.jsonl"
    results = evaluate(
        [],
        make_factory([]),
        str(out),
        model_id="m",
        service_type="ntrs",
        evaluation_data=[{"query": "a"}],
    )

    assert results == [{"query": "a"}]
    # Nothing was scored, so no file is written.
    assert not out.exists()


# ---------------------------------------------------------------------------
# load_previous_results
# ---------------------------------------------------------------------------


def test_load_previous_results_of_missing_file_is_empty(tmp_path):
    assert load_previous_results(str(tmp_path / "absent.jsonl")) == []


def test_load_previous_results_reads_jsonl_and_ignores_blank_lines(tmp_path):
    out = tmp_path / "eval.jsonl"
    out.write_text('{"query": "a"}\n\n{"query": "b"}\n')

    assert load_previous_results(str(out)) == [{"query": "a"}, {"query": "b"}]


def test_evaluate_round_trips_through_load_previous_results(tmp_path):
    out = tmp_path / "eval.jsonl"
    evaluate(
        [record("a")],
        make_factory([], [{"query": "a"}]),
        str(out),
        model_id="m",
        service_type="ntrs",
    )

    calls = []
    results = evaluate(
        [record("a"), record("b")],
        make_factory(calls, [{"query": "b"}]),
        str(out),
        model_id="m",
        service_type="ntrs",
        evaluation_data=load_previous_results(str(out)),
    )

    # The first record is recognized as done; only "b" is scored.
    assert len(calls) == 1
    assert [r["query"] for r in results] == ["a", "b"]
