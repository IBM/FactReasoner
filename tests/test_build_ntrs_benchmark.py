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

"""Unit tests for fact_reasoner.benchmarks.build_ntrs_benchmark."""

import copy
import json

import pytest

from fact_reasoner.benchmarks.build_ntrs_benchmark import (
    build_datasets,
    write_datasets,
)
from fact_reasoner.ntrs_query import strip_phrase_quotes, to_keyword_query


class FakeQueryBuilder:
    """Stands in for the LLM-backed QueryBuilder, recording every call."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def run(self, text):
        self.calls.append(text)
        return self.result


class FakeRetriever:
    """Stands in for SourceRetriever, recording the query text it receives."""

    def __init__(self, passages=None):
        self.passages = passages if passages is not None else []
        self.queries = []

    def query(self, text):
        self.queries.append(text)
        return self.passages


def _passage(i):
    return {
        "title": f"Title {i}",
        "text": f"Text {i}",
        "snippet": f"Snippet {i}",
        "link": f"https://example.com/{i}",
    }


def _records(num_atoms=1):
    return [
        {
            "input": "Question: Tell me about Mars.",
            "output": "Mars has liquid water.",
            "topic": "Mars",
            "atoms": [
                {
                    "id": f"a{i}",
                    "text": f"Atom {i} text",
                    "original": f"Atom {i} original",
                    "label": "S",
                    "contexts": [],
                }
                for i in range(num_atoms)
            ],
            "contexts": [],
        }
    ]


class TestSharedQueryFairness:
    """The same generated query must drive every backend."""

    def test_query_builder_runs_once_per_atom_not_per_service(self):
        qb = FakeQueryBuilder("Perseverance rover Jezero crater")
        retrievers = {"google": FakeRetriever(), "ntrs": FakeRetriever()}

        build_datasets(_records(num_atoms=2), ["google", "ntrs"], retrievers, qb)

        # Two atoms, two services, but only two generations.
        assert qb.calls == ["Atom 0 text", "Atom 1 text"]

    def test_google_gets_full_query_and_ntrs_gets_keyword_form(self):
        generated = '"liquid water" was detected on Mars by Perseverance'
        qb = FakeQueryBuilder(generated)
        retrievers = {"google": FakeRetriever(), "ntrs": FakeRetriever()}

        build_datasets(_records(), ["google", "ntrs"], retrievers, qb)

        expected_full = strip_phrase_quotes(generated)
        assert retrievers["google"].queries == [expected_full]
        assert retrievers["ntrs"].queries == [
            to_keyword_query(expected_full, max_terms=4)
        ]

    def test_phrase_quotes_are_stripped_before_either_backend(self):
        qb = FakeQueryBuilder('"Jezero crater" Perseverance')
        retrievers = {"google": FakeRetriever(), "ntrs": FakeRetriever()}

        build_datasets(_records(), ["google", "ntrs"], retrievers, qb)

        assert '"' not in retrievers["google"].queries[0]
        assert '"' not in retrievers["ntrs"].queries[0]

    def test_wikipedia_gets_the_full_query_like_google(self):
        qb = FakeQueryBuilder("Perseverance rover Jezero crater Mars")
        retrievers = {"wikipedia": FakeRetriever(), "ntrs": FakeRetriever()}

        build_datasets(_records(), ["wikipedia", "ntrs"], retrievers, qb)

        assert retrievers["wikipedia"].queries == [
            "Perseverance rover Jezero crater Mars"
        ]
        assert retrievers["ntrs"].queries != retrievers["wikipedia"].queries


class TestNtrsMaxTerms:
    """The NTRS word budget is forwarded to the normalization component."""

    def test_ntrs_max_terms_is_honored(self):
        generated = "Perseverance rover collected samples in Jezero crater on Mars"
        qb = FakeQueryBuilder(generated)
        retrievers = {"ntrs": FakeRetriever()}

        build_datasets(_records(), ["ntrs"], retrievers, qb, ntrs_max_terms=2)

        assert retrievers["ntrs"].queries == [to_keyword_query(generated, max_terms=2)]
        assert len(retrievers["ntrs"].queries[0].split()) <= 2


class TestSerializedSchema:
    """Output records must satisfy FactReasoner.from_dict_with_contexts."""

    def test_contexts_are_written_with_the_expected_keys(self):
        qb = FakeQueryBuilder("Mars")
        retrievers = {"ntrs": FakeRetriever([_passage(0), _passage(1)])}

        outputs = build_datasets(_records(), ["ntrs"], retrievers, qb)

        contexts = outputs["ntrs"][0]["contexts"]
        assert len(contexts) == 2
        for ctx in contexts:
            assert set(ctx) == {"id", "title", "text", "link", "snippet"}
        assert contexts[0]["title"] == "Title 0"
        assert contexts[0]["text"] == "Text 0"
        assert contexts[0]["snippet"] == "Snippet 0"
        assert contexts[0]["link"] == "https://example.com/0"

    def test_atom_context_ids_match_the_emitted_contexts(self):
        qb = FakeQueryBuilder("Mars")
        retrievers = {"ntrs": FakeRetriever([_passage(0), _passage(1)])}

        outputs = build_datasets(_records(num_atoms=2), ["ntrs"], retrievers, qb)

        record = outputs["ntrs"][0]
        assert record["atoms"][0]["contexts"] == ["c_a0_0", "c_a0_1"]
        assert record["atoms"][1]["contexts"] == ["c_a1_0", "c_a1_1"]
        # Every referenced id resolves to exactly one emitted context.
        emitted = [c["id"] for c in record["contexts"]]
        assert sorted(emitted) == ["c_a0_0", "c_a0_1", "c_a1_0", "c_a1_1"]

    def test_atom_text_original_and_label_are_preserved(self):
        qb = FakeQueryBuilder("Mars")
        retrievers = {"ntrs": FakeRetriever([_passage(0)])}

        outputs = build_datasets(_records(), ["ntrs"], retrievers, qb)

        atom = outputs["ntrs"][0]["atoms"][0]
        assert atom["text"] == "Atom 0 text"
        assert atom["original"] == "Atom 0 original"
        assert atom["label"] == "S"

    def test_record_level_fields_are_preserved(self):
        qb = FakeQueryBuilder("Mars")
        retrievers = {"ntrs": FakeRetriever()}

        outputs = build_datasets(_records(), ["ntrs"], retrievers, qb)

        record = outputs["ntrs"][0]
        assert record["input"] == "Question: Tell me about Mars."
        assert record["output"] == "Mars has liquid water."
        assert record["topic"] == "Mars"

    def test_missing_passage_fields_become_empty_strings(self):
        qb = FakeQueryBuilder("Mars")
        retrievers = {"ntrs": FakeRetriever([{"title": "Only a title"}])}

        outputs = build_datasets(_records(), ["ntrs"], retrievers, qb)

        ctx = outputs["ntrs"][0]["contexts"][0]
        assert ctx["title"] == "Only a title"
        assert ctx["text"] == ""
        assert ctx["snippet"] == ""
        assert ctx["link"] == ""


class TestPerServiceIsolation:
    """Each backend gets its own record copy; the input is never mutated."""

    def test_one_output_per_requested_service(self):
        qb = FakeQueryBuilder("Mars")
        retrievers = {"google": FakeRetriever(), "ntrs": FakeRetriever()}

        outputs = build_datasets(_records(), ["google", "ntrs"], retrievers, qb)

        assert set(outputs) == {"google", "ntrs"}

    def test_services_do_not_share_contexts(self):
        qb = FakeQueryBuilder("Mars")
        retrievers = {
            "google": FakeRetriever([_passage(0)]),
            "ntrs": FakeRetriever([_passage(1), _passage(2)]),
        }

        outputs = build_datasets(_records(), ["google", "ntrs"], retrievers, qb)

        assert len(outputs["google"][0]["contexts"]) == 1
        assert len(outputs["ntrs"][0]["contexts"]) == 2
        assert outputs["google"][0]["contexts"][0]["title"] == "Title 0"
        assert outputs["ntrs"][0]["contexts"][0]["title"] == "Title 1"

    def test_input_records_are_not_mutated(self):
        qb = FakeQueryBuilder("Mars")
        retrievers = {"ntrs": FakeRetriever([_passage(0)])}
        records = _records()
        before = copy.deepcopy(records)

        build_datasets(records, ["ntrs"], retrievers, qb)

        assert records == before

    def test_empty_retrieval_yields_no_contexts(self):
        qb = FakeQueryBuilder("Mars")
        retrievers = {"ntrs": FakeRetriever([])}

        outputs = build_datasets(_records(), ["ntrs"], retrievers, qb)

        assert outputs["ntrs"][0]["contexts"] == []
        assert outputs["ntrs"][0]["atoms"][0]["contexts"] == []


class TestWriteDatasets:
    """The generated files are checked in frozen, so writes must not clobber."""

    def test_refuses_to_overwrite_existing_output_without_the_flag(self, tmp_path):
        frozen = tmp_path / "nasa_science-labeled_ntrs.jsonl"
        frozen.write_text('{"input": "frozen"}\n')

        with pytest.raises(FileExistsError, match="--overwrite"):
            write_datasets(
                {"ntrs": _records()},
                str(tmp_path),
                "nasa_science-labeled",
            )

        assert frozen.read_text() == '{"input": "frozen"}\n'

    def test_overwrites_existing_output_with_the_flag(self, tmp_path):
        frozen = tmp_path / "nasa_science-labeled_ntrs.jsonl"
        frozen.write_text('{"input": "frozen"}\n')

        write_datasets(
            {"ntrs": _records()},
            str(tmp_path),
            "nasa_science-labeled",
            overwrite=True,
        )

        written = [json.loads(line) for line in frozen.read_text().splitlines()]
        assert written == _records()

    def test_writes_a_missing_output_without_the_flag(self, tmp_path):
        write_datasets(
            {"ntrs": _records()},
            str(tmp_path),
            "nasa_science-labeled",
        )

        out = tmp_path / "nasa_science-labeled_ntrs.jsonl"
        written = [json.loads(line) for line in out.read_text().splitlines()]
        assert written == _records()
