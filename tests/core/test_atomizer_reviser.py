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

"""Unit tests for fact_reasoner.core.atomizer_reviser."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mellea.backends import ModelOption

from fact_reasoner.backends import build_backend
from fact_reasoner.core.atomizer_reviser import (
    INSTRUCTION_ATOMIZER_REVISER,
    AtomizerReviser,
    parse_units,
)
from fact_reasoner.core.utils import build_atoms

ACHAT = "fact_reasoner.core.atomizer_reviser.mfuncs.achat"

# One training example for the adapter (input conversation and target output).
QUERY = "How deep into the Earth could humans dig? (If any equipment was available)"
RESPONSE = (
    '"The result was the Kola Superdeep Borehole and a drill-depth of more than 7.5 '
    "miles (12 kilometers). To put that in perspective, Kola descends further than "
    "the deepest point of the ocean, which lies at nearly 6.8 miles (11 kilometers). "
    'The borehole is located on the Kola Peninsula of Russia.Mar 11, 2015. " URL_0 '
    "They had to stop drilling because of the heat."
)
TRAINING_INSTRUCTION = (
    "<atomizer>As a judge agent, your role is to decompose the provided text into "
    "units that meet the given criteria, utilizing all available information, "
    "including conversations, documents, and tools.\n\n### Criteria: An atomic unit "
    "is a single, self-contained statement that expresses exactly one piece of "
    "information from the text. It should be minimal (it cannot be split further "
    "without losing meaning), independently verifiable, and understandable on its "
    "own without reference to the other units. Resolve pronouns and references so "
    "that each unit stands alone.\n\n### Scoring Schema: Decompose the last "
    "assistant's text into the units that meet the criteria. Return one unit per "
    "line, with no numbering, bullets, or extra commentary."
)


def _reviser():
    backend = MagicMock()
    backend.model_id = "fr-atomizer-reviser"
    return AtomizerReviser(backend)


def _reply(text):
    message = MagicMock()
    message.content = text
    return (message, MagicMock())


class TestParseUnits:
    def test_one_unit_per_line(self):
        assert parse_units("A is B.\nC is D.") == ["A is B.", "C is D."]

    def test_drops_blank_lines_and_whitespace(self):
        assert parse_units("\n  A is B.  \n\n\nC is D.\n") == ["A is B.", "C is D."]

    @pytest.mark.parametrize("prefix", ["- ", "* ", "• ", "1. ", "12) "])
    def test_strips_list_markers(self, prefix):
        assert parse_units(f"{prefix}A is B.") == ["A is B."]

    def test_keeps_leading_numbers_that_are_content(self):
        assert parse_units("1879 was the year Einstein was born.") == [
            "1879 was the year Einstein was born."
        ]

    def test_empty(self):
        assert parse_units("") == []
        assert parse_units("\n \n") == []


class TestInstruction:
    def test_matches_training_data_verbatim(self):
        assert INSTRUCTION_ATOMIZER_REVISER == TRAINING_INSTRUCTION


class TestAtomizerReviser:
    def test_none_backend_raises(self):
        with pytest.raises(ValueError, match="Mellea backend is None"):
            AtomizerReviser(backend=None)

    @pytest.mark.parametrize("value", ["false", "low", 0, 1])
    def test_reasoning_rejects_non_bool_values(self, value):
        with pytest.raises(TypeError, match="reasoning must be True, False or None"):
            AtomizerReviser(MagicMock(), reasoning=value)

    def test_marks_atoms_as_revised(self):
        assert AtomizerReviser.revises_atoms is True

    def test_run_returns_atomizer_shaped_dict(self):
        with patch(ACHAT, new=AsyncMock(return_value=_reply("A is B.\nC is D."))):
            assert _reviser().run(RESPONSE, query=QUERY) == {
                "0": "A is B.",
                "1": "C is D.",
            }

    def test_sends_conversation_and_disables_reasoning(self):
        achat = AsyncMock(return_value=_reply("A is B."))
        with patch(ACHAT, new=achat):
            _reviser().run(RESPONSE, query=QUERY)

        args, kwargs = achat.call_args
        assert args[0] == INSTRUCTION_ATOMIZER_REVISER
        turns = [(m.role, m.content) for m in kwargs["context"].as_list()]
        assert turns == [("user", QUERY), ("assistant", RESPONSE)]
        assert kwargs["model_options"] == {ModelOption.THINKING: False}

    def test_without_query_only_the_response_precedes(self):
        achat = AsyncMock(return_value=_reply("A is B."))
        with patch(ACHAT, new=achat):
            _reviser().run(RESPONSE)
        turns = [m.role for m in achat.call_args.kwargs["context"].as_list()]
        assert turns == ["assistant"]

    def test_reasoning_none_sends_no_setting(self):
        backend = MagicMock()
        backend.model_id = "fr-atomizer-reviser"
        achat = AsyncMock(return_value=_reply("A is B."))
        with patch(ACHAT, new=achat):
            AtomizerReviser(backend, reasoning=None).run(RESPONSE)
        assert achat.call_args.kwargs["model_options"] is None

    def test_empty_output_returns_empty_dict(self):
        with patch(ACHAT, new=AsyncMock(return_value=_reply("\n\n"))):
            assert _reviser().run(RESPONSE, query=QUERY) == {}

    def test_generation_error_returns_empty_dict(self):
        with patch(ACHAT, new=AsyncMock(side_effect=RuntimeError("boom"))):
            assert _reviser().run(RESPONSE, query=QUERY) == {}

    def test_run_batch_stays_aligned_when_one_fails(self):
        async def fake_achat(content, context, **kwargs):
            response = context.as_list()[-1].content
            if response == "bad":
                raise RuntimeError("boom")
            return _reply(f"{response} unit.")

        with patch(ACHAT, side_effect=fake_achat):
            results = asyncio.run(
                _reviser().run_batch(["ok1", "bad", "ok2"], ["q1", "q2", "q3"])
            )
        assert results == [{"0": "ok1 unit."}, {}, {"0": "ok2 unit."}]

    def test_run_batch_rejects_misaligned_queries(self):
        with pytest.raises(ValueError, match="aligned"):
            asyncio.run(_reviser().run_batch(["a", "b"], ["q"]))


class TestBuildAtoms:
    def test_passes_query_to_revising_extractor(self):
        extractor = MagicMock()
        extractor.revises_atoms = True
        extractor.run.return_value = {"0": "A is B."}
        atoms = build_atoms("resp", extractor, query="q")
        extractor.run.assert_called_once_with("resp", query="q")
        assert atoms["a0"].text == "A is B."

    def test_plain_atomizer_called_without_query(self):
        extractor = MagicMock(spec=["run"])
        extractor.run.return_value = {"0": "A is B."}
        build_atoms("resp", extractor, query="q")
        extractor.run.assert_called_once_with("resp")


class _FakeOpenAI(BaseHTTPRequestHandler):
    """A minimal /v1/chat/completions endpoint that records each request."""

    requests: list = []
    reply = ""

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(body)
        data = json.dumps(
            {
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": type(self).reply},
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class TestWireFormat:
    """End to end through Mellea's vLLM (OpenAI-compatible) backend."""

    def test_request_matches_training_conversation(self):
        _FakeOpenAI.requests = []
        _FakeOpenAI.reply = "The Kola Superdeep Borehole is in Russia.\n"
        server = HTTPServer(("127.0.0.1", 0), _FakeOpenAI)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            backend = build_backend(
                "vllm",
                model_id="fr-atomizer-reviser",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
            )
            atoms = build_atoms(RESPONSE, AtomizerReviser(backend), query=QUERY)
        finally:
            server.shutdown()

        (request,) = _FakeOpenAI.requests
        assert request["model"] == "fr-atomizer-reviser"
        # Exactly the training-data input conversation, nothing added.
        assert request["messages"] == [
            {"role": "user", "content": QUERY},
            {"role": "assistant", "content": RESPONSE},
            {"role": "user", "content": TRAINING_INSTRUCTION},
        ]
        # vLLM disables granite's thinking via the chat template.
        assert request["chat_template_kwargs"] == {"enable_thinking": False}
        assert {a.text for a in atoms.values()} == {
            "The Kola Superdeep Borehole is in Russia."
        }
