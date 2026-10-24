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

"""Tests for the opt-in ``raise_errors`` mode of the LLM pipeline stages.

By default a stage prints a failed LLM call and substitutes a fallback result.
With ``raise_errors=True`` the exception raised by the backend reaches the
caller unchanged.
"""

import asyncio
import time
import traceback
from unittest.mock import MagicMock, patch

import pytest
from mellea.backends.dummy import DummyBackend


@pytest.fixture(autouse=True)
def _no_transient_retries(monkeypatch):
    """These tests are about one failed call's fallback, not about retries.

    Their stub backends raise ``ConnectionError``, which ``run_throttled`` would
    otherwise retry (see ``fact_reasoner.utils.TRANSIENT_RETRIES``).
    """
    monkeypatch.setattr("fact_reasoner.utils.TRANSIENT_RETRIES", 0)

from fact_reasoner import telemetry
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.atomizer_reviser import AtomizerReviser
from fact_reasoner.core.base import Atom
from fact_reasoner.core.nli import NLIAdapterExtractor, NLIExtractor
from fact_reasoner.core.retriever import ContextRetriever
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.core.summarizer import ContextSummarizer


class FailingBackend(DummyBackend):
    """A real Mellea backend whose every generation fails with a fresh error."""

    def __init__(self):
        super().__init__(None)
        self.errors: list[ConnectionError] = []

    async def _generate_from_context(self, action, ctx, **kwargs):
        error = ConnectionError("Connection error.")
        self.errors.append(error)
        raise error


NEUTRAL = {"label": "neutral", "probability": 1.0}

# (stage call, default-mode fallback, line printed in default mode, telemetry
# component)
STAGE_CALLS = [
    pytest.param(
        lambda backend, **kw: Atomizer(backend, **kw).run("response"),
        {},
        "[Atomizer] Generation failed: ConnectionError: Connection error.",
        "atomizer",
        id="atomizer-run",
    ),
    pytest.param(
        lambda backend, **kw: asyncio.run(
            Atomizer(backend, **kw).run_batch(["r1", "r2"])
        ),
        [{}, {}],
        "[Atomizer] Batch item failed: ConnectionError: Connection error.",
        "atomizer",
        id="atomizer-run_batch",
    ),
    pytest.param(
        lambda backend, **kw: Reviser(backend, **kw).run(["a1"], "response"),
        [{"revised_unit": "a1", "rationale": "", "text": "a1"}],
        "[Reviser] Batch item failed: ConnectionError: Connection error.",
        "reviser",
        id="reviser-run",
    ),
    pytest.param(
        lambda backend, **kw: asyncio.run(
            Reviser(backend, **kw).run_batch(["a1", "a2"], "response")
        ),
        [
            {"revised_unit": "a1", "rationale": "", "text": "a1"},
            {"revised_unit": "a2", "rationale": "", "text": "a2"},
        ],
        "[Reviser] Batch item failed: ConnectionError: Connection error.",
        "reviser",
        id="reviser-run_batch",
    ),
    pytest.param(
        lambda backend, **kw: NLIExtractor(backend, **kw).run("premise", "hyp"),
        NEUTRAL,
        "[NLI] Generation failed: ConnectionError: Connection error.",
        "nli",
        id="nli-run",
    ),
    pytest.param(
        lambda backend, **kw: asyncio.run(
            NLIExtractor(backend, **kw).run_batch(["p1", "p2"], ["h1", "h2"])
        ),
        [NEUTRAL, NEUTRAL],
        "[NLI] Batch item failed: ConnectionError: Connection error.",
        "nli",
        id="nli-run_batch",
    ),
    pytest.param(
        lambda backend, **kw: NLIAdapterExtractor(backend, **kw).run("p", "h"),
        NEUTRAL,
        "[NLI] Batch item failed: ConnectionError: Connection error.",
        "nli",
        id="nli-adapter-run",
    ),
    pytest.param(
        lambda backend, **kw: ContextSummarizer(backend, **kw).run(["c1"], "atom"),
        [{"context": "c1", "summary": "", "probability": 0.0}],
        "[Summarizer] Batch item failed: ConnectionError: Connection error.",
        "summarizer",
        id="summarizer-run",
    ),
    pytest.param(
        lambda backend, **kw: asyncio.run(
            ContextSummarizer(backend, **kw).run_batch(["c1", "c2"], "atom")
        ),
        [
            {"context": "c1", "summary": "", "probability": 0.0},
            {"context": "c2", "summary": "", "probability": 0.0},
        ],
        "[Summarizer] Batch item failed: ConnectionError: Connection error.",
        "summarizer",
        id="summarizer-run_batch",
    ),
    pytest.param(
        lambda backend, **kw: AtomizerReviser(backend, **kw).run("response"),
        {},
        "[AtomizerReviser] Generation failed: ConnectionError: Connection error.",
        "atomizer_reviser",
        id="atomizer-reviser-run",
    ),
]


@pytest.mark.parametrize("call, fallback, message, component", STAGE_CALLS)
def test_default_mode_prints_and_falls_back(call, fallback, message, component, capsys):
    backend = FailingBackend()

    with patch.object(telemetry, "count_fallback") as count_fallback:
        assert call(backend) == fallback
    assert message in capsys.readouterr().out
    # One generation_error fallback per failed call, as before.
    assert count_fallback.call_count == len(backend.errors)
    assert {c.args for c in count_fallback.call_args_list} == {
        (component, "generation_error")
    }


@pytest.mark.parametrize("call, fallback, message, component", STAGE_CALLS)
def test_raise_mode_raises_the_backend_error(
    call, fallback, message, component, capsys
):
    backend = FailingBackend()

    with (
        patch.object(telemetry, "count_fallback") as count_fallback,
        pytest.raises(ConnectionError) as excinfo,
    ):
        call(backend, raise_errors=True)

    # The very exception the backend raised, not a copy or a wrapper ...
    assert any(error is excinfo.value for error in backend.errors)
    # ... still carrying the traceback down to the failing backend call.
    frames = [f.name for f in traceback.extract_tb(excinfo.value.__traceback__)]
    assert "_generate_from_context" in frames
    # Nothing was reported or counted as a fallback.
    assert message not in capsys.readouterr().out
    count_fallback.assert_not_called()


class TestContextRetriever:
    """The retriever summarizes contexts in its workers; raise mode must survive."""

    @staticmethod
    def _retriever(summarizer, source=None):
        if source is None:
            source = MagicMock()
            source.query.return_value = [
                {"text": "Some text.", "title": "T", "link": "l", "snippet": ""}
            ]
        return ContextRetriever(
            retriever=source, context_summarizer=summarizer, num_workers=2
        )

    def test_summarizer_failure_drops_contexts_by_default(self, capsys):
        retriever = self._retriever(ContextSummarizer(FailingBackend()))
        atoms = {"a0": Atom(id="a0", text="An atom.")}

        # An empty summary marks a context as irrelevant, so it is dropped.
        assert retriever.retrieve_all(atoms) == {}
        assert "[Summarizer] Batch item failed" in capsys.readouterr().out

    def test_summarizer_error_reaches_the_caller_in_raise_mode(self):
        backend = FailingBackend()
        retriever = self._retriever(ContextSummarizer(backend, raise_errors=True))
        atoms = {"a0": Atom(id="a0", text="An atom.")}

        with pytest.raises(ConnectionError) as excinfo:
            retriever.retrieve_all(atoms)
        assert any(error is excinfo.value for error in backend.errors)

    def test_summarizer_without_raise_mode_keeps_the_old_path(self):
        # Only raise_errors=True re-raises; any other summarizer error (here a
        # mock with no real raise_errors attribute) is logged and skipped.
        summarizer = MagicMock()
        summarizer.run.side_effect = RuntimeError("summarizer bug")
        retriever = self._retriever(summarizer)
        atoms = {"a0": Atom(id="a0", text="An atom.")}

        assert retriever.retrieve_all(atoms) == {}

    def test_raise_mode_cancels_the_queued_atoms(self):
        source = MagicMock()

        def query(text):
            time.sleep(0.05)
            return [{"text": "Some text.", "title": "T", "link": "l", "snippet": ""}]

        source.query.side_effect = query
        summarizer = ContextSummarizer(FailingBackend(), raise_errors=True)
        retriever = ContextRetriever(
            retriever=source, context_summarizer=summarizer, num_workers=1
        )
        atoms = {f"a{i}": Atom(id=f"a{i}", text=f"Atom {i}.") for i in range(8)}

        with pytest.raises(ConnectionError):
            retriever.retrieve_all(atoms)
        # Atoms still waiting in the queue were cancelled, not retrieved.
        assert source.query.call_count < len(atoms)

    def test_retrieval_errors_are_still_skipped_in_raise_mode(self):
        # raise_errors covers the LLM call; a failed search keeps its fallback.
        source = MagicMock()
        source.query.side_effect = RuntimeError("search failed")
        summarizer = ContextSummarizer(FailingBackend(), raise_errors=True)
        retriever = self._retriever(summarizer, source)
        atoms = {"a0": Atom(id="a0", text="An atom.")}

        assert retriever.retrieve_all(atoms) == {}
