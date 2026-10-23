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

"""Tests for the pipeline speedups: the shared backend throttle, the per-run
page cache, concurrent retrieval with a batch deadline, and the single
context-summarization batch in FactReasoner.build."""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fact_reasoner.assessor import FactReasoner
from fact_reasoner.core.base import Atom, Context
from fact_reasoner.core.retriever import ContextRetriever, SourceRetriever
from fact_reasoner.utils import BackendThrottle, run_throttled, throttle_for


class _Backend:
    """A weak-referenceable stand-in backend."""


class TestBackendThrottle:
    def test_bounds_concurrency_across_batches(self):
        throttle = BackendThrottle(max_concurrency=3)
        state = {"now": 0, "peak": 0}

        async def factory(item):
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            await asyncio.sleep(0.01)
            state["now"] -= 1
            return item

        async def two_batches():
            # Each batch alone would allow 32 in flight; together they share 3.
            return await asyncio.gather(
                run_throttled(factory, list(range(10)), throttle=throttle),
                run_throttled(factory, list(range(10, 20)), throttle=throttle),
            )

        first, second = asyncio.run(two_batches())
        assert first == list(range(10))
        assert second == list(range(10, 20))
        assert state["peak"] == 3

    def test_bounds_concurrency_across_event_loops(self):
        # build() and run_coroutine use different loops; one budget covers both.
        throttle = BackendThrottle(max_concurrency=2)
        lock = threading.Lock()
        state = {"now": 0, "peak": 0}

        async def factory(item):
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            await asyncio.sleep(0.02)
            with lock:
                state["now"] -= 1
            return item

        def worker(items, out):
            out.extend(asyncio.run(run_throttled(factory, items, throttle=throttle)))

        outs = [[], []]
        threads = [
            threading.Thread(target=worker, args=(list(range(i * 6, i * 6 + 6)), outs[i]))
            for i in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert outs == [list(range(6)), list(range(6, 12))]
        assert state["peak"] == 2

    def test_slot_released_when_item_fails(self):
        throttle = BackendThrottle(max_concurrency=1)

        async def factory(item):
            raise RuntimeError(f"fail {item}")

        results = asyncio.run(run_throttled(factory, [1, 2, 3], throttle=throttle))
        assert all(isinstance(r, RuntimeError) for r in results)
        assert throttle._in_flight == 0

    def test_rate_limit_applies(self):
        # 120/min = one token every 0.5s once the initial burst is spent.
        throttle = BackendThrottle(max_concurrency=10, rate_per_minute=120)
        throttle._tokens = 0.0
        start = time.monotonic()

        async def factory(item):
            return item

        asyncio.run(run_throttled(factory, [1, 2], throttle=throttle))
        assert time.monotonic() - start >= 0.9

    def test_rejects_invalid_limits(self):
        with pytest.raises(ValueError):
            BackendThrottle(max_concurrency=0)
        with pytest.raises(ValueError):
            BackendThrottle(rate_per_minute=0)

    def test_one_throttle_per_backend(self):
        a, b = _Backend(), _Backend()
        assert throttle_for(a) is throttle_for(a)
        assert throttle_for(a) is not throttle_for(b)

    def test_unhashable_backend_falls_back(self):
        # MagicMock supports weakrefs; a plain object() subclass without
        # __weakref__ (like int) does not.
        assert throttle_for(5) is None


class TestPageCache:
    @staticmethod
    def _retriever():
        with patch("fact_reasoner.core.retriever.SearchAPI"):
            return SourceRetriever(service_type="google", fetch_text=True)

    def test_each_page_fetched_once(self):
        retriever = self._retriever()
        with patch(
            "fact_reasoner.core.retriever.fetch_text_from_link",
            side_effect=lambda link, size: f"text of {link}",
        ) as fetch:
            assert retriever._fetch_page("u1", 100) == "text of u1"
            assert retriever._fetch_page("u1", 100) == "text of u1"
            assert retriever._fetch_page("u2", 100) == "text of u2"
        assert fetch.call_count == 2

    def test_concurrent_requests_share_one_fetch(self):
        retriever = self._retriever()
        calls = []

        def slow_fetch(link, size):
            calls.append(link)
            time.sleep(0.05)
            return "page"

        results = []
        with patch(
            "fact_reasoner.core.retriever.fetch_text_from_link", side_effect=slow_fetch
        ):
            threads = [
                threading.Thread(
                    target=lambda: results.append(retriever._fetch_page("u", None))
                )
                for _ in range(5)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        assert calls == ["u"]
        assert results == ["page"] * 5

    def test_failed_fetch_is_not_retried(self):
        retriever = self._retriever()
        with patch(
            "fact_reasoner.core.retriever.fetch_text_from_link",
            side_effect=TimeoutError("dead link"),
        ) as fetch:
            for _ in range(3):
                with pytest.raises(TimeoutError):
                    retriever._fetch_page("dead", None)
        assert fetch.call_count == 1


class TestConcurrentRetrieval:
    @staticmethod
    def _source(delay=0.0, slow_text=None):
        source = MagicMock()

        def query(text):
            time.sleep(1.0 if text == slow_text else delay)
            return [{"text": f"About {text}", "title": "T", "link": "l", "snippet": ""}]

        source.query.side_effect = query
        return source

    def test_items_retrieved_concurrently_by_default(self):
        source = self._source(delay=0.1)
        retriever = ContextRetriever(retriever=source)
        atoms = {f"a{i}": Atom(id=f"a{i}", text=f"Atom {i}.") for i in range(8)}

        start = time.monotonic()
        contexts = retriever.retrieve_all(atoms, query="Q")
        elapsed = time.monotonic() - start

        # Nine items of 0.1s each: ~0.1s concurrently, 0.9s one by one.
        assert elapsed < 0.5
        assert len(contexts) == 9
        assert "c_q_0" in contexts
        assert all(f"c_a{i}_0" in atoms[f"a{i}"].contexts for i in range(8))

    def test_explicit_num_workers_is_honoured(self):
        source = self._source(delay=0.05)
        retriever = ContextRetriever(retriever=source, num_workers=1)
        atoms = {f"a{i}": Atom(id=f"a{i}", text=f"Atom {i}.") for i in range(4)}

        start = time.monotonic()
        retriever.retrieve_all(atoms)
        assert time.monotonic() - start >= 0.2

    def test_stuck_item_skipped_at_deadline(self):
        source = self._source(slow_text="Atom 1.")
        retriever = ContextRetriever(retriever=source, per_atom_timeout=0.3)
        atoms = {f"a{i}": Atom(id=f"a{i}", text=f"Atom {i}.") for i in range(3)}

        start = time.monotonic()
        with patch("fact_reasoner.core.retriever.telemetry.count_fallback") as fb:
            contexts = retriever.retrieve_all(atoms)
        # Returns at the deadline instead of waiting for the stuck item.
        assert time.monotonic() - start < 0.9
        assert set(contexts) == {"c_a0_0", "c_a2_0"}
        fb.assert_called_once_with("retriever", "timeout")


def _pipeline(summarizer):
    return FactReasoner(
        atom_extractor=MagicMock(),
        atom_reviser=MagicMock(),
        context_retriever=MagicMock(),
        context_summarizer=summarizer,
        nli_extractor=MagicMock(),
        merlin_path="/path/to/merlin",
    )


class _StopAfterSummaries(Exception):
    pass


def _run_summarization(pipeline):
    """Run build() through the summarization stage only."""
    with patch(
        "fact_reasoner.assessor.build_relations", side_effect=_StopAfterSummaries
    ):
        with pytest.raises(_StopAfterSummaries):
            asyncio.run(
                pipeline.build(
                    query="Who is X?",
                    response="resp",
                    has_atoms=True,
                    has_contexts=True,
                    summarize_contexts=True,
                )
            )


class TestSingleSummarizationBatch:
    @staticmethod
    def _summary(text):
        return {"context": "", "summary": text, "probability": 0.5}

    def test_unsummarized_contexts_go_out_in_one_batch(self):
        summarizer = MagicMock()
        summarizer.run_batch_pairs = AsyncMock(
            return_value=[
                self._summary("X was born in 1900."),
                self._summary("X lived in Paris."),
                self._summary("X is a painter."),
            ]
        )
        pipeline = _pipeline(summarizer)
        a0, a1 = Atom(id="a0", text="X was born in 1900."), Atom(id="a1", text="X lived in Paris.")
        c0 = Context(id="c_a0_0", atom=a0, text="page 0")
        c1 = Context(id="c_a1_0", atom=a1, text="page 1")
        cq = Context(id="c_q_0", atom=None, text="page q")
        a0.add_context(c0)
        a1.add_context(c1)
        pipeline.atoms = {"a0": a0, "a1": a1}
        pipeline.contexts = {c.id: c for c in (c0, c1, cq)}

        _run_summarization(pipeline)

        # One call covering both atoms' contexts and the question's.
        summarizer.run_batch_pairs.assert_awaited_once_with(
            ["page 0", "page 1", "page q"],
            ["X was born in 1900.", "X lived in Paris.", "Who is X?"],
        )
        summarizer.run_batch.assert_not_called()
        assert pipeline.contexts["c_a0_0"].get_summary() == "X was born in 1900."
        assert pipeline.contexts["c_q_0"].get_summary() == "X is a painter."
        assert pipeline.contexts["c_a0_0"].get_probability() == pytest.approx(0.45)

    def test_already_summarized_contexts_are_not_summarized_again(self):
        summarizer = MagicMock()
        summarizer.run_batch_pairs = AsyncMock(return_value=[])
        pipeline = _pipeline(summarizer)
        a0 = Atom(id="a0", text="X was born in 1900.")
        c0 = Context(id="c_a0_0", atom=a0, text="page 0")
        c0.set_synthetic_summary("The page says X was born in 1900.")
        c0.set_probability(0.6)
        a0.add_context(c0)
        pipeline.atoms = {"a0": a0}
        pipeline.contexts = {"c_a0_0": c0}

        _run_summarization(pipeline)

        summarizer.run_batch_pairs.assert_awaited_once_with([], [])
        # Kept as the retriever left it: the prior is not discounted twice.
        assert pipeline.contexts["c_a0_0"].get_probability() == pytest.approx(0.6)

    def test_irrelevant_presummarized_context_is_dropped(self):
        summarizer = MagicMock()
        summarizer.run_batch_pairs = AsyncMock(return_value=[])
        pipeline = _pipeline(summarizer)
        a0 = Atom(id="a0", text="X was born in 1900.")
        c0 = Context(id="c_a0_0", atom=a0, text="page 0")
        c0.set_synthetic_summary(
            "The context does not provide information about the atom."
        )
        a0.add_context(c0)
        pipeline.atoms = {"a0": a0}
        pipeline.contexts = {"c_a0_0": c0}

        _run_summarization(pipeline)

        assert pipeline.contexts == {}
        assert pipeline.atoms["a0"].contexts == {}

    def test_irrelevant_new_summary_drops_context_from_its_atom(self):
        summarizer = MagicMock()
        summarizer.run_batch_pairs = AsyncMock(return_value=[self._summary("")])
        pipeline = _pipeline(summarizer)
        a0 = Atom(id="a0", text="X was born in 1900.")
        c0 = Context(id="c_a0_0", atom=a0, text="page 0")
        a0.add_context(c0)
        pipeline.atoms = {"a0": a0}
        pipeline.contexts = {"c_a0_0": c0}

        _run_summarization(pipeline)

        assert pipeline.contexts == {}
        assert pipeline.atoms["a0"].contexts == {}


class TestRunBatchPairs:
    @staticmethod
    def _output(text):
        out = MagicMock()
        out.success = True
        out.result = MagicMock()
        out.__str__ = lambda self: text
        return out

    def test_each_context_uses_its_own_reference(self):
        from fact_reasoner.core.summarizer import (
            INSTRUCTION_WITH_REF,
            INSTRUCTION_WITHOUT_REF,
            ContextSummarizer,
        )

        seen = []

        async def ainstruct(instruction, **kwargs):
            seen.append((instruction, kwargs["user_variables"]))
            return self._output(f"summary of {kwargs['user_variables']['context']}")

        with patch(
            "fact_reasoner.core.summarizer.mfuncs.ainstruct", side_effect=ainstruct
        ), patch.object(ContextSummarizer, "_get_probability", return_value=0.5):
            summarizer = ContextSummarizer(backend=_Backend())
            results = asyncio.run(
                summarizer.run_batch_pairs(["c1", "c2"], ["atom one", None])
            )

        assert [r["summary"] for r in results] == ["summary of c1", "summary of c2"]
        assert seen[0] == (INSTRUCTION_WITH_REF, {"context": "c1", "atom_text": "atom one"})
        assert seen[1] == (INSTRUCTION_WITHOUT_REF, {"context": "c2", "atom_text": None})


class TestLabelOnlyNLI:
    AACT = "fact_reasoner.core.nli.mfuncs.aact"

    def test_request_is_the_label_only_prompt_with_a_token_cap(self):
        from mellea.backends import ModelOption

        from fact_reasoner.core.nli import (
            LABEL_ONLY_MAX_NEW_TOKENS,
            LabelOnlyNLIExtractor,
        )

        aact = AsyncMock(return_value=MagicMock(success=False))
        with patch(self.AACT, new=aact):
            extractor = LabelOnlyNLIExtractor(_Backend())
            asyncio.run(extractor._request("The {premise}.", "The hypothesis."))

        message = aact.call_args.args[0]
        assert message.role == "user"
        # Braces in the inputs are kept as text, not treated as placeholders.
        assert "PREMISE: The {premise}.\nHYPOTHESIS: The hypothesis.\nANSWER:" in (
            message.content
        )
        assert "no explanation" in message.content
        options = aact.call_args.kwargs["model_options"]
        assert options[ModelOption.MAX_NEW_TOKENS] == LABEL_ONLY_MAX_NEW_TOKENS
        assert options["logprobs"] is True

    def test_bracketed_answer_is_parsed(self):
        from fact_reasoner.core.nli import LabelOnlyNLIExtractor

        # The label is read from the generation itself (output.result).
        output = MagicMock(success=True)
        output.result = MagicMock()
        output.result.__str__ = lambda self: "[contradiction]"
        with patch(self.AACT, new=AsyncMock(return_value=output)), patch.object(
            LabelOnlyNLIExtractor, "_get_probability", return_value=0.9
        ):
            extractor = LabelOnlyNLIExtractor(_Backend())
            result = asyncio.run(extractor.run_batch(["p"], ["h"]))[0]
        assert result["label"] == "contradiction"
        assert result["probability"] == 0.9

    def test_cached_verdicts_are_kept_apart_from_few_shot(self):
        from fact_reasoner.core.nli import LabelOnlyNLIExtractor, NLIExtractor
        from fact_reasoner.core.nli_cache import extractor_identity

        backend = _Backend()
        backend.model_id = "m"
        assert extractor_identity(LabelOnlyNLIExtractor(backend)) != (
            extractor_identity(NLIExtractor(backend))
        )


class TestNLIPromptSelection:
    def test_default_prompt_is_few_shot(self):
        from fact_reasoner.runner import FactualityRunner

        r = FactualityRunner(MagicMock(), pipeline="factscore")
        assert r.nli_prompt == "few_shot"
        assert type(r.nli_extractor).__name__ == "NLIExtractor"

    def test_label_only_prompt_selects_its_extractor(self):
        from fact_reasoner.core.nli import LabelOnlyNLIExtractor
        from fact_reasoner.runner import FactualityRunner

        r = FactualityRunner(MagicMock(), pipeline="factscore", nli_prompt="label_only")
        assert isinstance(r.nli_extractor, LabelOnlyNLIExtractor)

    def test_adapter_backend_wins_over_the_prompt(self):
        from fact_reasoner.core.nli import NLIAdapterExtractor
        from fact_reasoner.runner import FactualityRunner

        r = FactualityRunner(
            MagicMock(),
            pipeline="factscore",
            nli_prompt="label_only",
            nli_backend=MagicMock(),
        )
        assert isinstance(r.nli_extractor, NLIAdapterExtractor)

    def test_unknown_prompt_raises(self):
        from fact_reasoner.runner import FactualityRunner

        with pytest.raises(ValueError, match="Unknown nli_prompt"):
            FactualityRunner(MagicMock(), pipeline="factscore", nli_prompt="bogus")
