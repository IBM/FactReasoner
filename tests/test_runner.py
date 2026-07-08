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

"""Unit tests for fact_reasoner.runner.FactualityRunner (offline)."""

import json
from unittest.mock import MagicMock, patch

import pytest

from fact_reasoner import runner as runner_mod
from fact_reasoner.runner import FactualityRunner, _run_build


class TestRunBuild:
    def test_sync_build_called_directly(self):
        obj = MagicMock()
        obj.build.return_value = None  # sync
        _run_build(obj, has_atoms=True)
        obj.build.assert_called_once_with(has_atoms=True)

    def test_async_build_is_awaited(self):
        # A coroutine build (like FactReasoner) must be run via asyncio.run.
        awaited = {}

        async def _coro(**kwargs):
            awaited["ran"] = kwargs

        obj = MagicMock()
        obj.build.side_effect = lambda **kw: _coro(**kw)
        _run_build(obj, query="q", response="r")
        assert awaited["ran"] == {"query": "q", "response": "r"}


class TestConstruction:
    def test_unknown_pipeline_raises(self):
        with pytest.raises(ValueError, match="Unknown pipeline"):
            FactualityRunner(MagicMock(), pipeline="bogus")

    def test_factreasoner_requires_merlin(self):
        with pytest.raises(ValueError, match="requires a merlin_path"):
            FactualityRunner(MagicMock(), pipeline="factreasoner")

    def test_baseline_needs_no_merlin(self):
        # Should construct fine without merlin_path.
        r = FactualityRunner(MagicMock(), pipeline="factscore")
        assert r.pipeline == "factscore"

    def test_default_nli_method_is_logprobs(self):
        r = FactualityRunner(MagicMock(), pipeline="factscore")
        assert r.nli_method == "logprobs"
        assert r.nli_extractor.method == "logprobs"

    def test_nli_method_threaded_to_extractor(self):
        # The runner must forward nli_method (+ similarity metric) to NLIExtractor.
        captured = {}

        class _FakeNLIExtractor:
            def __init__(self, backend, **kw):
                captured.update(kw)
                self.method = kw.get("nli_method")

        with patch.object(runner_mod, "NLIExtractor", _FakeNLIExtractor):
            FactualityRunner(
                MagicMock(),
                pipeline="factscore",
                nli_method="simbauq",
                nli_similarity_metric="jaccard",
            )

        assert captured["nli_method"] == "simbauq"
        assert captured["simbauq_similarity_metric"] == "jaccard"

    def test_unknown_version_raises(self):
        with pytest.raises(ValueError, match="Unknown pipeline_version"):
            FactualityRunner(MagicMock(), pipeline="factscore", pipeline_version="v9")


class TestContextRetrieverWiring:
    def test_wraps_a_retriever(self):
        # Guards the fix: ContextRetriever must WRAP a SourceRetriever, and the
        # SourceRetriever must carry the service_type/top_k.
        r = FactualityRunner(MagicMock(), pipeline="factscore", service_type="google")
        captured = {}

        class _FakeSourceRetriever:
            def __init__(self, **kw):
                captured["retriever_kwargs"] = kw

        class _FakeContextRetriever:
            def __init__(self, **kw):
                captured["ctx_kwargs"] = kw

        with (
            patch.object(runner_mod, "SourceRetriever", _FakeSourceRetriever),
            patch.object(runner_mod, "ContextRetriever", _FakeContextRetriever),
            patch.object(runner_mod, "QueryBuilder"),
        ):
            r._build_context_retriever()

        assert captured["retriever_kwargs"]["service_type"] == "google"
        assert captured["retriever_kwargs"]["top_k"] == 3
        # ContextRetriever wraps the SourceRetriever instance.
        assert "retriever" in captured["ctx_kwargs"]
        assert isinstance(captured["ctx_kwargs"]["retriever"], _FakeSourceRetriever)


class TestAssessSingle:
    def _runner(self, pipeline="factscore"):
        kwargs = {"pipeline": pipeline}
        if pipeline == "factreasoner":
            kwargs["merlin_path"] = "/fake/merlin"
        return FactualityRunner(MagicMock(), **kwargs)

    def test_assess_builds_with_has_atoms_false_and_returns_dict(self):
        r = self._runner("factscore")
        fake_pipe = MagicMock()
        fake_pipe.build.return_value = None
        fake_pipe.score.return_value = {"factuality_score": 0.5}

        with (
            patch.object(r, "_build_context_retriever", return_value=MagicMock()),
            patch.object(r, "_make_pipeline", return_value=fake_pipe),
        ):
            out = r.assess("q", "resp", topic="t")

        # single mode generates atoms/contexts from scratch.
        _, build_kwargs = fake_pipe.build.call_args
        assert build_kwargs["has_atoms"] is False
        assert build_kwargs["has_contexts"] is False
        assert build_kwargs["revise_atoms"] is True
        assert out == {"factuality_score": 0.5}

    def test_assess_normalizes_factreasoner_tuple(self):
        r = self._runner("factreasoner")
        fake_pipe = MagicMock()
        fake_pipe.build.return_value = None
        # FactReasoner.score returns (results, marginals).
        fake_pipe.score.return_value = ({"factuality_score": 0.9}, [{"variable": "a0"}])

        with (
            patch.object(r, "_build_context_retriever", return_value=MagicMock()),
            patch.object(r, "_make_pipeline", return_value=fake_pipe),
        ):
            out = r.assess("q", "resp")
        assert out == {"factuality_score": 0.9}

    def test_assess_writes_json_when_output_file(self, tmp_path):
        r = self._runner("factscore")
        fake_pipe = MagicMock()
        fake_pipe.build.return_value = None
        fake_pipe.score.return_value = {"factuality_score": 0.7}
        out_file = tmp_path / "res.json"

        with (
            patch.object(r, "_build_context_retriever", return_value=MagicMock()),
            patch.object(r, "_make_pipeline", return_value=fake_pipe),
        ):
            r.assess("q", "resp", output_file=str(out_file))

        assert json.loads(out_file.read_text()) == {"factuality_score": 0.7}


class TestAssessFile:
    def test_file_mode_builds_with_has_atoms_true_and_writes_jsonl(self, tmp_path):
        r = FactualityRunner(MagicMock(), pipeline="factscore")
        data = [{"input": "q1", "output": "o1", "atoms": [], "contexts": []}]
        input_file = tmp_path / "in.jsonl"
        input_file.write_text("\n".join(json.dumps(d) for d in data))
        out_dir = tmp_path / "out"

        fake_pipe = MagicMock()
        fake_pipe.build.return_value = None
        fake_pipe.score.return_value = {"input": "q1", "factuality_score": 0.4}

        with (
            patch.object(r, "_build_context_retriever", return_value=MagicMock()),
            patch.object(r, "_make_pipeline", return_value=fake_pipe),
        ):
            results = r.assess_file(
                str(input_file), str(out_dir), dataset_name="d", model_id="m"
            )

        # file mode uses precomputed atoms/contexts.
        _, build_kwargs = fake_pipe.build.call_args
        assert build_kwargs["has_atoms"] is True
        assert build_kwargs["has_contexts"] is True
        assert len(results) == 1
        assert results[0]["model_name"] == "m"
        # An output jsonl was written.
        written = list(out_dir.glob("*.jsonl"))
        assert len(written) == 1
