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

"""Unit tests for fact_reasoner.lcs.runner.CoherenceRunner (offline).

Most cases stub the collaborators the runner wires together (the miner, the
factuality runner, the coherence pipeline) and assert on the wiring itself, in the
style of ``tests/test_runner.py``. ``TestEndToEndDryRun`` is the exception: it runs
the real runner, a real ``RelationMiner`` and a real ``LCSScorer`` with only the
leaves stubbed (the LLM via ``dry_run_patches``, Merlin via the brute-force
oracle), which is what pins the headline behaviour -- that the atom priors really
are the factuality assessor's posterior marginals.

Nothing here touches a network, an LLM or the Merlin binary.
"""

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from fact_reasoner.assessor import FactReasoner
from fact_reasoner.experiments.mock import (
    MAX_BRUTEFORCE_VARS,
    brute_force_run_merlin,
    dry_run_patches,
)
from fact_reasoner.lcs import runner as lcs_runner_mod
from fact_reasoner.lcs.pipeline import CoherenceResult, MRFCoherenceModel
from fact_reasoner.lcs.priors import (
    FactReasonerPriorProvider,
    PrecomputedPriorProvider,
    UniformPriorProvider,
)
from fact_reasoner.lcs.runner import (
    COHERENCE_PRIOR_SOURCES,
    DEFAULT_BACKEND_KIND,
    CoherenceRunner,
    atom_texts_from_item,
)

MERLIN = "/fake/merlin"


def _runner(**kwargs):
    """A runner on a mock backend, coherence-only unless told otherwise."""
    kwargs.setdefault("merlin_path", MERLIN)
    kwargs.setdefault("prior_source", "none")
    return CoherenceRunner(MagicMock(), **kwargs)


def _result(lcs=0.75):
    """A minimal CoherenceResult, as a stubbed pipeline would return."""
    return CoherenceResult(
        lcs=lcs, method="mean_marginal", scores={"mean_marginal": lcs}
    )


class _StubPipeline:
    """Records how the runner drove a CoherencePipeline."""

    def __init__(self, result=None, prior_provider=None):
        self.result = result or _result()
        self.prior_provider = prior_provider or UniformPriorProvider()
        self.run_calls = []
        self.arun_calls = []
        self.from_mining_calls = []

    def run(self, response, *, query=None, topic=None):
        self.run_calls.append({"response": response, "query": query, "topic": topic})
        return self.result

    async def arun(self, response, *, query=None, topic=None):
        self.arun_calls.append({"response": response, "query": query, "topic": topic})
        return self.result

    def run_from_mining(self, mining, *, priors=None):
        self.from_mining_calls.append({"mining": mining, "priors": priors})
        return self.result


class _StubAtomizer:
    """Returns a fixed atom set (``build_atoms`` only needs ``run(response)``)."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def run(self, response):
        self.calls += 1
        return {str(i): text for i, text in enumerate(self.texts)}


class _ItemRunner:
    """A FactualityRunner stand-in wrapping one real pre-built FactReasoner."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.calls = 0

    def assess_item_with_pipeline(self, item):
        self.calls += 1
        results, _marginals = self.pipeline.score()
        return results, self.pipeline


def _verdicts_by_atom(nli, per_atom, default=None):
    """Drive the mock NLI extractor from the ATOM text (the hypothesis).

    The premise the factuality pipeline builds is a formatted block, so keying on
    it would pin the test to that formatting; the hypothesis is the atom text
    verbatim. Mirrors the helper in ``tests/test_lcs_two_stage_dryrun.py``.
    """
    fallback = default or {"label": "neutral", "probability": 0.9}

    async def run_batch(premises, hypotheses):
        nli.calls.append((list(premises), list(hypotheses)))
        return [dict(per_atom.get(h, fallback)) for h in hypotheses]

    nli.run_batch = run_batch
    return nli


def _build_factuality(json_data, nli, summarizer):
    """A real FactReasoner over a pre-annotated item (no retrieval, no atomizing)."""
    pipeline = FactReasoner(
        nli_extractor=nli,
        context_summarizer=summarizer,
        merlin_path=MERLIN,
        use_priors=True,
    )
    pipeline.from_dict_with_contexts(json_data)
    asyncio.run(
        pipeline.build(
            has_atoms=True,
            has_contexts=True,
            revise_atoms=False,
            summarize_contexts=False,
        )
    )
    return pipeline


class TestAtomTextsFromItem:
    def test_dict_atoms(self):
        item = {"atoms": [{"id": "a0", "text": "One."}, {"id": "a1", "text": "Two."}]}
        assert atom_texts_from_item(item) == ["One.", "Two."]

    def test_bare_string_atoms(self):
        assert atom_texts_from_item({"atoms": ["One.", "Two."]}) == ["One.", "Two."]

    @pytest.mark.parametrize(
        "item", [{}, {"atoms": []}, {"atoms": "nope"}, {"atoms": [{"id": "a0"}]}]
    )
    def test_no_usable_atoms_is_none(self, item):
        # An empty list, a non-list, and text-less atom dicts all mean "atomize".
        assert atom_texts_from_item(item) is None


class TestConstruction:
    def test_requires_merlin_path(self):
        with pytest.raises(ValueError, match="requires a merlin_path"):
            CoherenceRunner(MagicMock(), merlin_path="")

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"prior_source": "nope"}, "Unknown prior_source"),
            ({"formulation": "nope"}, "Unknown formulation"),
            ({"methods": ()}, "at least one LCS readout"),
            ({"methods": ("nope",)}, "Unknown LCS method"),
            ({"pair_policy": "nope"}, "Unknown pair_policy"),
            ({"gate": "nope"}, "Unknown gate"),
            ({"prior_source": "file"}, "requires priors_file"),
        ],
    )
    def test_invalid_arguments_raise(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            CoherenceRunner(MagicMock(), merlin_path=MERLIN, **kwargs)

    def test_file_source_accepts_an_explicit_provider_instead_of_a_path(self):
        # An explicit provider overrides prior_source, so the path is not needed.
        r = _runner(prior_source="file", prior_provider=0.7)
        assert r._build_prior_provider() == 0.7

    def test_prior_source_choices(self):
        assert COHERENCE_PRIOR_SOURCES == ("factreasoner", "none", "file")

    def test_factreasoner_is_the_default_prior_source(self):
        # The headline default: atom marginals come from the assessor.
        r = CoherenceRunner(MagicMock(), merlin_path=MERLIN)
        assert r.prior_source == "factreasoner"

    def test_default_backend_kind_is_rits(self):
        assert DEFAULT_BACKEND_KIND == "rits"

    def test_mining_knobs_reach_the_miner(self):
        captured = {}

        class _FakeMiner:
            def __init__(self, backend, **kw):
                captured.update(kw)

        with patch.object(lcs_runner_mod, "RelationMiner", _FakeMiner):
            _runner(
                nli_method="simbauq",
                pair_policy="all_pairs",
                window=7,
                gate="entity",
                gate_threshold=0.42,
                strength_method="verbalized",
                strength_samples=3,
                concession_discount=0.25,
                show_progress=True,
            )

        assert captured["nli_method"] == "simbauq"
        assert captured["pair_policy"] == "all_pairs"
        assert captured["window"] == 7
        assert captured["gate"] == "entity"
        assert captured["gate_threshold"] == 0.42
        assert captured["strength_method"] == "verbalized"
        assert captured["strength_samples"] == 3
        assert captured["concession_discount"] == 0.25
        assert captured["show_progress"] is True

    def test_atomizer_is_shared_with_the_miner(self):
        r = _runner()
        assert r.miner.atomizer is r.atomizer

    def test_revise_atoms_controls_the_reviser(self):
        assert _runner().reviser is None
        r = _runner(revise_atoms=True)
        assert r.reviser is not None
        assert r.miner.reviser is r.reviser

    def test_methods_are_normalized_to_a_tuple(self):
        r = _runner(methods=["consistency", "mean_marginal"])
        assert r.methods == ("consistency", "mean_marginal")


class TestIboundThreaded:
    def test_ibound_reaches_the_scorer(self):
        # CoherencePipeline builds its default model WITHOUT an ibound, so the
        # runner must build the model itself for --ibound to have any effect.
        r = _runner(ibound=7)
        assert isinstance(r.coherence_model, MRFCoherenceModel)
        assert r.coherence_model.scorer.ibound == 7

    def test_default_ibound(self):
        assert _runner().coherence_model.scorer.ibound == 6

    def test_prebuilt_model_is_passed_to_the_pipeline(self):
        r = _runner(ibound=9)
        pipeline = r._make_pipeline()
        assert pipeline.coherence_model is r.coherence_model
        assert pipeline.coherence_model.scorer.ibound == 9

    def test_formulation_selects_the_model(self):
        assert _runner().coherence_model.formulation == "mrf"
        assert _runner(formulation="mln").coherence_model.formulation == "mln"


class TestFactualityRunnerWiring:
    def _capture(self, **kwargs):
        captured = {}

        class _FakeFactualityRunner:
            instances = 0

            def __init__(self, backend, **kw):
                type(self).instances += 1
                captured.update(kw)
                captured["backend"] = backend

        r = _runner(prior_source="factreasoner", **kwargs)
        with patch("fact_reasoner.runner.FactualityRunner", _FakeFactualityRunner):
            built = r._build_factuality_runner()
        return r, built, captured, _FakeFactualityRunner

    def test_factuality_arguments_are_forwarded(self):
        _r, _built, captured, _cls = self._capture(
            pipeline_version="v3",
            service_type="wikipedia",
            cache_dir="/tmp/cache",
            top_k=9,
            num_workers=2,
            use_summarizer=True,
            use_query_builder=True,
            nli_mode="all_pairs",
            nli_cache_dir="/tmp/nli",
            nli_method="simbauq",
            nli_similarity_metric="jaccard",
        )
        assert captured["pipeline"] == "factreasoner"
        assert captured["pipeline_version"] == "v3"
        assert captured["service_type"] == "wikipedia"
        assert captured["cache_dir"] == "/tmp/cache"
        assert captured["top_k"] == 9
        assert captured["num_workers"] == 2
        assert captured["use_summarizer"] is True
        assert captured["use_query_builder"] is True
        assert captured["nli_mode"] == "all_pairs"
        assert captured["nli_cache_dir"] == "/tmp/nli"
        assert captured["nli_method"] == "simbauq"
        assert captured["nli_similarity_metric"] == "jaccard"
        # The factuality stage exists to produce priors, so it must use them.
        assert captured["use_priors"] is True
        assert captured["merlin_path"] == MERLIN

    def test_backend_is_shared(self):
        r, _built, captured, _cls = self._capture()
        assert captured["backend"] is r.backend

    def test_nli_mode_defaults_to_fast(self):
        # The cheaper preset: this run only has to produce priors.
        _r, _built, captured, _cls = self._capture()
        assert captured["nli_mode"] == "fast"

    def test_is_memoized(self):
        r, built, _captured, cls = self._capture()
        with patch("fact_reasoner.runner.FactualityRunner", cls):
            again = r._build_factuality_runner()
        assert again is built
        assert cls.instances == 1

    def test_injected_runner_is_used_verbatim(self):
        sentinel = MagicMock()
        r = _runner(prior_source="factreasoner", factuality_runner=sentinel)

        class _Boom:
            def __init__(self, *a, **kw):  # pragma: no cover - must not be called
                raise AssertionError("must not build a FactualityRunner")

        with patch("fact_reasoner.runner.FactualityRunner", _Boom):
            assert r._build_factuality_runner() is sentinel


class TestPriorProviderSelection:
    def test_none_is_uniform(self):
        assert _runner(prior_source="none")._build_prior_provider() is None

    def test_file_builds_a_precomputed_provider(self, tmp_path):
        path = tmp_path / "factuality.json"
        path.write_text(json.dumps({"a0": 0.9, "a1": 0.2}))
        provider = _runner(
            prior_source="file", priors_file=str(path)
        )._build_prior_provider()
        assert isinstance(provider, PrecomputedPriorProvider)
        assert provider.priors_for(response="x").priors == {"a0": 0.9, "a1": 0.2}

    def test_factreasoner_uses_assess_mode_for_a_raw_response(self):
        r = _runner(prior_source="factreasoner", factuality_runner=MagicMock())
        provider = r._build_prior_provider()
        assert isinstance(provider, FactReasonerPriorProvider)
        assert provider.mode == "assess"
        assert provider.runner is r._factuality_runner

    def test_factreasoner_uses_file_item_mode_for_a_dataset_item(self):
        # A dataset item already carries contexts, so no retrieval must happen.
        item = {"input": "q", "output": "r", "atoms": [], "contexts": []}
        provider = _runner(
            prior_source="factreasoner", factuality_runner=MagicMock()
        )._build_prior_provider(item)
        assert provider.mode == "file_item"
        assert provider.item is item

    def test_on_degraded_is_threaded(self):
        provider = _runner(
            prior_source="factreasoner",
            factuality_runner=MagicMock(),
            on_degraded="raise",
        )._build_prior_provider()
        assert provider.on_degraded == "raise"

    def test_explicit_provider_short_circuits(self):
        sentinel = UniformPriorProvider(0.3)
        r = _runner(prior_source="factreasoner", prior_provider=sentinel)

        class _Boom:
            def __init__(self, *a, **kw):  # pragma: no cover - must not be called
                raise AssertionError("must not build a FactualityRunner")

        with patch("fact_reasoner.runner.FactualityRunner", _Boom):
            assert r._build_prior_provider() is sentinel
            # ... and it survives being handed to the pipeline.
            assert r._make_pipeline().prior_provider is sentinel


class TestAssessSingle:
    def test_assess_drives_the_pipeline_and_returns_the_result(self):
        r = _runner()
        stub = _StubPipeline(_result(0.61))
        with patch.object(r, "_make_pipeline", return_value=stub):
            out = r.assess("who?", "resp", topic="t")

        assert out.lcs == 0.61
        assert stub.run_calls == [{"response": "resp", "query": "who?", "topic": "t"}]

    def test_assess_with_pipeline_returns_the_pipeline(self):
        r = _runner()
        stub = _StubPipeline()
        with patch.object(r, "_make_pipeline", return_value=stub):
            out, pipeline = r.assess_with_pipeline("q", "resp")
        assert pipeline is stub
        assert out is stub.result

    def test_output_file_gets_the_json_view(self, tmp_path):
        r = _runner()
        path = tmp_path / "out.json"
        stub = _StubPipeline(_result(0.5))
        with patch.object(r, "_make_pipeline", return_value=stub):
            out = r.assess("q", "resp", output_file=str(path))

        assert json.loads(path.read_text()) == out.to_json()

    def test_atom_texts_are_mined_instead_of_atomizing(self):
        r = _runner()
        stub = _StubPipeline()
        with (
            patch.object(r, "_make_pipeline", return_value=stub),
            patch.object(r.miner, "mine_from_atoms", return_value="MINED") as mined,
            patch.object(r.miner, "mine_from_response") as from_response,
        ):
            r.assess("q", "resp", atom_texts=["One.", "Two."])

        mined.assert_called_once_with(["One.", "Two."], "resp")
        from_response.assert_not_called()
        # Scored off the mined graph, not re-run from the response.
        assert stub.run_calls == []
        assert stub.from_mining_calls[0]["mining"] == "MINED"

    async def test_aassess_awaits_arun(self):
        # A plain `async def` stub, not AsyncMock: asyncio_mode="auto" means this
        # test already runs inside a loop (see conftest's mock_nli_batch note).
        r = _runner()
        stub = _StubPipeline(_result(0.44))
        with patch.object(r, "_make_pipeline", return_value=stub):
            out = await r.aassess("q", "resp", topic="t")

        assert out.lcs == 0.44
        assert stub.arun_calls == [{"response": "resp", "query": "q", "topic": "t"}]
        assert stub.run_calls == []

    async def test_aassess_writes_the_output_file(self, tmp_path):
        r = _runner()
        path = tmp_path / "out.json"
        with patch.object(r, "_make_pipeline", return_value=_StubPipeline()):
            out = await r.aassess("q", "resp", output_file=str(path))
        assert json.loads(path.read_text()) == out.to_json()


class TestAssessItem:
    def _item(self, **extra):
        item = {"input": "q", "output": "The response.", "topic": "t"}
        item.update(extra)
        return item

    def test_item_with_atoms_mines_them(self):
        r = _runner()
        stub = _StubPipeline()
        item = self._item(atoms=[{"id": "a0", "text": "One."}])
        with (
            patch.object(r, "_make_pipeline", return_value=stub),
            patch.object(r.miner, "mine_from_atoms", return_value="MINED") as mined,
            patch.object(r.miner, "mine_from_response") as from_response,
        ):
            r.assess_item(item)

        mined.assert_called_once_with(["One."], "The response.")
        from_response.assert_not_called()
        assert stub.from_mining_calls[0]["mining"] == "MINED"
        assert stub.run_calls == []

    def test_item_without_atoms_runs_the_pipeline(self):
        r = _runner()
        stub = _StubPipeline()
        with patch.object(r, "_make_pipeline", return_value=stub):
            r.assess_item(self._item())

        assert stub.run_calls == [
            {"response": "The response.", "query": "q", "topic": "t"}
        ]
        assert stub.from_mining_calls == []

    def test_response_field_is_a_fallback_for_output(self):
        r = _runner()
        stub = _StubPipeline()
        with patch.object(r, "_make_pipeline", return_value=stub):
            r.assess_item({"input": "q", "response": "From response."})
        assert stub.run_calls[0]["response"] == "From response."

    @pytest.mark.parametrize("item", [{}, {"output": ""}, {"output": "   "}])
    def test_item_without_text_raises(self, item):
        with pytest.raises(ValueError, match="no 'output'/'response' text"):
            _runner().assess_item(item)

    def test_the_item_is_passed_to_the_prior_provider(self):
        r = _runner(prior_source="factreasoner", factuality_runner=MagicMock())
        item = self._item(contexts=[])
        with patch.object(
            r, "_build_prior_provider", return_value=None
        ) as build_provider:
            r.assess_item(item)
        # file_item mode is selected from this argument, so no retrieval happens.
        build_provider.assert_called_once_with(item)

    def test_item_with_pipeline_returns_the_pipeline(self):
        r = _runner()
        stub = _StubPipeline()
        with patch.object(r, "_make_pipeline", return_value=stub):
            _out, pipeline = r.assess_item_with_pipeline(self._item())
        assert pipeline is stub


class TestAssessFile:
    def _dataset(self, tmp_path, items, name="data.jsonl"):
        path = tmp_path / name
        path.write_text("".join(f"{json.dumps(i)}\n" for i in items))
        return str(path)

    def _items(self, n=2):
        return [
            {"input": f"q{i}", "output": f"response {i}", "topic": f"t{i}"}
            for i in range(n)
        ]

    def _out_file(self, out_dir):
        files = list(out_dir.iterdir())
        assert len(files) == 1, files
        return files[0]

    def test_writes_one_record_per_item(self, tmp_path):
        r = _runner()
        items = self._items(3)
        out_dir = tmp_path / "results"
        with patch.object(
            r, "_make_pipeline", return_value=_StubPipeline(_result(0.8))
        ):
            data = r.assess_file(
                self._dataset(tmp_path, items),
                str(out_dir),
                dataset_name="demo",
                model_id="m1",
            )

        assert len(data) == 3
        assert [d["input"] for d in data] == ["q0", "q1", "q2"]
        assert [d["output"] for d in data] == ["response 0", "response 1", "response 2"]
        assert [d["topic"] for d in data] == ["t0", "t1", "t2"]
        assert {d["model_name"] for d in data} == {"m1"}
        assert {d["lcs"] for d in data} == {0.8}

        # The file is valid jsonl and matches what was returned.
        lines = self._out_file(out_dir).read_text().splitlines()
        assert [json.loads(line) for line in lines] == data

    def test_output_dir_is_created(self, tmp_path):
        r = _runner()
        out_dir = tmp_path / "nested" / "results"
        with patch.object(r, "_make_pipeline", return_value=_StubPipeline()):
            r.assess_file(self._dataset(tmp_path, self._items(1)), str(out_dir))
        assert out_dir.is_dir()

    def test_filename_records_formulation_prior_source_dataset_and_model(
        self, tmp_path
    ):
        r = _runner(prior_source="none")
        out_dir = tmp_path / "results"
        with patch.object(r, "_make_pipeline", return_value=_StubPipeline()):
            r.assess_file(
                self._dataset(tmp_path, self._items(1)),
                str(out_dir),
                dataset_name="demo",
                model_id="granite",
            )
        assert self._out_file(out_dir).name == "lcs_mrf_none_demo_granite.jsonl"

    def test_prior_source_separates_an_ablation(self, tmp_path):
        # Two prior sources over one dataset must not overwrite each other.
        out_dir = tmp_path / "results"
        dataset = self._dataset(tmp_path, self._items(1))
        for source in ("none", "factreasoner"):
            r = _runner(prior_source=source, factuality_runner=MagicMock())
            with patch.object(r, "_make_pipeline", return_value=_StubPipeline()):
                r.assess_file(dataset, str(out_dir), dataset_name="d", model_id="m")

        assert sorted(p.name for p in out_dir.iterdir()) == [
            "lcs_mrf_factreasoner_d_m.jsonl",
            "lcs_mrf_none_d_m.jsonl",
        ]

    def test_resume_skips_processed_inputs_and_keeps_them(self, tmp_path):
        r = _runner()
        items = self._items(3)
        out_dir = tmp_path / "results"
        # Two distinct files: writing both through one path would let the second
        # _dataset call clobber the first pass's input before it ran.
        first_pass = self._dataset(tmp_path, items[:1], name="first.jsonl")
        dataset = self._dataset(tmp_path, items, name="all.jsonl")

        # First pass over the first item only.
        with patch.object(
            r, "_make_pipeline", return_value=_StubPipeline(_result(0.1))
        ):
            r.assess_file(first_pass, str(out_dir), dataset_name="d", model_id="m")

        # Second pass over all three: only the two new ones are scored.
        stub = _StubPipeline(_result(0.9))
        with patch.object(r, "_make_pipeline", return_value=stub):
            data = r.assess_file(dataset, str(out_dir), dataset_name="d", model_id="m")

        assert len(stub.run_calls) == 2
        assert [c["response"] for c in stub.run_calls] == ["response 1", "response 2"]
        assert [d["input"] for d in data] == ["q0", "q1", "q2"]
        # The pre-existing record keeps its original score.
        assert data[0]["lcs"] == 0.1
        assert [d["lcs"] for d in data[1:]] == [0.9, 0.9]

    def test_rerun_of_a_finished_sweep_does_nothing(self, tmp_path):
        r = _runner()
        out_dir = tmp_path / "results"
        dataset = self._dataset(tmp_path, self._items(2))
        with patch.object(r, "_make_pipeline", return_value=_StubPipeline()):
            first = r.assess_file(dataset, str(out_dir), dataset_name="d", model_id="m")

        stub = _StubPipeline()
        with patch.object(r, "_make_pipeline", return_value=stub):
            second = r.assess_file(
                dataset, str(out_dir), dataset_name="d", model_id="m"
            )

        assert stub.run_calls == []
        assert second == first

    def test_blank_lines_are_ignored(self, tmp_path):
        r = _runner()
        path = tmp_path / "data.jsonl"
        path.write_text(f"{json.dumps(self._items(1)[0])}\n\n")
        out_dir = tmp_path / "results"
        with patch.object(r, "_make_pipeline", return_value=_StubPipeline()):
            data = r.assess_file(str(path), str(out_dir))
        assert len(data) == 1

    def test_records_are_json_serializable(self, tmp_path):
        r = _runner()
        out_dir = tmp_path / "results"
        with patch.object(r, "_make_pipeline", return_value=_StubPipeline()):
            data = r.assess_file(self._dataset(tmp_path, self._items(1)), str(out_dir))
        json.dumps(data)


class TestFromBackendKind:
    def test_defaults_to_rits(self):
        with (
            patch.object(lcs_runner_mod, "CoherenceRunner", CoherenceRunner),
            patch("fact_reasoner.backends.build_backend") as build,
        ):
            build.return_value = MagicMock()
            r = CoherenceRunner.from_backend_kind(merlin_path=MERLIN)

        assert build.call_args[0][0] == "rits"
        assert isinstance(r, CoherenceRunner)

    def test_backend_arguments_are_forwarded(self):
        with patch("fact_reasoner.backends.build_backend") as build:
            build.return_value = MagicMock()
            CoherenceRunner.from_backend_kind(
                "vllm",
                model_id="granite4",
                base_url="http://localhost:8000/v1",
                api_key="k",
                model_options={"x": 1},
                merlin_path=MERLIN,
            )

        assert build.call_args[0][0] == "vllm"
        assert build.call_args.kwargs == {
            "model_id": "granite4",
            "base_url": "http://localhost:8000/v1",
            "api_key": "k",
            "model_options": {"x": 1},
        }

    @pytest.mark.parametrize("kind", ["rits", "ollama", "vllm", "openai"])
    def test_every_backend_kind_is_accepted(self, kind):
        with patch("fact_reasoner.backends.build_backend") as build:
            backend = MagicMock()
            build.return_value = backend
            r = CoherenceRunner.from_backend_kind(
                kind, merlin_path=MERLIN, nli_method="simbauq"
            )
        assert r.backend is backend

    def test_runner_kwargs_are_forwarded(self):
        with patch("fact_reasoner.backends.build_backend") as build:
            build.return_value = MagicMock()
            r = CoherenceRunner.from_backend_kind(
                merlin_path=MERLIN, prior_source="none", ibound=8
            )
        assert r.prior_source == "none"
        assert r.coherence_model.scorer.ibound == 8


class TestEndToEndDryRun:
    """The real runner, miner and scorer; only the LLM and Merlin are stubbed.

    The factuality network counts contexts as well as atoms, and the brute-force
    oracle refuses networks above ``MAX_BRUTEFORCE_VARS`` (20) variables, so the
    fixture stays tiny (2 atoms, 2 contexts). Do not grow it.
    """

    def _primed_runner(self, json_data, nli, summarizer, **kwargs):
        """A runner whose priors come from a real FactReasoner over the item."""
        factuality = _build_factuality(json_data, nli, summarizer)
        assert len(factuality.markov_network.nodes) <= MAX_BRUTEFORCE_VARS
        runner = CoherenceRunner(
            nli.backend,
            merlin_path=MERLIN,
            prior_source="factreasoner",
            factuality_runner=_ItemRunner(factuality),
            pair_policy="all_pairs",
            strength_method="verbalized",
            methods=("mean_marginal", "consistency"),
            **kwargs,
        )
        return runner, factuality

    def test_atom_priors_are_the_assessor_marginals(
        self, sample_json_data, mock_nli_batch, mock_summarizer, monkeypatch
    ):
        """The headline behaviour: stage 1's posteriors are stage 2's unary priors."""
        atoms = {a["id"]: a["text"] for a in sample_json_data["atoms"]}
        # Make the two atoms differ factually: a0 supported, a1 contradicted.
        _verdicts_by_atom(
            mock_nli_batch,
            {
                atoms["a0"]: {"label": "entailment", "probability": 0.95},
                atoms["a1"]: {"label": "contradiction", "probability": 0.95},
            },
        )
        monkeypatch.setattr(
            "fact_reasoner.assessor._run_merlin_shared", brute_force_run_merlin
        )

        with dry_run_patches():
            runner, _factuality = self._primed_runner(
                sample_json_data, mock_nli_batch, mock_summarizer
            )
            out = runner.assess_item(sample_json_data)

        # Real, differentiated posteriors -- not a flat 0.5.
        assert out.prior_coverage["source"] == "factreasoner"
        assert set(out.priors) == {"a0", "a1"}
        assert out.priors["a0"] > 0.5 > out.priors["a1"]
        assert set(out.priors.values()) != {0.5}
        # ... aligned onto the mined atoms without falling back to ids.
        assert out.prior_coverage["coverage"] == pytest.approx(1.0)
        assert out.prior_coverage["alignment"] in ("identity", "text")

        assert 0.0 <= out.lcs <= 1.0
        assert set(out.scores) == {"mean_marginal", "consistency"}
        # A joint model: coherence moved the atoms off their priors.
        assert out.marginals != pytest.approx(out.priors, abs=1e-9)
        # Stage-1 diagnostics are carried through.
        assert out.factuality and "factuality_score" in out.factuality
        json.dumps(out.to_json())

    def test_primed_and_coherence_only_runs_differ(
        self, sample_json_data, mock_nli_batch, mock_summarizer, monkeypatch
    ):
        """The priors must change the score, not merely be carried along."""
        atoms = {a["id"]: a["text"] for a in sample_json_data["atoms"]}
        _verdicts_by_atom(
            mock_nli_batch,
            {
                atoms["a0"]: {"label": "contradiction", "probability": 0.97},
                atoms["a1"]: {"label": "contradiction", "probability": 0.97},
            },
        )
        monkeypatch.setattr(
            "fact_reasoner.assessor._run_merlin_shared", brute_force_run_merlin
        )
        atom_texts = [a["text"] for a in sample_json_data["atoms"]]

        with dry_run_patches():
            primed, _factuality = self._primed_runner(
                sample_json_data, mock_nli_batch, mock_summarizer
            )
            primed_out = primed.assess_item(sample_json_data)

            # The coherence-only runner mines the same atoms, so the two runs
            # differ ONLY in their priors.
            plain = CoherenceRunner(
                mock_nli_batch.backend,
                merlin_path=MERLIN,
                prior_source="none",
                pair_policy="all_pairs",
                strength_method="verbalized",
            )
            plain_out = plain.assess(
                "q", sample_json_data["output"], atom_texts=atom_texts
            )

        assert set(plain_out.priors.values()) == {0.5}
        # Both atoms are factually contradicted, so priming pulls the score down.
        assert primed_out.lcs < plain_out.lcs

    def test_file_priors_replay_a_live_run(
        self, sample_json_data, mock_nli_batch, mock_summarizer, monkeypatch, tmp_path
    ):
        """prior_source='file' reproduces the live two-stage score, with no LLM."""
        monkeypatch.setattr(
            "fact_reasoner.assessor._run_merlin_shared", brute_force_run_merlin
        )
        factuality = _build_factuality(
            sample_json_data, mock_nli_batch, mock_summarizer
        )
        results, _marginals = factuality.score()
        path = tmp_path / "factuality.json"
        path.write_text(
            json.dumps(
                {
                    "factuality_score_per_atom": results["factuality_score_per_atom"],
                    "atoms": [
                        {"id": aid, "text": atom.get_text()}
                        for aid, atom in factuality.atoms.items()
                    ],
                }
            )
        )

        with dry_run_patches():
            live = CoherenceRunner(
                mock_nli_batch.backend,
                merlin_path=MERLIN,
                prior_source="factreasoner",
                factuality_runner=_ItemRunner(factuality),
                pair_policy="all_pairs",
                strength_method="verbalized",
            ).assess_item(sample_json_data)

            replayed = CoherenceRunner(
                mock_nli_batch.backend,
                merlin_path=MERLIN,
                prior_source="file",
                priors_file=str(path),
                pair_policy="all_pairs",
                strength_method="verbalized",
            ).assess_item(sample_json_data)

        assert replayed.lcs == pytest.approx(live.lcs, abs=1e-12)
        assert replayed.prior_coverage["source"] == "file"

    def test_assess_file_end_to_end(
        self, sample_json_data, mock_nli_batch, mock_summarizer, monkeypatch, tmp_path
    ):
        """A dataset sweep with real scoring, then a resume that adds nothing."""
        monkeypatch.setattr(
            "fact_reasoner.assessor._run_merlin_shared", brute_force_run_merlin
        )
        dataset = tmp_path / "data.jsonl"
        dataset.write_text(f"{json.dumps(sample_json_data)}\n")
        out_dir = tmp_path / "results"

        with dry_run_patches():
            runner, _factuality = self._primed_runner(
                sample_json_data, mock_nli_batch, mock_summarizer
            )
            data = runner.assess_file(
                str(dataset), str(out_dir), dataset_name="demo", model_id="mock"
            )
            resumed = runner.assess_file(
                str(dataset), str(out_dir), dataset_name="demo", model_id="mock"
            )

        assert len(data) == 1
        assert 0.0 <= data[0]["lcs"] <= 1.0
        assert data[0]["input"] == sample_json_data["input"]
        assert data[0]["model_name"] == "mock"
        assert data[0]["priors"] and set(data[0]["priors"].values()) != {0.5}
        # Resume is a no-op over a finished sweep.
        assert resumed == data

    def _coherence_only_runner(self, backend):
        """A coherence-only runner over a stubbed atomizer, for sync/async parity."""
        runner = CoherenceRunner(
            backend,
            merlin_path=MERLIN,
            prior_source="none",
            pair_policy="all_pairs",
            strength_method="verbalized",
        )
        return runner

    def test_sync_assess_end_to_end(
        self, sample_json_data, mock_nli_batch, mock_summarizer
    ):
        """The sync path mines and scores a raw response through the real miner."""
        atom_texts = [a["text"] for a in sample_json_data["atoms"]]
        with dry_run_patches():
            runner = self._coherence_only_runner(mock_nli_batch.backend)
            out = runner.assess("q", sample_json_data["output"], atom_texts=atom_texts)

        assert 0.0 <= out.lcs <= 1.0
        assert set(out.priors.values()) == {0.5}
        assert set(out.priors) == set(out.mining.atoms)

    async def test_aassess_end_to_end(
        self, sample_json_data, mock_nli_batch, mock_summarizer
    ):
        """``arun`` mines concurrently and reaches the same atom set.

        Note the sync equivalent cannot be called from here: ``mine_from_atoms``
        drives the miner with ``asyncio.run``, which refuses to nest inside the
        loop ``asyncio_mode="auto"`` already provides. The sync path is covered by
        :meth:`test_sync_assess_end_to_end`. The atoms are supplied explicitly
        because the real ``Atomizer`` cannot run against a bare mock backend.
        """
        atom_texts = [a["text"] for a in sample_json_data["atoms"]]
        with dry_run_patches():
            runner = self._coherence_only_runner(mock_nli_batch.backend)
            mining = await runner.miner.amine_from_atoms(
                atom_texts, sample_json_data["output"]
            )
            out = await runner.aassess(
                "q", sample_json_data["output"], atom_texts=atom_texts
            )

        assert 0.0 <= out.lcs <= 1.0
        assert set(out.priors.values()) == {0.5}
        # The async miner reached the same atom set the sync one would.
        assert set(out.mining.atoms) == set(mining.atoms)
