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

"""Unit tests for fact_reasoner.core.nli module."""

import asyncio
import math
import pytest
from unittest.mock import MagicMock, patch
from fact_reasoner.core.nli import NLIExtractor, INSTRUCTION_NLI


class TestNLIExtractorInit:
    """Tests for NLIExtractor initialization."""

    def test_nli_extractor_none_backend_raises(self):
        with pytest.raises(ValueError, match="Mellea backend is None"):
            NLIExtractor(backend=None)

    def test_nli_extractor_stores_backend(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        nli = NLIExtractor(backend=mock_backend)
        assert nli.backend == mock_backend

    def test_nli_extractor_default_method(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        nli = NLIExtractor(backend=mock_backend)
        assert nli.method == "logprobs"

    def test_nli_extractor_logprobs_builds_rejection_strategy(self):
        from mellea.stdlib.sampling import RejectionSamplingStrategy

        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        nli = NLIExtractor(backend=mock_backend, nli_method="logprobs")
        assert isinstance(nli._strategy, RejectionSamplingStrategy)
        assert nli._logprobs_model_options() == {
            "logprobs": True,
            "top_logprobs": 5,
        }

    def test_nli_extractor_simbauq_builds_simbauq_strategy(self):
        from fact_reasoner.uncertainty import SIMBAUQSamplingStrategy

        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        nli = NLIExtractor(
            backend=mock_backend,
            nli_method="simbauq",
            simbauq_similarity_metric="jaccard",
        )
        assert nli.method == "simbauq"
        assert isinstance(nli._strategy, SIMBAUQSamplingStrategy)
        # SIMBA-UQ must NOT request logprobs (Ollama rejects the option).
        assert nli._logprobs_model_options() is None

    def test_nli_extractor_unknown_method_raises(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        with pytest.raises(ValueError, match="Unknown nli_method"):
            NLIExtractor(backend=mock_backend, nli_method="bogus")

    def test_show_progress_defaults_false(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"
        nli = NLIExtractor(backend=mock_backend)
        assert nli.show_progress is False

    def test_show_progress_stored(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"
        nli = NLIExtractor(backend=mock_backend, show_progress=True)
        assert nli.show_progress is True


class TestNLIExtractorClassifierPath:
    """Tests for loading a trained SIMBA-UQ classifier from a path."""

    @staticmethod
    def _train_and_save(tmp_path, temperatures, n_per_temp):
        """Train a tiny classifier and save it; return its path."""
        import json

        from fact_reasoner.uncertainty import (
            save_classifier,
            train_classifier_from_jsonl,
        )

        n = len(temperatures) * n_per_temp
        samp = tmp_path / "s.jsonl"
        lines = []
        for g in range(6):
            samples = [f"[entailment] a{g}", "[entailment] b"]
            samples += ["[neutral] c", "[contradiction] d"]
            samples = (samples * n)[:n]  # pad/truncate to exactly n
            labels = [1 if "[entailment]" in s else 0 for s in samples]
            lines.append(
                json.dumps(
                    {
                        "premise": "p",
                        "hypothesis": "h",
                        "gold": "entailment",
                        "samples": samples,
                        "labels": labels,
                    }
                )
            )
        samp.write_text("\n".join(lines) + "\n")
        clf, meta = train_classifier_from_jsonl(
            str(samp),
            temperatures=temperatures,
            n_per_temp=n_per_temp,
            similarity_metric="jaccard",
        )
        out = tmp_path / "clf.joblib"
        save_classifier(clf, str(out), meta)
        return str(out)

    def test_loads_classifier_and_sets_method(self, tmp_path):
        pytest.importorskip("sklearn")
        pytest.importorskip("joblib")
        temps, n_per_temp = [0.5, 0.7], 2
        clf_path = self._train_and_save(tmp_path, temps, n_per_temp)

        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"
        nli = NLIExtractor(
            backend=mock_backend,
            nli_method="simbauq",
            simbauq_temperatures=temps,
            simbauq_n_per_temp=n_per_temp,
            simbauq_similarity_metric="jaccard",
            simbauq_confidence_method="classifier",
            simbauq_classifier_path=clf_path,
        )
        assert nli._strategy.confidence_method == "classifier"
        assert nli._strategy._classifier is not None
        assert nli._classifier_path == clf_path

    def test_feature_dim_mismatch_raises(self, tmp_path):
        pytest.importorskip("sklearn")
        pytest.importorskip("joblib")
        # Train with N=4 (2*2), then load into a config expecting N=16 (4*4).
        clf_path = self._train_and_save(tmp_path, [0.5, 0.7], 2)

        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"
        with pytest.raises(ValueError, match="expects .* features"):
            NLIExtractor(
                backend=mock_backend,
                nli_method="simbauq",
                simbauq_temperatures=[0.3, 0.5, 0.7, 1.0],
                simbauq_n_per_temp=4,
                simbauq_classifier_path=clf_path,
            )

    def test_explicit_classifier_object_takes_precedence(self, tmp_path):
        # When both an object and a path could apply, the object wins and the
        # path is not recorded (no load from disk).
        pytest.importorskip("sklearn")
        import numpy as np

        class DummyClf:
            n_features_in_ = 3

            def predict_proba(self, X):
                return np.tile([0.5, 0.5], (len(X), 1))

        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"
        nli = NLIExtractor(
            backend=mock_backend,
            nli_method="simbauq",
            simbauq_temperatures=[0.5, 0.7],
            simbauq_n_per_temp=2,
            simbauq_confidence_method="classifier",
            simbauq_classifier=DummyClf(),
        )
        assert nli._classifier_path is None
        assert isinstance(nli._strategy._classifier, DummyClf)


class TestNLIInstruction:
    """Tests for NLI instruction template."""

    def test_instruction_contains_examples(self):
        assert "Example 1:" in INSTRUCTION_NLI
        assert "Example 2:" in INSTRUCTION_NLI
        assert "Example 3:" in INSTRUCTION_NLI

    def test_instruction_contains_labels(self):
        assert "[entailment]" in INSTRUCTION_NLI
        assert "[contradiction]" in INSTRUCTION_NLI
        assert "[neutral]" in INSTRUCTION_NLI

    def test_instruction_contains_placeholders(self):
        assert "{{premise_text}}" in INSTRUCTION_NLI
        assert "{{hypothesis_text}}" in INSTRUCTION_NLI

    def test_instruction_contains_steps(self):
        assert "1. Evaluate Relationship:" in INSTRUCTION_NLI
        assert "2. Provide the reasoning" in INSTRUCTION_NLI
        assert "3. Final Answer:" in INSTRUCTION_NLI


class TestNLIExtractorGetLabel:
    """Tests for NLIExtractor._get_label method."""

    def test_get_label_entailment(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        nli = NLIExtractor(backend=mock_backend)

        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "The answer is [entailment]"

        result = nli._get_label(mock_output)
        assert result == "entailment"

    def test_get_label_contradiction(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        nli = NLIExtractor(backend=mock_backend)

        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "Based on evidence [contradiction]"

        result = nli._get_label(mock_output)
        assert result == "contradiction"

    def test_get_label_neutral(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        nli = NLIExtractor(backend=mock_backend)

        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "Cannot determine [neutral]"

        result = nli._get_label(mock_output)
        assert result == "neutral"

    def test_get_label_multiple_brackets(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        nli = NLIExtractor(backend=mock_backend)

        mock_output = MagicMock()
        mock_output.__str__ = lambda self: "[first] and [entailment]"

        result = nli._get_label(mock_output)
        assert result == "entailment"  # Should get the last one

    def test_get_label_json(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"
        nli = NLIExtractor(backend=mock_backend)

        mock_output = MagicMock()
        mock_output.__str__ = lambda self: '{"label": "contradiction"}'
        assert nli._get_label(mock_output) == "contradiction"

    def test_get_label_json_in_prose(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"
        nli = NLIExtractor(backend=mock_backend)

        mock_output = MagicMock()
        mock_output.__str__ = lambda self: 'After reasoning:\n{"label": "Entailment"}'
        assert nli._get_label(mock_output) == "entailment"  # lower-cased


class TestNLIExtractorGetProbability:
    """Tests for NLIExtractor._get_probability (token-span alignment)."""

    @staticmethod
    def _output(content):
        """Wrap a list of {token, logprob} as a mellea OpenAI-style output.

        Real OpenAI/vLLM ``content`` logprob arrays contain only emitted content
        tokens (no trailing EOS element), so these fixtures do not add one.
        """
        out = MagicMock()
        out._meta = {"oai_chat_response": {"choices": [{"logprobs": {"content": content}}]}}
        return out

    def _prob(self, content):
        nli = NLIExtractor(backend=self._backend())
        return nli._get_probability(self._output(content))

    @staticmethod
    def _backend():
        b = MagicMock()
        b.model_id = "test-model"
        return b

    def test_standalone_brackets(self):
        # exp(mean) over the tokens covering the label INTERIOR ("entail"); the
        # standalone bracket tokens fall outside the interior span.
        content = [
            {"token": "[", "logprob": -0.1},
            {"token": "ent", "logprob": -0.5},
            {"token": "ail", "logprob": -0.3},
            {"token": "]", "logprob": -0.1},
        ]
        expected = math.exp((-0.5 - 0.3) / 2)
        assert self._prob(content) == pytest.approx(expected)

    def test_fused_whole_label_token_no_eos(self):
        # The whole "[neutral]" is one fused token — the old bracket-walk broke
        # here (and the [:-1] EOS drop would have deleted it). Now robust.
        content = [
            {"token": "Final", "logprob": -0.1},
            {"token": " [neutral]", "logprob": -0.02},
        ]
        # Only the fused label token overlaps the span.
        assert self._prob(content) == pytest.approx(math.exp(-0.02))

    def test_last_span_wins_over_citation(self):
        # A citation "[1]" earlier in the reasoning must not be picked; the LAST
        # [...] label interior is measured (just the "contradiction" token).
        content = [
            {"token": "see [1] ", "logprob": -0.3},
            {"token": "[", "logprob": -0.2},
            {"token": "contradiction", "logprob": -0.03},
            {"token": "]", "logprob": -0.01},
        ]
        assert self._prob(content) == pytest.approx(math.exp(-0.03))

    def test_trailing_text_after_label(self):
        content = [
            {"token": "[neutral]", "logprob": -0.02},
            {"token": " is my final answer", "logprob": -0.5},
        ]
        assert self._prob(content) == pytest.approx(math.exp(-0.02))

    def test_close_bracket_last_token_not_dropped(self):
        # Regression: previously extract_logprobs did [:-1] and this confident
        # label collapsed to probability 0.0. Now it is a real value. The label
        # interior ("neutral") lives entirely in the fused "neutral]" token.
        content = [
            {"token": "[", "logprob": -0.2},
            {"token": "neutral]", "logprob": -0.03},
        ]
        result = self._prob(content)
        assert result > 0.0
        assert result == pytest.approx(math.exp(-0.03))

    def test_bare_word_label_probability(self):
        # No brackets/JSON: the bare NLI word is located (matching _get_label's
        # bare-word fallback), so the probability is a real value, not 0.5.
        content = [{"token": "entailment", "logprob": -0.1}]
        assert self._prob(content) == pytest.approx(math.exp(-0.1))

    def test_no_label_span_returns_unknown_default(self):
        # No JSON / bracket / bare-word label at all -> conservative 0.5.
        content = [{"token": "undecided", "logprob": -0.1}]
        assert self._prob(content) == NLIExtractor._UNKNOWN_PROBABILITY

    def test_empty_logprobs_returns_unknown_default(self):
        assert self._prob([]) == NLIExtractor._UNKNOWN_PROBABILITY

    def test_probability_in_unit_interval(self):
        content = [
            {"token": "[", "logprob": -0.1},
            {"token": "entailment", "logprob": -0.5},
            {"token": "]", "logprob": -0.1},
        ]
        result = self._prob(content)
        assert 0.0 < result <= 1.0

    def test_json_probability_over_value_tokens(self):
        # For JSON output the probability must be measured over the label VALUE
        # tokens ("entail","ment"), not the JSON boilerplate.
        content = [
            {"token": '{"label": "', "logprob": -0.01},
            {"token": "entail", "logprob": -0.05},
            {"token": "ment", "logprob": -0.03},
            {"token": '"}', "logprob": -0.01},
        ]
        expected = math.exp((-0.05 - 0.03) / 2)
        assert self._prob(content) == pytest.approx(expected)

    def test_json_fenced_whole_object_token(self):
        # Whole JSON object as one fused token overlaps the value span.
        content = [
            {"token": "```json\n", "logprob": -0.01},
            {"token": '{"label": "neutral"}', "logprob": -0.04},
            {"token": "\n```", "logprob": -0.01},
        ]
        assert self._prob(content) == pytest.approx(math.exp(-0.04))


class TestNLIExtractorRun:
    """Tests for NLIExtractor.run method."""

    def test_run_returns_dict(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        mock_result = MagicMock()
        mock_result.__str__ = lambda self: "[entailment]"
        mock_result._meta = {
            "oai_chat_response": {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {"token": "[", "logprob": -0.1},
                                {"token": "ent", "logprob": -0.2},
                                {"token": "]", "logprob": -0.1},
                                {"token": "<eos>", "logprob": -0.1},
                            ]
                        }
                    }
                ]
            }
        }

        mock_output = MagicMock()
        mock_output.success = True
        mock_output.result = mock_result

        with patch(
            "src.fact_reasoner.core.nli.mfuncs.instruct", return_value=mock_output
        ):
            nli = NLIExtractor(backend=mock_backend)
            result = nli.run(
                premise="The sky is blue.", hypothesis="The sky has color."
            )

            assert isinstance(result, dict)
            assert "label" in result
            assert "probability" in result
            assert result["label"] == "entailment"

    def test_run_json_output(self):
        # A JSON verdict must parse to the right label and a probability over the
        # label value tokens.
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        mock_result = MagicMock()
        mock_result.__str__ = lambda self: '{"label": "contradiction"}'
        mock_result._meta = {
            "oai_chat_response": {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {"token": '{"label": "', "logprob": -0.01},
                                {"token": "contradiction", "logprob": -0.05},
                                {"token": '"}', "logprob": -0.01},
                            ]
                        }
                    }
                ]
            }
        }

        mock_output = MagicMock()
        mock_output.success = True
        mock_output.result = mock_result

        with patch(
            "src.fact_reasoner.core.nli.mfuncs.instruct", return_value=mock_output
        ):
            nli = NLIExtractor(backend=mock_backend)
            result = nli.run(premise="p", hypothesis="h")

        assert result["label"] == "contradiction"
        assert result["probability"] == pytest.approx(math.exp(-0.05))

    def test_run_returns_neutral_on_failure(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        mock_output = MagicMock()
        mock_output.success = False

        with patch(
            "src.fact_reasoner.core.nli.mfuncs.instruct", return_value=mock_output
        ):
            nli = NLIExtractor(backend=mock_backend)
            result = nli.run(premise="Test premise", hypothesis="Test hypothesis")

            assert result["label"] == "neutral"
            assert result["probability"] == 1.0

    def test_run_returns_neutral_on_generation_exception(self):
        """A backend/network error during generation must not crash run()."""
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        with patch(
            "src.fact_reasoner.core.nli.mfuncs.instruct",
            side_effect=RuntimeError("backend exploded"),
        ):
            nli = NLIExtractor(backend=mock_backend)
            result = nli.run(premise="p", hypothesis="h")

            assert result["label"] == "neutral"
            assert result["probability"] == 1.0


class TestNLIExtractorRunBatch:
    """Tests for NLIExtractor.run_batch throttling and failure resilience."""

    @staticmethod
    def _mk_output(success: bool):
        out = MagicMock()
        out.success = success
        out.result = MagicMock()
        return out

    def test_run_batch_returns_aligned_results(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        labels = ["entailment", "contradiction"]
        outputs = [self._mk_output(True), self._mk_output(True)]

        async def fake_ainstruct(*args, **kwargs):
            return outputs.pop(0)

        with patch(
            "src.fact_reasoner.core.nli.mfuncs.ainstruct", side_effect=fake_ainstruct
        ):
            with patch.object(NLIExtractor, "_get_label", side_effect=labels):
                with patch.object(NLIExtractor, "_get_probability", return_value=0.9):
                    nli = NLIExtractor(backend=mock_backend)
                    results = asyncio.run(nli.run_batch(["p1", "p2"], ["h1", "h2"]))

        assert [r["label"] for r in results] == ["entailment", "contradiction"]

    def test_run_batch_single_failure_does_not_drop_others(self):
        """One raised call maps to neutral; results stay length/order aligned."""
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        good = self._mk_output(True)

        async def fake_ainstruct(*args, **kwargs):
            if kwargs["user_variables"]["premise_text"] == "bad":
                raise RuntimeError("boom")
            return good

        with patch(
            "src.fact_reasoner.core.nli.mfuncs.ainstruct", side_effect=fake_ainstruct
        ):
            with patch.object(NLIExtractor, "_get_label", return_value="entailment"):
                with patch.object(NLIExtractor, "_get_probability", return_value=0.9):
                    nli = NLIExtractor(backend=mock_backend)
                    results = asyncio.run(
                        nli.run_batch(["ok1", "bad", "ok2"], ["h1", "h2", "h3"])
                    )

        assert len(results) == 3
        assert results[0]["label"] == "entailment"
        assert results[1] == {"label": "neutral", "probability": 1.0}
        assert results[2]["label"] == "entailment"

    def test_run_batch_progress_bar_updates_per_pair(self):
        """With show_progress=True the tqdm bar advances once per pair."""
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        async def fake_ainstruct(*args, **kwargs):
            return self._mk_output(True)

        bar = MagicMock()
        # `from tqdm import tqdm` inside run_batch resolves to tqdm.tqdm.
        with patch("tqdm.tqdm", return_value=bar) as tqdm_ctor:
            with patch(
                "src.fact_reasoner.core.nli.mfuncs.ainstruct",
                side_effect=fake_ainstruct,
            ):
                with patch.object(NLIExtractor, "_get_label", return_value="neutral"):
                    with patch.object(
                        NLIExtractor, "_get_probability", return_value=0.5
                    ):
                        nli = NLIExtractor(backend=mock_backend, show_progress=True)
                        results = asyncio.run(
                            nli.run_batch(["p1", "p2", "p3"], ["h1", "h2", "h3"])
                        )

        assert len(results) == 3
        tqdm_ctor.assert_called_once()
        assert tqdm_ctor.call_args.kwargs["total"] == 3
        assert bar.update.call_count == 3  # one tick per completed pair
        bar.close.assert_called_once()

    def test_run_batch_no_bar_when_progress_disabled(self):
        """Default (show_progress=False) constructs no tqdm bar."""
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        async def fake_ainstruct(*args, **kwargs):
            return self._mk_output(True)

        with patch("tqdm.tqdm") as tqdm_ctor:
            with patch(
                "src.fact_reasoner.core.nli.mfuncs.ainstruct",
                side_effect=fake_ainstruct,
            ):
                with patch.object(NLIExtractor, "_get_label", return_value="neutral"):
                    with patch.object(
                        NLIExtractor, "_get_probability", return_value=0.5
                    ):
                        nli = NLIExtractor(backend=mock_backend)
                        asyncio.run(nli.run_batch(["p1"], ["h1"]))

        tqdm_ctor.assert_not_called()


class TestNLIExtractorSimbauqParse:
    """Tests for the SIMBA-UQ probability path in _parse_output."""

    @staticmethod
    def _mk_simbauq_output(text: str, confidence):
        """Build a fake successful sampling result carrying SIMBA-UQ metadata."""
        result = MagicMock()
        result.__str__ = lambda self: text
        result._meta = {"simba_uq": {"confidence": confidence}}
        output = MagicMock()
        output.success = True
        output.result = result
        return output

    def _nli(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"
        return NLIExtractor(backend=mock_backend, nli_method="simbauq")

    def test_confidence_becomes_label_probability(self):
        nli = self._nli()
        out = self._mk_simbauq_output("Final Answer:\n[entailment]", 0.83)
        result = nli._parse_output(out)
        assert result == {"label": "entailment", "probability": 0.83}

    def test_unknown_label_coerced_to_neutral_keeps_confidence(self):
        nli = self._nli()
        out = self._mk_simbauq_output("blah [supported]", 0.6)
        result = nli._parse_output(out)
        assert result == {"label": "neutral", "probability": 0.6}

    def test_degraded_confidence_none_falls_back_to_neutral(self):
        nli = self._nli()
        out = self._mk_simbauq_output("[contradiction]", None)
        result = nli._parse_output(out)
        assert result == {"label": "neutral", "probability": 1.0}

    def test_unsuccessful_sampling_falls_back_to_neutral(self):
        nli = self._nli()
        out = MagicMock()
        out.success = False
        result = nli._parse_output(out)
        assert result == {"label": "neutral", "probability": 1.0}
