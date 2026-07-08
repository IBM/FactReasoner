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


class TestNLIExtractorGetProbability:
    """Tests for NLIExtractor._get_probability method."""

    def test_get_probability_computes_exp_avg_logprob(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        nli = NLIExtractor(backend=mock_backend)

        # Mirrors the real OpenAI backend shape: mellea stores
        # ChatCompletion.model_dump() under "oai_chat_response", so logprobs
        # live at oai_chat_response["choices"][0]["logprobs"]["content"].
        mock_output = MagicMock()
        mock_output._meta = {
            "oai_chat_response": {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {"token": "[", "logprob": -0.1},
                                {"token": "ent", "logprob": -0.5},
                                {"token": "ail", "logprob": -0.3},
                                {"token": "]", "logprob": -0.1},
                                {"token": "<eos>", "logprob": -0.1},  # EOS token
                            ]
                        }
                    }
                ]
            }
        }

        result = nli._get_probability(mock_output)
        # Result should be exp of average logprob for tokens between [ and ]
        assert 0 < result <= 1

    def test_get_probability_handles_empty_logprobs(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        nli = NLIExtractor(backend=mock_backend)

        mock_output = MagicMock()
        mock_output._meta = {
            "oai_chat_response": {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {"token": "[", "logprob": -0.1},
                                {"token": "]", "logprob": -0.1},
                                {"token": "<eos>", "logprob": -0.1},
                            ]
                        }
                    }
                ]
            }
        }

        result = nli._get_probability(mock_output)
        # When count is 0, should return 0.0
        assert result == 0.0


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
