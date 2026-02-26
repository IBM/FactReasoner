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

import pytest
import math
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

        mock_output = MagicMock()
        mock_output._meta = {
            "oai_chat_response": {
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
                "logprobs": {
                    "content": [
                        {"token": "[", "logprob": -0.1},
                        {"token": "]", "logprob": -0.1},
                        {"token": "<eos>", "logprob": -0.1},
                    ]
                }
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
                "logprobs": {
                    "content": [
                        {"token": "[", "logprob": -0.1},
                        {"token": "ent", "logprob": -0.2},
                        {"token": "]", "logprob": -0.1},
                        {"token": "<eos>", "logprob": -0.1},
                    ]
                }
            }
        }

        mock_output = MagicMock()
        mock_output.success = True
        mock_output.result = mock_result

        with patch('src.fact_reasoner.core.nli.mfuncs.instruct', return_value=mock_output):
            nli = NLIExtractor(backend=mock_backend)
            result = nli.run(
                premise="The sky is blue.",
                hypothesis="The sky has color."
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

        with patch('src.fact_reasoner.core.nli.mfuncs.instruct', return_value=mock_output):
            nli = NLIExtractor(backend=mock_backend)
            result = nli.run(
                premise="Test premise",
                hypothesis="Test hypothesis"
            )

            assert result["label"] == "neutral"
            assert result["probability"] == 1.0
