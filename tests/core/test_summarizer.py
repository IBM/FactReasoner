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

"""Unit tests for fact_reasoner.core.summarizer module."""

import pytest
from unittest.mock import MagicMock, patch
from src.fact_reasoner.core.summarizer import (
    ContextSummarizer,
    INSTRUCTION_WITH_REF,
    INSTRUCTION_WITHOUT_REF,
)


class TestContextSummarizerInit:
    """Tests for ContextSummarizer initialization."""

    def test_summarizer_none_backend_raises(self):
        with pytest.raises(ValueError, match="Mellea backend is None"):
            ContextSummarizer(backend=None)

    def test_summarizer_stores_backend(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        summarizer = ContextSummarizer(backend=mock_backend)
        assert summarizer.backend == mock_backend


class TestSummarizerInstructions:
    """Tests for summarizer instruction templates."""

    def test_instruction_without_ref_contains_rules(self):
        assert "Rules:" in INSTRUCTION_WITHOUT_REF
        assert "Do NOT add any new information" in INSTRUCTION_WITHOUT_REF
        assert "Do NOT remove any information" in INSTRUCTION_WITHOUT_REF

    def test_instruction_without_ref_contains_examples(self):
        assert "EXAMPLE 1:" in INSTRUCTION_WITHOUT_REF
        assert "EXAMPLE 2:" in INSTRUCTION_WITHOUT_REF
        assert "EXAMPLE 3:" in INSTRUCTION_WITHOUT_REF

    def test_instruction_without_ref_contains_placeholder(self):
        assert "{{context}}" in INSTRUCTION_WITHOUT_REF

    def test_instruction_with_ref_contains_atom(self):
        assert "ATOM" in INSTRUCTION_WITH_REF
        assert "{{atom_text}}" in INSTRUCTION_WITH_REF
        assert "{{context}}" in INSTRUCTION_WITH_REF

    def test_instruction_with_ref_contains_none_option(self):
        # When context is irrelevant, summary should be "None"
        assert "None" in INSTRUCTION_WITH_REF

    def test_instruction_with_ref_contains_examples(self):
        assert "Example 1:" in INSTRUCTION_WITH_REF
        assert "Example 2:" in INSTRUCTION_WITH_REF
        assert "Example 3:" in INSTRUCTION_WITH_REF
        assert "Example 4:" in INSTRUCTION_WITH_REF
        assert "Example 5:" in INSTRUCTION_WITH_REF


class TestContextSummarizerGetProbability:
    """Tests for ContextSummarizer._get_probability method."""

    def test_get_probability_computes_correctly(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        summarizer = ContextSummarizer(backend=mock_backend)

        mock_output = MagicMock()
        mock_output._meta = {
            "oai_chat_response": {
                "logprobs": {
                    "content": [
                        {"token": "Test", "logprob": -0.5},
                        {"token": "summary", "logprob": -0.3},
                        {"token": "<eos>", "logprob": -0.1},  # EOS token
                    ]
                }
            }
        }

        result = summarizer._get_probability(mock_output)
        # exp((-0.5 + -0.3) / 2) = exp(-0.4) ≈ 0.67
        assert 0 < result <= 1

    def test_get_probability_handles_empty(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        summarizer = ContextSummarizer(backend=mock_backend)

        mock_output = MagicMock()
        mock_output._meta = {
            "oai_chat_response": {
                "logprobs": {
                    "content": [
                        {"token": "<eos>", "logprob": -0.1},  # Only EOS
                    ]
                }
            }
        }

        result = summarizer._get_probability(mock_output)
        assert result == 0.0  # Infinite logprob returns 0


class TestContextSummarizerRunBatch:
    """Tests for ContextSummarizer.run_batch method."""

    @pytest.mark.asyncio
    async def test_run_batch_with_atom_text(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        mock_result = MagicMock()
        mock_result.__str__ = lambda self: "This is a summary of the context."
        mock_result._meta = {
            "oai_chat_response": {
                "logprobs": {
                    "content": [
                        {"token": "This", "logprob": -0.2},
                        {"token": "<eos>", "logprob": -0.1},
                    ]
                }
            }
        }

        mock_output = MagicMock()
        mock_output.success = True
        mock_output.result = mock_result

        async def mock_ainstruct(*args, **kwargs):
            return mock_output

        with patch('src.fact_reasoner.core.summarizer.mfuncs.ainstruct', side_effect=mock_ainstruct):
            with patch('asyncio.gather', return_value=[mock_output]):
                summarizer = ContextSummarizer(backend=mock_backend)
                results = await summarizer.run_batch(
                    contexts=["Long context text here."],
                    atom_text="Test atom about something."
                )

                assert isinstance(results, list)
                assert len(results) == 1
                assert "summary" in results[0]
                assert "probability" in results[0]

    @pytest.mark.asyncio
    async def test_run_batch_without_atom_text(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        mock_result = MagicMock()
        mock_result.__str__ = lambda self: "General summary."
        mock_result._meta = {
            "oai_chat_response": {
                "logprobs": {
                    "content": [
                        {"token": "General", "logprob": -0.2},
                        {"token": "<eos>", "logprob": -0.1},
                    ]
                }
            }
        }

        mock_output = MagicMock()
        mock_output.success = True
        mock_output.result = mock_result

        with patch('asyncio.gather', return_value=[mock_output]):
            summarizer = ContextSummarizer(backend=mock_backend)
            # When atom_text is None, should use INSTRUCTION_WITHOUT_REF
            results = await summarizer.run_batch(
                contexts=["Context to summarize."],
                atom_text=None
            )

            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_run_batch_handles_none_summary(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        mock_result = MagicMock()
        mock_result.__str__ = lambda self: "None"
        mock_result._meta = {
            "oai_chat_response": {
                "logprobs": {
                    "content": [
                        {"token": "None", "logprob": -0.1},
                        {"token": "<eos>", "logprob": -0.1},
                    ]
                }
            }
        }

        mock_output = MagicMock()
        mock_output.success = True
        mock_output.result = mock_result

        with patch('asyncio.gather', return_value=[mock_output]):
            summarizer = ContextSummarizer(backend=mock_backend)
            results = await summarizer.run_batch(
                contexts=["Irrelevant context."],
                atom_text="Unrelated atom."
            )

            assert len(results) == 1
            # "None" should be converted to empty string
            assert results[0]["summary"] == ""
