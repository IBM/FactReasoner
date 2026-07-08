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
from fact_reasoner.core.summarizer import (
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
                                {"token": "Test", "logprob": -0.5},
                                {"token": "summary", "logprob": -0.3},
                                {"token": "<eos>", "logprob": -0.1},  # EOS token
                            ]
                        }
                    }
                ]
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
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {"token": "<eos>", "logprob": -0.1},  # Only EOS
                            ]
                        }
                    }
                ]
            }
        }

        result = summarizer._get_probability(mock_output)
        assert result == 0.0  # Infinite logprob returns 0


class TestContextSummarizerRunBatch:
    """Tests for ContextSummarizer.run_batch method.

    These patch mfuncs.ainstruct with a real async function (run_batch now
    routes through the throttled run_throttled helper rather than a bare
    asyncio.gather) and stub _get_probability, which is exercised separately.
    """

    @staticmethod
    def _mk_output(text: str, success: bool = True):
        out = MagicMock()
        out.success = success
        out.result = MagicMock()
        out.__str__ = lambda self: text
        return out

    @pytest.mark.asyncio
    async def test_run_batch_with_atom_text(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        output = self._mk_output("This is a summary of the context.")

        async def mock_ainstruct(*args, **kwargs):
            return output

        with patch(
            "src.fact_reasoner.core.summarizer.mfuncs.ainstruct",
            side_effect=mock_ainstruct,
        ):
            with patch.object(ContextSummarizer, "_get_probability", return_value=0.5):
                summarizer = ContextSummarizer(backend=mock_backend)
                results = await summarizer.run_batch(
                    contexts=["Long context text here."],
                    atom_text="Test atom about something.",
                )

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0]["summary"] == "This is a summary of the context."
        assert results[0]["probability"] == 0.5
        # Context is aligned to its input (fixes the loop-capture bug).
        assert results[0]["context"] == "Long context text here."

    @pytest.mark.asyncio
    async def test_run_batch_without_atom_text(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        output = self._mk_output("General summary.")

        async def mock_ainstruct(*args, **kwargs):
            return output

        with patch(
            "src.fact_reasoner.core.summarizer.mfuncs.ainstruct",
            side_effect=mock_ainstruct,
        ):
            with patch.object(ContextSummarizer, "_get_probability", return_value=0.5):
                summarizer = ContextSummarizer(backend=mock_backend)
                # When atom_text is None, should use INSTRUCTION_WITHOUT_REF
                results = await summarizer.run_batch(
                    contexts=["Context to summarize."], atom_text=None
                )

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_run_batch_handles_none_summary(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        output = self._mk_output("None")

        async def mock_ainstruct(*args, **kwargs):
            return output

        with patch(
            "src.fact_reasoner.core.summarizer.mfuncs.ainstruct",
            side_effect=mock_ainstruct,
        ):
            with patch.object(ContextSummarizer, "_get_probability", return_value=0.5):
                summarizer = ContextSummarizer(backend=mock_backend)
                results = await summarizer.run_batch(
                    contexts=["Irrelevant context."], atom_text="Unrelated atom."
                )

        assert len(results) == 1
        # "None" should be converted to empty string
        assert results[0]["summary"] == ""

    @pytest.mark.asyncio
    async def test_run_batch_single_failure_does_not_drop_others(self):
        """One raised call maps to an empty summary; results stay aligned."""
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        good = self._mk_output("Good summary.")

        async def mock_ainstruct(*args, **kwargs):
            if kwargs["user_variables"]["context"] == "bad":
                raise RuntimeError("boom")
            return good

        with patch(
            "src.fact_reasoner.core.summarizer.mfuncs.ainstruct",
            side_effect=mock_ainstruct,
        ):
            with patch.object(ContextSummarizer, "_get_probability", return_value=0.5):
                summarizer = ContextSummarizer(backend=mock_backend)
                results = await summarizer.run_batch(
                    contexts=["ok1", "bad", "ok2"], atom_text="atom"
                )

        assert len(results) == 3
        assert results[0]["summary"] == "Good summary."
        assert results[1]["summary"] == ""
        assert results[1]["probability"] == 0.0
        assert results[1]["context"] == "bad"
        assert results[2]["summary"] == "Good summary."
