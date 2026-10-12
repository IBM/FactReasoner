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

"""Unit tests for fact_reasoner.core.reviser module."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mellea.backends import ModelOption

from fact_reasoner.core.reviser import INSTRUCTION_REVISER, Reviser


class TestReviserInit:
    """Tests for Reviser initialization."""

    def test_reviser_none_backend_raises(self):
        with pytest.raises(ValueError, match="Mellea backend is None"):
            Reviser(backend=None)

    def test_reviser_stores_backend(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        reviser = Reviser(backend=mock_backend)
        assert reviser.backend == mock_backend


class TestReviserInstruction:
    """Tests for Reviser instruction template."""

    def test_instruction_contains_examples(self):
        assert "Example 1:" in INSTRUCTION_REVISER
        assert "Example 2:" in INSTRUCTION_REVISER
        assert "Example 3:" in INSTRUCTION_REVISER

    def test_instruction_mentions_vague_references(self):
        assert "Vague References:" in INSTRUCTION_REVISER
        assert "Pronouns" in INSTRUCTION_REVISER
        assert "Demonstrative pronouns" in INSTRUCTION_REVISER

    def test_instruction_contains_output_format(self):
        assert "```json" in INSTRUCTION_REVISER
        assert "revised_unit" in INSTRUCTION_REVISER
        # The rationale was dropped to keep the reply (and latency) short.
        assert "rationale" not in INSTRUCTION_REVISER

    def test_instruction_contains_placeholders(self):
        assert "{{atomic_unit}}" in INSTRUCTION_REVISER
        assert "{{response}}" in INSTRUCTION_REVISER


class TestReviserRun:
    """Tests for Reviser.run method."""

    def test_run_returns_list(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        mock_output = MagicMock()
        mock_output.success = True
        mock_output.__str__ = lambda self: (
            """```json
{
    "revised_unit": "Albert Einstein was German-born."
}
```"""
        )

        with patch(
            "src.fact_reasoner.core.reviser.mfuncs.ainstruct",
            new=AsyncMock(return_value=mock_output),
        ):
            reviser = Reviser(backend=mock_backend)
            result = reviser.run(
                units=["Einstein was German-born."],
                response="Albert Einstein was a German-born physicist.",
            )

            assert isinstance(result, list)
            assert len(result) == 1
            assert "revised_unit" in result[0]
            assert result[0]["revised_unit"] == "Albert Einstein was German-born."
            # Kept for callers of the old result shape.
            assert result[0]["rationale"] == ""

    def test_run_multiple_units(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        mock_output = MagicMock()
        mock_output.success = True
        mock_output.__str__ = lambda self: (
            """```json
{
    "revised_unit": "Revised atom"
}
```"""
        )

        with patch(
            "src.fact_reasoner.core.reviser.mfuncs.ainstruct",
            new=AsyncMock(return_value=mock_output),
        ):
            reviser = Reviser(backend=mock_backend)
            result = reviser.run(
                units=["He was born in 1879.", "She won the prize."],
                response="Albert Einstein was born in 1879. Marie Curie won the prize.",
            )

            assert len(result) == 2

    def test_run_handles_failure(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        mock_output = MagicMock()
        mock_output.success = False

        with patch(
            "src.fact_reasoner.core.reviser.mfuncs.ainstruct",
            new=AsyncMock(return_value=mock_output),
        ):
            reviser = Reviser(backend=mock_backend)
            result = reviser.run(units=["Test unit"], response="Test response")

            # Failures are now aligned no-op revisions (one entry per unit),
            # falling back to the original atom text.
            assert len(result) == 1
            assert result[0]["revised_unit"] == "Test unit"
            assert result[0]["text"] == "Test unit"

    def test_run_includes_original_text(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        mock_output = MagicMock()
        mock_output.success = True
        mock_output.__str__ = lambda self: (
            """```json
{
    "revised_unit": "Revised version"
}
```"""
        )

        with patch(
            "src.fact_reasoner.core.reviser.mfuncs.ainstruct",
            new=AsyncMock(return_value=mock_output),
        ):
            reviser = Reviser(backend=mock_backend)
            result = reviser.run(units=["Original text"], response="Full response")

            assert len(result) == 1
            assert "text" in result[0]

    def test_run_returns_empty_dict_on_generation_exception(self):
        """A backend/network error during generation must not crash run()."""
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        with patch(
            "src.fact_reasoner.core.reviser.mfuncs.ainstruct",
            new=AsyncMock(side_effect=RuntimeError("backend exploded")),
        ):
            reviser = Reviser(backend=mock_backend)
            result = reviser.run(units=["Atom A"], response="resp")

            assert len(result) == 1
            assert result[0]["revised_unit"] == "Atom A"
            assert result[0]["text"] == "Atom A"


class TestReviserRunBatch:
    """Tests for Reviser.run_batch throttling and failure resilience."""

    @staticmethod
    def _mk_output(success: bool, text: str = ""):
        out = MagicMock()
        out.success = success
        out.__str__ = lambda self: text
        return out

    def test_run_batch_returns_aligned_results(self):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        outputs = [
            self._mk_output(True, '```json\n{"revised_unit": "A2"}\n```'),
            self._mk_output(True, '```json\n{"revised_unit": "B2"}\n```'),
        ]

        async def fake_ainstruct(*args, **kwargs):
            return outputs.pop(0)

        with patch(
            "src.fact_reasoner.core.reviser.mfuncs.ainstruct",
            side_effect=fake_ainstruct,
        ):
            reviser = Reviser(backend=mock_backend)
            results = asyncio.run(reviser.run_batch(["A", "B"], "resp"))

        assert [r["revised_unit"] for r in results] == ["A2", "B2"]
        # The per-item original atom text is preserved (fixes loop-capture bug).
        assert [r["text"] for r in results] == ["A", "B"]

    def test_run_batch_single_failure_does_not_drop_others(self):
        """One raised call must not drop the others; results stay aligned."""
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        good = self._mk_output(True, '```json\n{"revised_unit": "OK"}\n```')

        async def fake_ainstruct(*args, **kwargs):
            if kwargs["user_variables"]["atomic_unit"] == "bad":
                raise RuntimeError("boom")
            return good

        with patch(
            "src.fact_reasoner.core.reviser.mfuncs.ainstruct",
            side_effect=fake_ainstruct,
        ):
            reviser = Reviser(backend=mock_backend)
            results = asyncio.run(reviser.run_batch(["ok1", "bad", "ok2"], "resp"))

        assert len(results) == 3
        assert results[0]["revised_unit"] == "OK"
        # Failed item falls back to a no-op revision aligned to its input.
        assert results[1]["revised_unit"] == "bad"
        assert results[1]["text"] == "bad"
        assert results[2]["revised_unit"] == "OK"

    def test_run_batch_aligned_when_calls_finish_out_of_order(self):
        """Results follow input order even if later atoms finish first."""
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"

        async def fake_ainstruct(*args, **kwargs):
            atom = kwargs["user_variables"]["atomic_unit"]
            # The first atom is the slowest, so completion order is reversed.
            await asyncio.sleep({"A": 0.03, "B": 0.02, "C": 0.0}[atom])
            return self._mk_output(True, '```json\n{"revised_unit": "%s2"}\n```' % atom)

        with patch(
            "src.fact_reasoner.core.reviser.mfuncs.ainstruct",
            side_effect=fake_ainstruct,
        ):
            reviser = Reviser(backend=mock_backend)
            results = asyncio.run(reviser.run_batch(["A", "B", "C"], "resp"))

        assert [r["revised_unit"] for r in results] == ["A2", "B2", "C2"]
        assert [r["text"] for r in results] == ["A", "B", "C"]


class TestReviserOutputValidation:
    """Replies that are valid JSON but not a usable revision."""

    @staticmethod
    def _run(reply):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"
        out = MagicMock()
        out.success = True
        out.__str__ = lambda self: reply
        with patch(
            "src.fact_reasoner.core.reviser.mfuncs.ainstruct",
            new=AsyncMock(return_value=out),
        ):
            return Reviser(backend=mock_backend).run(["He was born."], "resp")

    @pytest.mark.parametrize(
        "reply",
        [
            '```json\n["Einstein was born."]\n```',  # a list, not an object
            '```json\n"Einstein was born."\n```',  # a bare string
            '```json\n{"revised_unit": ""}\n```',  # empty
            '```json\n{"revised_unit": "   "}\n```',  # blank
            '```json\n{"revised_unit": 42}\n```',  # not a string
            '```json\n{"other": "x"}\n```',  # key missing
        ],
    )
    def test_unusable_reply_keeps_the_original_atom(self, reply):
        # Previously a list/string reply raised AttributeError out of run(),
        # and an empty or non-string revised_unit replaced the atom's text.
        (result,) = self._run(reply)
        assert result == {
            "revised_unit": "He was born.",
            "rationale": "",
            "text": "He was born.",
        }

    def test_valid_reply_is_trimmed_and_normalized(self):
        (result,) = self._run('```json\n{"revised_unit": "  Einstein was born. "}\n```')
        assert result == {
            "revised_unit": "Einstein was born.",
            "rationale": "",
            "text": "He was born.",
        }

    def test_validator_rejects_unusable_replies_so_sampling_retries(self):
        # The same check backs the rejection-sampling requirement.
        from fact_reasoner.core.reviser import _revised_unit

        assert _revised_unit('```json\n{"revised_unit": "A."}\n```') == "A."
        assert _revised_unit('```json\n["A."]\n```') is None
        assert _revised_unit("not json") is None


class TestReviserReasoning:
    """Tests for the Reviser reasoning ("thinking") setting."""

    @staticmethod
    def _model_options(**reviser_kwargs):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"
        out = MagicMock()
        out.success = True
        out.__str__ = lambda self: '```json\n{"revised_unit": "X"}\n```'
        ainstruct = AsyncMock(return_value=out)
        with patch("src.fact_reasoner.core.reviser.mfuncs.ainstruct", new=ainstruct):
            Reviser(backend=mock_backend, **reviser_kwargs).run(["x"], "resp")
        return ainstruct.call_args.kwargs["model_options"]

    def test_reasoning_off_by_default(self):
        assert self._model_options() == {ModelOption.THINKING: False}

    def test_reasoning_can_be_enabled(self):
        assert self._model_options(reasoning=True) == {ModelOption.THINKING: True}

    def test_reasoning_none_leaves_backend_default(self):
        assert self._model_options(reasoning=None) is None

    @pytest.mark.parametrize("value", ["false", "low", 0, 1])
    def test_reasoning_rejects_non_bool_values(self, value):
        mock_backend = MagicMock()
        mock_backend.model_id = "test-model"
        with pytest.raises(TypeError, match="reasoning must be True, False or None"):
            Reviser(backend=mock_backend, reasoning=value)
