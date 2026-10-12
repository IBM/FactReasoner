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

# One-shot atomization + decontextualization with a fine-tuned LoRA adapter.
#
# The adapter (trained on granite-4.2-8b) replaces the Atomizer -> Reviser pair:
# given the user query and the response, it returns the response's atomic units,
# already decontextualized, one per line. It is served by vLLM as its own model
# name next to the base model (`vllm serve ... --enable-lora --lora-modules
# <name>=<path>`), so on the Mellea side it is an ordinary OpenAI-compatible
# backend whose model_id is the adapter name.

import re
from typing import Any

import mellea.stdlib.functional as mfuncs
from mellea.backends import Backend, ModelOption
from mellea.core import MelleaLogger
from mellea.stdlib.components import Message
from mellea.stdlib.context import ChatContext

from fact_reasoner import telemetry
from fact_reasoner.utils import run_throttled

from ._async_runner import run_coroutine

# The final user turn, verbatim from the adapter's training data. The adapter
# only behaves as trained when it sees exactly this text.
INSTRUCTION_ATOMIZER_REVISER = (
    "<atomizer>As a judge agent, your role is to decompose the provided text into "
    "units that meet the given criteria, utilizing all available information, "
    "including conversations, documents, and tools.\n\n"
    "### Criteria: An atomic unit is a single, self-contained statement that "
    "expresses exactly one piece of information from the text. It should be "
    "minimal (it cannot be split further without losing meaning), independently "
    "verifiable, and understandable on its own without reference to the other "
    "units. Resolve pronouns and references so that each unit stands alone.\n\n"
    "### Scoring Schema: Decompose the last assistant's text into the units that "
    "meet the criteria. Return one unit per line, with no numbering, bullets, or "
    "extra commentary."
)

# Defensive: the adapter is trained not to number or bullet its lines, but a
# stray "1." / "-" / "*" / "•" prefix must not end up inside an atom.
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def parse_units(text: str) -> list[str]:
    """Split the adapter output into atoms: one per non-empty line, in order."""
    units = []
    for line in text.splitlines():
        unit = _LIST_MARKER.sub("", line).strip()
        if unit:
            units.append(unit)
    return units


class AtomizerReviser:
    """
    One-shot atomic unit extraction and decontextualization via a LoRA adapter.

    A drop-in replacement for :class:`~fact_reasoner.core.atomizer.Atomizer`
    (same ``run``/``run_batch`` result shape) whose atoms are already revised;
    ``revises_atoms`` tells the pipelines to skip the separate Reviser stage.
    """

    #: The atoms this extractor returns are already decontextualized.
    revises_atoms = True

    def __init__(self, backend: Backend, reasoning: bool | None = False):
        """
        Initialize the AtomizerReviser.

        Args:
            backend: Backend
                A Mellea backend whose model is the adapter -- e.g.
                ``build_backend("vllm", model_id="<adapter name>", base_url=...)``.
            reasoning: Optional[bool]
                Whether the model may reason ("think") before answering.
                Defaults to False, which matches how the adapter was trained
                (granite-4.2 thinks by default otherwise). None sends no setting
                and keeps the backend's default.
        """

        # Safety checks
        if backend is None:
            raise ValueError(
                "Mellea backend is None. Please provide a valid Mellea backend."
            )
        # Any other value would be forwarded as a reasoning effort level (e.g.
        # reasoning="false" -> reasoning_effort="false"), which backends reject.
        if reasoning is not None and not isinstance(reasoning, bool):
            raise TypeError(
                f"reasoning must be True, False or None, not {reasoning!r}."
            )

        self.backend = backend
        self.reasoning = reasoning

        # Print backend info
        print(f"[AtomizerReviser] Using Mellea backend: {self.backend.model_id}")

        # Disable Mellea logging
        MelleaLogger.get_logger().setLevel(MelleaLogger.ERROR)

    @staticmethod
    def _context(response: str, query: str | None) -> ChatContext:
        """The conversation the adapter was trained on: query, then response.

        Without a query (response-only inputs) the user turn is omitted and the
        response is the only prior turn.
        """
        context = ChatContext()
        if query:
            context = context.add(Message("user", query))
        return context.add(Message("assistant", response))

    def _request(self, response: str, query: str | None):
        """Build the (unawaited) adapter request for one response."""
        return mfuncs.achat(
            INSTRUCTION_ATOMIZER_REVISER,
            context=self._context(response, query),
            backend=self.backend,
            model_options=(
                None
                if self.reasoning is None
                else {ModelOption.THINKING: self.reasoning}
            ),
        )

    def run(self, response: str, query: str | None = None) -> dict[str, str]:
        """
        Extract the revised atomic units of a response.

        Synchronous wrapper around :meth:`run_batch`, run on the shared
        background event loop. Do not call it from async code; ``await``
        :meth:`run_batch` there instead.

        Args:
            response: str
                The response from which to extract atomic units.
            query: Optional[str]
                The user query the response answers (used to resolve
                references, as in training).
        Returns:
            dict: The revised atomic units keyed by position (``"0"``, ``"1"``,
            ...), or an empty dict on failure -- the same shape as
            ``Atomizer.run``.
        """

        return run_coroutine(self.run_batch([response], [query]))[0]

    async def run_batch(
        self, responses: list[str], queries: list[str | None] | None = None
    ) -> list[dict[str, str]]:
        """
        Extract the revised atomic units of several responses concurrently.

        Args:
            responses: List[str]
                The responses from which to extract atomic units.
            queries: Optional[List[Optional[str]]]
                The query for each response (positionally aligned), if any.
        Returns:
            List[dict]: One result per response, as for :meth:`run`.
        """

        if queries is None:
            queries = [None] * len(responses)
        if len(queries) != len(responses):
            raise ValueError("queries must be aligned with responses.")

        # run_throttled applies bounded concurrency plus a per-minute rate
        # limit, and captures per-item exceptions so a single backend failure
        # does not drop the rest.
        def factory(item: tuple[str, str | None]):
            return self._request(*item)

        with telemetry.span(
            "fact_reasoner.atomizer_reviser.run_batch",
            **{
                "fact_reasoner.responses": len(responses),
                "fact_reasoner.atomizer_reviser.reasoning": str(self.reasoning),
            },
        ):
            outputs = await run_throttled(factory, list(zip(responses, queries)))

        # Positionally aligned with `responses`; every failure maps to {}.
        return [self._parse_output(output) for output in outputs]

    @staticmethod
    def _parse_output(output: Any) -> dict[str, str]:
        """Map one ``achat`` result (or captured exception) to a units dict."""
        if isinstance(output, Exception):
            print(f"[AtomizerReviser] Generation failed: {output}")
            telemetry.count_fallback("atomizer_reviser", "generation_error")
            return {}
        message, _ = output
        units = parse_units(str(message.content or ""))
        if not units:
            print("[AtomizerReviser] The adapter returned no atomic units.")
            telemetry.count_fallback("atomizer_reviser", "empty_output")
            return {}
        return {str(i): unit for i, unit in enumerate(units)}

    def __str__(self) -> str:
        return "This is the atomizer-reviser"
