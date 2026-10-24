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

# Atomic fact decontextualization using LLMs

import json
from typing import Any

import mellea.stdlib.functional as mfuncs
from mellea.backends import Backend, ModelOption
from mellea.core import MelleaLogger
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.requirements import check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

# Local imports
from fact_reasoner import telemetry
from fact_reasoner.utils import (
    LOOP_BUDGET,
    handle_generation_error,
    run_throttled,
    throttle_for,
    strip_code_fences,
)

from ._async_runner import run_coroutine

INSTRUCTION_REVISER = """
Instructions:
You task is to decontextualize a UNIT to make it standalone. Each UNIT is an independent content unit or atomic unit extracted from the broader context of a RESPONSE.   

Vague References:
- Pronouns (e.g., "he", "she", "they", "it")
- Demonstrative pronouns (e.g., "this", "that", "these", "those")
- Unknown entities (e.g., "the event", "the research", "the invention")
- Incomplete names (e.g., "Jeff..." or "Bezos..." when referring to Jeff Bezos)

Follow the steps below for unit decontextualization:
1. If the UNIT contains vague references, minimally revise them with respect to the specific subjects they refer to in the RESPONSE.
2. The decontextualized UNIT should be minimally revised by ONLY resolving vague references. No additional information must be added.
3. UNIT extraction might decompose a conjunctive statement into multiple units (e.g. Democracy treats citizens as equals regardless of their race or religion -> (1) Democracy treats citizens as equals regardless of their race, (2) Democracy treats citizens as equals regardless of their religion). Avoid adding what is potentially part of another UNIT.
4. Output ONLY the revised UNIT, with no explanation, in the following JSON format with a markdown code block:

```json
{
  "revised_unit": "<REVISED_UNIT>"
}
```
Where <REVISED_UNIT> is the decontextualized UNIT after resolving vague references (or the UNIT unchanged if it has none).

Use the provided examples to learn your task.

Example 1: 
UNIT: 
Acorns is a financial technology company

RESPONSE:
Acorns is a financial technology company founded in 2012 by Walter Cruttenden, \
Jeff Cruttenden, and Mark Dru that provides micro-investing services. The \
company is headquartered in Irvine, California.

OUTPUT:
```json
{
    "revised_unit": "Acorns is a financial technology company."
}
```

Example 2: 
UNIT:
The victim had previously suffered a broken wrist.

RESPONSE:
The clip shows the victim, with his arm in a cast, being dragged to the floor \
by his neck as his attacker says "I'll drown you" on a school playing field, while forcing water from a bottle into the victim's mouth, \
simulating waterboarding. The video was filmed in a lunch break. The clip shows the victim walking away, without reacting, as the attacker \
and others can be heard continuing to verbally abuse him. The victim, a Syrian refugee, had previously suffered a broken wrist; this had also been \
investigated by the police, who had interviewed three youths but took no further action.

OUTPUT:
```json
{
    "revised_unit": "The Syrian refugee victim had previously suffered a broken wrist."
}
```

Example 3:
UNIT:
The difference is relatively small.

RESPONSE:
Both the RTX 3060 Ti and RTX 3060 are powerful GPUs, and the difference between them lies in their performance. \
The RTX 3060 Ti has more CUDA cores (4864 vs 3584) but a lower boost clock speed (1665 MHz vs 1777 MHz) compared to the RTX 3060. \
In terms of memory bandwidth, the RTX 3060 Ti has a slight edge over the RTX 3060 with a bandwidth of 448 GB/s compared to 360 GB/s. \
However, the difference is relatively small and may not be noticeable in real-world applications.

OUTPUT:
```json
{
    "revised_unit": "The difference in memory bandwidth between the RTX 3060 Ti and RTX 3060 is relatively small."
}
```

Your task:
UNIT:
{{atomic_unit}}

RESPONSE:
{{response}}

OUTPUT:
"""


def _revised_unit(text: str) -> str | None:
    """The revised unit in a reply, or None if the reply is not usable.

    A usable reply is a JSON object (optionally in a code fence) whose
    ``revised_unit`` is a non-empty string. ``validate_json_code_block`` alone
    also accepts JSON lists/strings and any ``revised_unit`` value, which would
    otherwise reach the pipeline as an atom's text.
    """
    try:
        data = json.loads(strip_code_fences(text))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    unit = data.get("revised_unit")
    if not isinstance(unit, str) or not unit.strip():
        return None
    return unit.strip()


class Reviser:
    """
    Atomic unit decontextualization using LLMs.

    """

    def __init__(
        self,
        backend: Backend,
        reasoning: bool | None = False,
        *,
        raise_errors: bool = False,
    ):
        """
        Initialize the Reviser.

        Args:
            backend: Backend
                The Mellea backend to use for LLM interactions.
            reasoning: Optional[bool]
                Whether the model may reason ("think") before answering.
                Defaults to False: revision is a small rewrite, and on models
                that think by default (e.g. granite4.2) the hidden reasoning is
                most of each call's cost -- hundreds to thousands of tokens for
                a ~25-token answer. Small models may resolve fewer references
                without it, so enable it if revision quality drops. None sends
                no setting and keeps the backend's own default; use it for
                hosted models that reject a reasoning parameter.
            raise_errors: bool
                If True, an exception raised by the backend (e.g. a connection
                error) propagates to the caller unchanged. Default False: the
                error is printed and the atom is returned unrevised.
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

        # Initialize the reviser
        self.backend = backend
        self.reasoning = reasoning
        self.raise_errors = raise_errors

        # Print backend info
        print(f"[Reviser] Using Mellea backend: {self.backend}, reasoning={self.reasoning}")

        # Disable Mellea logging
        MelleaLogger.get_logger().setLevel(MelleaLogger.ERROR)

    def run(self, units: list[str], response: str) -> list[dict[str, Any]]:
        """
        Decontextualize the input atomic units using the response as context.

        Synchronous wrapper around :meth:`run_batch`: the atoms are still
        revised concurrently, on the shared background event loop. Do not call
        it from async code; ``await`` :meth:`run_batch` there instead.

        Args:
            units: List[str]
                The atomic units to be decontextualized.
            response: str
                The response from which the atomic units are decontextualized.
        Returns:
            List[Dict[str, Any]]: One result per unit, in input order, each with
            the ``revised_unit``, an empty ``rationale`` and the original
            ``text``. A unit whose revision fails is returned unchanged.
        """

        return run_coroutine(self.run_batch(units, response)) # type: ignore

    def _request(self, atom_text: str, response: str):
        """Build the (unawaited) revision request for a single atom."""
        return mfuncs.ainstruct(
            INSTRUCTION_REVISER,
            context=SimpleContext(),
            backend=self.backend,
            requirements=[
                check(
                    'The output must be a JSON object {"revised_unit": "..."} '
                    "with a non-empty revised unit.",
                    validation_fn=simple_validate(
                        lambda s: _revised_unit(s) is not None
                    ),
                )
            ],
            user_variables={"atomic_unit": atom_text, "response": response},
            strategy=RejectionSamplingStrategy(loop_budget=LOOP_BUDGET),
            return_sampling_results=True,
            model_options=(
                None
                if self.reasoning is None
                else {ModelOption.THINKING: self.reasoning}
            ),
        )

    @staticmethod
    def _fallback(atom_text: str) -> dict[str, Any]:
        """Build a no-op revision result for a failed/unparsable atom."""
        return {"revised_unit": atom_text, "rationale": "", "text": atom_text}

    def _parse_output(self, output: Any, atom_text: str) -> dict[str, Any]:
        """Map a single sampling result to a revised-unit dict.

        On any failure (unsuccessful sampling or unparsable output) the original
        atom is returned unchanged so downstream callers that read
        ``result[i]["revised_unit"]`` positionally never crash.
        """
        if not getattr(output, "success", False):
            telemetry.count_fallback("reviser", "sampling_failed")
            return self._fallback(atom_text)
        revised_unit = _revised_unit(str(output))
        if revised_unit is None:
            print("[Reviser] Failed to parse output: no usable revised unit.")
            telemetry.count_fallback("reviser", "parse_error")
            return self._fallback(atom_text)
        # The prompt no longer asks for a rationale; keep the key so callers
        # reading the old result shape still work.
        return {"revised_unit": revised_unit, "rationale": "", "text": atom_text}

    async def run_batch(self, units: list[str], response: str) -> list[dict[str, Any]]:
        """
        Decontextualize the input atomic units using the response as context.

        Args:
            units: List[str]
                The atomic units to be decontextualized.
            response: str
                The response from which the atomic units are decontextualized.
        Returns:
            List[Dict[str, Any]]: One result per unit, in input order (see
            :meth:`run`).
        """

        # Build a fresh coroutine per atom. run_throttled applies bounded
        # concurrency plus a per-minute rate limit, and captures per-item
        # exceptions so a single backend failure does not drop the rest.
        def factory(atom_text: str):
            return self._request(atom_text, response)

        print(f"[Reviser] Running throttled batch of {len(units)} requests ...")
        with telemetry.span(
            "fact_reasoner.reviser.run_batch",
            **{
                "fact_reasoner.atoms": len(units),
                # str() so the unset (None) case is recorded rather than dropped.
                "fact_reasoner.reviser.reasoning": str(self.reasoning),
            },
        ):
            outputs = await run_throttled(
                factory, units, throttle=throttle_for(self.backend)
            )

        # Results are positionally aligned with `units`; every atom yields one
        # entry (a no-op revision on failure), so callers can index result[i].
        results: list[dict[str, Any]] = []
        for atom_text, output in zip(units, outputs):
            if isinstance(output, Exception):
                handle_generation_error(
                    output, "reviser", "[Reviser] Batch item failed", self.raise_errors
                )
                results.append(self._fallback(atom_text))
                continue
            results.append(self._parse_output(output, atom_text))

        return results
