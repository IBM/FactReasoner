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

"""Throttled NLI batching for the pairwise baselines.

The pairwise baselines issue :math:`O(n^2)` NLI calls per response -- 528 for the
33-atom narrative fixture -- so they must go through the same rate limiter as the
rest of the pipeline: ``run_throttled`` from :mod:`fact_reasoner.utils`, which
applies a 1500-requests-per-minute token bucket and a concurrency ceiling, and
captures per-item exceptions so one failure never drops the batch.

Why this exists rather than calling ``NLIExtractor.run_batch``
-------------------------------------------------------------
``run_batch`` is throttled correctly, but it maps a failed call onto
``{"label": "neutral", "probability": 1.0}`` -- byte-identical to a genuine
neutral verdict. For a *contradiction-rate* baseline that substitution is not
neutral at all: a throttled call becomes evidence of "no contradiction here", so
the score moves **up** when the endpoint degrades, and a rate-limited run reports
a *more* coherent response than a healthy one. The relation miner documents the
same hazard for mining ("a rate-limited call is ... parsed as 'no relation', so
throttling silently costs recall rather than raising").

So this module calls ``run_throttled`` directly and hands back the raw outcome per
pair, keeping ``Exception`` distinguishable from ``neutral``. The baselines then
exclude failures from the denominator and count them, which is the convention the
rest of the package already follows.
"""

from __future__ import annotations

import asyncio
import atexit
from typing import Any, Sequence

from fact_reasoner.utils import (
    MAX_CONCURRENT_REQUESTS,
    MAX_REQUESTS_PER_MINUTE,
    run_throttled,
)

#: Sentinel recorded for a pair whose call failed. Deliberately not a verdict
#: dict, so no caller can mistake it for a label.
CALL_FAILED = object()

#: One event loop reused across every :func:`run_pairs` call in a process.
#:
#: ``asyncio.run`` creates and *closes* a loop per call, which strands the HTTP
#: client the backend opened on it. When that client is later finalized it tries to
#: close a transport belonging to a dead loop, and asyncio reports
#: ``Task exception was never retrieved ... RuntimeError('Event loop is closed')``
#: -- once per baseline, drowning the log in tracebacks that look like failures but
#: are not (the batch has already returned its results by then). Keeping one loop
#: for the process avoids stranding anything.
_LOOP: asyncio.AbstractEventLoop | None = None


def _shared_loop() -> asyncio.AbstractEventLoop:
    """Return the process-wide loop, creating it on first use."""
    global _LOOP
    if _LOOP is None or _LOOP.is_closed():
        _LOOP = asyncio.new_event_loop()
    return _LOOP


@atexit.register
def _close_shared_loop() -> None:
    """Close the shared loop at interpreter exit, if one was created."""
    global _LOOP
    if _LOOP is not None and not _LOOP.is_closed():
        _LOOP.close()
    _LOOP = None


def _acall(extractor, premise: str, hypothesis: str):
    """Build one async NLI coroutine, mirroring ``NLIExtractor.run_batch``.

    ``NLIExtractor`` exposes a synchronous ``run`` and an async ``run_batch``, but
    no per-pair async entry point, so the coroutine is constructed here the same
    way ``run_batch`` builds its own: ``mfuncs.ainstruct`` with the extractor's
    prompt, validator, sampling strategy and logprob options. Reusing the
    extractor's private accessors keeps the label/probability semantics identical
    to the factuality pipeline's -- in particular the span-aligned logprob
    reading, which is what makes the two comparable.
    """
    import mellea.stdlib.functional as mfuncs
    from mellea.stdlib.context import SimpleContext
    from mellea.stdlib.requirements import check, simple_validate

    from fact_reasoner.core.nli import INSTRUCTION_NLI
    from fact_reasoner.utils import extract_nli_label_and_span

    return mfuncs.ainstruct(
        INSTRUCTION_NLI,
        context=SimpleContext(),
        backend=extractor.backend,
        requirements=[
            check(
                "The output must contain an NLI label, either as a JSON "
                'object {"label": "..."} or wrapped in square brackets.',
                validation_fn=simple_validate(
                    lambda s: extract_nli_label_and_span(s)[0] != ""
                ),
            )
        ],
        user_variables={"premise_text": premise, "hypothesis_text": hypothesis},
        strategy=extractor._strategy,
        return_sampling_results=True,
        model_options=extractor._logprobs_model_options(),
    )


async def _run_pairs_async(
    extractor: Any,
    pairs: Sequence[tuple[str, str]],
    *,
    max_concurrency: int,
    rate_per_minute: int,
    show_progress: bool,
) -> list[Any]:
    """Issue one NLI call per pair under the shared throttle."""

    def factory(pair):
        premise, hypothesis = pair
        return _acall(extractor, premise, hypothesis)

    bar = None
    on_progress = None
    if show_progress and pairs:
        from tqdm import tqdm

        bar = tqdm(total=len(pairs), desc="Baseline NLI", unit="pair")
        on_progress = bar.update
    try:
        return await run_throttled(
            factory,
            list(pairs),
            max_concurrency=max_concurrency,
            rate_per_minute=rate_per_minute,
            on_progress=on_progress,
        )
    finally:
        if bar is not None:
            bar.close()


def run_pairs(
    extractor: Any,
    pairs: Sequence[tuple[str, str]],
    *,
    max_concurrency: int = MAX_CONCURRENT_REQUESTS,
    rate_per_minute: int = MAX_REQUESTS_PER_MINUTE,
    show_progress: bool = False,
) -> list[Any]:
    """Score many premise/hypothesis pairs, throttled, preserving failures.

    Args:
        extractor: An NLI extractor. A real
            :class:`~fact_reasoner.core.nli.NLIExtractor` (detected by its
            ``backend`` attribute) goes down the throttled async path; anything
            else -- a stub in tests -- is called sequentially through ``run``.
        pairs: The ``(premise, hypothesis)`` pairs to score.
        max_concurrency: In-flight call ceiling.
        rate_per_minute: Token-bucket rate, defaulting to the pipeline's 1500/min.
        show_progress: Whether to draw a tqdm bar.

    Returns:
        One entry per pair, positionally aligned: either the extractor's verdict
        dict, or :data:`CALL_FAILED` when the call raised. Never a substituted
        neutral -- see the module docstring for why that distinction matters here.
    """
    if not pairs:
        return []

    if not hasattr(extractor, "backend"):
        # Sequential path: no event loop, no throttle needed (a stub extractor in
        # tests, or a backend without an async entry point).
        out: list[Any] = []
        for premise, hypothesis in pairs:
            try:
                out.append(extractor.run(premise, hypothesis))
            except Exception:  # noqa: BLE001
                out.append(CALL_FAILED)
        return out

    # Reuse one loop rather than asyncio.run()'s create-and-close-per-call, so the
    # backend's HTTP client is never stranded on a closed loop (see _LOOP).
    raw = _shared_loop().run_until_complete(
        _run_pairs_async(
            extractor,
            pairs,
            max_concurrency=max_concurrency,
            rate_per_minute=rate_per_minute,
            show_progress=show_progress,
        )
    )

    # `ainstruct` yields Mellea's raw generation object, not a verdict, so it must
    # go through the extractor's own parser -- the same one the synchronous `run`
    # uses. Parsing here (rather than letting callers do it) keeps the label and
    # the span-aligned probability exactly as the factuality pipeline reads them,
    # which is what makes the baselines comparable to it.
    out: list[Any] = []
    for item in raw:
        if isinstance(item, Exception):
            out.append(CALL_FAILED)
            continue
        try:
            out.append(extractor._parse_output(item))
        except Exception:  # noqa: BLE001 - an unparseable generation is a failure
            out.append(CALL_FAILED)
    return out
