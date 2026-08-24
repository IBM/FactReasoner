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

"""LLM judges: the practical incumbent the LCS is measured against.

Two judges, both reading the response as prose and returning a 1-5 coherence
rating that is rescaled to ``[0, 1]``:

``GEvalCoherence``
    A G-Eval-style judge (Liu et al., EMNLP 2023): the SummEval coherence rubric,
    chain-of-thought *then* a rating, and -- where the backend exposes logprobs --
    G-Eval's probability-weighted score, :math:`\\sum_k k \\cdot p(k)`, rather than
    the single emitted integer. The weighting matters because a 1-5 integer scale is
    coarse: on a ladder whose consecutive rungs differ by a few words, an unweighted
    judge returns the same integer for every rung and its ordering agreement is
    decided by tie-breaking.
``DirectCoherenceRating``
    "Rate the logical coherence from 1 to 5", no rubric and no reasoning. The naive
    incumbent, and the control that says how much of the judge's performance comes
    from the rubric rather than the model.

What the comparison is for
--------------------------
A judge is not a straw man -- on aggregate correlation judges are strong, and this
is the comparison reviewers will ask for first. What a judge cannot offer is
attribution (which claim is responsible), determinism (Remark 2: a
meaning-preserving edit provably cannot change the LCS), or auditability. Those are
the axes the paper should argue on, and they are only defensible with a *measured*
judge column beside them.

Which is why :func:`judge_with_variance` exists and why every reported judge number
should come through it. A judge's run-to-run spread is part of its result: if the
spread across seeds is wider than the gap between the judge and the LCS, the
comparison cannot support a ranking claim, and the honest thing is to say so.
"""

from __future__ import annotations

import math
import re
import statistics
import time
from collections.abc import Sequence
from typing import Any

from fact_reasoner.coherence_baselines.base import BaselineScore

#: The rating scale both judges use. G-Eval's SummEval coherence dimension is
#: 1-5, and keeping the same scale for the direct judge means the two differ only
#: in rubric and reasoning, not in the response format.
SCALE_MIN, SCALE_MAX = 1, 5

#: The SummEval coherence rubric, as G-Eval uses it. Reproduced rather than
#: paraphrased: a judge baseline whose rubric we invented would be a judge we
#: tuned, and the point is to compare against the incumbent as published.
GEVAL_COHERENCE_PROMPT = """
You will be given one response written for a query.

Your task is to rate the response on one metric.

Evaluation Criteria:

Coherence (1-5) - the collective quality of all sentences. We align this dimension
with the DUC quality question of structure and coherence, whereby the response
should be well-structured and well-organized. The response should not just be a
heap of related information, but should build from sentence to sentence to a
coherent body of information about a topic.

Evaluation Steps:

1. Read the response carefully and identify the main topic and key points.
2. Check whether the response presents them in a clear and logical order, and
   whether later statements are consistent with earlier ones.
3. Assign a score for coherence on a scale of 1 to 5, where 1 is the lowest and 5
   is the highest.

Response:

{response}

Reason briefly, then give your rating as a single digit in square brackets, for
example [3].
"""

#: The no-rubric control.
DIRECT_COHERENCE_PROMPT = """
Read the following response and rate its logical coherence -- how well it hangs
together as an argument, and whether it contradicts itself -- on a scale of 1 to 5,
where 1 is the lowest and 5 is the highest.

Response:

{response}

Give your rating as a single digit in square brackets, for example [3].
"""

#: Matches the bracketed rating. The *last* match is taken, since the prompt's own
#: example ("for example [3]") can be echoed back before the real answer.
_RATING = re.compile(r"\[\s*([1-5])\s*\]")


def _extract_rating(text: str) -> int | None:
    """Return the rating from a judge's completion, or None if absent."""
    matches = _RATING.findall(text or "")
    if matches:
        return int(matches[-1])
    # Fall back to a bare leading digit, which some models emit despite the
    # instruction. Anything else is a parse failure and must be counted as one.
    bare = re.search(r"\b([1-5])\b", text or "")
    return int(bare.group(1)) if bare else None


def _normalize(rating: float) -> float:
    """Map a 1-5 rating onto ``[0, 1]``."""
    return (rating - SCALE_MIN) / (SCALE_MAX - SCALE_MIN)


def weighted_rating(logprob_content: Any) -> tuple[float, dict[str, float]] | None:
    """G-Eval's probability-weighted score, read off the rating token's top-k.

    G-Eval computes :math:`\\sum_k k\\,p(k)` rather than taking the single emitted
    integer, and the reason matters here more than in its original setting: a 1-5
    integer scale is coarse, and on a ladder whose consecutive rungs differ by a
    few words an unweighted judge returns the *same* integer for every rung, so its
    ordering agreement is decided by tie-breaking rather than by judgement. The
    expectation over the digit distribution is continuous and can separate rungs the
    integer cannot.

    Only the emitted rating token's alternatives are used. Finding it is the fiddly
    part: models on this stack emit reasoning-channel tokens before the answer, so
    the *last* standalone digit token in ``1``-``5`` is taken, matching
    :func:`_extract_rating`'s last-match rule (both prompts contain an example
    rating that gets echoed).

    Args:
        logprob_content: The backend's per-token logprob list, each entry carrying
            ``token`` and ``top_logprobs``.

    Returns:
        ``(weighted_rating, distribution)`` with the distribution renormalized over
        the digits 1-5, or None when no rating token or no usable alternatives are
        found -- in which case the caller falls back to the emitted integer rather
        than inventing a number.
    """
    if not logprob_content:
        return None

    target = None
    for entry in logprob_content:
        token = str(entry.get("token") or "").strip()
        if token in {"1", "2", "3", "4", "5"}:
            target = entry  # keep going: the last such token is the answer
    if target is None:
        return None

    digits: dict[str, float] = {}
    for alt in target.get("top_logprobs") or []:
        tok = str(alt.get("token") or "").strip()
        if tok in {"1", "2", "3", "4", "5"}:
            lp = alt.get("logprob")
            if isinstance(lp, (int, float)):
                digits[tok] = digits.get(tok, 0.0) + math.exp(lp)

    total = sum(digits.values())
    if not digits or total <= 0:
        return None
    dist = {k: v / total for k, v in digits.items()}
    return sum(int(k) * v for k, v in dist.items()), dist


class _JudgeBaseline:
    """Shared plumbing: render a prompt, generate, parse a rating.

    Args:
        generate: A callable ``(prompt: str) -> str`` returning the model's
            completion. Injected rather than built here so this module stays
            independent of the backend layer -- the driver supplies a closure over
            the Mellea backend, and tests supply a stub.
        name: Report column name.
    """

    prompt_template = ""

    def __init__(self, generate, *, weighted: bool = True, name: str | None = None):
        self.generate = generate
        self.weighted = weighted
        if name:
            self.name = name

    def score(self, atoms: Sequence[str], response: str) -> BaselineScore:
        """Score one response by asking a judge.

        Args:
            atoms: Unused. A judge reads prose; it never sees the decomposition.
                Worth noting in reports -- unlike every other baseline here, a
                judge's input is not held fixed with the LCS's atoms, so its
                comparison is slightly less controlled.
            response: The response text.

        Returns:
            The :class:`BaselineScore`; abstains when the completion has no rating.
        """
        started = time.time()
        n_atoms = len([a for a in atoms if a and a.strip()])
        if not (response or "").strip():
            return BaselineScore(
                name=self.name,
                score=None,
                atoms_scored=n_atoms,
                diagnostics={"reason": "empty response"},
            )

        prompt = self.prompt_template.format(response=response.strip())
        try:
            completion = self.generate(prompt)
        except Exception as e:  # noqa: BLE001
            return BaselineScore(
                name=self.name,
                score=None,
                atoms_scored=n_atoms,
                diagnostics={"reason": f"generation failed: {e}"},
            )

        # A generate() may hand back either the text or (text, logprob_content); the
        # latter enables G-Eval's probability weighting. Accepting both keeps this
        # module free of backend imports and unit-testable with a plain lambda.
        logprob_content = None
        if isinstance(completion, tuple):
            completion, logprob_content = completion

        rating = _extract_rating(
            completion if isinstance(completion, str) else str(completion)
        )
        if rating is None:
            # An unparseable judgement is a missing measurement, not a low score.
            return BaselineScore(
                name=self.name,
                score=None,
                atoms_scored=n_atoms,
                diagnostics={
                    "reason": "no rating found in completion",
                    "completion": str(completion)[:400],
                },
            )

        diagnostics: dict[str, Any] = {
            "rating": rating,
            "scale": [SCALE_MIN, SCALE_MAX],
            "seconds": round(time.time() - started, 3),
        }

        effective = float(rating)
        if self.weighted:
            weighted = weighted_rating(logprob_content)
            if weighted is not None:
                effective, dist = weighted
                diagnostics["weighted_rating"] = effective
                diagnostics["digit_distribution"] = {
                    k: round(v, 6) for k, v in sorted(dist.items())
                }
            else:
                # No usable top-k: fall back to the emitted integer rather than
                # inventing a distribution, and record that the fallback happened
                # so a reader can tell a weighted column from a half-weighted one.
                diagnostics["weighted"] = False

        return BaselineScore(
            name=self.name,
            score=_normalize(effective),
            atoms_scored=n_atoms,
            diagnostics=diagnostics,
        )


class GEvalCoherence(_JudgeBaseline):
    """G-Eval's coherence dimension, with the SummEval rubric."""

    name = "judge_geval"
    prompt_template = GEVAL_COHERENCE_PROMPT


class DirectCoherenceRating(_JudgeBaseline):
    """A bare 1-5 coherence rating: no rubric, no reasoning."""

    name = "judge_direct"
    prompt_template = DIRECT_COHERENCE_PROMPT


def judge_with_variance(
    judge: Any,
    atoms: Sequence[str],
    response: str,
    *,
    seeds: int = 5,
) -> BaselineScore:
    """Run a judge repeatedly and report the mean with its spread.

    A single judge call is not a measurement of a judge -- LLM raters move between
    identical calls, and the paper's determinism argument is only meaningful next to
    a number for that movement. So the reported score is the mean across ``seeds``
    runs, and the standard deviation travels with it in diagnostics.

    Args:
        judge: A judge baseline exposing ``score(atoms, response)``.
        atoms: Passed through.
        response: Passed through.
        seeds: How many times to run. Five is the floor: fewer cannot show a spread.

    Returns:
        A :class:`BaselineScore` whose ``score`` is the mean over successful runs
        and whose diagnostics carry ``sd``, ``ratings`` and the abstention count.
        Abstains only when *every* run abstained.
    """
    if seeds < 1:
        raise ValueError(f"seeds must be at least 1, got {seeds}.")

    scores: list[float] = []
    ratings: list[int] = []
    abstained = 0
    for _ in range(seeds):
        out = judge.score(atoms, response)
        if out.score is None:
            abstained += 1
            continue
        scores.append(out.score)
        if "rating" in out.diagnostics:
            ratings.append(out.diagnostics["rating"])

    n_atoms = len([a for a in atoms if a and a.strip()])
    if not scores:
        return BaselineScore(
            name=getattr(judge, "name", "judge"),
            score=None,
            atoms_scored=n_atoms,
            diagnostics={"reason": "every judge run abstained", "seeds": seeds},
        )

    return BaselineScore(
        name=getattr(judge, "name", "judge"),
        score=statistics.fmean(scores),
        atoms_scored=n_atoms,
        diagnostics={
            "seeds": seeds,
            "runs_scored": len(scores),
            "abstained": abstained,
            "sd": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
            "min": min(scores),
            "max": max(scores),
            "ratings": ratings,
        },
    )


def make_backend_generate(backend: Any):
    """Return a ``prompt -> (text, logprob_content)`` callable over a Mellea backend.

    Lives here rather than in a driver script so the two entry points (fixtures and
    ladder) share one implementation; the judges themselves stay free of backend
    imports and unit-testable with a plain lambda.

    Two details are load-bearing:

    * ``mfuncs.instruct`` returns a ``(thunk, context)`` **tuple**, not a thunk.
      ``str()`` on the tuple yields its repr, and every rating parse would then read
      the example digit out of the echoed prompt instead of the answer.
    * The top-k alternatives that G-Eval's weighting needs live on the raw response
      metadata, not in the flattened per-token list, so they are read from
      ``_meta`` directly. Absent them the judge falls back to the emitted integer.
    """
    import mellea.stdlib.functional as mfuncs
    from mellea.stdlib.context import SimpleContext

    def generate(prompt: str):
        res = mfuncs.instruct(
            prompt,
            context=SimpleContext(),
            backend=backend,
            model_options={"logprobs": True, "top_logprobs": 5},
        )
        out = res[0] if isinstance(res, tuple) else res

        content = None
        try:
            meta = out._meta or {}
            container = meta.get("logprobs") or (
                meta.get("oai_chat_response", {})
                .get("choices", [{}])[0]
                .get("logprobs")
            )
            if isinstance(container, dict):
                content = container.get("content")
            elif isinstance(container, list):
                content = container
        except Exception:  # noqa: BLE001 - no logprobs is a supported fallback
            content = None
        return str(out), content

    return generate
