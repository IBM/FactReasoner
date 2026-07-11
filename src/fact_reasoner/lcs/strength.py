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

# Conditional-strength estimation helpers for the relation miner.
#
# The relation miner needs a calibrated scalar P(a_j | a_i, tau) -- "given a_i is
# true and coupling tau holds, how likely is a_j (the coupling's implication)?".
# Naive verbalized confidence (asking the LLM to say "0.72") is a weak, poorly
# calibrated UQ signal (Xiong et al. ICLR 2024, arXiv:2306.13063; for graphical-
# model parameters specifically, EPK arXiv:2505.15918 found it can be worse than
# random). This module provides the two pieces the miner uses to do better:
#
#   * ``surrogate_probability_from_logprobs`` -- read the probability from a
#     surrogate Yes/No token's logprobs, renormalized over the two options:
#     p = P("Yes") / (P("Yes") + P("No")) (Kadavath et al. arXiv:2207.05221).
#     This replaces the verbalized number with a quantity taken directly from the
#     model's own token distribution.
#
#   * ``StrengthCalibrator`` and friends -- an optional post-hoc calibration of
#     the raw strength (temperature / Platt scaling; Guo et al. ICML 2017,
#     arXiv:1706.04599). The default is the identity (no-op); a 1-2 parameter map
#     can be fit later on a small labeled set.

import math
from typing import Any, List, Optional, Sequence, Tuple

# Returned when a surrogate probability cannot be determined (mirrors the miner's
# own "unknown confidence" sentinel).
UNKNOWN_PROBABILITY = 0.5

# Numerical guards for logit/sigmoid round-trips.
_EPS = 1e-6


# ----------------------------------------------------------------------------
# Surrogate-token probability reader.
# ----------------------------------------------------------------------------


def _token_matches(token: str, target: str) -> bool:
    """Whether a (sub)token corresponds to the target word (yes / no).

    Tolerant of leading whitespace and punctuation and of case, so tokenizations
    like ``" Yes"``, ``"Yes"``, ``"yes"``, ``"YES"`` all match ``"yes"``. We match
    on the token's alphabetic prefix so ``"Yes,"`` or ``"Yes."`` also count.
    """
    t = token.strip().lower()
    # Keep the leading run of letters (drop punctuation / partial subwords tails).
    letters = ""
    for ch in t:
        if ch.isalpha():
            letters += ch
        else:
            break
    return letters == target


def _prob_for_word(alternatives: Sequence[Tuple[str, float]], word: str) -> float:
    """Total probability mass on ``word`` across ``(token, logprob)`` alternatives.

    Sums the exponentiated logprobs of every alternative whose token matches
    ``word`` (there can be more than one surface form, e.g. ``"Yes"`` / ``" Yes"``).
    """
    total = 0.0
    for token, logprob in alternatives:
        if _token_matches(token, word):
            total += math.exp(logprob)
    return total


def _first_token_alternatives(logprobs: List[Any]) -> List[Tuple[str, float]]:
    """Extract the first content token's ``(token, logprob)`` alternative list.

    Accepts the per-token entries returned by
    ``fact_reasoner.utils.extract_logprobs_from_output`` (OpenAI / vLLM ``content``
    items), each of which carries a ``top_logprobs`` list of alternatives. When no
    ``top_logprobs`` is present, falls back to the single realized token.
    """
    if not logprobs:
        return []
    first = logprobs[0]

    # OpenAI/vLLM content item: dict-like or object with ``.token`` / ``.logprob``
    # / ``.top_logprobs``. Support both mapping and attribute access.
    def _get(obj, key):
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    top = _get(first, "top_logprobs")
    alternatives: List[Tuple[str, float]] = []
    if top:
        for alt in top:
            tok = _get(alt, "token")
            lp = _get(alt, "logprob")
            if tok is not None and lp is not None:
                alternatives.append((str(tok), float(lp)))
    if alternatives:
        return alternatives

    # Fallback: only the realized token is available.
    tok = _get(first, "token")
    lp = _get(first, "logprob")
    if tok is not None and lp is not None:
        return [(str(tok), float(lp))]
    return []


def surrogate_probability_from_logprobs(
    logprobs: List[Any],
    *,
    positive: str = "yes",
    negative: str = "no",
) -> Optional[float]:
    """Renormalized surrogate-token probability p = P(pos) / (P(pos) + P(neg)).

    Reads the first generated token's top-logprob alternatives and renormalizes
    the mass on the positive vs negative surrogate words (Kadavath et al.
    arXiv:2207.05221). This is the calibrated stand-in for a verbalized number.

    Args:
        logprobs: Per-token logprob entries (as from
            ``extract_logprobs_from_output``); the first entry is the answer token.
        positive: The surrogate word meaning "the implication holds" (default
            "yes").
        negative: The surrogate word meaning "it does not" (default "no").

    Returns:
        The renormalized probability in [0, 1], or ``None`` when neither surrogate
        word appears among the first token's alternatives (caller substitutes its
        own unknown-probability sentinel).
    """
    alternatives = _first_token_alternatives(logprobs)
    if not alternatives:
        return None
    p_pos = _prob_for_word(alternatives, positive)
    p_neg = _prob_for_word(alternatives, negative)
    denom = p_pos + p_neg
    if denom <= 0.0:
        return None
    return p_pos / denom


def affirm_fraction(answers: Sequence[str], *, positive: str = "yes") -> Optional[float]:
    """Fraction of sampled answers whose first word is the positive surrogate.

    The backend-agnostic (no-logprobs) strength readout: sample N Yes/No answers
    and take the affirm fraction (SelfCheckGPT-Prompt style, Manakul et al.
    arXiv:2303.08896). Answers that are neither yes nor no are ignored; returns
    ``None`` if no answer is classifiable.

    Args:
        answers: The raw answer strings (first word is read as yes/no).
        positive: The affirming word (default "yes").

    Returns:
        The affirm fraction in [0, 1], or ``None`` if none are classifiable.
    """
    negative = "no"
    yes = no = 0
    for ans in answers:
        first = (ans or "").strip().split()
        word = first[0] if first else ""
        if _token_matches(word, positive):
            yes += 1
        elif _token_matches(word, negative):
            no += 1
    total = yes + no
    if total == 0:
        return None
    return yes / total


# ----------------------------------------------------------------------------
# Post-hoc calibration (optional; identity by default).
# ----------------------------------------------------------------------------


def _clamp(p: float) -> float:
    return max(_EPS, min(1.0 - _EPS, p))


def _logit(p: float) -> float:
    p = _clamp(p)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class StrengthCalibrator:
    """Base class / protocol for a post-hoc strength calibrator.

    A calibrator maps a raw strength probability in [0, 1] to a calibrated one.
    The default :class:`IdentityCalibrator` is a no-op; concrete calibrators apply
    a 1-2 parameter logit-space transform fit on a small labeled set.
    """

    def transform(self, p: float) -> float:  # pragma: no cover - overridden
        """Map a raw probability to a calibrated probability."""
        raise NotImplementedError


class IdentityCalibrator(StrengthCalibrator):
    """No-op calibrator (the default): returns the input unchanged."""

    def transform(self, p: float) -> float:
        return p


class TemperatureCalibrator(StrengthCalibrator):
    """One-parameter temperature scaling in logit space (Guo et al. 2017).

    ``T > 1`` softens (pulls probabilities toward 0.5), ``T < 1`` sharpens. ``T=1``
    is the identity.
    """

    def __init__(self, temperature: float = 1.0):
        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        self.temperature = float(temperature)

    def transform(self, p: float) -> float:
        return _sigmoid(_logit(p) / self.temperature)

    @classmethod
    def fit(
        cls,
        raw: Sequence[float],
        labels: Sequence[float],
        *,
        grid: Optional[Sequence[float]] = None,
    ) -> "TemperatureCalibrator":
        """Fit the temperature by minimizing negative log-likelihood on a grid.

        A grid search keeps this dependency-free and robust on the tiny label sets
        this is intended for (a handful of hand-labeled relations); it is not meant
        for large-scale calibration.

        Args:
            raw: Raw strength probabilities.
            labels: Ground-truth targets in {0, 1} (or soft in [0, 1]).
            grid: Candidate temperatures; defaults to a log-spaced range.

        Returns:
            The fitted :class:`TemperatureCalibrator`.
        """
        if len(raw) != len(labels):
            raise ValueError("raw and labels must have the same length.")
        if not raw:
            return cls(1.0)
        if grid is None:
            grid = [0.25, 0.4, 0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
        logits = [_logit(p) for p in raw]

        def nll(T: float) -> float:
            total = 0.0
            for z, y in zip(logits, labels):
                q = _clamp(_sigmoid(z / T))
                total -= y * math.log(q) + (1.0 - y) * math.log(1.0 - q)
            return total

        best_T = min(grid, key=nll)
        return cls(best_T)


class PlattCalibrator(StrengthCalibrator):
    """Two-parameter Platt scaling: sigmoid(a * logit(p) + b)."""

    def __init__(self, a: float = 1.0, b: float = 0.0):
        self.a = float(a)
        self.b = float(b)

    def transform(self, p: float) -> float:
        return _sigmoid(self.a * _logit(p) + self.b)
