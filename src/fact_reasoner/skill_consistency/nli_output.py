"""Parse a final NLI verdict and score exactly its emitted label tokens."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

PARSER_VERSION = "final-label-v1"
AMBIGUITY_PARSER_VERSION = "relation-ambiguity-json-v1"
LABELS = {"entailment", "contradiction", "neutral"}
AMBIGUITY_LEVELS = {
    "unambiguous",
    "minor_ambiguity",
    "moderate_ambiguity",
    "substantial_ambiguity",
    "closely_competing_relations",
}
_BRACKET = re.compile(
    r"\[\s*(entailment|contradiction|neutral)\s*\]\s*[.!?]?\s*$", re.I
)
_JSON_LABEL = re.compile(r'"label"\s*:\s*"(entailment|contradiction|neutral)"', re.I)
_TERMINATORS = {"", "<|eot_id|>", "<|end_of_text|>", "</s>", "<|im_end|>", "<|end|>"}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON verdict key")
        result[key] = value
    return result


def final_nli_label_and_span(text: str) -> tuple[str, tuple[int, int]]:
    """Require a terminal bracket verdict or a complete terminal JSON object.

    Earlier quoted labels and unmatched explanation brackets are not verdicts.
    A malformed final answer cannot fall back to an earlier label mention.
    Returned offsets cover the label alone, excluding wrapping punctuation.
    """
    match = _BRACKET.search(text)
    if match:
        return match.group(1).lower(), match.span(1)
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text, start)
        except ValueError:
            continue
        if text[end:].strip() not in {"", "```"}:
            continue
        if not isinstance(value, dict) or not isinstance(value.get("label"), str):
            continue
        label = value["label"].lower()
        if label not in LABELS:
            continue
        matches = list(_JSON_LABEL.finditer(text, start, end))
        if len(matches) == 1 and matches[0].group(1).lower() == label:
            return label, matches[0].span(1)
    raise ValueError("Missing or invalid final NLI label")


def final_ambiguity_output_and_span(
    text: str,
) -> tuple[dict[str, str], tuple[int, int]]:
    """Require one terminal relation-ambiguity JSON object and locate its relation."""

    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text, start)
        except ValueError:
            continue
        if text[end:].strip() not in {"", "```"}:
            continue
        if not isinstance(value, dict) or set(value) != {
            "relation",
            "relation_ambiguity",
            "basis",
        }:
            continue
        relation = value["relation"]
        ambiguity = value["relation_ambiguity"]
        basis = value["basis"]
        if relation not in LABELS or ambiguity not in AMBIGUITY_LEVELS:
            continue
        if not isinstance(basis, str) or not basis.strip():
            continue
        pattern = re.compile(r'"relation"\s*:\s*"(' + "|".join(LABELS) + r')"', re.I)
        matches = list(pattern.finditer(text, start, end))
        if len(matches) != 1 or matches[0].group(1).lower() != relation:
            continue
        return {
            "relation": relation,
            "relation_ambiguity": ambiguity,
            "basis": basis,
        }, matches[0].span(1)
    raise ValueError("Missing or invalid final relation-ambiguity output")


def final_ambiguity_logprob(text: str, rows: Sequence[Mapping[str, Any]]) -> float:
    """Score only relation-label tokens in a strict ambiguity JSON response."""

    _, span = final_ambiguity_output_and_span(text)
    return _logprob_for_span(text, rows, span)


def final_nli_logprob(text: str, rows: Sequence[Mapping[str, Any]]) -> float:
    """Geometric mean of token probabilities overlapping the final label."""

    _, span = final_nli_label_and_span(text)
    return _logprob_for_span(text, rows, span)


def _logprob_for_span(
    text: str, rows: Sequence[Mapping[str, Any]], span: tuple[int, int]
) -> float:
    """Score emitted tokens overlapping one validated character span."""

    start, end = span
    if not rows:
        raise ValueError("Missing label logprobs")
    chunks = []
    for row in rows:
        raw_bytes = row.get("bytes")
        if raw_bytes is not None:
            if not isinstance(raw_bytes, (list, tuple)) or not all(
                type(b) is int and 0 <= b <= 255 for b in raw_bytes
            ):
                raise ValueError("Invalid token bytes")
            chunks.append(bytes(raw_bytes))
        elif isinstance(row.get("token"), str):
            chunks.append(row["token"].encode("utf-8"))
        else:
            raise ValueError("Missing token text")
    emitted = b"".join(chunks)
    expected = text.encode("utf-8")
    if (
        not emitted.startswith(expected)
        or emitted[len(expected) :].decode("utf-8").strip() not in _TERMINATORS
    ):
        raise ValueError("Token text does not match model output")
    start, end = len(text[:start].encode("utf-8")), len(text[:end].encode("utf-8"))
    position = 0
    selected = []
    for row, chunk in zip(rows, chunks, strict=True):
        stop = position + len(chunk)
        if stop > start and position < end:
            score = row.get("logprob")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not -1000 < score <= 0
            ):
                raise ValueError("Invalid label logprob")
            selected.append(score)
        position = stop
    if not selected:
        raise ValueError("No tokens overlap final label")
    return math.exp(sum(selected) / len(selected))
