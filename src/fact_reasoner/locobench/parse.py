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

# One parser per prompt output.
#
# THE CONTRACT: every parser returns ``(value, error)`` and NEVER raises. A malformed
# generation is a gate failure with a reason, not a crash that loses the rest of a
# multi-day run -- so the failure mode of a model returning prose where JSON was asked
# for is a rejected family, recorded, retried.
#
# Parsers validate STRUCTURE only (shape, counts, legal values). Semantic thresholds
# live in `validate.THRESHOLDS`, so there is exactly one place to audit them.

from __future__ import annotations

import json
import re
from typing import Any

from fact_reasoner.locobench.taxonomy_bridge import (
    LEGAL_SENSES,
    coupling_for_sense,
)
from fact_reasoner.utils import extract_first_code_block, extract_first_square_brackets

# P2's claim tags (Phase 1 P2 instruction 7).
CLAIM_TAGS = (
    "correct",
    "incorrect",
    "alt-pair-1",
    "alt-pair-2",
    "disj-pair-1",
    "disj-pair-2",
    "equiv-pair-1",
    "equiv-pair-2",
    "holding",
)

# P3 structural bounds (Phase 1 P3 instructions 2, 3, 9).
N_CLAIMS_RANGE = (14, 18)
N_RELATIONS_RANGE = (8, 12)
N_NON_RELATIONS_RANGE = (4, 6)

STRENGTH_BANDS = {
    "strong": (0.85, 1.00),
    "moderate": (0.60, 0.84),
    "weak": (0.35, 0.59),
}

ERROR_KINDS = ("wrong_sense", "wrong_direction", "false_endpoint", "spurious")

# V2's four verdicts and V4's four statuses.
V2_VERDICTS = ("A", "B", "C", "D")
V4_STATUSES = ("asserted", "altered", "missing", "merged")

_TAG_RE = re.compile(r"\[([a-z0-9-]+)\]\s*$")


def _json_block(text: str) -> tuple[Any, str | None]:
    """Extract and parse the first JSON object in a fenced block, or bare.

    Args:
        text: The model output.

    Returns:
        ``(parsed, None)`` or ``(None, reason)``.
    """
    block = extract_first_code_block(text, ignore_language=True)
    candidate = block.strip() if block else text.strip()
    # A model that wraps JSON in prose still yields a parseable object if we take the
    # outermost braces.
    if not candidate.startswith(("{", "[")):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            return None, "no JSON object found in output"
        candidate = candidate[start : end + 1]
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as e:
        return None, f"output is not valid JSON: {e}"


# ----------------------------------------------------------------------------
# P1 -- topic to question
# ----------------------------------------------------------------------------


def parse_question(text: str) -> tuple[str | None, str | None]:
    """Extract P1's bracketed question.

    Args:
        text: P1's output.

    Returns:
        ``(question, None)`` or ``(None, reason)``.
    """
    inner = extract_first_square_brackets(text or "")
    if not inner:
        return None, "no bracketed question in output"
    q = " ".join(inner.split())
    if len(q) < 20:
        return None, f"question is implausibly short ({len(q)} chars): {q!r}"
    if not q.endswith("?"):
        return None, "question does not end with '?'"
    return q, None


# ----------------------------------------------------------------------------
# P2 -- atomic claims
# ----------------------------------------------------------------------------


def parse_claims(text: str) -> tuple[list[dict[str, str]] | None, str | None]:
    """Extract P2's tagged claim list.

    Args:
        text: P2's output -- a markdown code block of ``-`` bullets, each ending in a
            bracketed tag.

    Returns:
        ``([{"text": ..., "tag": ...}], None)`` or ``(None, reason)``.
    """
    block = extract_first_code_block(text or "", ignore_language=True) or (text or "")
    claims: list[dict[str, str]] = []
    for line in block.split("\n"):
        line = line.strip()
        if not line.startswith("-"):
            continue
        line = line[1:].strip()
        m = _TAG_RE.search(line)
        if not m:
            return None, f"claim has no [tag]: {line[:60]!r}"
        tag = m.group(1)
        if tag not in CLAIM_TAGS:
            return (
                None,
                f"unknown claim tag [{tag}] (expected one of {list(CLAIM_TAGS)})",
            )
        body = _TAG_RE.sub("", line).strip()
        if not body:
            return None, f"claim [{tag}] has no text"
        claims.append({"text": body, "tag": tag})

    if not claims:
        return None, "no '-' bulleted claims found"

    # Phase 1 P2 instruction 3 mandates the composition; the worked exemplar in the
    # appendix is abbreviated, so the instruction is what we validate against
    # (Phase 2, ambiguity 3).
    required_pairs = [
        ("alt-pair-1", "alt-pair-2"),
        ("disj-pair-1", "disj-pair-2"),
        ("equiv-pair-1", "equiv-pair-2"),
    ]
    tags = [c["tag"] for c in claims]
    for a, b in required_pairs:
        if tags.count(a) != 1 or tags.count(b) != 1:
            return None, f"expected exactly one [{a}] and one [{b}]"
    if tags.count("holding") < 1:
        return None, "no [holding] claim; a resolved Concession needs one"
    return claims, None


# ----------------------------------------------------------------------------
# P3 -- the relation plan
# ----------------------------------------------------------------------------


def parse_plan(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Extract and structurally validate P3's relation plan.

    Checks shape, counts, legal senses, the sense/coupling agreement with
    ``taxonomy.COMPILE``, band names, error kinds, and resolver references. Does *not*
    check the rare-facet floor, the window constraint or the validity split -- those are
    gates and live in :mod:`fact_reasoner.locobench.validate`.

    Args:
        text: P3's output.

    Returns:
        ``(plan, None)`` or ``(None, reason)``.
    """
    plan, err = _json_block(text or "")
    if err:
        return None, err
    if not isinstance(plan, dict):
        return None, f"plan must be a JSON object, got {type(plan).__name__}"

    for key in ("atoms", "relations", "non_relations"):
        if key not in plan:
            return None, f"plan is missing {key!r}"
        if not isinstance(plan[key], list):
            return None, f"plan[{key!r}] must be a list"

    atoms = plan["atoms"]
    lo, hi = N_CLAIMS_RANGE
    if not lo <= len(atoms) <= hi:
        return None, f"plan has {len(atoms)} atoms, expected {lo}-{hi}"

    positions: set[int] = set()
    for a in atoms:
        if not isinstance(a, dict) or "pos" not in a or "text" not in a:
            return None, f"atom must be an object with 'pos' and 'text': {a!r}"
        pos = a["pos"]
        if not isinstance(pos, int) or pos < 1:
            return None, f"atom 'pos' must be a positive int, got {pos!r}"
        if pos in positions:
            return None, f"duplicate atom position {pos}"
        positions.add(pos)
    if positions != set(range(1, len(atoms) + 1)):
        return None, f"atom positions must be 1..{len(atoms)} with no gaps"

    rels = plan["relations"]
    lo, hi = N_RELATIONS_RANGE
    if not lo <= len(rels) <= hi:
        return None, f"plan has {len(rels)} relations, expected {lo}-{hi}"

    for r in rels:
        if not isinstance(r, dict):
            return None, f"relation must be an object: {r!r}"
        for key in ("source_pos", "target_pos", "sense", "strength_band", "validity"):
            if key not in r:
                return None, f"relation is missing {key!r}: {r!r}"
        if r["source_pos"] not in positions or r["target_pos"] not in positions:
            return None, f"relation references an unknown position: {r!r}"
        if r["source_pos"] == r["target_pos"]:
            return None, f"relation is a self-loop: {r!r}"
        if r["sense"] not in LEGAL_SENSES:
            return None, f"unknown sense {r['sense']!r}"
        # The plan may name a coupling; if it does it must agree with COMPILE, which is
        # the single authority (Phase 1 Section 3.4).
        expected = coupling_for_sense(r["sense"])
        if "coupling" in r and r["coupling"] != expected:
            return None, (
                f"relation claims coupling {r['coupling']!r} for sense "
                f"{r['sense']!r}, but COMPILE says {expected!r}"
            )
        if r["strength_band"] not in STRENGTH_BANDS:
            return None, f"unknown strength_band {r['strength_band']!r}"
        if r["validity"] not in ("valid", "invalid"):
            return None, f"validity must be 'valid' or 'invalid', got {r['validity']!r}"
        if r["validity"] == "invalid":
            if r.get("error_kind") not in ERROR_KINDS:
                return None, (
                    f"an invalid relation needs an error_kind from {list(ERROR_KINDS)}, "
                    f"got {r.get('error_kind')!r}"
                )
        if r.get("resolved") and r.get("resolver_pos") not in positions:
            return None, f"resolved concession has no valid resolver_pos: {r!r}"

    nons = plan["non_relations"]
    lo, hi = N_NON_RELATIONS_RANGE
    if not lo <= len(nons) <= hi:
        return None, f"plan has {len(nons)} non_relations, expected {lo}-{hi}"
    related = {(r["source_pos"], r["target_pos"]) for r in rels}
    for nr in nons:
        if not isinstance(nr, dict) or "source_pos" not in nr or "target_pos" not in nr:
            return None, f"non_relation needs 'source_pos' and 'target_pos': {nr!r}"
        pair = (nr["source_pos"], nr["target_pos"])
        if pair in related or pair[::-1] in related:
            return None, f"non_relation {pair} is also a planned relation"
    return plan, None


# ----------------------------------------------------------------------------
# P4 / P5 -- prose, and the perturbation diff
# ----------------------------------------------------------------------------


def parse_response(text: str, *, min_words: int = 500) -> tuple[str | None, str | None]:
    """Extract P4's response prose.

    Args:
        text: P4's output, a code block per instruction 8.
        min_words: Phase-1 P4 instruction 6's floor.

    Returns:
        ``(response, None)`` or ``(None, reason)``.
    """
    block = extract_first_code_block(text or "", ignore_language=True)
    resp = (block or text or "").strip()
    if not resp:
        return None, "empty response"
    n = len(resp.split())
    if n < min_words:
        return None, f"response is {n} words, below the {min_words}-word floor"
    return resp, None


def parse_perturbation(
    text: str,
) -> tuple[tuple[str, dict[str, Any]] | None, str | None]:
    """Extract P5's perturbed response and its JSON diff.

    P5 emits two code blocks: the prose, then the diff. Taking the first block as prose
    and the last JSON object as the diff tolerates a model that adds commentary between
    them.

    Args:
        text: P5's output.

    Returns:
        ``((response, diff), None)`` or ``(None, reason)``.
    """
    raw = text or ""
    block = extract_first_code_block(raw, ignore_language=True)
    if not block:
        return None, "no code block in output (expected the perturbed response)"
    response = block.strip()
    if not response:
        return None, "perturbed response block is empty"

    rest = raw.split(block, 1)[-1]
    diff, err = _json_block(rest)
    if err:
        return None, f"could not read the JSON diff: {err}"
    if not isinstance(diff, dict) or "operator" not in diff:
        return None, "diff must be a JSON object with an 'operator' field"
    return (response, diff), None


# ----------------------------------------------------------------------------
# V1 -- V4
# ----------------------------------------------------------------------------


def parse_recovery(text: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Extract V1's recovered relation list.

    Args:
        text: V1's output -- a JSON list, per its function-call contract.

    Returns:
        ``(relations, None)`` or ``(None, reason)``. An empty list is valid: it means
        the model recovered nothing, which is a real (and gate-failing) verdict.
    """
    parsed, err = _json_block(text or "")
    if err:
        # V1 returns a bare list, so retry the brackets before giving up.
        raw = (text or "").strip()
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end <= start:
            return None, err
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as e:
            return None, f"output is not valid JSON: {e}"
    if not isinstance(parsed, list):
        return None, f"V1 must return a JSON list, got {type(parsed).__name__}"
    for r in parsed:
        if not isinstance(r, dict):
            return None, f"recovered relation must be an object: {r!r}"
        for key in ("source", "target", "sense", "coupling"):
            if key not in r:
                return None, f"recovered relation is missing {key!r}: {r!r}"
        # Endpoints must be integers -- the keys of the atoms mapping V1 was handed. A
        # string id ("a0") or a numeric string ("1") would never compare equal to a plan
        # position, so it would silently score as a total recovery failure rather than as
        # the malformed output it is. Coerce a numeric string, reject anything else.
        for key in ("source", "target"):
            val = r[key]
            if isinstance(val, bool) or not isinstance(val, int):
                if isinstance(val, str) and val.strip().lstrip("-").isdigit():
                    r[key] = int(val.strip())
                    continue
                return None, (
                    f"recovered {key!r} must be an atom number (an integer key of the "
                    f"atoms mapping), got {val!r}"
                )
        if r["source"] < 1 or r["target"] < 1:
            return None, (
                f"recovered endpoints are numbered from 1, got "
                f"source={r['source']!r} target={r['target']!r}"
            )
        if r["sense"] not in LEGAL_SENSES:
            return None, f"unknown recovered sense {r['sense']!r}"
    return parsed, None


def parse_verdict(text: str) -> tuple[str | None, str | None]:
    """Extract V2's single-letter exhaustiveness verdict.

    Args:
        text: V2's output.

    Returns:
        ``(letter, None)`` or ``(None, reason)``.
    """
    s = (text or "").strip().strip('"').strip("'").strip()
    if not s:
        return None, "empty verdict"
    letter = s[0].upper()
    if letter not in V2_VERDICTS:
        return None, f"verdict must be one of {list(V2_VERDICTS)}, got {s[:20]!r}"
    return letter, None


def parse_audit(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Extract V3's naturalness/leakage audit.

    Args:
        text: V3's output.

    Returns:
        ``(audit, None)`` or ``(None, reason)``.
    """
    audit, err = _json_block(text or "")
    if err:
        return None, err
    if not isinstance(audit, dict):
        return None, f"V3 must return an object, got {type(audit).__name__}"
    for key in ("fluency", "formality", "organization"):
        v = audit.get(key)
        if not isinstance(v, (int, float)) or not 1 <= v <= 5:
            return None, f"V3 {key!r} must be a number in 1..5, got {v!r}"
    for key in ("leakage", "hedging", "artifacts"):
        if not isinstance(audit.get(key, []), list):
            return None, f"V3 {key!r} must be a list of spans"
        audit.setdefault(key, [])
    return audit, None


def parse_coverage(text: str) -> tuple[list[dict[str, Any]] | None, str | None]:
    """Extract V4's per-atom coverage report.

    Args:
        text: V4's output.

    Returns:
        ``(entries, None)`` or ``(None, reason)``.
    """
    parsed, err = _json_block(text or "")
    if err:
        raw = (text or "").strip()
        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end <= start:
            return None, err
        try:
            parsed = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as e:
            return None, f"output is not valid JSON: {e}"
    if not isinstance(parsed, list):
        return None, f"V4 must return a JSON list, got {type(parsed).__name__}"
    for e in parsed:
        if not isinstance(e, dict) or "index" not in e or "status" not in e:
            return None, f"coverage entry needs 'index' and 'status': {e!r}"
        if e["status"] not in V4_STATUSES:
            return None, f"unknown V4 status {e['status']!r}"
    return parsed, None


PARSERS = {
    "P1": parse_question,
    "P2": parse_claims,
    "P3": parse_plan,
    "P4": parse_response,
    "P5": parse_perturbation,
    "V1": parse_recovery,
    "V2": parse_verdict,
    "V3": parse_audit,
    "V4": parse_coverage,
}

__all__ = [
    "CLAIM_TAGS",
    "ERROR_KINDS",
    "N_CLAIMS_RANGE",
    "N_NON_RELATIONS_RANGE",
    "N_RELATIONS_RANGE",
    "PARSERS",
    "STRENGTH_BANDS",
    "V2_VERDICTS",
    "V4_STATUSES",
    "parse_audit",
    "parse_claims",
    "parse_coverage",
    "parse_perturbation",
    "parse_plan",
    "parse_question",
    "parse_recovery",
    "parse_response",
    "parse_verdict",
]
