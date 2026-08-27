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
# Narrowed from (14, 18). P4's output length degrades as the atom count rises, because the
# model compresses to fit rather than writing more: measured on gpt-oss-120b, 16 atoms gave
# 581 and 596 words (~36 words/atom), 17 gave 516 and 543 (~31), and 18 gave **293** (~16) --
# below the 500-word floor, so the family died at `P4: SamplingFailed` however good the plan
# was. 18 atoms is a regime where P4 reliably breaks, and nothing in the benchmark needs it:
# the corpus property that matters is the relation graph, not atom count, and P3's own worked
# example uses 14. Keeping the floor at 14 preserves that example as an independent witness.
N_CLAIMS_RANGE = (14, 16)
N_RELATIONS_RANGE = (8, 12)
N_NON_RELATIONS_RANGE = (4, 6)

# The hedge words V3 flags, as the CANONICAL list -- the three prompt strings that recite it
# (P2, P4 instruction 7, V3) previously had no constant behind them, so they could drift.
#
# Enforced by `parse_response` because P4 and V3 disagree on SCOPE and neither side should
# move: P4 instruction 7 warns about these only "around planned-invalid relations", while V3
# flags them anywhere. Measured on gpt-oss-120b, r5 f001: the sole remaining rejection was
# `"possibly for body painting"` -- ordinary descriptive prose, so P4 permitted exactly what
# V3 rejects, and both auditors correctly flagged it. Widening P4's scope is the obvious fix
# and was tried before: it is guarded against by a test because added prohibitions measurably
# suppressed output (581 -> 308 -> 144 words). Checking here instead costs nothing and needs
# no prompt change -- the parser IS the rejection-sampling predicate (`build_llm` installs it
# as the Mellea requirement), so a hedge is re-sampled inside the same P4 call, against
# exactly the criterion V3 later applies.
HEDGE_WORDS = (
    "assume",
    "might",
    "possibly",
    "allegedly",
    "supposedly",
    "reportedly",
    "it is claimed",
)

STRENGTH_BANDS = {
    "strong": (0.85, 1.00),
    "moderate": (0.60, 0.84),
    "weak": (0.35, 0.59),
}

ERROR_KINDS = ("wrong_sense", "wrong_direction", "false_endpoint", "spurious")

# V4's four statuses.
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
        # `factual` is OPTIONAL here and authoritative nowhere: the pipeline derives it from
        # P2's [correct]/[incorrect] tag by text match, because the tag is the ground truth
        # and a model restating it can only drift. Type-checked when present so a string
        # "false" -- which is truthy -- cannot silently mark a false claim as true.
        if "factual" in a and not isinstance(a["factual"], bool):
            return None, (
                f"atom 'factual' must be true or false, got {a['factual']!r} at pos {pos}"
            )
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

    # At most ONE relation per atom pair, in either direction. The gold schema requires it
    # (`schema.validate_item` rejects a duplicate outright) because the Markov network
    # builds one factor per pair, so two edges over the same pair have no unambiguous
    # factor table. The parser did not enforce it and P3 never stated it, which made this a
    # stage-contract mismatch rather than a model error: a live gpt-oss plan carried two
    # relations over one pair, passed every gate -- plan, V1, V3, V4 and all seven P5
    # perturbations -- and then died at serialization with the whole family's work spent.
    # Enforced here so it is caught at the plan stage, where `_Caller.ask` can feed the
    # complaint back and re-plan, instead of terminally after the expensive stages.
    # Direction-insensitive because the pair, not the ordered pair, is what indexes the
    # factor: an (i,j) and a (j,i) edge collide just as surely.
    seen_pairs: dict[tuple[int, int], str] = {}
    for r in rels:
        key = (
            min(r["source_pos"], r["target_pos"]),
            max(r["source_pos"], r["target_pos"]),
        )
        if key in seen_pairs:
            return None, (
                f"atoms {key[0]} and {key[1]} carry two relations "
                f"({seen_pairs[key]!r} and {r['sense']!r}); each pair of atoms may take at "
                "most one relation, in one direction"
            )
        seen_pairs[key] = r["sense"]

    # A CONFLICT ladder's deepest rung applies `add_resolution` plus TWO DISTINCT
    # `drop_relation` calls, and only conflict-coupled edges are eligible for either. The
    # resolution consumes one, so the plan needs >= 3 conflict edges that are NOT already
    # resolved, i.e. >= 4 conflict edges with at most one pre-resolved.
    #
    # Enforced here rather than left to the prompt because prose did not work: P3
    # instruction 5 was extended to demand exactly this and a live model ignored it,
    # planning 3 conflict edges (one resolved) on six consecutive families across five
    # topics -- droppable 2, rejected 5/5 at `P5.rung4.edge_effect` after the whole
    # respond stage had been paid for. Only three senses compile to a conflict coupling
    # (Alternative -> exclusive, Concession/Contrast -> contradiction), so instruction 5's
    # one-of-each mandate yields exactly three and the mandated minimum was itself the
    # failing shape. As a parser error the complaint reaches `_Caller.ask`, which feeds it
    # back via `_retry_note` and re-plans within the same call -- the same reasoning as the
    # duplicate-pair check above.
    conflicts = [
        r
        for r in rels
        if coupling_for_sense(r["sense"]) in ("contradiction", "exclusive")
    ]
    unresolved = [r for r in conflicts if not r.get("resolved")]
    if len(unresolved) < 3:
        return None, (
            f"plan has {len(conflicts)} conflicting relation(s) "
            f"({sorted(r['sense'] for r in conflicts)}) of which only {len(unresolved)} "
            "are unresolved; a CONFLICT ladder needs AT LEAST 4 relations with a "
            "conflicting sense (Alternative, Concession or Contrast) and AT MOST 1 of them "
            "marked resolved, because its deepest rung resolves one and then drops two "
            "others. Only three senses conflict, so use one of them TWICE on a different "
            "pair of claims"
        )

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
    # P4 is the ONLY prompt whose code block holds prose rather than JSON -- P3 and P5
    # both say "JSON object in a code block", so a model readily generalizes the wrong
    # convention here and emits `{"response": "..."}`. Nothing caught that: the prose is
    # taken verbatim, and `ignore_language=True` strips a ```json tag but not the object
    # inside it, so a wrapped answer entered the corpus with a literal `{"response": "`
    # prefix and `\n` escapes as response text. Measured on a live deepseek-v3.2 run,
    # where it was masked by the word floor -- the answer was ALSO too short, so only the
    # length was reported and the format defect stayed invisible. Hence this check comes
    # BEFORE the word count: the wrapper is the more fundamental complaint, and naming it
    # is what lets `_Caller.ask` feed back something the model can act on.
    if resp[0] in "{[":
        return None, (
            "response must be the prose itself, not a JSON object wrapping it "
            "(the code block holds the answer, not a payload)"
        )
    n = len(resp.split())
    if n < min_words:
        return None, f"response is {n} words, below the {min_words}-word floor"
    # Hedges last, so the structural complaints above are reported first and a hedge is never
    # what a caller sees while the prose is also malformed or short.
    hedges = sorted(
        {
            w
            for w in HEDGE_WORDS
            if re.search(rf"\b{re.escape(w)}\b", resp, re.IGNORECASE)
        }
    )
    if hedges:
        # Word-bounded on purpose: "perhaps both" is MANDATED by P4 instruction 3 for
        # Disjunction, and an unbounded scan for "possibly" would also be wrong here anyway
        # -- it is "and possibly both" that P4 steers away from, via "perhaps both".
        return None, (
            f"response hedges with {hedges}: state every planned relation in the same "
            "confident register, whether or not it is sound, and use no hedging word "
            "anywhere in the prose"
        )
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
    "V3": parse_audit,
    "V4": parse_coverage,
}

__all__ = [
    "CLAIM_TAGS",
    "ERROR_KINDS",
    "HEDGE_WORDS",
    "N_CLAIMS_RANGE",
    "N_NON_RELATIONS_RANGE",
    "N_RELATIONS_RANGE",
    "PARSERS",
    "STRENGTH_BANDS",
    "V4_STATUSES",
    "parse_audit",
    "parse_claims",
    "parse_coverage",
    "parse_perturbation",
    "parse_plan",
    "parse_question",
    "parse_recovery",
    "parse_response",
]
