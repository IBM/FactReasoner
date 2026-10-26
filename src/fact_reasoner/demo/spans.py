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

# Claim -> response-span alignment for the demo.
#
# Atoms are decontextualized rewrites ("Voyager 1 is therefore in interstellar
# space"), not offsets into the response, so nothing upstream says WHERE in the
# text a claim lives. The demo needs that to highlight it. Alignment is lexical
# and runs in two steps:
#
#   1. anchor -- choose the run of 1..MAX_RUN consecutive sentences whose content
#      words best cover the claim's. Words are weighted down both by how many
#      sentences and by how many claims contain them, so the response's recurring
#      subject ("Voyager 1"), which a rewrite re-inserts everywhere, does not
#      decide the anchor -- the claim's distinguishing words do;
#   2. highlight -- inside the anchor, mark the tokens that match a claim word and
#      merge matches separated by at most MAX_GAP tokens into segments. A claim
#      drawn from a list ("credits include ... The Abyss") thus highlights the
#      head and its item, not the whole list. Segments carrying under
#      MIN_SEGMENT_SHARE of the matched weight are dropped as incidental matches.
#
# A claim whose best anchor covers under ``min_score`` of its weight is reported
# implicit: listed, but not pinned to a wrong span. Deterministic, dependency-free
# and free of LLM calls.

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

_SENTENCE_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")

_STOPWORDS = frozenset(
    """a an the of in on at to for from by with and or but not no is are was were be
    been being has have had do does did it its this that these those there their they
    he she his her him we our you your i as than then so such which who whom whose
    what when where why how also into over after before about all any both each more
    most other some very can could would should may might must will shall just only
    same own too s""".split()
)

#: The longest sentence run considered as one claim's anchor.
MAX_RUN = 2
#: Tokens allowed between two matches for them to share a highlight segment.
MAX_GAP = 3
#: Segments below this share of the claim's matched weight are dropped.
MIN_SEGMENT_SHARE = 0.2


@dataclass
class ClaimSpan:
    """Where one claim sits in the response.

    Attributes:
        atom_id: The claim's id.
        start / end: Character offsets of the whole highlighted region (``end``
            exclusive), or None when the claim is implicit.
        segments: The highlighted ``[start, end]`` pieces inside that region.
        score: Weighted share of the claim's content words found, in [0, 1].
        method: ``"lexical"`` or ``"implicit"``.
    """

    atom_id: str
    start: int | None
    end: int | None
    score: float
    method: str
    segments: list[list[int]] = field(default_factory=list)

    def to_json(self) -> dict:
        return asdict(self)


def _stem(word: str) -> str:
    """Lower-case and crudely stem (plural / -ed / -ing / possessive)."""
    w = word.lower()
    if w.endswith("'s"):
        w = w[:-2]
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _is_content(stem: str) -> bool:
    return len(stem) >= 2 and stem not in _STOPWORDS


def split_sentences(response: str) -> list[tuple[int, int]]:
    """Sentence offsets ``(start, end)`` in text order, trimmed of whitespace."""
    spans, pos = [], 0
    for m in _SENTENCE_RE.finditer(response):
        spans.append((pos, m.start() + len(m.group(0).rstrip())))
        pos = m.end()
    spans.append((pos, len(response)))
    out = []
    for a, b in spans:
        while a < b and response[a].isspace():
            a += 1
        while b > a and response[b - 1].isspace():
            b -= 1
        if b > a:
            out.append((a, b))
    return out


def align_claims(
    response: str,
    claims: dict[str, str],
    *,
    min_score: float = 0.4,
) -> dict[str, ClaimSpan]:
    """Align each claim to highlighted segments of the response.

    Args:
        response: The response the claims were extracted from.
        claims: ``{atom_id: claim_text}``. Pass the atom's ORIGINAL (pre-revision)
            text when available -- it is closer to the response wording.
        min_score: The weighted coverage an anchor must reach to be accepted.

    Returns:
        ``{atom_id: ClaimSpan}``, one entry per claim.
    """
    sentences = split_sentences(response)
    # Per sentence: [(stem, start, end)] for every token.
    tokens = [
        [(_stem(m.group(0)), a + m.start(), a + m.end())
         for m in _TOKEN_RE.finditer(response[a:b])]
        for a, b in sentences
    ]
    sent_words = [{t[0] for t in toks if _is_content(t[0])} for toks in tokens]
    df: dict[str, int] = {}
    for ws in sent_words:
        for w in ws:
            df[w] = df.get(w, 0) + 1

    claim_words = {
        aid: {s for s in map(_stem, _TOKEN_RE.findall(text)) if _is_content(s)}
        for aid, text in claims.items()
    }
    # Claim frequency: a word nearly every claim repeats (the subject a reviser
    # re-inserts) says little about where any ONE claim lives.
    cf: dict[str, int] = {}
    for ws in claim_words.values():
        for w in ws:
            cf[w] = cf.get(w, 0) + 1
    n_claims = max(1, len(claims))

    def weight(w: str) -> float:
        claim_share = cf.get(w, 1) / n_claims
        return (1.0 / df.get(w, 1)) * (1.0 - 0.8 * claim_share)

    out: dict[str, ClaimSpan] = {}
    for aid in claims:
        words = claim_words[aid]
        if not words or not sentences:
            out[aid] = ClaimSpan(aid, None, None, 0.0, "implicit")
            continue
        total = sum(weight(w) for w in words)

        # 1. anchor: best-covering sentence run, shorter runs winning near-ties.
        best = (-1.0, 0.0, 0, 0)  # (objective, recall, i, j)
        for i in range(len(sentences)):
            covered: set[str] = set()
            for j in range(i, min(i + MAX_RUN, len(sentences))):
                covered |= sent_words[j]
                recall = sum(weight(w) for w in words & covered) / total
                objective = recall - 0.1 * (j - i)
                if objective > best[0]:
                    best = (objective, recall, i, j)
        _, recall, i, j = best
        if recall < min_score:
            out[aid] = ClaimSpan(aid, None, None, round(recall, 3), "implicit")
            continue

        # 2. highlight: merge nearby matching tokens into weighted segments.
        run = [t for k in range(i, j + 1) for t in tokens[k]]
        segs: list[list] = []  # [start, end, last_token_index, weight]
        for idx, (stem, a, b) in enumerate(run):
            if stem not in words or not _is_content(stem):
                continue
            # Bridge a gap of stopwords freely, but at most one unmatched content
            # word: two of them mean we are crossing other list items.
            gap = run[segs[-1][2] + 1 : idx] if segs else []
            foreign = sum(1 for t in gap if _is_content(t[0]))
            if segs and len(gap) <= MAX_GAP and foreign <= 1:
                segs[-1][1], segs[-1][2] = b, idx
                segs[-1][3] += weight(stem)
            else:
                segs.append([a, b, idx, weight(stem)])
        matched = sum(s[3] for s in segs) or 1.0
        kept = [s for s in segs if s[3] / matched >= MIN_SEGMENT_SHARE] or segs
        segments = [[s[0], s[1]] for s in kept]
        out[aid] = ClaimSpan(
            aid, segments[0][0], segments[-1][1], round(recall, 3), "lexical", segments
        )
    return out


def locate_evidence(response: str, evidence: str | None) -> tuple[int, int] | None:
    """Find a miner-cited evidence quote in the response (case/space-insensitive).

    Args:
        response: The response text.
        evidence: The verbatim span the relation miner quoted, or None.

    Returns:
        ``(start, end)`` offsets, or None when absent or not found.
    """
    if not evidence or not evidence.strip():
        return None
    pattern = r"\s+".join(re.escape(tok) for tok in evidence.strip().split())
    m = re.search(pattern, response, re.IGNORECASE)
    return (m.start(), m.end()) if m else None


__all__ = ["ClaimSpan", "align_claims", "locate_evidence", "split_sentences"]
