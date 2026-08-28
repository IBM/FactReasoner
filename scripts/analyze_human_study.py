#!/usr/bin/env python
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

"""Analyse the LoCoBench human study.

Reports, in the order the paper needs them:

1. **Inter-annotator agreement** -- Krippendorff's alpha (nominal), overall and split by
   task kind. The increase and invariance screens are different questions and may agree
   very differently, so pooling them would hide exactly the contrast we care about.
   Screens with no majority are named rather than resolved: on a corpus whose ground
   truth is declared by construction, "readers do not agree here" is a finding about the
   item, not noise to be averaged away.
2. **Human versus the declared ordering** -- does the corpus's constructed ground truth
   match reader judgement? This is what the study exists to answer.
3. **Human versus each measure** -- the human-referenced version of the paper's
   `tab:summary` Increase column.
4. **Reason coding** -- whether the free-text reasons name the claims the perturbation
   record says were actually edited. Cheap evidence that readers responded to relational
   structure rather than to surface fluency.

The analysis plan is fixed before the data arrives (see the plan file), and BOTH
invariance outcomes are pre-declared publishable: if readers call reordered rungs equal,
the invariance requirement is validated; if they see a real difference, the requirement is
too strict. Neither result is a failure.

Run::

    python scripts/analyze_human_study.py --study-dir artifacts/human_study
"""

from __future__ import annotations

import argparse
import collections
import glob
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CHOICES = ("A", "B", "equal")


def krippendorff_alpha_nominal(units: list[list[str]]) -> float | None:
    """Krippendorff's alpha for nominal data, allowing missing ratings.

    Implemented directly rather than pulled in as a dependency: the nominal case is the
    coincidence-matrix definition and is short enough to verify by hand, which matters
    for a number that will be quoted in a paper.

    Uses the standard formulation over the coincidence matrix: units contribute pairs of
    ratings weighted by ``1 / (m_u - 1)`` where ``m_u`` is that unit's number of ratings,
    so a unit rated by fewer annotators contributes proportionally less rather than being
    dropped.

    Args:
        units: One list of category labels per unit. Entries may be shorter than the
            annotator count when a rating is missing; units with fewer than two ratings
            are skipped, as they carry no information about agreement.

    Returns:
        Alpha in ``(-inf, 1]``, or None when fewer than two units have >= 2 ratings, or
        when every rating in the study is the same category (alpha is undefined there --
        expected disagreement is zero, so the ratio has no meaning).
    """
    usable = [u for u in units if len(u) >= 2]
    if len(usable) < 2:
        return None

    cats = sorted({c for u in usable for c in u})
    if len(cats) < 2:
        # No variation at all: perfect agreement, but alpha's denominator is 0.
        return None
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)

    # Coincidence matrix: for each unit, every ORDERED pair of distinct rating slots.
    coinc = [[0.0] * k for _ in range(k)]
    for u in usable:
        m = len(u)
        w = 1.0 / (m - 1)
        for a, b in itertools.permutations(range(m), 2):
            coinc[idx[u[a]]][idx[u[b]]] += w

    n_total = sum(sum(row) for row in coinc)
    if n_total <= 0:
        return None

    # Observed disagreement: off-diagonal mass.
    do = sum(coinc[i][j] for i in range(k) for j in range(k) if i != j) / n_total
    # Expected disagreement from the marginals.
    marg = [sum(coinc[i]) for i in range(k)]
    de = sum(
        marg[i] * marg[j] for i in range(k) for j in range(k) if i != j
    ) / (n_total * (n_total - 1))
    if de == 0:
        return None
    return 1.0 - do / de


def majority(labels: list[str]) -> str | None:
    """The strict majority label, or None when there is none.

    Returns None on a three-way split or a tie, deliberately: forcing a verdict where
    readers genuinely divide would manufacture ground truth the study does not have.
    """
    if not labels:
        return None
    counts = collections.Counter(labels)
    top, n = counts.most_common(1)[0]
    if n * 2 > len(labels):
        return top
    return None


def load_responses(study_dir: str) -> dict[str, dict[str, dict]]:
    """Load ``responses_<annotator>.jsonl`` into ``{screen_id: {annotator: row}}``.

    Each row is expected to carry ``screen_id``, ``choice`` (one of A / B / equal), and
    optionally ``confidence``, ``why`` and ``noticed_reordering``.
    """
    out: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    paths = sorted(glob.glob(os.path.join(study_dir, "responses_*.jsonl")))
    if not paths:
        raise SystemExit(
            f"[human-study] no responses_*.jsonl in {study_dir}. Expected one file per "
            "annotator, each line {'screen_id':..., 'choice':'A'|'B'|'equal', ...}."
        )
    for path in paths:
        who = os.path.basename(path)[len("responses_") : -len(".jsonl")]
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                choice = row.get("choice")
                if choice not in CHOICES:
                    raise SystemExit(
                        f"[human-study] {path}: bad choice {choice!r} on "
                        f"{row.get('screen_id')!r}; expected one of {list(CHOICES)}"
                    )
                out[row["screen_id"]][who] = row
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-dir", default="artifacts/human_study")
    p.add_argument("--data-dir", default="data/locobench-claude-5-test")
    args = p.parse_args()

    key_path = os.path.join(args.study_dir, "answer_key.jsonl")
    if not os.path.exists(key_path):
        raise SystemExit(f"[human-study] missing {key_path}")
    key = {}
    with open(key_path) as f:
        for line in f:
            row = json.loads(line)
            key[row["screen_id"]] = row

    resp = load_responses(args.study_dir)
    annotators = sorted({w for r in resp.values() for w in r})
    print(f"[human-study] {len(annotators)} annotator(s): {', '.join(annotators)}")
    print(f"[human-study] {len(resp)} of {len(key)} screens answered")
    print()

    # ---- 1. agreement ------------------------------------------------------
    def units_for(kind: str | None) -> list[list[str]]:
        out = []
        for sid, k in key.items():
            if kind is not None and k["kind"] != kind:
                continue
            labels = [r["choice"] for r in resp.get(sid, {}).values()]
            out.append(labels)
        return out

    print("=== 1. inter-annotator agreement (Krippendorff alpha, nominal) ===")
    for label, kind in (("all screens", None), ("increase", "increase"),
                        ("invariance", "invariance")):
        a = krippendorff_alpha_nominal(units_for(kind))
        n = len([u for u in units_for(kind) if len(u) >= 2])
        print(f"  {label:<14} alpha = "
              f"{'undefined' if a is None else f'{a:+.3f}'}   ({n} screens)")
    nomaj = [sid for sid in key
             if majority([r["choice"] for r in resp.get(sid, {}).values()]) is None]
    print(f"  no majority   : {len(nomaj)} screen(s)"
          + (f" -> {', '.join(sorted(nomaj))}" if nomaj else ""))
    print("  (a screen with no majority is reported as genuinely ambiguous, not resolved)")
    print()

    # ---- 2. human vs the declared ordering ---------------------------------
    print("=== 2. human majority vs the corpus's declared ordering ===")
    agree = collections.Counter()
    total = collections.Counter()
    rows = []
    for sid, k in sorted(key.items()):
        labels = [r["choice"] for r in resp.get(sid, {}).values()]
        maj = majority(labels)
        declared = k["declared_answer"]
        total[k["kind"]] += 1
        hit = maj is not None and maj == declared
        if hit:
            agree[k["kind"]] += 1
        rows.append((sid, k, labels, maj, declared, hit))
    print("  %-6s %-6s %-9s %-11s %-5s %-9s %-8s %s"
          % ("screen", "fam", "ladder", "kind", "pair", "declared", "majority", "match"))
    for sid, k, labels, maj, declared, hit in rows:
        print("  %-6s %-6s %-9s %-11s %-5s %-9s %-8s %s"
              % (sid, k["family_id"], k["ladder"], k["kind"],
                 f"{k['lower_rung']}-{k['higher_rung']}", declared,
                 maj or "(none)", "yes" if hit else "no"))
    print()
    for kind in ("increase", "invariance"):
        if total[kind]:
            print(f"  {kind:<11} {agree[kind]}/{total[kind]} declared orderings "
                  f"confirmed by the human majority")
    print()
    print("  Reading the invariance row: a high number validates the invariance")
    print("  requirement and makes the LCS's 20/20 a human-referenced result; a low one")
    print("  says the requirement is too strict. Both were pre-declared publishable.")
    print()

    # ---- 3. confidence ------------------------------------------------------
    print("=== 3. confidence, by whether the majority matched the declared answer ===")
    buckets = collections.defaultdict(collections.Counter)
    for sid, k, labels, maj, declared, hit in rows:
        for r in resp.get(sid, {}).values():
            c = r.get("confidence")
            if c:
                buckets["match" if hit else "mismatch"][c] += 1
    for b in ("match", "mismatch"):
        if buckets[b]:
            tot = sum(buckets[b].values())
            parts = " ".join(f"{k}={v}" for k, v in sorted(buckets[b].items()))
            print(f"  {b:<9} n={tot:<4} {parts}")
    print("  (low confidence concentrated on mismatches means the disagreement is")
    print("   uncertainty; high confidence on mismatches means readers actively disagree)")
    print()

    # ---- 4. reason coding ---------------------------------------------------
    print("=== 4. do the stated reasons name the claims that were actually edited? ===")
    items = {}
    path = os.path.join(args.data_dir, "items.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                it = json.loads(line)
                exp = it.get("expected") or {}
                items[(exp.get("family_id"), exp.get("rung_index"))] = it
    named = checked = 0
    for sid, k, labels, maj, declared, hit in rows:
        it = items.get((k["family_id"], k["higher_rung"]))
        if not it:
            continue
        # The atom ids the perturbation record says this rung touched.
        touched = set()
        for eff in ((it.get("expected") or {}).get("perturbation") or {}).get(
            "edge_effects", []
        ):
            for tok in str(eff.get("detail", "")).replace("->", " ").split():
                if tok.startswith("a") and tok[1:].isdigit():
                    touched.add(tok)
        if not touched:
            continue
        texts = {a["id"]: a["text"] for a in it.get("atoms", [])}
        for r in resp.get(sid, {}).values():
            why = (r.get("why") or "").lower()
            if not why:
                continue
            checked += 1
            # A reason "names" an edited claim when it shares a distinctive content word
            # with that claim's text. Crude but honest, and reported as such.
            for aid in touched:
                words = {
                    w.strip(".,;:()").lower()
                    for w in texts.get(aid, "").split()
                    if len(w) > 6
                }
                if words & set(why.replace(",", " ").split()):
                    named += 1
                    break
    if checked:
        print(f"  {named}/{checked} reasons mention a content word from a claim the")
        print("  perturbation record says was edited. This is a keyword heuristic, not")
        print("  semantic matching, so treat it as a floor rather than a measurement.")
    else:
        print("  no codable reasons (empty `why` fields, or no perturbation record)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
