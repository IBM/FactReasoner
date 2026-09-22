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
5. **The invariance result split by ladder type.** Pooling ORDER and CONTROL into one
   "invariance" number hides the study's most informative contrast, because the two
   ladders perturb different things: CONTROL swaps a relation's direction, which is
   genuinely meaning-preserving, while ORDER shuffles *sentences*, which can strand an
   anaphor without touching a single claim or edge. Reported separately with Fisher's
   exact test on the 2x2, since a pooled fraction would average a validated invariance
   against a blind spot.
6. **Every measure against the human majority.** The human-referenced version of the
   paper's Increase column: each LCS arm and readout, and each baseline column, mapped to
   an A / B / equal verdict per screen under the same 1e-6 tie tolerance the ladder
   scorer uses, then compared to what readers said. This is the only place in the paper
   where a measure is scored against human judgement rather than against a constructed
   ground truth.

The analysis plan is fixed before the data arrives (see the plan file), and BOTH
invariance outcomes are pre-declared publishable: if readers call reordered rungs equal,
the invariance requirement is validated; if they see a real difference, the requirement is
too strict. Neither result is a failure.

The study directory is gitignored on purpose -- the answer key maps each screen back to its
rung and declared answer, so committing it beside the annotator forms would put the ground
truth one directory away from people being asked to judge blind. What IS tracked is this
script, the export/import tooling, and ``results/human_study/summary.json``, which holds
every number the paper cites. To reproduce from the raw exports::

    python scripts/import_label_studio.py \\
        --export artifacts/human_study/annotation_locobench_1.json \\
        --export artifacts/human_study/annotation_locobench_2.json \\
        --export artifacts/human_study/annotation_locobench_3.json \\
        --export artifacts/human_study/annotation_locobench_4.json
    python scripts/analyze_human_study.py \\
        --json-out results/human_study/summary.json

The second command self-checks against :data:`PUBLISHED`, so a run that no longer
reproduces the paper's table says so rather than quietly emitting different numbers.

Run::

    python scripts/analyze_human_study.py --study-dir artifacts/human_study
"""

from __future__ import annotations

import argparse
import collections
import glob
import itertools
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

CHOICES = ("A", "B", "equal")

#: Two scores closer than this count as equal, matching ``report_ladder_baselines.py`` so
#: a measure's human-referenced column and its ladder column are computed the same way.
TIE_TOLERANCE = 1e-6

#: The four readouts, in the order the paper reports them.
READOUTS = ("mean_marginal", "consistency", "reified", "log_partition")

#: The cells the paper prints, asserted rather than assumed. If the analysis drifts -- a
#: changed tie tolerance, a re-imported export, a different results tree -- the run says so
#: instead of quietly publishing different numbers under the same prose. Keys are the
#: report's own labels; values are ``(vs_human, vs_declared)`` as fractions.
PUBLISHED = {
    "alpha": {"all": 0.748, "increase": 0.744, "invariance": 0.667},
    "declared": {"increase": (8, 10), "invariance": (1, 4)},
    "ratings_increase": (32, 40),
    "invariance_split": {"ORDER": (0, 8), "CONTROL": (6, 8)},
    "fisher_p": 0.007,
    "measures": {
        "LCS[gold].mean_marginal": ((9, 12), (14, 14)),
        "LCS[gold].consistency": ((8, 12), (13, 14)),
        "LCS[mined:gpt-oss-120b-a100:bidirectional].mean_marginal": ((7, 12), (7, 14)),
        "LCS[mined:llama-3.3-70b-instruct:windowed].mean_marginal": ((3, 12), (4, 14)),
        "judge_direct@gpt-oss-120b-a100": ((8, 12), (5, 14)),
        "control_length@gpt-oss-120b-a100": ((7, 12), (9, 14)),
        "judge_geval@gpt-oss-120b-a100": ((6, 12), (7, 14)),
        "roscoe_sc_mean@gpt-oss-120b-a100": ((6, 12), (6, 14)),
        "discourse_rc@gpt-oss-120b-a100": ((4, 12), (8, 14)),
        "control_claim_count@gpt-oss-120b-a100": ((2, 12), (4, 14)),
        "nli_contradiction@gpt-oss-120b-a100": ((1, 12), (2, 14)),
    },
}


def check_published(summary: dict) -> list[str]:
    """Compare a run's numbers against the cells the paper prints.

    Returns:
        One human-readable line per disagreement; empty when the run reproduces the
        published table. Returned rather than raised, so a partial run (no results tree,
        a subset of annotators) reports drift without failing the analysis.
    """
    out: list[str] = []
    for k, want in PUBLISHED["alpha"].items():
        got = (summary.get("alpha") or {}).get(k)
        if got is None or round(got, 3) != want:
            out.append(f"alpha[{k}]: published {want}, got "
                       f"{'none' if got is None else round(got, 3)}")
    for k, want in PUBLISHED["declared"].items():
        got = tuple((summary.get("declared") or {}).get(k) or ())
        if got != want:
            out.append(f"declared[{k}]: published {want}, got {got or 'none'}")
    got = tuple((summary.get("significance") or {}).get("rating") or ())
    if got != PUBLISHED["ratings_increase"]:
        out.append(f"increase ratings: published "
                   f"{PUBLISHED['ratings_increase']}, got {got or 'none'}")
    for ladder, (eq, n) in PUBLISHED["invariance_split"].items():
        blk = (summary.get("invariance_split") or {}).get(ladder) or {}
        if (blk.get("n_equal"), blk.get("n_ratings")) != (eq, n):
            out.append(f"invariance[{ladder}]: published {eq}/{n}, got "
                       f"{blk.get('n_equal')}/{blk.get('n_ratings')}")
    fp = (summary.get("invariance_split") or {}).get("fisher_p")
    if fp is None or round(fp, 3) != PUBLISHED["fisher_p"]:
        out.append(f"fisher p: published {PUBLISHED['fisher_p']}, got "
                   f"{'none' if fp is None else round(fp, 3)}")
    by_label = {m["measure"]: m for m in summary.get("measures") or []}
    for label, (wh, wd) in PUBLISHED["measures"].items():
        row = by_label.get(label)
        if row is None:
            out.append(f"{label}: published {wh[0]}/{wh[1]} but absent from this run")
            continue
        if tuple(row["vs_human"]) != wh or tuple(row["vs_declared"]) != wd:
            out.append(
                f"{label}: published {wh[0]}/{wh[1]} and {wd[0]}/{wd[1]}, got "
                f"{row['vs_human'][0]}/{row['vs_human'][1]} and "
                f"{row['vs_declared'][0]}/{row['vs_declared'][1]}"
            )
    return out


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


def binom_sf(k: int, n: int, p: float) -> float:
    """``P(X >= k)`` for ``X ~ Binomial(n, p)``, computed exactly.

    Written out rather than pulled from SciPy because the study is small enough that the
    exact sum is cheap, and because a p-value quoted in a paper should come from a line
    someone can check by eye.

    Args:
        k: Observed successes.
        n: Trials.
        p: Null success probability.

    Returns:
        The upper-tail probability, including ``k`` itself.
    """
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for the 2x2 table ``[[a, b], [c, d]]``.

    Used for the ORDER-versus-CONTROL contrast, where the counts are small enough that a
    chi-square would be inappropriate. The two-sided p sums the probability of every table
    with the same margins whose probability is no greater than the observed one -- the
    conventional definition, and the one that does not require choosing a direction in
    advance.

    Args:
        a, b, c, d: The table's cells, row-major.

    Returns:
        The two-sided p-value.
    """
    n = a + b + c + d
    row1, col1 = a + b, a + c

    def prob(a_: int) -> float:
        b_, c_, d_ = row1 - a_, col1 - a_, n - row1 - col1 + a_
        if min(a_, b_, c_, d_) < 0:
            return 0.0
        return (math.comb(row1, a_) * math.comb(n - row1, c_)) / math.comb(n, col1)

    observed = prob(a)
    return sum(
        pr
        for a_ in range(0, min(row1, col1) + 1)
        if (pr := prob(a_)) <= observed + 1e-12
    )


def verdict_from_scores(
    lower: float | None,
    higher: float | None,
    higher_side: str,
    tol: float = TIE_TOLERANCE,
) -> str | None:
    """Turn a measure's two rung scores into the A / B / equal verdict a reader gave.

    A measure does not answer "A or B"; it returns a number per response. To compare it
    with a reader it has to be asked the reader's question, which means mapping the sign
    of the difference onto the side that actually held the higher rung -- ``higher_side``
    from the answer key, since the export randomizes which side that is.

    The tolerance is the ladder scorer's own ``1e-6``: below it the measure is treated as
    saying "equal" rather than as expressing a preference, so floating-point noise cannot
    be scored as a judgement. This matters here more than on the ladder, because "equal"
    is a real answer readers gave on the invariance screens rather than an abstention.

    Args:
        lower: The measure's score on the lower rung, or None when missing.
        higher: The measure's score on the higher rung, or None when missing.
        higher_side: ``"A"`` or ``"B"`` -- which side of the screen held the higher rung.
        tol: Scores closer than this count as equal.

    Returns:
        ``"A"``, ``"B"``, ``"equal"``, or None when either score is missing.
    """
    if lower is None or higher is None:
        return None
    if abs(higher - lower) <= tol:
        return "equal"
    other = "B" if higher_side == "A" else "A"
    return higher_side if higher > lower else other


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


def load_lcs_arms(results_dir: str) -> dict[str, dict[tuple[str, int], dict]]:
    """Read ``by_item/*.json`` into ``{arm: {(family_id, rung): readouts}}``.

    Args:
        results_dir: An LCS results directory holding a ``by_item`` subdirectory.

    Returns:
        One entry per arm (``gold``, ``mined:<model>:<policy>``, ...), empty when the
        directory is absent -- the study's own numbers do not depend on it, so a missing
        results tree degrades this one section rather than failing the run.
    """
    out: dict[str, dict[tuple[str, int], dict]] = collections.defaultdict(dict)
    for path in glob.glob(os.path.join(results_dir, "by_item", "*.json")):
        with open(path) as f:
            item = json.load(f)
        for run in item.get("runs") or []:
            lcs = run.get("lcs")
            if isinstance(lcs, dict):
                out[run["arm"]][(run["family_id"], run["rung_index"])] = lcs
    return dict(out)


def load_baseline_arms(path: str) -> dict[str, dict[tuple[str, int], float]]:
    """Read a ``ladder_scores.jsonl`` into ``{baseline_name: {(family, rung): score}}``."""
    out: dict[str, dict[tuple[str, int], float]] = collections.defaultdict(dict)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("score") is not None:
                out[row["name"]][(row["family_id"], row["rung"])] = row["score"]
    return dict(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-dir", default="artifacts/human_study")
    p.add_argument("--data-dir", default="data/locobench-claude-5-test")
    p.add_argument(
        "--lcs-results",
        default="results/locobench_claude_5_v3_mined",
        help="LCS results directory (the one report_mined_ladder_lcs.py reads), used for "
        "the measure-versus-human comparison.",
    )
    p.add_argument(
        "--baseline-results",
        action="append",
        default=None,
        help="A baselines ladder_scores.jsonl. Repeatable, once per judge backend.",
    )
    p.add_argument(
        "--json-out",
        default=None,
        help="Write every reported number to this path, so the paper cites generated "
        "values rather than retyped ones.",
    )
    args = p.parse_args()
    if args.baseline_results is None:
        args.baseline_results = sorted(
            glob.glob("results/ladder_baselines_v2_*/ladder_scores.jsonl")
        )

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
    print()

    summary: dict = {
        "annotators": annotators,
        "n_screens": len(key),
        "n_ratings": sum(len(v) for v in resp.values()),
        "alpha": {
            lbl: krippendorff_alpha_nominal(units_for(kind))
            for lbl, kind in (("all", None), ("increase", "increase"),
                              ("invariance", "invariance"))
        },
        "no_majority": sorted(nomaj),
        "declared": {k: [agree[k], total[k]] for k in ("increase", "invariance")},
        "reason_coding": {"named": named, "checked": checked},
    }

    # ---- 5. the invariance result, split by ladder type ---------------------
    print("=== 5. invariance, split by ladder type (the pooled number hides this) ===")
    split = {}
    for ladder in ("ORDER", "CONTROL"):
        sids = sorted(s for s, k in key.items() if k["ladder"] == ladder)
        ratings = [r["choice"] for s in sids for r in resp.get(s, {}).values()]
        n_equal = sum(1 for c in ratings if c == "equal")
        split[ladder] = {
            "screens": sids, "n_ratings": len(ratings), "n_equal": n_equal,
            "per_screen": {s: [r["choice"] for r in resp.get(s, {}).values()]
                           for s in sids},
        }
        print(f"  {ladder:<8} {n_equal}/{len(ratings)} ratings called the pair equal")
        for sid in sids:
            print(f"    {sid}: {split[ladder]['per_screen'][sid]}")
    o, c = split.get("ORDER"), split.get("CONTROL")
    if o and c and o["n_ratings"] and c["n_ratings"]:
        pv = fisher_exact_two_sided(
            o["n_equal"], o["n_ratings"] - o["n_equal"],
            c["n_equal"], c["n_ratings"] - c["n_equal"],
        )
        split["fisher_p"] = pv
        print(f"\n  Fisher exact (two-sided) on equal-vs-not by ladder: p = {pv:.4f}")
        print("  ORDER shuffles SENTENCES, which can strand an anaphor without touching a")
        print("  claim or an edge; CONTROL reverses a relation's direction, which is")
        print("  meaning-preserving at the claim level. A significant split says the two")
        print("  must not be pooled: one invariance requirement is validated by readers")
        print("  and the other is not.")
        print("  The ratings within a ladder are 4 annotators x 2 screens of ONE family,")
        print("  so they are not 8 independent trials; this p treats them as exchangeable")
        print("  ratings and should be read as a descriptive contrast, not as a")
        print("  family-level significance claim. A per-annotator sign test would be")
        print("  worse, since it would double-count the same two responses.")
    summary["invariance_split"] = split
    print()

    # ---- 6. significance of the increase result -----------------------------
    print("=== 6. are readers better than chance on the increase screens? ===")
    inc = [s for s, k in key.items() if k["kind"] == "increase"]
    rating_hits = sum(
        1 for s in inc for r in resp.get(s, {}).values()
        if r["choice"] == key[s]["declared_answer"]
    )
    rating_n = sum(len(resp.get(s, {})) for s in inc)
    maj_hits, maj_n = agree["increase"], total["increase"]
    sig = {"rating": [rating_hits, rating_n], "majority": [maj_hits, maj_n]}
    for p_null, lbl in ((1 / 3, "uniform over {A,B,equal}"), (0.5, "coin flip A vs B")):
        pr = binom_sf(rating_hits, rating_n, p_null)
        pm = binom_sf(maj_hits, maj_n, p_null)
        sig[f"rating_p_{lbl}"] = pr
        sig[f"majority_p_{lbl}"] = pm
        print(f"  vs {lbl:<24} ratings {rating_hits}/{rating_n}: p = {pr:.2e}"
              f"   majorities {maj_hits}/{maj_n}: p = {pm:.4f}")
    print("  Report both nulls: 'equal' is available on every screen, so uniform-over-three")
    print("  is the honest chance rate, but a reader who never says 'equal' on an increase")
    print("  screen is effectively flipping a coin, and the majority-level test at this n")
    print("  may not clear 0.05 against it. Saying so is part of the result.")
    summary["significance"] = sig
    print()

    print("=== 6b. per-annotator accuracy on the increase screens ===")
    per_ann = {}
    for who in annotators:
        hit = sum(1 for s in inc
                  if resp.get(s, {}).get(who, {}).get("choice")
                  == key[s]["declared_answer"])
        seen = sum(1 for s in inc if who in resp.get(s, {}))
        per_ann[who] = [hit, seen]
        print(f"  {who}: {hit}/{seen}")
    summary["per_annotator_increase"] = per_ann
    times = [r["lead_time_s"] for d in resp.values() for r in d.values()
             if r.get("lead_time_s")]
    if times:
        med = statistics.median(times) / 60
        summary["lead_time_median_min"] = med
        summary["lead_time_range_min"] = [min(times) / 60, max(times) / 60]
        print(f"\n  median time per screen {med:.1f} min "
              f"(range {min(times)/60:.1f}-{max(times)/60:.1f})")
        print("  The maximum is a tab left open, not reading time, so the mean is")
        print("  meaningless here and only the median is reported.")
    print()

    # ---- 7. every measure against the human majority ------------------------
    print("=== 7. measures vs the human majority, and vs the declared ordering ===")
    columns: list[tuple[str, dict]] = []
    for arm, scores in sorted(load_lcs_arms(args.lcs_results).items()):
        for ro in READOUTS:
            columns.append((f"LCS[{arm}].{ro}",
                            {kk: vv.get(ro) for kk, vv in scores.items()}))
    for path in args.baseline_results:
        tag = os.path.basename(os.path.dirname(path)).replace(
            "ladder_baselines_v2_", "")
        for name, scores in sorted(load_baseline_arms(path).items()):
            columns.append((f"{name}@{tag}", scores))

    table: list[tuple[int, int, int, int, str]] = []
    partial: list[tuple[str, int]] = []
    for label, scores in columns:
        ah = nh = ad = nd = 0
        for sid, k in key.items():
            says = verdict_from_scores(
                scores.get((k["family_id"], k["lower_rung"])),
                scores.get((k["family_id"], k["higher_rung"])),
                k["higher_side"],
            )
            if says is None:
                continue
            nd += 1
            ad += says == k["declared_answer"]
            maj = majority([r["choice"] for r in resp.get(sid, {}).values()])
            if maj is not None:
                nh += 1
                ah += says == maj
        # A partial results tree (an abandoned or still-running backend) covers only a
        # few screens, and a row reading "1/1" next to another reading "9/12" invites a
        # false comparison. Require the full screen set rather than silently ranking a
        # measure on a subset of the questions.
        if nd == len(key):
            table.append((ah, nh, ad, nd, label))
        elif nd:
            partial.append((label, nd))
    if table:
        print(f"  {'measure':<46} {'vs human':>10} {'vs declared':>12}")
        for ah, nh, ad, nd, label in sorted(table, reverse=True):
            print(f"  {label:<46} {ah:5d}/{nh:<4d} {ad:7d}/{nd:<4d}")
        print("\n  Screens with no human majority are excluded from the human column, so")
        print("  its denominator is smaller. A measure cannot be credited or faulted for")
        print("  agreeing with a majority that does not exist.")
        if partial:
            print(f"\n  Excluded, covering fewer than {len(key)} screens: "
                  + ", ".join(f"{l} ({n})" for l, n in sorted(partial)))
        summary["measures"] = [
            {"measure": l, "vs_human": [ah, nh], "vs_declared": [ad, nd]}
            for ah, nh, ad, nd, l in sorted(table, reverse=True)
        ]
    else:
        print("  no measure results found; pass --lcs-results / --baseline-results")

    drift = check_published(summary)
    print()
    if drift:
        print("=== DRIFT from the cells the paper prints ===")
        for line in drift:
            print(f"  {line}")
        print("  Either the paper's table needs updating or this run is not the one it")
        print("  was written from. Do not republish the old numbers over new data.")
    else:
        print("[human-study] every published cell reproduced.")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(summary, f, indent=1, sort_keys=True)
        print(f"\n[human-study] wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
