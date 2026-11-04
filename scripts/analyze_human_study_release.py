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

"""Analyse the LoCoBench validation study: every hypothesis of the paper's Table 12.

Reads the judgements written by ``import_human_study_release.py`` and reports, for each
hypothesis, its units, its estimate with a 95% Wilson interval or its test, the
pre-registered pass criterion, and whether it passed. Agreement is Krippendorff's
alpha per question.

    python scripts/analyze_human_study_release.py \\
        --judgements artifacts/human_study_release/judgements.jsonl

Everything is computed on **majorities, one per unit** (a claim pair, a claim, an edited
version, an ORDER pair seen by three annotators), so a unit counts once however many
annotators judged it. A unit with no strict majority -- a three-way split -- is reported
and left out of the rate, rather than resolved by a tie-break that would invent a verdict.

Attention checks are scored first. An annotator who fails either check is excluded from
everything that follows, as the paper registers; ``--keep-failed-attention`` overrides
for diagnosis only.

How the three answers to a claim pair recover the coupling. "Could both be true?" and
"Could both be false?" are answered from the claims' meaning alone, and each coupling
the scheme compiles forbids a different cell of the truth table:

    both true  both false   coupling
    no         yes          contradiction  (Contrast, Concession)
    no         no           exclusive      (Alternative)
    yes        no           co_necessity   (Disjunction)
    yes        yes          entailment, equivalence or none (no conflict)

So a valid gold relation is "confirmed at the coupling level" when the majority marks it
linked AND its truth answers fall in the gold coupling's row.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# The pilot's statistics, reused rather than reimplemented, so one tested implementation
# backs both studies (tests/test_human_study.py).
from analyze_human_study import (  # noqa: E402
    binom_sf,
    fisher_exact_two_sided,
    krippendorff_alpha_nominal,
    majority,
)

CONFLICT_ROW = {(False, True): "contradiction", (False, False): "exclusive",
                (True, False): "co_necessity", (True, True): "no_conflict"}
COUPLING_CLASS = {"contradiction": "contradiction", "exclusive": "exclusive",
                  "co_necessity": "co_necessity", "entailment": "no_conflict",
                  "equivalence": "no_conflict", "none": "no_conflict"}

# Pre-registered pass thresholds (paper, Table 12).
PASS = {"A1": 0.90, "A2": 0.80, "A4": 0.95, "B": 0.90}
ALPHA_TARGET = 0.67
SIG = 0.05


# -- statistics ----------------------------------------------------------------------


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float] | None:
    """The 95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> float | None:
    """One-sided p for H1: p1 > p2, by the pooled two-proportion z test."""
    if not n1 or not n2:
        return None
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return None
    z = (k1 / n1 - k2 / n2) / se
    return 0.5 * math.erfc(z / math.sqrt(2))


# -- loading and exclusion -----------------------------------------------------------


def load(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def attention(rows: list[dict]) -> dict[str, dict]:
    """Per annotator: did they pass the repeated-claim and identical-version checks?

    Only answers with one defensible value are scored. A claim paired with itself must be
    marked **linked**; its two truth answers are not scored, because "could both be
    false?" about one proposition stated twice is a question readers can fairly read two
    ways. A version identical to the base must read **about as coherent** with **nothing
    else changed**.
    """
    out: dict[str, dict] = collections.defaultdict(dict)
    for r in rows:
        if not r.get("attention_check"):
            continue
        if r["kind"] == "pair":
            out[r["annotator"]]["repeated_claim"] = bool(r["linked"])
        elif r["kind"] == "version":
            out[r["annotator"]]["identical_version"] = (r["coherence"] == "same"
                                                        and r["collateral"] == "none")
    return out


# -- majorities ----------------------------------------------------------------------


def units(rows: list[dict], kind: str, field) -> dict[str, list]:
    """unit -> the list of answers to ``field`` (a key or a function of the row)."""
    get = field if callable(field) else (lambda r: r[field])
    u: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        if r["kind"] == kind and not r.get("attention_check"):
            u[r["unit"]].append(get(r))
    return u


def first(rows: list[dict], kind: str) -> dict[str, dict]:
    """unit -> one representative row, for the unit's fixed (key-side) fields."""
    out: dict[str, dict] = {}
    for r in rows:
        if r["kind"] == kind and not r.get("attention_check"):
            out.setdefault(r["unit"], r)
    return out


def majorities(u: dict[str, list]) -> dict[str, object]:
    return {k: majority([json.dumps(v) for v in vs]) for k, vs in u.items()}


def _undump(v):
    return None if v is None else json.loads(v)


def alpha_of(u: dict[str, list]) -> float | None:
    return krippendorff_alpha_nominal([[json.dumps(v) for v in vs] for vs in u.values()])


# -- hypotheses ----------------------------------------------------------------------


def rate(name: str, k: int, n: int, splits: int, threshold: float | None) -> dict:
    ci = wilson(k, n)
    return {"hypothesis": name, "units": n, "no_majority": splits, "k": k,
            "rate": (k / n) if n else None, "ci95": ci, "threshold": threshold,
            "passed": None if threshold is None or ci is None else (k / n) >= threshold}


def analyse(rows: list[dict]) -> dict:
    pairs = first(rows, "pair")
    pair_answer = units(rows, "pair", lambda r: [r["linked"], r["both_true"], r["both_false"]])
    pm = {u: _undump(m) for u, m in majorities(pair_answer).items()}

    def confirmed_coupling(u):
        m = pm[u]
        if m is None:
            return None
        linked, bt, bf = m
        return bool(linked) and CONFLICT_ROW[(bt, bf)] == COUPLING_CLASS[pairs[u]["gold_coupling"]]

    # Majority per QUESTION, not per answer triple, for the link question alone: a pair
    # can have a clear "linked" majority while its truth answers split.
    link = {u: _undump(m) for u, m in majorities(units(rows, "pair", "linked")).items()}

    out: dict[str, object] = {}
    valid = [u for u, r in pairs.items() if r["gold_kind"] == "rel" and r["validity"] == "valid"]
    a1 = [confirmed_coupling(u) for u in valid]
    out["A1"] = rate("A1 valid gold relations confirmed (coupling level)",
                     sum(1 for x in a1 if x), sum(1 for x in a1 if x is not None),
                     sum(1 for x in a1 if x is None), PASS["A1"])
    out["A1_by_coupling"] = {
        c: rate(f"A1 {c}", sum(1 for u in valid if pairs[u]["gold_coupling"] == c and confirmed_coupling(u)),
                sum(1 for u in valid if pairs[u]["gold_coupling"] == c and confirmed_coupling(u) is not None),
                0, None)
        for c in sorted({pairs[u]["gold_coupling"] for u in valid})}

    non = [u for u, r in pairs.items() if r["gold_kind"] == "non"]
    a2 = [link[u] for u in non]
    out["A2"] = rate("A2 declared non-relations confirmed (not linked)",
                     sum(1 for x in a2 if x is False), sum(1 for x in a2 if x is not None),
                     sum(1 for x in a2 if x is None), PASS["A2"])

    inval = [u for u, r in pairs.items() if r["gold_kind"] == "rel" and r["validity"] == "invalid"]
    by_err: dict[str, list] = collections.defaultdict(list)
    for u in inval:
        by_err[pairs[u]["error_kind"]].append(confirmed_coupling(u))
    # A3, descriptive by error kind. Planted-invalid relations are ASSERTED in the prose
    # (paper, sec-plan), so a careful reader marks them linked; what can differ is whether
    # the reader's truth answers match the coupling the label claims. A planted relation
    # the majority "accepts" -- linked, at its labelled coupling -- reads as valid, so
    # either the error is subtle or the writer realized it too well and the label should
    # be repaired. The truth questions can catch a wrong coupling (wrong_sense) but not
    # a wrong direction or a link with no basis, so the rate is reported, never thresholded.
    out["A3"] = {e: {**rate(f"A3 {e} accepted at its labelled coupling", sum(1 for x in xs if x),
                            sum(1 for x in xs if x is not None), sum(1 for x in xs if x is None), None),
                     "accepted_units": [u for u, x in zip([v for v in inval if pairs[v]["error_kind"] == e], xs) if x]}
                 for e, xs in sorted(by_err.items())}

    claims = first(rows, "claim")
    cm = {u: _undump(m) for u, m in majorities(units(rows, "claim", "asserted")).items()}
    false_units = [u for u, r in claims.items() if not r["factual"]]
    a4 = [cm[u] for u in false_units]
    out["A4"] = {**rate("A4 planted-false claims asserted as written",
                        sum(1 for x in a4 if x == "asserted"), sum(1 for x in a4 if x is not None),
                        sum(1 for x in a4 if x is None), PASS["A4"]),
                 "exceptions": sorted(u for u in false_units if cm[u] not in (None, "asserted"))}
    true_units = [u for u, r in claims.items() if r["factual"]]
    out["A4_true_claims"] = rate("true claims asserted as written (context for A4)",
                                 sum(1 for u in true_units if cm[u] == "asserted"),
                                 sum(1 for u in true_units if cm[u] is not None), 0, None)

    versions = first(rows, "version")
    coll = {u: _undump(m) for u, m in majorities(units(rows, "version", "collateral")).items()}
    bmaj = [coll[u] for u in versions]
    by_call: dict[str, list] = collections.defaultdict(list)
    for u, v in versions.items():
        by_call["+".join(v["calls"])].append(coll[u])
    out["B"] = {**rate("B edits realize their declared effect only (no material collateral)",
                       sum(1 for x in bmaj if x in ("none", "minor")), sum(1 for x in bmaj if x is not None),
                       sum(1 for x in bmaj if x is None), PASS["B"]),
                "by_call": {c: rate(c, sum(1 for x in xs if x in ("none", "minor")),
                                    sum(1 for x in xs if x is not None), sum(1 for x in xs if x is None),
                                    PASS["B"] if len(xs) >= 10 else None)
                            for c, xs in sorted(by_call.items())},
                "material_units": sorted(u for u in versions if coll[u] == "material")}

    coh = {u: _undump(m) for u, m in majorities(units(rows, "version", "coherence")).items()}

    def declared(v):
        if v["ladder"] == "CONTROL":
            return "same"
        return "less" if v["rungs"][0] < v["parent"] else "more"

    inc = [u for u, v in versions.items() if v["ladder"] in ("CONFLICT", "CHAIN")]
    agree = [None if coh[u] is None else coh[u] == declared(versions[u]) for u in inc]
    k, n = sum(1 for x in agree if x), sum(1 for x in agree if x is not None)
    # Chance is one of three answers; H1 is one-sided against it.
    out["H1"] = {**rate("H1 readers agree with the declared direction", k, n,
                        sum(1 for x in agree if x is None), None),
                 "chance": 1 / 3, "p_one_sided": binom_sf(k, n, 1 / 3) if n else None}
    out["H1"]["passed"] = None if out["H1"]["p_one_sided"] is None else out["H1"]["p_one_sided"] < SIG

    first_step = [u for u in inc if versions[u]["ladder"] == "CONFLICT" and versions[u]["rungs"][0] == 0]
    fs = [None if coh[u] is None else coh[u] == "less" for u in first_step]
    other = [x for u, x in zip(inc, agree) if u not in first_step]
    out["H2"] = {**rate("H2 agreement at CONFLICT's first step (descriptive)",
                        sum(1 for x in fs if x), sum(1 for x in fs if x is not None),
                        sum(1 for x in fs if x is None), None),
                 "other_steps": rate("H2 agreement at every other step", sum(1 for x in other if x),
                                     sum(1 for x in other if x is not None), 0, None)}

    ctl = [u for u, v in versions.items() if v["ladder"] == "CONTROL"]
    eq = [None if coh[u] is None else coh[u] == "same" for u in ctl]
    out["H3"] = {**rate("H3 CONTROL versions read as about equal (descriptive)",
                        sum(1 for x in eq if x), sum(1 for x in eq if x is not None),
                        sum(1 for x in eq if x is None), None)}
    out["H3"]["passed"] = None if not out["H3"]["units"] else out["H3"]["rate"] > 0.5

    # H4: per judgement, not per majority -- the instruction is assigned per annotator
    # within a pair, so the pair is not one unit under one condition.
    ins = [r for r in rows if r["kind"] == "instruction"]
    arms = {name: [r for r in ins if r["instruction"] == name] for name in ("consistency", "argument")}
    eqc = {name: sum(1 for r in rs if r["choice"] == "equal") for name, rs in arms.items()}
    nc = {name: len(rs) for name, rs in arms.items()}
    p = two_proportion_z(eqc["consistency"], nc["consistency"], eqc["argument"], nc["argument"])
    out["H4"] = {"hypothesis": "H4 'about equal' more often under the consistency instruction",
                 "equal_rate": {k_: (eqc[k_] / nc[k_]) if nc[k_] else None for k_ in arms},
                 "n": nc, "p_one_sided": p,
                 "fisher_two_sided": fisher_exact_two_sided(
                     eqc["consistency"], nc["consistency"] - eqc["consistency"],
                     eqc["argument"], nc["argument"] - eqc["argument"]) if all(nc.values()) else None,
                 "passed": None if p is None else p < SIG,
                 "base_preferred_rate": {k_: (sum(1 for r in rs if r["choice"] == r["base_side"]) / len(rs)) if rs else None
                                         for k_, rs in arms.items()}}

    # Coverage: how many annotators each core family still has. An excluded annotator, or
    # a session stopped at 3 hours, leaves families with fewer than three, where a
    # two-way split has no majority; those families should be re-annotated.
    per_fam: dict[str, set] = collections.defaultdict(set)
    for r in rows:
        if r["kind"] in ("pair", "claim", "version"):
            per_fam[r["family_id"]].add(r["annotator"])
    out["coverage"] = {"families": len(per_fam),
                       "below_three": {f: sorted(a) for f, a in sorted(per_fam.items()) if len(a) < 3}}

    out["agreement"] = {
        "linked": alpha_of(units(rows, "pair", "linked")),
        "coupling": alpha_of(units(rows, "pair", lambda r: CONFLICT_ROW[(r["both_true"], r["both_false"])])),
        "asserted": alpha_of(units(rows, "claim", "asserted")),
        "coherence": alpha_of(units(rows, "version", "coherence")),
        "collateral": alpha_of(units(rows, "version", "collateral")),
        "instruction_choice": alpha_of(units(rows, "instruction", "choice")),
        "target_on_coupling": ALPHA_TARGET,
    }
    return out


# -- report --------------------------------------------------------------------------


def _fmt_rate(r: dict) -> str:
    if not r["units"]:
        return f"{r['hypothesis']}: no units"
    lo, hi = r["ci95"]
    s = f"{r['hypothesis']}: {r['k']}/{r['units']} = {100 * r['rate']:.1f}% [95% CI {100 * lo:.1f}, {100 * hi:.1f}]"
    if r["no_majority"]:
        s += f", {r['no_majority']} without a majority"
    if r["threshold"] is not None:
        s += f"  -- threshold {100 * r['threshold']:.0f}%: {'PASS' if r['passed'] else 'FAIL'}"
    return s


def report(res: dict, excluded: list[str], att: dict) -> str:
    L = []
    L.append("Attention checks: " + ", ".join(f"{a} {'ok' if all(v.values()) else 'FAILED ' + str(v)}"
                                              for a, v in sorted(att.items())))
    L.append(f"Excluded annotators: {excluded or 'none'}")
    cov = res["coverage"]
    if cov["below_three"]:
        L.append(f"Families with fewer than 3 annotators ({len(cov['below_three'])} of {cov['families']}); "
                 "re-annotate these: " + ", ".join(f"{f} {a}" for f, a in cov["below_three"].items()))
    else:
        L.append(f"All {cov['families']} core families have 3 annotators.")
    L.append("")
    L.append(_fmt_rate(res["A1"]))
    for r in res["A1_by_coupling"].values():
        L.append("    " + _fmt_rate(r))
    L.append(_fmt_rate(res["A2"]))
    L.append("A3 planted-invalid relations accepted as valid, by error kind (descriptive, no threshold):")
    for e, r in res["A3"].items():
        L.append("    " + _fmt_rate(r))
    L.append(_fmt_rate(res["A4"]) + (f"; exceptions: {res['A4']['exceptions']}" if res["A4"]["exceptions"] else ""))
    L.append("    " + _fmt_rate(res["A4_true_claims"]))
    L.append(_fmt_rate(res["B"]))
    for r in res["B"]["by_call"].values():
        L.append("    " + _fmt_rate(r))
    if res["B"]["material_units"]:
        L.append(f"    material collateral change (flag these rungs in the release): {res['B']['material_units']}")
    h1 = res["H1"]
    L.append(_fmt_rate(h1) + (f"; one-sided binomial vs chance 1/3: p = {h1['p_one_sided']:.2g} -- "
                              f"{'PASS' if h1['passed'] else 'FAIL'}" if h1["p_one_sided"] is not None else ""))
    L.append(_fmt_rate(res["H2"]))
    L.append("    " + _fmt_rate(res["H2"]["other_steps"]))
    L.append(_fmt_rate(res["H3"]) + ("" if res["H3"]["passed"] is None else f"  -- majority equal: {'yes' if res['H3']['passed'] else 'no'}"))
    h4 = res["H4"]
    if all(h4["n"].values()):
        L.append(f"H4 'about equal' rate: consistency {100 * h4['equal_rate']['consistency']:.1f}% "
                 f"(n={h4['n']['consistency']}) vs argument {100 * h4['equal_rate']['argument']:.1f}% "
                 f"(n={h4['n']['argument']}); one-sided p = {h4['p_one_sided']:.2g}, Fisher two-sided "
                 f"p = {h4['fisher_two_sided']:.2g} -- {'PASS' if h4['passed'] else 'FAIL'}")
        L.append(f"    original preferred: consistency {100 * h4['base_preferred_rate']['consistency']:.0f}%, "
                 f"argument {100 * h4['base_preferred_rate']['argument']:.0f}%")
    else:
        L.append("H4: no instruction-test judgements")
    a = res["agreement"]
    L.append("\nKrippendorff's alpha: " + ", ".join(f"{k} {'n/a' if v is None else f'{v:.2f}'}"
                                                   for k, v in a.items() if k != "target_on_coupling"))
    if a["coupling"] is not None:
        L.append(f"    coupling alpha {a['coupling']:.2f} vs target {ALPHA_TARGET}: "
                 f"{'met' if a['coupling'] >= ALPHA_TARGET else 'NOT met'}")
    return "\n".join(L)


def main() -> int:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judgements", default=os.path.join(repo, "artifacts", "human_study_release", "judgements.jsonl"))
    ap.add_argument("--json", help="Also write the full results as JSON here.")
    ap.add_argument("--keep-failed-attention", action="store_true",
                    help="Do not exclude annotators who failed an attention check (diagnosis only).")
    args = ap.parse_args()

    rows = [r for r in load(args.judgements) if not r.get("practice")]
    att = attention(rows)
    excluded = sorted(a for a, v in att.items() if not all(v.values()))
    if excluded and not args.keep_failed_attention:
        rows = [r for r in rows if r["annotator"] not in excluded]
    res = analyse(rows)
    print(report(res, [] if args.keep_failed_attention else excluded, att))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"excluded": excluded, "attention": att, "results": res}, f, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
