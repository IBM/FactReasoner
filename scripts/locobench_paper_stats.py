"""Release statistics for the LoCoBench dataset paper (docs/arr2027/locobench).

Reads the slot3 test split (families.json, items.jsonl, attempts.jsonl, spend.jsonl) and the
development split, and writes LaTeX macros (prefix ``\\lb``) and a JSON file the figure script
reads. Every number in the paper's data sections comes from here.

    .venv/bin/python scripts/locobench_paper_stats.py \\
        --macros docs/arr2027/locobench/numbers_lb.tex --json docs/arr2027/locobench/stats.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import statistics as st

TEST = "data/locobench-slot3-test"
DEV = "data/locobench-slot3-pilot"
EVAL = "results/arr2027_slot3_test/eval.json"
TYPES = ("CONFLICT", "CHAIN", "SIGN", "ORDER", "CONTROL")
PRIMARY = ("CONFLICT", "CHAIN", "SIGN", "ORDER")


def load(d):
    fams = json.load(open(os.path.join(d, "families.json")))["families"]
    items = [json.loads(l) for l in open(os.path.join(d, "items.jsonl")) if l.strip()]
    return fams, items


def gates(d):
    """Per-gate pass rates over every evaluation of every attempt (retries included)."""
    recs = [json.loads(l) for l in open(os.path.join(d, "attempts.jsonl")) if l.strip()]
    gate = collections.defaultdict(lambda: [0, 0])
    sv5 = collections.defaultdict(lambda: [0, 0])
    calls = collections.Counter()
    for r in recs:
        calls.update(r.get("calls", {}))
        for e in r.get("events", []):
            g = gate[e["gate"]]
            g[0] += bool(e["passed"])
            g[1] += 1
            if e["gate"] == "SV5" and e.get("attempt") == 0:
                rr = sv5[f"{e.get('role')}/{e.get('kind')}"]
                rr[0] += bool(e["passed"])
                rr[1] += 1
    fams = collections.OrderedDict()
    for r in recs:
        fams.setdefault(r["family_id"], []).append(r)
    first = sum(rs[0]["outcome"] == "admitted" for rs in fams.values())
    stages = collections.Counter(r.get("stage") for r in recs if r["outcome"] != "admitted")
    why = collections.Counter()
    for r in recs:
        if r["outcome"] == "admitted":
            continue
        reason = r.get("reason") or ""
        key = ("VThreeSpans" if reason.startswith("V3.spans") else
               "VThreeScores" if reason.startswith("V3.scores") else
               "VFour" if reason.startswith("V4") else
               "ContentFilter" if reason.startswith("P2") else
               "SFour" if reason.startswith("S4") else
               "SVFive" if reason.startswith("SV5") else "Other")
        why[key] += 1
    return {"n_attempts": len(recs), "n_families_attempted": len(fams), "first_try": first,
            "gate": {k: tuple(v) for k, v in gate.items()},
            "sv5_first": {k: tuple(v) for k, v in sv5.items()},
            "calls": dict(calls), "rejected_by_stage": dict(stages), "rejected_why": dict(why),
            "admitted_attempts": sum(r["outcome"] == "admitted" for r in recs)}


def spend(d):
    p = os.path.join(d, "spend.jsonl")
    rows = [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []
    vals = [r["key_spend"] for r in rows if r.get("key_spend") is not None]
    return vals[-1] - vals[0] if len(vals) >= 2 else None


def release(fams, items):
    by_id = {i["id"]: i for i in items}
    lad = [l for f in fams for l in f["ladders"]]
    out = {"n_families": len(fams), "n_ladders": len(lad), "n_items": len(items),
           "ladders_by_type": {t: sum(l["type"] == t for l in lad) for t in TYPES}}
    out["topics"] = len({f["canonical_topic"] for f in fams})
    dom = collections.Counter(f["domain"] for f in fams)
    out["domains"] = dict(dom)
    out["domain_x_type"] = {d: {t: sum(f["domain"] == d and f["primary"] == t for f in fams)
                                for t in PRIMARY} for d in sorted(dom)}
    out["topic_families"] = dict(collections.Counter(f["canonical_topic"] for f in fams))
    out["rare_senses"] = dict(collections.Counter(s for f in fams for s in f["rare_senses"]))
    words = [i["meta"]["word_count"] for i in items]
    claims = [i["num_atoms"] for i in items]
    false = [sum(not a["factual"] for a in i["atoms"]) for i in items]
    rels = [len(i["relations"]) for i in items]
    factor = [sum(r["level1_coupling"] != "none" for r in i["relations"]) for i in items]
    nonrel = [len(i["non_relations"]) for i in items]
    out["dist"] = {"words": words, "claims": claims, "false": false, "relations": rels,
                   "factor_relations": factor, "non_relations": nonrel}
    out["summary"] = {k: {"min": min(v), "max": max(v), "mean": st.mean(v),
                          "median": st.median(v)} for k, v in out["dist"].items()}
    C = collections.Counter
    allr = [r for i in items for r in i["relations"]]
    out["relations"] = {
        "total": len(allr),
        "sense": dict(C(r["level2_sense"] for r in allr)),
        "coupling": dict(C(r["level1_coupling"] for r in allr)),
        "band": dict(C(r["intended_strength_band"] for r in allr)),
        "sign_verdict": dict(C(r["sign_verdict"] for r in allr)),
        "sign_assertion": dict(C(r["sign_assertion"] for r in allr)),
        "slot": sum(r["provenance"]["source"] == "slot" for r in allr),
        "plan": sum(r["provenance"]["source"] == "plan" for r in allr),
    }
    # Relations per FAMILY template (the base splice's plan relations), not per response.
    out["stance_denied"] = sum(a["stance"] == "denied" for i in items for a in i["atoms"])
    out["n_atoms_total"] = sum(claims)
    corr = [c for f in fams for c in f.get("gold_corrections", [])]
    out["gold_corrections"] = {"total": len(corr),
                               "by_action": dict(C(c["action"] for c in corr)),
                               "families": sum(bool(f.get("gold_corrections")) for f in fams)}
    out["missing_control"] = sum(not any(l["type"] == "CONTROL" for l in f["ladders"])
                                 for f in fams)
    # Response count: 3 rungs per ladder, minus the rung a family's two ladders share
    # (the primary's rung 1 is the CONTROL ladder's rung 0).
    slots = sum(len(l["item_ids"]) for l in lad)
    member = collections.Counter(i for l in lad for i in l["item_ids"])
    shared = sum(n == 2 for n in member.values())
    assert set(member.values()) <= {1, 2} and len(member) == len(items) == slots - shared
    assert shared == out["n_families"] - out["missing_control"]
    out["rung_slots"], out["shared_rungs"] = slots, shared
    # Gold LCS per rung, per ladder (stored by the generator's exact ladder check).
    out["gold_lcs"] = {t: [l["gold"]["lcs"] for l in lad if l["type"] == t] for t in TYPES}
    out["gold_lambda"] = {t: [l["gold"]["lambda"] for l in lad if l["type"] == t] for t in TYPES}
    assert all(i in by_id for l in lad for i in l["item_ids"])
    return out


def fillers(d, fams):
    """Admitted filler lengths (words) and the stored committee readings, incl. rejected ones."""
    st_ = json.load(open(os.path.join(d, "state.json")))["families"]
    keep = {f["family_id"] for f in fams}
    words, logged, rejected = [], 0, 0
    for fid, f in st_.items():
        if fid not in keep:
            continue
        art = f.get("artifacts", {})
        for slot in (art.get("fillers") or {}).values():
            words += [len(t.split()) for t in slot.values()]
        for x in art.get("filler_log", []):
            logged += 1
            rejected += not x["passed"]
    return {"words": words, "logged": logged, "rejected": rejected}


def write_macros(path, R, G, Dv, spent, F=None):
    L = ["% Generated by scripts/locobench_paper_stats.py -- do not edit by hand."]

    words = "Zero One Two Three Four Five Six Seven Eight Nine".split()

    def m(name, val):
        # LaTeX command names cannot contain digits: P1 -> POne, SV5 -> SVFive.
        name = "".join(words[int(c)] if c.isdigit() else c for c in name)
        L.append(f"\\newcommand{{\\lb{name}}}{{{val}}}")

    def num(x):
        return f"{x:,}".replace(",", "{,}")

    m("Families", R["n_families"])
    m("Ladders", R["n_ladders"])
    m("Items", num(R["n_items"]))
    for t, k in R["ladders_by_type"].items():
        m(f"N{t.title()}", k)
    m("Topics", R["topics"])
    m("Domains", len(R["domains"]))
    s = R["summary"]
    for k, tag in (("words", "Words"), ("claims", "Claims"), ("false", "False"),
                   ("relations", "Rels"), ("factor_relations", "FactorRels"),
                   ("non_relations", "NonRels")):
        m(f"{tag}Min", s[k]["min"])
        m(f"{tag}Max", s[k]["max"])
        m(f"{tag}Mean", f"{s[k]['mean']:.1f}")
        m(f"{tag}Median", f"{s[k]['median']:g}")
    r = R["relations"]
    m("RelTotal", num(r["total"]))
    m("RelFactor", num(r["total"] - r["coupling"].get("none", 0)))
    m("RelNone", num(r["coupling"].get("none", 0)))
    m("RelSlot", num(r["slot"]))
    m("RelPlan", num(r["plan"]))
    for k, tag in (("support", "Sup"), ("conflict", "Con")):
        m(f"RelVerdict{tag}", num(r["sign_verdict"].get(k, 0)))
        m(f"RelAssert{tag}", num(r["sign_assertion"].get(k, 0)))
    for sense, n in r["sense"].items():
        m("Sense" + sense.replace("-", ""), num(n))
    for c, n in r["coupling"].items():
        m("Coup" + c.replace("_", "").title(), num(n))
    for b, n in r["band"].items():
        m("Band" + b.title(), num(n))
    m("AtomsTotal", num(R["n_atoms_total"]))
    m("Denied", R["stance_denied"])
    m("GoldCorr", R["gold_corrections"]["total"])
    m("GoldCorrFamilies", R["gold_corrections"]["families"])
    for a, n in R["gold_corrections"]["by_action"].items():
        m("GoldCorr" + a.title(), n)
    m("MissingControl", R["missing_control"])
    m("FullFamilies", R["n_families"] - R["missing_control"])
    m("FullFamilyItems", num(5 * (R["n_families"] - R["missing_control"])))
    m("PrimaryOnlyItems", 3 * R["missing_control"])
    m("RungSlots", num(R["rung_slots"]))
    m("SharedRungs", R["shared_rungs"])
    for sense, n in R["rare_senses"].items():
        m("Rare" + sense.replace("-", ""), n)
    # Generation record.
    m("Attempts", G["n_attempts"])
    m("FirstTry", G["first_try"])
    calls = G["calls"]
    m("AttemptsAdmitted", G["admitted_attempts"])
    m("AttemptsRejected", G["n_attempts"] - G["admitted_attempts"])
    for k, v in G["rejected_by_stage"].items():
        m("RejStage" + k.title(), v)
    for k, v in G["rejected_why"].items():
        m("Rej" + k, v)
    m("Calls", num(sum(calls.values())))
    for k, v in calls.items():
        m("Calls" + k.split("@")[0], num(v))
    n_adm_lad = sum(R["ladders_by_type"].values())
    m("CallsPerLadder", f"{sum(calls.values()) / n_adm_lad:.1f}")
    for g, (ok, n) in G["gate"].items():
        tag = g.replace(".", "").replace("@", "").replace("_", "").replace("ladder", "Lad")
        tag = "".join(p[:1].upper() + p[1:] for p in tag.split())
        m(f"Gate{tag}", f"{ok}/{n}")
        m(f"Gate{tag}Pct", f"{100 * ok / n:.0f}")
    for k, (ok, n) in G["sv5_first"].items():
        role, kind = k.split("/")
        tag = role.replace("_", "").title() + kind.title()
        m(f"Read{tag}", f"{ok}/{n}")
        m(f"Read{tag}Pct", f"{100 * ok / n:.0f}")
    if spent is not None:
        m("Spend", f"{spent:,.0f}")
    if F:
        m("FillerWordsMedian", f"{st.median(F['words']):g}")
        m("FillerWordsMax", max(F["words"]))
        m("FillersAdmitted", num(len(F["words"])))
        m("FillerReadingsLogged", num(F["logged"]))
        m("FillerReadingsRejected", F["rejected"])
    # Development split.
    m("DevFamilies", Dv["n_families"])
    m("DevLadders", Dv["n_ladders"])
    m("DevItems", Dv["n_items"])
    open(path, "w").write("\n".join(L) + "\n")


def write_topic_table(path, fams):
    """Topics by domain, with families per primary ladder type (appendix)."""
    by = collections.defaultdict(lambda: collections.Counter())
    dom = {}
    for f in fams:
        by[f["canonical_topic"]][f["primary"]] += 1
        dom[f["canonical_topic"]] = f["domain"]
    L = ["% Generated by scripts/locobench_paper_stats.py -- do not edit by hand.",
         "\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
         "Domain & Topic & \\cp{conf.} & \\cp{chain} & \\cp{sign} & \\cp{order} & all \\\\",
         "\\midrule"]
    last = None
    for t in sorted(by, key=lambda t: (dom[t], t)):
        d = {"Engineering & Technology": "Eng.\\,\\&\\,Tech.", "Humanities": "Humanities",
             "Natural Sciences": "Nat.\\ Sci.", "Social Sciences": "Soc.\\ Sci."}[dom[t]]
        c = by[t]
        L.append(f"{d if d != last else ''} & {t} & " + " & ".join(
            str(c[x]) for x in PRIMARY) + f" & {sum(c.values())} \\\\")
        last = d
    L += ["\\bottomrule", "\\end{tabular}"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(L) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--macros", default="docs/arr2027/locobench/numbers_lb.tex")
    ap.add_argument("--json", default="docs/arr2027/locobench/stats.json")
    args = ap.parse_args()
    fams, items = load(TEST)
    R = release(fams, items)
    G = gates(TEST)
    Dv = release(*load(DEV))
    spent = spend(TEST)
    # Cross-check the stored gold LCS against the companion paper's evaluation.
    if os.path.exists(EVAL):
        ev = json.load(open(EVAL))["gold_shape"]
        for t in TYPES:
            for k in range(3):
                mine = st.mean(v[k] for v in R["gold_lcs"][t])
                assert abs(mine - ev[t]["lcs"][k]) < 2e-3, (t, k, mine, ev[t]["lcs"][k])
    json.dump({"release": R, "gates": G, "dev": {k: v for k, v in Dv.items() if k != "dist"},
               "spend": spent}, open(args.json, "w"), indent=1)
    F = fillers(TEST, fams)
    write_macros(args.macros, R, G, Dv, spent, F)
    write_topic_table(os.path.join(os.path.dirname(args.macros), "examples", "topics.tex"),
                      fams)
    print(f"wrote {args.macros} and {args.json}")
    print(json.dumps({k: R[k] for k in ("ladders_by_type", "domains", "rare_senses",
                                        "gold_corrections", "missing_control", "stance_denied")}))
    print(json.dumps(R["summary"]))
    print(json.dumps(G["rejected_by_stage"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
