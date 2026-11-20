"""Worked examples for the LoCoBench dataset paper, extracted from the released test split.

Nothing in the paper's examples is typed by hand: every text, label, relation, committee reading
and score below is read from ``data/locobench-slot3-test`` (families.json, items.jsonl and the
generator's stored state, state.json) and written as LaTeX snippets to
``docs/arr2027/locobench/examples/``:

* ``family_<fid>.tex``   -- one family end to end (appendix): question, claims, template with
  slot markers, fillers, rungs with their assignments and gold LCS at both worlds, one response
  in full with its slot sentences marked, and the base response's gold relations.
* ``teaser_<fid>.tex``   -- the primary ladder's slot sentences per rung, with mini relation
  graphs and gold LCS (main-text Figure 1).
* ``worlds_<fid>.tex``   -- a SIGN ladder's slot under the verdict and the asserted world.
* ``readings_<fid>.tex`` -- a filler's blind committee readings across its attempts.

    .venv/bin/python scripts/locobench_paper_examples.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from fact_reasoner.locobench import slots as S  # noqa: E402
import arr_slot3_eval as E  # noqa: E402  (the companion paper's scorer)

_SCORES: dict[str, dict] = {}


def score(item):
    """Gold LCS at the verdict and at the asserted world, from the companion paper's scorer."""
    if item["id"] not in _SCORES:
        sc = E.score_item(item, E.gold_relations(item))
        _SCORES[item["id"]] = {"v": sc["lcs"], "a": sc["lcs_assert"]}
    return _SCORES[item["id"]]

DATA = "data/locobench-slot3-test"
OUT = "docs/arr2027/locobench/examples"
FAMILIES = {"CONFLICT": "f009", "CHAIN": "f224", "SIGN": "f003", "ORDER": "f055"}
TEASER = "f009"
WORLDS = "f003"
READINGS = ("f075", "S1", "conflict")
RUNG_NAMES = {
    "CONFLICT": ["both slots set their claims against each other", "one slot does",
                 "neither does"],
    "CHAIN": ["neither link drawn", "the first link drawn", "both links drawn"],
    "SIGN": ["the falsehood derived from the truth", "no link", "the falsehood refuted"],
    "ORDER": ["paragraph order 1", "paragraph order 2", "paragraph order 3"],
    "CONTROL": ["wording 1", "wording 2", "wording 3"],
}
ROLE_NAMES = {"conflict_tt": "conflict", "chain_link": "chain link",
              "sign_tf": "sign", "control": "control"}
KIND_NAMES = {"none": "no link", "conflict": "conflict", "support": "support",
              "para1": "wording 1", "para2": "wording 2", "para3": "wording 3"}


def tex(t: str) -> str:
    t = re.sub(r'"([^"]*)"', r"``\1''", t)
    t = t.replace("\\", "\\textbackslash{}")
    for a, b in (("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"),
                 ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"),
                 ("^", "\\textasciicircum{}")):
        t = t.replace(a, b)
    return t.replace("\\textbackslash\\{\\}", "\\textbackslash{}")


def load():
    fams = {f["family_id"]: f for f in
            json.load(open(os.path.join(DATA, "families.json")))["families"]}
    items = {}
    for line in open(os.path.join(DATA, "items.jsonl")):
        if line.strip():
            it = json.loads(line)
            items[it["id"]] = it
    state = json.load(open(os.path.join(DATA, "state.json")))["families"]
    return fams, items, state


def slot_rel(item, sid):
    for r in item["relations"]:
        if r.get("slot_id") == sid:
            return r
    return None


def lad(fam, t):
    return next(l for l in fam["ladders"] if l["type"] == t)


def primary_slots(art, primary, items, ladder):
    """Slots whose filler differs across the primary ladder's rungs, in plan order."""
    asg = [items[i]["expected"]["assignment"] for i in ladder["item_ids"]]
    return [s for s in art["plan"]["slots"] if len({a.get(s["id"]) for a in asg}) > 1]


def render_text(template, fillers, assignment, mark=None, order=None):
    """A response as LaTeX paragraphs; slots in ``mark`` are wrapped in \\slotfill."""
    def rep(m):
        sid = m.group(1)
        text = tex(fillers[sid][assignment[sid]].strip())
        return f"\\slotfill{{{sid}}}{{{text}}}" if (mark is None or sid in mark) else text
    pars = [p.strip() for p in re.split(r"\n\s*\n", template) if p.strip()]
    if order is not None:
        pars = [pars[i] for i in order]
    out = []
    for p in pars:
        p = re.sub(r"\s+", " ", p)
        p = S.MARKER_RE.sub(lambda m: "\x00" + m.group(1) + "\x00", p)
        parts = p.split("\x00")
        buf = []
        for k, part in enumerate(parts):
            if k % 2:
                buf.append(rep(re.match(r"(S\d+)", part)))
            else:
                buf.append(tex(part))
        out.append("".join(buf))
    return "\n\n".join(out)


def render_template(template):
    pars = [re.sub(r"\s+", " ", p.strip()) for p in re.split(r"\n\s*\n", template) if p.strip()]
    out = []
    for p in pars:
        bits = S.MARKER_RE.split(p)
        out.append("".join(f"\\slotmarker{{{b}}}" if k % 2 else tex(b)
                           for k, b in enumerate(bits)))
    return "\n\n".join(out)


def claims_table(atoms, slot_pos):
    L = ["\\setlength{\\tabcolsep}{2.5pt}",
         "\\begin{tabular}{@{}rp{0.76\\linewidth}l@{}}", "\\toprule",
         "& Claim & Label \\\\", "\\midrule"]
    for a in atoms:
        mark = "\\slotclaim{" + f"$a_{{{a['pos']}}}$" + "}" if a["pos"] in slot_pos \
            else f"$a_{{{a['pos']}}}$"
        lab = "true" if a["factual"] else "\\textbf{false}"
        L.append(f"{mark} & {tex(a['text'])} & {lab} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L)


def fillers_table(art, slot_list, width="0.66\\linewidth", used=None):
    """Every slot's fillers; slots not in ``used`` are marked as unused spares.

    One spanning header row per slot, then one row per filler kind, so the table fits a
    column (``width`` is the text column's width).
    """
    L = ["\\setlength{\\tabcolsep}{2.5pt}",
         f"\\begin{{tabular}}{{@{{}}lp{{{width}}}@{{}}}}", "\\toprule",
         "Filler & Text \\\\", "\\midrule"]
    for s in slot_list:
        spare = used is not None and s["id"] not in used
        role = ROLE_NAMES[s["role"]] + (" (spare)" if spare else "")
        L.append(f"\\multicolumn{{2}}{{@{{}}l}}{{\\slotmarker{{{s['id']}}} {role} slot, "
                 f"$a_{{{s['source_pos']}}}$--$a_{{{s['target_pos']}}}$}} \\\\")
        for kind in art["fillers"][s["id"]]:
            L.append(f"\\quad {KIND_NAMES.get(kind, kind)} & "
                     f"{tex(art['fillers'][s['id']][kind])} \\\\")
        L.append("\\addlinespace")
    L[-1] = "\\bottomrule"
    L.append("\\end{tabular}")
    return "\n".join(L)


def family_block(fid, fams, items, state):
    fam, art = fams[fid], state[fid]["artifacts"]
    primary = fam["primary"]
    pl = lad(fam, primary)
    ctl = next((l for l in fam["ladders"] if l["type"] == "CONTROL"), None)
    pslots = primary_slots(art, primary, items, pl)
    cslot = [s for s in art["plan"]["slots"] if s["role"] == "control"]
    slot_pos = {p for s in pslots + cslot for p in (s["source_pos"], s["target_pos"])}
    atoms = sorted(art["plan"]["atoms"], key=lambda a: a["pos"])
    base = items[pl["item_ids"][1]]
    L = [f"% Generated by scripts/locobench_paper_examples.py from {fid}.",
         f"\\paragraph{{Question.}} {tex(art['question'])} "
         f"({tex(fam['canonical_topic'])}, {tex(fam['domain'])}.)", "",
         "\\paragraph{Claims.} The plan's claims in assertion order, with their factuality "
         "labels; claims joined by a ladder's slots are boxed.", "",
         "\\begin{center}\\scriptsize", claims_table(atoms, slot_pos), "\\end{center}", "",
         "\\paragraph{Template.} The answer as written once, with its slot markers.", "",
         "\\begin{quote}\\small", render_template(art["template"]), "\\end{quote}", "",
         "\\paragraph{Fillers.} Each slot's fillers, as admitted by the committee.", "",
         "\\begin{center}\\scriptsize",
         fillers_table(art, art["plan"]["slots"], width="0.80\\linewidth",
                       used={x["id"] for x in pslots + cslot}),
         "\\end{center}", ""]
    # Rungs.
    L += ["\\paragraph{Rungs.} The five responses of the family, which filler each carries in "
          "the edited slots, and gold $\\LCS$ at the verdict and the asserted world.", "",
          "\\begin{center}\\scriptsize", "\\setlength{\\tabcolsep}{2.5pt}",
          "\\begin{tabular}{@{}l>{\\raggedright\\arraybackslash}p{0.27\\linewidth}"
          ">{\\raggedright\\arraybackslash}p{0.30\\linewidth}cc@{}}", "\\toprule",
          "& Ladder, rung & Fillers & \\multicolumn{2}{c}{$\\LCS$ at world} \\\\",
          "& & & verdict & asserted \\\\",
          "\\midrule"]
    rows = []
    for name, ladder in ((primary, pl), ("CONTROL", ctl)):
        if ladder is None:
            continue
        for k, iid in enumerate(ladder["item_ids"]):
            if name == "CONTROL" and k == 0:
                continue  # the primary's middle rung
            asg = items[iid]["expected"]["assignment"]
            sl = pslots if name == primary else cslot
            what = ", ".join(f"{s['id']}: {KIND_NAMES[asg[s['id']]]}" for s in sl)
            if name == "ORDER":
                po = items[iid]["expected"].get("paragraph_order") or []
                what = ("paragraphs " + "-".join(str(i + 1) for i in po)) if po \
                    else "paragraphs as written"
            lab = f"{name.lower()} $\\rho_{k}$"
            if name == primary and k == 1 and ctl is not None:
                lab += ", control $\\rho_0$"
            sc = score(items[iid])
            rows.append((iid.rsplit("-", 1)[1], lab, what, f"{sc['v']:.3f}", f"{sc['a']:.3f}"))
    for r in rows:
        L.append(" & ".join(r) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{center}", ""]
    # One response in full.
    order = base["expected"].get("paragraph_order")
    L += [f"\\paragraph{{Response {base['id'].rsplit('-', 1)[1]} in full}} "
          f"({primary.lower()} $\\rho_1$; edited slots highlighted, "
          f"{base['meta']['word_count']} words).", "",
          "\\begin{quote}\\small",
          render_text(art["template"], art["fillers"], base["expected"]["assignment"],
                      mark={s["id"] for s in pslots + cslot}, order=order),
          "\\end{quote}", ""]
    # Gold relations of the base response.
    L += [f"\\paragraph{{Gold relations of response {base['id'].rsplit('-', 1)[1]}.}} Sense, "
          "compiled coupling, strength band, and the relation's sign at the verdict (v.) and "
          "asserted (a.) world ($+$ support, $-$ conflict, $\\cdot$ no factor); a slot id marks "
          "a relation drawn by a filler.", "",
          "\\begin{center}\\scriptsize", "\\setlength{\\tabcolsep}{2.2pt}",
          "\\begin{tabular}{@{}lllllcc@{}}", "\\toprule",
          "Pair & & Sense & Coupling & Band & v. & a. \\\\", "\\midrule"]
    sg = {"support": "$+$", "conflict": "$-$", "none": "$\\cdot$"}
    for r in sorted(base["relations"], key=lambda r: (int(r["source_id"][1:]),
                                                      int(r["target_id"][1:]))):
        s, t = int(r["source_id"][1:]) + 1, int(r["target_id"][1:]) + 1
        src = r.get("slot_id") or ""
        L.append(f"$a_{{{s}}}\\to a_{{{t}}}$ & {src} & {r['level2_sense']} & "
                 f"{r['level1_coupling'].replace('_', '-')} & {r['intended_strength_band']} & "
                 f"{sg[r['sign_verdict']]} & {sg[r['sign_assertion']]} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    return "\n".join(L) + "\n"


def teaser(fid, fams, items, state):
    """Per rung: the edited slot sentences (macros the figure arranges) and gold LCS."""
    fam, art = fams[fid], state[fid]["artifacts"]
    pl = lad(fam, fam["primary"])
    pslots = primary_slots(art, fam["primary"], items, pl)
    L = [f"% Generated by scripts/locobench_paper_examples.py from {fid}."]
    for k, iid in enumerate(pl["item_ids"]):
        asg = items[iid]["expected"]["assignment"]
        for j, s in enumerate(pslots):
            L.append(f"\\expandafter\\def\\csname teaser{'ABC'[k]}{'ab'[j]}\\endcsname"
                     f"{{{tex(art['fillers'][s['id']][asg[s['id']]].strip())}}}")
            L.append(f"\\expandafter\\def\\csname teaserKind{'ABC'[k]}{'ab'[j]}\\endcsname"
                     f"{{{asg[s['id']]}}}")
            edge = {"conflict": "cnfedge", "support": "supedge"}.get(asg[s["id"]], "noedge")
            L.append(f"\\expandafter\\def\\csname teaserEdge{'ABC'[k]}{'ab'[j]}\\endcsname"
                     f"{{{edge}}}")
        L.append(f"\\expandafter\\def\\csname teaserLcs{'ABC'[k]}\\endcsname"
                 f"{{{score(items[iid])['v']:.2f}}}")
    for j, s in enumerate(pslots):
        L.append(f"\\expandafter\\def\\csname teaserPair{'ab'[j]}\\endcsname"
                 f"{{$a_{{{s['source_pos']}}}$--$a_{{{s['target_pos']}}}$}}")
    L.append(f"\\def\\teaserQuestion{{{tex(art['question'])}}}")
    L.append(f"\\def\\teaserFamily{{{fid}}}")
    L.append(f"\\def\\teaserWords{{{items[pl['item_ids'][1]]['meta']['word_count']}}}")
    return "\n".join(L) + "\n"


def worlds(fid, fams, items, state):
    """A SIGN ladder's slot: claims, labels, stances, and the coupling's sign per world."""
    fam, art = fams[fid], state[fid]["artifacts"]
    pl = lad(fam, "SIGN")
    s = primary_slots(art, "SIGN", items, pl)[0]
    atoms = {a["pos"]: a for a in art["plan"]["atoms"]}
    L = [f"% Generated by scripts/locobench_paper_examples.py from {fid}."]
    L.append(f"\\def\\wTrue{{{tex(atoms[s['source_pos']]['text'])}}}")
    L.append(f"\\def\\wFalse{{{tex(atoms[s['target_pos']]['text'])}}}")
    L.append(f"\\def\\wTruePos{{{s['source_pos']}}}")
    L.append(f"\\def\\wFalsePos{{{s['target_pos']}}}")
    L.append(f"\\def\\wFamily{{{fid}}}")
    for k, iid in enumerate(pl["item_ids"]):
        it = items[iid]
        r = slot_rel(it, s["id"])
        stance = next(a["stance"] for a in it["atoms"] if a["pos"] == s["target_pos"])
        C = "ABC"[k]
        L.append(f"\\expandafter\\def\\csname wText{C}\\endcsname"
                 f"{{{tex(art['fillers'][s['id']][it['expected']['assignment'][s['id']]])}}}")
        L.append(f"\\expandafter\\def\\csname wCoup{C}\\endcsname"
                 f"{{{(r['level1_coupling'] if r else 'none').replace('_', '-')}}}")
        L.append(f"\\expandafter\\def\\csname wSignV{C}\\endcsname"
                 f"{{{r['sign_verdict'] if r else 'none'}}}")
        L.append(f"\\expandafter\\def\\csname wSignA{C}\\endcsname"
                 f"{{{r['sign_assertion'] if r else 'none'}}}")
        L.append(f"\\expandafter\\def\\csname wStance{C}\\endcsname{{{stance}}}")
        L.append(f"\\expandafter\\def\\csname wXa{C}\\endcsname"
                 f"{{{0 if stance == 'denied' else 1}}}")
        col = {"support": "\\textcolor{lbsupport}{\\textbf{support}}",
               "conflict": "\\textcolor{lbconflict}{\\textbf{conflict}}",
               "none": "no factor"}
        for tag, key in (("V", "sign_verdict"), ("A", "sign_assertion")):
            L.append(f"\\expandafter\\def\\csname wSign{tag}Tex{C}\\endcsname"
                     f"{{{col[r[key]] if r else col['none']}}}")
        sc = score(it)
        L.append(f"\\expandafter\\def\\csname wLcsV{C}\\endcsname{{{sc['v']:.3f}}}")
        L.append(f"\\expandafter\\def\\csname wLcsA{C}\\endcsname{{{sc['a']:.3f}}}")
    return "\n".join(L) + "\n"


def readings(fid, sid, kind, fams, state):
    art = state[fid]["artifacts"]
    slot = next(s for s in art["plan"]["slots"] if s["id"] == sid)
    atoms = {a["pos"]: a for a in art["plan"]["atoms"]}
    log = [x for x in art["filler_log"] if x["slot"] == sid and x["kind"] == kind]
    L = [f"% Generated by scripts/locobench_paper_examples.py from {fid} {sid}/{kind}.",
         f"\\def\\rdFamily{{{fid}}}",
         f"\\def\\rdX{{{tex(atoms[slot['source_pos']]['text'])}}}",
         f"\\def\\rdY{{{tex(atoms[slot['target_pos']]['text'])}}}",
         f"\\def\\rdXPos{{{slot['source_pos']}}}", f"\\def\\rdYPos{{{slot['target_pos']}}}",
         "\\def\\rdTable{%",
         "\\begin{tabular}{@{}p{0.30\\linewidth}lp{0.10\\linewidth}p{0.14\\linewidth}"
         "p{0.20\\linewidth}@{}}", "\\toprule",
         "Filler (attempt) & Rater & Link read & Stances & Link to a third statement \\\\",
         "\\midrule"]
    names = {"gpt-6.1-sol": "GPT-6.1", "gemini-3.1-pro": "Gemini 3.1 Pro",
             "claude-opus-5-5": "Claude Opus 5.5"}
    for x in log:
        rd = x["readings"]
        for j, r in enumerate(rd):
            st = ", ".join(f"$a_{{{p}}}$ {v}" for p, v in sorted(r["status"].items(),
                                                                  key=lambda kv: int(kv[0])))
            other = "; ".join(tex(o) for o in r.get("other_links") or []) or "none"
            ok = "" if (r["link_ok"] and r["status_ok"] and not r.get("other_links")) \
                else " \\textbf{(fails)}"
            first = (f"\\multirow{{{len(rd)}}}{{=}}{{``{tex(x['filler'])}'' "
                     f"(attempt {x['attempt'] + 1}: {'admitted' if x['passed'] else 'rejected'})}}"
                     if j == 0 else "")
            L.append(f"{first} & {names.get(r['rater'], r['rater'])}{ok} & "
                     f"{r['link'].replace('_', ' ')} & {st} & {other} \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L += ["\\end{tabular}}"]
    return "\n".join(L) + "\n"


def schema_example(iid, items, fams, fid):
    """A released item and its family record, trimmed for the page (marked with "...")."""
    it = json.loads(json.dumps(items[iid]))
    it["response"] = it["response"][:160].rsplit(" ", 1)[0] + " ..."
    it["atoms"] = it["atoms"][:2] + ["..."]
    it["relations"] = it["relations"][:2] + ["..."]
    it["non_relations"] = it["non_relations"][:1] + ["..."]
    fam = json.loads(json.dumps(fams[fid]))
    fam["ladders"] = fam["ladders"][:1]
    fam["gold_corrections"] = fam["gold_corrections"][:1]
    return (json.dumps(it, indent=1, ensure_ascii=False) + "\n",
            json.dumps(fam, indent=1, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fams, items, state = load()
    for _t, fid in FAMILIES.items():
        open(os.path.join(args.out, f"family_{fid}.tex"), "w").write(
            family_block(fid, fams, items, state))
        art = state[fid]["artifacts"]
        open(os.path.join(args.out, f"template_{fid}.tex"), "w").write(
            render_template(art["template"]) + "\n")
        fam = fams[fid]
        sl = primary_slots(art, fam["primary"], items, lad(fam, fam["primary"])) + \
            [x for x in art["plan"]["slots"] if x["role"] == "control"]
        open(os.path.join(args.out, f"fillers_{fid}.tex"), "w").write(
            fillers_table(art, art["plan"]["slots"], width="0.86\\textwidth",
                          used={x["id"] for x in sl}) + "\n")
    open(os.path.join(args.out, f"teaser_{TEASER}.tex"), "w").write(
        teaser(TEASER, fams, items, state))
    open(os.path.join(args.out, f"worlds_{WORLDS}.tex"), "w").write(
        worlds(WORLDS, fams, items, state))
    a, b = schema_example(f"locobench-slot3-test-{TEASER}-t1", items, fams, TEASER)
    open(os.path.join(args.out, "schema_item.json"), "w").write(a)
    open(os.path.join(args.out, "schema_family.json"), "w").write(b)
    fid, sid, kind = READINGS
    open(os.path.join(args.out, f"readings_{fid}.tex"), "w").write(
        readings(fid, sid, kind, fams, state))
    print(f"wrote examples for {sorted(FAMILIES.values())}, teaser {TEASER}, worlds {WORLDS}, "
          f"readings {fid} to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
