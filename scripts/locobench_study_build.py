"""Build the LoCoBench human validation study: samples, assignment, Label Studio tasks, gold key.

Design: ``docs/arr2027/locobench/study/preregistration.md``. Two tasks, five annotators, each item
judged by three of them (balanced over the ten possible triples):

* **Task 1, filler reading** (aspects 1 and 3): 120 slot fillers -- 80 the committee admitted, 40
  it rejected -- each shown highlighted in its paragraph, with the slot's two claims. The
  annotator reports the connection the highlighted text draws between them, each claim's stance,
  and any link to another statement (the human counterpart of the committee's SV5 reading).
* **Task 2, rung comparison** (aspect 2): every rung pair of the primary and CONTROL ladders of
  30 families (10 CONFLICT, 10 CHAIN, 10 SIGN): which response is more logically coherent.

Outputs, under ``--out`` (default ``docs/arr2027/locobench/study/locobench-study``):

* ``bundles/annotator_<k>/`` -- what annotator k receives and nothing else: the two task files
  (Label Studio JSON), the practice files, the two labeling configs, the brief.
* ``key/key.json`` -- the gold key (sample, intended answers, committee verdicts, declared
  orders, attention-check answers). It is never imported into Label Studio and not opened until
  analysis; its SHA-256 goes into the pre-registration (``key/key.sha256``).

Every bundle is checked to carry no gold field (``check_no_leak``). The build is deterministic in
``--seed``.

    .venv/bin/python scripts/locobench_study_build.py [--seed 20261120]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import html
import itertools
import json
import os
import random
import re
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fact_reasoner.locobench import slots as S  # noqa: E402

TEST = "data/locobench-slot3-test"
DEV = "data/locobench-slot3-pilot"
STUDY_DOCS = "docs/arr2027/locobench/study"
ANNOTATORS = [1, 2, 3, 4, 5]
TRIPLES = list(itertools.combinations(ANNOTATORS, 3))          # 10
COMBOS = [("conflict_tt", "conflict"), ("conflict_tt", "none"),
          ("chain_link", "support"), ("chain_link", "none"),
          ("sign_tf", "conflict"), ("sign_tf", "none"), ("sign_tf", "support"),
          ("control", "para1"), ("control", "para2"), ("control", "para3")]
N_ADMITTED_PER_COMBO = 8                                        # 80
N_REJECTED_PER_COMBO = 4                                        # 40 (topped up, see below)
N_FAMILIES_PER_TYPE = 10                                        # Task 2: 30 families
TASK2_TYPES = ("CONFLICT", "CHAIN", "SIGN")

# ---------------------------------------------------------------------------------------------
# The answer options, shared by the labeling configs and the analysis (one source of truth).
# Each Task-1 link option maps to the committee's SV5 label vocabulary, so human and committee
# readings are scored with the same rule (slots.label_matches).
# ---------------------------------------------------------------------------------------------
LINK_OPTIONS = [
    ("X is given as a reason, cause or evidence for Y", "i_supports_j"),
    ("Y is given as a reason, cause or evidence for X", "j_supports_i"),
    ("X and Y cannot both be true (one rules out the other)", "incompatible"),
    ("X and Y say the same thing", "equivalent"),
    ("Exactly one of X and Y holds (not both, not neither)", "exactly_one"),
    ("At least one of X and Y holds (perhaps both)", "at_least_one"),
    ("Only contrasted, conceded, or ordered in time: no dependence", "contrast_only"),
    ("No connection is drawn between X and Y", "unlinked"),
]
STANCE_OPTIONS = [("asserted", "asserted"), ("denied", "denied"),
                  ("not stated in this passage", "absent")]
OTHER_OPTIONS = ["No", "Yes"]
CONFIDENCE_OPTIONS = ["1 (unsure)", "2", "3 (sure)"]
PREFERENCE_OPTIONS = ["A is more logically coherent", "About equal",
                      "B is more logically coherent"]

#: Data fields a task may carry. Anything else in a bundle is a leak.
TASK1_FIELDS = {"task_id", "context_html", "claim_x", "claim_y"}
TASK2_FIELDS = {"task_id", "question", "facts_html", "a_html", "b_html"}

STYLE = ("<style>.lb p{margin:0 0 .6em 0;line-height:1.45} .lb mark{background:#ffe08a;"
         "padding:0 .15em} .lb .facts{background:#f4f4f2;border-left:3px solid #898781;"
         "padding:.4em .7em;margin:.3em 0 .6em 0}</style>")


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------
def load(d):
    fams = {f["family_id"]: f for f in
            json.load(open(os.path.join(d, "families.json")))["families"]}
    items = {}
    for line in open(os.path.join(d, "items.jsonl")):
        if line.strip():
            it = json.loads(line)
            items[it["id"]] = it
    state = json.load(open(os.path.join(d, "state.json")))["families"]
    return fams, items, state


def base_assignment(fam, items):
    """The assignment of the family's primary middle rung (also CONTROL's first rung)."""
    prim = next(l for l in fam["ladders"] if l["type"] != "CONTROL") if any(
        l["type"] != "CONTROL" for l in fam["ladders"]) else fam["ladders"][0]
    return dict(items[prim["item_ids"][1]]["expected"]["assignment"])


def paragraphs(template):
    return [re.sub(r"\s+", " ", p.strip()) for p in re.split(r"\n\s*\n", template) if p.strip()]


def esc(t):
    return html.escape(t, quote=False)


def render(template, fillers, assignment, mark=(), override=None, only_par_of=None,
           order=None):
    """Template -> HTML paragraphs. Slots in ``mark`` are highlighted; ``override`` replaces one
    slot's text ({slot: text}); ``only_par_of`` keeps only the paragraph holding that marker."""
    override = override or {}
    pars = paragraphs(template)
    if order is not None:
        pars = [pars[i] for i in order]
    out = []
    for p in pars:
        if only_par_of and "{{" + only_par_of + "}}" not in p:
            continue
        bits = S.MARKER_RE.split(p)
        buf = []
        for k, b in enumerate(bits):
            if k % 2 == 0:
                buf.append(esc(b))
                continue
            text = override.get(b, fillers[b][assignment[b]]).strip()
            buf.append(f"<mark>{esc(text)}</mark>" if b in mark else esc(text))
        out.append("<p>" + re.sub(r" +([.,;:])", r"\1", "".join(buf)).strip() + "</p>")
    return out


def wrap(pars_html):
    return STYLE + "<div class='lb'>" + "".join(pars_html) + "</div>"


# ---------------------------------------------------------------------------------------------
# Task 1: fillers
# ---------------------------------------------------------------------------------------------
def filler_candidates(fams, items, state, exclude):
    """Every SV5-read filler of the released families, with its context and committee verdict."""
    out = collections.defaultdict(lambda: {"admitted": [], "rejected": []})
    for fid, fam in fams.items():
        if fid in exclude:
            continue
        art = state[fid]["artifacts"]
        slots = {s["id"]: s for s in art["plan"]["slots"]}
        owns = S.slot_owns(art["plan"])
        atoms = {a["pos"]: a for a in art["plan"]["atoms"]}
        base = base_assignment(fam, items)
        for x in art.get("filler_log", []):
            if not x.get("readings"):
                continue
            s = slots[x["slot"]]
            combo = (s["role"], x["kind"])
            if combo not in COMBOS:
                continue
            want = {s["source_pos"]: "asserted", s["target_pos"]: "asserted"}
            if s["role"] == "sign_tf" and x["kind"] == "support":
                want[s["target_pos"]] = "denied"
            asg = dict(base)
            asg[s["id"]] = x["kind"]
            ctx = render(art["template"], art["fillers"], asg, mark={s["id"]},
                         override={s["id"]: x["filler"]}, only_par_of=s["id"])
            rec = {
                "family_id": fid, "slot": s["id"], "role": s["role"], "kind": x["kind"],
                "attempt": x["attempt"], "committee_admitted": bool(x["passed"]),
                "committee_links": [r["link"] for r in x["readings"]],
                "expected_link": S.expected_label(S.slot_links(s)[x["kind"]]),
                "x_pos": s["source_pos"], "y_pos": s["target_pos"],
                "owned": owns.get(s["id"], [s["source_pos"], s["target_pos"]]),
                "expected_stance": want,
                "data": {"context_html": wrap(ctx), "claim_x": atoms[s["source_pos"]]["text"],
                         "claim_y": atoms[s["target_pos"]]["text"]},
            }
            out[combo]["admitted" if x["passed"] else "rejected"].append(rec)
    return out


def sample_fillers(cands, rng, n_adm=N_ADMITTED_PER_COMBO, n_rej=N_REJECTED_PER_COMBO):
    """10 * n_adm admitted + 10 * n_rej rejected fillers (topped up), one per family."""
    used, admitted, rejected = set(), [], []

    def take(pool, n, sink):
        pool = pool[:]
        rng.shuffle(pool)
        k = 0
        for r in pool:
            if k == n:
                break
            if r["family_id"] in used:
                continue
            used.add(r["family_id"])
            sink.append(r)
            k += 1
        return k

    short = 0
    for combo in COMBOS:  # rejected first: the scarcer pool claims its families first
        short += n_rej - take(cands[combo]["rejected"], n_rej, rejected)
    # Top up the rejected sample from the largest rejected pools, in order of size.
    for combo in sorted(COMBOS, key=lambda c: -len(cands[c]["rejected"])):
        if short <= 0:
            break
        short -= take(cands[combo]["rejected"], 1, rejected)
    for combo in COMBOS:
        got = take(cands[combo]["admitted"], n_adm, admitted)
        assert got == n_adm, (combo, got)
    assert len(admitted) == 10 * n_adm and len(rejected) == 10 * n_rej, (len(admitted),
                                                                          len(rejected))
    return admitted + rejected


# ---------------------------------------------------------------------------------------------
# Task 2: rung pairs
# ---------------------------------------------------------------------------------------------
def sample_families(fams, rng):
    out = []
    for t in TASK2_TYPES:
        cands = [f for f in fams.values() if f["primary"] == t
                 and any(l["type"] == "CONTROL" for l in f["ladders"])]
        by_dom = collections.defaultdict(list)
        for f in cands:
            by_dom[f["domain"]].append(f)
        for v in by_dom.values():
            rng.shuffle(v)
        doms = sorted(by_dom)
        rng.shuffle(doms)
        pick, i = [], 0
        while len(pick) < N_FAMILIES_PER_TYPE:  # round-robin over domains
            d = doms[i % len(doms)]
            if by_dom[d]:
                pick.append(by_dom[d].pop())
            i += 1
        out += pick
    return out


def family_pairs(fam, items, state):
    """The family's rung pairs: (ladder type, declared, i, j, html_i, html_j)."""
    art = state[fam["family_id"]]["artifacts"]
    atoms = sorted(art["plan"]["atoms"], key=lambda a: a["pos"])
    false = [a["text"] for a in atoms if not a["factual"]]
    facts = (STYLE + "<div class='lb'><div class='facts'><b>Take as given:</b> the following "
             "claims made in these answers are <b>false</b>.<ul>"
             + "".join(f"<li>{esc(t)}</li>" for t in false) + "</ul></div></div>")
    pairs = []
    for lad in fam["ladders"]:
        ids = lad["item_ids"]
        asg = [items[i]["expected"]["assignment"] for i in ids]
        for i, j in itertools.combinations(range(3), 2):
            diff = {s for s in asg[i] if asg[i][s] != asg[j][s]}
            hi = wrap(render(art["template"], art["fillers"], asg[i], mark=diff))
            hj = wrap(render(art["template"], art["fillers"], asg[j], mark=diff))
            pairs.append({"family_id": fam["family_id"], "ladder": lad["ladder_id"],
                          "type": lad["type"], "declared": lad["declared"],
                          "rungs": [i, j], "item_ids": [ids[i], ids[j]], "html": [hi, hj]})
    return art["question"], facts, pairs


# ---------------------------------------------------------------------------------------------
# Practice and attention checks (development split only)
# ---------------------------------------------------------------------------------------------
def dev_fillers(rng):
    fams, items, state = load(DEV)
    cands = filler_candidates(fams, items, state, exclude=set())
    clear = {}
    for combo in COMBOS:
        # unanimous, first-attempt admitted readings: unambiguous practice and check items
        pool = [r for r in cands[combo]["admitted"] if r["attempt"] == 0
                and len(set(r["committee_links"])) == 1]
        rng.shuffle(pool)
        clear[combo] = pool
    return fams, items, state, clear


def _choices(name, to, options, required=True, inline=False):
    opts = "".join(f'\n    <Choice value="{html.escape(o)}"/>' for o in options)
    return (f'  <Choices name="{name}" toName="{to}" choice="single-radio"'
            f'{" required=\"true\"" if required else ""}{" showInline=\"true\"" if inline else ""}>'
            f"{opts}\n  </Choices>")


def write_configs(d):
    """The two Label Studio labeling configs, generated from the option lists above."""
    os.makedirs(d, exist_ok=True)
    t1 = "\n".join([
        "<View>",
        '  <Header value="Read the passage. Judge only the HIGHLIGHTED text." size="4"/>',
        '  <HyperText name="ctx" value="$context_html" inline="true"/>',
        '  <View style="background:#f4f4f2;padding:0.5em 0.8em;margin:0.6em 0">',
        '    <Header value="Claim X" size="6"/>',
        '    <Text name="x" value="$claim_x"/>',
        '    <Header value="Claim Y" size="6"/>',
        '    <Text name="y" value="$claim_y"/>',
        "  </View>",
        '  <Header value="1. What connection does the highlighted text draw between X and Y?"'
        ' size="5"/>',
        _choices("link", "ctx", [o for o, _ in LINK_OPTIONS]),
        '  <Header value="2. How does the highlighted text treat claim X?" size="5"/>',
        _choices("stance_x", "ctx", [o for o, _ in STANCE_OPTIONS], inline=True),
        '  <Header value="3. How does the highlighted text treat claim Y?" size="5"/>',
        _choices("stance_y", "ctx", [o for o, _ in STANCE_OPTIONS], inline=True),
        '  <Header value="4. Does the highlighted text also connect X or Y to some OTHER'
        ' statement?" size="5"/>',
        _choices("other", "ctx", OTHER_OPTIONS, inline=True),
        '  <View visibleWhen="choice-selected" whenTagName="other" whenChoiceValue="Yes">',
        '    <TextArea name="other_quote" toName="ctx" rows="2" maxSubmissions="1"'
        ' placeholder="Quote the words that make that connection"/>',
        "  </View>",
        '  <Header value="5. How sure are you?" size="5"/>',
        _choices("confidence", "ctx", CONFIDENCE_OPTIONS, inline=True),
        "</View>"])
    t2 = "\n".join([
        "<View>",
        '  <Header value="$question" size="4"/>',
        '  <HyperText name="facts" value="$facts_html" inline="true"/>',
        '  <View style="display:flex;gap:1.2em;align-items:flex-start">',
        '    <View style="flex:1;min-width:0">',
        '      <Header value="Response A" size="5"/>',
        '      <HyperText name="a" value="$a_html" inline="true"/>',
        "    </View>",
        '    <View style="flex:1;min-width:0">',
        '      <Header value="Response B" size="5"/>',
        '      <HyperText name="b" value="$b_html" inline="true"/>',
        "    </View>",
        "  </View>",
        '  <Header value="Which response is more logically coherent, given the facts above?"'
        ' size="5"/>',
        _choices("preference", "a", PREFERENCE_OPTIONS, inline=True),
        '  <Header value="How sure are you?" size="5"/>',
        _choices("confidence", "a", CONFIDENCE_OPTIONS, inline=True),
        '  <TextArea name="reason" toName="a" rows="1" maxSubmissions="1"'
        ' placeholder="Optional: one-line reason"/>',
        "</View>"])
    open(os.path.join(d, "filler_reading.xml"), "w").write(t1 + "\n")
    open(os.path.join(d, "rung_comparison.xml"), "w").write(t2 + "\n")


def build(seed, out, reduced=False):
    write_configs(os.path.join(STUDY_DOCS, "labelstudio"))
    rng = random.Random(seed)
    fams, items, state = load(TEST)
    task2_fams = sample_families(fams, rng)
    exclude = {f["family_id"] for f in task2_fams}
    fillers = sample_fillers(filler_candidates(fams, items, state, exclude), rng,
                             *((7, 3) if reduced else ()))

    # Practice (6 fillers, 1 family) and checks (4 fillers, 2 identical pairs): dev split.
    dfams, ditems, dstate, clear = dev_fillers(rng)
    practice_combos = [("conflict_tt", "conflict"), ("conflict_tt", "none"),
                       ("chain_link", "support"), ("sign_tf", "conflict"),
                       ("sign_tf", "support"), ("control", "para2")]
    check_combos = [("conflict_tt", "conflict"), ("chain_link", "support"),
                    ("control", "para1"), ("chain_link", "none")]
    practice, checks1, used_dev = [], [], set()
    for combo in practice_combos + check_combos:
        r = next(x for x in clear[combo] if (x["family_id"], x["slot"]) not in used_dev)
        used_dev.add((r["family_id"], r["slot"]))
        (practice if len(practice) < len(practice_combos) else checks1).append(r)
    dev_t2 = [f for f in dfams.values() if f["primary"] in TASK2_TYPES
              and any(l["type"] == "CONTROL" for l in f["ladders"])]
    rng.shuffle(dev_t2)
    pq, pfacts, ppairs = family_pairs(dev_t2[0], ditems, dstate)
    cq, cfacts, cpairs = family_pairs(dev_t2[1], ditems, dstate)
    # Development families already shown to annotators (practice, attention checks): the brief's
    # real walk-through must come from another one. Computed here, never read from the key.
    dev_shown = {f for f, _ in used_dev} | {dev_t2[0]["family_id"], dev_t2[1]["family_id"]}

    allc = filler_candidates(fams, items, state, exclude=set())
    population = {f"{r}/{k}": {"admitted": len(allc[(r, k)]["admitted"]),
                               "rejected": len(allc[(r, k)]["rejected"])} for r, k in COMBOS}
    key = {"seed": seed, "task1": {}, "task2": {}, "practice": {}, "assignment": {},
           "population": population}
    tid = iter(rng.sample(range(10000, 99999), 2000))

    def new_id(prefix):
        return f"{prefix}-{next(tid)}"

    # ---- Task 1 items, balanced over triples (stratified: shuffle within combo, deal out) ---
    order = sorted(fillers, key=lambda r: (COMBOS.index((r["role"], r["kind"])),
                                           not r["committee_admitted"], rng.random()))
    t1_items = []
    for k, r in enumerate(order):
        t = new_id("F")
        key["task1"][t] = {kk: v for kk, v in r.items() if kk != "data"}
        key["task1"][t]["check"] = False
        t1_items.append((t, r["data"], TRIPLES[k % len(TRIPLES)]))
    # ---- Task 2 families ---------------------------------------------------------------------
    t2_fams = []
    rng.shuffle(task2_fams)
    for k, fam in enumerate(task2_fams):
        q, facts, pairs = family_pairs(fam, items, state)
        t2_fams.append((fam["family_id"], q, facts, pairs, TRIPLES[k % len(TRIPLES)]))

    bundles = {a: {"task1": [], "task2": []} for a in ANNOTATORS}
    for t, data, tri in t1_items:
        for a in tri:
            bundles[a]["task1"].append({"data": {"task_id": t, **data}})
    side = collections.Counter()
    for fid, q, facts, pairs, tri in t2_fams:
        for a in tri:
            block = []
            prs = pairs[:]
            rng.shuffle(prs)
            for p in prs:
                swap = side[a] % 2 == 1  # alternate sides per annotator: exactly balanced
                side[a] += 1
                t = new_id("P")
                lo, hi = (1, 0) if swap else (0, 1)
                key["task2"][t] = {kk: v for kk, v in p.items() if kk != "html"}
                key["task2"][t].update({"annotator": a, "a_is_rung": p["rungs"][lo],
                                        "b_is_rung": p["rungs"][hi], "check": False})
                block.append({"data": {"task_id": t, "question": q, "facts_html": facts,
                                       "a_html": p["html"][lo], "b_html": p["html"][hi]}})
            bundles[a]["task2"].append(block)
    # ---- attention checks, inserted at seeded positions ------------------------------------
    for a in ANNOTATORS:
        t1 = bundles[a]["task1"]
        rng.shuffle(t1)
        for r in checks1:
            t = new_id("F")
            key["task1"][t] = {kk: v for kk, v in r.items() if kk != "data"}
            key["task1"][t].update({"check": True, "split": "dev"})
            t1.insert(rng.randrange(5, len(t1)), {"data": {"task_id": t, **r["data"]}})
        blocks = bundles[a]["task2"]
        rng.shuffle(blocks)
        for c in range(2):  # an identical pair: the only right answer is "About equal"
            p = cpairs[c]
            t = new_id("P")
            key["task2"][t] = {"check": True, "split": "dev", "expected": "About equal",
                               "family_id": p["family_id"], "annotator": a}
            blocks.insert(rng.randrange(1, len(blocks)), [
                {"data": {"task_id": t, "question": cq, "facts_html": cfacts,
                          "a_html": p["html"][0], "b_html": p["html"][0]}}])
        bundles[a]["task2"] = [x for blk in blocks for x in blk]
        key["assignment"][str(a)] = {"task1": len(t1), "task2": len(bundles[a]["task2"])}
    # ---- practice (same for everyone; answers discussed in the briefing) -------------------
    prac1, prac2 = [], []
    for r in practice:
        t = new_id("X")
        key["practice"][t] = {kk: v for kk, v in r.items() if kk != "data"}
        prac1.append({"data": {"task_id": t, **r["data"]}})
    for p in ppairs:
        t = new_id("X")
        key["practice"][t] = {kk: v for kk, v in p.items() if kk != "html"}
        prac2.append({"data": {"task_id": t, "question": pq, "facts_html": pfacts,
                               "a_html": p["html"][0], "b_html": p["html"][1]}})
    practice_md = practice_answers(key, prac1, prac2)
    # The brief: the master text with its real Task 1 walk-through filled in. No rng is used, so
    # the key is the same with or without the walk-through.
    brief = render_brief(clear, dev_shown)
    open(os.path.join(STUDY_DOCS, "brief_rendered.md"), "w").write(brief)
    # ---- write --------------------------------------------------------------------------------
    if os.path.isdir(os.path.join(out, "bundles")):
        shutil.rmtree(os.path.join(out, "bundles"))
    os.makedirs(os.path.join(out, "key"), exist_ok=True)
    for a in ANNOTATORS:
        d = os.path.join(out, "bundles", f"annotator_{a}")
        os.makedirs(d, exist_ok=True)
        json.dump(bundles[a]["task1"], open(os.path.join(d, "task1_fillers.json"), "w"), indent=1)
        json.dump(bundles[a]["task2"], open(os.path.join(d, "task2_pairs.json"), "w"), indent=1)
        json.dump(prac1, open(os.path.join(d, "practice_task1.json"), "w"), indent=1)
        json.dump(prac2, open(os.path.join(d, "practice_task2.json"), "w"), indent=1)
        open(os.path.join(d, "practice_answers.md"), "w").write(practice_md)
        for f in ("filler_reading.xml", "rung_comparison.xml", "setup.md"):
            src = os.path.join(STUDY_DOCS, "labelstudio" if f.endswith(".xml") else "", f)
            if os.path.exists(src):
                shutil.copy2(src, d)
        open(os.path.join(d, "brief.md"), "w").write(brief)
        shutil.copy2(os.path.join(os.path.dirname(__file__), "locobench_study_projects.py"), d)
        check_no_leak(d)
    kp = os.path.join(out, "key", "key.json")
    json.dump(key, open(kp, "w"), indent=1, sort_keys=True)
    digest = hashlib.sha256(open(kp, "rb").read()).hexdigest()
    open(os.path.join(out, "key", "key.sha256"), "w").write(digest + "  key.json\n")
    return key, digest, bundles


def practice_answers(key, prac1, prac2):
    """The practice items' intended answers (development split: safe to share), for the briefing."""
    inv = {v: k for k, v in LINK_OPTIONS}
    stance = {v: k for k, v in STANCE_OPTIONS}
    L = ["# Practice answers (development split)", "",
         "Discuss these together after everyone has done the practice projects. They are the",
         "answers the benchmark intends; the point is to agree on how to read the questions.", "",
         "## Practice 1: filler reading", ""]
    for i, t in enumerate(prac1, 1):
        k = key["practice"][t["data"]["task_id"]]
        st = {int(p): v for p, v in k["expected_stance"].items()}
        owned = set(k["owned"])
        sx = stance[st[k["x_pos"]]] if k["x_pos"] in owned else "(not scored)"
        sy = stance[st[k["y_pos"]]] if k["y_pos"] in owned else "(not scored)"
        L += [f"{i}. `{t['data']['task_id']}` -- connection: **{inv[k['expected_link']]}**; "
              f"X: {sx}; Y: {sy}; other statement: No."]
    L += ["", "Any of the two no-dependence answers (\"Only contrasted, conceded, or ordered in "
          "time\" and \"No connection\") counts the same.", "",
          "## Practice 2: response comparison", "",
          "The six screens are every pair of one family's responses. In each pair, the response "
          "whose highlighted text sets two true claims against each other, or derives a false "
          "claim from a true one, is the less coherent one; one that adds a sound inference, or "
          "refutes a false claim with a true one, is the more coherent one; two wordings of the "
          "same inference are about equal.", ""]
    for i, t in enumerate(prac2, 1):
        k = key["practice"][t["data"]["task_id"]]
        hi = max(k["rungs"])
        if k["declared"] == "invariant":
            ans = "About equal"
        else:
            ans = "B is more logically coherent" if k["rungs"][1] == hi else "A is more logically coherent"
        L += [f"{i}. `{t['data']['task_id']}` -- **{ans}**"]
    return "\n".join(L) + "\n"


EXAMPLE_MARKER = "<!-- EXAMPLE:TASK1-REAL -->"


def _md(html_text):
    """Task HTML -> Markdown: <mark> becomes bold, paragraphs become blank-line separated."""
    t = re.sub(r"<style>.*?</style>", "", html_text, flags=re.S)
    t = re.sub(r"</p>\s*<p>", "\n\n", t)
    t = re.sub(r"<mark>(.*?)</mark>", r"**\1**", t, flags=re.S)
    t = re.sub(r"<[^>]+>", "", t)
    return html.unescape(t).strip()


WHY = {"i_supports_j": "The highlighted text gives X as the reason for, or evidence of, Y.",
       "j_supports_i": "The highlighted text gives Y as the reason for, or evidence of, X.",
       "incompatible": "The highlighted text says that the two claims cannot both hold.",
       "unlinked": "Both claims are stated, but nothing in the highlighted text links them."}


def render_brief(clear, dev_shown, combo=("chain_link", "support")):
    """The master brief with one real development-split screen and its intended answer.

    The filler is an admitted, unanimous, first-attempt reading of ``combo`` from the first
    development family (in id order) that the practice items and attention checks do not use.
    """
    master = open(os.path.join(STUDY_DOCS, "brief.md")).read()
    if EXAMPLE_MARKER not in master:
        return master
    pool = sorted((r for r in clear[combo] if r["family_id"] not in dev_shown),
                  key=lambda r: (r["family_id"], r["slot"]))
    r = pool[0]
    inv = {v: k for k, v in LINK_OPTIONS}
    stance = {v: k for k, v in STANCE_OPTIONS}
    st = {int(p): v for p, v in r["expected_stance"].items()}
    sx, sy = stance[st[r["x_pos"]]], stance[st[r["y_pos"]]]
    quote = "\n".join("> " + line if line else ">" for line in _md(r["data"]["context_html"]).split("\n"))
    block = "\n".join([
        "### Task 1: one real screen",
        "",
        "This passage comes from the development split, from an answer that is not used anywhere",
        "in the study or the practice. It shows what a screen looks like.",
        "",
        quote,
        "",
        f"X: *{r['data']['claim_x']}*",
        f"Y: *{r['data']['claim_y']}*",
        "",
        f"- Connection: **{inv[r['expected_link']]}**. {WHY[r['expected_link']]}",
        f"- X: {sx}. Y: {sy}.",
        "- Other statement: No. The highlighted text links only X and Y.",
    ])
    out = master.replace(EXAMPLE_MARKER, block)
    for bad in (r"\bf\d{3}\b", r"locobench-slot3", r"conflict_tt|chain_link|sign_tf",
                r"\bpara[123]\b", r"i_supports_j|j_supports_i|contrast_only|exactly_one",
                r"\{\{S\d+\}\}"):
        assert not re.search(bad, out), ("brief", bad)
    return out


def check_no_leak(d):
    """A bundle carries task text only: no gold field, no family or ladder id, no slot kind."""
    for name, allowed in (("task1_fillers.json", TASK1_FIELDS), ("task2_pairs.json",
                                                                  TASK2_FIELDS),
                          ("practice_task1.json", TASK1_FIELDS),
                          ("practice_task2.json", TASK2_FIELDS)):
        tasks = json.load(open(os.path.join(d, name)))
        for t in tasks:
            assert set(t) == {"data"}, (name, set(t))
            assert set(t["data"]) <= allowed, (name, set(t["data"]) - allowed)
            blob = json.dumps(t["data"])
            # identifiers and internal labels only: words such as "committee" or "ladder" can
            # occur in the responses themselves
            for bad in (r"\bf\d{3}\b", r"locobench-slot3", r"conflict_tt|chain_link|sign_tf",
                        r"\bpara[123]\b", r"i_supports_j|j_supports_i|contrast_only|exactly_one",
                        r"\{\{S\d+\}\}"):
                assert not re.search(bad, blob), (name, bad, t["data"]["task_id"])
            assert re.fullmatch(r"[FPX]-\d{5}", t["data"]["task_id"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20261120)
    ap.add_argument("--out", default="docs/arr2027/locobench/study/locobench-study")
    ap.add_argument("--reduced", action="store_true",
                    help="The pre-registered timing fallback: Task 1 with 100 fillers (7+3).")
    args = ap.parse_args()
    key, digest, bundles = build(args.seed, args.out, args.reduced)
    t1 = [v for v in key["task1"].values() if not v["check"]]
    t2 = [v for v in key["task2"].values() if not v["check"]]
    print(f"task 1: {len(t1)} fillers ({sum(v['committee_admitted'] for v in t1)} admitted), "
          f"{sum(v['check'] for v in key['task1'].values())} check items")
    print(f"task 2: {len({v['family_id'] for v in t2})} families, "
          f"{len({(v['ladder'], tuple(v['rungs'])) for v in t2})} distinct pairs, "
          f"{len(t2)} judgements")
    for a in ANNOTATORS:
        print(f"  annotator {a}: {len(bundles[a]['task1'])} task-1 screens, "
              f"{len(bundles[a]['task2'])} task-2 screens")
    print(f"key sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
