"""Fairer LLM judges for the coherence paper (review M6 and M9, Phase 3.2 and 3.7).

Modes:

``absolute``  LoCoBench: a 1-5 rating of logical coherence, given the response AND its claims
              marked TRUE/FALSE by the factuality labels, with the construct spelled out.
``pairwise``  LoCoBench: for every rung pair of every ladder, which of two responses is more
              logically coherent ([A], [B] or [=]), asked in both orders to cancel position
              bias; ``--verdicts`` gives the judge the claims and their truth values.
``rubric``    ASAP 2.0: a 1-6 holistic score with the essay rubric (a rubric-prompted grader),
              written where scripts/asap_lcs.py reads baselines.
``eval``      LoCoBench: rise / fall / tie of each judge on the declared orders, and macros.

The gateway cache is bypassed (LOCOBENCH_NO_CACHE=1) so repeated runs are real samples.

    LOCOBENCH_NO_CACHE=1 .venv/bin/python scripts/run_fair_judges.py absolute --seeds 3
    LOCOBENCH_NO_CACHE=1 .venv/bin/python scripts/run_fair_judges.py pairwise --verdicts
    LOCOBENCH_NO_CACHE=1 .venv/bin/python scripts/run_fair_judges.py pairwise
    LOCOBENCH_NO_CACHE=1 .venv/bin/python scripts/run_fair_judges.py rubric --seeds 3
    .venv/bin/python scripts/run_fair_judges.py eval --macros docs/arr2027/coherence/numbers_fairjudge.tex
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import statistics as st
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

DATA = os.path.join(REPO, "data", "locobench-slot3-test")
OUT = os.path.join(REPO, "results", "arr2027_slot3_test", "fair_judges")
ASAP = os.path.join(REPO, "data", "ext", "asap-120")
ASAP_OUT = os.path.join(REPO, "results", "ext", "asap_120", "baselines")

CONSTRUCT = (
    "Logical coherence is how well the logical relations a response draws between its claims "
    "(inferences, explanations, refutations, contradictions) agree with the claims' truth "
    "values. Supporting or deriving a true claim from true claims, and refuting a false claim, "
    "are coherent. Setting two true claims against each other, or deriving a false claim from a "
    "true one, is incoherent. Do not judge whether the claims are true, and do not judge style "
    "or organization: judge only whether the relations the response draws fit the truth values.")

ABSOLUTE = """You will read a response and the claims it makes, each marked TRUE or FALSE by a
fact-check.

{construct}

Claims:
{claims}

Response:
{response}

Rate the response's logical coherence from 1 (lowest) to 5 (highest). Reason briefly, then give
your rating as a single digit in square brackets, for example [3]."""

PAIRWISE_V = """You will compare two responses that make the same claims, each marked TRUE or FALSE
by a fact-check.

{construct}

Claims:
{claims}

Response A:
{a}

Response B:
{b}

Which response is more logically coherent? If they are equally coherent, say so. Reason briefly,
then answer [A], [B] or [=]."""

PAIRWISE = """You will compare two responses to the same query.

Which response is more logically coherent -- which hangs together better as an argument, without
contradicting itself? Ignore style and organization. If they are equally coherent, say so.

Response A:
{a}

Response B:
{b}

Reason briefly, then answer [A], [B] or [=]."""

RUBRIC = """You will score a persuasive essay written by a student in grades 6-10, who read an
article and argued a position about it. Use this holistic rubric:

6 - clear and consistent mastery: develops a point of view effectively, with outstanding critical
    thinking; uses clearly appropriate examples, reasons and evidence from the source; is well
    organized and focused, with a smooth progression of ideas; uses language skillfully.
5 - reasonably consistent mastery: develops a point of view effectively, with strong critical
    thinking; generally appropriate evidence; well organized, with coherence and progression;
    skillful language, with occasional lapses.
4 - adequate mastery: develops a point of view, with competent critical thinking; adequate
    evidence; generally organized and focused, with some lapses in coherence; adequate language.
3 - developing mastery: develops a point of view with some critical thinking, inconsistently;
    inadequate evidence; limited organization or focus; weak or inconsistent language.
2 - little mastery: a vague or seriously limited point of view; weak critical thinking;
    inappropriate or insufficient evidence; poor organization; frequent language errors.
1 - very little or no mastery: no viable point of view, or little or no evidence; disorganized
    or unfocused; severe language errors that interfere with meaning.

Essay:
{response}

Reason briefly, then give your score as a single digit in square brackets, for example [3]."""

_DIGIT = re.compile(r"\[\s*([1-6])\s*\]")
_CHOICE = re.compile(r"\[\s*(A|B|=)\s*\]")
_LOCK = threading.Lock()


def jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()] if os.path.exists(path) else []


def claims_block(item, verdicts=True):
    atoms = sorted(item["atoms"], key=lambda a: a["pos"])
    return "\n".join(f"{k + 1}. {a['text']}" + (f" [{'TRUE' if a['factual'] else 'FALSE'}]"
                                                if verdicts else "")
                     for k, a in enumerate(atoms))


def make_generate(model):
    from run_slot3_baselines import _backend
    from fact_reasoner.coherence_baselines.judges import make_backend_generate
    gen = make_backend_generate(_backend(model))
    return lambda prompt: gen(prompt)[0]


def run_jobs(jobs, path, worker, workers):
    done = {json.dumps(r["key"], sort_keys=True) for r in jsonl(path) if r.get("ok")}
    todo = [j for j in jobs if json.dumps(j["key"], sort_keys=True) not in done]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} done, {len(todo)} to run -> {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def one(job):
        try:
            row = worker(job)
        except Exception as e:  # noqa: BLE001 - logged and retried on resume
            row = {"key": job["key"], "ok": False, "error": str(e)[:300]}
        with _LOCK, open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, row in enumerate(ex.map(one, todo)):
            if k % 50 == 0:
                print(f"  {k + 1}/{len(todo)} {row.get('ok')}", flush=True)


def load_loco():
    import arr_slot3_eval as E
    E.DATA = DATA
    items, _fams, ladders = E.load()
    return items, ladders


def cmd_absolute(args):
    items, _ = load_loco()
    gen = make_generate(args.model)
    jobs = [{"key": [iid, s], "iid": iid} for iid in sorted(items) for s in range(args.seeds)]

    def worker(job):
        it = items[job["iid"]]
        text = gen(ABSOLUTE.format(construct=CONSTRUCT, claims=claims_block(it),
                                   response=it["response"].strip()))
        m = _DIGIT.findall(text)
        r = int(m[-1]) if m and int(m[-1]) <= 5 else None
        return {"key": job["key"], "ok": r is not None, "rating": r, "tail": text[-300:]}

    run_jobs(jobs, os.path.join(OUT, f"absolute__{args.model}.jsonl"), worker, args.workers)
    return 0


def cmd_pairwise(args):
    items, ladders = load_loco()
    gen = make_generate(args.model)
    jobs = []
    for lad in ladders:
        ids = lad["item_ids"]
        for i, j in itertools.combinations(range(3), 2):
            for order in ("ij", "ji"):
                jobs.append({"key": [lad["ladder_id"], i, j, order], "a": ids[i] if order == "ij"
                             else ids[j], "b": ids[j] if order == "ij" else ids[i]})
    tmpl = PAIRWISE_V if args.verdicts else PAIRWISE

    def worker(job):
        a, b = items[job["a"]], items[job["b"]]
        text = gen(tmpl.format(construct=CONSTRUCT, claims=claims_block(a),
                               a=a["response"].strip(), b=b["response"].strip()))
        m = _CHOICE.findall(text)
        c = m[-1] if m else None
        return {"key": job["key"], "ok": c is not None, "choice": c, "tail": text[-300:]}

    tag = "pairwise_verdicts" if args.verdicts else "pairwise"
    run_jobs(jobs, os.path.join(OUT, f"{tag}__{args.model}.jsonl"), worker, args.workers)
    return 0


def cmd_rubric(args):
    items = {r["id"]: r for r in jsonl(os.path.join(ASAP, "items.jsonl"))}
    gen = make_generate(args.model)
    jobs = [{"key": [iid, s], "iid": iid} for iid in sorted(items) for s in range(args.seeds)]
    raw = os.path.join(ASAP_OUT, "raw", f"rubric__{args.model}.jsonl")

    def worker(job):
        text = gen(RUBRIC.format(response=items[job["iid"]]["response"].strip()))
        m = _DIGIT.findall(text)
        r = int(m[-1]) if m else None
        return {"key": job["key"], "ok": r is not None, "rating": r, "tail": text[-300:]}

    run_jobs(jobs, raw, worker, args.workers)
    by = defaultdict(list)
    for r in jsonl(raw):
        if r.get("ok"):
            by[r["key"][0]].append(r["rating"])
    with open(os.path.join(ASAP_OUT, f"judge__{args.model}__rubric.jsonl"), "w") as f:
        for iid, v in sorted(by.items()):
            f.write(json.dumps({"item_id": iid, "name": f"judge_rubric@{args.model}",
                                "score": st.mean(v), "diagnostics": {"runs": v}}) + "\n")
    print(f"wrote rubric scores for {len(by)} essays")
    return 0


def cmd_eval(args):
    import arr_slot3_eval as E
    items, ladders = load_loco()
    inc_types = E.INCREASE
    res, pair_rows = {}, {}
    # absolute: mean rating over seeds per item, decided like every other score (relative tie)
    for path in sorted(p for p in os.listdir(OUT) if p.startswith("absolute__")):
        by = defaultdict(list)
        for r in jsonl(os.path.join(OUT, path)):
            if r.get("ok"):
                by[r["key"][0]].append(r["rating"])
        mean = {k: st.mean(v) for k, v in by.items()}
        sd = {k: (st.pstdev(v) if len(v) > 1 else 0.0) for k, v in by.items()}
        c = defaultdict(lambda: [0, 0, 0, 0])  # rise, fall, tie, n
        for lad in ladders:
            ids = lad["item_ids"]
            if any(i not in mean for i in ids):
                continue
            for i, j in itertools.combinations(range(3), 2):
                d = mean[ids[j]] - mean[ids[i]]
                g = "inc" if lad["type"] in inc_types else "inv"
                for key in (g, lad["type"]):
                    c[key][0] += d > 1e-9
                    c[key][1] += d < -1e-9
                    c[key][2] += abs(d) <= 1e-9
                    c[key][3] += 1
                if g == "inv":
                    c["inv_spread"][0] += abs(d) <= max(sd[ids[i]], sd[ids[j]]) + 1e-9
                    c["inv_spread"][3] += 1
        res[path[:-6]] = dict(c)
        prs = []
        for lad in ladders:
            ids = lad["item_ids"]
            if any(i not in mean for i in ids):
                continue
            for i, j in itertools.combinations(range(3), 2):
                d = mean[ids[j]] - mean[ids[i]]
                tie = abs(d) <= 1e-9
                ok = (d > 1e-9) if lad["type"] in inc_types else tie
                prs.append({"family": lad["family_id"], "ladder": lad["ladder_id"],
                            "type": lad["type"], "pair": [i, j], "ok": bool(ok),
                            "ok_tol": bool(ok), "tie": bool(tie)})
        pair_rows[path[:-6]] = prs
    # pairwise: net preference for the later rung over the two orders
    for path in sorted(p for p in os.listdir(OUT) if p.startswith("pairwise")):
        votes = defaultdict(int)
        seen = defaultdict(int)
        for r in jsonl(os.path.join(OUT, path)):
            if not r.get("ok"):
                continue
            lid, i, j, order = r["key"]
            later = "B" if order == "ij" else "A"
            v = 0 if r["choice"] == "=" else (1 if r["choice"] == later else -1)
            votes[(lid, i, j)] += v
            seen[(lid, i, j)] += 1
        c = defaultdict(lambda: [0, 0, 0, 0])
        prs = []
        for lad in ladders:
            for i, j in itertools.combinations(range(3), 2):
                k = (lad["ladder_id"], i, j)
                if seen.get(k) != 2:
                    continue
                tie = votes[k] == 0
                ok = (votes[k] > 0) if lad["type"] in inc_types else tie
                prs.append({"family": lad["family_id"], "ladder": lad["ladder_id"],
                            "type": lad["type"], "pair": [i, j], "ok": bool(ok),
                            "ok_tol": bool(ok), "tie": bool(tie)})
                g = "inc" if lad["type"] in inc_types else "inv"
                for key in (g, lad["type"]):
                    c[key][0] += votes[k] > 0
                    c[key][1] += votes[k] < 0
                    c[key][2] += votes[k] == 0
                    c[key][3] += 1
        res[path[:-6]] = dict(c)
        pair_rows[path[:-6]] = prs
    json.dump({"counts": res, "pairs": pair_rows}, open(os.path.join(OUT, "eval.json"), "w"))
    for k, v in res.items():
        print(k, {a: b for a, b in v.items()})
    if args.macros:
        L = ["% Generated by scripts/run_fair_judges.py eval -- do not edit by hand."]
        tagmap = {"absolute__gpt-6-sol": "Abs", "pairwise_verdicts__gpt-6-sol": "PairV",
                  "pairwise__gpt-6-sol": "Pair"}
        for k, v in res.items():
            t = tagmap.get(k)
            if not t:
                continue
            for key, kt in (("inc", "Inc"), ("inv", "Inv"), ("CONFLICT", "Conflict"),
                            ("CHAIN", "Chain"), ("SIGN", "Sign"), ("ORDER", "Order"),
                            ("CONTROL", "Control"), ("inv_spread", "InvSpread")):
                if key not in v:
                    continue
                rise, fall, tie, n = v[key]
                pct = lambda a: f"{100 * a / n:.0f}" if n else "--"  # noqa: E731
                L.append(f"\\newcommand{{\\fj{t}{kt}Rise}}{{{pct(rise)}}}")
                L.append(f"\\newcommand{{\\fj{t}{kt}Fall}}{{{pct(fall)}}}")
                L.append(f"\\newcommand{{\\fj{t}{kt}Tie}}{{{pct(tie)}}}")
                L.append(f"\\newcommand{{\\fj{t}{kt}N}}{{{n}}}")
                L.append(f"\\newcommand{{\\fj{t}{kt}Counts}}{{{rise}/{fall}/{tie}}}")
        open(args.macros, "w").write("\n".join(L) + "\n")
        print(f"wrote {args.macros}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("absolute", "pairwise", "rubric"):
        s = sub.add_parser(name)
        s.add_argument("--model", default="gpt-6-sol")
        s.add_argument("--seeds", type=int, default=3)
        s.add_argument("--workers", type=int, default=12)
        if name == "pairwise":
            s.add_argument("--verdicts", action="store_true")
    e = sub.add_parser("eval")
    e.add_argument("--macros", default=None)
    args = ap.parse_args()
    return {"absolute": cmd_absolute, "pairwise": cmd_pairwise, "rubric": cmd_rubric,
            "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
