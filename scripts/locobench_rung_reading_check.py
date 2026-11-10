#!/usr/bin/env python
# coding=utf-8
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

"""Rung reading check: does the perturbed prose actually realize the planted change?

The gold-swap diagnostic showed the graph miners reproduce the full gold pattern on
only 1-2 of the 39 ladder-bearing edges. Two explanations predict that: the miner
misses the perturbation, or the perturbed text does not realize it (V1 only ever
checked the BASE rung). This script separates them.

For every ladder-bearing edge, at the one rung boundary where gold changes it, a
judge reads the two adjacent rung texts SIDE BY SIDE -- blind to the planted change
-- and reports, for the edge's two claims, how each text relates them and whether a
conflict is presented as resolved. Each item is judged in both orders (A/B, B/A) by
each judge, so position bias shows up as order disagreement.

The verdict per transition compares the judge's reading to gold's change:
  realized   the judge sees the planted change in the planted direction
  absent     the judge sees no difference on this edge
  other      the judge sees a different change
Then, against the miner, per transition: did the miner's graphs on the two rungs
show the change? That yields the 2x2 (text realizes x miner follows) that decides
between "miner misses it" and "text doesn't carry it".

    python scripts/locobench_rung_reading_check.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from locobench_gold_swap import gold_edges, mined_edges, upair  # noqa: E402

from fact_reasoner.env import load_dotenv  # noqa: E402
from fact_reasoner.locoeval.graph_miner import ChatClient  # noqa: E402
from fact_reasoner.locoeval.models import load_model_specs  # noqa: E402

JUDGES = ("gpt-6.1-sol", "claude-opus-5-5")
MINERS = {
    "claude-opus-5-5": ("opus55", "mined:claude-opus-5-5:graph:k5"),
    "gpt-6.1-sol": ("gpt61", "mined:gpt-6.1-sol:graph:k5"),
}
PRICES = {
    "gpt-6.1-sol": (2e-06, 1e-05),
    "claude-opus-5-5": (3.04e-06, 1.52e-05),
    "claude-opus-5": (3.8e-06, 1.9e-05),
}

PROMPT = """You are a careful reader comparing two versions of the same expert text.
The versions are nearly identical but may differ in a few sentences.

Focus ONLY on the two claims below and on what each TEXT itself says about how
they relate. Do not use outside knowledge about whether the claims are true.

CLAIM X: {x}
CLAIM Y: {y}

For EACH text, answer:
  "relation": how the text relates X and Y, one of
     "conflict"   - the text presents X and Y as in tension / incompatible /
                    competing (contrast, concession, mutually exclusive alternatives)
     "support"    - the text presents one as supporting, causing, implying or
                    exemplifying the other
     "equivalent" - the text presents them as restatements of each other
     "either"     - the text says at least one of them holds
     "none"       - the text draws no connection between X and Y (they may both
                    appear, but the text does not link them)
  "resolved": only when relation is "conflict": true if the text explicitly
     dissolves or adjudicates the tension (e.g. explains that both hold under
     different conditions, or that the opposition is only apparent); false if the
     tension is left standing; null otherwise.
  "evidence": the shortest verbatim quote from that text that carries the answer
     (empty string for "none").

Then say whether the two texts differ in how they treat X and Y:
  "differs": true or false
  "change": one short sentence describing the difference (or "").

Return ONLY a JSON object:
{{"text_1": {{"relation": ..., "resolved": ..., "evidence": ...}},
  "text_2": {{"relation": ..., "resolved": ..., "evidence": ...}},
  "differs": ..., "change": ...}}

TEXT 1:
{t1}

TEXT 2:
{t2}
"""

_REL_OF_COUPLING = {
    "contradiction": "conflict",
    "exclusive": "conflict",
    "entailment": "support",
    "equivalence": "equivalent",
    "co_necessity": "either",
}


def sig(e):
    return None if e is None else (e["coupling"], bool(e["resolved"]))


def gold_reading(e):
    """What a faithful reader should report for a gold edge state."""
    if e is None:
        return ("none", None)
    rel = _REL_OF_COUPLING[e[0]]
    return (rel, e[1] if rel == "conflict" else None)


def transitions(items):
    """Every (edge, rung boundary) where gold changes the edge's signature."""
    fam = defaultdict(dict)
    for it in items:
        fam[it["expected"]["family_id"]][it["expected"]["rung_index"]] = it
    out = []
    for fid, rungs in sorted(fam.items()):
        G = {r: gold_edges(it) for r, it in rungs.items()}
        for k in sorted(set().union(*G.values())):
            order = sorted(rungs)
            gs = [sig(G[r].get(k)) for r in order]
            if len(set(gs)) < 2:
                continue
            for i in range(len(order) - 1):
                a, b = gs[i], gs[i + 1]
                if a == b:
                    continue
                if a and not b:
                    kind = "spurious_removed" if order[i] == 0 else "dropped"
                elif b and not a:
                    kind = "added"
                else:
                    kind = "resolved"
                out.append(
                    {
                        "family_id": fid,
                        "family": rungs[order[0]]["expected"]["family"],
                        "pair": list(k),
                        "rungs": [order[i], order[i + 1]],
                        "items": [rungs[order[i]]["id"], rungs[order[i + 1]]["id"]],
                        "gold": [a, b],
                        "kind": kind,
                    }
                )
    return out


def parse_json(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def norm_reading(t):
    rel = str((t or {}).get("relation") or "none").lower()
    res = (t or {}).get("resolved")
    return (rel, bool(res) if rel == "conflict" and res is not None else None)


async def judge_all(trans, items_by_id, specs, cache_path, judges=JUDGES):
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    clients = {j: ChatClient(specs[j]) for j in judges}
    sem = asyncio.Semaphore(8)

    async def one(t, judge, flip):
        key = f"{judge}|{t['items'][0]}|{t['items'][1]}|{'-'.join(t['pair'])}|{int(flip)}"
        if key in cache and cache[key].get("parsed"):
            return
        lo, hi = (items_by_id[i] for i in t["items"])
        atoms = {a["id"]: a["text"] for a in lo["atoms"]}
        t1, t2 = (hi, lo) if flip else (lo, hi)
        prompt = PROMPT.format(
            x=atoms[t["pair"][0]], y=atoms[t["pair"][1]],
            t1=t1["response"], t2=t2["response"],
        )
        async with sem:
            for _ in range(3):
                try:
                    out = await clients[judge].complete(prompt)
                except Exception as e:  # noqa: BLE001
                    out = {"text": "", "usage": {}, "error": repr(e)}
                    await asyncio.sleep(5)
                    continue
                parsed = parse_json(out["text"])
                if parsed:
                    break
        cache[key] = {"parsed": parsed, "usage": out.get("usage"), "raw": out.get("text")}

    jobs = [one(t, j, flip) for t in trans for j in judges for flip in (False, True)]
    await asyncio.gather(*jobs)
    for c in clients.values():
        await c.aclose()
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=1)
    return cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/locobench-claude-5-test")
    ap.add_argument("--root", default="results/locobench_claude_5_graph")
    ap.add_argument("--out-dir", default=None,
                    help="Where the cache and reading_check.json go (default: <root>/diag).")
    ap.add_argument("--judges", default=",".join(JUDGES),
                    help="Comma-separated judge models. Exclude the dataset's generator.")
    ap.add_argument("--no-miners", action="store_true",
                    help="Skip the miner comparison (no graph-arm records for this dataset).")
    args = ap.parse_args()
    judges = tuple(j.strip() for j in args.judges.split(",") if j.strip())
    gen = {g["model_id"] for g in json.load(open(os.path.join(args.data_dir, "config.json"))).get("generators", [])}
    specs = load_model_specs()
    clash = [j for j in judges if specs[j].model_id in gen]
    if clash:
        ap.error(f"judge(s) {clash} generated this dataset; a generator may not judge its own prose.")
    load_dotenv()
    os.environ.setdefault("OPENAI_API_KEY", os.environ.get("ANTHROPIC_AUTH_TOKEN", ""))

    items = [json.loads(line) for line in open(os.path.join(args.data_dir, "items.jsonl"))]
    by_id = {it["id"]: it for it in items}
    trans = transitions(items)
    out_dir = args.out_dir or os.path.join(args.root, "diag")
    os.makedirs(out_dir, exist_ok=True)
    cache = asyncio.run(
        judge_all(trans, by_id, specs, os.path.join(out_dir, "reading_cache.json"), judges)
    )

    # Per transition, per judge: does each order's reading realize gold's change?
    rows = []
    for t in trans:
        want = [gold_reading(t["gold"][0]), gold_reading(t["gold"][1])]
        row = dict(t)
        row["want"] = want
        row["judges"] = {}
        for j in judges:
            verdicts = []
            for flip in (False, True):
                key = f"{j}|{t['items'][0]}|{t['items'][1]}|{'-'.join(t['pair'])}|{int(flip)}"
                p = (cache.get(key) or {}).get("parsed")
                if not p:
                    verdicts.append("unparsed")
                    continue
                r1, r2 = norm_reading(p.get("text_1")), norm_reading(p.get("text_2"))
                lo_r, hi_r = (r2, r1) if flip else (r1, r2)
                if t["kind"] == "resolved":
                    got = (lo_r[1], hi_r[1])
                    exp = (want[0][1], want[1][1])
                    sees_change = lo_r[1] != hi_r[1]
                else:
                    has = lambda r: r[0] != "none"  # noqa: E731
                    got = (has(lo_r), has(hi_r))
                    exp = (want[0][0] != "none", want[1][0] != "none")
                    sees_change = got[0] != got[1]
                if got == exp:
                    verdicts.append("realized")
                elif not sees_change:
                    verdicts.append("absent")
                else:
                    verdicts.append("other")
            row["judges"][j] = verdicts
        # Did each miner follow the change?
        row["miners"] = {}
        for m, (d, arm) in ({} if args.no_miners else MINERS).items():
            slug = arm
            for ch in (":", "/", ".", " "):
                slug = slug.replace(ch, "_")
            ms = []
            for iid in t["items"]:
                with open(os.path.join(args.root, "arms", d, "records", f"{iid}__{slug}.json")) as f:
                    me = mined_edges(json.load(f))
                ms.append(sig(me.get(upair(*t["pair"]))))
            if t["kind"] == "resolved":
                ok = ms[0] is not None and ms[1] is not None and ms[0][1] == t["gold"][0][1] and ms[1][1] == t["gold"][1][1]
            else:
                ok = (ms[0] is not None) == (t["gold"][0] is not None) and (ms[1] is not None) == (t["gold"][1] is not None)
            row["miners"][m] = {"follows": ok, "mined": ms}
        rows.append(row)

    with open(os.path.join(out_dir, "reading_check.json"), "w") as f:
        json.dump(rows, f, indent=1)

    # Consensus "text realizes": a strict majority of the 4 readings (2 judges x 2
    # orders) say realized. Also per judge, both orders agreeing.
    def realized_by(row, judges):
        v = [x for j in judges for x in row["judges"][j]]
        return sum(x == "realized" for x in v) * 2 > len(v)

    print(f"{len(rows)} transitions; kinds: {dict(Counter(r['kind'] for r in rows))}\n")
    print("Text realizes the planted change (majority of 2 orders):")
    print(f"  {'kind':18s} {'n':>3s} " + " ".join(f"{j:>16s}" for j in judges) + f" {'both judges':>12s} {'order-agree':>12s}")
    for kind in ("dropped", "spurious_removed", "resolved", "added", "ALL"):
        sub = [r for r in rows if kind == "ALL" or r["kind"] == kind]
        if not sub:
            continue
        per = [sum(realized_by(r, [j]) for r in sub) for j in judges]
        both = sum(realized_by(r, list(judges)) for r in sub)
        agree = sum(r["judges"][j][0] == r["judges"][j][1] for r in sub for j in judges)
        print(f"  {kind:18s} {len(sub):3d} " + " ".join(f"{p:>10d}/{len(sub):<5d}" for p in per)
              + f" {both:>6d}/{len(sub):<5d} {agree:>6d}/{2*len(sub):<5d}")
    print("\nVerdict mix (all 4 readings per transition):")
    for kind in ("dropped", "spurious_removed", "resolved", "added"):
        c = Counter(x for r in rows if r["kind"] == kind for j in judges for x in r["judges"][j])
        print(f"  {kind:18s} {dict(c)}")

    print("\nText realizes (consensus) x miner follows:")
    for m in ({} if args.no_miners else MINERS):
        tab = Counter((realized_by(r, list(judges)), r["miners"][m]["follows"]) for r in rows)
        print(f"  {m:18s} text-yes/miner-yes={tab[(True, True)]:2d}  text-yes/miner-no={tab[(True, False)]:2d}  "
              f"text-no/miner-yes={tab[(False, True)]:2d}  text-no/miner-no={tab[(False, False)]:2d}")
        for kind in ("dropped", "spurious_removed", "resolved", "added"):
            sub = [r for r in rows if r["kind"] == kind]
            t2 = Counter((realized_by(r, list(judges)), r["miners"][m]["follows"]) for r in sub)
            print(f"      {kind:16s} yy={t2[(True, True)]:2d} yn={t2[(True, False)]:2d} ny={t2[(False, True)]:2d} nn={t2[(False, False)]:2d}")

    cost = 0.0
    for key, v in cache.items():
        j = key.split("|")[0]
        u = v.get("usage") or {}
        pin, pout = PRICES.get(j, (0.0, 0.0))
        cost += (u.get("prompt_tokens") or 0) * pin + (u.get("completion_tokens") or 0) * pout
    print(f"\nJudge spend: ${cost:.2f} over {len(cache)} calls")


if __name__ == "__main__":
    main()
