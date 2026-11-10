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

"""Summarize the graph-mining experiment against the pairwise RITS baselines.

Reads every `results.json` under the graph run's `arms/` folders plus the earlier
pairwise run, and prints one table: edge-level agreement, ladder pass rates per
readout, gap closure against the best pairwise arm, and the gateway spend actually
recorded in the raw-sample cache (fresh tokens only, priced per model).

    python scripts/summarize_locobench_graph.py [--json out.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

ROOT = "results/locobench_claude_5_graph"
BASELINE = "results/locobench_claude_5_v3_mined/results.json"
READOUTS = ("mean_marginal", "consistency", "log_partition")

# $/token on the IBM LiteLLM gateway (read from /model/info, 2026-11-06).
PRICES = {
    "gpt-6.1-sol": (2e-06, 1e-05),
    "claude-opus-5-5": (3.04e-06, 1.52e-05),
    "gemini-3.1-pro": (2e-06, 1.2e-05),
}

# Arms whose numbers are reference-only: the model sat on the V1 validation
# committee that admitted these items, so it is graded on its own filter.
SELECTION_BIASED = ("gemini-3.1-pro",)


def ladder_rates(results: dict) -> dict[str, dict[str, tuple[int, int]]]:
    agg: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for fam in results.get("families", []):
        for arm, block in (fam.get("arms") or {}).items():
            for c in block.get("checks", []):
                cell = agg[arm][c["readout"]]
                cell[0] += bool(c["passed"])
                cell[1] += 1
    return {a: {r: tuple(v) for r, v in d.items()} for a, d in agg.items()}


def cost_from_cache(root: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for path in glob.glob(os.path.join(root, "samples", "*", "*", "*.json")):
        model, mode = path.split(os.sep)[-3], path.split(os.sep)[-2]
        with open(path) as f:
            samples = json.load(f).get("samples", [])
        tot = Counter()
        for s in samples:
            tot.update(s.get("usage") or {})
            tot["samples"] += 1
        slot = out.setdefault(f"{model}/{mode}", Counter())
        slot.update(tot)
    priced = {}
    for key, tot in sorted(out.items()):
        model = key.split("/")[0]
        pin, pout = PRICES.get(model, (0.0, 0.0))
        priced[key] = {
            **dict(tot),
            "usd": tot["prompt_tokens"] * pin + tot["completion_tokens"] * pout,
        }
    return priced


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows: dict[str, dict] = {}
    sources = [args.baseline] + sorted(
        glob.glob(os.path.join(args.root, "arms", "*", "results.json"))
    )
    for src in sources:
        if not os.path.exists(src):
            continue
        with open(src) as f:
            res = json.load(f)
        rates = ladder_rates(res)
        n_failed = Counter(r["arm"] for r in res.get("records", []) if "error" in r)
        for arm, rr in rates.items():
            if arm in rows and arm.startswith("gold"):
                continue
            m = (res.get("mining") or {}).get(arm, {})
            rows[arm] = {
                "source": src,
                "rates": rr,
                "failed_cells": n_failed.get(arm, 0),
                "pair": m.get("pair"),
                "coupling": m.get("coupling"),
                "sense": m.get("sense"),
                "backward_recall": ((m.get("recall_by_direction") or {}).get("backward") or {}).get("recall"),
                "gate_recall": ((m.get("recall_by_window_admission") or {}).get("gate") or {}).get("recall"),
                "nonrel_viol": m.get("non_relation_violation_rate"),
                "invalid_recall": ((m.get("recall_by_validity") or {}).get("invalid") or {}).get("recall"),
            }

    def rate(arm, ro):
        p, n = rows[arm]["rates"].get(ro, (0, 0))
        return p / n if n else float("nan")

    pairwise = [a for a in rows if a.startswith("mined:") and ":graph" not in a and ":verify" not in a]
    best_pw = {ro: max((rate(a, ro) for a in pairwise), default=float("nan")) for ro in READOUTS}
    gold = {ro: rate("gold", ro) for ro in READOUTS} if "gold" in rows else {}

    f = lambda x: "  -  " if x is None else f"{x:.3f}"  # noqa: E731
    hdr = (f"{'arm':52s} {'pairF1':>6s} {'P':>5s} {'R':>5s} {'cplF1':>6s} {'snsF1':>6s} "
           f"{'bwdR':>5s} {'gateR':>5s} {'nrViol':>6s} {'invR':>5s} | "
           + " ".join(f"{ro[:8]:>8s}" for ro in READOUTS) + " | gap(mm) fail")
    print(hdr)
    print("-" * len(hdr))
    for arm in sorted(rows, key=lambda a: (not a.startswith("gold"), ":graph" in a or ":verify" in a, a)):
        r = rows[arm]
        pr = r["pair"] or {}
        tag = " *sel-bias" if any(b in arm for b in SELECTION_BIASED) else ""
        gap = ""
        if gold and arm.startswith("mined:"):
            den = gold["mean_marginal"] - best_pw["mean_marginal"]
            gap = f"{(rate(arm, 'mean_marginal') - best_pw['mean_marginal']) / den:+.2f}" if den else ""
        print(
            f"{(arm + tag)[:52]:52s} {f((r['pair'] or {}).get('f1')):>6s} {f(pr.get('precision')):>5s} "
            f"{f(pr.get('recall')):>5s} {f((r['coupling'] or {}).get('f1')):>6s} "
            f"{f((r['sense'] or {}).get('f1')):>6s} {f(r['backward_recall']):>5s} "
            f"{f(r['gate_recall']):>5s} {f(r['nonrel_viol']):>6s} {f(r['invalid_recall']):>5s} | "
            + " ".join(f"{rate(arm, ro):8.3f}" for ro in READOUTS)
            + f" | {gap:>7s} {r['failed_cells']:>4d}"
        )

    print("\nGateway spend recorded in the sample cache (all samples drawn):")
    costs = cost_from_cache(args.root)
    total = 0.0
    for key, c in costs.items():
        total += c["usd"]
        print(
            f"  {key:28s} samples={int(c.get('samples', 0)):4d} in={int(c.get('prompt_tokens', 0)):>9,d} "
            f"out={int(c.get('completion_tokens', 0)):>9,d} (reasoning {int(c.get('reasoning_tokens', 0)):,d}) "
            f"${c['usd']:.2f}"
        )
    print(f"  {'TOTAL':28s} ${total:.2f}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"rows": rows, "costs": costs, "best_pairwise": best_pw, "gold": gold}, fh, indent=2)


if __name__ == "__main__":
    main()
