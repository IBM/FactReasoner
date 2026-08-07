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

"""Score post-filters of an already-mined arm against gold, with ZERO LLM calls.

A candidate-pair or output-vocabulary change is a *post-filter*: it only ever
removes relations the miner already produced. So re-filtering the records of a
superset arm (`all_pairs` visits every ordered pair) and re-scoring is not an
estimate of what a narrower policy would do -- it is an exact ablation, on the
same model, the same prompt and the same samples.

That makes prompt-free levers measurable in seconds instead of one live sweep
each, which matters because a live windowed arm is ~264 LLM calls.

**Validity boundary.** This is exact ONLY for post-filters. It cannot simulate a
prompt change: a different prompt sees a different pair set and answers
differently, so nothing here predicts that. Variants below are all pure filters.

Scoring imports `_pair_key` and `_prf` from `locoeval.mined_graph` rather than
reimplementing them, so an ablation number and a live report number can never
diverge through drift. `--variants baseline` is the self-check: it must reproduce
the stored `mining[arm]["coupling"]` block digit-for-digit.

Run::

    python scripts/ablate_mined_records.py \\
        --records-dir results/locobench_claude_5_mined_lcs/records \\
        --data-dir data/locobench-claude-5-test
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fact_reasoner.lcs.taxonomy import LEVEL1_NONE  # noqa: E402
from fact_reasoner.locoeval.mined_graph import (  # noqa: E402
    _atom_index,
    _pair_key,
    _prf,
)

# The (sense, coupling) combinations the corpus actually uses. Gold's mapping is a
# bijection: 9 senses, 9 couplings, no cross-talk. The miner emits `Instantiation`
# and `Condition`, which appear ZERO times in gold, so admitting only these pairs
# is free precision.
GOLD9: set[tuple[str, str]] = {
    ("Evidence", "entailment"),
    ("Cause-Effect", "entailment"),
    ("Effect-Cause", "entailment"),
    ("Restatement", "equivalence"),
    ("Contrast", "contradiction"),
    ("Concession", "contradiction"),
    ("Alternative", "exclusive"),
    ("Disjunction", "co_necessity"),
    ("Precedence", "none"),
}


def load_items(data_dir: str) -> dict[str, dict[str, Any]]:
    path = os.path.join(data_dir, "items.jsonl")
    with open(path) as f:
        return {json.loads(line)["id"]: json.loads(line) for line in f}


def gold_keys(item: dict[str, Any]) -> set[tuple[Any, ...]]:
    """Coupling-level keys of the item's edge-producing gold relations."""
    return {
        (
            *_pair_key(
                rel["source_id"], rel["target_id"], bool(rel.get("directed", True))
            ),
            rel["level1_coupling"],
        )
        for rel in item.get("relations", [])
        if rel.get("level1_coupling") != LEVEL1_NONE
    }


def apply_filters(
    relations: list[dict[str, Any]],
    *,
    max_distance: int | None,
    legal_only: bool,
    dedup: bool,
    degree_cap: int | None,
) -> list[dict[str, Any]]:
    """Post-filter one cell's mined relations."""
    kept = []
    for rel in relations:
        if max_distance is not None:
            dist = abs(_atom_index(rel["target"]) - _atom_index(rel["source"]))
            if dist > max_distance:
                continue
        if legal_only and (rel["sense"], rel["type"]) not in GOLD9:
            continue
        kept.append(rel)

    if dedup:
        # One edge per unordered pair, keeping the most confident.
        best: dict[tuple[str, str], dict[str, Any]] = {}
        for rel in sorted(kept, key=lambda r: -float(r.get("probability") or 0.0)):
            key = tuple(sorted((rel["source"], rel["target"]), key=_atom_index))
            best.setdefault(key, rel)  # type: ignore[arg-type]
        kept = list(best.values())

    if degree_cap is not None:
        # Greedy by descending probability. Measured HARMFUL at every cap -- kept
        # only so the report can show the measurement rather than assert it.
        deg: Counter = Counter()
        out = []
        for rel in sorted(kept, key=lambda r: -float(r.get("probability") or 0.0)):
            if deg[rel["source"]] < degree_cap and deg[rel["target"]] < degree_cap:
                out.append(rel)
                deg[rel["source"]] += 1
                deg[rel["target"]] += 1
        kept = out
    return kept


VARIANTS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "legal": {"legal_only": True},
    "d1": {"max_distance": 1},
    "d2": {"max_distance": 2},
    "d3": {"max_distance": 3},
    "d1+legal": {"max_distance": 1, "legal_only": True},
    "d2+legal": {"max_distance": 2, "legal_only": True},
    "d3+legal": {"max_distance": 3, "legal_only": True},
    "d1+legal+dedup": {"max_distance": 1, "legal_only": True, "dedup": True},
    "d1+legal+degcap2": {"max_distance": 1, "legal_only": True, "degree_cap": 2},
    "d1+legal+degcap3": {"max_distance": 1, "legal_only": True, "degree_cap": 3},
}


def score(
    records_dir: str,
    items: dict[str, dict[str, Any]],
    policy: str,
    **filters: Any,
) -> dict[str, Any]:
    """Coupling-level P/R/F1 of one filtered variant of one source arm."""
    opts = {
        "max_distance": filters.get("max_distance"),
        "legal_only": bool(filters.get("legal_only")),
        "dedup": bool(filters.get("dedup")),
        "degree_cap": filters.get("degree_cap"),
    }
    tp = fp = fn = 0
    n_kept = n_cells = 0
    illegal = 0
    for path in sorted(glob.glob(os.path.join(records_dir, f"*{policy}.json"))):
        with open(path) as f:
            rec = json.load(f)
        if "error" in rec or not rec.get("relations"):
            continue
        item = items.get(rec["item_id"])
        if item is None:
            continue
        n_cells += 1
        illegal += sum(
            1 for r in rec["relations"] if (r["sense"], r["type"]) not in GOLD9
        )
        kept = apply_filters(rec["relations"], **opts)
        n_kept += len(kept)
        keys = {
            (*_pair_key(r["source"], r["target"], r["directed"]), r["type"])
            for r in kept
        }
        gold = gold_keys(item)
        tp += len(gold & keys)
        fp += len(keys - gold)
        fn += len(gold - keys)
    out = _prf(tp, fp, fn)
    out.update(cells=n_cells, edges=n_kept, illegal_in_source=illegal)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--records-dir", default="results/locobench_claude_5_mined_lcs/records"
    )
    ap.add_argument("--data-dir", default="data/locobench-claude-5-test")
    ap.add_argument(
        "--source-policy",
        default="all_pairs",
        help="Arm whose records are re-filtered. Must be a SUPERSET of the pool "
        "each variant simulates, else the ablation understates recall "
        "(default: all_pairs, which visits every ordered pair).",
    )
    ap.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help="Comma-separated variant names, or 'all'.",
    )
    ap.add_argument(
        "--compare-arms",
        default="windowed,gated,all_pairs",
        help="Stored arms to print unfiltered, as the self-check baseline.",
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = ap.parse_args(argv)

    items = load_items(args.data_dir)
    rows: list[tuple[str, dict[str, Any]]] = []

    # Self-check: the stored arms, unfiltered. These MUST match the `mining` block
    # of results.json, or every number below is void.
    for arm in [a.strip() for a in args.compare_arms.split(",") if a.strip()]:
        rows.append((f"[stored] {arm}", score(args.records_dir, items, arm)))

    names = list(VARIANTS) if args.variants == "all" else [
        v.strip() for v in args.variants.split(",") if v.strip()
    ]
    for name in names:
        if name not in VARIANTS:
            ap.error(f"Unknown variant {name!r} (have: {sorted(VARIANTS)}).")
        rows.append(
            (
                f"{args.source_policy}:{name}",
                score(args.records_dir, items, args.source_policy, **VARIANTS[name]),
            )
        )

    if args.json:
        print(json.dumps({n: r for n, r in rows}, indent=2))
        return 0

    def fmt(x: Any) -> str:
        return "  --  " if x is None else f"{float(x):.3f}"

    print(
        f"{'variant':34s} {'edges':>6s} {'tp':>4s} {'fp':>4s} {'fn':>4s} "
        f"{'P':>7s} {'R':>7s} {'F1':>7s}"
    )
    print("-" * 80)
    for name, r in rows:
        print(
            f"{name:34s} {r['edges']:6d} {r['tp']:4d} {r['fp']:4d} {r['fn']:4d} "
            f"{fmt(r['precision'])} {fmt(r['recall'])} {fmt(r['f1'])}"
        )
    print()
    print(
        "Coupling level (the level that builds the MRF factor). Exact for "
        "post-filters only; a prompt change cannot be simulated here."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
