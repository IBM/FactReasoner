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

"""Assemble a repaired LoCoBench dataset from several `locobench-repair` outputs.

Each source contributes the families it ADMITTED by repair or regeneration; a later
source overrides an earlier one for the same family. A family admitted by no source is
reported and the merge refuses to write, so a v2 can never silently carry an unrepaired
family mixed with repaired ones.

    python scripts/merge_locobench_repair.py --out data/locobench-claude-5-test-v2 \\
        --base data/locobench-claude-5-test \\
        data/locobench-claude-5-test-v2-p2a \\
        data/locobench-claude-5-test-v2-pilot3 data/locobench-claude-5-test-v2-pilot4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from fact_reasoner.locobench.schema import validate_item, validate_manifest_entry  # noqa: E402
from fact_reasoner.locoeval.gold_graph import build_gold_result  # noqa: E402


def _jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", required=True, help="The unrepaired source (for the family list).")
    ap.add_argument("--allow-missing", action="store_true",
                    help="Write anyway, copying unrepaired families from --base (marked).")
    ap.add_argument("--exclude-missing", action="store_true",
                    help="Write only the repaired families; list the rest in config.json.")
    ap.add_argument("sources", nargs="+", help="Repair outputs, lowest priority first.")
    args = ap.parse_args()

    base_state = json.load(open(os.path.join(args.base, "state.json")))["families"]
    base_items = _jsonl(os.path.join(args.base, "items.jsonl"))
    base_fams = {f["family_id"]: f for f in json.load(open(os.path.join(args.base, "families.json")))["families"]}

    chosen: dict[str, dict] = {}
    for src in args.sources:
        log = json.load(open(os.path.join(src, "repair_log.json")))
        st = json.load(open(os.path.join(src, "state.json")))["families"]
        items = _jsonl(os.path.join(src, "items.jsonl"))
        fams = {f["family_id"]: f for f in json.load(open(os.path.join(src, "families.json")))["families"]}
        for e in log:
            if e["action"] in ("repaired", "regenerated") and e.get("admitted"):
                fid = e["family_id"]
                chosen[fid] = {
                    "source": src,
                    "action": e["action"],
                    "items": [i for i in items if i["expected"]["family_id"] == fid],
                    "manifest": fams[fid],
                    "state": st[fid],
                }

    missing = sorted(set(base_state) - set(chosen))
    if missing and not (args.allow_missing or args.exclude_missing):
        print(f"[merge] {len(missing)} family(ies) admitted by no source: {missing}. "
              "Refusing to write a partially repaired dataset (pass --allow-missing).")
        return 1
    for fid in ([] if args.exclude_missing else missing):
        chosen[fid] = {
            "source": args.base, "action": "UNREPAIRED",
            "items": [i for i in base_items if i["expected"]["family_id"] == fid],
            "manifest": base_fams[fid], "state": base_state[fid],
        }

    os.makedirs(args.out, exist_ok=True)
    out_items, out_fams, out_state, provenance = [], [], {}, {}
    for fid in sorted(chosen):
        c = chosen[fid]
        assert len(c["items"]) == 5, f"{fid}: {len(c['items'])} items"
        for it in sorted(c["items"], key=lambda x: x["expected"]["rung_index"]):
            validate_item(it)
            build_gold_result(it)  # the gold MRF must build
            out_items.append(it)
        validate_manifest_entry(c["manifest"])
        out_fams.append(c["manifest"])
        out_state[fid] = c["state"]
        provenance[fid] = {"source": c["source"], "action": c["action"]}

    ids = [i["id"] for i in out_items]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    with open(os.path.join(args.out, "items.jsonl"), "w") as f:
        f.writelines(json.dumps(i) + "\n" for i in out_items)
    json.dump({"manifest_version": 3, "families": out_fams},
              open(os.path.join(args.out, "families.json"), "w"), indent=1)
    json.dump({"families": out_state}, open(os.path.join(args.out, "state.json"), "w"), indent=1)
    cfg = json.load(open(os.path.join(args.base, "config.json")))
    cfg["repaired_from"] = args.base
    cfg["repair_provenance"] = provenance
    if args.exclude_missing and missing:
        cfg["excluded_unrepaired"] = {
            fid: base_state[fid].get("family") for fid in missing
        }
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)
    print(f"[merge] wrote {len(out_items)} items / {len(out_fams)} families to {args.out}")
    for fid, p in provenance.items():
        print(f"  {fid}: {p['action']:12s} <- {p['source']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
