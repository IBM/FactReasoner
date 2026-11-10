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

"""Repair an admitted LoCoBench dataset's perturbed rungs, keeping every base rung.

A blind side-by-side reading found the planted rung changes realized in 5/39 dev and
132/268 release transitions. This command re-runs ONLY the perturb stage of each selected
family under the gated operators (`locobench.realize`), resuming from the family's stored
question, claims, plan and base response -- so item ids, atoms, base text and base gold
labels are unchanged (checked: all 14 dev families rebuild byte-identical atoms and base
relations). The base is trusted, not re-audited (see `generate_family(trust_base=...)`).

Families not selected are copied through unchanged. Families whose plan cannot support
their ladder under the current targeting rule (the dev CHAIN families predate the
2026-10-31 valid-entailment-chain fix) must be regenerated whole with `--regenerate`,
which runs the full pipeline from P1 on the same topic and family id.

    locobench-repair --src data/locobench-claude-5-test \\
        --out data/locobench-claude-5-test-v2 --families f001,f005 \\
        --regenerate f005 --config configs/locobench_claude.json

Run with `--dry-run` for an offline smoke test on the mock.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from typing import Any

from fact_reasoner.env import load_dotenv
from fact_reasoner.locobench.cli import _build_auditors, _build_generators
from fact_reasoner.locobench.config import GenConfig
from fact_reasoner.locobench.pipeline import generate_family, make_mock_llm
from fact_reasoner.locobench.store import FamilyState, Store


def _csv(v: str) -> list[str]:
    return [t.strip() for t in (v or "").split(",") if t.strip()]


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="locobench-repair", description=__doc__.split("\n")[0])
    p.add_argument("--src", required=True, help="Admitted dataset to repair.")
    p.add_argument("--out", required=True, help="New dataset directory (must not exist).")
    p.add_argument("--config", required=True, help="Generation config (generator, committee).")
    p.add_argument("--families", type=_csv, default=None,
                   help="Families to repair (default: every family).")
    p.add_argument("--regenerate", type=_csv, default=[],
                   help="Families to regenerate WHOLE from P1, same topic and id.")
    p.add_argument("--dry-run", action="store_true", help="Run on the offline mock.")
    p.add_argument("--resume", action="store_true",
                   help="Reuse families already written to --out.")
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if os.path.exists(args.out) and not args.resume:
        raise SystemExit(f"[repair] {args.out} exists; pass --resume or choose a new path.")

    src_state = json.load(open(os.path.join(args.src, "state.json")))["families"]
    src_fams = {
        f["family_id"]: f
        for f in json.load(open(os.path.join(args.src, "families.json")))["families"]
    }
    src_items = _load_jsonl(os.path.join(args.src, "items.jsonl"))
    src_cfg = json.load(open(os.path.join(args.src, "config.json")))

    with open(args.config) as f:
        cfg = GenConfig.from_dict(json.load(f))
    # Keep the source's dataset name: item ids embed it, and they must stay stable.
    cfg = cfg.with_overrides(dataset_name=src_cfg["dataset_name"], out_dir=args.out)
    if args.dry_run:
        cfg = cfg.with_overrides(dry_run=True)
    cfg.validate()

    selected = args.families or sorted(src_state)
    unknown = sorted(set(selected) | set(args.regenerate))
    unknown = [f for f in unknown if f not in src_state]
    if unknown:
        raise SystemExit(f"[repair] unknown family id(s): {unknown}")
    regenerate = set(args.regenerate)
    selected = sorted(set(selected) | regenerate)

    store = Store(args.out)
    out_cfg = dict(src_cfg)
    out_cfg["repaired_from"] = args.src
    out_cfg["repair"] = {
        "families": selected,
        "regenerated": sorted(regenerate),
        "gates": ["P5.spans", "V5", "V5.screen"],
        "committee": [m.to_dict() for m in cfg.committee],
        "generators": [m.to_dict() for m in cfg.generators],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    store.save_config(out_cfg)

    if cfg.dry_run:
        # `llm=None` lets `generate_family` build its own mock wired to its plan holder,
        # which the mock P5 needs to resolve edge targets; a pre-built mock has a separate
        # holder and edits nothing. Auditors are mocks too, so the gated path runs.
        llms = {(cfg.generators[0].name if cfg.generators else "mock"): None}
        shared: dict[str, Any] = {}
        auditors = {
            name: [(f"mock-{i}", make_mock_llm(cfg, plan_holder=shared)) for i in range(3)]
            for name in llms
        }
    else:
        llms = _build_generators(cfg)
        auditors = _build_auditors(cfg)
    gen_name = next(iter(llms))

    done = {fs for fs in store.state if store.state[fs].stage == "admitted"} if args.resume else set()
    log: list[dict[str, Any]] = []
    for fid in sorted(src_state):
        e = src_state[fid]
        if fid in done:
            continue
        if fid not in selected:
            # Copy through unchanged: same items, same manifest entry, same state.
            items = [i for i in src_items if (i.get("expected") or {}).get("family_id") == fid]
            store.append_items(items)
            store.put_manifest(src_fams[fid])
            store.put(FamilyState.from_dict(e))
            store.flush_state()
            log.append({"family_id": fid, "action": "copied"})
            continue

        t0 = time.perf_counter()
        resume = None if fid in regenerate else dict(e["artifacts"])
        res = generate_family(
            fid,
            e["canonical_topic"],
            e["family"],
            cfg,
            llm=llms[gen_name],
            generator=gen_name,
            resume_from=resume,
            auditor_llms=auditors.get(gen_name),
            trust_base=resume is not None,
            plan_holder=shared if cfg.dry_run else None,
        )
        fs = FamilyState(
            family_id=fid,
            canonical_topic=e["canonical_topic"],
            family=e["family"],
            stage=res.stage,
            attempts=1,
            rejected_reason="" if res.admitted else res.verdict.reason(),
            item_ids=[i["id"] for i in res.items],
            artifacts=res.artifacts,
        )
        if res.admitted:
            store.append_items(res.items)
            store.put_manifest(res.manifest)
        else:
            store.reject(fid, res.verdict.to_dict(), stage=res.stage)
        store.put(fs)
        store.flush_state()
        failed = [g for g in res.verdict.results if not g.passed]
        entry = {
            "family_id": fid,
            "action": "regenerated" if fid in regenerate else "repaired",
            "admitted": res.admitted,
            "stage": res.stage,
            "calls": res.calls,
            "gate_failures_retried": len(failed) if res.admitted else None,
            "reason": None if res.admitted else res.verdict.reason(),
            "screen": next((g.observed for g in res.verdict.results if g.gate == "V5.screen"), None),
            "elapsed_s": round(time.perf_counter() - t0, 1),
        }
        log.append(entry)
        print(f"[repair] {fid} {entry['action']}: "
              f"{'ADMITTED' if res.admitted else 'REJECTED at ' + res.stage} "
              f"calls={res.calls} ({entry['elapsed_s']}s)", flush=True)
        if not res.admitted:
            print(f"         reason: {entry['reason'][:400]}", flush=True)
        with open(os.path.join(args.out, "repair_log.json"), "w") as f:
            json.dump(log, f, indent=1)

    n_ok = sum(1 for x in log if x.get("admitted"))
    n_try = sum(1 for x in log if x["action"] != "copied")
    print(f"[repair] {n_ok}/{n_try} families admitted; log at {args.out}/repair_log.json")
    return 0 if n_ok == n_try else 1


if __name__ == "__main__":
    raise SystemExit(main())
