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

"""Generate a slot3 LoCoBench corpus (3-rung ladders spliced from one template).

Run::

    # Offline, every stage, no credentials:
    python -m fact_reasoner.locobench.cli_slot3 --config configs/locobench_slot3_test.json \\
        --dry-run --out /tmp/slot3

    # Inspect the allocation and the committees, generating nothing (no calls):
    python -m fact_reasoner.locobench.cli_slot3 --config ... --check
    # ... and send one 5-token request to every configured model:
    python -m fact_reasoner.locobench.cli_slot3 --config ... --check --ping

    # Generate. Re-running resumes: admitted families are skipped, rejected ones retried.
    python -m fact_reasoner.locobench.cli_slot3 --config ... --track-spend --max-spend 900

    # Per-gate pass rates, per-generator admission and cost, from attempts.jsonl:
    python -m fact_reasoner.locobench.cli_slot3 --config ... --report

Families run in parallel threads (``--workers``, default the config's ``max_concurrency``).
Each thread builds its OWN generator and committee callables, per generator, since a
`build_llm` callable owns one event loop and must not be shared across threads; store writes
are serialized by a lock.

ALLOCATION. A single-generator run of at most 36 families (the pilot) keeps the original
allocation, so its directory still resumes. Any larger or multi-generator run uses
:func:`balanced_slots`: every (ladder type, generator) cell gets the same number of families,
and each topic rotates through the ladder types and the generators.

SPEND. ``--track-spend`` reads the gateway's running spend for the API key (LiteLLM's
``x-litellm-key-spend`` response header, from one 5-token request) before the run, every few
families and at the end, into ``spend.jsonl``. The figure is the key's total, so concurrent
use of the same key by other jobs is included in the deltas. ``--max-spend`` stops starting
new families once the run's delta exceeds it; families in flight finish.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import threading
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from fact_reasoner.locobench import slots as S
from fact_reasoner.locobench.cli import _interleave
from fact_reasoner.locobench.config import GenConfig, ModelRef, load_config
from fact_reasoner.locobench.pipeline import build_llm
from fact_reasoner.locobench.pipeline_slot3 import generate_family_slot3
from fact_reasoner.locobench.slots_mock import make_slot3_mock_llm
from fact_reasoner.locobench.store import FamilyState, Store
from fact_reasoner.locobench.topics import TOPICS, domain_of

#: The cheapest gateway model, used only to read the key's spend header.
SPEND_PROBE_MODEL = "aws/claude-haiku-5-5"
#: Probe the spend after every this many completed families.
SPEND_EVERY = 5


def _topics_by_domain() -> list[str]:
    """The 36 topics, round-robin over domains, so any prefix spans the domains."""
    by_dom: dict[str, list[str]] = collections.defaultdict(list)
    for t in sorted(TOPICS):
        by_dom[domain_of(t)].append(t)
    topics: list[str] = []
    while len(topics) < len(TOPICS):
        for d in sorted(by_dom):
            if by_dom[d]:
                topics.append(by_dom[d].pop(0))
    return topics


def family_slots(n: int) -> list[tuple[str, str, str]]:
    """The pilot's allocation: ``(family_id, topic, primary)``, types interleaved.

    Kept verbatim so the 16-family pilot resumes with the same slots. Beyond 36 families it
    gives every topic a single ladder type, which is why larger runs use
    :func:`balanced_slots`.
    """
    topics = _topics_by_domain()
    types = _interleave([S.PRIMARY_TYPES[i % len(S.PRIMARY_TYPES)] for i in range(n)])
    return [(f"f{i + 1:03d}", topics[i % len(topics)], types[i]) for i in range(n)]


def balanced_slots(n: int, generators: list[str]) -> list[tuple[str, str, str, str]]:
    """``(family_id, topic, primary, generator)`` for a balanced multi-generator corpus.

    Family ``i`` takes topic ``i mod 36`` and combination ``c = (t + s*k) mod L`` of the
    ``L = |types| * |generators|`` (type, generator) cells, where ``t`` is the topic index,
    ``k = i div 36`` the pass over the topics, and the shift ``s`` is chosen so that every
    cell gets the same number of families (within one), each topic sees as many types and
    generators as its passes allow, and no topic repeats a cell. For 240 families and three
    generators that is 20 per cell, every topic with all 4 types and all 3 generators.
    """
    import math

    topics = _topics_by_domain()
    nt, types, ng = len(topics), list(S.PRIMARY_TYPES), len(generators)
    L = len(types) * ng
    # With coprime counts (4 types, 3 generators) the cell index maps to (c mod 4, c mod 3),
    # a bijection by the Chinese remainder theorem, so CONSECUTIVE families rotate both the
    # type and the generator and any prefix of the run is balanced too (a run stopped early,
    # e.g. by the spend cap, is not skewed towards one generator).
    crt = math.gcd(len(types), ng) == 1

    def cell(c: int) -> tuple[str, str]:
        if crt:
            return types[c % len(types)], generators[c % ng]
        return types[c % len(types)], generators[c // len(types)]

    def build(shift: int):
        out = []
        for i in range(n):
            t, k = i % nt, i // nt
            p, g = cell((t + shift * k) % L)
            out.append((f"f{i + 1:03d}", topics[t], p, g))
        return out

    def quality(slots):
        cells = collections.Counter((p, g) for _f, _t, p, g in slots)
        spread = (max(cells.values()) - min(cells.values())) if len(cells) == L else n
        per_topic = collections.defaultdict(list)
        for _f, t, p, g in slots:
            per_topic[t].append((p, g))
        dups = sum(len(v) - len(set(v)) for v in per_topic.values())
        types_seen = sum(len({p for p, _ in v}) for v in per_topic.values())
        gens_seen = sum(len({g for _, g in v}) for v in per_topic.values())
        return (spread, dups, -types_seen, -gens_seen)

    best = min(range(1, L), key=lambda s: (quality(build(s)), s)) if L > 1 else 1
    return build(best)


def plan_slots(cfg: GenConfig) -> list[tuple[str, str, str, str]]:
    """The run's slots, ``(family_id, topic, primary, generator name)``."""
    names = [g.name for g in cfg.generators] or ["mock"]
    if len(names) == 1 and cfg.n_families <= len(TOPICS):
        return [(f, t, p, names[0]) for f, t, p in family_slots(cfg.n_families)]
    return balanced_slots(cfg.n_families, names)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="locobench-slot3", description=__doc__.split("\n")[0])
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--check", action="store_true",
                   help="Validate the config and print the allocation and committees; "
                        "generate nothing.")
    p.add_argument("--ping", action="store_true",
                   help="With --check: send one 5-token request to every configured model.")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--families", default=None, help="Comma-separated family ids to run.")
    p.add_argument("--family-attempts", type=int, default=2,
                   help="Whole-family attempts per run (question and claims are reused).")
    p.add_argument("--track-spend", action="store_true",
                   help="Record the gateway key's spend before, during and after the run.")
    p.add_argument("--max-spend", type=float, default=None,
                   help="USD: stop starting new families once the run has spent this much "
                        "(implies --track-spend).")
    return p


# ---------------------------------------------------------------------------- gateway


def _gateway(cfg: GenConfig) -> tuple[str, str] | None:
    """(base_url, token) of the LiteLLM gateway the models use, or None."""
    token = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    urls = {m.base_url for m in cfg.generators if m.base_url}
    if not token or len(urls) != 1:
        return None
    return urls.pop().rstrip("/"), token


def _chat(base: str, token: str, model: str, *, timeout: int = 90) -> tuple[int, dict, str]:
    """One 5-token chat request; (status, response headers, text or error)."""
    body = json.dumps({"model": model, "max_tokens": 5,
                       "messages": [{"role": "user", "content": "Say ok"}]}).encode()
    req = urllib.request.Request(f"{base}/chat/completions", data=body, method="POST",
                                 headers={"Authorization": f"Bearer {token}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read() or b"{}")
            text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content", "")
            return r.status, dict(r.headers), text or ""
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers or {}), e.read().decode(errors="replace")[:200]
    except Exception as e:  # noqa: BLE001 -- reported to the caller
        return 0, {}, f"{type(e).__name__}: {e}"


def key_spend(cfg: GenConfig) -> float | None:
    """The gateway key's running spend in USD, read off one tiny request's headers."""
    gw = _gateway(cfg)
    if gw is None:
        return None
    status, headers, _ = _chat(*gw, SPEND_PROBE_MODEL)
    low = {k.lower(): v for k, v in headers.items()}
    try:
        return float(low["x-litellm-key-spend"]) if status == 200 else None
    except (KeyError, ValueError):
        return None


# ---------------------------------------------------------------------------- run


class _Run:
    def __init__(self, cfg: GenConfig, store: Store, dry: bool, max_spend: float | None,
                 track: bool):
        self.cfg = cfg
        self.store = store
        self.dry = dry
        self.lock = threading.Lock()
        self.local = threading.local()
        self.attempts_path = os.path.join(store.out_dir, "attempts.jsonl")
        self.spend_path = os.path.join(store.out_dir, "spend.jsonl")
        self.gens = {g.name: g for g in cfg.generators}
        self.track = track or max_spend is not None
        self.max_spend = max_spend
        self.spend0: float | None = None
        self.completed = 0
        self.stop = threading.Event()

    def models(self, gen_name: str) -> tuple[Any, list[tuple[str, Any]]]:
        """This thread's (generator, committee) callables for one generator, built once."""
        cache = getattr(self.local, "models", None)
        if cache is None:
            cache = self.local.models = {}
        if gen_name not in cache:
            gen = self.gens[gen_name]
            panel = [(m.name, build_llm(m, self.cfg)) for m in self.cfg.eligible_auditors(gen)]
            cache[gen_name] = (build_llm(gen, self.cfg), panel)
        return cache[gen_name]

    def log(self, path: str, rec: dict[str, Any]) -> None:
        with self.lock:
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def snapshot(self, note: str) -> float | None:
        if not self.track or self.dry:
            return None
        spend = key_spend(self.cfg)
        self.log(self.spend_path, {"ts": time.time(), "note": note, "key_spend": spend,
                                   "completed": self.completed})
        return spend

    def after_family(self) -> None:
        with self.lock:
            self.completed += 1
            due = self.completed % SPEND_EVERY == 0
        if due:
            spend = self.snapshot("progress")
            if (self.max_spend is not None and spend is not None and self.spend0 is not None
                    and spend - self.spend0 > self.max_spend):
                self.stop.set()

    def family(self, index: int, fid: str, topic: str, primary: str, gen_name: str,
               attempts: int) -> str:
        if self.stop.is_set():
            return f"{fid} {topic} ({primary}, {gen_name}): not started -- spend cap reached"
        rare = S.rare_senses_for(index)
        prev = self.store.get(fid)
        resume = {}
        if prev and prev.artifacts:
            resume = {k: prev.artifacts[k] for k in ("question", "claims")
                      if prev.artifacts.get(k)}
        done = (prev.attempts if prev else 0)
        msg = ""
        try:
            for k in range(attempts):
                if self.dry:
                    holder: dict[str, Any] = {}
                    llm = make_slot3_mock_llm(self.cfg.seed, holder)
                    panel = [(f"mock{j}", make_slot3_mock_llm(self.cfg.seed, holder))
                             for j in range(3)]
                else:
                    llm, panel = self.models(gen_name)
                try:
                    res = generate_family_slot3(fid, topic, primary, self.cfg, llm=llm,
                                                auditor_llms=panel, generator=gen_name,
                                                rare=rare, resume_from=resume,
                                                split=self.cfg.split)
                except Exception as e:  # noqa: BLE001 -- recorded; the run continues
                    self.log(self.attempts_path, {
                        "family_id": fid, "topic": topic, "primary": primary,
                        "generator": gen_name, "attempt": done + k + 1, "outcome": "error",
                        "error": f"{type(e).__name__}: {e}",
                        "traceback": traceback.format_exc()[-2000:]})
                    msg = f"ERROR {type(e).__name__}: {e}"
                    continue
                rec = {
                    "family_id": fid, "topic": topic, "primary": primary,
                    "generator": gen_name, "rare": rare, "attempt": done + k + 1,
                    "outcome": "admitted" if res.admitted else "rejected",
                    "stage": res.stage,
                    "ladders": [l["type"] for l in (res.manifest or {}).get("ladders", [])],
                    "rejected_ladders": (res.manifest or {}).get("rejected_ladders", {}),
                    "reason": "" if res.admitted else res.verdict.reason()[:600],
                    "calls": res.calls, "timing": res.timing, "events": res.events,
                }
                self.log(self.attempts_path, rec)
                with self.lock:
                    if res.admitted:
                        self.store.append_items(res.items)
                        self.store.put_manifest(res.manifest)
                        self.store.clear_rejection(fid)
                        self.store.put(FamilyState(
                            fid, topic, primary, stage="admitted", attempts=done + k + 1,
                            item_ids=[it["id"] for it in res.items],
                            artifacts={**res.artifacts, "verdict": res.verdict.to_dict(),
                                       "calls": res.calls, "generator": gen_name}))
                    else:
                        self.store.reject(fid, res.verdict.to_dict(), stage=res.stage)
                        self.store.put(FamilyState(
                            fid, topic, primary, stage=res.stage, attempts=done + k + 1,
                            rejected_reason=res.verdict.reason(),
                            artifacts={**res.artifacts, "generator": gen_name}))
                if res.admitted:
                    return (f"{fid} {topic} ({primary}, {gen_name}): admitted "
                            f"{rec['ladders']}"
                            + (f", lost {rec['rejected_ladders']}" if rec["rejected_ladders"]
                               else ""))
                resume = {k2: res.artifacts[k2] for k2 in ("question", "claims")
                          if res.artifacts.get(k2)}
                msg = f"REJECTED at {res.stage}: {rec['reason'][:140]}"
            return f"{fid} {topic} ({primary}, {gen_name}): {msg}"
        finally:
            self.after_family()


# ---------------------------------------------------------------------------- report


def report(out_dir: str) -> None:
    """Per-gate pass rates, admission per type and generator, calls, spend."""
    path = os.path.join(out_dir, "attempts.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"no attempts.jsonl under {out_dir}")
    recs = [json.loads(l) for l in open(path) if l.strip()]
    fams = collections.OrderedDict()
    for r in recs:
        fams.setdefault(r["family_id"], []).append(r)
    gate = collections.defaultdict(lambda: [0, 0])
    sv5_role = collections.defaultdict(lambda: [0, 0])
    for r in recs:
        for e in r.get("events", []):
            g = gate[e["gate"]]
            g[0] += bool(e["passed"])
            g[1] += 1
            if e["gate"] == "SV5" and e.get("attempt") == 0:
                rr = sv5_role[f"{e.get('role')}/{e.get('kind')}"]
                rr[0] += bool(e["passed"])
                rr[1] += 1
    adm = [f for f, rs in fams.items() if any(x["outcome"] == "admitted" for x in rs)]
    first = [f for f, rs in fams.items() if rs[0]["outcome"] == "admitted"]
    ladders = collections.Counter()
    lost = collections.Counter()
    for rs in fams.values():
        for x in rs:
            if x["outcome"] == "admitted":
                ladders.update(x["ladders"])
                lost.update(x["rejected_ladders"].keys())
    calls = collections.Counter()
    for r in recs:
        calls.update(r.get("calls", {}))
    n_lad = sum(ladders.values())
    print(f"[slot3] {out_dir}")
    print(f"  families attempted   : {len(fams)}   attempts: {len(recs)}")
    print(f"  admitted (any ladder): {len(adm)}/{len(fams)}   first try: {len(first)}/{len(fams)}")
    print(f"  ladders admitted     : {dict(ladders)}   (lost in admitted families: {dict(lost)})")
    for key in ("primary", "generator"):
        by = collections.defaultdict(lambda: [0, 0])
        for f, rs in fams.items():
            v = rs[0].get(key, "?")
            by[v][1] += 1
            by[v][0] += any(x["outcome"] == "admitted" and rs[0]["primary"] in x["ladders"]
                            for x in rs)
        print(f"  primary ladder admitted, by {key}: "
              + ", ".join(f"{t} {a}/{n}" for t, (a, n) in sorted(by.items())))
    print(f"  calls total          : {sum(calls.values())}   per admitted ladder: "
          f"{sum(calls.values()) / max(1, n_lad):.1f}")
    print("  calls by prompt      : " + ", ".join(f"{k} {v}" for k, v in sorted(calls.items())))
    print("  gate pass rates (every evaluation, retries included):")
    for g, (ok, n) in sorted(gate.items()):
        print(f"    {g:22s} {ok:4d}/{n:<4d} {ok / n:.2f}")
    print("  SV5 first-reading pass rate, by role/kind:")
    for g, (ok, n) in sorted(sv5_role.items()):
        print(f"    {g:22s} {ok:4d}/{n:<4d} {ok / n:.2f}")
    errs = [r for r in recs if r["outcome"] == "error"]
    if errs:
        print(f"  ERRORS: {len(errs)} -- e.g. {errs[0]['error'][:200]}")
    sp = os.path.join(out_dir, "spend.jsonl")
    if os.path.exists(sp):
        rows = [json.loads(l) for l in open(sp) if l.strip()]
        vals = [r["key_spend"] for r in rows if r.get("key_spend") is not None]
        if len(vals) >= 2:
            print(f"  key spend during run : ${vals[-1] - vals[0]:.2f} (key total ${vals[-1]:.2f}; "
                  "includes any concurrent use of the key)")


def check(cfg: GenConfig, slots: list[tuple[str, str, str, str]], ping: bool) -> None:
    """Print the allocation and committees; with ``ping``, call every model once."""
    print(f"[slot3] {cfg.dataset_name}: {len(slots)} families -> {cfg.out_dir} "
          f"(split {cfg.split!r})")
    cells = collections.Counter((p, g) for _f, _t, p, g in slots)
    gens = sorted({g for *_x, g in slots})
    print("  families per (type, generator):")
    print(f"    {'':9s}" + "".join(f"{g:>18s}" for g in gens))
    for p in S.PRIMARY_TYPES:
        print(f"    {p:9s}" + "".join(f"{cells[(p, g)]:>18d}" for g in gens))
    per_topic = collections.defaultdict(list)
    for _f, t, p, g in slots:
        per_topic[t].append((p, g))
    print(f"  topics: {len(per_topic)}; families per topic "
          f"{dict(sorted(collections.Counter(len(v) for v in per_topic.values()).items()))}; "
          f"types per topic {dict(collections.Counter(len({p for p, _ in v}) for v in per_topic.values()))}; "
          f"generators per topic {dict(collections.Counter(len({g for _, g in v}) for v in per_topic.values()))}")
    rare = collections.Counter(s for i in range(len(slots)) for s in S.rare_senses_for(i))
    print(f"  rare senses allocated: {dict(rare)}")
    models: dict[str, ModelRef] = {}
    for g in cfg.generators:
        panel = cfg.eligible_auditors(g)
        print(f"  {g.name:16s} ({g.model_id}) -> committee "
              + ", ".join(f"{m.name} [{m.family}]" for m in panel))
        models[g.model_id] = g
        for m in panel:
            models[m.model_id] = m
    if not ping:
        return
    gw = _gateway(cfg)
    if gw is None:
        raise SystemExit("[slot3] --ping needs one gateway base_url and OPENAI_API_KEY set")
    print("  ping (one 5-token request each):")
    for mid in sorted(models):
        t0 = time.time()
        status, _h, text = _chat(*gw, mid)
        print(f"    {mid:32s} {status} {time.time() - t0:5.1f}s {text[:60]!r}")
    spend = key_spend(cfg)
    if spend is not None:
        print(f"  key spend so far: ${spend:.2f}")


def main() -> None:
    args = _parser().parse_args()
    cfg = load_config(args.config)
    if args.out:
        cfg = cfg.with_overrides(out_dir=args.out)
    if args.dry_run:
        cfg = cfg.with_overrides(dry_run=True)
    if args.report:
        report(cfg.out_dir)
        return
    try:
        cfg.validate()
    except ValueError as e:
        raise SystemExit(f"[slot3] invalid configuration: {e}") from None
    all_slots = plan_slots(cfg)
    if args.check:
        check(cfg, all_slots, args.ping)
        return
    store = Store(cfg.out_dir)
    store.save_config({**cfg.to_dict(), "design": S.DESIGN, "taxonomy": S.TAXONOMY})
    slots = all_slots
    if args.families:
        want = set(args.families.split(","))
        slots = [s for s in slots if s[0] in want]
    if cfg.limit is not None:
        slots = slots[: cfg.limit]
    index = {fid: i for i, (fid, *_x) in enumerate(all_slots)}
    todo = [s for s in slots if not (store.get(s[0]) and store.get(s[0]).done)]
    if not args.dry_run:
        for g in cfg.generators:
            names = [m.name for m in cfg.eligible_auditors(g)]
            print(f"[slot3] generator {g.name} ({g.model_id}); committee {names}")
    print(f"[slot3] {len(slots)} families, {len(slots) - len(todo)} already admitted, "
          f"{len(todo)} to run -> {cfg.out_dir}")
    run = _Run(cfg, store, args.dry_run, args.max_spend, args.track_spend)
    run.spend0 = run.snapshot("start")
    if run.track and not args.dry_run and run.spend0 is None:
        print("[slot3] WARNING: could not read the key's spend; --max-spend is inactive")
    workers = args.workers or cfg.max_concurrency or 1
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run.family, index[f], f, t, p, g, args.family_attempts): f
                for f, t, p, g in todo}
        for fut in as_completed(futs):
            print("  " + fut.result(), flush=True)
    run.snapshot("end")
    report(cfg.out_dir)


if __name__ == "__main__":
    main()
