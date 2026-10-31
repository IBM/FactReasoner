#!/usr/bin/env python
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

"""Generate a LoCoBench corpus in parallel shards, then merge them into one corpus.

The generator is sequential (one model call at a time, ~45 s each), so a 100-family corpus
is ~29 hours in one process. Sharding runs several generator processes at once, each in its
own output directory, and merges them at the end.

HOW SHARDS STAY DISJOINT. Every shard is launched with the SAME ``--n-families`` and a
disjoint set of ``--only-topic`` flags. The CLI assigns ``(family_id, topic, ladder type)``
to slots from ``--n-families`` alone, then filters by topic, so each shard generates exactly
the global slots whose topic it owns -- the same family ids, topics and ladder types a single
unsharded run would have produced. Nothing about a family depends on which shard ran it.

Subcommands::

    plan    assign topics to shards, balancing projected calls; write shards.json
    launch  start every shard detached (nohup + caffeinate), recording pids and logs
    reset   return every family to its stored base response (committee or P5 changed)
    regen   regenerate one family from P1 in a scratch dir and splice it into its shard
    status  one table over all shards: admitted / rejected / in progress / running
    merge   consolidate the shards into one corpus directory, validating everything

Run::

    python scripts/locobench_shards.py plan --config configs/locobench_opus55.json \\
        --n-families 100 --shards 8 --root data/locobench-opus-5-5-shards
    python scripts/locobench_shards.py launch --root data/locobench-opus-5-5-shards
    python scripts/locobench_shards.py status --root data/locobench-opus-5-5-shards
    python scripts/locobench_shards.py merge --root data/locobench-opus-5-5-shards \\
        --out data/locobench-opus-5-5
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fact_reasoner.locobench import cli, perturb  # noqa: E402
from fact_reasoner.locobench.config import load_config  # noqa: E402
from fact_reasoner.locobench.schema import (  # noqa: E402
    validate_item,
    validate_manifest_entry,
)
from fact_reasoner.locobench.store import (  # noqa: E402
    CONFIG_FILE,
    ITEMS_FILE,
    MANIFEST_FILE,
    REJECTED_DIR,
    STATE_FILE,
)

PLAN_FILE = "shards.json"
#: Config keys allowed to differ between shards. Everything else -- models, thresholds,
#: temperatures, corpus size, dataset name -- must match, or the shards are not one corpus.
SHARD_LOCAL_KEYS = ("only_topics", "out_dir", "limit")


def _slots(n_families: int) -> list[tuple[str, str, str]]:
    """The global ``(family_id, topic, type)`` slots, exactly as the CLI assigns them."""
    return cli._slots(SimpleNamespace(n_families=n_families, only_topics=[], limit=None))


def _git_commit() -> str:
    """The current commit, with a ``-dirty`` suffix when the tree has local changes."""
    def git(*a: str) -> str:
        return subprocess.run(
            ["git", *a], capture_output=True, text=True, check=False
        ).stdout.strip()

    sha = git("rev-parse", "HEAD") or "unknown"
    return sha + ("-dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


# -- plan ---------------------------------------------------------------------


def plan(
    config: str,
    n_families: int,
    n_shards: int,
    root: str,
    *,
    python: str = sys.executable,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Assign topics to shards and write ``shards.json``.

    Topics are the unit of assignment, so balance is approximate: each topic is placed, in
    descending order of projected calls, on the currently lightest shard (LPT scheduling).
    A topic's projected calls are its families' one-pass budget from
    :func:`perturb.call_budget`, which CONFLICT and CHAIN dominate.
    """
    cfg = load_config(config)
    slots = _slots(n_families)
    by_topic: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for fid, topic, fam in slots:
        by_topic[topic].append((fid, fam))
    n_voters = 3
    if cfg.generators and cfg.committee:
        n_voters = len(cfg.eligible_auditors(cfg.generators[0]))
    cost = {
        t: perturb.call_budget([f for _, f in fams], n_voters=n_voters)["total"]
        for t, fams in by_topic.items()
    }
    shards = [{"calls": 0, "topics": []} for _ in range(n_shards)]
    for topic in sorted(cost, key=lambda t: (-cost[t], t)):
        target = min(shards, key=lambda s: s["calls"])
        target["calls"] += cost[topic]
        target["topics"].append(topic)

    entries = []
    for i, s in enumerate(shards, 1):
        topics = sorted(s["topics"])
        fams = sorted(f for t in topics for f in by_topic[t])
        out = os.path.join(root, f"s{i:02d}")
        cmd = [python, "-u", "-m", "fact_reasoner.locobench.cli", "--config", config,
               "--out", out, "--n-families", str(n_families), "--limit", str(n_families)]
        if dry_run:
            cmd.append("--dry-run")
        for t in topics:
            cmd += ["--only-topic", t]
        entries.append(
            {
                "shard": f"s{i:02d}",
                "out_dir": out,
                "topics": topics,
                "family_ids": [fid for fid, _ in fams],
                "ladders": dict(collections.Counter(f for _, f in fams)),
                "projected_calls": s["calls"],
                "command": cmd,
                "log": os.path.join(root, f"s{i:02d}.log"),
                "pid": None,
            }
        )
    manifest = {
        "config": config,
        "n_families": n_families,
        "dry_run": dry_run,
        "git_commit": _git_commit(),
        "shards": entries,
    }
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, PLAN_FILE), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _load_plan(root: str) -> dict[str, Any]:
    with open(os.path.join(root, PLAN_FILE)) as f:
        return json.load(f)


def _save_plan(root: str, manifest: dict[str, Any]) -> None:
    with open(os.path.join(root, PLAN_FILE), "w") as f:
        json.dump(manifest, f, indent=2)


# -- launch -------------------------------------------------------------------


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def launch(root: str, *, only: list[str] | None = None, wait: bool = False) -> list[str]:
    """Start each shard detached, appending to its log. Skips a shard already running.

    A shard is resumable, so re-launching one that crashed continues where it stopped.
    ``caffeinate -i`` (macOS) keeps the machine from idle-sleeping under a long run; it is
    used when present. ``wait`` blocks until every launched shard exits (tests, dry runs).
    """
    manifest = _load_plan(root)
    if manifest["git_commit"] != _git_commit():
        print(
            f"[shards] WARNING: plan was made at {manifest['git_commit']}, tree is now "
            f"{_git_commit()}; shards launched now run different code than the plan."
        )
    started, procs = [], []
    caff = shutil.which("caffeinate")
    for e in manifest["shards"]:
        if only and e["shard"] not in only:
            continue
        if _alive(e.get("pid")):
            print(f"[shards] {e['shard']} already running (pid {e['pid']}); skipped")
            continue
        cmd = ([caff, "-i"] if caff else []) + e["command"]
        log = open(e["log"], "a")
        env = {**os.environ, "PYTHONPATH": os.path.join(os.getcwd(), "src")}
        p = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, env=env,
        )
        e["pid"] = p.pid
        started.append(e["shard"])
        procs.append(p)
    _save_plan(root, manifest)
    if wait:
        for p in procs:
            p.wait()
    return started


# -- reset --------------------------------------------------------------------


#: Artifacts a reset keeps: everything up to and including the base response. The pipeline
#: resumes from a stored response by re-running the committee on it, then regenerating the
#: rungs -- so a reset re-validates and re-perturbs without re-planning or re-writing.
KEEP_ON_RESET = ("question", "claims", "plan", "response")


def reset(
    root: str,
    *,
    reason: str,
    families: list[str] | None = None,
    keep: tuple[str, ...] = KEEP_ON_RESET,
) -> dict[str, int]:
    """Return families to "base response written, not yet validated" (all, by default).

    Used when the committee or P5 changes mid-run, so that every admitted family carries a
    verdict from the same panel and rungs from the same P5. Items and manifest entries of
    previously admitted families are removed (they will be regenerated); rejection records
    are cleared; attempts restart at zero. With ``families``, only those are reset and every
    other family's items, manifest entry and rejection record are kept in place. ``keep``
    names the artifacts to retain; dropping ``plan`` makes the family re-plan from P3. Each shard's prior ``state.json`` is kept as
    ``state.pre-reset-<n>.json`` and the reason is logged in ``shards.json``.
    """
    manifest = _load_plan(root)
    n = len(manifest.get("resets", [])) + 1
    counts = collections.Counter()
    for e in manifest["shards"]:
        d = e["out_dir"]
        sp = os.path.join(d, STATE_FILE)
        if not os.path.exists(sp):
            continue
        if families is not None and not set(families) & set(e["family_ids"]):
            continue
        if _alive(e.get("pid")):
            raise RuntimeError(f"{e['shard']} is running; stop it before resetting")
        raw = _read_json(sp)
        shutil.copy2(sp, os.path.join(d, f"state.pre-reset-{n}.json"))
        fams = raw.get("families", {})
        for fid, fs in fams.items():
            if families is not None and fid not in families:
                continue
            kept = {k: fs["artifacts"][k] for k in keep if k in fs.get("artifacts", {})}
            counts["with_response" if "response" in kept else "replanned"] += 1
            counts[f"was_{'admitted' if fs.get('stage') == 'admitted' else 'other'}"] += 1
            fams[fid] = {**fs, "stage": "", "attempts": 0, "rejected_reason": "",
                         "item_ids": [], "artifacts": kept}
        with open(sp, "w") as f:
            json.dump(raw, f, indent=2)
        if families is None:
            for name in (ITEMS_FILE, MANIFEST_FILE):
                path = os.path.join(d, name)
                if os.path.exists(path):
                    os.replace(path, os.path.join(d, f"{name}.pre-reset-{n}"))
            rdir = os.path.join(d, REJECTED_DIR)
            if os.path.isdir(rdir):
                shutil.move(rdir, os.path.join(d, f"{REJECTED_DIR}.pre-reset-{n}"))
                os.makedirs(rdir)
        else:
            _drop_families(d, set(families), n)
    manifest.setdefault("resets", []).append(
        {"n": n, "reason": reason, "git_commit": _git_commit(), "counts": dict(counts),
         "families": sorted(families) if families is not None else "all",
         "kept": list(keep)}
    )
    _save_plan(root, manifest)
    return dict(counts)


def _drop_families(d: str, fids: set[str], n: int) -> None:
    """Remove ``fids``' items, manifest entries and rejection records from shard ``d``.

    The originals are kept beside the live files with a ``.pre-reset-<n>`` suffix.
    """
    ip = os.path.join(d, ITEMS_FILE)
    if os.path.exists(ip):
        shutil.copy2(ip, f"{ip}.pre-reset-{n}")
        with open(ip) as f:
            keep = [ln for ln in f if ln.strip()
                    and json.loads(ln)["expected"]["family_id"] not in fids]
        with open(ip, "w") as f:
            f.writelines(keep)
    mp = os.path.join(d, MANIFEST_FILE)
    if os.path.exists(mp):
        shutil.copy2(mp, f"{mp}.pre-reset-{n}")
        m = _read_json(mp)
        m["families"] = [e for e in m["families"] if e["family_id"] not in fids]
        with open(mp, "w") as f:
            json.dump(m, f, indent=2)
    rdir = os.path.join(d, REJECTED_DIR)
    for fid in fids:
        rp = os.path.join(rdir, f"{fid}.json")
        if os.path.exists(rp):
            os.makedirs(f"{rdir}.pre-reset-{n}", exist_ok=True)
            shutil.move(rp, os.path.join(f"{rdir}.pre-reset-{n}", f"{fid}.json"))


# -- regen --------------------------------------------------------------------


def regen(root: str, family: str, *, reason: str, no_cache: bool = True) -> str:
    """Regenerate one family from P1 in a scratch directory, then splice it into its shard.

    For a family whose PLAN is the problem -- repeated, agreeing committee rejections on
    fresh prose -- retrying from the stored plan cannot help. This discards every stored
    artifact and reruns the family's own slot (same id, topic and ladder type) from P1.

    ``no_cache`` sends LiteLLM's ``cache: {no-cache: true}`` on every generator call. The
    gateway caches identical requests, and P1's request for a slot is identical every time,
    so without it the first call replays the question that led to the bad plan. The bypass
    lives in a scratch copy of the config, not in the shard's: shard configs must agree for
    the merge, and the corpus config should not record a one-family transport setting.

    The other families sharing the topic are seeded as admitted in the scratch state so the
    CLI runs only this slot. Returns the scratch directory. The family is spliced into its
    shard only if admitted; the shard's prior files are kept as ``*.pre-regen-<family>``.
    """
    manifest = _load_plan(root)
    entry = next(e for e in manifest["shards"] if family in e["family_ids"])
    if _alive(entry.get("pid")):
        raise RuntimeError(f"{entry['shard']} is running; stop it first")
    slot = next(sl for sl in _slots(manifest["n_families"]) if sl[0] == family)
    topic = slot[1]
    siblings = [sl[0] for sl in _slots(manifest["n_families"]) if sl[1] == topic]
    d = entry["out_dir"]
    shard_state = _read_json(os.path.join(d, STATE_FILE))["families"]

    scratch = os.path.join(root, f"regen-{family}")
    if os.path.exists(scratch):
        shutil.rmtree(scratch)
    os.makedirs(scratch)
    seed = {f: shard_state[f] for f in siblings if f != family}
    if any(v.get("stage") != "admitted" for v in seed.values()):
        raise RuntimeError(f"siblings of {family} on {topic} are not all admitted: {list(seed)}")
    with open(os.path.join(scratch, STATE_FILE), "w") as f:
        json.dump({"families": seed}, f, indent=2)
    cfg = _read_json(manifest["config"])
    if no_cache:
        for g in cfg["generators"]:
            g.setdefault("model_options", {}).setdefault("extra_body", {})["cache"] = {
                "no-cache": True
            }
    cfg_path = os.path.join(scratch, "regen_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    cmd = list(entry["command"])
    cmd[cmd.index("--config") + 1] = cfg_path
    cmd[cmd.index("--out") + 1] = scratch
    i = cmd.index("--only-topic")
    cmd = cmd[:i] + ["--only-topic", topic]
    env = {**os.environ, "PYTHONPATH": os.path.join(os.getcwd(), "src")}
    with open(os.path.join(scratch, "run.log"), "w") as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)

    new = _read_json(os.path.join(scratch, STATE_FILE))["families"].get(family, {})
    if new.get("stage") != "admitted":
        print(f"[shards] {family} not admitted in {scratch}: {new.get('rejected_reason', '')[:200]}")
        return scratch
    tag = f"pre-regen-{family}"
    for name in (STATE_FILE, ITEMS_FILE, MANIFEST_FILE):
        shutil.copy2(os.path.join(d, name), os.path.join(d, f"{name}.{tag}"))
    new.setdefault("artifacts", {})["regenerated"] = {
        "from": scratch, "reason": reason, "no_cache": no_cache, "config": cfg_path,
        "previous_attempts": shard_state.get(family, {}).get("attempts"),
    }
    st = _read_json(os.path.join(d, STATE_FILE))
    st["families"][family] = new
    with open(os.path.join(d, STATE_FILE), "w") as f:
        json.dump(st, f, indent=2)
    _drop_families(d, {family}, f"regen-{family}")
    with open(os.path.join(scratch, ITEMS_FILE)) as f:
        fresh = [ln for ln in f if ln.strip()]
    with open(os.path.join(d, ITEMS_FILE), "a") as f:
        f.writelines(fresh)
    m = _read_json(os.path.join(d, MANIFEST_FILE))
    m["families"] = sorted(
        m["families"] + [e for e in _read_json(os.path.join(scratch, MANIFEST_FILE))["families"]
                         if e["family_id"] == family],
        key=lambda e: e["family_id"],
    )
    with open(os.path.join(d, MANIFEST_FILE), "w") as f:
        json.dump(m, f, indent=2)
    manifest.setdefault("regens", []).append(
        {"family": family, "shard": entry["shard"], "reason": reason, "scratch": scratch,
         "git_commit": _git_commit(), "no_cache": no_cache}
    )
    _save_plan(root, manifest)
    return scratch


# -- status -------------------------------------------------------------------


def shard_status(entry: dict[str, Any], max_attempts: int) -> dict[str, Any]:
    """Count a shard's families by outcome, read straight from its ``state.json``."""
    state_path = os.path.join(entry["out_dir"], STATE_FILE)
    state = {}
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f).get("families", {})
    admitted = rejected = exhausted = progress = 0
    for fid in entry["family_ids"]:
        fs = state.get(fid)
        if not fs:
            continue
        if fs.get("stage") == "admitted":
            admitted += 1
        elif fs.get("rejected_reason"):
            if int(fs.get("attempts", 0)) >= max_attempts:
                exhausted += 1
            else:
                rejected += 1
        else:
            progress += 1
    total = len(entry["family_ids"])
    return {
        "shard": entry["shard"],
        "families": total,
        "admitted": admitted,
        "retrying": rejected,
        "exhausted": exhausted,
        "in_progress": progress,
        "unstarted": total - admitted - rejected - exhausted - progress,
        "running": _alive(entry.get("pid")),
    }


def status(root: str) -> list[dict[str, Any]]:
    manifest = _load_plan(root)
    max_attempts = load_config(manifest["config"]).max_attempts
    return [shard_status(e, max_attempts) for e in manifest["shards"]]


def _print_status(rows: list[dict[str, Any]]) -> None:
    cols = ("shard", "families", "admitted", "retrying", "exhausted", "in_progress",
            "unstarted", "running")
    print("  ".join(f"{c:>11s}" for c in cols))
    for r in rows:
        print("  ".join(f"{str(r[c]):>11s}" for c in cols))
    tot = {c: sum(r[c] for r in rows) for c in cols[1:7]}
    print("  ".join([f"{'total':>11s}"] + [f"{tot[c]:>11d}" for c in cols[1:7]]
                    + [f"{sum(r['running'] for r in rows):>8d} up"]))


# -- merge --------------------------------------------------------------------


class MergeError(RuntimeError):
    """A precondition of the merge failed; nothing was written."""


def _read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def merge(root: str, out: str, *, allow_incomplete: bool = False) -> dict[str, Any]:
    """Consolidate every shard into ``out``. All-or-nothing.

    Preconditions, each a :class:`MergeError` if violated: no shard is still running; every
    shard's family slots are admitted (unless ``allow_incomplete``, which still requires no
    shard to be running); the shard configs agree on everything but the shard-local keys;
    no family id or item id occurs in two shards; and every family the plan assigned to a
    shard was produced by that shard and no other. Then every item and manifest entry is
    re-validated against the schema, and each family must carry exactly five rungs.

    The result is built in a temporary sibling directory and renamed into place, so a
    failed merge leaves ``out`` absent rather than half-written. ``out`` must not exist.
    """
    manifest = _load_plan(root)
    if os.path.exists(out):
        raise MergeError(f"{out} already exists; refusing to overwrite a corpus")
    max_attempts = load_config(manifest["config"]).max_attempts

    rows = [shard_status(e, max_attempts) for e in manifest["shards"]]
    running = [r["shard"] for r in rows if r["running"]]
    if running:
        raise MergeError(f"shards still running: {running}")
    short = [r["shard"] for r in rows if r["admitted"] < r["families"]]
    if short and not allow_incomplete:
        raise MergeError(
            f"shards with unadmitted families: {short}; re-launch them, or pass "
            "--allow-incomplete to release what was admitted"
        )

    configs = {}
    for e in manifest["shards"]:
        c = _read_json(os.path.join(e["out_dir"], CONFIG_FILE))
        configs[e["shard"]] = {k: v for k, v in c.items() if k not in SHARD_LOCAL_KEYS}
    first = next(iter(configs.values()))
    differing = [s for s, c in configs.items() if c != first]
    if differing:
        diff = sorted(
            k for s in differing for k in set(first) | set(configs[s])
            if first.get(k) != configs[s].get(k)
        )
        raise MergeError(f"shard configs differ ({differing}) on keys {sorted(set(diff))}")

    items: list[dict[str, Any]] = []
    fams: dict[str, dict[str, Any]] = {}
    state: dict[str, Any] = {}
    rejected: dict[str, str] = {}
    owner: dict[str, str] = {}
    seen_items: dict[str, str] = {}
    for e in manifest["shards"]:
        d = e["out_dir"]
        planned = set(e["family_ids"])
        st = _read_json(os.path.join(d, STATE_FILE)).get("families", {})
        mf = _read_json(os.path.join(d, MANIFEST_FILE)).get("families", [])
        for fid in st:
            if fid not in planned:
                raise MergeError(f"{e['shard']} produced {fid}, which the plan gave elsewhere")
            if fid in owner:
                raise MergeError(f"{fid} produced by both {owner[fid]} and {e['shard']}")
            owner[fid] = e["shard"]
            state[fid] = st[fid]
        for entry in mf:
            fams[entry["family_id"]] = entry
        with open(os.path.join(d, ITEMS_FILE)) as f:
            for line in f:
                if not line.strip():
                    continue
                it = json.loads(line)
                if it["id"] in seen_items:
                    raise MergeError(
                        f"item {it['id']} in both {seen_items[it['id']]} and {e['shard']}"
                    )
                seen_items[it["id"]] = e["shard"]
                items.append(it)
        rdir = os.path.join(d, REJECTED_DIR)
        if os.path.isdir(rdir):
            for name in os.listdir(rdir):
                if name.endswith(".json"):
                    rejected[name] = os.path.join(rdir, name)

    admitted = {fid for fid, s in state.items() if s.get("stage") == "admitted"}
    if set(fams) != admitted:
        raise MergeError(
            f"manifest/state disagree: manifest-only {sorted(set(fams) - admitted)}, "
            f"state-only {sorted(admitted - set(fams))}"
        )
    per_family = collections.Counter(it["expected"]["family_id"] for it in items)
    bad = {fid: n for fid, n in per_family.items() if n != 5 or fid not in admitted}
    missing = sorted(admitted - set(per_family))
    if bad or missing:
        raise MergeError(f"families without exactly 5 admitted items: {bad}, missing {missing}")
    for it in items:
        validate_item(it)
    for entry in fams.values():
        validate_manifest_entry(entry)

    items.sort(key=lambda it: (it["expected"]["family_id"], it["expected"]["rung_index"]))
    merged_cfg = {
        **first,
        "out_dir": out,
        "only_topics": [],
        "limit": None,
        "merged_from": {
            "root": root,
            "git_commit": manifest["git_commit"],
            "shards": {e["shard"]: e["topics"] for e in manifest["shards"]},
        },
    }

    parent = os.path.dirname(os.path.abspath(out)) or "."
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.mkdtemp(dir=parent, prefix=".merge-")
    try:
        with open(os.path.join(tmp, ITEMS_FILE), "w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
        with open(os.path.join(tmp, MANIFEST_FILE), "w") as f:
            json.dump(
                {"manifest_version": "1.0", "families": [fams[k] for k in sorted(fams)]},
                f, indent=2,
            )
        with open(os.path.join(tmp, STATE_FILE), "w") as f:
            json.dump({"families": {k: state[k] for k in sorted(state)}}, f, indent=2)
        with open(os.path.join(tmp, CONFIG_FILE), "w") as f:
            json.dump(merged_cfg, f, indent=2)
        os.makedirs(os.path.join(tmp, REJECTED_DIR))
        for name, src in rejected.items():
            shutil.copy2(src, os.path.join(tmp, REJECTED_DIR, name))
        os.replace(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {
        "families": len(fams),
        "items": len(items),
        "rejected_records": len(rejected),
        "ladders": dict(collections.Counter(e["family"] for e in fams.values())),
        "topics": len({e["canonical_topic"] for e in fams.values()}),
    }


# -- cli ----------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--config", required=True)
    p.add_argument("--n-families", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("launch")
    p.add_argument("--root", required=True)
    p.add_argument("--only", action="append", help="shard id, repeatable")
    p.add_argument("--wait", action="store_true")
    p = sub.add_parser("reset")
    p.add_argument("--root", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--family", action="append", help="family id, repeatable; default all")
    p.add_argument("--keep", default=",".join(KEEP_ON_RESET),
                   help="comma-separated artifacts to retain (default: through the response)")
    p = sub.add_parser("regen")
    p.add_argument("--root", required=True)
    p.add_argument("--family", required=True)
    p.add_argument("--reason", required=True)
    p.add_argument("--keep-cache", action="store_true")
    p = sub.add_parser("status")
    p.add_argument("--root", required=True)
    p = sub.add_parser("merge")
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-incomplete", action="store_true")
    a = ap.parse_args()

    if a.cmd == "plan":
        m = plan(a.config, a.n_families, a.shards, a.root, python=a.python,
                 dry_run=a.dry_run)
        for e in m["shards"]:
            print(f"{e['shard']}: {len(e['family_ids']):3d} families, "
                  f"{e['projected_calls']:4d} calls, {e['ladders']}  {e['topics']}")
        print(f"[shards] plan written to {os.path.join(a.root, PLAN_FILE)} "
              f"at {m['git_commit']}")
    elif a.cmd == "launch":
        print(f"[shards] started: {launch(a.root, only=a.only, wait=a.wait)}")
    elif a.cmd == "reset":
        keep = tuple(k for k in a.keep.split(",") if k)
        print(f"[shards] reset: "
              f"{reset(a.root, reason=a.reason, families=a.family, keep=keep)}")
    elif a.cmd == "regen":
        print(f"[shards] regen scratch: "
              f"{regen(a.root, a.family, reason=a.reason, no_cache=not a.keep_cache)}")
    elif a.cmd == "status":
        _print_status(status(a.root))
    elif a.cmd == "merge":
        try:
            summary = merge(a.root, a.out, allow_incomplete=a.allow_incomplete)
        except MergeError as e:
            print(f"[shards] merge refused: {e}")
            return 1
        print(f"[shards] merged into {a.out}: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
