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

"""The two-stage factuality+coherence experiment over the data/factuality corpora.

For each dataset item (which already carries atoms and retrieved contexts) this
driver runs, in order:

  1. **FactReasoner v2** over atoms + contexts -> posterior marginals ``q_i``.
     Nothing is atomized and nothing is retrieved: the corpus already has both, so
     the only LLM cost is the atom<->context NLI, scored with ``--nli-mode fast``.
  2. **The coherence MRF**, ONCE PER PAIR POLICY (windowed and bidirectional),
     with the stage-1 marginals as the atoms' unary priors, reading out all four
     LCS scores (mean_marginal, consistency, reified, log_partition).
  3. **The coherence baselines** -- model-free (controls, discourse), NLI-based
     (contradiction counting, ROSCOE-SC + ablation arms) and the LLM judges.

Three design points that are the reason this is one driver rather than three:

* **Stage 1 runs once per item, not once per policy.** The marginals do not depend
  on the candidate-pair policy, so they are computed once and passed to both
  coherence runs via ``run_from_mining``. That halves the factuality cost and, more
  importantly, makes the two policies an exact ablation: they see byte-identical
  priors, so any difference is attributable to pair selection alone.
* **Baselines see the same atoms.** They are handed the item's own atom texts,
  never re-atomized, which is what makes the comparison an ablation rather than a
  different experiment.
* **One record per item, written incrementally.** Every datapoint is persisted as
  it completes and already-done items are skipped on restart, so a multi-hour run
  survives interruption. A record carries the marginals, all four readouts per
  policy, the mining diagnostics and every baseline score, so the report and the
  paper section are built from disk without re-running anything.

Usage:
    # Smoke test (2 items, no judges)
    python scripts/run_factuality_coherence.py --rits-model llama-3.3-70b-instruct \
        --merlin-path ~/git/merlin/bin/merlin --limit 2 --no-judges

    # A dataset
    python scripts/run_factuality_coherence.py --rits-model llama-3.3-70b-instruct \
        --merlin-path ~/git/merlin/bin/merlin --datasets bio
"""

import argparse
import json
import os
import sys
import time
import traceback

from dotenv import load_dotenv

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(REPO, ".env"))

from fact_reasoner.coherence_baselines import (  # noqa: E402
    CONTROL_BASELINES,
    DISCOURSE_BASELINES,
    DirectCoherenceRating,
    GEvalCoherence,
    PairwiseNLIContradiction,
    RoscoeSelfConsistency,
    make_backend_generate,
)
from fact_reasoner.lcs.lcs_scorer import LCS_METHODS  # noqa: E402

# The two candidate-pair policies compared. "windowed" is forward-only within an
# order window; "bidirectional" emits both arc directions within the same radius,
# so it can express a relation that runs backward in assertion order.
PAIR_POLICIES = ("windowed", "bidirectional")

DATASET_LABELS = ("bio", "askhist", "books", "eli5", "lfobj")


class _SeededJudge:
    """Run a judge ``seeds`` times and report the mean, keeping the spread.

    A judge's run-to-run variance bounds what any comparison against it can
    conclude, so a single sample would overstate its precision. Mirrors the
    wrapper in ``scripts/run_coherence_baselines.py``.
    """

    def __init__(self, judge, seeds: int):
        self.judge = judge
        self.seeds = seeds
        self.name = f"{judge.name}_x{seeds}"

    def score(self, atoms, response):
        runs = [self.judge.score(atoms, response) for _ in range(self.seeds)]
        vals = [r.score for r in runs if r.score is not None]
        first = runs[0]
        if not vals:
            first.name = self.name
            return first
        mean = sum(vals) / len(vals)
        spread = max(vals) - min(vals)
        first.score = mean
        first.name = self.name
        first.diagnostics = dict(first.diagnostics or {})
        first.diagnostics.update(
            {"seed_scores": vals, "seed_spread": spread, "seeds": self.seeds}
        )
        return first


def _apply_rits_model(args) -> None:
    """Resolve --rits-model against configs/rits_models.json into backend flags."""
    if not args.rits_model:
        return
    path = os.path.join(REPO, "configs", "rits_models.json")
    with open(path) as f:
        entries = {e["name"]: e for e in json.load(f)}
    if args.rits_model not in entries:
        raise SystemExit(
            f"Unknown --rits-model {args.rits_model!r}. "
            f"Known: {', '.join(sorted(entries))}"
        )
    e = entries[args.rits_model]
    args.backend = e.get("backend", "rits")
    args.model_id = e["model_id"]
    args.base_url = e.get("base_url")


def _load_items(sample_dir: str, datasets: list[str], limit: int | None):
    """Load the sampled items for the requested datasets."""
    items = []
    for label in datasets:
        path = os.path.join(sample_dir, f"{label}.jsonl")
        if not os.path.isfile(path):
            raise SystemExit(
                f"Missing {path!r}. Run scripts/sample_factuality_corpus.py first."
            )
        with open(path) as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    if limit:
        # Interleave across datasets so a --limit smoke test is not all one domain.
        by_ds: dict[str, list] = {}
        for it in items:
            by_ds.setdefault(it.get("dataset", "?"), []).append(it)
        items, i = [], 0
        while len(items) < limit and any(v[i:] for v in by_ds.values()):
            for v in by_ds.values():
                if i < len(v) and len(items) < limit:
                    items.append(v[i])
            i += 1
    return items


def _build_baselines(args, backend, nli, verdict_cache=None):
    """Instantiate the requested baselines (see the module docstring)."""
    baselines = []
    if not args.no_model_free:
        baselines += list(CONTROL_BASELINES) + list(DISCOURSE_BASELINES)
    if not args.no_nli_baselines:
        # The cache rides in on the throttle dict, which the baselines splat into
        # run_pairs. All five score the SAME pairs, so without it the model sees
        # each pair five times over.
        throttle = {"verdict_cache": verdict_cache} if verdict_cache else {}
        baselines += [
            PairwiseNLIContradiction(nli, throttle=throttle),
            PairwiseNLIContradiction(nli, soft=True, throttle=throttle),
            RoscoeSelfConsistency(nli, throttle=throttle),
            # The ablation arms separate "the max saturates" from "it is
            # forward-only" from "it is untyped".
            RoscoeSelfConsistency(nli, aggregate="mean", throttle=throttle),
            RoscoeSelfConsistency(nli, symmetric=True, throttle=throttle),
        ]
    if not args.no_judges:
        generate = make_backend_generate(backend)
        baselines += [
            _SeededJudge(GEvalCoherence(generate), args.judge_seeds),
            _SeededJudge(DirectCoherenceRating(generate), args.judge_seeds),
        ]
    return baselines


def _atom_texts(item) -> list[str]:
    """The item's atom texts, in assertion order."""
    return [a["text"] for a in item.get("atoms") or []]


def _relations_payload(mining) -> dict:
    """The mined relations, in full, plus type/sense histograms.

    Recording only ``num_relations`` (as an earlier version of this driver did) throws
    away the two labels the coherence model actually keys on -- the Level-1 ``type``,
    which selects the factor table, and the Level-2 ``sense``, which compiles to it --
    so no post-hoc analysis of *what kind* of relations a miner finds is possible
    without re-mining. Both the per-relation list and the aggregate counts are kept:
    the list so any later question can be asked of the raw graph, the histograms so the
    common questions need no reduction pass.
    """
    from dataclasses import asdict, is_dataclass

    rels = list(getattr(mining, "relations", None) or [])
    out: list[dict] = []
    for r in rels:
        out.append(asdict(r) if is_dataclass(r) else dict(r))

    # The dataclass names these `level1_type` / `level2_sense`; the LoCoBench harness
    # serializes the same two as `type` / `sense`. Accept either so a histogram cannot
    # silently come back all-None if the serialization changes.
    def _hist(*fields: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in out:
            key = next(
                (str(d[f]) for f in fields if d.get(f) is not None), "unknown"
            )
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    return {
        "relations": out,
        "type_counts": _hist("level1_type", "type"),
        "sense_counts": _hist("level2_sense", "sense"),
        "num_directed": sum(1 for d in out if d.get("directed")),
        "num_concession_resolved": sum(
            1 for d in out if d.get("concession_resolved")
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--sample-dir",
                    default=os.path.join(REPO, "data", "factuality_sample"))
    ap.add_argument("--out-dir",
                    default=os.path.join(REPO, "results", "factuality_coherence"))
    ap.add_argument("--datasets", default="all",
                    help=f"Comma-separated subset of {','.join(DATASET_LABELS)}, "
                         "or 'all' (default).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most this many items (smoke tests).")
    ap.add_argument("--merlin-path", default=os.environ.get("MERLIN_PATH"),
                    help="Merlin executable (or set MERLIN_PATH). Required.")
    ap.add_argument("--rits-model", default=None,
                    help="Name from configs/rits_models.json, e.g. "
                         "llama-3.3-70b-instruct or gpt-oss-120b-a100.")
    ap.add_argument("--nli-method", default="direct",
                    choices=("logprobs", "direct", "simbauq"),
                    help="Relation-probability estimator (default: direct, the "
                         "no-reasoning prompt; 'logprobs' saturates at ~1.0).")
    ap.add_argument("--nli-mode", default="fast", choices=("all_pairs", "fast"),
                    help="Factuality NLI candidate-pair preset (default: fast).")
    ap.add_argument("--pipeline-version", default="v2", choices=("v1", "v2", "v3"))
    ap.add_argument("--priors", default="factreasoner",
                    choices=("factreasoner", "none"),
                    help="Where each claim's unary prior comes from. 'factreasoner' "
                         "(default) runs stage 1 and uses its posterior marginals -- "
                         "the two-stage model. 'none' uses a flat 0.5, i.e. coherence "
                         "ONLY: no retrieval, no atom-context NLI, no factuality "
                         "assessment at all. The ablation that isolates what the "
                         "factuality stage contributes.")
    ap.add_argument("--window", type=int, default=4,
                    help="Order-window radius for both pair policies.")
    ap.add_argument("--ibound", type=int, default=6, help="Merlin WMB i-bound.")
    ap.add_argument("--judge-seeds", type=int, default=5,
                    help="Judge runs per item (default: 5; fewer cannot show a "
                         "spread).")
    ap.add_argument("--no-judges", action="store_true")
    ap.add_argument("--no-nli-baselines", action="store_true")
    ap.add_argument("--no-model-free", action="store_true")
    ap.add_argument("--nli-cache-dir", default=None,
                    help="Cross-run NLI verdict cache (recommended: reuse across "
                         "policies and restarts).")
    ap.add_argument("--progress-bar", action="store_true")
    args = ap.parse_args()

    if not args.merlin_path:
        raise SystemExit("--merlin-path is required (the MRF is solved with Merlin).")
    if not os.path.isfile(os.path.expanduser(args.merlin_path)):
        raise SystemExit(f"--merlin-path not found: {args.merlin_path!r}")
    args.merlin_path = os.path.expanduser(args.merlin_path)
    _apply_rits_model(args)
    if not getattr(args, "model_id", None):
        raise SystemExit("--rits-model is required (it resolves the backend).")

    datasets = (list(DATASET_LABELS) if args.datasets == "all"
                else [d.strip() for d in args.datasets.split(",") if d.strip()])
    unknown = [d for d in datasets if d not in DATASET_LABELS]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {unknown}")

    items = _load_items(args.sample_dir, datasets, args.limit)
    if not items:
        raise SystemExit("No items to process.")

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.rits_model or "model"
    # The prior source is part of the arm identity, so a coherence-only run writes to
    # its own file rather than overwriting the two-stage results (whose records the
    # resume check would otherwise treat as already done).
    prior_tag = "" if args.priors == "factreasoner" else f"_{args.priors}priors"
    out_path = os.path.join(
        args.out_dir, f"fc_{tag}_{args.nli_method}{prior_tag}.jsonl"
    )

    # Resume: skip items already present in the output.
    records: list[dict] = []
    if os.path.isfile(out_path):
        with open(out_path) as f:
            records = [json.loads(li) for li in f if li.strip()]
    done = {r.get("item_id") for r in records}
    print(f"[fc] {len(items)} items requested; {len(done)} already done.")

    from fact_reasoner.backends import build_backend
    from fact_reasoner.core.nli import NLIExtractor
    from fact_reasoner.lcs.runner import CoherenceRunner

    backend = build_backend(
        args.backend, model_id=args.model_id, base_url=args.base_url
    )
    nli = NLIExtractor(backend, nli_method=args.nli_method)
    disk_cache = None
    if args.nli_cache_dir:
        from fact_reasoner.core.nli_cache import NLIVerdictCache

        disk_cache = NLIVerdictCache(args.nli_cache_dir)
    baselines = _build_baselines(args, backend, nli, verdict_cache=disk_cache)
    print(f"[fc] baselines: {', '.join(b.name for b in baselines) or '(none)'}")

    # One coherence runner per pair policy. They share the backend and, crucially,
    # the factuality stage is NOT run by them -- we run it once per item ourselves
    # and hand the same marginals to both, so the policies are an exact ablation.
    runners = {
        policy: CoherenceRunner(
            backend,
            merlin_path=args.merlin_path,
            methods=tuple(LCS_METHODS),
            ibound=args.ibound,
            prior_source=args.priors,
            pipeline_version=args.pipeline_version,
            nli_mode=args.nli_mode,
            nli_method=args.nli_method,
            nli_cache_dir=args.nli_cache_dir,
            pair_policy=policy,
            window=args.window,
            show_progress=args.progress_bar,
        )
        for policy in PAIR_POLICIES
    }

    n_ok = n_fail = 0
    for idx, item in enumerate(items, 1):
        iid = item.get("item_id")
        if iid in done:
            continue
        t0 = time.perf_counter()
        print(f"\n[fc] ({idx}/{len(items)}) {iid} "
              f"atoms={len(item.get('atoms') or [])} "
              f"contexts={len(item.get('contexts') or [])}")
        try:
            record = {
                "item_id": iid,
                "dataset": item.get("dataset"),
                "input": item.get("input"),
                "output": item.get("output") or item.get("response"),
                "topic": item.get("topic"),
                "model": args.rits_model,
                "nli_method": args.nli_method,
                "num_atoms": len(item.get("atoms") or []),
                "num_contexts": len(item.get("contexts") or []),
                "gold_atom_labels": [a.get("label") for a in item.get("atoms") or []],
            }

            # -- Stage 1 + stage 2, per policy. The FIRST policy's run computes the
            # factuality marginals (assess_item_with_pipeline runs the provider);
            # the second reuses them so stage 1 is paid for exactly once.
            priors_map: dict[str, float] | None = None
            factuality: dict | None = None
            for policy in PAIR_POLICIES:
                ts = time.perf_counter()
                runner = runners[policy]
                if priors_map is None:
                    result = runner.assess_item(item)
                    priors_map = dict(result.priors)
                    factuality = result.factuality
                else:
                    # Mine under this policy, then score under the SAME priors.
                    mining = runner.miner.mine_from_atoms(
                        _atom_texts(item), record["output"]
                    )
                    pipeline = runner._make_pipeline(item)
                    result = pipeline.run_from_mining(mining, priors=priors_map)
                rel_payload = (
                    _relations_payload(result.mining) if result.mining else {}
                )
                record[f"lcs_{policy}"] = {
                    "scores": result.scores,
                    "headline": result.lcs,
                    "method": result.method,
                    "marginals": result.marginals,
                    "diagnostics": result.diagnostics,
                    "num_relations": (
                        len(result.mining.relations) if result.mining else 0
                    ),
                    # The mined graph itself, plus type/sense histograms: the labels
                    # the MRF keys on, kept so later analysis needs no re-mining.
                    **rel_payload,
                    "mining_coverage": (
                        result.mining.coverage if result.mining else {}
                    ),
                    "seconds": round(time.perf_counter() - ts, 2),
                }
                print(f"[fc]   {policy:14s} "
                      + " ".join(f"{k}={('None' if v is None else f'{v:.4f}')}"
                                 for k, v in result.scores.items()))

            record["factuality"] = factuality
            record["priors"] = priors_map

            # -- Stage 3: the baselines, on the SAME atoms.
            atoms = _atom_texts(item)
            bl: dict[str, dict] = {}
            for baseline in baselines:
                try:
                    score = baseline.score(atoms, record["output"])
                    bl[baseline.name] = score.to_json()
                    shown = "None" if score.score is None else f"{score.score:.4f}"
                    print(f"[fc]   baseline {baseline.name:30s} {shown}")
                except Exception as e:  # noqa: BLE001
                    # A baseline failure must not lose the item's LCS work, and it
                    # must be recorded as a failure rather than absorbed as a score.
                    print(f"[fc]   baseline {baseline.name} FAILED: {e}")
                    bl[baseline.name] = {"name": baseline.name, "score": None,
                                         "error": str(e)}
            record["baselines"] = bl
            record["seconds"] = round(time.perf_counter() - t0, 2)

            records.append(record)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"[fc] ITEM FAILED {iid}: {e}")
            traceback.print_exc(limit=3)
            records.append({"item_id": iid, "dataset": item.get("dataset"),
                            "model": args.rits_model, "error": str(e)})

        # Persist after every item so an interruption keeps completed work.
        with open(out_path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)

    print(f"\n[fc] done: {n_ok} ok, {n_fail} failed. -> {out_path}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
