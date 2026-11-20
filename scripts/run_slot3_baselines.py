"""Score the coherence baselines on a slot3 corpus, once per unique item.

slot3 ladders share items (a CONTROL ladder's rung 0 is its family's primary rung 1), so
the unit of scoring is the item, not the (family, rung) pair the v1 driver
(`run_ladder_baselines.py`) uses. Ladders are assembled from these scores by
`scripts/arr_slot3_eval.py`.

Groups, run as separate processes so the slow ones overlap (``--group``):

* ``controls``  claim count, response length, edit distance (no model)
* ``discourse`` DiscoScore RC/LC and the entity graph (local encoder)
* ``nli``       pairwise NLI contradiction and ROSCOE self-consistency (max and mean),
                over the CLAIMS, with verdicts cached per claim pair: every item of a
                family has the same claim list, so these are constant within a family
* ``judge:<model>:<geval|direct>``  an LLM judge, ``--seeds`` runs per item (mean and
                spread), with ``<model>`` a name in ``configs/rits_models.json``

Run::

    python scripts/run_slot3_baselines.py --group controls
    python scripts/run_slot3_baselines.py --group judge:gpt-6-sol:geval
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from fact_reasoner.env import load_dotenv  # noqa: E402

load_dotenv(verbose=False)
if not os.environ.get("OPENAI_API_KEY") and os.environ.get("ANTHROPIC_AUTH_TOKEN"):
    os.environ["OPENAI_API_KEY"] = os.environ["ANTHROPIC_AUTH_TOKEN"]


def _backend(name: str):
    from fact_reasoner.backends import build_backend
    from fact_reasoner.locoeval.models import load_model_specs

    spec = load_model_specs(os.path.join(REPO, "configs", "rits_models.json"))[name]
    return build_backend(spec.backend, model_id=spec.model_id, base_url=spec.base_url)


def _baselines(group: str, args):
    from fact_reasoner import coherence_baselines as cb

    if group == "controls":
        return list(cb.CONTROL_BASELINES)
    if group == "discourse":
        return list(cb.DISCOURSE_BASELINES)
    if group == "nli":
        from fact_reasoner.core.nli import NLIExtractor
        from fact_reasoner.core.nli_cache import NLIVerdictCache

        nli = NLIExtractor(_backend(args.nli_model), nli_method="logprobs")
        cache = "nli_cache" if args.shard is None else f"nli_cache.shard{args.shard}"
        throttle = {"verdict_cache": NLIVerdictCache(os.path.join(args.out_dir, cache))}
        return [
            cb.PairwiseNLIContradiction(nli, throttle=throttle),
            cb.RoscoeSelfConsistency(nli, throttle=throttle),
            cb.RoscoeSelfConsistency(nli, aggregate="mean", throttle=throttle),
        ]
    if group.startswith("judge:"):
        _, model, kind = group.split(":")
        gen = cb.make_backend_generate(_backend(model))
        judge = {"geval": cb.GEvalCoherence, "direct": cb.DirectCoherenceRating}[kind](gen)

        class Seeded:
            name = f"{judge.name}@{model}"

            def score(self, atoms, response):
                return cb.judge_with_variance(judge, atoms, response, seeds=args.seeds)

        return [Seeded()]
    raise SystemExit(f"unknown group {group!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=os.path.join(REPO, "data", "locobench-slot3-pilot"))
    ap.add_argument("--out-dir", default=os.path.join(REPO, "results", "arr2027_slot3",
                                                      "baselines"))
    ap.add_argument("--group", required=True)
    ap.add_argument("--nli-model", default="gpt-oss-120b")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--shard", type=int, default=None, help="This process's shard index.")
    ap.add_argument("--of", type=int, default=None, help="Number of shards.")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(os.path.join(args.data_dir, "items.jsonl"))]
    os.makedirs(args.out_dir, exist_ok=True)
    stem = args.group.replace(":", "__")
    suffix = f".shard{args.shard}" if args.shard is not None else ""
    out_path = os.path.join(args.out_dir, stem + suffix + ".jsonl")
    # Done items are read from EVERY file of this group, so shards started after a
    # sequential run (or each other) do not redo its work.
    done = set()
    for f in os.listdir(args.out_dir):
        if f.startswith(stem + ".") and f.endswith(".jsonl"):
            for line in open(os.path.join(args.out_dir, f)):
                if line.strip():
                    r = json.loads(line)
                    if r.get("score") is not None:
                        done.add((r["item_id"], r["name"]))
    if args.shard is not None:
        # Shard by FAMILY, not by item: a family's items share one claim list, so the NLI
        # baselines' cached verdicts are computed once per family only if a family stays in
        # one shard. Each shard also gets its own NLI cache directory (no SQLite contention).
        fams = sorted({it["expected"]["family_id"] for it in items})
        mine = {f for i, f in enumerate(fams) if i % args.of == args.shard}
        items = [it for it in items if it["expected"]["family_id"] in mine]
    baselines = _baselines(args.group, args)
    with open(out_path, "a") as out:
        for it in items:
            atoms = [a["text"] for a in sorted(it["atoms"], key=lambda a: a["pos"])]
            for b in baselines:
                if (it["id"], b.name) in done:
                    continue
                res = b.score(atoms, it["response"])
                row = {"item_id": it["id"], "family_id": it["expected"]["family_id"],
                       **res.to_json()}
                row["name"] = b.name
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(f"{it['id']} {b.name} {res.score}", flush=True)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
