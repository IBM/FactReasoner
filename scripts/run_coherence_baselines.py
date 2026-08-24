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

"""Run the coherence baselines over the LCS fixtures.

The baselines are the comparison the Logical Coherence Score is measured against
(see ``fact_reasoner.coherence_baselines``). This driver runs them on the same
atoms the LCS is scored on, so the numbers form an ablation rather than a
separate experiment, and writes one resumable jsonl.

Run::

    # No LLM: the controls only. Useful as a wiring check and to see whether a
    # ladder ordering is reproducible from surface properties alone.
    python scripts/run_coherence_baselines.py --controls-only \\
        --out-dir results/coherence_baselines

    # With the NLI baselines (needs a backend).
    python scripts/run_coherence_baselines.py \\
        --backend rits --model-id llama-3-3-70b-instruct \\
        --out-dir results/coherence_baselines

Each output row is one (item, baseline) pair, so adding a baseline later does not
invalidate the rows already computed; already-present (item, baseline) pairs are
skipped on a re-run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from fact_reasoner.env import load_dotenv, require_env  # noqa: E402

# Credentials live in a gitignored .env at the project root; load them before any
# backend is constructed, or RITSBackend dies on a bare KeyError.
load_dotenv(verbose=True)

from fact_reasoner.coherence_baselines import (  # noqa: E402
    CONTROL_BASELINES,
    DISCOURSE_BASELINES,
    DirectCoherenceRating,
    GEvalCoherence,
    PairwiseNLIContradiction,
    RoscoeSelfConsistency,
    judge_with_variance,
    make_backend_generate,
)

#: Fixture pairs whose *relative* ordering is declared by the fixtures' own
#: notes. Reported as a pass/fail direction check per baseline, which is the only
#: ground truth these standalone fixtures carry -- they have no coherence rating,
#: only an intended direction.
DECLARED_ORDERINGS = (
    # (higher, lower, why)
    (
        "example-2-biography",
        "example-2-biography-contradicted",
        "five planted contradictions",
    ),
    (
        "example-5-renda-K",
        "example-5-renda-S",
        "claim-identical pair; K resolves each tension in place",
    ),
)


def _apply_rits_model(args, repo: str) -> None:
    """Resolve ``--rits-model`` into the backend flags, in place.

    The endpoint URLs in ``configs/rits_models.json`` are long and easy to mistype,
    and a wrong one fails with an opaque auth error rather than a clear "no such
    model", so the name is resolved from the config rather than retyped.
    """
    if not args.rits_model:
        return
    path = os.path.join(repo, "configs", "rits_models.json")
    with open(path) as f:
        entries = json.load(f)
    by_name = {e["name"]: e for e in entries}
    if args.rits_model not in by_name:
        raise SystemExit(
            f"Unknown --rits-model {args.rits_model!r}. Available: "
            f"{', '.join(sorted(by_name))}"
        )
    entry = by_name[args.rits_model]
    args.backend = entry.get("backend", "rits")
    args.model_id = entry["model_id"]
    args.base_url = entry.get("base_url")
    print(f"[config] {args.rits_model} -> {args.model_id} @ {args.base_url}")


def _load_fixtures(data_dir: str) -> list[dict]:
    """Load every LCS fixture from ``data_dir``, sorted by id."""
    items = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(data_dir, fname)) as f:
            item = json.load(f)
        if item.get("atoms") and item.get("response"):
            items.append(item)
    return items


class _SeededJudge:
    """Wraps a judge so ``score`` runs it ``seeds`` times and reports the spread.

    Kept as an adapter rather than folded into the judge classes because a single
    call is the natural unit for a judge and repetition is a *reporting* decision:
    the driver always wants the spread, an interactive caller may not.
    """

    def __init__(self, judge, seeds: int):
        self.judge = judge
        self.seeds = seeds
        self.name = judge.name

    def score(self, atoms, response):
        return judge_with_variance(self.judge, atoms, response, seeds=self.seeds)


def _build_baselines(args, backend):
    """Instantiate the baselines requested by ``args``."""
    # The model-free baselines always run: they need no backend, and the controls
    # are the gate that says whether a ladder ordering is evidence at all.
    baselines = list(CONTROL_BASELINES) + list(DISCOURSE_BASELINES)
    if args.controls_only:
        return baselines

    from fact_reasoner.core.nli import NLIExtractor

    nli = NLIExtractor(backend, nli_method=args.nli_method)
    # Throttle settings are applied inside the baselines' batched call path; None
    # means "use the pipeline-wide default" (1500/min), so the live run matches the
    # factuality pipeline's budget without restating it.
    throttle = {}
    if args.rate_per_minute is not None:
        throttle["rate_per_minute"] = args.rate_per_minute
    if args.max_concurrency is not None:
        throttle["max_concurrency"] = args.max_concurrency
    baselines += [
        PairwiseNLIContradiction(nli, show_progress=True, throttle=throttle),
        PairwiseNLIContradiction(
            nli, soft=True, show_progress=True, throttle=throttle
        ),
        RoscoeSelfConsistency(nli, show_progress=True, throttle=throttle),
        # The two ablation arms: they separate "the max saturates" from "it is
        # forward-only" from "it is untyped". Without these, a single losing
        # number would not say which difference mattered.
        RoscoeSelfConsistency(nli, aggregate="mean", throttle=throttle),
        RoscoeSelfConsistency(nli, symmetric=True, throttle=throttle),
    ]

    if not args.no_judges:
        generate = _make_generate(backend, args)
        baselines += [
            _SeededJudge(GEvalCoherence(generate), args.judge_seeds),
            _SeededJudge(DirectCoherenceRating(generate), args.judge_seeds),
        ]
    return baselines


def _make_generate(backend, args):
    """Thin alias for the package's shared generate factory."""
    return make_backend_generate(backend)


def _already_done(path: str) -> set[tuple[str, str]]:
    """The (item id, baseline name) pairs already present in ``path``."""
    done: set[tuple[str, str]] = set()
    if not os.path.isfile(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("item_id") and row.get("name"):
                done.add((row["item_id"], row["name"]))
    return done


def _report_direction_checks(rows: list[dict]) -> int:
    """Print the declared-direction checks. Returns the number of failures.

    A baseline that gets a direction wrong is either mis-implemented or is
    genuinely blind to that defect -- both worth knowing before a ladder run, and
    much cheaper to discover here.
    """
    # Compare on the *raw* metric where one is recorded. A clamped score can hide
    # an ordering: DiscoScore's EntityGraph is an unbounded sum, so two responses
    # can both clamp to 1.0 and read as a tie while the underlying values differ
    # by a factor of two. Reporting the clamped tie would understate how wrong the
    # baseline is.
    by_key = {}
    for r in rows:
        diag = r.get("diagnostics") or {}
        raw = diag.get("raw")
        by_key[(r["item_id"], r["name"])] = {
            "score": r.get("score"),
            "cmp": raw if isinstance(raw, (int, float)) else r.get("score"),
            "clamped": bool(diag.get("clamped")),
        }
    names = sorted({r["name"] for r in rows})
    failures = 0

    print("\nDeclared-direction checks")
    print("-" * 78)
    for higher, lower, why in DECLARED_ORDERINGS:
        print(f"\n  {higher}  >  {lower}")
        print(f"  ({why})")
        for name in names:
            h_rec, l_rec = by_key.get((higher, name)), by_key.get((lower, name))
            if h_rec is None or l_rec is None:
                print(f"    {name:28s}  n/a (missing row)")
                continue
            hi, lo = h_rec["cmp"], l_rec["cmp"]
            if hi is None or lo is None:
                print(f"    {name:28s}  n/a (abstained)")
                continue
            # Round to reporting precision first: scores that print identically are
            # a tie. Otherwise a float difference in an intermediate term shows up
            # as a "correct" ordering between two visually equal values.
            if round(hi, 6) == round(lo, 6):
                verdict = "FLAT"
            elif hi > lo:
                verdict = "OK"
            else:
                verdict = "WRONG"
                failures += 1
            clamped = h_rec["clamped"] or l_rec["clamped"]
            note = "  (raw; score clamped)" if clamped else ""
            print(f"    {name:28s}  {hi:.4f} vs {lo:.4f}   {verdict}{note}")
    print()
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        "--data-dir",
        default=os.path.join(repo, "data", "lcs"),
        help="Directory of LCS fixture JSONs (default: data/lcs).",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(repo, "results", "coherence_baselines"),
        help="Output directory for the results jsonl.",
    )
    parser.add_argument(
        "--controls-only",
        action="store_true",
        help="Run only the model-free controls. No backend or credentials needed.",
    )
    parser.add_argument(
        "--nli-method",
        default="logprobs",
        choices=("logprobs", "simbauq"),
        help="How the NLI extractor estimates label probabilities.",
    )
    parser.add_argument(
        "--rits-model",
        default=None,
        help="Name from configs/rits_models.json (e.g. gpt-oss-120b-a100, "
        "llama-3.3-70b-instruct). Resolves --backend/--model-id/--base-url from "
        "that entry, so the endpoint URL is never retyped by hand. Requires "
        "RITS_API_KEY in the environment.",
    )
    parser.add_argument(
        "--rate-per-minute",
        type=int,
        default=None,
        help="Requests-per-minute ceiling for the pairwise NLI batches. Defaults "
        "to the pipeline-wide 1500 (fact_reasoner.utils.MAX_REQUESTS_PER_MINUTE).",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="In-flight call ceiling. Defaults to the pipeline-wide value.",
    )
    parser.add_argument(
        "--no-judges",
        action="store_true",
        help="Skip the LLM judges (they cost judge-seeds calls per item).",
    )
    parser.add_argument(
        "--judge-seeds",
        type=int,
        default=5,
        help="Judge runs per item. Five is the floor: fewer cannot show a spread.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score at most this many fixtures (smoke tests).",
    )
    # Reuse the shared backend flags so this driver selects a model with exactly
    # the same options as `fact-reasoner` and `fact-reasoner-lcs`.
    from fact_reasoner.cli import _add_backend_args

    _add_backend_args(parser, default_kind="rits")
    args = parser.parse_args()
    _apply_rits_model(args, repo)
    if not args.controls_only and (args.backend or "rits") == "rits":
        require_env(
            "RITS_API_KEY",
            hint="RITS endpoints need it; --controls-only runs without any backend.",
        )

    items = _load_fixtures(args.data_dir)
    if args.limit:
        items = items[: args.limit]
    if not items:
        print(f"No usable fixtures in {args.data_dir!r}.", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "baselines.jsonl")
    done = _already_done(out_path)
    if done:
        print(f"Resuming: {len(done)} (item, baseline) rows already present.")

    rows: list[dict] = []
    if os.path.isfile(out_path):
        with open(out_path) as f:
            rows = [json.loads(li) for li in f if li.strip()]

    def run_all(backend):
        baselines = _build_baselines(args, backend)
        todo = sum(
            1 for i in items for b in baselines if (i["id"], b.name) not in done
        )
        print(
            f"Scoring {len(items)} fixtures x {len(baselines)} baselines "
            f"({todo} to compute)."
        )
        with open(out_path, "a") as out:
            for item in items:
                atoms = [a["text"] for a in item["atoms"]]
                for baseline in baselines:
                    if (item["id"], baseline.name) in done:
                        continue
                    result = baseline.score(atoms, item["response"])
                    row = {
                        "item_id": item["id"],
                        # Recorded so a report can prove every baseline saw the
                        # same decomposition; without it the comparison is not an
                        # ablation.
                        "num_atoms": len(atoms),
                        **result.to_json(),
                    }
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    rows.append(row)
                    shown = "None" if result.score is None else f"{result.score:.4f}"
                    print(f"  {item['id']:36s} {baseline.name:28s} {shown}")

    if args.controls_only:
        run_all(None)
    else:
        from fact_reasoner.cli import _backend_context

        with _backend_context(args) as backend:
            run_all(backend)

    print(f"\nWrote {out_path}")
    failures = _report_direction_checks(rows)
    if failures:
        print(
            f"{failures} declared-direction check(s) went the wrong way. That is a "
            f"result, not necessarily a bug -- see each baseline's docstring for "
            f"which defects it is expected to miss."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
