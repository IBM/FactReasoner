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

"""Score the coherence baselines against the ladder corpus' declared orderings.

The ladder is the paper's controlled contrast: each family is five responses over
one claim set, and the family declares how consecutive rungs must order. Because
every rung carries the same number of claims, the length and pair-count confounds
that dominate the hand-authored fixtures cancel here -- which makes this the
comparison that can actually separate a coherence measure from a size proxy.

Which constraints apply to a baseline
-------------------------------------
The corpus states constraints per *readout* (``mean_marginal``, ``consistency``,
``log_partition``), because the LCS reports several. A baseline has one score, so
only the readout-independent content of each constraint transfers:

* **C1** -- strict increase between consecutive rungs. Applies directly; the union
  of the declared pairs is used, so a baseline is asked the same four questions the
  LCS is asked.
* **C3** -- endpoint separation, rung 0 below rung 4. Applies directly.
* **C2** -- a *predicted* decrease or invariance at the concession rung. Deliberately
  **excluded**: it is a prediction about the internals of specific readouts (see the
  paper's belief-versus-event-activity argument), not about coherence ordering, and
  scoring a baseline against it would credit or punish it for a property it makes no
  claim about.

So each family contributes four C1 assertions plus one C3 assertion, and a baseline
is scored on the fraction it satisfies across families.

Run::

    python scripts/run_ladder_baselines.py \\
        --data-dir data/locobench-claude-5-test \\
        --rits-model gpt-oss-120b-a100 \\
        --out-dir results/ladder_baselines
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

#: Tolerance for calling two rung scores equal. A baseline that moves by less than
#: this is treated as flat rather than as satisfying a strict-increase constraint:
#: crediting a 1e-9 difference would let floating-point noise pass for judgement.
TIE_TOLERANCE = 1e-6


class _SeededJudge:
    """Run a judge ``seeds`` times per item and report mean + spread."""

    def __init__(self, judge, seeds: int):
        self.judge = judge
        self.seeds = seeds
        self.name = judge.name

    def score(self, atoms, response):
        return judge_with_variance(self.judge, atoms, response, seeds=self.seeds)


def _load(data_dir: str):
    """Return ``(items_by_family, families)``."""
    with open(os.path.join(data_dir, "items.jsonl")) as f:
        items = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(data_dir, "families.json")) as f:
        manifest = json.load(f)

    by_family: dict[str, dict[int, dict]] = {}
    for item in items:
        exp = item.get("expected") or {}
        fid, rung = exp.get("family_id"), exp.get("rung_index")
        if fid is None or rung is None:
            continue
        by_family.setdefault(fid, {})[rung] = item
    return by_family, manifest["families"]


def _assertions(family: dict) -> list[tuple[str, int, int]]:
    """The readout-independent assertions a baseline can be scored on.

    Returns:
        ``(constraint_class, lower_rung, higher_rung)`` triples, each asserting
        that the higher rung must score strictly above the lower one.
    """
    out: list[tuple[str, int, int]] = []
    for constraint in family.get("ordering_constraints") or []:
        cls = constraint.get("class")
        if cls == "C1":
            for pair in sorted({tuple(p["pair"]) for p in constraint.get("pairs", [])}):
                out.append(("C1", pair[0], pair[1]))
        elif cls == "C3":
            for pair in constraint.get("required") or []:
                out.append(("C3", pair[0], pair[1]))
        # C2 is readout-specific; see the module docstring.
    return out


def _build_baselines(args, backend):
    baselines = list(CONTROL_BASELINES) + list(DISCOURSE_BASELINES)
    if args.controls_only:
        return baselines

    from fact_reasoner.core.nli import NLIExtractor

    nli = NLIExtractor(backend, nli_method=args.nli_method)
    throttle = {}
    if args.rate_per_minute is not None:
        throttle["rate_per_minute"] = args.rate_per_minute

    baselines += [
        PairwiseNLIContradiction(nli, show_progress=True, throttle=throttle),
        RoscoeSelfConsistency(nli, show_progress=True, throttle=throttle),
        RoscoeSelfConsistency(nli, aggregate="mean", throttle=throttle),
    ]
    if not args.no_judges:
        generate = make_backend_generate(backend)
        baselines += [
            _SeededJudge(GEvalCoherence(generate), args.judge_seeds),
            _SeededJudge(DirectCoherenceRating(generate), args.judge_seeds),
        ]
    return baselines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument(
        "--data-dir", default=os.path.join(repo, "data", "locobench-claude-5-test")
    )
    parser.add_argument(
        "--out-dir", default=os.path.join(repo, "results", "ladder_baselines")
    )
    parser.add_argument("--controls-only", action="store_true")
    parser.add_argument("--no-judges", action="store_true")
    parser.add_argument("--judge-seeds", type=int, default=5)
    parser.add_argument(
        "--nli-method", default="logprobs", choices=("logprobs", "simbauq")
    )
    parser.add_argument("--rate-per-minute", type=int, default=None)
    parser.add_argument("--rits-model", default=None)
    from fact_reasoner.cli import _add_backend_args

    _add_backend_args(parser, default_kind="rits")
    args = parser.parse_args()

    if args.rits_model:
        with open(os.path.join(repo, "configs", "rits_models.json")) as f:
            entries = {e["name"]: e for e in json.load(f)}
        if args.rits_model not in entries:
            raise SystemExit(
                f"Unknown --rits-model. Have: {', '.join(sorted(entries))}"
            )
        entry = entries[args.rits_model]
        args.backend, args.model_id, args.base_url = (
            entry.get("backend", "rits"),
            entry["model_id"],
            entry.get("base_url"),
        )
        print(f"[config] {args.rits_model} -> {args.model_id}")

    if not args.controls_only and (args.backend or "rits") == "rits":
        require_env("RITS_API_KEY")

    by_family, families = _load(args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "ladder_scores.jsonl")

    done = set()
    if os.path.isfile(out_path):
        with open(out_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["family_id"], r["rung"], r["name"]))
        print(f"Resuming: {len(done)} rows present.")

    def run(backend):
        baselines = _build_baselines(args, backend)
        with open(out_path, "a") as out:
            for fid, rungs in sorted(by_family.items()):
                for rung, item in sorted(rungs.items()):
                    atoms = [a["text"] for a in item["atoms"]]
                    for baseline in baselines:
                        if (fid, rung, baseline.name) in done:
                            continue
                        res = baseline.score(atoms, item["response"])
                        row = {
                            "family_id": fid,
                            "rung": rung,
                            "rung_name": (item.get("expected") or {}).get("rung_name"),
                            "num_atoms": len(atoms),
                            **res.to_json(),
                        }
                        out.write(json.dumps(row) + "\n")
                        out.flush()
                        shown = "None" if res.score is None else f"{res.score:.4f}"
                        print(f"  {fid} r{rung} {baseline.name:26s} {shown}")

    if args.controls_only:
        run(None)
    else:
        from fact_reasoner.cli import _backend_context

        with _backend_context(args) as backend:
            run(backend)

    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
