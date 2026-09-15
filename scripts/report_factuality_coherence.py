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

"""Summarize the factuality+coherence runs into report tables (and LaTeX).

Reads the jsonl written by ``scripts/run_factuality_coherence.py`` and reports, per
model and per dataset:

* the four LCS readouts under each pair policy (windowed / bidirectional);
* the discriminative power of each measure -- its spread across items. A measure
  that returns the same number for every response cannot rank anything, so this
  is reported as ``distinct`` values and stdev alongside the mean. This is the
  column that separates a working measure from a saturated one, and it is why the
  judges' means are not reported on their own.
* stage 1's factuality marginals, whose spread is what makes the two-stage model
  more than a coherence-only score;
* agreement with the human S/NS atom labels where the corpus carries them (bio),
  as an external check that stage 1 is measuring support rather than fluency.

Usage:
    python scripts/report_factuality_coherence.py
    python scripts/report_factuality_coherence.py --latex > section.tex
"""

import argparse
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICIES = ("windowed", "bidirectional")
READOUTS = ("mean_marginal", "consistency", "reified", "log_partition")


def _load(out_dir: str) -> list[dict]:
    """Load every run record, skipping the error rows."""
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, "fc_*.jsonl"))):
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("error"):
                    continue
                rows.append(r)
    return rows


def _describe(values: list[float]) -> dict:
    """Mean/stdev/range/distinct for one measure's values across items.

    ``distinct`` is the load-bearing statistic: a measure with one distinct value
    is a constant, and a constant cannot order responses no matter what its mean is.
    """
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None,
                "distinct": 0}
    return {
        "n": len(vals),
        "mean": st.mean(vals),
        "sd": st.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
        "distinct": len({round(v, 4) for v in vals}),
    }


def _fmt(d: dict) -> str:
    if not d["n"]:
        return f"{'--':>8}"
    return (f"{d['mean']:.4f} +-{d['sd']:.4f}  [{d['min']:.4f},{d['max']:.4f}]  "
            f"d={d['distinct']:3d}/{d['n']:3d}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir",
                    default=os.path.join(REPO, "results", "factuality_coherence"))
    ap.add_argument("--latex", action="store_true",
                    help="Emit a LaTeX table instead of the text report.")
    args = ap.parse_args()

    rows = _load(args.out_dir)
    if not rows:
        raise SystemExit(f"No records in {args.out_dir!r}.")

    # An arm is (model, prior source), not just the model: the same model appears in
    # both the two-stage and the coherence-only runs, and merging them would average
    # two different experiments into one meaningless column. Records do not carry an
    # explicit prior-source field, so it is inferred from the signature the
    # coherence-only arm necessarily has -- a uniform 0.5 prior on every claim and no
    # factuality block.
    def _prior_source(r: dict) -> str:
        pri = list((r.get("priors") or {}).values())
        if r.get("factuality") is None and pri and {round(p, 6) for p in pri} == {0.5}:
            return "uniform"
        return "factreasoner"

    by_model = defaultdict(list)
    for r in rows:
        by_model[f"{r.get('model') or '?'} [{_prior_source(r)} priors]"].append(r)

    print(f"# {len(rows)} items across {len(by_model)} arm(s) "
          f"(arm = model x prior source)\n")

    for model, items in sorted(by_model.items()):
        ds = defaultdict(int)
        for r in items:
            ds[r.get("dataset")] += 1
        print(f"== {model}  n={len(items)}  "
              f"({', '.join(f'{k}:{v}' for k, v in sorted(ds.items()))})")
        atoms = [r["num_atoms"] for r in items]
        print(f"   atoms/item mean={st.mean(atoms):.1f} "
              f"median={st.median(atoms)} max={max(atoms)}")
        secs = [r["seconds"] for r in items if r.get("seconds")]
        if secs:
            print(f"   seconds/item mean={st.mean(secs):.0f} total={sum(secs)/3600:.1f} h")

        # -- Stage 1: the factuality marginals.
        pri = [p for r in items for p in (r.get("priors") or {}).values()]
        print(f"\n   stage 1 (FactReasoner marginals over {len(pri)} atoms)")
        print(f"     {_fmt(_describe(pri))}")

        # External check: do the marginals separate the human S from NS atoms?
        s_vals, ns_vals = [], []
        for r in items:
            labels = r.get("gold_atom_labels") or []
            priors = r.get("priors") or {}
            # priors are keyed a0..aN in atom order, which is the label order.
            for i, lab in enumerate(labels):
                v = priors.get(f"a{i}")
                if v is None:
                    continue
                if lab == "S":
                    s_vals.append(v)
                elif lab == "NS":
                    ns_vals.append(v)
        if s_vals and ns_vals:
            print(f"     vs human labels: S   {_fmt(_describe(s_vals))}")
            print(f"                      NS  {_fmt(_describe(ns_vals))}")
            print(f"                      separation (mean S - mean NS) = "
                  f"{st.mean(s_vals) - st.mean(ns_vals):+.4f}")

        # -- Stage 2: the readouts, per policy.
        for policy in POLICIES:
            key = f"lcs_{policy}"
            present = [r for r in items if key in r]
            if not present:
                continue
            rels = [r[key]["num_relations"] for r in present]
            print(f"\n   stage 2 [{policy}]  relations/item mean={st.mean(rels):.1f} "
                  f"max={max(rels)}")
            for readout in READOUTS:
                vals = [r[key]["scores"].get(readout) for r in present]
                print(f"     {readout:14s} {_fmt(_describe(vals))}")

        # -- Stage 3: the baselines.
        names = sorted({n for r in items for n in (r.get("baselines") or {})})
        if names:
            print("\n   baselines")
            for name in names:
                vals = [(r.get("baselines") or {}).get(name, {}).get("score")
                        for r in items]
                print(f"     {name:26s} {_fmt(_describe(vals))}")
        print()

    if args.latex:
        _emit_latex(by_model)
    return 0


def _emit_latex(by_model) -> None:
    """Emit the main results table in the paper's style."""
    print("\n%% ---- generated by scripts/report_factuality_coherence.py ----")
    print(r"\begin{table}[t]")
    print(r"\centering\small")
    print(r"\caption{The two-stage model on the factuality corpora. \textbf{d} is the "
          r"number of distinct values a measure takes across the items: a measure with "
          r"$d=1$ is a constant and cannot rank responses whatever its mean.}")
    print(r"\label{tab:factuality-coherence}")
    print(r"\begin{tabular}{@{}llrrrr@{}}")
    print(r"\toprule")
    print(r"Model & Measure & Mean & SD & Range & d \\")
    print(r"\midrule")
    for model, items in sorted(by_model.items()):
        first = True
        for policy in POLICIES:
            key = f"lcs_{policy}"
            present = [r for r in items if key in r]
            if not present:
                continue
            for readout in READOUTS:
                d = _describe([r[key]["scores"].get(readout) for r in present])
                if not d["n"]:
                    continue
                label = model.replace("_", r"\_") if first else ""
                first = False
                name = f"$\\LCS$ {readout.replace('_', ' ')} ({policy})"
                print(f"{label} & {name} & {d['mean']:.4f} & {d['sd']:.4f} & "
                      f"[{d['min']:.3f},{d['max']:.3f}] & {d['distinct']} \\\\")
        names = sorted({n for r in items for n in (r.get("baselines") or {})})
        for name in names:
            d = _describe([(r.get("baselines") or {}).get(name, {}).get("score")
                           for r in items])
            if not d["n"]:
                continue
            print(f" & \\textsc{{{name.replace('_', ' ')}}} & {d['mean']:.4f} & "
                  f"{d['sd']:.4f} & [{d['min']:.3f},{d['max']:.3f}] & "
                  f"{d['distinct']} \\\\")
        print(r"\midrule")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    raise SystemExit(main())
