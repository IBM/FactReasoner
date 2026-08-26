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

"""Score generated passages with BOTH the LCS readouts and the prompt-based LLM judges.

This is the prompt-iteration harness for the incoherent corpus. The corpus is only
useful if two things hold at once, and they pull against each other:

* **LCS must see the defect** -- the relation miner has to extract at least one conflict
  edge. Measured failure: a smooth paraphrase that read well mined ZERO conflict edges and
  scored ``mean_marginal`` 0.634 with ``log_partition`` 1.000, i.e. it fooled the metric
  as thoroughly as the judge. Surface smoothing hides the conflict from *both*.
* **A prompt-based judge must not** -- otherwise the corpus proves nothing, since a metric
  that only wins where a judge already wins is not evidence. Measured: the v1
  ``assert-both-then-negate`` corpus scores ``judge_direct`` 0.25 on Claude, because the
  passage announces its own contradiction.

So the verdict is a conjunction, not a single score. ``PASS`` requires a mined conflict
edge AND a low ``mean_marginal`` AND a fooled judge.

Judges are model-dependent and this reports every combination rather than picking a
flattering one. Measured on one passage: ``judge_geval`` on ``llama-3-3-70b`` gave 0.903
(fooled) while ``judge_direct`` on the same model gave 0.011 (not fooled at all).

**Cost warning, measured.** LCS scoring dominates and scales with atom count, which
scales with passage length. The v1 corpus averaged 602 chars -> 9 atoms -> ~163 s per
instance. The v2 ``invented-subtle`` passages are ~2.2x longer (1314 chars -> **18
atoms**), and mining is windowed over pairs while the ``consistency``/``reified`` readouts
add one auxiliary variable per edge, so per-instance cost grows much faster than linearly:
a 3-instance run left the process blocked on network I/O with ~0 CPU for 40 minutes. Use
``--no-lcs`` while iterating on the prompt (judges alone are fast and are the thing being
tuned), score a small sample separately, and prefer ``--methods mean_marginal`` style
narrowing if the aux-variable readouts are not needed.

Usage::

    # Iterate on a couple of instances (the intended loop).
    python scripts/eval_incoherent_judges.py \\
        --input results/incoherent-v2/conflictbank-invented-claude.json --limit 3 \\
        --merlin-path /path/to/merlin

    # Judges only, no Merlin needed (fast prompt iteration).
    python scripts/eval_incoherent_judges.py --input <file> --limit 3 --no-lcs

    # Compare a control corpus the same way.
    python scripts/eval_incoherent_judges.py --input results/.../control-n20.json --limit 20
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
import traceback

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
)

from fact_reasoner.env import load_dotenv  # noqa: E402

# Credentials before any backend is constructed, or RITSBackend dies on a bare KeyError.
load_dotenv(verbose=True)

from fact_reasoner.coherence_baselines import (  # noqa: E402
    DirectCoherenceRating,
    GEvalCoherence,
    judge_with_variance,
    make_backend_generate,
)

GATEWAY_BASE_URL = "https://ete-litellm.bx.cloud9.ibm.com/v1"
RITS_GPTOSS = (
    "openai/gpt-oss-120b-a100",
    "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/gpt-oss-120b-a100",
)

#: Judge models to run. ``llama-3-3-70b-instruct`` is the judge behind the paper's
#: recorded baselines (results/coherence_baselines/baselines.jsonl), so it is the one a
#: claim about "fooling a judge baseline" has to be made against. Claude is included as
#: the stronger adversary -- it is NOT fooled by anything measured so far, and saying so
#: is part of the result.
JUDGE_MODELS = {
    "llama": {"kind": "rits", "model_id": "llama-3-3-70b-instruct", "base_url": None},
    "claude": {
        "kind": "openai",
        "model_id": "aws/claude-opus-5",
        "base_url": GATEWAY_BASE_URL,
    },
}

#: A judge is "fooled" at or above this normalized rating (4 of 5 on the 1-5 scale).
FOOLED_AT = 0.75

#: ``mean_marginal`` at or below this counts as "LCS sees a defect". The v1 corpus
#: measured 0.539 and its coherent control 0.737, so this sits between them.
LCS_LOW_AT = 0.65


def _bridge_openai_key() -> None:
    """Expose the IBM gateway token to the OpenAI SDK (see gen_incoherent_responses)."""
    if not os.environ.get("OPENAI_API_KEY") and os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        os.environ["OPENAI_API_KEY"] = os.environ["ANTHROPIC_AUTH_TOKEN"]
        print("[config] OPENAI_API_KEY <- ANTHROPIC_AUTH_TOKEN (gateway token)")


def load_responses(path: str, limit: int | None) -> list[dict]:
    """Load ``{id, response}`` rows from a generated corpus (JSON or JSONL)."""
    if path.endswith(".jsonl"):
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        with open(path) as f:
            payload = json.load(f)
        rows = payload["records"] if isinstance(payload, dict) else payload
    rows = [r for r in rows if r.get("response")]
    if not rows:
        raise SystemExit(f"{path}: no records with a 'response' field.")
    return rows[: limit or len(rows)]


def judge_one(response: str, judges: dict, seeds: int) -> dict:
    """Run every (judge, model) pair on one response.

    Returns ``{"<model>.<judge>": {score, sd, ratings, ...}}``. A judge that cannot parse
    a rating returns ``score=None``; that is a missing measurement, never a zero, and it
    is reported as such.
    """
    out: dict[str, dict] = {}
    for model_name, judge_list in judges.items():
        for judge in judge_list:
            key = f"{model_name}.{judge.name}"
            try:
                r = judge_with_variance(judge, [], response, seeds=seeds)
                out[key] = {
                    "score": r.score,
                    "sd": r.diagnostics.get("sd"),
                    "ratings": r.diagnostics.get("ratings"),
                    "abstained": r.diagnostics.get("abstained"),
                }
            except Exception as e:  # one judge failing must not lose the other rows
                out[key] = {"score": None, "error": f"{type(e).__name__}: {e}"}
    return out


def verdict(lcs: dict | None, judged: dict, *, target: str) -> tuple[str, list[str]]:
    """Decide PASS/FAIL for one instance and say which conditions failed.

    Three conjuncts, each guarding a measured failure mode:
      1. a mined conflict edge -- else the miner never saw the defect;
      2. ``mean_marginal <= LCS_LOW_AT`` -- else LCS calls it coherent;
      3. the target judge at or above :data:`FOOLED_AT` -- else the corpus is no harder
         than the v1 one.
    """
    reasons = []
    if lcs is not None:
        if lcs["n_conflict_edges"] < 1:
            reasons.append("no conflict edge mined")
        mm = lcs["scores"].get("mean_marginal")
        if mm is None or mm > LCS_LOW_AT:
            reasons.append(f"mean_marginal {mm} not <= {LCS_LOW_AT}")
    tgt = judged.get(target, {}).get("score")
    if tgt is None:
        reasons.append(f"{target} abstained")
    elif tgt < FOOLED_AT:
        reasons.append(f"{target} {tgt:.3f} < {FOOLED_AT} (judge not fooled)")
    return ("PASS" if not reasons else "FAIL"), reasons


def main(argv: list[str] | None = None) -> int:
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, help="Generated corpus (.json or .jsonl).")
    p.add_argument("--output", default=None, help="Where to write the report JSON.")
    p.add_argument("--limit", type=int, default=3, help="Instances to evaluate.")
    p.add_argument(
        "--judge-seeds",
        type=int,
        default=5,
        help="Repeats per judge. 5 is the floor: fewer cannot show a spread, and the "
        "spread is part of the result (recorded sd reaches 0.187).",
    )
    p.add_argument(
        "--judge-models",
        default="llama,claude",
        help="Comma-separated subset of: " + ", ".join(sorted(JUDGE_MODELS)),
    )
    p.add_argument(
        "--target",
        default="llama.judge_geval",
        help="The (model.judge) pair the PASS verdict is measured against. Default "
        "llama.judge_geval -- the judge behind the paper's recorded baselines.",
    )
    p.add_argument(
        "--no-lcs",
        action="store_true",
        help="Skip mining/Merlin and report judges only. Much faster for prompt "
        "iteration (LCS scoring is ~163 s/instance).",
    )
    p.add_argument(
        "--merlin-path",
        default=os.environ.get("MERLIN_PATH"),
        help="Merlin executable (or set MERLIN_PATH). Required unless --no-lcs.",
    )
    args = p.parse_args(argv)

    if not args.no_lcs and not args.merlin_path:
        raise SystemExit(
            "LCS scoring needs Merlin: pass --merlin-path, set MERLIN_PATH, or use "
            "--no-lcs to report judges only."
        )
    models = [m.strip() for m in args.judge_models.split(",") if m.strip()]
    unknown = [m for m in models if m not in JUDGE_MODELS]
    if unknown:
        raise SystemExit(
            f"Unknown --judge-models {unknown}. Available: {sorted(JUDGE_MODELS)}"
        )

    rows = load_responses(args.input, args.limit)
    _bridge_openai_key()

    from fact_reasoner.backends import build_backend

    # One backend per judge model; both judges on a model share its generate closure.
    judges: dict[str, list] = {}
    for name in models:
        spec = JUDGE_MODELS[name]
        b = build_backend(
            spec["kind"], model_id=spec["model_id"], base_url=spec["base_url"]
        )
        gen = make_backend_generate(b)
        judges[name] = [GEvalCoherence(gen), DirectCoherenceRating(gen)]
        print(f"[judge] {name} -> {spec['model_id']}")

    score_response = None
    if not args.no_lcs:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "gen_incoherent", os.path.join(repo, "scripts", "gen_incoherent_responses.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        score_response = mod.score_response
        miner_backend = build_backend(
            "rits", model_id=RITS_GPTOSS[0], base_url=RITS_GPTOSS[1]
        )
        print(f"[miner] {RITS_GPTOSS[0]}")

    print(
        f"\n[eval] {len(rows)} instance(s) x {sum(len(v) for v in judges.values())} "
        f"judge(s) x {args.judge_seeds} seeds"
        + ("" if args.no_lcs else " + LCS")
    )

    out_rows = []
    for i, r in enumerate(rows, start=1):
        rec: dict = {"id": r.get("id"), "topic": r.get("topic")}
        if r.get("conflict_type"):
            rec["conflict_type"] = r["conflict_type"]
        rec["n_chars"] = len(r["response"])
        rec["tells"] = r.get("tells")
        start = time.perf_counter()

        lcs = None
        if score_response is not None:
            try:
                sc = score_response(
                    r["response"],
                    backend=miner_backend,
                    merlin_path=args.merlin_path,
                    pair_policy="windowed",
                    window=4,
                    strength_method="verbalized",
                    nli_method="logprobs",
                )
                nconf = sum(
                    1
                    for x in sc["mining"]["relations"]
                    if x["coupling"] in ("contradiction", "exclusive")
                )
                lcs = {
                    "scores": sc["scores"],
                    "n_atoms": sc["mining"]["n_atoms"],
                    "n_relations": sc["mining"]["n_relations"],
                    "n_conflict_edges": nconf,
                    "saturated": sc["mining"]["saturated"],
                }
            except Exception as e:
                lcs = {"error": f"{type(e).__name__}: {e}"}
                print(f"  [{i}] LCS FAILED: {e}")
                traceback.print_exc()
        rec["lcs"] = lcs

        rec["judges"] = judge_one(r["response"], judges, args.judge_seeds)
        v, why = verdict(
            lcs if (lcs and "error" not in lcs) else None, rec["judges"], target=args.target
        )
        rec["verdict"], rec["failed_conditions"] = v, why
        rec["elapsed_s"] = round(time.perf_counter() - start, 1)
        out_rows.append(rec)

        bits = []
        if lcs and "error" not in lcs:
            s = lcs["scores"]
            bits.append(
                f"mm={s['mean_marginal']:.3f} cons={s['consistency']:.3f} "
                f"conf={lcs['n_conflict_edges']} atoms={lcs['n_atoms']}"
            )
        for k in sorted(rec["judges"]):
            sc = rec["judges"][k]["score"]
            bits.append(f"{k}={'None' if sc is None else format(sc, '.3f')}")
        print(f"  [{i}/{len(rows)}] {rec['id']} {v}  " + "  ".join(bits))
        if why:
            print(f"        why: {'; '.join(why)}")

    # -- aggregate ---------------------------------------------------------
    print("\n" + "=" * 78)
    npass = sum(1 for r in out_rows if r["verdict"] == "PASS")
    print(f"VERDICT: {npass}/{len(out_rows)} PASS  (target {args.target})")
    scored = [r for r in out_rows if r.get("lcs") and "error" not in r["lcs"]]
    if scored:
        for k in ("mean_marginal", "consistency", "reified", "log_partition"):
            vals = [r["lcs"]["scores"][k] for r in scored if r["lcs"]["scores"].get(k) is not None]
            if vals:
                print(f"  LCS {k:16s} mean={st.mean(vals):.3f}"
                      + (f" sd={st.stdev(vals):.3f}" if len(vals) > 1 else ""))
        nconf = [r["lcs"]["n_conflict_edges"] for r in scored]
        print(f"  conflict edges: mean={st.mean(nconf):.1f}, "
              f"{sum(1 for c in nconf if c >= 1)}/{len(nconf)} have >=1")
    keys = sorted({k for r in out_rows for k in r["judges"]})
    for k in keys:
        vals = [r["judges"][k]["score"] for r in out_rows if r["judges"][k].get("score") is not None]
        if vals:
            fooled = sum(1 for v in vals if v >= FOOLED_AT)
            print(f"  JUDGE {k:22s} mean={st.mean(vals):.3f}"
                  + (f" sd={st.stdev(vals):.3f}" if len(vals) > 1 else "")
                  + f"  fooled(>={FOOLED_AT}) {fooled}/{len(vals)}")
        else:
            print(f"  JUDGE {k:22s} all abstained")

    out_path = args.output or os.path.join(
        os.path.dirname(os.path.abspath(args.input)), "judges-report.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "config": {
                    "input": args.input,
                    "limit": args.limit,
                    "judge_seeds": args.judge_seeds,
                    "judge_models": models,
                    "target": args.target,
                    "fooled_at": FOOLED_AT,
                    "lcs_low_at": LCS_LOW_AT,
                    "lcs_scored": not args.no_lcs,
                },
                "counts": {"n_total": len(out_rows), "n_pass": npass},
                "records": out_rows,
            },
            f,
            indent=2,
        )
    print(f"\n[eval] report -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
