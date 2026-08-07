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

"""Command line interface for the LoCoBench LCS evaluation.

Scores every item of a generated dataset under one or more *arms*. A gold arm
builds the MRF from the relations the item file already carries -- no LLM, fully
deterministic, Merlin the only subprocess. A **mined** arm runs the real relation
miner over the item's response prose instead, so it measures whether the pipeline
can recover the graph the corpus asserts. Both arms hold the atoms and the
factuality priors (0.9 / 0.1) fixed, which is what makes them comparable.

Gold only (the original behaviour)::

    locobench-lcs-eval --data-dir data/locobench-claude-5-test \\
        --out-dir results/locobench_claude_5_lcs \\
        --merlin-path /path/to/merlin

Gold plus a mined sweep over models x pair policies::

    export RITS_API_KEY=...
    locobench-lcs-eval --merlin-path /path/to/merlin \\
        --models llama-3.3-70b-instruct --pair-policies windowed,all_pairs \\
        --resume

``--models`` x ``--pair-policies`` expands into ``mined:<model>:<policy>`` arms
appended to ``--arms``, so the sweep is stated as axes but recorded as arms.
``--estimate-only`` prints the projected LLM call count and exits; ``--resume``
reuses per-cell records from an earlier invocation, which is what makes a long
sweep restartable and lets the cheap and expensive cells run as separate commands.

Re-render the report from an existing run without re-scoring::

    locobench-lcs-eval --out-dir results/... --report-only

Outputs, under ``--out-dir``: ``results.json`` (config, every record, the
per-family ladder checks and the mined-vs-gold summary), ``records/`` (one file
per item x arm), ``by_item/`` (one file per item with its text, atoms and gold
relations), ``report.tex`` and ``report.pdf``.
"""

from __future__ import annotations

import argparse
import json
import os

from fact_reasoner.lcs.candidate_pairs import PAIR_POLICIES
from fact_reasoner.lcs.lcs_scorer import LCS_METHODS
from fact_reasoner.locoeval.gold_graph import DEFAULT_CONCESSION_DISCOUNT, item_atoms
from fact_reasoner.locoeval.mined_graph import (
    DEFAULT_MAX_CALL_ERROR_RATE,
    format_arm,
    parse_arm,
)
from fact_reasoner.locoeval.models import DEFAULT_MODELS_FILE, load_model_specs
from fact_reasoner.locoeval.report import build_pdf, write_report
from fact_reasoner.locoeval.runner import GOLD_ARMS, GoldEvalRunner, load_items

# Default dataset and output directory. The dataset is the 10-item Claude-5 test
# set; the output directory is deliberately NOT the gold baseline run
# (`results/locobench_claude_5_fixed_lcs`), which the mined arms are compared
# against and must not be overwritten.
DEFAULT_DATA_DIR = "data/locobench-claude-5-test"
DEFAULT_OUT_DIR = "results/locobench_claude_5_mined_lcs"


def _csv(value: str) -> list[str]:
    """Parse a comma-separated option into a list of non-empty tokens."""
    return [tok.strip() for tok in (value or "").split(",") if tok.strip()]


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (exposed so tests can exercise it)."""
    parser = argparse.ArgumentParser(
        prog="locobench-lcs-eval",
        description=(
            "Evaluate the LCS pipeline on a generated LoCoBench dataset, scoring "
            "each item from its own gold relations."
        ),
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Dataset directory holding items.jsonl (and optionally families.json) "
        f"(default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help="Where results.json, records/, by_item/ and the report are written "
        f"(default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--merlin-path",
        default=os.environ.get("MERLIN_PATH"),
        help="Path to the Merlin executable (or set MERLIN_PATH). Required unless "
        "--report-only.",
    )
    parser.add_argument(
        "--items",
        type=_csv,
        default=None,
        help="Comma-separated item ids to score (default: every item).",
    )
    parser.add_argument(
        "--arms",
        type=_csv,
        default=list(GOLD_ARMS),
        help=f"Which arms to score: any of {','.join(GOLD_ARMS)} plus explicit "
        "mined:<model>:<policy> arms. --models/--pair-policies append to this "
        f"(default: {','.join(GOLD_ARMS)}).",
    )

    mined = parser.add_argument_group(
        "mined arms",
        "Run the real relation miner instead of reading the item's gold labels. "
        "Needs a served model, so unlike a gold arm these arms are neither free "
        "nor deterministic.",
    )
    mined.add_argument(
        "--models",
        type=_csv,
        default=None,
        help="Model short names from --models-file; crossed with --pair-policies "
        "into mined arms.",
    )
    mined.add_argument(
        "--pair-policies",
        type=_csv,
        default=None,
        help=f"Candidate-pair policies to sweep, from {list(PAIR_POLICIES)} "
        "(default: windowed, when --models is given).",
    )
    mined.add_argument(
        "--models-file",
        default=DEFAULT_MODELS_FILE,
        help=f"Served-model inventory (default: {DEFAULT_MODELS_FILE}). Model names "
        "are never resolved against the catalog: it cannot see RITS endpoints.",
    )
    mined.add_argument(
        "--window",
        type=int,
        default=4,
        help="Order-window radius for the windowed/gated policies (default: 4).",
    )
    mined.add_argument(
        "--gate",
        default="none",
        help="Long-range gate for the gated policy (default: none).",
    )
    mined.add_argument(
        "--nli-method",
        default="auto",
        choices=("auto", "logprobs", "simbauq"),
        help="Type-confidence method; auto picks logprobs on a logprobs-capable "
        "backend (default: auto).",
    )
    mined.add_argument(
        "--strength-method",
        default="auto",
        help="Conditional-strength method, or auto (default: auto).",
    )
    mined.add_argument(
        "--strength-samples",
        type=int,
        default=8,
        help="Samples per edge for surrogate_sampled (default: 8).",
    )
    mined.add_argument(
        "--max-concurrency",
        type=int,
        default=16,
        help="Concurrent LLM calls per item (default: 16). Lower it for an endpoint "
        "that throttles: a rate-limited call parses as 'no relation'.",
    )
    mined.add_argument(
        "--max-call-error-rate",
        type=float,
        default=DEFAULT_MAX_CALL_ERROR_RATE,
        help="Fail a cell when more than this fraction of its LLM calls error "
        f"(default: {DEFAULT_MAX_CALL_ERROR_RATE}). Failed calls are silently "
        "indistinguishable from genuine negatives, so the ceiling is strict.",
    )
    mined.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing successful per-cell records whose run configuration "
        "matches; re-run everything else.",
    )
    mined.add_argument(
        "--dry-run",
        action="store_true",
        help="Stub the LLM (offline). Scoring still uses the real Merlin, so a "
        "realistically-sized item can be exercised without spending tokens.",
    )
    mined.add_argument(
        "--estimate-only",
        action="store_true",
        help="Print the projected LLM call count for the requested sweep and exit.",
    )
    mined.add_argument(
        "--show-progress",
        action="store_true",
        help="Show the miner's per-item progress bar.",
    )
    mined.add_argument(
        "--baseline-results",
        default=None,
        help="An earlier results.json whose gold cells this run must reproduce; "
        "adds a reproduction check to the report.",
    )
    parser.add_argument(
        "--methods",
        type=_csv,
        default=list(LCS_METHODS),
        help=f"LCS readouts to compute (default: {','.join(LCS_METHODS)}).",
    )
    parser.add_argument(
        "--examples",
        type=_csv,
        default=None,
        help="Item ids to write up as worked examples (default: each family's base "
        "rung).",
    )
    parser.add_argument(
        "--concession-discount",
        type=float,
        default=DEFAULT_CONCESSION_DISCOUNT,
        help="Lambda for a resolved concession; 0 disables the discount "
        f"(default: {DEFAULT_CONCESSION_DISCOUNT}).",
    )
    parser.add_argument(
        "--reified-prior",
        type=float,
        default=0.5,
        help="Bernoulli prior on the reified coherence node (default: 0.5).",
    )
    parser.add_argument(
        "--ibound", type=int, default=6, help="Merlin i-bound (default: 6)."
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Re-render the report from an existing results.json; do not score.",
    )
    parser.add_argument(
        "--no-report", action="store_true", help="Score only; write no report."
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Write report.tex but do not run pdflatex.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Let the Merlin helper print progress."
    )
    return parser


def _expand_arms(args, parser) -> list[str]:
    """Resolve --arms plus the --models x --pair-policies cross product.

    Order is preserved and duplicates dropped, so `--arms gold` with
    `--models m --pair-policies windowed,all_pairs` yields
    `[gold, mined:m:windowed, mined:m:all_pairs]`.
    """
    arms = list(args.arms or [])
    if args.models:
        policies = args.pair_policies or ["windowed"]
        for policy in policies:
            if policy not in PAIR_POLICIES:
                parser.error(
                    f"Unknown pair policy {policy!r} (expected one of "
                    f"{list(PAIR_POLICIES)})."
                )
        for model in args.models:
            for policy in policies:
                arms.append(format_arm(model, policy))
    elif args.pair_policies:
        parser.error("--pair-policies needs --models to name the model(s) to sweep.")

    seen: set[str] = set()
    ordered: list[str] = []
    for arm in arms:
        if arm not in seen:
            seen.add(arm)
            ordered.append(arm)
    return ordered


def _estimate(args, arms: list[str], parser) -> None:
    """Print the projected LLM call count per mined arm.

    Pair selection is pure -- no LLM, no Merlin -- so the Prompt A count is exact.
    Prompt B fires once per pair that yields an edge, which is the only unknown, so
    it is bracketed rather than guessed.
    """
    from fact_reasoner.lcs import candidate_pairs as cp

    mined = [(a, parse_arm(a)) for a in arms]
    mined = [(a, s) for a, s in mined if s is not None]
    if not mined:
        print("[locoeval] no mined arms: gold arms make no LLM calls.")
        return

    items = load_items(args.data_dir, args.items)
    print(f"[locoeval] estimate over {len(items)} item(s):")
    grand = 0
    for arm, spec in mined:
        pairs = 0
        for item in items:
            selected, _ = cp.select(
                item_atoms(item),
                response=item.get("response") or "",
                policy=spec.pair_policy,
                window=args.window,
                gate=args.gate,
            )
            pairs += len(selected)
        # Prompt A once per pair; Prompt B once per edge found (10-50% of pairs).
        lo, hi = pairs + int(0.10 * pairs), pairs + int(0.50 * pairs)
        grand += hi
        print(
            f"  {arm:<48s} {pairs:5d} pairs -> {lo}-{hi} LLM calls "
            f"(Prompt A + Prompt B per edge)"
        )
    print(f"[locoeval] upper bound across mined arms: ~{grand} LLM calls")


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation. Returns a process exit code (non-zero if a cell failed)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    arms = _expand_arms(args, parser)
    mined_arms = [a for a in arms if parse_arm(a) is not None]

    model_specs = None
    if mined_arms:
        try:
            model_specs = load_model_specs(args.models_file)
        except (FileNotFoundError, ValueError) as e:
            parser.error(str(e))
        unknown = sorted(
            {parse_arm(a).model for a in mined_arms} - set(model_specs)  # type: ignore[union-attr]
        )
        if unknown:
            parser.error(
                f"Unknown model(s) {unknown} in --models/--arms. Available in "
                f"{args.models_file}: {sorted(model_specs)}."
            )

    if args.estimate_only:
        _estimate(args, arms, parser)
        return 0

    if mined_arms and not args.dry_run and not args.report_only:
        # Fail before a backend is built, not 20 minutes into a sweep.
        needs_rits = any(
            model_specs[parse_arm(a).model].backend == "rits"  # type: ignore[union-attr,index]
            for a in mined_arms
        )
        if needs_rits and not os.environ.get("RITS_API_KEY"):
            parser.error(
                "Mined arms on RITS need a key. Export it first:\n"
                "    export RITS_API_KEY=...\n"
                "(or pass --dry-run to stub the LLM offline)."
            )

    if args.report_only:
        results_path = os.path.join(args.out_dir, "results.json")
        if not os.path.exists(results_path):
            parser.error(
                f"--report-only needs an existing {results_path}; run the sweep first."
            )
        with open(results_path) as f:
            results = json.load(f)
    else:
        if not args.merlin_path:
            parser.error(
                "--merlin-path is required (or set MERLIN_PATH): the LCS readouts "
                "are read off Merlin's inference."
            )
        if not os.path.exists(args.merlin_path):
            parser.error(f"Merlin executable not found: {args.merlin_path!r}")
        try:
            runner = GoldEvalRunner(
                data_dir=args.data_dir,
                output_dir=args.out_dir,
                merlin_path=args.merlin_path,
                item_ids=args.items,
                arms=arms,
                methods=args.methods,
                concession_discount=args.concession_discount,
                reified_prior=args.reified_prior,
                ibound=args.ibound,
                verbose=args.verbose,
                model_specs=model_specs,
                window=args.window,
                gate=args.gate,
                nli_method=args.nli_method,
                strength_method=args.strength_method,
                strength_samples=args.strength_samples,
                max_concurrency=args.max_concurrency,
                max_call_error_rate=args.max_call_error_rate,
                resume=args.resume,
                show_progress=args.show_progress,
            )
        except ValueError as e:  # unknown arm / policy / model
            parser.error(str(e))

        if args.dry_run:
            from unittest.mock import MagicMock

            from fact_reasoner.experiments.mock import dry_run_patches

            runner.backend_factory = lambda spec: MagicMock(name=spec.name)
            # Stub the LLM but keep the REAL Merlin: the offline brute-force oracle
            # refuses networks this size (16 atoms plus the scorer's auxiliary
            # variables), so patching it would fail every cell.
            with dry_run_patches(patch_merlin=False):
                results = runner.run()
        else:
            results = runner.run()

    n_failed = sum(1 for r in results.get("records", []) if "error" in r)
    if n_failed:
        print(f"[locoeval] WARNING: {n_failed} cell(s) failed; see results.json.")

    baseline = None
    if args.baseline_results:
        if not os.path.exists(args.baseline_results):
            parser.error(f"No baseline results at {args.baseline_results!r}.")
        with open(args.baseline_results) as f:
            baseline = json.load(f)

    if not args.no_report:
        tex = write_report(
            results, args.out_dir, example_ids=args.examples, baseline=baseline
        )
        if not args.no_pdf:
            build_pdf(tex)

    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
