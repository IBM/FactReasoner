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

Scores every item of a generated dataset from the relations the item file already
carries -- no LLM, so the whole run is deterministic and offline (Merlin is the
only subprocess). Atom priors come from the corpus's ``factual`` labels
(0.9 / 0.1), edge probabilities from the midpoint of each relation's strength
band, and resolved concessions are discounted using the item's own
``resolver_atom_id``.

Run::

    locobench-lcs-eval --data-dir data/locobench-claude-5 \\
        --out-dir results/locobench_claude_5_lcs \\
        --merlin-path /path/to/merlin

    # or, from a checkout:
    python scripts/eval_locobench_lcs.py --merlin-path /path/to/merlin

Re-render the report from an existing run without re-scoring::

    locobench-lcs-eval --out-dir results/... --report-only

Outputs, under ``--out-dir``: ``results.json`` (config, every record, and the
per-family ladder checks), ``records/`` (one file per item x arm), ``by_item/``
(one file per item with its text, atoms and gold relations), ``report.tex`` and
``report.pdf``.
"""

from __future__ import annotations

import argparse
import json
import os

from fact_reasoner.lcs.lcs_scorer import LCS_METHODS
from fact_reasoner.locoeval.gold_graph import DEFAULT_CONCESSION_DISCOUNT
from fact_reasoner.locoeval.report import build_pdf, write_report
from fact_reasoner.locoeval.runner import GOLD_ARMS, GoldEvalRunner


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
        default="data/locobench-claude-5",
        help="Dataset directory holding items.jsonl (and optionally families.json).",
    )
    parser.add_argument(
        "--out-dir",
        default="results/locobench_claude_5_lcs",
        help="Where results.json, records/, by_item/ and the report are written.",
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
        help=f"Which gold variants to score (default: {','.join(GOLD_ARMS)}).",
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


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation. Returns a process exit code (non-zero if a cell failed)."""
    parser = build_parser()
    args = parser.parse_args(argv)

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
        results = GoldEvalRunner(
            data_dir=args.data_dir,
            output_dir=args.out_dir,
            merlin_path=args.merlin_path,
            item_ids=args.items,
            arms=args.arms,
            methods=args.methods,
            concession_discount=args.concession_discount,
            reified_prior=args.reified_prior,
            ibound=args.ibound,
            verbose=args.verbose,
        ).run()

    n_failed = sum(1 for r in results.get("records", []) if "error" in r)
    if n_failed:
        print(f"[locoeval] WARNING: {n_failed} cell(s) failed; see results.json.")

    if not args.no_report:
        tex = write_report(results, args.out_dir, example_ids=args.examples)
        if not args.no_pdf:
            build_pdf(tex)

    return 1 if n_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
