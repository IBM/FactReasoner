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

"""CLI for the LCS experiment harness.

Examples::

    # Offline dry-run (no backends, no Merlin): full pipeline + report.
    python -m fact_reasoner.experiments.run_experiments --dry-run \
        --output-dir results/lcs_dryrun

    # Real run: two models on their own backends + a Merlin executable.
    python -m fact_reasoner.experiments.run_experiments \
        --model granite-4-1-30b:vllm:http://localhost:8000/v1 \
        --model gpt-oss-120b:rits \
        --merlin-path /path/to/merlin --output-dir results/lcs_exp

    # Regenerate the .tex report from an existing results.json.
    python -m fact_reasoner.experiments.run_experiments \
        --report-only --output-dir results/lcs_exp
"""

import argparse
import json
import os
from typing import List

from fact_reasoner.lcs.lcs_scorer import LCS_METHODS
from fact_reasoner.lcs.relation_miner import STRENGTH_METHODS

from fact_reasoner.experiments.config import DEFAULT_MODELS, ExperimentConfig, ModelSpec
from fact_reasoner.experiments.report import combine_results, write_report
from fact_reasoner.experiments.runner import ExperimentRunner


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model", action="append", default=None, metavar="ID:BACKEND[:URL]",
        help="A model to evaluate as 'model_id:backend[:base_url]'. Repeatable. "
        "Defaults to granite-4-1-30b:vllm and gpt-oss-120b:rits.",
    )
    p.add_argument("--examples", default=None,
                   help="Comma-separated example ids (default: all in data/lcs).")
    p.add_argument("--strength-methods", default=None,
                   help=f"Comma-separated subset of {list(STRENGTH_METHODS)}.")
    p.add_argument("--lcs-methods", default=None,
                   help=f"Comma-separated subset of {list(LCS_METHODS)}.")
    p.add_argument("--pair-policy", default="all_pairs",
                   choices=["all_pairs", "windowed", "gated"])
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--gate", default="none", choices=["embedding", "entity", "none"])
    p.add_argument("--strength-samples", type=int, default=8)
    p.add_argument("--reified-prior", type=float, default=0.5)
    p.add_argument("--merlin-path", default=None, help="Path to the Merlin executable.")
    p.add_argument("--data-dir", default="data/lcs")
    p.add_argument("--output-dir", default="results/lcs_experiments")
    p.add_argument("--dry-run", action="store_true",
                   help="Run offline with a stubbed LLM and brute-force inference.")
    p.add_argument("--report-only", action="store_true",
                   help="Skip the sweep; regenerate the report from results.json.")
    p.add_argument(
        "--combine", default=None,
        help="Comma-separated list of results.json paths (or their dirs) to merge "
        "into a single combined report written to --output-dir. Columns are keyed "
        "by (model, pair-policy, strength) so multiple policies appear side by side.",
    )
    return p


def _models_from_args(args) -> List[ModelSpec]:
    if args.model:
        return [ModelSpec.parse(s) for s in args.model]
    return list(DEFAULT_MODELS)


def _csv(val):
    return [x.strip() for x in val.split(",")] if val else None


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)

    if args.combine:
        parts = []
        for p in _csv(args.combine):
            path = p if p.endswith(".json") else os.path.join(p, "results.json")
            with open(path) as f:
                parts.append(json.load(f))
        combined = combine_results(parts)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(combined, f, indent=2)
        write_report(combined, args.output_dir)
        print(f"[experiments] combined {len(parts)} runs -> {args.output_dir}")
        return

    if args.report_only:
        results_path = os.path.join(args.output_dir, "results.json")
        with open(results_path) as f:
            results = json.load(f)
        write_report(results, args.output_dir)
        return

    if not args.dry_run and not args.merlin_path:
        raise SystemExit(
            "A real run requires --merlin-path (or use --dry-run for the offline "
            "oracle)."
        )

    config = ExperimentConfig(
        models=_models_from_args(args),
        example_ids=_csv(args.examples),
        strength_methods=_csv(args.strength_methods) or list(STRENGTH_METHODS),
        lcs_methods=_csv(args.lcs_methods) or list(LCS_METHODS),
        pair_policy=args.pair_policy,
        window=args.window,
        gate=args.gate,
        strength_samples=args.strength_samples,
        reified_prior=args.reified_prior,
        merlin_path=args.merlin_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
    )
    results = ExperimentRunner(config).run()
    write_report(results, args.output_dir)


if __name__ == "__main__":
    main()
