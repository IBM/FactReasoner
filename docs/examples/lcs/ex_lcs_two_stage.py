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

"""Logical Coherence Score: coherence only, and primed with factuality.

Three ways to set the coherence MRF's per-atom priors, in increasing cost:

  1. UNIFORM (the default) -- every atom starts at 0.5, so the score measures how
     well the response hangs together and nothing else. No retrieval.
  2. PRECOMPUTED -- priors from a saved FactReasoner results file. No LLM calls at
     all, so one factuality run can prime many coherence experiments.
  3. LIVE FACTUALITY -- a real FactReasoner run supplies each atom's posterior
     marginal, so the score reflects external support AND internal coherence. This
     is the two-stage model; the factuality run's atoms are reused, so the response
     is atomized once rather than once per stage.

Run it::

    # coherence only (needs a backend + merlin)
    python docs/examples/lcs/ex_lcs_two_stage.py --merlin-path /path/to/merlin

    # two-stage, with a live factuality run (needs a retriever too)
    python docs/examples/lcs/ex_lcs_two_stage.py --merlin-path /path/to/merlin \
        --priors factreasoner --backend rits --model-id llama-3-3-70b-instruct

    # offline demo -- no backend, no merlin, no retrieval
    python docs/examples/lcs/ex_lcs_two_stage.py --dry-run

The equivalent one-liner is the `fact-reasoner-lcs` console command; see the
accompanying .md.
"""

import argparse
import json

RESPONSE = (
    "AeroParts issued a recall of its turbine blades in March. The recall "
    "followed a fatigue crack found during a routine inspection. Because the "
    "crack could propagate under load, the regulator grounded the fleet. "
    "No injuries were reported. Two passengers were treated for minor injuries "
    "after the incident. The tribunal ultimately held that the pilots were not "
    "at fault."
)
QUERY = "What happened with the AeroParts turbine blade recall?"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-path", default=None, help="Path to Merlin.")
    parser.add_argument(
        "--priors",
        default="none",
        choices=["none", "factreasoner", "file"],
        help="Where the atom priors come from (default: none = coherence only).",
    )
    parser.add_argument(
        "--priors-file", default=None, help="For --priors file: a results JSON."
    )
    parser.add_argument("--backend", default="rits", help="Backend kind.")
    parser.add_argument("--model-id", default="llama-3-3-70b-instruct")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run fully offline with stubbed LLM + brute-force inference.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    """Score the response under the selected prior source."""
    from fact_reasoner.backends import build_backend
    from fact_reasoner.core.atomizer import Atomizer
    from fact_reasoner.lcs import CoherencePipeline, RelationMiner

    backend = build_backend(args.backend, model_id=args.model_id)
    miner = RelationMiner(
        backend,
        atomizer=Atomizer(backend),
        pair_policy="windowed",
        window=3,
    )

    # -- pick the prior source -------------------------------------------------
    prior_provider = None  # None => uniform 0.5, i.e. coherence only
    if args.priors == "file":
        from fact_reasoner.lcs import PrecomputedPriorProvider

        prior_provider = PrecomputedPriorProvider(args.priors_file)
    elif args.priors == "factreasoner":
        from fact_reasoner.lcs import FactReasonerPriorProvider
        from fact_reasoner.runner import FactualityRunner

        # Every FactualityRunner axis applies unchanged: pipeline_version, nli_mode,
        # nli_method, caches, backend. `fast` is much cheaper than `all_pairs` for
        # the same graph semantics.
        runner = FactualityRunner(
            backend,
            merlin_path=args.merlin_path,
            nli_mode="fast",
            use_priors=True,
        )
        prior_provider = FactReasonerPriorProvider(runner=runner)

    pipeline = CoherencePipeline(
        miner=miner,
        merlin_path=args.merlin_path,
        prior_provider=prior_provider,
        # All four readouts share the base inference runs, so asking for every one
        # costs 6 Merlin invocations rather than 12.
        methods=("mean_marginal", "consistency", "reified", "log_partition"),
    )

    result = pipeline.run(RESPONSE, query=QUERY)
    result.describe()

    print("\nPer-atom priors -> coherence posteriors:")
    for aid in sorted(result.marginals, key=lambda a: int(a.lstrip("a") or 0)):
        prior = result.priors.get(aid, 0.5)
        post = result.marginals[aid]
        arrow = "v" if post < prior else "^"
        print(f"  {aid}: prior={prior:.3f} -> posterior={post:.3f}  {arrow}")

    if result.factuality:
        print(f"\nStage-1 (factuality) diagnostics: {result.factuality}")
    if result.prior_coverage.get("degraded"):
        print(f"\nNOTE: priors degraded -- {result.prior_coverage}")


def run_dry(args: argparse.Namespace) -> None:
    """The same flow with every external service stubbed out.

    Uses the experiment harness's dry-run stubs: a deterministic ``ainstruct`` for
    the miner and the exact brute-force enumerator in place of Merlin. Useful to
    see the shape of the output without a backend.
    """
    from fact_reasoner.experiments.mock import dry_run_patches, make_mock_backend
    from fact_reasoner.lcs import (
        CoherencePipeline,
        PrecomputedPriorProvider,
        RelationMiner,
    )

    atoms = [
        "AeroParts recalled its turbine blades in March.",
        "A fatigue crack was found during a routine inspection.",
        "The regulator grounded the fleet.",
        "No injuries were reported.",
        "Two passengers were treated for minor injuries.",
    ]
    # Stand in for a factuality run: the "no injuries" / "two treated" pair is the
    # incoherent one, and the last atom is also poorly supported factually.
    factuality_posteriors = {
        "a0": 0.95, "a1": 0.90, "a2": 0.88, "a3": 0.60, "a4": 0.20,
    }

    with dry_run_patches():
        miner = RelationMiner(
            make_mock_backend(), pair_policy="all_pairs", strength_method="verbalized"
        )
        mining = miner.mine_from_atoms(atoms, " ".join(atoms))

        for label, priors in (
            ("uniform (coherence only)", None),
            ("factuality-primed", factuality_posteriors),
        ):
            pipeline = CoherencePipeline(
                miner=miner,
                merlin_path="dry-run",
                prior_provider=(
                    PrecomputedPriorProvider(priors) if priors else None
                ),
                methods=("mean_marginal", "consistency"),
            )
            out = pipeline.run_from_mining(mining, priors=priors)
            print(f"\n--- {label} ---")
            print(f"  LCS (mean_marginal): {out.scores['mean_marginal']:.4f}")
            print(f"  consistency:         {out.scores['consistency']:.4f}")
            print(f"  atoms below their own prior: {out.diagnostics.get('num_below_prior')}")
            print(f"  priors: {json.dumps({k: round(v, 2) for k, v in out.priors.items()})}")


def main() -> None:
    args = _parse_args()
    if args.dry_run:
        run_dry(args)
        return
    if not args.merlin_path:
        raise SystemExit("--merlin-path is required (or pass --dry-run).")
    if args.priors == "file" and not args.priors_file:
        raise SystemExit("--priors file requires --priors-file.")
    run(args)


if __name__ == "__main__":
    main()
