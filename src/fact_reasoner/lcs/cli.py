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

# Command-line entrypoint for the Logical Coherence Score.
#
# Installed as the ``fact-reasoner-lcs`` console command. Scores how well a
# response hangs together, optionally priming each atom with its factuality
# posterior from the ``fact-reasoner`` pipeline (the two-stage model; see
# ``fact_reasoner.lcs.priors``).
#
# Kept separate from the ``fact-reasoner`` command on purpose: that one is about
# factuality and already carries a large flag surface, and the coherence knobs
# (pair policy, strength method, readouts, prior source, formulation) are a
# different axis. The backend/model flags are shared via
# ``fact_reasoner.cli._add_backend_args``, so both commands select a model the
# same way.

import argparse
import json
import os

from fact_reasoner.cli import _add_backend_args, _backend_context
from fact_reasoner.core.nli_config import NLI_MODES
from fact_reasoner.lcs.candidate_pairs import PAIR_POLICIES
from fact_reasoner.lcs.lcs_scorer import LCS_METHODS
from fact_reasoner.lcs.pipeline import COHERENCE_FORMULATIONS
from fact_reasoner.lcs.relation_miner import STRENGTH_METHODS
from fact_reasoner.lcs.runner import CoherenceRunner, atom_texts_from_item

# Where the atom priors come from.
PRIOR_SOURCE_CHOICES = ("none", "factreasoner", "file")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fact-reasoner-lcs",
        description=(
            "Compute the Logical Coherence Score (LCS) of a response: mine the "
            "relations between its atoms, build the coherence MRF and read a score "
            "off it. With --priors factreasoner the atoms' unary priors are the "
            "factuality posteriors from the FactReasoner pipeline, so the score "
            "reflects both external support and internal coherence. For factuality "
            "alone, see the `fact-reasoner` command."
        ),
    )

    # --- Input ---
    i = parser.add_argument_group("input")
    i.add_argument("--response", default=None, help="The response text to score.")
    i.add_argument(
        "--response-file",
        default=None,
        help="Read the response from a file: either plain text, or a JSON object "
        "with a 'response' (or 'output') field and optionally an 'atoms' list "
        "(as data/lcs/*.json has), in which case those atoms are mined directly "
        "instead of re-atomizing.",
    )
    i.add_argument("--query", default=None, help="The query the response answers.")
    i.add_argument("--topic", default=None, help="Optional topic hint.")
    i.add_argument(
        "--input-file",
        default=None,
        help="Dataset mode: a jsonl file of items that already carry atoms and "
        "contexts (as the factuality pipeline writes them). Nothing is atomized "
        "or retrieved; results are written incrementally to --output-dir and "
        "already-processed inputs are skipped, so the run is resumable.",
    )
    i.add_argument(
        "--output-dir",
        default=None,
        help="Dataset mode: directory for the output jsonl.",
    )
    i.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset mode: dataset label used in the output filename.",
    )

    # --- Scoring ---
    s = parser.add_argument_group("scoring")
    s.add_argument(
        "--merlin-path",
        default=None,
        required=False,
        help="Path to Merlin (required: the MRF is solved with it).",
    )
    s.add_argument(
        "--methods",
        default="mean_marginal",
        help="Comma-separated LCS readouts to compute, or 'all'. The first is the "
        f"headline. Choices: {', '.join(LCS_METHODS)}. Several readouts share the "
        "base inference runs, so asking for all four costs 7 Merlin calls, not 13.",
    )
    s.add_argument(
        "--formulation",
        default="mrf",
        choices=list(COHERENCE_FORMULATIONS),
        help="Which coherence model to score with (default: mrf). 'mln' is the "
        "Markov-logic research branch; its pairwise fragment is exactly the MRF "
        "and its beyond-pairwise inference is not implemented yet.",
    )
    s.add_argument(
        "--reified-prior",
        type=float,
        default=0.5,
        help="Bernoulli prior on the reified coherence node (default: 0.5; only "
        "used by the 'reified' readout).",
    )
    s.add_argument("--ibound", type=int, default=6, help="Merlin WMB i-bound.")

    # --- Atom priors (the factuality stage) ---
    p = parser.add_argument_group("atom priors (factuality stage)")
    p.add_argument(
        "--priors",
        default="none",
        choices=list(PRIOR_SOURCE_CHOICES),
        help="Where each atom's unary prior comes from. 'none' (default) uses a "
        "flat 0.5, i.e. coherence only. 'factreasoner' runs the factuality "
        "pipeline and uses its posterior marginals -- the two-stage model, which "
        "also reuses the factuality run's atoms so the response is atomized once. "
        "'file' loads priors from a saved FactReasoner results JSON (costs no LLM "
        "calls).",
    )
    p.add_argument(
        "--priors-file",
        default=None,
        help="For --priors file: a FactReasoner results JSON (or a bare "
        "{atom_id: probability} map).",
    )
    p.add_argument(
        "--on-low-coverage",
        default="warn",
        choices=["warn", "raise", "uniform"],
        help="What to do when few atoms carry a real prior (default: warn). "
        "'uniform' discards them all, giving a clean coherence-only score rather "
        "than a half-primed mixture.",
    )
    p.add_argument(
        "--pipeline-version",
        default="v2",
        choices=["v1", "v2", "v3"],
        help="FactReasoner graph shape for --priors factreasoner (default: v2).",
    )
    p.add_argument(
        "--nli-mode",
        default="fast",
        choices=list(NLI_MODES),
        help="NLI candidate-pair preset for the factuality stage (default: fast, "
        "which is far cheaper than all_pairs for the same graph semantics).",
    )
    p.add_argument(
        "--service-type",
        default="google",
        choices=["google", "wikipedia", "chromadb"],
        help="Retrieval service for the factuality stage (default: google).",
    )
    p.add_argument("--cache-dir", default=None, help="Retriever cache directory.")
    p.add_argument(
        "--nli-cache-dir",
        default=None,
        help="Cross-run NLI verdict cache for the factuality stage.",
    )
    p.add_argument("--top-k", type=int, default=3, help="Top-k contexts per atom.")
    p.add_argument(
        "--use-summarizer",
        action="store_true",
        help="Summarize retrieved contexts in the factuality stage.",
    )

    # --- Mining ---
    m = parser.add_argument_group("relation mining")
    m.add_argument(
        "--nli-method",
        default="logprobs",
        choices=["logprobs", "simbauq"],
        help="How the relation type-confidence is estimated: 'logprobs' needs a "
        "logprobs-capable backend (rits/vllm/OpenAI); 'simbauq' works on any "
        "backend (required for ollama and for Claude). Default: logprobs.",
    )
    m.add_argument(
        "--strength-method",
        default="auto",
        choices=[*STRENGTH_METHODS, "auto"],
        help="How the conditional relation strength is estimated (default: auto, "
        "which picks surrogate_logprobs when logprobs are available).",
    )
    m.add_argument(
        "--strength-samples",
        type=int,
        default=8,
        help="Samples per edge for --strength-method surrogate_sampled.",
    )
    m.add_argument(
        "--pair-policy",
        default="windowed",
        choices=list(PAIR_POLICIES),
        help="Candidate atom-pair policy (default: windowed). 'all_pairs' is "
        "quadratic in atoms and over-connects long responses.",
    )
    m.add_argument("--window", type=int, default=4, help="Order-window radius.")
    m.add_argument(
        "--gate",
        default="embedding",
        choices=["embedding", "entity", "none"],
        help="Long-range gate for the gated policy (default: embedding).",
    )
    m.add_argument(
        "--revise-atoms",
        action="store_true",
        help="Decontextualize atoms before mining (when atomizing a raw response).",
    )
    m.add_argument(
        "--progress-bar", action="store_true", help="Show a mining progress bar."
    )

    # RITS by default here (unlike the factuality command's ollama): the default
    # relation-strength estimator reads logprobs, which ollama does not expose.
    _add_backend_args(parser, default_kind="rits")

    parser.add_argument(
        "--output-file",
        default=None,
        help="Single mode: write the full result to this JSON file (else print a "
        "summary).",
    )
    return parser


def _resolve_methods(spec: str) -> tuple[str, ...]:
    """Parse ``--methods`` into a tuple of readout names.

    Args:
        spec: A comma-separated list of readout names, or ``"all"``.

    Returns:
        The readouts, in the order given (the first is the headline).

    Raises:
        SystemExit: If a name is not a known readout.
    """
    if spec.strip().lower() == "all":
        return tuple(LCS_METHODS)
    methods = tuple(m.strip() for m in spec.split(",") if m.strip())
    unknown = [m for m in methods if m not in LCS_METHODS]
    if unknown:
        raise SystemExit(
            f"Unknown --methods entry/entries: {', '.join(unknown)} "
            f"(expected from {', '.join(LCS_METHODS)}, or 'all')."
        )
    if not methods:
        raise SystemExit("--methods must name at least one readout.")
    return methods


def _read_response(args) -> tuple[str, list[str] | None]:
    """Resolve the response text (and pre-extracted atoms, when the file has them).

    Args:
        args: The parsed arguments.

    Returns:
        ``(response, atom_texts_or_None)``.

    Raises:
        SystemExit: If neither/both input forms are given, or the file is unusable.
    """
    if bool(args.response) == bool(args.response_file):
        raise SystemExit("Provide exactly one of --response or --response-file.")

    if args.response:
        return args.response, None

    path = args.response_file
    if not os.path.exists(path):
        raise SystemExit(f"--response-file not found: {path!r}.")
    with open(path) as f:
        raw = f.read()

    stripped = raw.lstrip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            raise SystemExit(f"--response-file is not valid JSON: {e}.") from e
        response = payload.get("response") or payload.get("output")
        if not response:
            raise SystemExit(f"{path!r} has no 'response'/'output' field to score.")
        return response, atom_texts_from_item(payload)

    if not raw.strip():
        raise SystemExit(f"--response-file is empty: {path!r}.")
    return raw, None


def _build_runner(args, backend, methods: tuple[str, ...]) -> CoherenceRunner:
    """Build the :class:`CoherenceRunner` the parsed arguments describe.

    Args:
        args: The parsed arguments.
        backend: The Mellea backend.
        methods: The resolved LCS readouts.

    Returns:
        The runner.
    """
    return CoherenceRunner(
        backend,
        merlin_path=args.merlin_path,
        methods=methods,
        formulation=args.formulation,
        reified_prior=args.reified_prior,
        ibound=args.ibound,
        on_low_coverage=args.on_low_coverage,
        prior_source=args.priors,
        priors_file=args.priors_file,
        pipeline_version=args.pipeline_version,
        service_type=args.service_type,
        cache_dir=args.cache_dir,
        top_k=args.top_k,
        use_summarizer=args.use_summarizer,
        nli_mode=args.nli_mode,
        nli_cache_dir=args.nli_cache_dir,
        nli_method=args.nli_method,
        strength_method=args.strength_method,
        strength_samples=args.strength_samples,
        pair_policy=args.pair_policy,
        window=args.window,
        gate=args.gate,
        revise_atoms=args.revise_atoms,
        show_progress=args.progress_bar,
    )


def main() -> None:
    args = _build_arg_parser().parse_args()

    if not args.merlin_path:
        raise SystemExit("--merlin-path is required (the MRF is solved with Merlin).")

    file_mode = bool(args.input_file)
    if file_mode and (args.response or args.response_file):
        raise SystemExit(
            "Provide either --input-file (dataset mode) or "
            "--response/--response-file (single mode), not both."
        )
    if file_mode and not args.output_dir:
        raise SystemExit("--input-file requires --output-dir.")
    if file_mode and not os.path.exists(args.input_file):
        raise SystemExit(f"--input-file not found: {args.input_file!r}.")
    if args.priors == "file":
        if not args.priors_file:
            raise SystemExit("--priors file requires --priors-file.")
        if not os.path.exists(args.priors_file):
            raise SystemExit(f"--priors-file not found: {args.priors_file!r}.")

    methods = _resolve_methods(args.methods)
    # Validated before the backend is built (and before any credentials are
    # needed), so a bad flag combination fails fast.
    response, atom_texts = (None, None) if file_mode else _read_response(args)

    with _backend_context(args) as backend:
        runner = _build_runner(args, backend, methods)

        if file_mode:
            runner.assess_file(
                args.input_file,
                args.output_dir,
                dataset_name=args.dataset_name,
                model_id=args.model_id,
            )
            return

        # When the input file already carries atoms, those are mined rather than
        # re-atomizing the response.
        result = runner.assess(
            args.query or "", response, topic=args.topic, atom_texts=atom_texts
        )

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(result.to_json(), f, indent=2)
        print(f"[fact-reasoner-lcs] Result written to: {args.output_file}")
    else:
        result.describe()


if __name__ == "__main__":
    main()
