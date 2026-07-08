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

# Command-line entrypoint for the factuality runner.
#
# Installed as the ``fact-reasoner`` console command. Runs any factuality
# assessor (FactReasoner or a baseline) with any backend (ollama, rits, or a
# local vLLM instance), over a single query/response or a jsonl dataset.

import argparse
import contextlib
import json

from fact_reasoner.backends import build_backend
from fact_reasoner.runner import PIPELINES, FactualityRunner

# Friendly RITS model shortcuts -> concrete RITS model identifiers (resolved lazily).
_RITS_SHORTCUTS = {
    "llama3": "LLAMA_3_3_70B_INSTRUCT",
    "granite4": "GRANITE_4_H_SMALL",
    "mistral": "MISTRAL_LARGE_3_675B_2512",
    "gpt-oss": "GPT_OSS_120B",
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fact-reasoner",
        description="Run a factuality assessor (FactReasoner or a baseline) over "
        "a single query/response or a dataset, with an Ollama, RITS, or local "
        "vLLM backend.",
    )

    # --- Pipeline ---
    p = parser.add_argument_group("assessor")
    p.add_argument(
        "--pipeline",
        default="factreasoner",
        choices=list(PIPELINES),
        help="Factuality assessor to run (default: factreasoner).",
    )
    p.add_argument(
        "--pipeline-version",
        default="v2",
        choices=["v1", "v2", "v3"],
        help="FactReasoner version (default: v2; ignored by baselines).",
    )
    p.add_argument(
        "--merlin-path",
        default=None,
        help="Path to Merlin (required for factreasoner).",
    )
    p.add_argument("--use-priors", action="store_true", help="Use atom/context priors.")
    p.add_argument("--use-summarizer", action="store_true", help="Summarize contexts.")
    p.add_argument(
        "--use-query-builder", action="store_true", help="Use the QueryBuilder."
    )
    p.add_argument(
        "--nli-method",
        default="logprobs",
        choices=["logprobs", "simbauq"],
        help="How the NLI extractor estimates relation probabilities: "
        "'logprobs' needs a logprobs-capable backend (rits/vllm); 'simbauq' "
        "uses self-consistency and works on any backend (required for ollama, "
        "which does not expose logprobs). Default: logprobs.",
    )
    p.add_argument(
        "--nli-similarity-metric",
        default="rouge",
        choices=["rouge", "jaccard", "sbert", "difflib", "levenshtein"],
        help="Similarity metric for the SIMBA-UQ NLI method "
        "(only used with --nli-method simbauq; default: rouge).",
    )

    # --- Retrieval ---
    r = parser.add_argument_group("retrieval")
    r.add_argument(
        "--service-type",
        default="google",
        choices=["google", "wikipedia", "chromadb"],
        help="Retrieval service (default: google).",
    )
    r.add_argument("--cache-dir", default=None, help="Retriever cache directory.")
    r.add_argument("--top-k", type=int, default=3, help="Top-k contexts per atom.")

    # --- Backend ---
    b = parser.add_argument_group("backend")
    b.add_argument(
        "--backend",
        default="ollama",
        choices=["ollama", "rits", "vllm"],
        help="Backend to use (default: ollama).",
    )
    b.add_argument(
        "--model-id",
        default=None,
        help="Model id. For rits, a shortcut (%s); for ollama/vllm-client, the "
        "model / served-model name." % ", ".join(_RITS_SHORTCUTS),
    )
    b.add_argument(
        "--base-url",
        default=None,
        help="Base URL for a vllm client (defaults to VLLM_BASE_URL env).",
    )
    # vLLM local-server options: passing --model starts a local server.
    b.add_argument(
        "--model",
        default=None,
        help="For --backend vllm: local weights path or HF repo id. Supplying "
        "this starts a LOCAL vLLM server (otherwise connect as a client).",
    )
    b.add_argument("--served-model", default=None, help="vLLM served-model name.")
    b.add_argument(
        "--tensor-parallel-size", type=int, default=None, help="vLLM TP size."
    )
    b.add_argument(
        "--gpu-memory-utilization", type=float, default=0.90, help="vLLM GPU mem frac."
    )
    b.add_argument("--max-model-len", type=int, default=None, help="vLLM max ctx len.")

    # --- Input (single vs file) ---
    i = parser.add_argument_group("input")
    i.add_argument("--query", default=None, help="Single-mode: the input query.")
    i.add_argument("--response", default=None, help="Single-mode: the response.")
    i.add_argument("--topic", default=None, help="Single-mode: optional topic hint.")
    i.add_argument("--input-file", default=None, help="File-mode: input jsonl dataset.")
    i.add_argument("--output-dir", default=None, help="File-mode: output directory.")
    i.add_argument("--dataset-name", default=None, help="File-mode: dataset label.")

    # --- Output ---
    parser.add_argument(
        "--output-file",
        default=None,
        help="Single-mode: write the results dict to this JSON file (else print).",
    )
    return parser


def _resolve_rits_model(model_id):
    """Map a RITS shortcut to a concrete RITS model identifier."""
    from mellea_ibm.rits import RITS

    if model_id not in _RITS_SHORTCUTS:
        raise SystemExit(
            f"Unknown RITS model shortcut: {model_id!r} "
            f"(expected one of {sorted(_RITS_SHORTCUTS)})."
        )
    return getattr(RITS, _RITS_SHORTCUTS[model_id])


@contextlib.contextmanager
def _backend_context(args):
    """Yield a Mellea backend, starting/stopping a local vLLM server if requested.

    - rits: resolve the model shortcut, build a RITS backend.
    - vllm + --model: start a local VLLMServer and yield its backend.
    - vllm (no --model): connect as a client to --base-url / VLLM_BASE_URL.
    - ollama: build an Ollama backend.
    """
    if args.backend == "rits":
        yield build_backend("rits", model_id=_resolve_rits_model(args.model_id))
    elif args.backend == "vllm" and args.model:
        # Import lazily so the vllm-server path is only required when used.
        from fact_reasoner.serving import VLLMServer

        with VLLMServer(
            args.model,
            served_model_name=args.served_model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        ) as server:
            yield server.build_backend()
    elif args.backend == "vllm":
        yield build_backend(
            "vllm", model_id=args.served_model or args.model_id, base_url=args.base_url
        )
    else:  # ollama
        yield build_backend("ollama", model_id=args.model_id)


def main() -> None:
    args = _build_arg_parser().parse_args()

    # Validate input mode: exactly one of single / file.
    single = args.query is not None or args.response is not None
    file_mode = args.input_file is not None
    if single and file_mode:
        raise SystemExit("Provide either --query/--response OR --input-file, not both.")
    if not single and not file_mode:
        raise SystemExit("Provide either --query/--response (single) or --input-file.")
    if single and (args.query is None or args.response is None):
        raise SystemExit("Single mode requires both --query and --response.")
    if file_mode and not args.output_dir:
        raise SystemExit("File mode requires --output-dir.")
    if args.pipeline == "factreasoner" and not args.merlin_path:
        raise SystemExit("The 'factreasoner' pipeline requires --merlin-path.")
    if args.backend == "vllm" and args.model and not args.served_model:
        raise SystemExit(
            "Starting a local vLLM server requires --served-model "
            "(the vLLM --served-model-name)."
        )
    # Ollama does not expose token logprobs, so the default NLI method degrades
    # to all-neutral relations. Steer the user to the SIMBA-UQ method.
    if args.backend == "ollama" and args.nli_method == "logprobs":
        print(
            "[warning] The 'ollama' backend does not expose logprobs, so "
            "--nli-method logprobs yields all-neutral NLI relations. Use "
            "--nli-method simbauq for meaningful NLI probabilities on Ollama."
        )

    with _backend_context(args) as backend:
        runner = FactualityRunner(
            backend,
            pipeline=args.pipeline,
            pipeline_version=args.pipeline_version,
            service_type=args.service_type,
            cache_dir=args.cache_dir,
            top_k=args.top_k,
            use_priors=args.use_priors,
            use_summarizer=args.use_summarizer,
            use_query_builder=args.use_query_builder,
            merlin_path=args.merlin_path,
            nli_method=args.nli_method,
            nli_similarity_metric=args.nli_similarity_metric,
        )

        if file_mode:
            model_label = args.served_model or args.model_id
            runner.assess_file(
                args.input_file,
                args.output_dir,
                dataset_name=args.dataset_name,
                model_id=model_label,
            )
        else:
            results = runner.assess(
                args.query,
                args.response,
                topic=args.topic,
                output_file=args.output_file,
            )
            if not args.output_file:
                print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
