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

"""Run a factuality assessor against a locally-served vLLM model in one process.

Intended for one LSF job on a GPU node: this entrypoint starts a local vLLM
server from the given weights, waits until it is ready, runs the selected
factuality pipeline against ``localhost``, and tears the server down on exit.

Any of the factuality assessors are selectable via ``--pipeline``: the
probabilistic ``factreasoner`` pipeline, or the ``factscore`` / ``veriscore`` /
``factverify`` baselines in ``src/fact_reasoner/baselines``.

The server lifecycle is owned by :class:`fact_reasoner.serving.VLLMServer` (a
context manager), so the vLLM process is always cleaned up, even on error. The
pipeline itself is the same importable :func:`fact_reasoner.eval.eval_dataset.run`
used by the standalone CLI.

Examples:
    # FactReasoner (probabilistic; needs Merlin):
    python scripts/run_vllm.py \\
        --model /weights/granite-4.1-8b \\
        --served-model granite-4.1-8b \\
        --input-file data/example.jsonl \\
        --output-dir results/ \\
        --pipeline factreasoner --merlin-path /path/to/merlin

    # A baseline assessor (no Merlin needed):
    python scripts/run_vllm.py \\
        --model /weights/granite-4.1-8b \\
        --served-model granite-4.1-8b \\
        --input-file data/example.jsonl \\
        --output-dir results/ \\
        --pipeline factscore
"""

import argparse

from fact_reasoner.eval.eval_dataset import run
from fact_reasoner.serving import VLLMServer


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve a local vLLM model and run a factuality assessor "
        "(FactReasoner or a baseline) against it."
    )

    # --- vLLM server options ---
    server = parser.add_argument_group("vLLM server")
    server.add_argument(
        "--model",
        required=True,
        help="Local filesystem path to the weights, or a HuggingFace repo id.",
    )
    server.add_argument(
        "--served-model",
        default=None,
        help="Served model name (also used as the client model id). Defaults to "
        "the final component of --model.",
    )
    server.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface for vLLM to bind (default: 127.0.0.1).",
    )
    server.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for vLLM (default: an auto-picked free port).",
    )
    server.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="vLLM tensor-parallel size (default: number of GPUs in "
        "CUDA_VISIBLE_DEVICES, else 1).",
    )
    server.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory vLLM may use (default: 0.90).",
    )
    server.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional maximum context length override.",
    )
    server.add_argument(
        "--dtype",
        default="auto",
        help="vLLM dtype ('auto' is safe on A100/H100; 'bfloat16' is common).",
    )
    server.add_argument(
        "--startup-timeout",
        type=float,
        default=600.0,
        help="Seconds to wait for the server to become ready (default: 600).",
    )
    server.add_argument(
        "--vllm-log",
        default=None,
        help="Path for the vLLM server log (default: vllm.<port>.log).",
    )

    # --- FactReasoner pipeline options (mirror eval_dataset.py) ---
    pipe = parser.add_argument_group("FactReasoner pipeline")
    pipe.add_argument("--input-file", required=True, help="Input dataset (jsonl).")
    pipe.add_argument("--output-dir", required=True, help="Output directory.")
    pipe.add_argument("--cache-dir", default=None, help="Retriever cache directory.")
    pipe.add_argument("--dataset-name", default=None, help="Dataset name (for output).")
    pipe.add_argument(
        "--service-type",
        default="google",
        help="Retrieval service (wikipedia, chromadb, google).",
    )
    pipe.add_argument(
        "--pipeline",
        default="factreasoner",
        choices=["factreasoner", "factscore", "veriscore", "factverify"],
        help="Factuality assessor to run: 'factreasoner' (probabilistic, needs "
        "Merlin) or a baseline ('factscore', 'veriscore', 'factverify').",
    )
    pipe.add_argument(
        "--pipeline-version",
        default="v2",
        help="FactReasoner version: v1, v2 or v3 (factreasoner only).",
    )
    pipe.add_argument("--top-k", type=int, default=3, help="Top-k contexts per atom.")
    pipe.add_argument(
        "--use-priors",
        action="store_true",
        help="Use atom/context priors (factreasoner only).",
    )
    pipe.add_argument(
        "--use-summarizer",
        action="store_true",
        help="Summarize contexts (factreasoner only).",
    )
    pipe.add_argument(
        "--use-query-builder", action="store_true", help="Use the QueryBuilder."
    )
    pipe.add_argument(
        "--merlin-path",
        default=None,
        help="Path to the Merlin inference engine (required for factreasoner).",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    # Start the local vLLM server; the context manager guarantees teardown.
    with VLLMServer(
        args.model,
        served_model_name=args.served_model,
        host=args.host,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        startup_timeout_s=args.startup_timeout,
        log_path=args.vllm_log,
    ) as server:
        backend = server.build_backend()

        run(
            backend,
            input_file=args.input_file,
            output_dir=args.output_dir,
            pipeline=args.pipeline,
            pipeline_version=args.pipeline_version,
            service_type=args.service_type,
            cache_dir=args.cache_dir,
            dataset_name=args.dataset_name,
            model_id=server.served_model_name,
            top_k=args.top_k,
            use_priors=args.use_priors,
            use_summarizer=args.use_summarizer,
            use_query_builder=args.use_query_builder,
            merlin_path=args.merlin_path,
        )


if __name__ == "__main__":
    main()
