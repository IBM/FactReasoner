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

# Main runner script

import os
import json
import argparse
import pandas as pd

from typing import Optional

from mellea.backends import Backend

# Local imports
from fact_reasoner.backends import build_backend
from fact_reasoner.assessor import FactReasoner
from fact_reasoner.baselines.factscore import FactScore
from fact_reasoner.baselines.factverify import FactVerify
from fact_reasoner.baselines.veriscore import VeriScore
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.core.retriever import ContextRetriever
from fact_reasoner.core.query_builder import QueryBuilder
from fact_reasoner.core.summarizer import ContextSummarizer
from fact_reasoner.core.nli import NLIExtractor


def build_backend_from_args(
    backend_kind: str,
    model_id: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Backend:
    """Create a Mellea backend from the CLI ``--backend`` selection.

    For ``rits`` the ``model_id`` is a friendly shortcut (``llama3``,
    ``granite4``, ``mistral``, ``gpt-oss``); for ``ollama`` and ``vllm`` it is
    passed through as the model / served-model name.

    Args:
        backend_kind: One of ``"rits"``, ``"ollama"`` or ``"vllm"``.
        model_id: Model shortcut (rits) or model / served-model name.
        base_url: Base URL for the ``vllm`` backend (optional).

    Returns:
        A ready-to-use Mellea backend.
    """
    if backend_kind == "rits":
        # Resolve the friendly RITS model shortcuts to concrete RITS models.
        from mellea_ibm.rits import RITS

        rits_models = {
            "llama3": RITS.LLAMA_3_3_70B_INSTRUCT,
            "granite4": RITS.GRANITE_4_H_SMALL,
            "mistral": RITS.MISTRAL_LARGE_3_675B_2512,
            "gpt-oss": RITS.GPT_OSS_120B,
        }
        if model_id not in rits_models:
            raise ValueError(
                f"Unknown RITS model shortcut: {model_id!r} "
                f"(expected one of {sorted(rits_models)})."
            )
        return build_backend("rits", model_id=rits_models[model_id])

    # ollama / vllm: pass the model id (served-model name for vllm) through.
    return build_backend(backend_kind, model_id=model_id, base_url=base_url)


def run(
    backend: Backend,
    *,
    input_file: str,
    output_dir: str,
    pipeline: str = "factreasoner",
    pipeline_version: str = "v2",
    service_type: str = "google",
    cache_dir: Optional[str] = None,
    dataset_name: Optional[str] = None,
    model_id: Optional[str] = None,
    top_k: int = 3,
    use_priors: bool = False,
    use_summarizer: bool = False,
    use_query_builder: bool = False,
    merlin_path: Optional[str] = None,
):
    """Run a factuality pipeline over a dataset with a pre-built backend.

    This is the importable core of the evaluation driver: the CLI (``__main__``)
    and the vLLM job entrypoint both construct a Mellea ``backend`` and then call
    this function.

    Args:
        backend: The Mellea backend that drives all components.
        input_file: Path to the input dataset (jsonl).
        output_dir: Directory for the output jsonl.
        pipeline: Factuality pipeline: ``factreasoner``, ``factscore``,
            ``veriscore`` or ``factverify``.
        pipeline_version: FactReasoner version ``v1``, ``v2`` or ``v3``.
        service_type: Retrieval service (``wikipedia``, ``chromadb``, ``google``).
        cache_dir: Optional cache directory for the retriever.
        dataset_name: Dataset name (used in the output filename).
        model_id: Model name/label recorded in the results and output filename.
        top_k: Top-k contexts retrieved per atom.
        use_priors: Use atom/context priors in the factor definition
            (FactReasoner only).
        use_summarizer: Summarize contexts (FactReasoner only).
        use_query_builder: Use the QueryBuilder for search queries.
        merlin_path: Path to the Merlin inference engine (FactReasoner only).
    """
    # FactReasoner versions:
    if pipeline_version == "v1":
        # 1 - context-atom relationships only, allow duplicated contexts
        rel_context_context = False
        remove_duplicates = False
        contexts_per_atom_only = True
    elif pipeline_version == "v2":
        # 2 - context-atom relationships only, no duplicated contexts
        rel_context_context = False
        remove_duplicates = True
        contexts_per_atom_only = False
    elif pipeline_version == "v3":
        # 3 - context-atom and context-context relationships, no duplicated contexts
        rel_context_context = True
        remove_duplicates = True
        contexts_per_atom_only = False
    else:
        raise ValueError(f"Unknown FactReasoner version: {pipeline_version}")

    # Create the atom extractor
    atom_extractor = Atomizer(backend)

    # Create the atom reviser
    atom_reviser = Reviser(backend)

    # Create the NLI extractor
    nli_extractor = NLIExtractor(backend)

    # Create the Query Builder
    query_builder = QueryBuilder(backend) if use_query_builder else None

    # Create context retriever and summarizer
    context_summarizer = ContextSummarizer(backend)
    context_retriever = ContextRetriever(
        service_type=service_type,
        top_k=top_k,
        cache_dir=cache_dir,
        query_builder=query_builder,
        fetch_text=True if pipeline != "factverify" else False,
    )

    print(f"Processing input dataset: {input_file}")

    # Load the dataset
    with open(input_file) as f:
        lines = f.read().splitlines()
    df_inter = pd.DataFrame(lines)
    df_inter.columns = ["json_element"]
    df_inter["json_element"].apply(json.loads)
    df = pd.json_normalize(df_inter["json_element"].apply(json.loads))
    dataset = df.to_dict("records")

    print(f"Loading data from: {input_file}")
    print(f"Found {len(dataset)} elements")

    # Set the pipeline name
    if pipeline in ["factscore", "factverify", "veriscore"]:
        pipeline_name = pipeline
    elif pipeline == "factreasoner":
        pipeline_name = f"{pipeline}-{pipeline_version}"
    else:
        raise ValueError(f"Unknown pipeline: {pipeline}. Aborting.")

    # Check if previous results exist. If yes, load them and skip over them
    # when processing the input dataset.
    filename = "eval_{}_{}_{}_{}.jsonl".format(
        pipeline_name, service_type, dataset_name, model_id
    )

    # Prepare the output file
    output_filename = os.path.join(output_dir, filename)
    print(f"Reading previous results from: {output_filename}")
    evaluation_data = []
    if os.path.isfile(output_filename):
        with open(output_filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                evaluation_data.append(json.loads(line))

    print(f"Found {len(evaluation_data)} existing evaluations data.")
    print(f"Using factuality pipeline: {pipeline_name}")

    # Loop over the data points in the dataset
    for input_data in dataset:
        # Check if current data has been processed already
        processed = False
        for eval_data in evaluation_data:
            if eval_data["input"] == input_data["input"]:
                processed = True
                break
        if processed:
            prompt = input_data["input"]
            print(f"Input: {prompt} already processed.")
            continue

        # Process the data point with the FactReasoner pipeline
        if pipeline == "factreasoner":
            pipeline_obj = FactReasoner(
                atom_extractor=atom_extractor,
                atom_reviser=atom_reviser,
                nli_extractor=nli_extractor,
                context_retriever=context_retriever,
                context_summarizer=context_summarizer,
                merlin_path=merlin_path,
                use_priors=use_priors,
            )
        elif pipeline == "factscore":
            pipeline_obj = FactScore(
                backend=backend,
                atom_extractor=atom_extractor,
                atom_reviser=atom_reviser,
                context_retriever=context_retriever,
            )
        elif pipeline == "veriscore":
            pipeline_obj = VeriScore(
                backend=backend,
                atom_extractor=atom_extractor,
                atom_reviser=atom_reviser,
                context_retriever=context_retriever,
            )
        elif pipeline == "factverify":
            pipeline_obj = FactVerify(
                atom_extractor=atom_extractor,
                atom_reviser=atom_reviser,
                context_retriever=context_retriever,
            )

        # Load the problem instance from a file or dict
        ok = pipeline_obj.from_dict_with_contexts(input_data)
        if not ok:
            continue  # annotations are null (ignore)

        # Build the pipeline and score
        if pipeline == "factreasoner":
            pipeline_obj.build(
                remove_duplicates=remove_duplicates,
                contexts_per_atom_only=contexts_per_atom_only,
                has_atoms=True,
                has_contexts=True,
                revise_atoms=False,
                rel_atom_context=True,
                rel_context_context=rel_context_context,
                summarize_contexts=use_summarizer,
            )

            results, marginals = pipeline_obj.score()
            results["model_name"] = model_id
            evaluation_data.append(results)
            print(f"[FactReasoner] Marginals: {marginals}")
            print(f"[FactReasoner] Results: {results}")
        elif pipeline == "factscore":
            pipeline_obj.build(has_atoms=True, has_contexts=True, revise_atoms=False)

            # Print the results
            results = pipeline_obj.score()
            results["model_name"] = model_id
            evaluation_data.append(results)
            print(f"[FactScore] Results: {results}")
        elif pipeline == "veriscore":
            pipeline_obj.build(has_atoms=True, has_contexts=True, revise_atoms=False)

            # Print the results
            results = pipeline_obj.score()
            results["model_name"] = model_id
            evaluation_data.append(results)
            print(f"[VeriScore] Results: {results}")
        elif pipeline == "factverify":
            pipeline_obj.build(has_atoms=True, has_contexts=True, revise_atoms=False)

            # Print the results
            results = pipeline_obj.score()
            results["model_name"] = model_id
            evaluation_data.append(results)
            print(f"[FactVerify] Results: {results}")

        # Save results to a file
        print(f"Writing results to: {output_filename}")
        with open(output_filename, "w") as f:
            for res in evaluation_data:
                f.write(f"{json.dumps(res)}\n")

    print("Done.")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the evaluation driver."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to the input dataset (jsonl).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Path to the output directory."
    )
    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Path to the cache directory."
    )
    parser.add_argument(
        "--dataset_name", type=str, default=None, help="Name of the dataset."
    )
    parser.add_argument(
        "--service_type",
        type=str,
        default="google",
        help="Service type (wikipedia, chromadb, google).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="rits",
        choices=["rits", "ollama", "vllm"],
        help="Which Mellea backend to use: 'rits' (remote IBM RITS, default), "
        "'ollama' (local Ollama server), or 'vllm' (vLLM OpenAI-compatible server).",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Name of the model used by the pipeline. For 'rits' this is a "
        "shortcut (llama3, granite4, mistral, gpt-oss); for 'ollama' and "
        "'vllm' it is passed through as the model / served-model name.",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Base URL for the 'vllm' backend (defaults to VLLM_BASE_URL env "
        "or http://localhost:8000/v1).",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="factreasoner",
        required=True,
        help="Factuality pipeline (factreasoner, factscore, veriscore, factverify).",
    )
    parser.add_argument(
        "--pipeline_version",
        type=str,
        default="v2",
        help="FactReasoner version: v1, v2 or v3",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Top k results retrieved as contexts per atom.",
    )
    parser.add_argument(
        "--use_priors",
        default=False,
        action="store_true",
        help="Use the atom and context priors in the factor definition.",
    )
    parser.add_argument(
        "--use_summarizer",
        default=False,
        action="store_true",
        help="Use the ContextSummarizer to summarize contexts (FactReasoner only).",
    )
    parser.add_argument(
        "--use_query_builder",
        default=False,
        action="store_true",
        help="Use the QueryBuilder to generate queries for Google search.",
    )
    parser.add_argument(
        "--merlin_path",
        type=str,
        default="/home/radu/git/fm-factual/lib/merlin",
        help="Path to the probabilistic inference engine merlin.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    # Create the Mellea backend via the shared factory.
    backend = build_backend_from_args(
        args.backend, model_id=args.model_id, base_url=args.base_url
    )

    run(
        backend,
        input_file=args.input_file,
        output_dir=args.output_dir,
        pipeline=args.pipeline,
        pipeline_version=args.pipeline_version,
        service_type=args.service_type,
        cache_dir=args.cache_dir,
        dataset_name=args.dataset_name,
        model_id=args.model_id,
        top_k=args.top_k,
        use_priors=args.use_priors,
        use_summarizer=args.use_summarizer,
        use_query_builder=args.use_query_builder,
        merlin_path=args.merlin_path,
    )
