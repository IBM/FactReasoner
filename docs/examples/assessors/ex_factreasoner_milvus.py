# This is an example of running FactReasoner with retrieval from a Milvus
# vector index instead of live web search.
#
# Uses the same `fast` NLI mode as ex_factreasoner_fast.py -- see that file's
# docstring for what `fast` mode changes and trades off. The only structural
# difference here is the retriever: SourceRetriever(service_type="milvus")
# in place of service_type="google".


import argparse
import asyncio
import json
import os
from pathlib import Path

from fact_reasoner.assessor import FactReasoner

# Local imports
from fact_reasoner.backends import build_backend
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.nli import NLIExtractor
from fact_reasoner.core.nli_config import get_pair_config
from fact_reasoner.core.retriever import (
    EMBEDDING_MODEL,
    ContextRetriever,
    SourceRetriever,
)
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.core.summarizer import ContextSummarizer

# The NLI candidate-pair mode this example demonstrates.
NLI_MODE = "fast"

# Example query and response, used when no --input-file is given. Grounded in
# two indexed papers (SAFE/LongFact and FactBench) so retrieval has to pull
# from more than one source document to fully support the response.
QUERY = (
    "Summarize how the SAFE method evaluates long-form factuality, and "
    "briefly contrast it with FactBench's approach."
)
RESPONSE = (
    "SAFE (Search-Augmented Factuality Evaluator) decomposes a long-form model response into "
    "individual facts and checks each one against Google Search results, using LongFact -- a "
    "benchmark of 2,280 fact-seeking prompts across 38 topics -- for evaluation. On a set of "
    "roughly 16,000 individual facts, SAFE agrees with crowdsourced human annotators 72% of the "
    "time, and on a random subset of 100 disagreement cases, SAFE's judgment was preferred 76% "
    "of the time, while being more than 20 times cheaper than human annotation. In contrast, "
    "FactBench focuses on in-the-wild language model factuality by sourcing prompts from real "
    "user interactions rather than a static, pre-generated prompt set."
)
TOPIC = "Long-form factuality evaluation methods"


def main() -> None:
    # Select the Mellea backend from the command line (RITS by default).
    parser = argparse.ArgumentParser(
        description=f"FactReasoner assessor example against a Milvus index ({NLI_MODE} NLI mode)."
    )
    parser.add_argument(
        "--backend",
        choices=["rits", "ollama", "vllm", "openai"],
        default="rits",
        help="Which Mellea backend to use: 'rits' (remote IBM RITS, default), "
        "'ollama' (local Ollama server), 'vllm' (vLLM OpenAI-compatible server), "
        "or 'openai' (hosted frontier model: OpenAI, or Claude via --base-url "
        "https://api.anthropic.com/v1/).",
    )
    parser.add_argument(
        "--served-model",
        default=None,
        help="Model / served-model name. Optional: when omitted, build_backend "
        "uses the shared default model (Granite 4 Micro) for the chosen backend.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="API endpoint. For --backend vllm: the server base URL "
        "(defaults to VLLM_BASE_URL env or http://localhost:8000/v1). For "
        "--backend rits: a custom RITS endpoint, in which case --served-model "
        "is the raw RITS model name (RITS appends /v1; key from RITS_API_KEY).",
    )
    parser.add_argument(
        "--merlin-path",
        required=True,
        help="Path to the Merlin probabilistic inference binary (required).",
    )
    parser.add_argument(
        "--milvus-uri",
        default="http://localhost:19530",
        help="Milvus connection URI (see the author-reasoner project for how "
        "to reach a deployed instance), or a local Milvus Lite file path "
        "for testing.",
    )
    parser.add_argument(
        "--collection",
        default="factuality_papers",
        help="Milvus collection to retrieve contexts from (built by "
        "author-reasoner/milvus/examples/index_pdfs.py).",
    )
    parser.add_argument(
        "--embedding-model",
        default=EMBEDDING_MODEL,
        help="Must match whatever model indexed --collection.",
    )
    parser.add_argument(
        "--top-k", type=int, default=5, help="Number of contexts retrieved per atom."
    )

    # The two input modes are mutually exclusive: either assess a response from
    # scratch, or score precomputed atoms/contexts loaded from a file.
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--response",
        default=None,
        help="Response text to assess from scratch (atomize + retrieve contexts). "
        f"Defaults to a built-in example about {TOPIC}.",
    )
    source.add_argument(
        "--input-file",
        default=None,
        help="JSON file with precomputed atoms and contexts. Scores those "
        "directly -- no retrieval.",
    )

    parser.add_argument(
        "--query", default=QUERY, help="Query that produced the response."
    )
    parser.add_argument("--topic", default=TOPIC, help="Optional topic hint.")
    parser.add_argument(
        "--nli-cache-dir",
        default=None,
        help="Optional directory for the cross-run NLI verdict cache. "
        "Score-neutral: a cache hit returns the verdict the model already gave, "
        "so re-scoring the same data costs no LLM calls.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help=f"Where to write the pipeline JSON (default: "
        f"factreasoner_milvus_{NLI_MODE}_output.json beside this script).",
    )
    args = parser.parse_args()

    # When no --served-model is given, build_backend falls back to the shared
    # default model (Granite 4 Micro), resolved appropriately for the backend.
    backend = build_backend(
        args.backend, model_id=args.served_model, base_url=args.base_url
    )

    cwd = Path(__file__).resolve().parent

    # Create the retriever, atomizer and reviser. ContextRetriever wraps a
    # SourceRetriever, so build the SourceRetriever first.
    atom_extractor = Atomizer(backend)
    atom_reviser = Reviser(backend)
    retriever = SourceRetriever(
        service_type="milvus",
        collection_name=args.collection,
        persist_dir=args.milvus_uri,
        top_k=args.top_k,
        embedding_model=args.embedding_model,
    )
    context_summarizer = ContextSummarizer(backend)
    nli_extractor = NLIExtractor(backend)
    context_retriever = ContextRetriever(
        retriever=retriever,
        context_summarizer=context_summarizer,
        num_workers=4,
    )

    # Create the FactReasoner pipeline.
    pipeline = FactReasoner(
        context_retriever=context_retriever,
        context_summarizer=context_summarizer,
        atom_extractor=atom_extractor,
        atom_reviser=atom_reviser,
        nli_extractor=nli_extractor,
        merlin_path=args.merlin_path,
        nli_pair_config=get_pair_config(NLI_MODE),
        nli_cache_dir=args.nli_cache_dir,
    )

    # Flags shared by both input modes (FR2: atom-context relations only).
    build_kwargs = {
        "remove_duplicates": True,
        "contexts_per_atom_only": False,
        "rel_atom_context": True,
        "rel_context_context": False,
        "use_fast_retriever": True,
    }

    if args.input_file:
        # File mode: atoms and contexts are already computed, so skip atomization,
        # retrieval, revision and summarization -- just score what was loaded.
        json_file = args.input_file
        with open(json_file, "r") as f:
            data = json.load(f)

        print(f"[FactReasoner] Initializing the pipeline from {json_file}")
        pipeline.from_dict_with_contexts(data)

        build_kwargs.update(
            has_atoms=True,
            has_contexts=True,
            revise_atoms=False,
            summarize_contexts=False,
        )
    else:
        # Live mode: atomize the response, retrieve contexts, then score.
        build_kwargs.update(
            query=args.query,
            response=args.response or RESPONSE,
            topic=args.topic,
            has_atoms=False,
            has_contexts=False,
            revise_atoms=True,
            summarize_contexts=True,
        )

    # Build the FactReasoner pipeline (FR2 version). FactReasoner.build is async.
    print(f"[FactReasoner] NLI mode: {NLI_MODE}")
    print(
        f"[FactReasoner] Retrieving from Milvus collection '{args.collection}' "
        f"at {args.milvus_uri} (embedding_model={args.embedding_model})"
    )
    asyncio.run(pipeline.build(**build_kwargs))

    # score() returns a (results, marginals) pair for FactReasoner. (The baseline
    # assessors return just `results`.)
    results, marginals = pipeline.score()
    print(f"[FactReasoner] Marginals: {marginals}")
    print(f"[FactReasoner] Results: {results}")
    print(
        f"[FactReasoner] Factuality score ({NLI_MODE}): "
        f"{results['factuality_score']:.2%} over {results['num_atoms']} atoms"
    )

    # Save the pipeline to a JSON file
    output_file = args.output_file or os.path.join(
        cwd, f"factreasoner_milvus_{NLI_MODE}_output.json"
    )
    output = pipeline.to_json()
    output["results"] = results
    with open(output_file, "w") as fp:
        json.dump(output, fp, indent=4)
    print(f"Done. Wrote {output_file}")


if __name__ == "__main__":
    main()
