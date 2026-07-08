import os
import json
import asyncio
import argparse
from pathlib import Path

# Local imports
from fact_reasoner.backends import build_backend
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.core.retriever import ContextRetriever, Retriever
from fact_reasoner.core.summarizer import ContextSummarizer
from fact_reasoner.core.nli import NLIExtractor
from fact_reasoner.core.query_builder import QueryBuilder
from fact_reasoner.assessor import FactReasoner


def main() -> None:
    # Select the Mellea backend from the command line (RITS by default).
    parser = argparse.ArgumentParser(description="FactReasoner (from file) example.")
    parser.add_argument(
        "--backend",
        choices=["rits", "ollama", "vllm"],
        default="rits",
        help="Which Mellea backend to use: 'rits' (remote IBM RITS, default), "
        "'ollama' (local Ollama server), or 'vllm' (vLLM OpenAI-compatible server).",
    )
    parser.add_argument(
        "--served-model",
        default=None,
        help="Model / served-model name (required for 'vllm').",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL for the 'vllm' backend (defaults to VLLM_BASE_URL env "
        "or http://localhost:8000/v1).",
    )
    args = parser.parse_args()

    backend = build_backend(
        args.backend, model_id=args.served_model, base_url=args.base_url
    )

    # Set cache dir for context retriever
    cache_dir = None  # "/home/radu/data/cache"
    cwd = Path(__file__).resolve().parent

    # Create the retriever, atomizer and reviser. ContextRetriever wraps a
    # Retriever, so build the Retriever first.
    qb = QueryBuilder(backend)
    atom_extractor = Atomizer(backend)
    atom_reviser = Reviser(backend)
    retriever = Retriever(
        service_type="google",
        top_k=5,
        cache_dir=cache_dir,
        fetch_text=True,
        query_builder=qb,
        num_workers=4,
    )
    context_summarizer = ContextSummarizer(backend)
    nli_extractor = NLIExtractor(backend)
    context_retriever = ContextRetriever(
        retriever=retriever,
        context_summarizer=context_summarizer,
        num_workers=4,
    )

    # Path to merlin (probabilistic inference engine)
    merlin_path = os.path.join(os.getcwd(), "lib", "merlin")  # Linux RedHat version

    # Create the FactReasoner pipeline
    pipeline = FactReasoner(
        context_retriever=context_retriever,
        context_summarizer=context_summarizer,
        atom_extractor=atom_extractor,
        atom_reviser=atom_reviser,
        nli_extractor=nli_extractor,
        merlin_path=merlin_path,
    )

    # Load the problem instance from a file (contains precomputed atoms/contexts)
    json_file = os.path.join(cwd, "flaherty_wikipedia.json")
    with open(json_file, "r") as f:
        data = json.load(f)

    print(f"[FactReasoner] Initializing the pipeline from {json_file}")
    pipeline.from_dict_with_contexts(data)

    # Build the FactReasoner pipeline (FR2 version). FactReasoner.build is async.
    asyncio.run(
        pipeline.build(
            has_atoms=True,
            has_contexts=True,
            revise_atoms=False,
            remove_duplicates=True,
            summarize_contexts=False,
            contexts_per_atom_only=False,
            rel_atom_context=True,
            rel_context_context=False,
        )
    )

    # Print the results
    results, marginals = pipeline.score()
    print(f"[FactReasoner] Marginals: {marginals}")
    print(f"[FactReasoner] Results: {results}")

    # Save the pipeline to a JSON file
    output_file = os.path.join(cwd, "factreasoner_output.json")
    output = pipeline.to_json()
    output["results"] = results
    with open(output_file, "w") as fp:
        json.dump(output, fp, indent=4)
    print("Done.")


if __name__ == "__main__":
    main()
