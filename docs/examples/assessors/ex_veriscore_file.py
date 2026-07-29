import os
import json
import argparse
from pathlib import Path

# Local imports
from fact_reasoner.backends import build_backend
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.core.retriever import ContextRetriever, SourceRetriever
from fact_reasoner.core.query_builder import QueryBuilder
from fact_reasoner.baselines.veriscore import VeriScore


def main() -> None:
    # Select the Mellea backend from the command line (RITS by default).
    parser = argparse.ArgumentParser(description="VeriScore (from file) example.")
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
        help="Model / served-model name. Optional: when omitted, the shared "
        "default model (Granite 4 Micro) is used for the chosen backend.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="API endpoint. For --backend vllm: the server base URL "
        "(defaults to VLLM_BASE_URL env or http://localhost:8000/v1). For "
        "--backend rits: a custom RITS endpoint, in which case --served-model "
        "is the raw RITS model name (RITS appends /v1; key from RITS_API_KEY).",
    )
    args = parser.parse_args()

    backend = build_backend(
        args.backend, model_id=args.served_model, base_url=args.base_url
    )

    # Set cache dir for context retriever
    cache_dir = None  # "/home/radu/data/cache"
    cwd = Path(__file__).resolve().parent

    # Create the retriever, atomizer and reviser. ContextRetriever wraps a
    # SourceRetriever, so build the SourceRetriever first.
    qb = QueryBuilder(backend)
    atom_extractor = Atomizer(backend)
    atom_reviser = Reviser(backend)
    retriever = SourceRetriever(
        service_type="google",
        top_k=5,
        cache_dir=cache_dir,
        fetch_text=True,
        query_builder=qb,
        num_workers=4,
    )
    context_retriever = ContextRetriever(retriever=retriever, num_workers=4)

    # Create the VeriScore pipeline
    pipeline = VeriScore(
        backend=backend,
        context_retriever=context_retriever,
        atom_extractor=atom_extractor,
        atom_reviser=atom_reviser,
    )

    # Load the problem instance from a file (contains precomputed atoms/contexts)
    json_file = os.path.join(cwd, "flaherty_wikipedia.json")
    with open(json_file, "r") as f:
        data = json.load(f)

    print(f"[VeriScore] Initializing pipeline from: {json_file}")
    pipeline.from_dict_with_contexts(data)

    # Build the scorer (atoms and contexts already present in the file)
    pipeline.build(has_atoms=True, has_contexts=True, revise_atoms=False)

    # Print the results
    results = pipeline.score()
    print(f"[VeriScore] Results: {results}")

    # Save the pipeline to a JSON file
    output_file = os.path.join(cwd, "veriscore_output.json")
    output = pipeline.to_json()
    output["results"] = results
    with open(output_file, "w") as fp:
        json.dump(output, fp, indent=4)
    print("Done.")


if __name__ == "__main__":
    main()
