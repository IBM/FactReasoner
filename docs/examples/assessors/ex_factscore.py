import os
import json
import argparse
from pathlib import Path

# Local imports
from fact_reasoner.backends import build_backend
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.core.retriever import ContextRetriever, Retriever
from fact_reasoner.core.query_builder import QueryBuilder
from fact_reasoner.baselines.factscore import FactScore

# Example query and response
QUERY = "Tell me a biography of Lanny Flaherty"
RESPONSE = 'Lanny Flaherty is an American actor born on December 18, 1949, in Pensacola, Florida. He has appeared in numerous films, television shows, and theater productions throughout his career, which began in the late 1970s. Some of his notable film credits include "King of New York," "The Abyss," "Natural Born Killers," "The Game," and "The Straight Story." On television, he has appeared in shows such as "Law & Order," "The Sopranos," "Boardwalk Empire," and "The Leftovers." Flaherty has also worked extensively in theater, including productions at the Public Theater and the New York Shakespeare Festival. He is known for his distinctive looks and deep gravelly voice, which have made him a memorable character actor in the industry.'
TOPIC = "Lanny Flaherty"


def main() -> None:
    # Select the Mellea backend from the command line (RITS by default).
    parser = argparse.ArgumentParser(description="FactScore assessor example.")
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

    # Create the retriever, atomizer and reviser.
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
    context_retriever = ContextRetriever(retriever=retriever, num_workers=4)

    # Create the FactScore pipeline
    pipeline = FactScore(
        backend=backend,
        context_retriever=context_retriever,
        atom_extractor=atom_extractor,
        atom_reviser=atom_reviser,
    )

    # Build the scorer
    pipeline.build(
        query=QUERY,
        response=RESPONSE,
        topic=TOPIC,
        has_atoms=False,
        has_contexts=False,
        revise_atoms=True,
        use_fast_retriever=True,
    )

    # Print the results
    results = pipeline.score()
    print(f"[FactScore] Results: {results}")

    # Save the pipeline to a JSON file
    output_file = os.path.join(cwd, "factscore_output.json")
    output = pipeline.to_json()
    output["results"] = results
    with open(output_file, "w") as fp:
        json.dump(output, fp, indent=4)
    print("Done.")


if __name__ == "__main__":
    main()
