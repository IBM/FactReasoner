# This is an example of using ContextRetriever for parallel context retrieval.

import argparse

# Local imports
from fact_reasoner.backends import build_backend
from fact_reasoner.core.query_builder import QueryBuilder
from fact_reasoner.core.retriever import ContextRetriever, Retriever
from fact_reasoner.core.base import Atom

# A set of atoms to retrieve contexts for, and a standalone query
ATOMS = {
    "a0": Atom(id="a0", text="The Eiffel Tower was completed in 1889."),
    "a1": Atom(id="a1", text="Marie Curie won two Nobel Prizes."),
    "a2": Atom(id="a2", text="The speed of light is approximately 300,000 km/s."),
}
QUERY = "Facts about famous landmarks and scientists"


def run_all(fast_retriever: ContextRetriever, atoms: dict, query: str) -> None:
    """Retrieve contexts for all atoms in parallel and print them."""

    contexts = fast_retriever.retrieve_all(atoms=atoms, query=query)
    print(f"\nTotal contexts retrieved: {len(contexts)}")
    for cid, context in contexts.items():
        print(context)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel context retriever example.")
    parser.add_argument(
        "--backend",
        choices=["rits", "ollama", "vllm"],
        default="rits",
        help="Which Mellea backend to use for the query builder: 'rits' "
        "(remote IBM RITS, default), 'ollama' (local Ollama server), or "
        "'vllm' (vLLM OpenAI-compatible server).",
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

    # Create the selected Mellea backend (used by the query builder)
    backend = build_backend(
        args.backend, model_id=args.served_model, base_url=args.base_url
    )

    # Build a query builder and retriever
    query_builder = QueryBuilder(backend)

    retriever = Retriever(
        top_k=3,
        service_type="google",
        cache_dir=None,
        fetch_text=True,
        use_in_memory_vectorstore=False,
        query_builder=query_builder,
        num_workers=4,
    )

    # Wrap the retriever for parallel retrieval across 4 worker threads
    fast_retriever = ContextRetriever(
        retriever=retriever,
        num_workers=4,
    )

    # Retrieve contexts for all atoms in parallel
    run_all(fast_retriever, ATOMS, QUERY)

    print("Done.")


if __name__ == "__main__":
    main()
