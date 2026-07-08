# This is a simple example

import argparse

# Local imports
from fact_reasoner.backends import build_backend
from fact_reasoner.core.query_builder import QueryBuilder
from fact_reasoner.core.retriever import SourceRetriever

# The text to retrieve supporting contexts for
QUERY_TEXT = "rootstock for honey crisp apples in wayne county, ny"


def run_single(retriever: SourceRetriever, query_text: str) -> None:
    """Retrieve and print contexts for a single query."""

    contexts = retriever.query(text=query_text)
    print(f"Number of contexts: {len(contexts)}")
    for context in contexts:
        print(context)


def main() -> None:
    parser = argparse.ArgumentParser(description="Context retriever example.")
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
    cache_dir = None  # e.g. "my_database.db" to cache results

    retriever = SourceRetriever(
        top_k=10,
        service_type="google",
        cache_dir=cache_dir,
        fetch_text=True,
        use_in_memory_vectorstore=False,
        query_builder=query_builder,
    )

    # Retrieve contexts for a single query
    run_single(retriever, QUERY_TEXT)

    print("Done.")


if __name__ == "__main__":
    main()
