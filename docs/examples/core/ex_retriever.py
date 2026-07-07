# This is a simple example

import argparse

from mellea.backends import Backend, ModelOption

# Local imports
from fact_reasoner.core.query_builder import QueryBuilder
from fact_reasoner.core.retriever import Retriever

# The text to retrieve supporting contexts for
QUERY_TEXT = "rootstock for honey crisp apples in wayne county, ny"


def run_single(retriever: Retriever, query_text: str) -> None:
    """Retrieve and print contexts for a single query."""

    contexts = retriever.query(text=query_text)
    print(f"Number of contexts: {len(contexts)}")
    for context in contexts:
        print(context)


def build_backend(kind: str) -> Backend:
    """Create the Mellea backend used by the query builder.

    Args:
        kind: str
            Which backend to build: "rits" for the remote IBM RITS service, or
            "ollama" for a local Ollama server.
    Returns:
        Backend: A ready-to-use Mellea backend.
    """
    if kind == "rits":
        # Remote IBM RITS backend (requires the mellea_ibm package and RITS
        # credentials/config in the environment).
        from mellea_ibm.rits import RITSBackend, RITS

        return RITSBackend(
            RITS.LLAMA_3_3_70B_INSTRUCT,
            model_options={ModelOption.MAX_NEW_TOKENS: 4096},
        )
    elif kind == "ollama":
        # Local Ollama backend (requires a running Ollama server at
        # http://localhost:11434; the model is pulled on first use). Pass
        # base_url=... to OllamaModelBackend to target a non-default host.
        from mellea.backends.ollama import OllamaModelBackend
        from mellea.backends.model_ids import IBM_GRANITE_4_MICRO_3B

        return OllamaModelBackend(
            IBM_GRANITE_4_MICRO_3B,
            model_options={ModelOption.MAX_NEW_TOKENS: 4096},
        )
    else:
        raise ValueError(f"Unknown backend: {kind!r} (expected 'rits' or 'ollama')")


def main() -> None:
    parser = argparse.ArgumentParser(description="Context retriever example.")
    parser.add_argument(
        "--backend",
        choices=["rits", "ollama"],
        default="rits",
        help="Which Mellea backend to use for the query builder: 'rits' "
        "(remote IBM RITS, default) or 'ollama' (local Ollama server).",
    )
    args = parser.parse_args()

    # Create the selected Mellea backend (used by the query builder)
    backend = build_backend(args.backend)

    # Build a query builder and retriever
    query_builder = QueryBuilder(backend)
    cache_dir = None  # e.g. "my_database.db" to cache results

    retriever = Retriever(
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
