# This is an example of using ContextRetriever for parallel context retrieval.

import argparse

from mellea.backends import Backend, ModelOption

# Local imports
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
    parser = argparse.ArgumentParser(
        description="Parallel context retriever example."
    )
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
