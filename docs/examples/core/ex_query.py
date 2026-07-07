# This is a simple example

import argparse

from mellea.backends import Backend, ModelOption

# Local imports
from fact_reasoner.core.query_builder import QueryBuilder

# The text (typically an atomic claim) to turn into a search query
TEXT = "rootstock for honey crisp apples in wayne county, ny"


def run_single(qb: QueryBuilder, text: str) -> None:
    """Build a search query for a single piece of text and print it."""

    result = qb.run(text)
    print(f"Query builder result: {result}")
    print(f"Initial Text: {text}")
    print(f"Query: {result}")


def build_backend(kind: str) -> Backend:
    """Create the Mellea backend to drive the query builder.

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
    parser = argparse.ArgumentParser(description="Query builder example.")
    parser.add_argument(
        "--backend",
        choices=["rits", "ollama"],
        default="rits",
        help="Which Mellea backend to use: 'rits' (remote IBM RITS, default) "
        "or 'ollama' (local Ollama server).",
    )
    args = parser.parse_args()

    # Create the selected Mellea backend
    backend = build_backend(args.backend)

    # Create the query builder
    qb = QueryBuilder(backend)

    # Build a query for a single piece of text
    run_single(qb, TEXT)

    print("Done.")


if __name__ == "__main__":
    main()
