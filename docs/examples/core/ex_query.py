# This is a simple example

import argparse

# Local imports
from fact_reasoner.backends import build_backend
from fact_reasoner.core.query_builder import QueryBuilder

# The text (typically an atomic claim) to turn into a search query
TEXT = "rootstock for honey crisp apples in wayne county, ny"


def run_single(qb: QueryBuilder, text: str) -> None:
    """Build a search query for a single piece of text and print it."""

    result = qb.run(text)
    print(f"Query builder result: {result}")
    print(f"Initial Text: {text}")
    print(f"Query: {result}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query builder example.")
    parser.add_argument(
        "--backend",
        choices=["rits", "ollama", "vllm", "openai"],
        default="rits",
        help="Which Mellea backend to use: 'rits' (remote IBM RITS, default), "
        "'ollama' (local Ollama server), 'vllm' (vLLM OpenAI-compatible "
        "server), or 'openai' (hosted frontier model: OpenAI, or Claude via "
        "--base-url https://api.anthropic.com/v1/).",
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

    # Create the selected Mellea backend
    backend = build_backend(
        args.backend, model_id=args.served_model, base_url=args.base_url
    )

    # Create the query builder
    qb = QueryBuilder(backend)

    # Build a query for a single piece of text
    run_single(qb, TEXT)

    print("Done.")


if __name__ == "__main__":
    main()
