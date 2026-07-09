# This is a simple example

import argparse
import asyncio

# Local imports
from fact_reasoner.backends import build_backend
from fact_reasoner.core.reviser import Reviser

# The original response that provides context for decontextualization
RESPONSE = 'Lanny Flaherty is an American actor born on December 18, 1949, \
    in Pensacola, Florida. He has appeared in numerous films, television \
    shows, and theater productions throughout his career, which began in the \
    late 1970s. Some of his notable film credits include "King of New York," \
    "The Abyss," "Natural Born Killers," "The Game," and "The Straight Story." \
    On television, he has appeared in shows such as "Law & Order," "The Sopranos," \
    "Boardwalk Empire," and "The Leftovers." Flaherty has also worked \
    extensively in theater, including productions at the Public Theater and \
    the New York Shakespeare Festival. He is known for his distinctive looks \
    and deep gravelly voice, which have made him a memorable character \
    actor in the industry.'

# Atomic units with vague references to be decontextualized
ATOMS = [
    "He has appeared in numerous films.",
    "He has appeared in numerous television shows.",
    "He has appeared in numerous theater productions.",
    "His career began in the late 1970s.",
]


def print_results(result: list) -> None:
    """Print the revised atomic units."""

    print(f"Number of revised atomic units: {len(result)}")
    for atom in result:
        print(f"Original Atom: {atom['text']}")
        print(f"Revised Atom:  {atom['revised_unit']}")
        print(f"Rationale: {atom['rationale']}")
        print("-----")


def run_single(reviser: Reviser, atoms: list[str], response: str) -> None:
    """Decontextualize a list of atoms synchronously and print them."""

    result = reviser.run(atoms, response)
    print(f"Reviser result: {result}")
    print_results(result)


async def run_batch(reviser: Reviser, atoms: list[str], response: str) -> None:
    """Decontextualize a batch of atoms and print them.

    run_batch is throttled and failure-resilient:
      - requests are rate-limited (default 1500/min) and run with bounded
        concurrency, so large batches do not trigger provider rate limits;
      - if a single request fails or produces unparsable output, that item
        falls back to a no-op revision (the original atom) instead of aborting
        the whole batch;
      - the returned list is positionally aligned with `atoms`.
    """

    print("Process a batch of atoms ...")
    result = await reviser.run_batch(atoms, response)
    print_results(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reviser example.")
    parser.add_argument(
        "--backend",
        choices=["rits", "ollama", "vllm"],
        default="rits",
        help="Which Mellea backend to use: 'rits' (remote IBM RITS, default), "
        "'ollama' (local Ollama server), or 'vllm' (vLLM OpenAI-compatible "
        "server).",
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

    # Create the reviser
    reviser = Reviser(backend=backend)

    # Single (synchronous) processing
    run_single(reviser, ATOMS, RESPONSE)

    # Batch processing
    asyncio.run(run_batch(reviser, ATOMS, RESPONSE))

    print("Done.")


if __name__ == "__main__":
    main()
