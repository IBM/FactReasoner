# This is a simple example

import argparse
import asyncio

from mellea.backends import Backend, ModelOption

# Local imports
from fact_reasoner.core.reviser import Reviser

# The original response that provides context for decontextualization
RESPONSE = "Lanny Flaherty is an American actor born on December 18, 1949, \
    in Pensacola, Florida. He has appeared in numerous films, television \
    shows, and theater productions throughout his career, which began in the \
    late 1970s. Some of his notable film credits include \"King of New York,\" \
    \"The Abyss,\" \"Natural Born Killers,\" \"The Game,\" and \"The Straight Story.\" \
    On television, he has appeared in shows such as \"Law & Order,\" \"The Sopranos,\" \
    \"Boardwalk Empire,\" and \"The Leftovers.\" Flaherty has also worked \
    extensively in theater, including productions at the Public Theater and \
    the New York Shakespeare Festival. He is known for his distinctive looks \
    and deep gravelly voice, which have made him a memorable character \
    actor in the industry."

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


def build_backend(kind: str) -> Backend:
    """Create the Mellea backend to drive the reviser.

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
    parser = argparse.ArgumentParser(description="Reviser example.")
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

    # Create the reviser
    reviser = Reviser(backend=backend)

    # Single (synchronous) processing
    run_single(reviser, ATOMS, RESPONSE)

    # Batch processing
    asyncio.run(run_batch(reviser, ATOMS, RESPONSE))

    print("Done.")


if __name__ == "__main__":
    main()
