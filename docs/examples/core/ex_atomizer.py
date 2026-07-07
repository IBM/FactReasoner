# This is a simple example

import argparse
import asyncio

from mellea.backends import Backend, ModelOption

# Local imports
from fact_reasoner.core.atomizer import Atomizer

# A single response to process
RESPONSE = "The Apollo 14 mission to the Moon took place on January 31, 1971. \
    This mission was significant as it marked the third time humans set \
    foot on the lunar surface, with astronauts Alan Shepard and Edgar \
    Mitchell joining Captain Stuart Roosa, who had previously flown on \
    Apollo 13. The mission lasted for approximately 8 days, during which \
    the crew conducted various experiments and collected samples from the \
    lunar surface. Apollo 14 brought back approximately 70 kilograms of \
    lunar material, including rocks, soil, and core samples, which have \
    been invaluable for scientific research ever since."

# A batch of responses to process
RESPONSES = [
    "The Apollo 14 mission to the Moon took place on January 31, 1971. \
    This mission was significant as it marked the third time humans set \
    foot on the lunar surface, with astronauts Alan Shepard and Edgar \
    Mitchell joining Captain Stuart Roosa, who had previously flown on \
    Apollo 13. The mission lasted for approximately 8 days, during which \
    the crew conducted various experiments and collected samples from the \
    lunar surface. Apollo 14 brought back approximately 70 kilograms of \
    lunar material, including rocks, soil, and core samples, which have \
    been invaluable for scientific research ever since.",
    "Lanny Flaherty is an American actor born on December 18, 1949, in \
    Pensacola, Florida. He has appeared in numerous films, television \
    shows, and theater productions throughout his career, which began in \
    the late 1970s. Some of his notable film credits include \"King of New \
    York,\" \"The Abyss,\" \"Natural Born Killers,\" \"The Game,\" \
    and \"The Straight Story.\" On television, he has appeared in shows \
    such as \"Law & Order,\" \"The Sopranos,\" \"Boardwalk Empire,\" \
    and \"The Leftovers.\" Flaherty has also worked extensively in theater, \
    including productions at the Public Theater and the New York Shakespeare \
    Festival. He is known for his distinctive looks and deep gravelly \
    voice, which have made him a memorable character actor in the industry."
]


def run_single(atomizer: Atomizer, response: str) -> None:
    """Extract atomic units from a single response and print them."""

    # Process the response to extract atomic units
    result = atomizer.run(response)
    print(f"Atomization result: {result}")

    # Print the extracted atomic units
    print(f"Extracted {len(result)} atomic units:")
    for k, v in result.items():
        print(f"Atom {k}: {v}")


async def run_batch(atomizer: Atomizer, responses: list[str]) -> None:
    """Extract atomic units from a batch of responses and print them.

    run_batch is throttled and failure-resilient:
      - requests are rate-limited (default 1500/min) and run with bounded
        concurrency, so large batches do not trigger provider rate limits;
      - if a single request fails (backend/network error) or produces
        unparsable output, that item comes back as an empty dict {} instead of
        aborting the whole batch;
      - the returned list is positionally aligned with `responses` (same length,
        same order), so results[i] always corresponds to responses[i].
    """

    print("Process a batch of responses ...")
    results = await atomizer.run_batch(responses)
    for i, result in enumerate(results):
        if not result:
            print(f"Response {i}: no atoms extracted (failed or empty)")
            continue
        print(f"Response {i}: extracted {len(result)} atomic units:")
        for k, v in result.items():
            print(f"Atom {k}: {v}")


def build_backend(kind: str) -> Backend:
    """Create the Mellea backend to drive the atomizer.

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
    parser = argparse.ArgumentParser(description="Atomizer example.")
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

    # Create the atomizer
    atomizer = Atomizer(backend=backend)

    # Single-response processing
    run_single(atomizer, RESPONSE)

    # Batch processing
    asyncio.run(run_batch(atomizer, RESPONSES))

    print("Done.")


if __name__ == "__main__":
    main()
