# This is a simple example

import argparse
import asyncio

# Local imports
from fact_reasoner.backends import build_backend
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
    'Lanny Flaherty is an American actor born on December 18, 1949, in \
    Pensacola, Florida. He has appeared in numerous films, television \
    shows, and theater productions throughout his career, which began in \
    the late 1970s. Some of his notable film credits include "King of New \
    York," "The Abyss," "Natural Born Killers," "The Game," \
    and "The Straight Story." On television, he has appeared in shows \
    such as "Law & Order," "The Sopranos," "Boardwalk Empire," \
    and "The Leftovers." Flaherty has also worked extensively in theater, \
    including productions at the Public Theater and the New York Shakespeare \
    Festival. He is known for his distinctive looks and deep gravelly \
    voice, which have made him a memorable character actor in the industry.',
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Atomizer example.")
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

    # Create the atomizer
    atomizer = Atomizer(backend=backend)

    # Single-response processing
    run_single(atomizer, RESPONSE)

    # Batch processing
    asyncio.run(run_batch(atomizer, RESPONSES))

    print("Done.")


if __name__ == "__main__":
    main()
