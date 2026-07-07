# This is a simple example

import argparse
import asyncio

from mellea.backends import Backend, ModelOption

# Local imports
from fact_reasoner.core.summarizer import ContextSummarizer

# An atomic claim used when summarizing with reference
ATOM = "The city council has approved new regulations for electric scooters."

# Contexts summarized with reference to the atom (relevant, partially relevant,
# empty, and irrelevant passages).
CONTEXTS_WITH_REF = [
    "In the past year, the city had seen a rapid increase in the use of \
    electric scooters. They seemed like a perfect solution to reduce traffic \
    and provide an eco-friendly transportation option. However, problems arose \
    quickly. Riders often ignored traffic laws, riding on sidewalks, and \
    causing accidents. Additionally, the scooters were frequently left \
    haphazardly around public spaces, obstructing pedestrians. City officials \
    were under increasing pressure to act, and after numerous public \
    consultations and debates, the council finally passed new regulations. \
    The new rules included mandatory helmet use, restricted riding areas, \
    and designated parking zones for scooters. The implementation of these \
    regulations was expected to improve safety and the overall experience for \
    both scooter users and pedestrians.",
    "With the rise of shared electric scooters and bikes in cities across the \
    country, municipal governments have been scrambling to develop effective \
    policies to handle this new form of transportation. Many cities, including \
    the local area, were caught off guard by the sudden popularity of \
    scooters, and their original infrastructure was ill-prepared for this new \
    trend. The city council's recent approval of new regulations was part of a \
    larger effort to stay ahead of the curve and provide a balanced approach \
    to regulating modern transportation options while encouraging their growth.",
    "",
    "The sun hung low in the sky, casting a warm golden glow over the city as \
    Emily wandered through the bustling streets, her mind drifting between \
    thoughts of the past and the uncertain future. She passed the familiar \
    old bookstore that always smelled like aged paper and adventure, a place \
    she used to frequent with her grandmother, whose absence still left a \
    hollow ache in her chest.",
]

# A single context summarized without reference (generic summarization)
CONTEXT_WITHOUT_REF = """In the past year, the city had seen a rapid increase in the \
use of electric scooters. They seemed like a perfect solution to reduce \
traffic and provide an eco-friendly transportation option. However, \
problems arose quickly. Riders often ignored traffic laws, riding on \
sidewalks, and causing accidents. Additionally, the scooters were frequently \
left haphazardly around public spaces, obstructing pedestrians. City officials \
were under increasing pressure to act, and after numerous public \
consultations and debates, the council finally passed new regulations. \
The new rules included mandatory helmet use, restricted riding areas, and \
designated parking zones for scooters. The implementation of these regulations \
was expected to improve safety and the overall experience for both scooter \
users and pedestrians."""


def print_results(result: list) -> None:
    """Print each context, its summary, and its relevance probability."""

    for i, elem in enumerate(result):
        context = elem["context"]
        summary = elem["summary"]
        probability = elem["probability"]
        print(
            f"\n\nContext #{i + 1}: {context}"
            f"\n--> Summary #{i + 1}: {summary}"
            f"\n--> Probability #{i + 1}: {probability}"
        )


async def run_with_reference(summarizer: ContextSummarizer) -> None:
    """Summarize each context relative to a specific atomic claim.

    Whether summarization is done with respect to an atom is controlled by
    passing ``atom_text`` to ``run_batch`` (not by a constructor flag).
    """

    print("Summarizing contexts WITH reference to an atom ...")
    result = await summarizer.run_batch(CONTEXTS_WITH_REF, ATOM)
    print(f"Summarizer result: {result}")
    print_results(result)


async def run_without_reference(summarizer: ContextSummarizer) -> None:
    """Summarize a single context independently (no reference atom)."""

    print("Summarizing a context WITHOUT reference ...")
    result = await summarizer.run_batch([CONTEXT_WITHOUT_REF], None)
    print(f"Summarizer result: {result}")
    print_results(result)


def build_backend(kind: str) -> Backend:
    """Create the Mellea backend to drive the context summarizer.

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
    parser = argparse.ArgumentParser(description="Context summarizer example.")
    parser.add_argument(
        "--backend",
        choices=["rits", "ollama"],
        default="rits",
        help="Which Mellea backend to use: 'rits' (remote IBM RITS, default) "
        "or 'ollama' (local Ollama server).",
    )
    parser.add_argument(
        "--with-reference",
        action="store_true",
        help="Summarize contexts relative to a reference atom instead of "
        "summarizing a single context independently.",
    )
    args = parser.parse_args()

    # Create the selected Mellea backend
    backend = build_backend(args.backend)

    # Create the context summarizer
    summarizer = ContextSummarizer(backend=backend)

    if args.with_reference:
        asyncio.run(run_with_reference(summarizer))
    else:
        asyncio.run(run_without_reference(summarizer))

    print("Done.")


if __name__ == "__main__":
    main()
