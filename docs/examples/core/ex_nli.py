# This is a simple example

import argparse
import asyncio

# Local imports
from fact_reasoner.backends import build_backend
from fact_reasoner.core.nli import NLIExtractor

# A single premise/hypothesis pair to evaluate
PREMISE = "natural born killers is a 1994 american romantic crime action film \
    directed by oliver stone and starring woody harrelson, juliette lewis, \
    robert downey jr., tommy lee jones, and tom sizemore. the film tells the \
    story of two victims of traumatic childhoods who become lovers and mass \
    murderers, and are irresponsibly glorified by the mass media. the film is \
    based on an original screenplay by quentin tarantino that was heavily \
    revised by stone, writer david veloz, and associate producer richard \
    rutowski. natural born killers was released on august 26, 1994 in the \
    united states, and screened at the venice film festival on august 29, 1994."
HYPOTHESIS = "Lanny Flaherty has appeared in numerous films."

# A batch of premise/hypothesis pairs to evaluate
PREMISES = [
    "The biggest risk facing the world's insurance companies is possibly the \
    rapid change now taking place within their own ranks. Sluggish growth in \
    core markets and intense price competition, coupled with shifting patterns \
    of customer demand and the rising cost of losses, are threatening to \
    overwhelm those too slow to react.",
    "The biggest risk facing the world's insurance companies is possibly the \
    rapid change now taking place within their own ranks. Sluggish growth in \
    core markets and intense price competition, coupled with shifting patterns \
    of customer demand and the rising cost of losses, are threatening to \
    overwhelm those too slow to react.",
    "The biggest risk facing the world's insurance companies is possibly the \
    rapid change now taking place within their own ranks. Sluggish growth in \
    core markets and intense price competition, coupled with shifting patterns \
    of customer demand and the rising cost of losses, are threatening to \
    overwhelm those too slow to react.",
]
HYPOTHESES = [
    "Insurance companies are experiencing a boom in their core markets.",
    "Insurance companies are competing to provide the best service to their customers.",
    "Customers don't trust insurance companies as much as they once were.",
]


def run_single(extractor: NLIExtractor, premise: str, hypothesis: str) -> None:
    """Evaluate the entailment for a single premise/hypothesis pair."""

    result = extractor.run(premise=premise, hypothesis=hypothesis)
    print(f"H -> P: {result}")


async def run_batch(
    extractor: NLIExtractor, premises: list[str], hypotheses: list[str]
) -> None:
    """Evaluate the entailment for a batch of premise/hypothesis pairs.

    run_batch is throttled and failure-resilient:
      - requests are rate-limited (default 1500/min) and run with bounded
        concurrency, so large batches do not trigger provider rate limits;
      - if a single request fails, that item falls back to a neutral
        relationship instead of aborting the whole batch;
      - the returned list is positionally aligned with the input pairs.
    """

    print("Process a batch of premise/hypothesis pairs ...")
    results = await extractor.run_batch(premises=premises, hypotheses=hypotheses)
    for i, result in enumerate(results):
        print(f"Pair {i} -> {result}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NLI extractor example.")
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
        help="Base URL for the 'vllm' backend (defaults to VLLM_BASE_URL env "
        "or http://localhost:8000/v1).",
    )
    args = parser.parse_args()

    # Create the selected Mellea backend
    backend = build_backend(
        args.backend, model_id=args.served_model, base_url=args.base_url
    )

    # Create the NLI extractor
    extractor = NLIExtractor(backend)

    # Single pair processing
    run_single(extractor, PREMISE, HYPOTHESIS)

    # Batch processing
    asyncio.run(run_batch(extractor, PREMISES, HYPOTHESES))

    print("Done.")


if __name__ == "__main__":
    main()
