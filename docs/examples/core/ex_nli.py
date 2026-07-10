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

# A premise (e.g. a retrieved context) and a hypothesis (e.g. an atom).
# Ground truth: entailment
PREMISE1 = (
    "Robert Haldane Smith, Baron Smith of Kelvin, is a British businessman and "
    "former Governor of the British Broadcasting Corporation."
)
HYPOTHESIS1 = "Robert Smith holds the title of Baron Smith of Kelvin."

# Ground truth: neutral
PREMISE2 = (
    "Some time on the night of October 1st, the Copacabana Club was burnt to "
    "the ground. The police are treating the fire as suspicious. The only facts "
    "known at this stage are: The club was insured for more than its real value. "
    "The club belonged to John Hodges. Les Braithwaite was known to dislike "
    "John Hodges. Between October 1st and October 2nd, Les Braithwaite was away "
    "from home on a business trip. There were no fatalities. A plan of the club "
    "was found in Les Braithwaite's flat."
)
HYPOTHESIS2 = "If the insurance company pays out in full, John Hodges stands to profit from the fire."

# Ground truth: contradiction
PREMISE3 = (
    "To determine whether interbreeding took place among Homo species before the "
    "populations that became modern humans left Africa, evolutionary biologists "
    "studied DNA from two African hunter-gatherer groups, the Biaka Pygmies and "
    "the San, and from a West African agricultural population, the Mandenka. "
    "Each of these groups is descended from populations thought to have remained "
    "in Africa, meaning they would have avoided the genetic bottleneck effect "
    "that usually occurs with migration. This means the groups show particularly "
    "high genetic diversity, which makes their genomes more likely to have retained "
    "evidence of ancient genetic mixing. The researchers looked at 61 non-coding DNA "
    "regions in all three groups. Because direct comparison to archaic specimens wasn't "
    "possible, the authors used computer models to simulate how infiltration from different "
    "populations might have affected patterns of variation within modern genomes. On "
    "chromosomes 4, 13 and 18 of the three African populations, the researchers found "
    "genetic regions that were more divergent on average than known modern sequences "
    "at the same locations, hinting at a different origin."
)
HYPOTHESIS3 = "Since the genetic diversity of the three African populations was high, while that of the indigenous population was low, researchers concluded that the three African populations had interbred."

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

    # Create the NLI extractor
    extractor = NLIExtractor(backend)

    # Single pair processing
    run_single(extractor, PREMISE, HYPOTHESIS)

    # Batch processing
    asyncio.run(run_batch(extractor, PREMISES, HYPOTHESES))

    print("****" * 20)

    # Build the NLI extractor with the logprobs method.
    nli = NLIExtractor(
        backend,
        nli_method="logprobs",
    )

    # Predict the NLI relationship and its probability.
    result = nli.run(premise=PREMISE1, hypothesis=HYPOTHESIS1)
    print(f"Premise:    {PREMISE1}")
    print(f"Hypothesis: {HYPOTHESIS1}")
    print(f"Label:       {result['label']}")
    print(f"Probability: {result['probability']:.4f}")

    result = nli.run(premise=PREMISE2, hypothesis=HYPOTHESIS2)
    print(f"Premise:    {PREMISE2}")
    print(f"Hypothesis: {HYPOTHESIS2}")
    print(f"Label:       {result['label']}")
    print(f"Probability: {result['probability']:.4f}")

    result = nli.run(premise=PREMISE3, hypothesis=HYPOTHESIS3)
    print(f"Premise:    {PREMISE3}")
    print(f"Hypothesis: {HYPOTHESIS3}")
    print(f"Label:       {result['label']}")
    print(f"Probability: {result['probability']:.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
