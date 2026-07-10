# This is an example of estimating NLI relation probabilities with SIMBA-UQ.
#
# The default (logprobs) NLI method needs a backend that exposes token
# logprobs. Ollama does not, so on Ollama every NLI call degrades to a fixed
# neutral relation. The SIMBA-UQ method estimates the probability of the
# predicted label via self-consistency (sampling across temperatures and
# scoring by consensus) and works on any backend.

import argparse

# Local imports
from fact_reasoner.backends import build_backend
from fact_reasoner.core.nli import NLIExtractor

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
    "Though no heavy rain has been received in the city and water is receding "
    "from most areas in Chennai and massive relief operations are underway, the "
    "city is staring at an outbreak of epidemics with tones of stinking garbage "
    "littering the streets as bright sunshine further eased the situation."
)
HYPOTHESIS3 = "Improper drainage system in Chennai is the major cause of flood in the city."

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NLI relation probability via SIMBA-UQ."
    )
    parser.add_argument(
        "--backend",
        choices=["rits", "ollama", "vllm"],
        default="ollama",
        help="Which Mellea backend to use (default: ollama). SIMBA-UQ works on "
        "any backend, including ones without logprobs like Ollama.",
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
    parser.add_argument(
        "--similarity-metric",
        default="rouge",
        choices=["rouge", "jaccard", "sbert", "difflib", "levenshtein"],
        help="SIMBA-UQ similarity metric (default: rouge).",
    )
    args = parser.parse_args()

    # Create the selected Mellea backend.
    backend = build_backend(
        args.backend, model_id=args.served_model, base_url=args.base_url
    )

    # Build the NLI extractor with the SIMBA-UQ method. The confidence of the
    # selected sample is used as the probability of the predicted label.
    nli = NLIExtractor(
        backend,
        nli_method="simbauq",
        simbauq_similarity_metric=args.similarity_metric,
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
