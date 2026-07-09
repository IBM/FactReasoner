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
PREMISE = (
    "Robert Haldane Smith, Baron Smith of Kelvin, is a British businessman and "
    "former Governor of the British Broadcasting Corporation."
)
HYPOTHESIS = "Robert Smith holds the title of Baron Smith of Kelvin."


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
        help="Base URL for the 'vllm' backend (defaults to VLLM_BASE_URL env "
        "or http://localhost:8000/v1).",
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
    result = nli.run(premise=PREMISE, hypothesis=HYPOTHESIS)
    print(f"Premise:    {PREMISE}")
    print(f"Hypothesis: {HYPOTHESIS}")
    print(f"Label:       {result['label']}")
    print(f"Probability: {result['probability']:.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
