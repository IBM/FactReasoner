# This example estimates NLI relation probabilities with SIMBA-UQ using a
# trained probabilistic *classifier* to score sample confidence (instead of the
# default data-free aggregation).
#
# First train and save a classifier with scripts/train_simbauq_nli.py, e.g.:
#
#   python scripts/train_simbauq_nli.py --stage all \
#     --nli-data /Users/radu/tmp/raw_nli/train_balanced.json --num-pairs 900 \
#     --backend ollama --similarity-metric rouge \
#     --samples artifacts/simbauq_nli_samples.jsonl \
#     --out artifacts/simbauq_nli_clf.joblib
#
# Then run this example, pointing --classifier-path at the saved .joblib. The
# temperature schedule / n-per-temp / similarity metric MUST match what the
# classifier was trained with (the extractor validates the feature dimension).

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
        description="NLI relation probability via SIMBA-UQ with a trained classifier."
    )
    parser.add_argument(
        "--backend",
        choices=["rits", "ollama", "vllm"],
        default="ollama",
        help="Which Mellea backend to use (default: ollama).",
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
        "--classifier-path",
        required=True,
        help="Path to a classifier saved by scripts/train_simbauq_nli.py.",
    )
    parser.add_argument(
        "--similarity-metric",
        default="rouge",
        choices=["rouge", "jaccard", "sbert", "difflib", "levenshtein"],
        help="SIMBA-UQ similarity metric (must match training; default: rouge).",
    )
    args = parser.parse_args()

    # Create the selected Mellea backend.
    backend = build_backend(
        args.backend, model_id=args.served_model, base_url=args.base_url
    )

    # Build the NLI extractor with SIMBA-UQ + the trained classifier. Passing
    # simbauq_classifier_path loads the classifier, validates its feature
    # dimension, and switches the confidence method to "classifier".
    nli = NLIExtractor(
        backend,
        nli_method="simbauq",
        simbauq_similarity_metric=args.similarity_metric,
        simbauq_confidence_method="classifier",
        simbauq_classifier_path=args.classifier_path,
    )

    # Predict the NLI relationship and its probability.
    result = nli.run(premise=PREMISE, hypothesis=HYPOTHESIS)
    print(f"Premise:     {PREMISE}")
    print(f"Hypothesis:  {HYPOTHESIS}")
    print(f"Label:       {result['label']}")
    print(f"Probability: {result['probability']:.4f}")

    print("Done.")


if __name__ == "__main__":
    main()
