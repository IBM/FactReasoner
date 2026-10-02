# Copyright 2023-present the International Business Machines.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate the frozen NTRS benchmark datasets and report per-backend metrics.

The benchmark datasets are frozen: contexts were retrieved once by
``build_ntrs_benchmark`` and stored in the ``.jsonl`` files. Evaluation
therefore performs NO retrieval -- ``context_retriever`` is deliberately
``None`` so that any accidental retrieval attempt fails loudly instead of
silently querying a search backend and making the Google-vs-NTRS comparison
irreproducible. ``--service-type`` is only a label.

This is why the runner does not go through ``FactualityRunner``, which builds a
live ``SourceRetriever`` for every record and offers no way to opt out.

Only ``factverify`` and ``factscore`` are supported; both score atoms directly
with the LLM and so avoid the external ``merlin`` dependency::

    python -m fact_reasoner.benchmarks.eval_ntrs_benchmark \
        --input-file data/nasa_science-labeled_ntrs.jsonl \
        --output-dir results --dataset-name nasa_science \
        --service-type ntrs --model-id gpt-4o-mini --pipeline factverify
"""

import argparse
import json
import os
from typing import Any, Callable

from dotenv import load_dotenv

PIPELINES = ("factverify", "factscore")


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-response results into micro-averaged benchmark metrics.

    Precision/recall/F1 are computed over the atom-level confusion matrix
    against the gold labels, pooled across all responses.

    Args:
        results: The per-response result dicts produced by the pipeline's
            ``score()``.

    Returns:
        The aggregated metrics.
    """

    tp = sum(r.get("true_positive", 0) for r in results)
    tn = sum(r.get("true_negative", 0) for r in results)
    fp = sum(r.get("false_positive", 0) for r in results)
    fn = sum(r.get("false_negative", 0) for r in results)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    num_atoms = sum(r.get("num_atoms", 0) for r in results)
    accuracy = (tp + tn) / num_atoms if num_atoms else 0.0
    scores = [r["factuality_score"] for r in results if "factuality_score" in r]

    return {
        "num_responses": len(results),
        "num_atoms": num_atoms,
        "num_contexts": sum(r.get("num_contexts", 0) for r in results),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "avg_factuality_score": sum(scores) / len(scores) if scores else 0.0,
    }


def load_previous_results(output_filename: str) -> list[dict[str, Any]]:
    """Load already-written results so an interrupted run can be resumed."""

    if not os.path.isfile(output_filename):
        return []
    with open(output_filename) as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(
    records: list[dict[str, Any]],
    make_pipeline: Callable[[], Any],
    output_filename: str,
    model_id: str,
    service_type: str,
    evaluation_data: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Score each frozen record and write the results incrementally.

    Args:
        records: Frozen benchmark records, each carrying its atoms and contexts.
        make_pipeline: Factory returning a fresh assessor per record.
        output_filename: Where the per-response results are written.
        model_id: Recorded on each result as ``model_name``.
        service_type: Recorded on each result; a label only, since no retrieval
            happens here.
        evaluation_data: Results from a previous run to resume from.

    Returns:
        All results, including any resumed ones.
    """

    evaluation_data = list(evaluation_data or [])
    # score() reports the response under "query", not "input".
    processed_inputs = {res["query"] for res in evaluation_data if "query" in res}

    for record in records:
        if record["input"] in processed_inputs:
            print(f"Skipping already evaluated response: {record.get('topic')}")
            continue

        pipeline = make_pipeline()
        pipeline.from_dict_with_contexts(record)
        pipeline.build(
            topic=record.get("topic"),
            has_atoms=True,
            has_contexts=True,
            revise_atoms=False,
        )

        results = pipeline.score()
        results["model_name"] = model_id
        results["service_type"] = service_type
        evaluation_data.append(results)

        # Write incrementally so a crash keeps completed work.
        with open(output_filename, "w") as f:
            f.writelines(f"{json.dumps(res)}\n" for res in evaluation_data)
        print(f"Wrote {len(evaluation_data)} results to {output_filename}")

    return evaluation_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen NTRS benchmark dataset."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to a frozen benchmark dataset (jsonl with atoms and contexts).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory where the evaluation results are written.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="nasa_science",
        help="Dataset name used in the output filename.",
    )
    parser.add_argument(
        "--service-type",
        type=str,
        default="ntrs",
        help="Retrieval backend that produced the dataset "
        "(label only; no retrieval is performed).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        help="Mellea backend used by the evaluation pipeline.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="gpt-4o-mini",
        help="Model identifier passed to the evaluation backend.",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default="factverify",
        choices=PIPELINES,
        help="Factuality pipeline. Both avoid the merlin dependency.",
    )
    args = parser.parse_args()

    # Imported here so that summarize() and evaluate() stay importable without
    # pulling in the heavy model dependencies.
    from fact_reasoner.backends import build_backend
    from fact_reasoner.baselines.factscore import FactScore
    from fact_reasoner.baselines.factverify import FactVerify
    from fact_reasoner.core.atomizer import Atomizer
    from fact_reasoner.core.reviser import Reviser

    pipeline_cls = {"factverify": FactVerify, "factscore": FactScore}[args.pipeline]

    load_dotenv(override=True)

    backend = build_backend(args.backend, model_id=args.model_id)

    # The atom extractor and reviser are asserted non-None by build(), but are
    # never used here: the benchmark atoms are pre-defined and frozen
    # (has_atoms=True, revise_atoms=False).
    atom_extractor = Atomizer(backend)
    atom_reviser = Reviser(backend)

    with open(args.input_file) as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(records)} records from {args.input_file}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_filename = os.path.join(
        args.output_dir,
        f"eval_{args.pipeline}_{args.service_type}"
        f"_{args.dataset_name}_{args.model_id}.jsonl",
    )

    evaluation_data = load_previous_results(output_filename)
    print(f"Found {len(evaluation_data)} existing evaluations in {output_filename}")

    evaluation_data = evaluate(
        records,
        lambda: pipeline_cls(
            backend=backend,
            atom_extractor=atom_extractor,
            atom_reviser=atom_reviser,
            context_retriever=None,  # frozen contexts: retrieval must not happen
        ),
        output_filename,
        model_id=args.model_id,
        service_type=args.service_type,
        evaluation_data=evaluation_data,
    )

    summary = summarize(evaluation_data)
    print(
        f"\n[{args.service_type}] Benchmark summary "
        f"({args.pipeline}, {args.model_id}):"
    )
    for key, value in summary.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else value
        print(f"  {key}: {formatted}")

    print("Done.")


if __name__ == "__main__":
    main()
