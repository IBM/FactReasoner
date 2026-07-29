# coding=utf-8
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

"""Train a SIMBA-UQ classifier for the NLI use case.

Two stages, decoupled via a JSONL samples cache so the expensive generation step
is not repeated when you re-tune classifier hyper-parameters:

  * ``generate`` — for each labeled ``{premise, hypothesis, label}`` pair, run the
    SIMBA-UQ generation and write per-pair sample groups + correctness labels to a
    JSONL cache (resumable).
  * ``train`` — read the JSONL cache, fit a classifier, and save it (joblib).

Use ``--stage all`` to run both. Example::

    python scripts/train_simbauq_nli.py --stage all \\
      --nli-data /Users/radu/tmp/raw_nli/train_balanced.json --num-pairs 900 \\
      --backend ollama --similarity-metric rouge \\
      --samples artifacts/simbauq_nli_samples.jsonl \\
      --out artifacts/simbauq_nli_clf.joblib

Generation cost is ``num_pairs * len(temperatures) * n_per_temp`` LLM calls, so
scale ``--num-pairs`` to your compute budget.
"""

import argparse
import asyncio

from fact_reasoner.backends import build_backend
from fact_reasoner.uncertainty import (
    evaluate_classifier,
    generate_training_samples,
    load_classifier,
    load_nli_pairs,
    save_classifier,
    train_classifier_from_jsonl,
)


def _parse_temperatures(text):
    """Parse a comma-separated list of temperatures, or None for the default."""
    if text is None:
        return None
    return [float(t) for t in text.split(",") if t.strip()]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a SIMBA-UQ NLI classifier.")
    p.add_argument(
        "--stage",
        choices=["generate", "train", "all", "evaluate"],
        default="all",
        help="Which stage(s) to run (default: all).",
    )

    # Data.
    p.add_argument(
        "--nli-data",
        default=None,
        help="Path to a JSON array of {premise, hypothesis, label} triples "
        "(required for --stage generate/all).",
    )
    p.add_argument(
        "--num-pairs",
        type=int,
        default=900,
        help="Max number of pairs to generate over (class-balanced subset). "
        "Generation cost scales with this. Default: 900.",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="Seed for the balanced subset selection."
    )

    # Backend (generation only).
    p.add_argument(
        "--backend",
        choices=["rits", "ollama", "vllm", "openai"],
        default="ollama",
        help="Mellea backend used for generation (default: ollama).",
    )
    p.add_argument(
        "--model-id",
        default=None,
        help="Model id / served-model name (optional; defaults to Granite 4 Micro). "
        "With --backend rits and a custom --base-url, this is the raw RITS model "
        "name (required in that case).",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="API endpoint. For --backend vllm: the server base URL (defaults to "
        "VLLM_BASE_URL env or http://localhost:8000/v1). For --backend rits: a "
        "custom RITS endpoint, in which case --model-id is the raw RITS model "
        "name (RITS appends /v1; key from RITS_API_KEY).",
    )
    p.add_argument(
        "--num-workers", type=int, default=4, help="Concurrent pair generations."
    )

    # SIMBA-UQ config (must be consistent between generate and train).
    p.add_argument(
        "--temperatures",
        default=None,
        help="Comma-separated temperatures (default: SIMBA-UQ default 0.3,0.5,0.7,1.0).",
    )
    p.add_argument("--n-per-temp", type=int, default=5, help="Samples per temperature.")
    p.add_argument(
        "--similarity-metric",
        default="rouge",
        choices=["rouge", "jaccard", "sbert", "difflib", "levenshtein"],
        help="Similarity metric for classifier features (default: rouge).",
    )

    # Classifier hyper-parameters (train stage).
    p.add_argument("--clf-max-depth", type=int, default=4, help="Random forest depth.")
    p.add_argument(
        "--clf-random-state", type=int, default=0, help="Random forest seed."
    )

    # Artifacts.
    p.add_argument(
        "--samples",
        default="artifacts/simbauq_nli_samples.jsonl",
        help="Samples JSONL cache (written by generate, read by train/evaluate).",
    )
    p.add_argument(
        "--out",
        default="artifacts/simbauq_nli_clf.joblib",
        help="Output path for the trained classifier (train stage).",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    temperatures = _parse_temperatures(args.temperatures)

    if args.stage in ("generate", "all"):
        if not args.nli_data:
            raise SystemExit("--nli-data is required for --stage generate/all.")
        # A custom RITS endpoint serves its own model, so it needs an explicit name.
        if args.backend == "rits" and args.base_url and not args.model_id:
            raise SystemExit(
                "A custom RITS endpoint (--base-url) requires --model-id "
                "(the RITS model name)."
            )
        backend = build_backend(
            args.backend, model_id=args.model_id, base_url=args.base_url
        )
        pairs = load_nli_pairs(args.nli_data, num_pairs=args.num_pairs, seed=args.seed)
        print(f"[train] Loaded {len(pairs)} labeled NLI pairs from {args.nli_data}.")
        asyncio.run(
            generate_training_samples(
                pairs,
                backend,
                args.samples,
                temperatures=temperatures,
                n_per_temp=args.n_per_temp,
                similarity_metric=args.similarity_metric,
                num_workers=args.num_workers,
                progress=True,
            )
        )

    if args.stage in ("train", "all"):
        clf, metadata = train_classifier_from_jsonl(
            args.samples,
            temperatures=temperatures,
            n_per_temp=args.n_per_temp,
            similarity_metric=args.similarity_metric,
            clf_max_depth=args.clf_max_depth,
            clf_random_state=args.clf_random_state,
            progress=True,
        )
        save_classifier(clf, args.out, metadata)

    if args.stage == "evaluate":
        clf, _ = load_classifier(args.out)
        evaluate_classifier(
            clf,
            args.samples,
            temperatures=temperatures,
            n_per_temp=args.n_per_temp,
            similarity_metric=args.similarity_metric,
            progress=True,
        )

    print("[train] Done.")


if __name__ == "__main__":
    main()
