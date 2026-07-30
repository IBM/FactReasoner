# This is an example of running FactReasoner with the `fast` NLI mode.
#
# `fast` scores far fewer NLI candidate pairs than the default, while producing the
# **same graph shape**. It bundles four settings at once:
#   * atom-context pairs restricted to the atoms that actually retrieved each
#     context (plus query-level contexts, near-neighbor atoms, and a
#     similarity-gate rescue for genuine cross-atom evidence);
#   * context-context pairs gated by embedding similarity;
#   * near-duplicate contexts collapsed before mining; and
#   * each context pair scored in one direction, mirrored only where the reverse
#     could change the reconciled outcome.
#
# How much it saves is workload-dependent -- measured between ~1.2x (a narrative
# where every atom shares characters, so the cross-product holds little waste) and
# ~5x (unrelated subtopics). Because it prunes, it can miss a relation: use
# `ex_factreasoner_all_pairs.py` when you need bit-for-bit reproducibility.
#
# Two things to watch on startup:
#   * the assessor prints "[FactReasoner] NLI pair policy: provenance (...)" --
#     that line is your confirmation the mode is active;
#   * the similarity gate needs the sentence-transformers embedding model. If it
#     cannot be loaded the gate degrades to token Jaccard and prints a warning --
#     do not ignore it, the lexical fallback measurably drops real relations.
#
# The script accepts either input mode:
#   * --response/--query/--topic  -- assess a response from scratch (atomize,
#     retrieve contexts, then score).
#   * --input-file <json>         -- load precomputed atoms + contexts via
#     `from_dict_with_contexts` and score those directly (no retrieval, no LLM
#     calls for atomization).

import argparse
import asyncio
import json
import os
from pathlib import Path

from fact_reasoner.assessor import FactReasoner

# Local imports
from fact_reasoner.backends import build_backend
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.nli import NLIExtractor
from fact_reasoner.core.nli_config import get_pair_config
from fact_reasoner.core.query_builder import QueryBuilder
from fact_reasoner.core.retriever import ContextRetriever, SourceRetriever
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.core.summarizer import ContextSummarizer

# The NLI candidate-pair mode this example demonstrates.
NLI_MODE = "fast"

# Example query and response, used when no --input-file is given.
QUERY = "Tell me a biography of Lanny Flaherty"
RESPONSE = 'Lanny Flaherty is an American actor born on December 18, 1949, in Pensacola, Florida. He has appeared in numerous films, television shows, and theater productions throughout his career, which began in the late 1970s. Some of his notable film credits include "King of New York," "The Abyss," "Natural Born Killers," "The Game," and "The Straight Story." On television, he has appeared in shows such as "Law & Order," "The Sopranos," "Boardwalk Empire," and "The Leftovers." Flaherty has also worked extensively in theater, including productions at the Public Theater and the New York Shakespeare Festival. He is known for his distinctive looks and deep gravelly voice, which have made him a memorable character actor in the industry.'
TOPIC = "Lanny Flaherty"


def main() -> None:
    # Select the Mellea backend from the command line (RITS by default).
    parser = argparse.ArgumentParser(
        description=f"FactReasoner assessor example ({NLI_MODE} NLI mode)."
    )
    parser.add_argument(
        "--backend",
        choices=["rits", "ollama", "vllm", "openai"],
        default="rits",
        help="Which Mellea backend to use: 'rits' (remote IBM RITS, default), "
        "'ollama' (local Ollama server), 'vllm' (vLLM OpenAI-compatible server), "
        "or 'openai' (hosted frontier model: OpenAI, or Claude via --base-url "
        "https://api.anthropic.com/v1/).",
    )
    parser.add_argument(
        "--served-model",
        default=None,
        help="Model / served-model name. Optional: when omitted, build_backend "
        "uses the shared default model (Granite 4 Micro) for the chosen backend.",
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
        "--merlin-path",
        required=True,
        help="Path to the Merlin probabilistic inference binary (required).",
    )

    # The two input modes are mutually exclusive: either assess a response from
    # scratch, or score precomputed atoms/contexts loaded from a file.
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--response",
        default=None,
        help="Response text to assess from scratch (atomize + retrieve contexts). "
        f"Defaults to a built-in example about {TOPIC}.",
    )
    source.add_argument(
        "--input-file",
        default=None,
        help="JSON file with precomputed atoms and contexts (e.g. "
        "flaherty_wikipedia.json). Scores those directly -- no retrieval.",
    )

    parser.add_argument(
        "--query", default=QUERY, help="Query that produced the response."
    )
    parser.add_argument("--topic", default=TOPIC, help="Optional topic hint.")
    parser.add_argument(
        "--nli-cache-dir",
        default=None,
        help="Optional directory for the cross-run NLI verdict cache. "
        "Score-neutral: a cache hit returns the verdict the model already gave, "
        "so re-scoring the same data costs no LLM calls.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help=f"Where to write the pipeline JSON (default: "
        f"factreasoner_{NLI_MODE}_output.json beside this script).",
    )
    args = parser.parse_args()

    # When no --served-model is given, build_backend falls back to the shared
    # default model (Granite 4 Micro), resolved appropriately for the backend.
    backend = build_backend(
        args.backend, model_id=args.served_model, base_url=args.base_url
    )

    # Set cache dir for context retriever
    cache_dir = None  # "/home/radu/data/cache"
    cwd = Path(__file__).resolve().parent

    # Create the retriever, atomizer and reviser. ContextRetriever wraps a
    # SourceRetriever, so build the SourceRetriever first.
    qb = QueryBuilder(backend)
    atom_extractor = Atomizer(backend)
    atom_reviser = Reviser(backend)
    retriever = SourceRetriever(
        service_type="google",
        top_k=5,
        cache_dir=cache_dir,
        fetch_text=True,
        query_builder=qb,
        num_workers=4,
    )
    context_summarizer = ContextSummarizer(backend)
    nli_extractor = NLIExtractor(backend)
    context_retriever = ContextRetriever(
        retriever=retriever,
        context_summarizer=context_summarizer,
        num_workers=4,
    )

    # Create the FactReasoner pipeline.
    #
    # NOTE: this low-level class takes a pair-config *object*, not the `nli_mode`
    # name that the CLI and FactualityRunner accept -- passing nli_mode="fast"
    # here would be a TypeError. Resolve the name with get_pair_config().
    pipeline = FactReasoner(
        context_retriever=context_retriever,
        context_summarizer=context_summarizer,
        atom_extractor=atom_extractor,
        atom_reviser=atom_reviser,
        nli_extractor=nli_extractor,
        merlin_path=args.merlin_path,
        nli_pair_config=get_pair_config(NLI_MODE),
        nli_cache_dir=args.nli_cache_dir,
    )

    # Flags shared by both input modes (FR2: atom-context relations only).
    build_kwargs = {
        "remove_duplicates": True,
        "contexts_per_atom_only": False,
        "rel_atom_context": True,
        "rel_context_context": False,
        "use_fast_retriever": True,
    }

    if args.input_file:
        # File mode: atoms and contexts are already computed, so skip atomization,
        # retrieval, revision and summarization -- just score what was loaded.
        json_file = args.input_file
        with open(json_file, "r") as f:
            data = json.load(f)

        print(f"[FactReasoner] Initializing the pipeline from {json_file}")
        pipeline.from_dict_with_contexts(data)

        build_kwargs.update(
            has_atoms=True,
            has_contexts=True,
            revise_atoms=False,
            summarize_contexts=False,
        )
    else:
        # Live mode: atomize the response, retrieve contexts, then score.
        build_kwargs.update(
            query=args.query,
            response=args.response or RESPONSE,
            topic=args.topic,
            has_atoms=False,
            has_contexts=False,
            revise_atoms=True,
            summarize_contexts=True,
        )

    # Build the FactReasoner pipeline (FR2 version). FactReasoner.build is async.
    print(f"[FactReasoner] NLI mode: {NLI_MODE}")
    asyncio.run(pipeline.build(**build_kwargs))

    # score() returns a (results, marginals) pair for FactReasoner. (The baseline
    # assessors return just `results`.)
    results, marginals = pipeline.score()
    print(f"[FactReasoner] Marginals: {marginals}")
    print(f"[FactReasoner] Results: {results}")
    print(
        f"[FactReasoner] Factuality score ({NLI_MODE}): "
        f"{results['factuality_score']:.2%} over {results['num_atoms']} atoms"
    )

    # Save the pipeline to a JSON file
    output_file = args.output_file or os.path.join(
        cwd, f"factreasoner_{NLI_MODE}_output.json"
    )
    output = pipeline.to_json()
    output["results"] = results
    with open(output_file, "w") as fp:
        json.dump(output, fp, indent=4)
    print(f"Done. Wrote {output_file}")


if __name__ == "__main__":
    main()
