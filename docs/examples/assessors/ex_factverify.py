# This is an example of running the FactVerify baseline assessor.
#
# The script accepts either input mode:
#   * --response/--query/--topic  -- assess a response from scratch (atomize,
#     retrieve contexts, then score).
#   * --input-file <json>         -- load precomputed atoms + contexts via
#     `from_dict_with_contexts` and score those directly (no retrieval). When
#     the file carries gold labels, score() additionally reports how the
#     predictions compare against them.

import argparse
import json
import os
from pathlib import Path

# Local imports
from fact_reasoner.backends import build_backend
from fact_reasoner.baselines.factverify import FactVerify
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.query_builder import QueryBuilder
from fact_reasoner.core.retriever import ContextRetriever, SourceRetriever
from fact_reasoner.core.reviser import Reviser

# Example query and response, used when no --input-file is given.
QUERY = "Tell me a biography of Lanny Flaherty"
RESPONSE = 'Lanny Flaherty is an American actor born on December 18, 1949, in Pensacola, Florida. He has appeared in numerous films, television shows, and theater productions throughout his career, which began in the late 1970s. Some of his notable film credits include "King of New York," "The Abyss," "Natural Born Killers," "The Game," and "The Straight Story." On television, he has appeared in shows such as "Law & Order," "The Sopranos," "Boardwalk Empire," and "The Leftovers." Flaherty has also worked extensively in theater, including productions at the Public Theater and the New York Shakespeare Festival. He is known for his distinctive looks and deep gravelly voice, which have made him a memorable character actor in the industry.'
TOPIC = "Lanny Flaherty"


def main() -> None:
    # Select the Mellea backend from the command line (RITS by default).
    parser = argparse.ArgumentParser(description="FactVerify assessor example.")
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
        nargs="?",
        const="flaherty_google.json",
        default=None,
        help="JSON file with precomputed atoms and contexts. Scores those directly "
        "-- no retrieval. Pass the flag with no value to use the bundled "
        "flaherty_google.json beside this script.",
    )

    parser.add_argument(
        "--query", default=QUERY, help="Query that produced the response."
    )
    parser.add_argument("--topic", default=TOPIC, help="Optional topic hint.")
    parser.add_argument(
        "--output-file",
        default=None,
        help="Where to write the pipeline JSON (default: factverify_output.json "
        "beside this script).",
    )
    args = parser.parse_args()

    backend = build_backend(
        args.backend, model_id=args.served_model, base_url=args.base_url
    )

    # Set cache dir for context retriever
    cache_dir = None  # "/home/radu/data/cache"
    cwd = Path(__file__).resolve().parent

    # Create the retriever, atomizer and reviser.
    qb = QueryBuilder(backend)
    atom_extractor = Atomizer(backend)
    atom_reviser = Reviser(backend)
    retriever = SourceRetriever(
        service_type="google",
        top_k=5,
        cache_dir=cache_dir,
        fetch_text=False,  # no retrieving from the link
        query_builder=qb,
        num_workers=4,
    )
    context_retriever = ContextRetriever(retriever=retriever, num_workers=4)

    # Create the FactVerify pipeline
    pipeline = FactVerify(
        backend=backend,
        context_retriever=context_retriever,
        atom_extractor=atom_extractor,
        atom_reviser=atom_reviser,
    )

    if args.input_file:
        # File mode: atoms and contexts are already computed, so skip atomization,
        # retrieval and revision -- just score what was loaded. If the file carries
        # gold labels, from_dict_with_contexts picks them up and score() reports the
        # comparison, which the live path cannot do.
        json_file = args.input_file
        if not os.path.isabs(json_file) and not os.path.exists(json_file):
            json_file = os.path.join(cwd, json_file)
        with open(json_file, "r") as f:
            data = json.load(f)

        print(f"[FactVerify] Initializing pipeline from: {json_file}")
        pipeline.from_dict_with_contexts(data)

        pipeline.build(has_atoms=True, has_contexts=True, revise_atoms=False)
    else:
        # Live mode: atomize the response, retrieve contexts, then score.
        pipeline.build(
            query=args.query,
            response=args.response or RESPONSE,
            topic=args.topic,
            has_atoms=False,
            has_contexts=False,
            revise_atoms=True,
            use_fast_retriever=True,
        )

    # Print the results
    results = pipeline.score()
    print(f"[FactVerify] Results: {results}")

    # Save the pipeline to a JSON file
    output_file = args.output_file or os.path.join(
        cwd, "factverify_output.json"
    )
    output = pipeline.to_json()
    output["results"] = results
    with open(output_file, "w") as fp:
        json.dump(output, fp, indent=4)
    print(f"Done. Wrote {output_file}")


if __name__ == "__main__":
    main()
