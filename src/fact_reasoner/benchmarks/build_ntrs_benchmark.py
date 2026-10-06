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

"""Frozen per-backend benchmark generation for the NTRS retrieval backend.

Reads a base benchmark file (atoms + gold labels, empty contexts) and, for each
atom, generates ONE search query with the shared ``QueryBuilder`` and fans that
identical query out to each retrieval backend. The result is one frozen .jsonl
per backend, directly consumable by ``FactualityRunner.assess_file``
(``has_atoms=True``, ``has_contexts=True``).

Fairness: NTRS is a keyword index, so raw natural-language claims match almost
nothing while Google/Serper tolerates them. Generating the query once and
reusing it for every backend ensures the comparison reflects evidence quality,
not query formatting. The NTRS keyword form is derived deterministically by
``fact_reasoner.ntrs_query``, so it costs no additional LLM calls.

Examples::

    # Google + NTRS, both driven by the same generated queries.
    python -m fact_reasoner.benchmarks.build_ntrs_benchmark \
        --input-file data/nasa_science-labeled.jsonl \
        --dataset-name nasa_science-labeled \
        --services google,ntrs --top-k 5

    # Generate queries with a local Ollama model instead of OpenAI.
    python -m fact_reasoner.benchmarks.build_ntrs_benchmark \
        --backend ollama --model-id granite4 --services ntrs
"""

import argparse
import copy
import json
import os
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

from fact_reasoner.ntrs_query import strip_phrase_quotes, to_keyword_query

if TYPE_CHECKING:
    # Imported for type checking only. At runtime these are pulled in inside
    # `main`, so `build_datasets` stays importable (and testable) without the
    # retrieval/Mellea dependency stack.
    from fact_reasoner.core.query_builder import QueryBuilder
    from fact_reasoner.core.retriever import SourceRetriever

# Backend kinds accepted by `fact_reasoner.backends.build_backend`.
BACKEND_KINDS = ["openai", "ollama", "vllm", "rits"]

# Retrieval backends this builder generates. `chromadb` is deliberately absent:
# it needs a collection name and a persisted store, which this builder does not
# plumb through.
SERVICES = ["google", "ntrs", "wikipedia"]


def build_datasets(
    records: list[dict[str, Any]],
    services: list[str],
    retrievers: dict[str, "SourceRetriever"],
    query_builder: "QueryBuilder",
    ntrs_max_terms: int = 4,
) -> dict[str, list[dict[str, Any]]]:
    """
    Build one frozen dataset per retrieval backend from the base records.

    For each atom a single query is generated with the QueryBuilder and reused
    across every backend, so all backends receive identical search queries. Atom
    text/original/label are preserved; only the contexts are filled in.

    Args:
        records: list
            The base benchmark records (one response per element).
        services: list
            The retrieval backends to generate (e.g. ["google", "ntrs"]).
        retrievers: dict
            Mapping of service name -> initialized SourceRetriever
            (query_builder=None).
        query_builder: QueryBuilder
            The shared QueryBuilder used to generate the query for every atom.
        ntrs_max_terms: int
            Word cap for the keyword query derived for the NTRS backend.

    Returns:
        dict: Mapping of service name -> list of populated records.
    """

    outputs = {service: copy.deepcopy(records) for service in services}

    for record_idx, record in enumerate(records):
        for atom_idx, atom in enumerate(record["atoms"]):
            query = strip_phrase_quotes(query_builder.run(atom["text"]))
            ntrs_query = to_keyword_query(query, max_terms=ntrs_max_terms)
            print(f"[{record.get('topic')}] {atom['id']}: query={query!r}")
            if "ntrs" in services:
                print(f"    ntrs_query={ntrs_query!r}")

            for service in services:
                # NTRS is a keyword index; every other backend receives the full
                # query. Both are derived from the same QueryBuilder output.
                service_query = ntrs_query if service == "ntrs" else query
                passages = retrievers[service].query(text=service_query)
                out_atom = outputs[service][record_idx]["atoms"][atom_idx]

                context_ids = []
                for j, passage in enumerate(passages):
                    cid = f"c_{out_atom['id']}_{j}"
                    context_ids.append(cid)
                    outputs[service][record_idx]["contexts"].append({
                        "id": cid,
                        "title": passage.get("title", ""),
                        "text": passage.get("text", ""),
                        "link": passage.get("link", ""),
                        "snippet": passage.get("snippet", ""),
                    })
                out_atom["contexts"] = context_ids

    return outputs


def _output_path(output_dir: str, dataset_name: str, service: str) -> str:
    """Path of the frozen file generated for ``service``."""

    return os.path.join(output_dir, f"{dataset_name}_{service}.jsonl")


def _check_no_overwrite(output_filenames: list[str], overwrite: bool) -> None:
    """Refuse to replace existing outputs unless the caller opted in.

    The generated files are checked in as frozen benchmark artifacts, so the
    default must never clobber them. Nothing is written unless every target is
    clear, so a refusal cannot leave a partially regenerated set behind.
    """

    if overwrite:
        return
    existing = [name for name in output_filenames if os.path.exists(name)]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing benchmark file(s): "
            f"{', '.join(existing)}. These may be frozen artifacts; pass "
            "--overwrite to replace them intentionally."
        )


def write_datasets(
    outputs: dict[str, list[dict[str, Any]]],
    output_dir: str,
    dataset_name: str,
    overwrite: bool = False,
) -> None:
    """
    Write one frozen .jsonl per retrieval backend.

    Args:
        outputs: dict
            Mapping of service name -> records, as returned by `build_datasets`.
        output_dir: str
            Directory the per-backend files are written to.
        dataset_name: str
            Base dataset name; output is {dataset_name}_{service}.jsonl.
        overwrite: bool
            Replace existing output files instead of refusing.

    Raises:
        FileExistsError: If an output file exists and `overwrite` is False.
    """

    _check_no_overwrite(
        [_output_path(output_dir, dataset_name, service) for service in outputs],
        overwrite,
    )

    os.makedirs(output_dir, exist_ok=True)
    for service, records in outputs.items():
        output_filename = _output_path(output_dir, dataset_name, service)
        with open(output_filename, "w") as f:
            for record in records:
                f.write(f"{json.dumps(record)}\n")
        print(
            f"[build-ntrs-benchmark] Wrote {len(records)} records "
            f"to {output_filename}"
        )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input-file",
        default="data/nasa_science-labeled.jsonl",
        help="Base benchmark file (atoms + gold labels, empty contexts).",
    )
    p.add_argument(
        "--output-dir",
        default="data",
        help="Directory where the per-backend files are written.",
    )
    p.add_argument(
        "--dataset-name",
        default="nasa_science-labeled",
        help="Base dataset name; output is {dataset-name}_{service}.jsonl.",
    )
    p.add_argument(
        "--services",
        default="google,ntrs",
        help=f"Comma-separated retrieval backends ({', '.join(SERVICES)}).",
    )
    p.add_argument(
        "--backend",
        default="openai",
        choices=BACKEND_KINDS,
        help="Mellea backend used by the QueryBuilder (default: openai).",
    )
    p.add_argument(
        "--model-id",
        default=None,
        help="Model id for the query-builder backend. Accepts a unified catalog "
        "id/alias or a raw provider value; when omitted, the backend's own "
        "default is used. Endpoints and keys come from the environment (see "
        "fact_reasoner.backends).",
    )
    p.add_argument(
        "--top-k", type=int, default=5, help="Number of contexts retrieved per atom."
    )
    p.add_argument(
        "--ntrs-max-terms",
        type=int,
        default=4,
        help="Word cap for the keyword query derived for the NTRS backend.",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Optional cache directory (used by the google backend).",
    )
    p.add_argument(
        "--fetch-text",
        action="store_true",
        help="Fetch full page text for retrieved links (google backend only).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output files. Without this flag the builder "
        "refuses to overwrite them, so checked-in frozen benchmark artifacts "
        "survive an accidental run with default arguments.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # Deferred so that importing this module (and unit-testing `build_datasets`)
    # does not require the retrieval/Mellea dependency stack.
    from fact_reasoner.backends import build_backend
    from fact_reasoner.core.query_builder import QueryBuilder
    from fact_reasoner.core.retriever import SourceRetriever

    # Load SERPER_API_KEY / OPENAI_API_KEY from a local .env if present.
    load_dotenv(override=True)

    services = [s.strip() for s in args.services.split(",") if s.strip()]
    if not services:
        raise ValueError("No retrieval services requested (see --services).")

    # Checked up front as well as at write time, so a refusal costs no LLM or
    # retrieval calls.
    _check_no_overwrite(
        [_output_path(args.output_dir, args.dataset_name, s) for s in services],
        args.overwrite,
    )

    # Build the shared QueryBuilder through the standard backend factory, which
    # resolves the model id against the catalog and applies provider defaults.
    query_builder = QueryBuilder(build_backend(args.backend, model_id=args.model_id))

    # One retriever per backend. query_builder is None here because the query is
    # generated once per atom and passed in as the query text, so every backend
    # receives the identical query.
    retrievers = {
        service: SourceRetriever(
            service_type=service,
            top_k=args.top_k,
            cache_dir=args.cache_dir,
            fetch_text=args.fetch_text,
            query_builder=None,
        )
        for service in services
    }

    # Load the base benchmark records (one JSON object per line).
    with open(args.input_file) as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(
        f"[build-ntrs-benchmark] Loaded {len(records)} records "
        f"from {args.input_file}"
    )
    print(
        f"[build-ntrs-benchmark] Generating services={services} "
        f"with backend={args.backend} model_id={args.model_id}"
    )

    outputs = build_datasets(
        records,
        services,
        retrievers,
        query_builder,
        ntrs_max_terms=args.ntrs_max_terms,
    )

    write_datasets(
        outputs,
        args.output_dir,
        args.dataset_name,
        overwrite=args.overwrite,
    )

    print("Done.")


if __name__ == "__main__":
    main()
