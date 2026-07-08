import os
import json
import argparse
from pathlib import Path

# Local imports
from fact_reasoner.backends import build_backend
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.core.retriever import ContextRetriever
from fact_reasoner.core.query_builder import QueryBuilder
from fact_reasoner.baselines.factverify import FactVerify

# Select the Mellea backend from the command line (RITS by default).
parser = argparse.ArgumentParser(description="FactVerify (from file) example.")
parser.add_argument(
    "--backend",
    choices=["rits", "ollama", "vllm"],
    default="rits",
    help="Which Mellea backend to use: 'rits' (remote IBM RITS, default), "
    "'ollama' (local Ollama server), or 'vllm' (vLLM OpenAI-compatible server).",
)
parser.add_argument(
    "--served-model",
    default=None,
    help="Model / served-model name (required for 'vllm').",
)
parser.add_argument(
    "--base-url",
    default=None,
    help="Base URL for the 'vllm' backend (defaults to VLLM_BASE_URL env "
    "or http://localhost:8000/v1).",
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
context_retriever = ContextRetriever(
    service_type="google",
    top_k=5,
    cache_dir=cache_dir,
    fetch_text=False,  # no retrieving from the link
    query_builder=qb,
)

# Create the FactScore pipeline
pipeline = FactVerify(
    backend=backend,
    context_retriever=context_retriever,
    atom_extractor=atom_extractor,
    atom_reviser=atom_reviser,
)

# Load the problem instance from a file
json_file = os.path.join(cwd, "flaherty_google.json")
with open(json_file, "r") as f:
    data = json.load(f)

print(f"[FactVerify] Initializing pipeline from: {json_file}")
pipeline.from_dict_with_contexts(data)

# Build the FactVerify pipeline
pipeline.build(has_atoms=True, has_contexts=True, revise_atoms=False)

# Print the results
results = pipeline.score()
print(f"[FactVerify] Results: {results}")

# Save the pipeline to a JSON file
output_file = os.path.join(cwd, "factverify_output.json")
output = pipeline.to_json()
output["results"] = results
with open(output_file, "w") as fp:
    json.dump(output, fp, indent=4)
print("Done.")
