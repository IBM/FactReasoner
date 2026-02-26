import os
import json
from pathlib import Path
from mellea.backends import ModelOption

# Local imports
from src.fact_reasoner.core.atomizer import Atomizer
from src.fact_reasoner.core.reviser import Reviser
from src.fact_reasoner.core.retriever import ContextRetriever
from src.fact_reasoner.core.query_builder import QueryBuilder
from src.fact_reasoner.baselines.factscore import FactScore

# Create a Mellea RITS backend
from mellea_ibm.rits import RITSBackend, RITS
backend = RITSBackend(
    RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 4096},
)

# Set cache dir for context retriever
cache_dir = None # "/home/radu/data/cache"
cwd = Path(__file__).resolve().parent

# Create the retriever, atomizer and reviser.
qb = QueryBuilder(backend)
atom_extractor = Atomizer(backend)
atom_reviser = Reviser(backend)
context_retriever = ContextRetriever(
    service_type="google", 
    top_k=5, 
    cache_dir=cache_dir, 
    fetch_text=True, 
    query_builder=qb
)

# Create the FactScore pipeline
pipeline = FactScore(
    backend=backend,
    context_retriever=context_retriever,
    atom_extractor=atom_extractor,
    atom_reviser=atom_reviser,
)

# Load the problem instance from a file
json_file = os.path.join(cwd, "flaherty_wikipedia.json")
with open(json_file, "r") as f:
    data = json.load(f)

# Load the file (json)
print(f"[FactScore] Initializing pipeline from: {json_file}")
pipeline.from_dict_with_contexts(data)

# Build the scorer
pipeline.build(
    has_atoms=True,
    has_contexts=True,
    revise_atoms=False
)

# Print the results
results = pipeline.score()
print(f"[FactScore] Results: {results}")

# Save the pipeline to a JSON file
output_file = os.path.join(cwd, "factscore_output.json")
output = pipeline.to_json()
output["results"] = results
with open(output_file, "w") as fp:
    json.dump(output, fp, indent=4)
print(f"Done.")

