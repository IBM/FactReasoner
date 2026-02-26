import os
import json
from pathlib import Path

from mellea.backends import ModelOption

# Local imports
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.core.retriever import ContextRetriever
from fact_reasoner.core.summarizer import ContextSummarizer
from fact_reasoner.core.nli import NLIExtractor
from fact_reasoner.core.query_builder import QueryBuilder
from fact_reasoner.assessor import FactReasoner

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
context_summarizer = ContextSummarizer(backend)
nli_extractor = NLIExtractor(backend)

# Path to merlin (probabilistic inference engine)
merlin_path = os.path.join(os.getcwd(), "lib", "merlin") # Linux RedHat version

# Create the FactReasoner pipeline
pipeline = FactReasoner(
    context_retriever=context_retriever,
    context_summarizer=context_summarizer,
    atom_extractor=atom_extractor,
    atom_reviser=atom_reviser,
    nli_extractor=nli_extractor,
    merlin_path=merlin_path,
)

# Load the problem instance from a file
json_file = os.path.join(cwd, "flaherty_wikipedia.json")
with open(json_file, "r") as f:
    data = json.load(f)

print(f"[FactReasoner] Initializing the pipeline from {json_file}")
pipeline.from_dict_with_contexts(data)

# Build the FactReasoner pipeline (FR2 version)
pipeline.build(
    has_atoms=True,
    has_contexts=True,
    revise_atoms=False,
    remove_duplicates=True,
    summarize_contexts=False,
    contexts_per_atom_only=False,
    rel_atom_context=True, 
    rel_context_context=False,
)

# Print the results
results, marginals = pipeline.score()
print(f"[FactReasoner] Marginals: {marginals}")
print(f"[FactReasoner] Results: {results}")

# Save the pipeline to a JSON file
output_file = os.path.join(cwd, "factreasoner_output.json")
output = pipeline.to_json()
output["results"] = results
with open(output_file, "w") as fp:
    json.dump(output, fp, indent=4)
print(f"Done.")

