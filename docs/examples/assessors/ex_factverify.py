import os
import json
from pathlib import Path
from mellea.backends import ModelOption

# Local imports
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.core.retriever import ContextRetriever, ContextRetrieverFast
from fact_reasoner.core.query_builder import QueryBuilder
from fact_reasoner.baselines.factverify import FactVerify

# Example query and response
query = "Tell me a biography of Lanny Flaherty"
response = "Lanny Flaherty is an American actor born on December 18, 1949, in Pensacola, Florida. He has appeared in numerous films, television shows, and theater productions throughout his career, which began in the late 1970s. Some of his notable film credits include \"King of New York,\" \"The Abyss,\" \"Natural Born Killers,\" \"The Game,\" and \"The Straight Story.\" On television, he has appeared in shows such as \"Law & Order,\" \"The Sopranos,\" \"Boardwalk Empire,\" and \"The Leftovers.\" Flaherty has also worked extensively in theater, including productions at the Public Theater and the New York Shakespeare Festival. He is known for his distinctive looks and deep gravelly voice, which have made him a memorable character actor in the industry."
topic = "Lanny Flaherty"

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
    fetch_text=False, # no retrieving from the link
    query_builder=qb
    num_workers=4
)
context_retriever_fast = ContextRetrieverFast(
    context_retriever=context_retriever,
    num_workers=4
)

# Create the FactScore pipeline
pipeline = FactVerify(
    backend=backend,
    context_retriever=context_retriever_fast,
    atom_extractor=atom_extractor,
    atom_reviser=atom_reviser,
)

# Build the FactVerify pipeline
pipeline.build(
    query=query,
    response=response,
    topic=topic,
    has_atoms=False,
    has_contexts=False,
    revise_atoms=True,
    use_fast_retriever=True
)

# Print the results
results = pipeline.score()
print(f"[FactVerify] Results: {results}")

# Save the pipeline to a JSON file
output_file = os.path.join(cwd, "factverify_output.json")
output = pipeline.to_json()
output["results"] = results
with open(output_file, "w") as fp:
    json.dump(output, fp, indent=4)
print(f"Done.")

