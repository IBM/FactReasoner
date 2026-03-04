# This is an example of using ContextRetrieverFast for parallel context retrieval.

from mellea.backends import ModelOption
from mellea_ibm.rits import RITSBackend, RITS

# Local imports
from fact_reasoner.core.query_builder import QueryBuilder
from fact_reasoner.core.retriever import ContextRetriever
from fact_reasoner.core.retriever_fast import ContextRetrieverFast
from fact_reasoner.core.utils import Atom

# Create a Mellea RITS backend (used by the query builder)
backend = RITSBackend(
    RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 4096}
)

# Build a query builder and retriever
query_builder = QueryBuilder(backend)

retriever = ContextRetriever(
    top_k=3,
    service_type="google",
    cache_dir=None,
    fetch_text=True,
    use_in_memory_vectorstore=False,
    query_builder=query_builder,
)

# Create a set of atoms to retrieve contexts for
atoms = {
    "a0": Atom(id="a0", text="The Eiffel Tower was completed in 1889."),
    "a1": Atom(id="a1", text="Marie Curie won two Nobel Prizes."),
    "a2": Atom(id="a2", text="The speed of light is approximately 300,000 km/s."),
}

query = "Facts about famous landmarks and scientists"

# Create the fast retriever with 4 worker threads
fast_retriever = ContextRetrieverFast(
    context_retriever=retriever,
    num_workers=4,
)

# Retrieve contexts for all atoms in parallel
contexts = fast_retriever.retrieve_all(atoms=atoms, query=query)

print(f"\nTotal contexts retrieved: {len(contexts)}")
for cid, context in contexts.items():
    print(f"  {cid}: [{context.get_title()}] {context.get_text()[:100]}...")

print("Done.")
