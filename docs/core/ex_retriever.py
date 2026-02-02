# This is a simple example

from mellea.backends import ModelOption
from mellea_ibm.rits import RITSBackend, RITS

# Local imports
from src.fact_reasoner.core.query_builder import QueryBuilder
from src.fact_reasoner.core.retriever import ContextRetriever, fetch_text_from_link

# Create a Mellea RITS backend
from mellea_ibm.rits import RITSBackend, RITS
backend = RITSBackend(
    RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 4096}
)

# query_text = "Lanny Flaherty has appeared in Law & Order."
query_text = "Unsupervised learning is the primary method used for analyzing soil quality in oil palm plantations"
cache_dir = None #"my_database.db"
query_builder = QueryBuilder(backend)

retriever = ContextRetriever(
    top_k=5,
    service_type="google",
    cache_dir=cache_dir,
    fetch_text=True,
    use_in_memory_vectorstore=False,
    query_builder=query_builder
)

contexts = retriever.query(text=query_text)

print(f"Number of contexts: {len(contexts)}")
for context in contexts:
    print(context)

link = "https://www.ancientportsantiques.com/wp-content/uploads/Documents/AUTHORS/SeaPeoples/SeaPeoples-Fischer&B%C3%BCrge2017.pdf"
text = fetch_text_from_link(link, max_size=4000)
print(f"Text length: {len(text)}")
print(f"Text: {text}")  # Print text
print("Done.")
