# This is a simple example

from mellea.backends import ModelOption
from mellea_ibm.rits import RITSBackend, RITS

# Local imports
from src.fact_reasoner.core.query_builder import QueryBuilder

# Create a Mellea RITS backend
from mellea_ibm.rits import RITSBackend, RITS
backend = RITSBackend(
    RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 4096},
)

# Create the query builder
qb = QueryBuilder(backend)

# Process a single atom (no knowledge)        
# text = "The Apollo 14 mission to the Moon took place on January 31, 1971."
# text = "You'd have to yell if your friend is outside the same location"
text = "rootstock for honey crisp apples in wayne county, ny"

result = qb.run(text)
print(f"Query builder result: {result}")

# Print the query
print(f"Initial Text: {text}")
print(f"Query: {result}")

print("Done.")
