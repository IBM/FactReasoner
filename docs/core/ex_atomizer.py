# This is a simple example

import asyncio

from mellea.backends import ModelOption
from mellea_ibm.rits import RITSBackend, RITS

# Local imports
from src.fact_reasoner.core.atomizer import Atomizer

# Create a Mellea RITS backend
backend = RITSBackend(
    RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 4096}
)

# Create the atomizer
atomizer = Atomizer(backend=backend)

response = "The Apollo 14 mission to the Moon took place on January 31, 1971. \
    This mission was significant as it marked the third time humans set \
    foot on the lunar surface, with astronauts Alan Shepard and Edgar \
    Mitchell joining Captain Stuart Roosa, who had previously flown on \
    Apollo 13. The mission lasted for approximately 8 days, during which \
    the crew conducted various experiments and collected samples from the \
    lunar surface. Apollo 14 brought back approximately 70 kilograms of \
    lunar material, including rocks, soil, and core samples, which have \
    been invaluable for scientific research ever since."

responses = [
    "The Apollo 14 mission to the Moon took place on January 31, 1971. \
    This mission was significant as it marked the third time humans set \
    foot on the lunar surface, with astronauts Alan Shepard and Edgar \
    Mitchell joining Captain Stuart Roosa, who had previously flown on \
    Apollo 13. The mission lasted for approximately 8 days, during which \
    the crew conducted various experiments and collected samples from the \
    lunar surface. Apollo 14 brought back approximately 70 kilograms of \
    lunar material, including rocks, soil, and core samples, which have \
    been invaluable for scientific research ever since.",
    "Lanny Flaherty is an American actor born on December 18, 1949, in Pensacola, Florida. He has appeared in numerous films, television shows, and theater productions throughout his career, which began in the late 1970s. Some of his notable film credits include \"King of New York,\" \"The Abyss,\" \"Natural Born Killers,\" \"The Game,\" and \"The Straight Story.\" On television, he has appeared in shows such as \"Law & Order,\" \"The Sopranos,\" \"Boardwalk Empire,\" and \"The Leftovers.\" Flaherty has also worked extensively in theater, including productions at the Public Theater and the New York Shakespeare Festival. He is known for his distinctive looks and deep gravelly voice, which have made him a memorable character actor in the industry."
]


# Process the response to extract atomic units
result = atomizer.run(response)
print(f"Atomization result: {result}")

# Print the extracted atomic units
print(f"Extracted {len(result)} atomic units:")
for k, v in result.items():
    print(f"Atom {k}: {v}")

# Process the batch
print(f"Process a batch of responses ...")
results = asyncio.run(atomizer.run_batch(responses))
for result in results:
    for k, v in result.items():
        print(f"Atom {k}: {v}")

print("Done.")
