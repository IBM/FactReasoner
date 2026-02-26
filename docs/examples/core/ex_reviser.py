from mellea.backends import ModelOption
from mellea_ibm.rits import RITSBackend, RITS

# Local imports
from fact_reasoner.core.reviser import Reviser

# Create a Mellea RITS backend
backend = RITSBackend(
    RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 4096}
)

# Create the reviser
reviser = Reviser(backend=backend)

response = "Lanny Flaherty is an American actor born on December 18, 1949, \
    in Pensacola, Florida. He has appeared in numerous films, television \
    shows, and theater productions throughout his career, which began in the \
    late 1970s. Some of his notable film credits include \"King of New York,\" \
    \"The Abyss,\" \"Natural Born Killers,\" \"The Game,\" and \"The Straight Story.\" \
    On television, he has appeared in shows such as \"Law & Order,\" \"The Sopranos,\" \
    \"Boardwalk Empire,\" and \"The Leftovers.\" Flaherty has also worked \
    extensively in theater, including productions at the Public Theater and \
    the New York Shakespeare Festival. He is known for his distinctive looks \
    and deep gravelly voice, which have made him a memorable character \
    actor in the industry."

atoms = [
    "He has appeared in numerous films.",
    "He has appeared in numerous television shows.",
    "He has appeared in numerous theater productions.",
    "His career began in the late 1970s."
]

# Process the atoms
result = reviser.run(atoms, response)
print(f"Reviser result: {result}")

# Print the revised atomic units
print(f"Number of revised atomic units: {len(result)}")
for atom in result:
    print(f"Original Atom: {atom['text']}")
    print(f"Revised Atom:  {atom['revised_unit']}")
    print(f"Rationale: {atom['rationale']}")
    print("-----")

print("Done.")