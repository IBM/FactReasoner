# This is a simple example

from mellea.backends import ModelOption
from mellea_ibm.rits import RITSBackend, RITS

# Local imports
from fact_reasoner.core.summarizer import ContextSummarizer

with_ref = False

# Create a Mellea RITS backend
from mellea_ibm.rits import RITSBackend, RITS
backend = RITSBackend(
    RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 4096}
)

# Create the context summarizer
summarizer = ContextSummarizer(backend=backend, with_reference=with_ref)

if with_ref:
    atom = "The city council has approved new regulations for electric scooters."
    contexts = ["In the past year, the city had seen a rapid increase in the use of electric scooters. They seemed like a perfect solution to reduce traffic and provide an eco-friendly transportation option. However, problems arose quickly. Riders often ignored traffic laws, riding on sidewalks, and causing accidents. Additionally, the scooters were frequently left haphazardly around public spaces, obstructing pedestrians. City officials were under increasing pressure to act, and after numerous public consultations and debates, the council finally passed new regulations. The new rules included mandatory helmet use, restricted riding areas, and designated parking zones for scooters. The implementation of these regulations was expected to improve safety and the overall experience for both scooter users and pedestrians.",
        "With the rise of shared electric scooters and bikes in cities across the country, municipal governments have been scrambling to develop effective policies to handle this new form of transportation. Many cities, including the local area, were caught off guard by the sudden popularity of scooters, and their original infrastructure was ill-prepared for this new trend. Early attempts to regulate the scooters were chaotic and ineffective, often leading to public frustration. Some cities took drastic steps, such as banning scooters altogether, while others focused on infrastructure improvements, like adding dedicated lanes for scooters and bicycles. The city council's recent approval of new regulations was part of a larger effort to stay ahead of the curve and provide a balanced approach to regulating modern transportation options while encouraging their growth. These regulations were designed not only to ensure the safety of riders but also to integrate the scooters more seamlessly into the city's broader transportation network.",
        "",
        "The sun hung low in the sky, casting a warm golden glow over the city as Emily wandered through the bustling streets, her mind drifting between thoughts of the past and the uncertain future. She passed the familiar old bookstore that always smelled like aged paper and adventure, a place she used to frequent with her grandmother, whose absence still left a hollow ache in her chest. The air was thick with the scent of coffee wafting from nearby cafés, mingling with the earthy smell of rain that had yet to fall. Despite the noise of the traffic, the chatter of pedestrians, and the hum of city life, there was a strange sense of stillness around her. It was as if time had slowed down, giving her a moment to breathe, to collect her scattered thoughts. She glanced up at the towering buildings that seemed to stretch endlessly into the sky, their glass facades reflecting the fading light. Everything around her was in constant motion, yet she felt an unexpected calm. Her phone buzzed in her pocket, pulling her back to reality, and she sighed, reluctantly slipping it out. It was a message from her best friend, asking if they still wanted to meet up later."
    ]

    result = summarizer.run(contexts, atom)
    print(f"Summarizer result: {result}")

    # Print the results
    for i, elem in enumerate(result):
        context = elem["context"]
        summary = elem["summary"]
        probability = elem["probability"]
        print(f"\n\nContext #{i + 1}: {context}\n--> Summary #{i + 1}: {summary}\n--> Probability #{i + 1}: {probability}")
else:
    context = """In the past year, the city had seen a rapid increase in the \
use of electric scooters. They seemed like a perfect solution to reduce \
traffic and provide an eco-friendly transportation option. However, \
problems arose quickly. Riders often ignored traffic laws, riding on \
sidewalks, and causing accidents. Additionally, the scooters were frequently \
left haphazardly around public spaces, obstructing pedestrians. City officials \
were under increasing pressure to act, and after numerous public \
consultations and debates, the council finally passed new regulations. \
The new rules included mandatory helmet use, restricted riding areas, and \
designated parking zones for scooters. The implementation of these regulations \
was expected to improve safety and the overall experience for both scooter \
users and pedestrians."""

    result = summarizer.run([context], None)
    print(f"Summarizer result: {result}")

    # Print the results
    for i, elem in enumerate(result):
        context = elem["context"]
        summary = elem["summary"]
        probability = elem["probability"]
        print(f"\n\nContext #{i + 1}: {context}\n--> Summary #{i + 1}: {summary}\n--> Probability #{i + 1}: {probability}")

print("Done.")
