# coding=utf-8
# Copyright 2023-present the International Business Machines.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Context summarization using LLMs

import json
import string
import asyncio
import numpy as np
import mellea.stdlib.functional as mfuncs

from typing import Any, Dict, List, Union
from mellea.backends import Backend
from mellea.backends.types import ModelOption
from mellea.stdlib.base import SimpleContext, CBlock
from mellea.stdlib.requirement import req, check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.stdlib.base import ModelOutputThunk
from mellea.stdlib.sampling import SamplingResult

# Local imports
from src.fact_reasoner.utils import strip_code_fences, validate_markdown_code_block

INSTRUCTION_CONTEXT_SUMMARY = """

Your task is to summarize the CONTEXT with respect to the ATOM.

Instructions: 
Follow the steps below for CONTEXT summarization:
1. The ATOM can be true, false or not verifiable according to the SUMMARY.
2. It is very possible that no relevant information about the ATOM or related to the ATOM can be found in the CONTEXT. In this case, the SUMMARY must be: "None".
3. If the CONTEXT does not provide information about the ATOM, or if the CONTEXT does not mention anything related to the ATOM, the SUMMARY must be: "None".
4. If the CONTEXT provides information about the ATOM, the SUMMARY must contain the most relevant information of the CONTEXT and be such that we can fact-check the ATOM using this SUMMARY. 
5. The SUMMARY must not use reported speech to refer to the CONTEXT, for instance the SUMMARY must NOT state: "according to the context", "this context mentions", or "this article outlines", but instead the SUMMARY must only summarize the CONTEXT.
6. If the CONTEXT provides information about the ATOM, provide the SUMMARY and wrap the SUMMARY in a markdown code block.
7. If the CONTEXT does not provide information about the ATOM, the SUMMARY must only provide "None". Provide "None" and wrap it in a markdown code block. Do not mention that the context does not provide any information about the atom. Do not provide anything else.

Use the provided examples to learn the task better.

Your task:
ATOM: {{atom_text}}

CONTEXT: {{context}}

SUMMARY:
"""

# TODO: implement this once we know how to get the logits from Mellea
def get_probability(output: Union[ModelOutputThunk, SamplingResult]) -> float:
    """
    Compute the average log probability of the generated tokens.

    Args:
        output: Union[ModelOutputThunk, SamplingResult]
            The model raw output (via Mellea).

    Returns:
        float: The average log probability of the generated tokens.
    """

    return 1.0

class ContextSummarizer:
    """
    Context summarization given the atom.
    """
    def __init__(
            self,
            backend: Backend
    ):
        """
        Initialize the ContextSummarizer.

        Args:
            backend: str
                The Mellea backend to use for LLM interaction.
        """
        
        # Safety checks        
        if backend is None:
            raise ValueError("Mellea backend is None. Please provide a valid Mellea backend.")

        # Initialize the extractor
        self.backend = backend
        
        # Print info
        print(f"[Summarizer] Using Mellea backend: {self.backend.model_id}")

        # In-context learning examples
        self.icl_examples = [
            """Example 1:
            ATOM: Sense and Sensibility was published in the summer of 1811.

            CONTEXT: + Sense and Sensibility + Sense and Sensibility is a novel by Jane \
            Austen , published in 1811 . + Jane Austen + Jane Austen ( 16 December 1775 - 18 July \
            1817 ) was an English novelist known primarily for her six major novels , which interpret , \
            critique and comment upon the British landed gentry at the end of the 18th century .

            SUMMARY:
            ```
            Sense and Sensibility was published in 1811, however it is not known whether it \
            has been published in summer.
            ```
            """,
            """Example 2:
            ATOM: Filmfare is about cheese.

            CONTEXT: + Filmfare + Filmfare is an English-language , tabloid-sized magazine \
            about Hindi-language cinema , popularly known as Bollywood . + Bollywood + Bollywood \
            is the sobriquet for India 's Hindi language film industry , based in the city of Mumbai , \
            Maharashtra .

            SUMMARY: 
            ```
            Filmfare is about Hindi-language cinema, not about cheese.
            ```
            """,
            """Example 3:
            ATOM: The 19th G7 summit only included Russia.

            CONTEXT:
            + 19th G7 summit + The Group of Seven ( G7 ) was an unofficial forum \
            which brought together the heads of the richest industrialized countries : France , Germany \
            , Italy , Japan , the United Kingdom , the United States , Canada ( since 1976 ) and the \
            President of the European Commission ( starting officially in 1981 ) .

            SUMMARY: 
            ```
            The 19th G7 summit did not only include Russia, but also the heads of the six \
            other richest industrialized countries and the President of the European Commission.
            ```
            """,
            """Example 4:
            ATOM: Quantum mechanics describes the behavior of particles at the smallest scales, where classical physics no longer applies.

            CONTEXT:
            The Amazon rainforest, often referred to as the "lungs of the Earth," spans over 5.5 million square kilometers across nine countries. \
            It is home to millions of species, many of which are yet to be discovered. The rainforest plays a crucial role in global oxygen production \
            and carbon dioxide absorption. However, it faces severe threats from deforestation, illegal mining, and climate change. Conservation efforts \
            are ongoing, with governments, environmental organizations, and indigenous communities working together to protect this vital ecosystem. 

            SUMMARY:
            ```
            None
            ```
            """,
            """Example 5:
            ATOM: Zeus was the creator of Nazgul.

            CONTEXT:
            + Artemis + She was the Hellenic goddess of the hunt , wild animals , \
            wilderness , childbirth , virginity and protector of young girls , bringing and relieving \
            disease in women ; she often was depicted as a huntress carrying a bow and arrows .

            SUMMARY:
            ```
            None
            ```
            """,
        ]

    def run(self, contexts: List[str], atom_text: str) -> List[Dict[str, Any]]:
        """
        Summarize a list of contexts with respect to an atomic unit.
        
        Args:
            contexts: List[str]
                The list of contexts to be summarized.
            atom_text: str
                The reference atomic unit text
        Returns:
            List[Dict[str, Any]]: A list of summarized contexts.
        """
        
        # Perform the instruction with validation
        results = []
        for context in contexts:
            output = mfuncs.instruct(
                INSTRUCTION_CONTEXT_SUMMARY,
                context=SimpleContext(),
                backend=self.backend,
                requirements=[
                    check(
                        "The output must be a valid markdown code block.",
                        validation_fn=simple_validate(
                            lambda s: validate_markdown_code_block(s)
                        ),
                    )
                ],
                user_variables={"context": context, "atom_text": atom_text},
                icl_examples=self.icl_examples,
                strategy=RejectionSamplingStrategy(loop_budget=3),
                return_sampling_results=True,
            )

            # The output is a validated JSON string; parse it
            if output.success:
                cleaned = strip_code_fences(str(output))
                results.append(
                    {
                        "context": context,
                        "summary": cleaned if cleaned != "None" else "",
                        "probability": get_probability(output)
                    }
                )

        return results

    async def arun(self, contexts: List[str], atom_text: str) -> List[str]:
        """
        Summarize a list of contexts with respect to an atomic unit.
        
        Args:
            contexts: List[str]
                The list of contexts to be summarized.
            atom_text: str
                The reference atomic unit text
        Returns:
            List[str]: A list of summarized contexts.
        """
        
        # Perform the instruction with validation
        results = []
        for context in contexts:
            output = await mfuncs.ainstruct(
                INSTRUCTION_CONTEXT_SUMMARY,
                context=SimpleContext(),
                backend=self.backend,
                requirements=[
                    check(
                        "The output must be a valid markdown code block.",
                        validation_fn=simple_validate(
                            lambda s: validate_markdown_code_block(s)
                        ),
                    )
                ],
                user_variables={"context": context, "atom_text": atom_text},
                icl_examples=self.icl_examples,
                strategy=RejectionSamplingStrategy(loop_budget=3),
                return_sampling_results=True,
            )

            # The output is a validated JSON string; parse it
            if output.success:
                cleaned = strip_code_fences(str(output))
                results.append(
                    {
                        "context": context,
                        "summary": cleaned if cleaned != "None" else "",
                        "probability": get_probability(output)
                    }
                )

        return results
       
if __name__ == "__main__":
    
    use_async = False

    # Create a Mellea RITS backend
    from mellea_ibm.rits import RITSBackend, RITS
    backend = RITSBackend(
        RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 500}
    )

    # Create the context summarizer
    summarizer = ContextSummarizer(backend=backend)

    atom = "The city council has approved new regulations for electric scooters."
    contexts = ["In the past year, the city had seen a rapid increase in the use of electric scooters. They seemed like a perfect solution to reduce traffic and provide an eco-friendly transportation option. However, problems arose quickly. Riders often ignored traffic laws, riding on sidewalks, and causing accidents. Additionally, the scooters were frequently left haphazardly around public spaces, obstructing pedestrians. City officials were under increasing pressure to act, and after numerous public consultations and debates, the council finally passed new regulations. The new rules included mandatory helmet use, restricted riding areas, and designated parking zones for scooters. The implementation of these regulations was expected to improve safety and the overall experience for both scooter users and pedestrians.",
        "With the rise of shared electric scooters and bikes in cities across the country, municipal governments have been scrambling to develop effective policies to handle this new form of transportation. Many cities, including the local area, were caught off guard by the sudden popularity of scooters, and their original infrastructure was ill-prepared for this new trend. Early attempts to regulate the scooters were chaotic and ineffective, often leading to public frustration. Some cities took drastic steps, such as banning scooters altogether, while others focused on infrastructure improvements, like adding dedicated lanes for scooters and bicycles. The city council's recent approval of new regulations was part of a larger effort to stay ahead of the curve and provide a balanced approach to regulating modern transportation options while encouraging their growth. These regulations were designed not only to ensure the safety of riders but also to integrate the scooters more seamlessly into the city's broader transportation network.",
        "",
        "The sun hung low in the sky, casting a warm golden glow over the city as Emily wandered through the bustling streets, her mind drifting between thoughts of the past and the uncertain future. She passed the familiar old bookstore that always smelled like aged paper and adventure, a place she used to frequent with her grandmother, whose absence still left a hollow ache in her chest. The air was thick with the scent of coffee wafting from nearby cafés, mingling with the earthy smell of rain that had yet to fall. Despite the noise of the traffic, the chatter of pedestrians, and the hum of city life, there was a strange sense of stillness around her. It was as if time had slowed down, giving her a moment to breathe, to collect her scattered thoughts. She glanced up at the towering buildings that seemed to stretch endlessly into the sky, their glass facades reflecting the fading light. Everything around her was in constant motion, yet she felt an unexpected calm. Her phone buzzed in her pocket, pulling her back to reality, and she sighed, reluctantly slipping it out. It was a message from her best friend, asking if they still wanted to meet up later."
    ]

    if not use_async:
        result = summarizer.run(contexts, atom)
        print(f"Summarizer result: {result}")
    else:
        result = asyncio.run(summarizer.arun(contexts, atom))
        print(f"Summarizer result: {result}")

    # Print the results
    for i, elem in enumerate(result):
        context = elem["context"]
        summary = elem["summary"]
        probability = elem["probability"]
        print(f"\n\nContext #{i + 1}: {context}\n--> Summary #{i + 1}: {summary}\n--> Probability #{i + 1}: {probability}")


    print("Done.")
