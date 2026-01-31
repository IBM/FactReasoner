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

# Decompose the input string into atomic units. Use the same Mellea session 
# (context) to revise or decontextualize the atomc units if needed.

import json
import asyncio
import mellea.stdlib.functional as mfuncs

from typing import Any, Dict, List
from mellea.backends import Backend
from mellea.backends.types import ModelOption
from mellea.stdlib.base import SimpleContext
from mellea.stdlib.requirement import check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

# Local imports
from src.fact_reasoner.utils import validate_json_code_block, strip_code_fences, LOOP_BUDGET

INSTRUCTION_ATOMIZER = """
Instructions:
Your task is to break down a given paragraph into a set of atomic units without adding any new information.

Rules:
- An atomic unit is the smallest sentence containing a singular piece of information directly extracted from the provided paragraph.
- Atomic units may contradict one another.
- The paragraph may contain information that is factually incorrect. Even in such cases, you are not to alter any information contained in the paragraph and must produce atomic units that are completely faithful to the information in the paragraph.
- Each atomic unit in the output must check a different piece of information found explicitly in the paragraph.
- Each atomic unit is standalone in that any actual nouns or proper nouns should be used in place of pronouns or anaphors.
- Each atomic unit must not include any information beyond what is explicitly stated in the provided paragraph.
- Where possible, avoid paraphrasing and instead try to only use language used in the paragraph without introducing new words. 
- The output must be a JSON dictionary with the following format and markdown code fences such that each atomic unit has a unique ID:

```json
{
    "id1": "<first atomic unit>",
    "id2": "<second atomic unit>",
    ...
}
```

Use the provided examples to learn your task.

Example 1:
INPUT: Glenn Allen Anzalone (born June 23, 1955), better known by his stage name Glenn Danzig, is an American singer, songwriter, musician, and record producer. He is the founder of the rock bands Misfits, Samhain, and Danzig. He owns the Evilive record label as well as Verotik, an adult-oriented comic book publishing company.
OUTPUT:
```json
{
    "id1": "Glenn Allen Anzalone was born on June 23, 1955.",
    "id2": "Glenn Allen Anzalone is better known by his stage name Glenn Danzig.",
    "id3": "Glenn Danzig is an American singer, songwriter, musician, and record producer.",
    "id4": "Glenn Danzig is the founder of several rock bands, including Misfits, Samhain, and Danzig.",
    "id5": "Glenn Danzig owns the Evilive record label.",
    "id6": "Glenn Danzig owns Verotik, which is an adult-oriented comic book publishing company."
}
```

Example 2:
INPUT: Luiz Inácio Lula da Silva (born 27 October 1945), also known as Lula da Silva or simply Lula, is a Brazilian politician who is the 39th and current president of Brazil since 2023. A member of the Workers' Party, Lula was also the 35th president from 2003 to 2010. He also holds the presidency of the G20 since 2023. Lula quit school after second grade to work, and did not learn to read until he was ten years old. As a teenager, he worked as a metalworker and became a trade unionist.
OUTPUT:
```json
{
    "id1": "Luiz Inácio Lula da Silva was born on October 27, 1945.",
    "id2": "Luiz Inácio Lula da Silva is also known as Lula da Silva or simply Lula.",
    "id3": "Lula is a Brazilian politician.",
    "id4": "Lula is the 39th and current president of Brazil since 2023.",
    "id5": "Lula is a member of the Workers' Party.",
    "id6": "Lula served as the 35th president of Brazil from 2003 to 2010.",
    "id7": "Lula holds the presidency of the G20 since 2023.",
    "id8": "Lula quit school after the second grade to work.",
    "id9": "Lula did not learn to read until he was ten years old.",
    "id10": "As a teenager, Lula worked as a metalworker.",
    "id11": "Lula became a trade unionist."
}
```

Example 3:
INPUT: Zhejiang Huafang Pharmaceutical Co., Ltd. is a leading chemical company based in China that specializes in the research, manufacturing, and sales of various pharmaceutical products, including excipients and intermediates. The company was founded in 2018 and is located in Hangzhou, a city with a rich history in eastern China. Zhejiang Huafang Pharmaceutical Co., Ltd. is committed to providing high-quality products to its customers in the healthcare industry. The company's manufacturing facilities are equipped with state-of-the-art technology and infrastructure that ensure the production of high-quality products. Overall, Zhejiang Huafang Pharmaceutical Co., Ltd. is a reputable pharmaceutical company with a long history of success in the healthcare industry. The company's commitment to quality, innovation, and customer service has made it a leader in the field of pharmaceutical research and development.
OUTPUT:
```json
{
    "id1": "Zhejiang Huafang Pharmaceutical Co., Ltd. is a leading chemical company.",
    "id2": "Zhejiang Huafang Pharmaceutical Co., Ltd. is based in China.",
    "id3": "Zhejiang Huafang Pharmaceutical Co., Ltd. specializes in the research of various pharmaceutical products",
    "id4": "Zhejiang Huafang Pharmaceutical Co., Ltd. specializes in the manufacturing of various pharmaceutical products.",
    "id5": "Zhejiang Huafang Pharmaceutical Co., Ltd. specializes in the sales of various pharmaceutical products.",
    "id6": "Excipients are the pharmaceutical products of the Zhejiang Huafang Pharmaceutical Co., Ltd.",
    "id7": "Intermediates are the pharmaceutical products of the Zhejiang Huafang Pharmaceutical Co., Ltd.",
    "id8": "The company was founded in 2018.",
    "id9": "The company is located in Hangzhou.",
    "id10": "Hangzhou is a city.",
    "id11": "Hangzhou has a rich history in eastern China.",
    "id12": "Zhejiang Huafang Pharmaceutical Co., Ltd. is committed to providing high-quality products to its customers in the healthcare industry.",
    "id13": "The company's manufacturing facilities are equipped with state-of-the-art technology.",
    "id14": "The company's manufacturing facilities are equipped with state-of-the-art infrastructure.",
    "id15": "The company's manufacturing facilities are equipped with state-of-the-art technology and infrastructure that ensure the production of high-quality products.",
    "id16": "Zhejiang Huafang Pharmaceutical Co., Ltd. is a reputable pharmaceutical company.",
    "id17": "Zhejiang Huafang Pharmaceutical Co., Ltd. has a long history of success in the healthcare industry.",
    "id18": "The company is committed to quality.",
    "id19": "The company is committed to innovation.",
    "id20": "The company is committed to customer service.",
    "id21": "The company's commitment to quality, innovation, and customer service has made it a leader in the field of pharmaceutical research.",
    "id22": "The company's commitment to quality, innovation, and customer service has made it a leader in the field of pharmaceutical development."
}
```

Example 4:
INPUT: I'm here to help you make an informed decision. Both the RTX 3060 Ti and RTX 3060 are powerful GPUs, and the difference between them lies in their performance. The RTX 3060 Ti has more CUDA cores (4864 vs 3584) but a lower boost clock speed (1665 MHz vs 1777 MHz) compared to the RTX 3060. In terms of memory bandwidth, the RTX 3060 Ti has a slight edge over the RTX 3060 with a bandwidth of 448 GB/s compared to 360 GB/s. However, the difference is relatively small. It's important to consider other factors such as the power consumption, cooling system, and compatibility with your system when making a decision."
OUTPUT:
```json
{
    "id1": "The RTX 3060 Ti is a powerful GPU.",
    "id2": "The RTX 3060 is a powerful GPU.",
    "id3": "The difference between them lies in their performance.",
    "id4": "The RTX 3060 Ti has more CUDA cores compared to the RTX 3060.",
    "id5": "The RTX 3060 Ti has 4864 CUDA cores.",
    "id6": "The RTX 3060 has 3584 CUDA cores.",
    "id7": "The RTX 3060 Ti has a lower boost clock speed compared to the RTX 3060.",
    "id8": "The RTX 3060 Ti has a boost clock speed of 1665 MHz.",
    "id9": "The RTX 3060 has a boost clock speed of 1777 MHz.",
    "id10": "The RTX 3060 Ti has a slight edge over the RTX 3060 in terms of memory bandwidth.",
    "id11": "The RTX 3060 Ti has a memory bandwidth of 448 GB/s.",
    "id12": "The RTX 3060 has a memory bandwidth of 360 GB/s.",
    "id13": "The difference is relatively small.",
}
```

Your task:
INPUT: {{response}}
OUTPUT:
"""

class Atomizer(object):
    """
    The Atomizer class implements the atomic decomposition of the response.
    For our purpose, an atomic unit or atom is either a fact or a claim.
    """

    def __init__(
        self,
        backend: Backend,
    ):
        """
        Initialize the Atomizer.

        Args:
            backend: Backend
                The Mellea backend to use for LLM interactions.
        """ 

        # Safety checks        
        if backend is None:
            raise ValueError("Mellea backend is None. Please provide a valid Mellea backend.")

        # Initialize the extractor
        self.backend = backend
        
        # Print info
        print(f"[Atomizer] Using Mellea backend: {self.backend.model_id}")

    def run(self, response: str) -> Dict[str, str]:
        """
        Extract atomic units from a single response.
        
        Args:
            response: str
                The response from which to extract atomic units.
        Returns:
            Dict[str, str]: A dictionary containing the atomic units, each with
            a unique identifier.
        """
        
        # Perform the instruction with validation
        output = mfuncs.instruct(
            INSTRUCTION_ATOMIZER,
            context=SimpleContext(),
            backend=self.backend,
            requirements=[
                check(
                    "The output must be a valid JSON dictionary with markdown code fences",
                    validation_fn=simple_validate(
                        lambda s: validate_json_code_block(s)
                    ),
                )
            ],
            user_variables={"response": response},
            strategy=RejectionSamplingStrategy(loop_budget=LOOP_BUDGET),
            return_sampling_results=True,
        )

        # The output is a validated JSON string; parse it
        if output.success:
            cleaned = strip_code_fences(str(output))
            return json.loads(cleaned)
        else:
            return {} # empty dict on failure
        
    async def run_batch(self, responses: List[str]) -> List[Dict[str, str]]:
        """
        Extract atomic units from a list of responses.
        
        Args:
            responses: List[str]
                The list of response from which to extract atomic units.
        Returns:
            dict: A dictionary containing the number of atomic units, the units themselves,
            all atomic units as dictionaries, and all facts as dictionaries.
        """
        
        # Perform the instruction with validation
        corutines = []
        for response in responses:
            corutine = mfuncs.ainstruct(
                INSTRUCTION_ATOMIZER,
                context=SimpleContext(),
                backend=self.backend,
                requirements=[
                    check(
                        "The output must be a valid JSON dictionary with markdown code fences",
                        validation_fn=simple_validate(
                            lambda s: validate_json_code_block(s)
                        ),
                    )
                ],
                user_variables={"response": response},
                strategy=RejectionSamplingStrategy(loop_budget=LOOP_BUDGET),
                return_sampling_results=True,
            )
            corutines.append(corutine)

        results = []
        print(f"[Atomizer] Awaiting for the async execution ...")
        outputs = await asyncio.gather(*(corutines[i] for i in range(len(corutines))))
        for output in outputs:
            # The output is a validated JSON string; parse it
            if output.success:
                cleaned = strip_code_fences(str(output))
                results.append(json.loads(cleaned))
            else:
                results.append({}) # empty dict on failure

        return results
    
if __name__ == "__main__":

    # Create a Mellea RITS backend
    from mellea_ibm.rits import RITSBackend, RITS
    backend = RITSBackend(
        RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 4096}
    )

    # Disable Mellea logging
    from mellea.helpers.fancy_logger import FancyLogger
    FancyLogger.get_logger().setLevel(FancyLogger.ERROR)

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
