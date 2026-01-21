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

from typing import Any, Dict
from mellea.backends import Backend
from mellea.backends.types import ModelOption
from mellea.stdlib.base import SimpleContext, CBlock
from mellea.stdlib.requirement import req, check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

# Local imports
from src.fact_reasoner.utils import validate_json_code_block, strip_code_fences

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
- The output must be a JSON dictionary with the following format and markdown code fences:

```json
{
  "atomic_units": [
    {"id": 1, "text": "<first atomic unit>."},
    {"id": 2, "text": "<second atomic unit>."},
    ...
  ]
}
```

Use the provided examples to learn your task.

Example 1:
INPUT: Glenn Allen Anzalone (born June 23, 1955), better known by his stage name Glenn Danzig, is an American singer, songwriter, musician, and record producer. He is the founder of the rock bands Misfits, Samhain, and Danzig. He owns the Evilive record label as well as Verotik, an adult-oriented comic book publishing company.
OUTPUT:
```json
{
    "atomic_units": [
        {"id": 1, "text": "Glenn Allen Anzalone was born on June 23, 1955."},
        {"id": 2, "text": "Glenn Allen Anzalone is better known by his stage name Glenn Danzig."},
        {"id": 3, "text": "Glenn Danzig is an American singer, songwriter, musician, and record producer."},
        {"id": 4, "text": "Glenn Danzig is the founder of several rock bands, including Misfits, Samhain, and Danzig."},
        {"id": 5, "text": "Glenn Danzig owns the Evilive record label."},
        {"id": 6, "text": "Glenn Danzig owns Verotik, which is an adult-oriented comic book publishing company."}
    ]
}

Example 2:
INPUT: Luiz Inácio Lula da Silva (born 27 October 1945), also known as Lula da Silva or simply Lula, is a Brazilian politician who is the 39th and current president of Brazil since 2023. A member of the Workers' Party, Lula was also the 35th president from 2003 to 2010. He also holds the presidency of the G20 since 2023. Lula quit school after second grade to work, and did not learn to read until he was ten years old. As a teenager, he worked as a metalworker and became a trade unionist.
OUTPUT:
```json
{
    "atomic_units": [
        {"id": 1, "text": "Luiz Inácio Lula da Silva was born on October 27, 1945."},
        {"id": 2, "text": "Luiz Inácio Lula da Silva is also known as Lula da Silva or simply Lula."},
        {"id": 3, "text": "Lula is a Brazilian politician."},
        {"id": 4, "text": "Lula is the 39th and current president of Brazil since 2023."},
        {"id": 5, "text": "Lula is a member of the Workers' Party."},
        {"id": 6, "text": "Lula served as the 35th president of Brazil from 2003 to 2010."},
        {"id": 7, "text": "Lula holds the presidency of the G20 since 2023."},
        {"id": 8, "text": "Lula quit school after the second grade to work."},
        {"id": 9, "text": "Lula did not learn to read until he was ten years old."},
        {"id": 10, "text": "As a teenager, Lula worked as a metalworker."},
        {"id": 11, "text": "Lula became a trade unionist."}
    ]
}

Example 3:
INPUT: Zhejiang Huafang Pharmaceutical Co., Ltd. is a leading chemical company based in China that specializes in the research, manufacturing, and sales of various pharmaceutical products, including excipients and intermediates. The company was founded in 2018 and is located in Hangzhou, a city with a rich history in eastern China. Zhejiang Huafang Pharmaceutical Co., Ltd. is committed to providing high-quality products to its customers in the healthcare industry. The company's manufacturing facilities are equipped with state-of-the-art technology and infrastructure that ensure the production of high-quality products. Overall, Zhejiang Huafang Pharmaceutical Co., Ltd. is a reputable pharmaceutical company with a long history of success in the healthcare industry. The company's commitment to quality, innovation, and customer service has made it a leader in the field of pharmaceutical research and development.
OUTPUT:
```json
{
    "atomic_units": [
        {"id": 1, "text": "Zhejiang Huafang Pharmaceutical Co., Ltd. is a leading chemical company."},
        {"id": 2, "text": "Zhejiang Huafang Pharmaceutical Co., Ltd. is based in China."},
        {"id": 3, "text": "Zhejiang Huafang Pharmaceutical Co., Ltd. specializes in the research of various pharmaceutical products"},
        {"id": 4, "text": "Zhejiang Huafang Pharmaceutical Co., Ltd. specializes in the manufacturing of various pharmaceutical products."},
        {"id": 5, "text": "Zhejiang Huafang Pharmaceutical Co., Ltd. specializes in the sales of various pharmaceutical products."},
        {"id": 6, "text": "Excipients are the pharmaceutical products of the Zhejiang Huafang Pharmaceutical Co., Ltd."},
        {"id": 7, "text": "Intermediates are the pharmaceutical products of the Zhejiang Huafang Pharmaceutical Co., Ltd."},
        {"id": 8, "text": "The company was founded in 2018."},
        {"id": 9, "text": "The company is located in Hangzhou."},
        {"id": 10, "text": "Hangzhou is a city."},
        {"id": 11, "text": "Hangzhou has a rich history in eastern China."},
        {"id": 12, "text": "Zhejiang Huafang Pharmaceutical Co., Ltd. is committed to providing high-quality products to its customers in the healthcare industry."},
        {"id": 13, "text": "The company's manufacturing facilities are equipped with state-of-the-art technology."},
        {"id": 14, "text": "The company's manufacturing facilities are equipped with state-of-the-art infrastructure."},
        {"id": 15, "text": "The company's manufacturing facilities are equipped with state-of-the-art technology and infrastructure that ensure the production of high-quality products."},
        {"id": 16, "text": "Zhejiang Huafang Pharmaceutical Co., Ltd. is a reputable pharmaceutical company."},
        {"id": 17, "text": "Zhejiang Huafang Pharmaceutical Co., Ltd. has a long history of success in the healthcare industry."},
        {"id": 18, "text": "The company is committed to quality."},
        {"id": 19, "text": "The company is committed to innovation."},
        {"id": 20, "text": "The company is committed to customer service."},
        {"id": 21, "text": "The company's commitment to quality, innovation, and customer service has made it a leader in the field of pharmaceutical research."},
        {"id": 22, "text": "The company's commitment to quality, innovation, and customer service has made it a leader in the field of pharmaceutical development."}
    }

Example 4:
INPUT: I'm here to help you make an informed decision. Both the RTX 3060 Ti and RTX 3060 are powerful GPUs, and the difference between them lies in their performance. The RTX 3060 Ti has more CUDA cores (4864 vs 3584) but a lower boost clock speed (1665 MHz vs 1777 MHz) compared to the RTX 3060. In terms of memory bandwidth, the RTX 3060 Ti has a slight edge over the RTX 3060 with a bandwidth of 448 GB/s compared to 360 GB/s. However, the difference is relatively small. It's important to consider other factors such as the power consumption, cooling system, and compatibility with your system when making a decision."
OUTPUT:
```json
{
    "atomic_units": [ 
        {"id": 1, "text": "The RTX 3060 Ti is a powerful GPU."},
        {"id": 2, "text": "The RTX 3060 is a powerful GPU."},
        {"id": 3, "text": "The difference between them lies in their performance."},
        {"id": 4, "text": "The RTX 3060 Ti has more CUDA cores compared to the RTX 3060."},
        {"id": 5, "text": "The RTX 3060 Ti has 4864 CUDA cores."},
        {"id": 6, "text": "The RTX 3060 has 3584 CUDA cores."},
        {"id": 7, "text": "The RTX 3060 Ti has a lower boost clock speed compared to the RTX 3060."},
        {"id": 8, "text": "The RTX 3060 Ti has a boost clock speed of 1665 MHz."},
        {"id": 9, "text": "The RTX 3060 has a boost clock speed of 1777 MHz."},
        {"id": 10, "text": "The RTX 3060 Ti has a slight edge over the RTX 3060 in terms of memory bandwidth."},
        {"id": 11, "text": "The RTX 3060 Ti has a memory bandwidth of 448 GB/s."},
        {"id": 12, "text": "The RTX 3060 has a memory bandwidth of 360 GB/s."},
        {"id": 13, "text": "The difference is relatively small."},
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

    def run(self, response: str) -> Dict[str, Any]:
        """
        Extract atomic units from a single response.
        
        Args:
            response: str
                The response from which to extract atomic units.
        Returns:
            dict: A dictionary containing the number of atomic units, the units themselves,
            all atomic units as dictionaries, and all facts as dictionaries.
        """
        
        # Perform the instruction with validation
        output = mfuncs.instruct(
            INSTRUCTION_ATOMIZER,
            context=SimpleContext(),
            backend=self.backend,
            requirements=[
                check(
                    "The output must be a valid JSON dictionary with markdown code fences.",
                    validation_fn=simple_validate(
                        lambda s: validate_json_code_block(s, required_keys=["atomic_units"])
                    ),
                )
            ],
            user_variables={"response": response},
            strategy=RejectionSamplingStrategy(loop_budget=3),
            return_sampling_results=True,
        )

        # The output is a validated JSON string; parse it
        if output.success:
            cleaned = strip_code_fences(str(output))
            return json.loads(cleaned)
        else:
            return {} # empty dict on failure
                        
    async def arun(self, response: str) -> Dict[str, Any]:
        """
        Extract atomic units from a single response.
        
        Args:
            response: str
                The response from which to extract atomic units.
        Returns:
            dict: A dictionary containing the atomic units, each one being a dict
            with 'id' and 'text' as keys.
        """
        
        # Perform the instruction with validation
        output = await mfuncs.ainstruct(
            INSTRUCTION_ATOMIZER,
            context=SimpleContext(),
            backend=self.backend,
            requirements=[
                check(
                    "The output must be a valid JSON dictionary with markdown code fences.",
                    validation_fn=simple_validate(
                        lambda s: validate_json_code_block(s, required_keys=["atomic_units"])
                    ),
                )
            ],
            user_variables={"response": response},
            strategy=RejectionSamplingStrategy(loop_budget=3),
            return_sampling_results=True,
        )

        # The output is a validated JSON string; parse it
        if output.success:
            cleaned = strip_code_fences(str(output))
            return json.loads(cleaned)
        else:
            return {} # empty dict on failure

if __name__ == "__main__":

    use_async = False

    # Create a Mellea RITS backend
    from mellea_ibm.rits import RITSBackend, RITS
    backend = RITSBackend(
        RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 500}
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
      
    
    # Process the response to extract atomic units
    if not use_async:
        result = atomizer.run(response)
        print(f"Atomization result: {result}")
    else:
        result = asyncio.run(atomizer.arun(response))
        print(f"Atomization result: {result}")

    # Print the extracted atomic units
    atoms = result.get("atomic_units", [])
    print(f"Extracted {len(atoms)} atomic units:")
    for atom in atoms:
        print(f"Atom {atom['id']}: {atom['text']}")

    print("Done.")
