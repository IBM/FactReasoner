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

from typing import Any, Dict
from mellea import MelleaSession
from mellea.backends.types import ModelOption
from mellea.stdlib.requirement import req, check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea_ibm.rits import RITSBackend, RITS

# Local imports
from src.fact_reasoner.utils import validate_json_code_block, strip_code_fences

# v1
INSTRUCTION_ATOMIZER_V1 = """
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

Use the following examples to learn your task.

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
```

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
```

Your task:
INPUT: {{response}}
OUTPUT:
"""

# v2
INSTRUCTION_ATOMIZER_V2 = """
Instructions:
Your task is to exhaustively break down a given paragraph into a set of independent content units without adding any new information.

Rules:
- An independent content unit is the smallest piece of information directly extracted from the provided paragraph.
- Each content unit can take one of the following forms:
  a. Fact: An objective piece of information that can be proven or verified.
  b. Claim: A statement or assertion that expresses a position or viewpoint on a particular topic.
  c. Instruction: A directive or guidance on how to perform a specific task.
  d. Data Format: Any content presented in a specific format, including code, mathematical notations, equations, variables, technical symbols, tables, or structured data formats.
  e. Meta Statement: Disclaimers, acknowledgments, or any other statements about the nature of the response or the responder.
  f. Question: A query or inquiry about a particular topic.
  g. Other: Any other relevant content that doesn't fit into the above categories.
- Label each content unit with its corresponding unit type
- The output must be a JSON dictionary with the following format and markdown code fences: 

```json
{
  "atomic_units": [
    {"id": 1, "text": "<first atomic unit>.", "type": "<unit type>"},
    {"id": 2, "text": "<second atomic unit>.", "type": "<unit type>"},
    ...
  ]
}
```

Refer to the following examples to understand the task and output formats. 

Example 1:
INPUT: Zhejiang Huafang Pharmaceutical Co., Ltd. is a leading chemical company based in China that specializes in the research, manufacturing, and sales of various pharmaceutical products, including excipients and intermediates. The company was founded in 2018 and is located in Hangzhou, a city with a rich history in eastern China. Zhejiang Huafang Pharmaceutical Co., Ltd. is committed to providing high-quality products to its customers in the healthcare industry. The company's manufacturing facilities are equipped with state-of-the-art technology and infrastructure that ensure the production of high-quality products. Overall, Zhejiang Huafang Pharmaceutical Co., Ltd. is a reputable pharmaceutical company with a long history of success in the healthcare industry. The company's commitment to quality, innovation, and customer service has made it a leader in the field of pharmaceutical research and development.
OUTPUT:
```json
{
  "atomic_units": [
    {"id": 1, "text": "Zhejiang Huafang Pharmaceutical Co., Ltd. is a leading chemical company.", "type": "Fact"},
    {"id": 2, "text": "Zhejiang Huafang Pharmaceutical Co., Ltd. is based in China.", "type": "Fact"},
    {"id": 3, "text": "Zhejiang Huafang Pharmaceutical Co., Ltd. specializes in the research of various pharmaceutical products", "type": "Fact"},
    {"id": 4, "text": "Zhejiang Huafang Pharmaceutical Co., Ltd. specializes in the manufacturing of various pharmaceutical products.", "type": "Fact"},
    {"id": 5, "text": "Zhejiang Huafang Pharmaceutical Co., Ltd. specializes in the sales of various pharmaceutical products.", "type": "Fact"},
    {"id": 6, "text": "Excipients are the pharmaceutical products of the Zhejiang Huafang Pharmaceutical Co., Ltd.", "type": "Fact"},
    {"id": 7, "text": "Intermediates are the pharmaceutical products of the Zhejiang Huafang Pharmaceutical Co., Ltd.", "type": Fact},
    {"id": 8, "text": "The company was founded in 2018.", "type": "Fact"},
    {"id": 9, "text": "The company is located in Hangzhou.", "type": "Fact"},
    {"id": 10, "text": "Hangzhou is a city.", "type": "Fact"},
    {"id": 11, "text": "Hangzhou has a rich history in eastern China.", "type": "Fact"},
    {"id": 12, "text": "Zhejiang Huafang Pharmaceutical Co., Ltd. is committed to providing high-quality products to its customers in the healthcare industry.", "type": "Claim"},
    {"id": 13, "text": "The company's manufacturing facilities are equipped with state-of-the-art technology.", "type": "Fact"},
    {"id": 14, "text": "The company's manufacturing facilities are equipped with state-of-the-art infrastructure.", "type": "Fact"},
    {"id": 15, "text": "The company's manufacturing facilities are equipped with state-of-the-art technology and infrastructure that ensure the production of high-quality products.", "type": "Claim"},
    {"id": 16, "text": "Zhejiang Huafang Pharmaceutical Co., Ltd. is a reputable pharmaceutical company.", "type": "Claim"},
    {"id": 17, "text": "Zhejiang Huafang Pharmaceutical Co., Ltd. has a long history of success in the healthcare industry.", "type": "Claim"},
    {"id": 18, "text": "The company is committed to quality.", "type": "Claim"},
    {"id": 19, "text": "The company is committed to innovation.", "type": "Claim"},
    {"id": 20, "text": "The company is committed to customer service.", "type": "Claim"},
    {"id": 21, "text": "The company's commitment to quality, innovation, and customer service has made it a leader in the field of pharmaceutical research.", "type": "Claim"},
    {"id": 22, "text": "The company's commitment to quality, innovation, and customer service has made it a leader in the field of pharmaceutical development.", "type": "Claim"}
}
```

Example 2:
INPUT: I'm here to help you make an informed decision. Both the RTX 3060 Ti and RTX 3060 are powerful GPUs, and the difference between them lies in their performance. The RTX 3060 Ti has more CUDA cores (4864 vs 3584) but a lower boost clock speed (1665 MHz vs 1777 MHz) compared to the RTX 3060. In terms of memory bandwidth, the RTX 3060 Ti has a slight edge over the RTX 3060 with a bandwidth of 448 GB/s compared to 360 GB/s. However, the difference is relatively small. It's important to consider other factors such as the power consumption, cooling system, and compatibility with your system when making a decision."
OUTPUT:
```json
{
  "atomic_units": [ 
    {"id": 1, "text": "I'm here to help you make an informed decision.", "type": "Meta Statement"},
    {"id": 2, "text": "The RTX 3060 Ti is a powerful GPU.", "type": "Claim"},
    {"id": 3, "text": "The RTX 3060 is a powerful GPU." "type": "Claim"},
    {"id": 4, "text": "The difference between them lies in their performance.", "type": "Claim"},
    {"id": 5, "text": "The RTX 3060 Ti has more CUDA cores compared to the RTX 3060.", "type": "Fact"},
    {"id": 6, "text": "The RTX 3060 Ti has 4864 CUDA cores.", "type": "Fact"},
    {"id": 7, "text": "The RTX 3060 has 3584 CUDA cores.", "type": "Fact"},
    {"id": 8, "text": "The RTX 3060 Ti has a lower boost clock speed compared to the RTX 3060.", "type": "Fact"},
    {"id": 9, "text": "The RTX 3060 Ti has a boost clock speed of 1665 MHz.", "type": "Fact"},
    {"id": 10, "text": "The RTX 3060 has a boost clock speed of 1777 MHz.", "type": "Fact"},
    {"id": 11, "text": "The RTX 3060 Ti has a slight edge over the RTX 3060 in terms of memory bandwidth.", "type": "Fact"},
    {"id": 12, "text": "The RTX 3060 Ti has a memory bandwidth of 448 GB/s.", "type": "Fact"},
    {"id": 13, "text": "The RTX 3060 has a memory bandwidth of 360 GB/s.", "type": "Fact"},
    {"id": 14, "text": "The difference is relatively small.", "type": "Claim"},
    {"id": 15, "text": "It's important to consider other factors such as power consumption when making a decision.", "type": "Instruction"},
    {"id": 16, "text": "It's important to consider other factors such as cooling system when making a decision.", "type": "Instruction"},
    {"id": 17, "text": "It's important to consider other factors such as compatibility with your system when making a decision.", "type": "Instruction"}
}
```

Your Task:
INPUT: {{response}}
OUTPUT:
# """


class AtomExtractor(object):
    """
    The AtomExtractor class implements the atomic decomposition of the response.
    For our purpose, an atomic unit or atom is either a fact or a claim.
    """

    def __init__(
        self,
        session: MelleaSession,
        version: str = "v1"
    ):
        """
        Initialize the AtomExtractor.

        Args:
            session: MelleaSession
                The Mellea session to use for LLM interactions.
            version: str
                The version of the atomizer. Supported versions are "v1" and "v2".
        """ 

        # Safety checks        
        if version not in ["v1", "v2"]:
            raise ValueError(f"Unknown Atomizer version: {version}. Supported versions are: 'v1', 'v2'.")
        if session is None:
            raise ValueError("Mellea session is None. Please provide a valid Mellea session.")

        # Initialize the extractor
        self.session = session
        self.version = version
        
        print(f"[Atomizer] Using LLM in Mellea session: {self.session.backend.model_id}")
        print(f"[Atomizer] Using prompt version: {self.version}")

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

        # Prepare instruction based on version
        if self.version == "v1":
            instruction = INSTRUCTION_ATOMIZER_V1
        elif self.version == "v2":
            instruction = INSTRUCTION_ATOMIZER_V2
        else:
            raise ValueError(f"Unknown Atomizer version: {self.version}. Supported versions are: 'v1', 'v2'.")
        
        # Perform the instruction with validation
        output = self.session.instruct(
            instruction,
            requirements=[
                check(
                    "The output must be a valid JSON dictionary with markdown code fences.",
                    validation_fn=simple_validate(lambda s: validate_json_code_block(s)),
                )
            ],
            user_variables={"response": response},
            strategy=RejectionSamplingStrategy(loop_budget=3),
            return_sampling_results=True
        )

        # The output is a validated JSON string; parse it
        if output.success:
            cleaned = strip_code_fences(str(output))
            return json.loads(cleaned)
        else:
            return {} # empty dict on failure
                        

if __name__ == "__main__":

    # Create a Mellea session with RITS backend
    m = MelleaSession(
        backend=RITSBackend(
            RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 500}
        )
    )

    # Create the atomizer
    atomizer = AtomExtractor(session=m, version="v2")

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
    result = atomizer.run(response)
    print(f"Atomization result: {result}")

    # Print the extracted atomic units
    atoms = result.get("atomic_units", [])
    print(f"Extracted {len(atoms)} atomic units:")
    for atom in atoms:
        print(f"Atom {atom['id']}: {atom['text']}")


    print("Done.")
