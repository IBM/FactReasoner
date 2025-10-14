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

# Split the input text into atomic facts/claims (based on FactBench/VERIFY).

import sys
from typing import Any, Dict, List

from fm_factual.llm_handler import LLMHandler
from fm_factual.utils import DEFAULT_PROMPT_BEGIN, DEFAULT_PROMPT_END, MODELS
from tqdm import tqdm

# v1
ATOM_EXTRACTION_PROMPT1 = """{_PROMPT_BEGIN_PLACEHOLDER}
Instructions:
1. You are given a paragraph. Your task is to break the sentence down into a list of atomic statements without adding any new information.
2. An atomic statement is a sentence containing a singular piece of information directly extracted from the provided paragraph.
3. Atomic statements may contradict one another.
4. The paragraph may contain information that is factually incorrect. Even in such cases, you are not to alter any information contained in the paragraph and must produce atomic statements that are completely faithful to the information in the paragraph.
5. Each atomic statement in the outputted list should check a different piece of information found explicitly in the paragraph.
6. Each atomic statement is standalone in that any actual nouns or proper nouns should be used in place of pronouns or anaphors.
7. Each atomic statement must not include any information beyond what is explicitly stated in the provided paragraph.
8. Where possible, avoid paraphrasing and instead try to only use language used in the paragraph without introducing new words. 
9. Use the previous examples to learn how to do this.
10. You should only output the atomic statement as a list, with each item starting with "- ". Do not include other formatting.
11. Your task is to do this for the last paragraph that is given. 

Example 1:
Please breakdown the following paragraph into independent statements: Glenn Allen Anzalone (born June 23, 1955), better known by his stage name Glenn Danzig, is an American singer, songwriter, musician, and record producer. He is the founder of the rock bands Misfits, Samhain, and Danzig. He owns the Evilive record label as well as Verotik, an adult-oriented comic book publishing company.
- Glenn Allen Anzalone was born on June 23, 1955.
- Glenn Allen Anzalone is better known by his stage name Glenn Danzig.
- Glenn Danzig is an American singer, songwriter, musician, and record producer.
- Glenn Danzig is the founder of several rock bands, including Misfits, Samhain, and Danzig.
- Glenn Danzig owns the Evilive record label.
- Glenn Danzig owns Verotik, which is an adult-oriented comic book publishing company.

Example 2:
Please breakdown the following paragraph into independent statements: Luiz Inácio Lula da Silva (born 27 October 1945), also known as Lula da Silva or simply Lula, is a Brazilian politician who is the 39th and current president of Brazil since 2023. A member of the Workers' Party, Lula was also the 35th president from 2003 to 2010. He also holds the presidency of the G20 since 2023. Lula quit school after second grade to work, and did not learn to read until he was ten years old. As a teenager, he worked as a metalworker and became a trade unionist.
- Luiz Inácio Lula da Silva was born on October 27, 1945.
- Luiz Inácio Lula da Silva is also known as Lula da Silva or simply Lula.
- Lula is a Brazilian politician.
- Lula is the 39th and current president of Brazil since 2023.
- Lula is a member of the Workers' Party.
- Lula served as the 35th president of Brazil from 2003 to 2010.
- Lula holds the presidency of the G20 since 2023.
- Lula quit school after the second grade to work.
- Lula did not learn to read until he was ten years old.
- As a teenager, Lula worked as a metalworker.
- Lula became a trade unionist.

Your task:
Please breakdown the following paragraph into independent statements: {_RESPONSE_PLACEHOLDER}{_PROMPT_END_PLACEHOLDER}
"""

# v2
ATOM_EXTRACTION_PROMPT2 = """{_PROMPT_BEGIN_PLACEHOLDER}

Instructions: 
- Exhaustively break down the following text into independent content units. Each content unit can take one of the following forms:
  a. Fact: An objective piece of information that can be proven or verified.
  b. Claim: A statement or assertion that expresses a position or viewpoint on a particular topic.
  c. Instruction: A directive or guidance on how to perform a specific task.
  d. Data Format: Any content presented in a specific format, including code, mathematical notations, equations, variables, technical symbols, tables, or structured data formats.
  e. Meta Statement: Disclaimers, acknowledgments, or any other statements about the nature of the response or the responder.
  f. Question: A query or inquiry about a particular topic.
  g. Other: Any other relevant content that doesn't fit into the above categories.
- Label each content unit with its corresponding unit type using the format: [content unit]: [content unit type]
- You should only output the independent content units as a list, with each item starting with "- ". Do not include other formatting or preamble text.
- Refer to the following examples to understand the task and output formats. 

Example 1:
TEXT: Zhejiang Huafang Pharmaceutical Co., Ltd. is a leading chemical company based in China that specializes in the research, manufacturing, and sales of various pharmaceutical products, including excipients and intermediates. The company was founded in 2018 and is located in Hangzhou, a city with a rich history in eastern China. Zhejiang Huafang Pharmaceutical Co., Ltd. is committed to providing high-quality products to its customers in the healthcare industry. The company's manufacturing facilities are equipped with state-of-the-art technology and infrastructure that ensure the production of high-quality products. Overall, Zhejiang Huafang Pharmaceutical Co., Ltd. is a reputable pharmaceutical company with a long history of success in the healthcare industry. The company's commitment to quality, innovation, and customer service has made it a leader in the field of pharmaceutical research and development.

UNITS:
- Zhejiang Huafang Pharmaceutical Co., Ltd. is a leading chemical company: Fact
- Zhejiang Huafang Pharmaceutical Co., Ltd. is based in China: Fact
- Zhejiang Huafang Pharmaceutical Co., Ltd. specializes in the research of various pharmaceutical products: Fact
- Zhejiang Huafang Pharmaceutical Co., Ltd. specializes in the manufacturing of various pharmaceutical products: Fact
- Zhejiang Huafang Pharmaceutical Co., Ltd. specializes in the sales of various pharmaceutical products: Fact
- excipients are the pharmaceutical products of the Zhejiang Huafang Pharmaceutical Co., Ltd.: Fact
- intermediates are the pharmaceutical products of the Zhejiang Huafang Pharmaceutical Co., Ltd.: Fact
- The company was founded in 2018: Fact
- The company is located in Hangzhou: Fact
- Hangzhou is a city: Fact
- Hangzhou has a rich history in eastern China: Fact
- Zhejiang Huafang Pharmaceutical Co., Ltd. is committed to providing high-quality products to its customers in the healthcare industry: Claim
- The company's manufacturing facilities are equipped with state-of-the-art technology: Fact
- The company's manufacturing facilities are equipped with state-of-the-art infrastructure: Fact
- The company's manufacturing facilities are equipped with state-of-the-art technology and infrastructure that ensure the production of high-quality products: Claim
- Zhejiang Huafang Pharmaceutical Co., Ltd. is a reputable pharmaceutical company: Claim
- Zhejiang Huafang Pharmaceutical Co., Ltd. has a long history of success in the healthcare industry: Claim
- The company is committed to quality: Claim
- The company is committed to innovation: Claim
- The company is committed to customer service: Claim
- The company's commitment to quality, innovation, and customer service has made it a leader in the field of pharmaceutical research: Claim
- The company's commitment to quality, innovation, and customer service has made it a leader in the field of pharmaceutical development: Claim

Example 2:
TEXT: I'm here to help you make an informed decision. Both the RTX 3060 Ti and RTX 3060 are powerful GPUs, and the difference between them lies in their performance. The RTX 3060 Ti has more CUDA cores (4864 vs 3584) but a lower boost clock speed (1665 MHz vs 1777 MHz) compared to the RTX 3060. In terms of memory bandwidth, the RTX 3060 Ti has a slight edge over the RTX 3060 with a bandwidth of 448 GB/s compared to 360 GB/s. However, the difference is relatively small. It's important to consider other factors such as the power consumption, cooling system, and compatibility with your system when making a decision."

UNITS: 
- I'm here to help you make an informed decision: Meta Statement
- The RTX 3060 Ti is a powerful GPU: Claim
- The RTX 3060 is a powerful GPU: Claim
- The difference between them lies in their performance: Claim
- The RTX 3060 Ti has more CUDA cores compared to the RTX 3060: Fact
- The RTX 3060 Ti has 4864 CUDA cores: Fact
- The RTX 3060 has 3584 CUDA cores: Fact
- The RTX 3060 Ti has a lower boost clock speed compared to the RTX 3060: Fact
- The RTX 3060 Ti has a boost clock speed of 1665 MHz: Fact
- The RTX 3060 has a boost clock speed of 1777 MHz: Fact
- The RTX 3060 Ti has a slight edge over the RTX 3060 in terms of memory bandwidth: Fact
- The RTX 3060 Ti has a memory bandwidth of 448 GB/s: Fact
- The RTX 3060 has a memory bandwidth of 360 GB/s: Fact
- The difference is relatively small: Claim
- It's important to consider other factors such as power consumption when making a decision: Instruction
- It's important to consider other factors such as cooling system when making a decision: Instruction
- It's important to consider other factors such as compatibility with your system when making a decision: Instruction

Your Task:
TEXT: {_RESPONSE_PLACEHOLDER}

UNITS:{_PROMPT_END_PLACEHOLDER}
"""

_ATOM = 'atom'
_LABEL = 'label'

def text_to_units(text: str, separator: str = '- ') -> List[str]:
    parsed_units = []
    parsed_labels = []
    current_unit = []
    preamble = True
    for line in text.strip().splitlines():
        line = line.strip()
        
        if line.startswith(separator):
            if preamble: preamble = False
            if current_unit:
                # Process the previous unit if it's completed
                full_unit = "\n".join(current_unit).strip()
                if ": " in full_unit: # the format is - atomic unit: atomic unit type
                    unit, label = full_unit.rsplit(": ", 1)
                    parsed_units.append(unit.strip())
                    parsed_labels.append(label.strip())
                else: # the format is just - atomic unit
                    unit, label = full_unit.strip(), "Fact"
                    parsed_units.append(unit.strip())
                    parsed_labels.append(label.strip())
                current_unit = []
            # Add the new line to the current unit (without leading '- ')
            current_unit.append(line[2:].strip())
        else:
            if preamble:
                continue # skip preamble lines that do not start with '-'
            # Continue adding lines to the current unit
            current_unit.append(line.strip())
    
    # Process the last unit
    if current_unit:
        full_unit = "\n".join(current_unit).strip()
        if ": " in full_unit:
            unit, label = full_unit.rsplit(": ", 1)
            parsed_units.append(unit.strip())
            parsed_labels.append(label.strip())
        else:
            unit, label = full_unit.strip(), "Fact"
            parsed_units.append(unit.strip())
            parsed_labels.append(label.strip())
    
    return parsed_units, parsed_labels

def convert_atomic_units_to_dicts_(
    labels: List[str], 
    units: List[str]
) -> List[dict[str, Any]]:

    return [
        {_LABEL: label, _ATOM: atom}
        for label, atom in zip(labels, units)
    ]
 
class AtomExtractor(object):
    """
    Main class for atomic unit decomposition. An atomic unit is either a fact
    or a claim.
    """

    def __init__(
        self, 
        model: str = "llama-3.3-70b-instruct",
        prompt_version: str = "v1",
        llm_inference_provider: str = "rits"
    ):
        """
        Initialize the AtomExtractor.

        Args:
            model: str
                The model id (RITS) used for extraction.
            prompt_version: str
                The prompt version used for the model (v1 - original, v2 - newer)
        """ 
            
        self.model = model
        self.prompt_version = prompt_version

        self.model_info = MODELS[llm_inference_provider][model] # only used to ger prompt begin/end
        self.prompt_begin = self.model_info.get("prompt_begin", DEFAULT_PROMPT_BEGIN)
        self.prompt_end = self.model_info.get("prompt_end", DEFAULT_PROMPT_END)
        
        self.llm_handler = LLMHandler(self.model, llm_inference_provider=llm_inference_provider)
     
        print(f"[AtomExtractor] Using LLM on {llm_inference_provider}: {self.model}")
        print(f"[AtomExtractor] Using prompt version: {self.prompt_version}")

    def make_prompt(self, response: str) -> str:
        if self.prompt_version == "v1":
            prompt = ATOM_EXTRACTION_PROMPT1.format(
                _RESPONSE_PLACEHOLDER=response,
                _PROMPT_BEGIN_PLACEHOLDER=self.prompt_begin,
                _PROMPT_END_PLACEHOLDER=self.prompt_end
            )
        elif self.prompt_version == "v2":
            prompt = ATOM_EXTRACTION_PROMPT2.format(
                _RESPONSE_PLACEHOLDER=response,
                _PROMPT_BEGIN_PLACEHOLDER=self.prompt_begin,
                _PROMPT_END_PLACEHOLDER=self.prompt_end
            )
        else:
            raise ValueError(f"AtomExtractor: Uknown prompt version {self.prompt_version}.")
        
        return prompt

    def get_atoms_from_response(self, response: str, credentials: Dict[str, Any]):
        print(f"[AtomExtractor] Prompt created: 1")
        prompt = self.make_prompt(response)
        response = self.llm_handler.completion(prompt, credentials)
        output = response.choices[0].message.content
        units, labels = text_to_units(output)

        return units, labels

    def get_atoms_from_responses(self, responses: List[str], credentials: Dict[str, Any]):
        results = []
        prompts = [self.make_prompt(response) for response in responses]
        print(f"[AtomExtractor] Prompts created: {len(prompts)}")

        for _, response in tqdm(
            enumerate(
                self.llm_handler.batch_completion(prompts, credentials)
            ),
            total=len(prompts),
            desc="Extractor",
            unit="prompts",
            ):
                results.append(response.choices[0].message.content)

        all_units = []
        all_labels = []
        for result in results:
            units, labels = text_to_units(result)
            all_units.append(units)
            all_labels.append(labels)
        return all_units, all_labels

    def run(self, response: str, credentials: Dict[str, Any]):
        units, labels = self.get_atoms_from_response(response, credentials)
        # print(f"units: {units}, labels: {labels}")
        units_as_dict = convert_atomic_units_to_dicts_(labels, units)
        facts_as_dict = [unit for unit in units_as_dict if unit[_LABEL].lower() in ["fact", "claim"]]
        
        return {
            'num_atoms': len(units),
            'atoms': units,
            'all_atoms': units_as_dict,
            'all_facts': facts_as_dict
        }
    
    def runall(self, responses: List[str], credentials: Dict[str, Any]):

        results = []
        units, labels = self.get_atoms_from_responses(responses, credentials) 
        # print(f"units: {units}, labels: {labels}")
        for i in range(len(responses)):
            units_as_dict = convert_atomic_units_to_dicts_(labels[i], units[i])
            facts_as_dict = [unit for unit in units_as_dict if unit[_LABEL].lower() in ["fact", "claim"]]
            results.append({
                'num_atoms': len(units[i]),
                'atoms': units[i],
                'all_atoms': units_as_dict,
                'all_facts': facts_as_dict
            })

        return results

if __name__ == "__main__":
    model = "gpt-3.5-turbo"
    prompt_version = "v2"
    llm_inference_provider = "openai"
    credentials = {"OPENAI_API_KEY": "sk-proj-.."} 

    extractor = AtomExtractor(model=model, prompt_version=prompt_version, llm_inference_provider=llm_inference_provider)

    response = "The Apollo 14 mission to the Moon took place on January 31, 1971. \
        This mission was significant as it marked the third time humans set \
        foot on the lunar surface, with astronauts Alan Shepard and Edgar \
        Mitchell joining Captain Stuart Roosa, who had previously flown on \
        Apollo 13. The mission lasted for approximately 8 days, during which \
        the crew conducted various experiments and collected samples from the \
        lunar surface. Apollo 14 brought back approximately 70 kilograms of \
        lunar material, including rocks, soil, and core samples, which have \
        been invaluable for scientific research ever since."

    result = extractor.run(response, credentials=credentials)
    num_atoms = result["num_atoms"]
    print(f"Number of atoms: {num_atoms}")
    for i, elem in enumerate(result["all_facts"]):
        label = elem["label"]
        text = elem["atom"]
        print(f"{i}: [{label}] - {text}")

    responses = [
        "Gerhard Fischer is an inventor and entrepreneur who is best known \
        for inventing the first handheld, battery-operated metal detector in 1931. \
        He was born on July 23, 1904, in Frankfurt, Germany, and moved to the \
        United States in 1929, where he became a citizen in 1941.\n\nFischer's metal \
        detector was originally designed to find and remove nails and other metal \
        debris from wood used in construction projects. However, it soon became \
        popular among treasure hunters looking for buried artifacts and coins.\n\nIn addition \
        to his work on metal detectors, Fischer also invented a number of other \
        devices, including a waterproof flashlight and a portable radio receiver. \
        He founded the Fischer Research Laboratory in 1936, which became one of the \
        leading manufacturers of metal detectors in the world.\n\nFischer received \
        numerous awards and honors for his inventions, including the Thomas A. \
        Edison Foundation Gold Medal in 1987. He passed away on February 23, 1995, \
        leaving behind a legacy of innovation and entrepreneurship.",
        
        "Lanny Flaherty is an American actor born on December 18, 1949, in \
        Pensacola, Florida. He has appeared in numerous films, television shows, \
        and theater productions throughout his career, which began in the late 1970s. \
        Some of his notable film credits include \"King of New York,\" \"The Abyss,\" \
        \"Natural Born Killers,\" \"The Game,\" and \"The Straight Story.\" On television, \
        he has appeared in shows such as \"Law & Order,\" \"The Sopranos,\" \"Boardwalk Empire,\" \
        and \"The Leftovers.\" Flaherty has also worked extensively in theater, \
        including productions at the Public Theater and the New York Shakespeare \
        Festival. He is known for his distinctive looks and deep gravelly voice, \
        which have made him a memorable character actor in the industry."
    ]

    results = extractor.runall(responses, credentials=credentials)
    for result in results:
        num_atoms = result["num_atoms"]
        print(f"Number of atoms: {num_atoms}")
        for i, elem in enumerate(result["all_facts"]):
            label = elem["label"]
            text = elem["atom"]
            print(f"{i}: [{label}] - {text}")

    print("Done.")
