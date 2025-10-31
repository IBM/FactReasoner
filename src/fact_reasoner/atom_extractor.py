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

# Split the input text into atomic facts/claims (LLM based).

from typing import Any, List
from tqdm import tqdm

# if not __package__:
#     # Make CLI runnable from source tree with
#     #    python src/package
#     package_source_path = os.path.dirname(os.path.dirname(__file__))
#     sys.path.insert(0, package_source_path)

# Local imports
from src.fact_reasoner.prompts import ATOM_EXTRACTION_PROMPT_V1, ATOM_EXTRACTION_PROMPT_V2
from src.fact_reasoner.llm_handler import LLMHandler

_ATOM = 'atom'
_LABEL = 'label'

def text_to_units(text: str, separator: str = '- ') -> List[str]:
    """
    Parse the input text into atomic units and their labels.

    Args:
        text: str
            The input text containing atomic units.
        separator: str
            The separator used to identify the start of each atomic unit.

    Returns:
        List[str]: A list of atomic units.
    """

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

def convert_atomic_units_to_dicts_(labels: List[str], units: List[str]) -> List[dict[str, Any]]:
    """
    Convert atomic units and their labels into a list of dictionaries.
    
    Args:
        labels: List[str]
            A list of labels for the atomic units.
        units: List[str]
            A list of atomic units (facts or claims).

    Returns:
        List[dict[str, Any]]: A list of dictionaries where each dictionary contains
        a label and its corresponding atomic unit.
    """
    return [
        {_LABEL: label, _ATOM: atom}
        for label, atom in zip(labels, units)
    ]
 
class AtomExtractor(object):
    """
    Main class for atomic unit decomposition (i.e., atom extraction). 
    An atomic unit is either a fact or a claim.
    """

    def __init__(
        self, 
        model_id: str = "llama-3.1-70b-instruct",
        prompt_version: str = "v1",
        backend: str = "rits"
    ):
        """
        Initialize the AtomExtractor.

        Args:
            model_id: str
                The model id used for extraction.
            prompt_version: str
                The prompt version used for the model (v1 - original, v2 - newer)
            backend: str
                The model's backend (rits, hf or wx).
        """ 
        
        # Initialize the extractor
        self.model_id = model_id
        self.backend = backend
        self.prompt_version = prompt_version
        self.llm_handler = LLMHandler(self.model_id, backend=backend)

        # Set the prompt begin and end templates
        self.prompt_begin = self.llm_handler.get_prompt_begin()
        self.prompt_end = self.llm_handler.get_prompt_end()
        
        print(f"[AtomExtractor] Using LLM on {self.backend}: {self.model_id}")
        print(f"[AtomExtractor] Using prompt version: {self.prompt_version}")

    def make_prompt(self, response: str) -> str:
        """
        Create the prompt for atom extraction based on the response.
        
        Args:
            response: str
                The response from which to extract atomic units.
        
        Returns:
            str: The formatted prompt for the LLM.
        """

        if self.prompt_version == "v1":
            prompt = ATOM_EXTRACTION_PROMPT_V1.format(
                _RESPONSE_PLACEHOLDER=response,
                _PROMPT_BEGIN_PLACEHOLDER=self.prompt_begin,
                _PROMPT_END_PLACEHOLDER=self.prompt_end
            )
        elif self.prompt_version == "v2":
            prompt = ATOM_EXTRACTION_PROMPT_V2.format(
                _RESPONSE_PLACEHOLDER=response,
                _PROMPT_BEGIN_PLACEHOLDER=self.prompt_begin,
                _PROMPT_END_PLACEHOLDER=self.prompt_end
            )
        else:
            raise ValueError(f"Unknown prompt version: {self.prompt_version}. "
                    f"Supported versions are: 'v1', 'v2'.")
        
        return prompt

    def get_atoms_from_response(self, response: str):
        """
        Extract atomic units from a single response.
        Args:
            response: str
                The response from which to extract atomic units.

        Returns:
            List[str]: A list of atomic units (facts or claims).
            List[str]: A list of labels corresponding to the atomic units.
        """

        print(f"[AtomExtractor] Prompts created: 1")
        prompt = self.make_prompt(response)
        response = self.llm_handler.completion(prompt)
        output = response.choices[0].message.content
        units, labels = text_to_units(output)

        return units, labels

    def get_atoms_from_responses(self, responses: List[str]):
        """
        Extract atomic units from multiple responses.
        Args:
            responses: List[str]
                A list of responses from which to extract atomic units.
        
        Returns:
            List[List[str]]: A list of lists, where each inner list contains atomic units (facts or claims).
            List[List[str]]: A list of lists, where each inner list contains labels corresponding to the atomic units.
        """

        results = []
        prompts = [self.make_prompt(response) for response in responses]
        print(f"[AtomExtractor] Prompts created: {len(prompts)}")

        for _, response in tqdm(
            enumerate(
                self.llm_handler.batch_completion(prompts)
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

    def run(self, response: str) -> dict[str, Any]:
        """ Extract atomic units from a single response.
        Args:
            response: str
                The response from which to extract atomic units.
        Returns:
            dict: A dictionary containing the number of atomic units, the units themselves,
            all atomic units as dictionaries, and all facts as dictionaries.
        """

        units, labels = self.get_atoms_from_response(response)
        # print(f"units: {units}, labels: {labels}")
        units_as_dict = convert_atomic_units_to_dicts_(labels, units)
        facts_as_dict = [unit for unit in units_as_dict if unit[_LABEL].lower() in ["fact", "claim", "data format"]]
        
        return {
            'num_atoms': len(units),
            'atoms': units,
            'all_atoms': units_as_dict,
            'all_facts': facts_as_dict
        }
    
    def runall(self, responses: List[str]) -> List[dict[str, Any]]:
        """ Extract atomic units from multiple responses.
        Args:
            responses: List[str]
                A list of responses from which to extract atomic units.

        Returns:
            List[dict]: A list of dictionaries, where each dictionary contains the number of atomic units,
            the units themselves, all atomic units as dictionaries, and all facts as dictionaries."""

        results = []
        units, labels = self.get_atoms_from_responses(responses)
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

    model_id = "llama-3.3-70b-instruct"
    prompt_version = "v2"
    backend = "rits"

    extractor = AtomExtractor(model_id=model_id, prompt_version=prompt_version, backend=backend)

    response = "The Apollo 14 mission to the Moon took place on January 31, 1971. \
        This mission was significant as it marked the third time humans set \
        foot on the lunar surface, with astronauts Alan Shepard and Edgar \
        Mitchell joining Captain Stuart Roosa, who had previously flown on \
        Apollo 13. The mission lasted for approximately 8 days, during which \
        the crew conducted various experiments and collected samples from the \
        lunar surface. Apollo 14 brought back approximately 70 kilograms of \
        lunar material, including rocks, soil, and core samples, which have \
        been invaluable for scientific research ever since."
      
    result = extractor.run(response)
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

    results = extractor.runall(responses)
    for result in results:
        num_atoms = result["num_atoms"]
        print(f"Number of atoms: {num_atoms}")
        for i, elem in enumerate(result["all_facts"]):
            label = elem["label"]
            text = elem["atom"]
            print(f"{i}: [{label}] - {text}")


    print("Done.")
