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

from typing import Dict, List
from mellea.backends import Backend
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.requirements import check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.core import FancyLogger

# Local imports
from fact_reasoner.utils import validate_json_code_block, strip_code_fences, LOOP_BUDGET

ALLOWED_SCIENCE_ATOM_TYPES = {
    "Result",
    "Data",
    "Observation",
    "Method",
    "Conclusion",
    "Claim",
    "Hypothesis",
    "Opinion",
    "Background Fact",
    "Limitation",
    "Meta Statement",
    "Citation",
    "Reference",
    "Attribution",
    "Instruction",
    "Question",
    "Other",
}

FILTERED_OUT_TYPES = {
    "Meta Statement",
    "Citation",
    "Reference",
    "Attribution",
    "Instruction",
    "Question",
    "Other",
}

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


INSTRUCTION_ATOMIZER_SCIENCE = """
Instructions:
You are given a TEXT extracted from scientific literature (e.g., research papers, abstracts, reports, reviews, technical documentation, or supplementary material).

Your task is to decompose the TEXT into a set of independent, minimal, and scientifically meaningful content units ("atoms").

Each atom must represent a single, self-contained piece of information that can be evaluated independently.

Only extract information explicitly stated in the TEXT. Do NOT infer missing values, causes, mechanisms, or relationships.

Preserve qualifiers exactly as written, including:
- may, might, suggests, likely
- not, no, absence of
- confidence levels (e.g., p-values, confidence intervals)
- comparative or hedging language

Scientific content unit types:

Core Evidence:
- Result: Measured or reported findings (effects, comparisons, statistical outcomes). It may include numbers.
- Data: Raw numerical values, percentages, ranges, quantities, or units without implied interpretation.
- Observation: Descriptive findings without strong interpretation.
- Method: Study design, datasets, instruments, procedures, materials, interventions, experimental setup, or analysis steps.

Interpretation:
- Conclusion: Interpretation of results within the study scope. It is directly supported by results stated in the same text.
- Claim: Generalized or extrapolated statement beyond direct evidence in the same text.
- Hypothesis: Proposed explanation, mechanism, or causal account.
- Opinion: Subjective evaluation, judgment, or preference stated by the authors or quoted in the text, especially when it is normative, evaluative, or not directly testable as stated, such as:
    - "we believe"
    - "the reviewer described"
    - "promising"
    - "elegant"
    - "practical"
    - "intuitive"
    - "important" when clearly evaluative rather than evidential

Context:
- Background Fact: Established knowledge, definitions, or general scientific context without explicit attribution.
- Limitation: Uncertainty, caveat, constraint, bias, weakness, or restricted applicability.
- Meta Statement: Statements about the paper, section, appendix, document structure, or discourse.

Structure & Attribution:
- Citation: Inline references (e.g., [12], (Smith et al., 2020)).
- Reference: Full bibliographic, technical, software, database, or dataset references.
- Attribution: Statements referring to prior work or external findings. It must explicitly reference a source, prior study, report, guideline, or citation.

Other:
- Instruction: Recommendation, procedural guidance, or suggested action.
- Question: Research question or inquiry.
- Other: Content that does not fit the above categories. Use "Other" ONLY if no category clearly applies.

Follow these steps:
1. Decompose the TEXT into minimal units, with one main idea per unit.
2. Preserve numbers, units, variables, qualifiers, uncertainty, and scope exactly as written.
3. Extract citations separately AND keep their associated meaning. A citation must appear as a separate atom immediately after the statement it supports whenever possible.
4. Preserve attribution explicitly.
5. Do not add external knowledge.
6. Avoid redundancy.
7. Use the smallest standalone wording possible while remaining faithful to the TEXT.
8. Each atom must have:
   - "text": the atomic unit text
   - "atom_type": one of the allowed content unit types listed above
9. The output must be a valid JSON dictionary inside a markdown code block.
10. Each atom must have a unique key: "id1", "id2", "id3", ...
11. Do not output any text before or after the JSON code block.

SPLITTING RULE:
- If a sentence contains multiple independent attributes (e.g., time, location, quantity, entity), you MUST split them into separate atoms.

Output format:
```json
{
  "id1": {
    "text": "<first atomic unit>",
    "atom_type": "<first atom type>"
  },
  "id2": {
    "text": "<second atomic unit>",
    "atom_type": "<second atom type>"
  },
  ...
}

Use the provided examples to learn your task.

Example 1:
TEXT:
Inflammation plays a central role in atherosclerosis. Previous studies have shown that elevated CRP levels are associated with increased cardiovascular risk (Ridker et al., 2000). In this randomized controlled trial, 500 patients received Drug D for 24 weeks. CRP levels decreased from 5.0 mg/L to 2.3 mg/L in the treatment group (p < 0.001). No serious adverse events were observed. These findings suggest that Drug D reduces systemic inflammation. However, the study excluded patients over 70 years old. Future trials should include older populations. Can Drug D reduce long-term mortality? Full protocol details are provided in Appendix B. [22]
OUTPUT:
```json
{
  "id1": {
    "text": "Inflammation plays a central role in atherosclerosis.",
    "atom_type": "Background Fact"
  },
  "id2": {
    "text": "Previous studies have shown that elevated CRP levels are associated with increased cardiovascular risk.",
    "atom_type": "Attribution"
  },
  "id3": {
    "text": "(Ridker et al., 2000)",
    "atom_type": "Citation"
  },
  "id4": {
    "text": "The study was a randomized controlled trial.",
    "atom_type": "Method"
  },
  "id5": {
    "text": "The study included 500 patients.",
    "atom_type": "Method"
  },
  "id6": {
    "text": "Patients received Drug D for 24 weeks.",
    "atom_type": "Method"
  },
  "id7": {
    "text": "The baseline CRP level was 5.0 mg/L.",
    "atom_type": "Data"
  },
  "id8": {
    "text": "The CRP level after treatment was 2.3 mg/L.",
    "atom_type": "Data"
  },
  "id9": {
    "text": "CRP levels decreased in the treatment group.",
    "atom_type": "Result"
  },
  "id10": {
    "text": "The reduction was statistically significant (p < 0.001).",
    "atom_type": "Result"
  },
  "id11": {
    "text": "No serious adverse events were observed.",
    "atom_type": "Observation"
  },
  "id12": {
    "text": "These findings suggest that Drug D reduces systemic inflammation.",
    "atom_type": "Conclusion"
  },
  "id13": {
    "text": "The study excluded patients over 70 years old.",
    "atom_type": "Limitation"
  },
  "id14": {
    "text": "Future trials should include older populations.",
    "atom_type": "Instruction"
  },
  "id15": {
    "text": "Can Drug D reduce long-term mortality?",
    "atom_type": "Question"
  },
  "id16": {
    "text": "Full protocol details are provided in Appendix B.",
    "atom_type": "Meta Statement"
  },
  "id17": {
    "text": "[22]",
    "atom_type": "Citation"
  }
}
```

Example 2:
TEXT:
Greenhouse gases contribute to radiative forcing in the Earth's atmosphere. According to the IPCC Sixth Assessment Report (IPCC, 2021), global temperature has increased by approximately 1.1°C since pre-industrial times. We used CMIP6 climate models to simulate warming under different emission scenarios. Simulations including anthropogenic forcing reproduced observed trends, while those excluding it did not. These findings indicate that human activities are the primary driver of warming. However, regional projections remain uncertain. Full report: IPCC (2021) Climate Change 2021: The Physical Science Basis.
OUTPUT:
```json
{
  "id1": {
    "text": "Greenhouse gases contribute to radiative forcing in the Earth's atmosphere.",
    "atom_type": "Background Fact"
  },
  "id2": {
    "text": "According to the IPCC Sixth Assessment Report, global temperature has increased by approximately 1.1°C since pre-industrial times.",
    "atom_type": "Attribution"
  },
  "id3": {
    "text": "(IPCC, 2021)",
    "atom_type": "Citation"
  },
  "id4": {
    "text": "CMIP6 climate models were used.",
    "atom_type": "Method"
  },
  "id5": {
    "text": "Warming was simulated under different emission scenarios.",
    "atom_type": "Method"
  },
  "id6": {
    "text": "Simulations including anthropogenic forcing reproduced observed trends.",
    "atom_type": "Result"
  },
  "id7": {
    "text": "Simulations excluding anthropogenic forcing did not reproduce observed trends.",
    "atom_type": "Result"
  },
  "id8": {
    "text": "These findings indicate that human activities are the primary driver of warming.",
    "atom_type": "Conclusion"
  },
  "id9": {
    "text": "Regional projections remain uncertain.",
    "atom_type": "Limitation"
  },
  "id10": {
    "text": "IPCC (2021) Climate Change 2021: The Physical Science Basis.",
    "atom_type": "Reference"
  }
}
```

Example 3:
TEXT:
Remote sensing enables large-scale environmental monitoring. Using Sentinel-2 imagery, vegetation cover was analyzed across the region from 2015 to 2022. NDVI increased by 18% over the study period. Ground validation showed an accuracy of 91%. The results suggest that vegetation productivity has increased. The authors claim that this method can be applied globally. However, cloud contamination may affect optical data quality.
OUTPUT:
```json
{
  "id1": {
    "text": "Remote sensing enables large-scale environmental monitoring.",
    "atom_type": "Background Fact"
  },
  "id2": {
    "text": "Sentinel-2 imagery was used.",
    "atom_type": "Method"
  },
  "id3": {
    "text": "Vegetation cover was analyzed across the region from 2015 to 2022.",
    "atom_type": "Method"
  },
  "id4": {
    "text": "NDVI increased by 18% over the study period.",
    "atom_type": "Result"
  },
  "id5": {
    "text": "18% is the magnitude of NDVI increase.",
    "atom_type": "Data"
  },
  "id6": {
    "text": "Ground validation was performed.",
    "atom_type": "Method"
  },
  "id7": {
    "text": "The accuracy was 91%.",
    "atom_type": "Data"
  },
  "id8": {
    "text": "Ground validation showed an accuracy of 91%.",
    "atom_type": "Result"
  },
  "id9": {
    "text": "The results suggest that vegetation productivity has increased.",
    "atom_type": "Conclusion"
  },
  "id10": {
    "text": "The authors claim that this method can be applied globally.",
    "atom_type": "Claim"
  },
  "id11": {
    "text": "Cloud contamination may affect optical data quality.",
    "atom_type": "Limitation"
  }
}
```

Example 4:
TEXT:
Students improved their scores from 60% to 78% after the intervention, corresponding to a relative increase of 30%. Classroom engagement was observed to increase during sessions. The authors hypothesize that active learning contributed to this improvement. This study demonstrates the effectiveness of interactive teaching methods. Section 4.2 describes the scoring methodology.
OUTPUT:
```json
{
  "id1": {
    "text": "The baseline score was 60%.",
    "atom_type": "Data"
  },
  "id2": {
    "text": "The post-intervention score was 78%.",
    "atom_type": "Data"
  },
  "id3": {
    "text": "Scores increased after the intervention.",
    "atom_type": "Result"
  },
  "id4": {
    "text": "The relative increase was 30%.",
    "atom_type": "Result"
  },
  "id5": {
    "text": "Classroom engagement was observed to increase during sessions.",
    "atom_type": "Observation"
  },
  "id6": {
    "text": "The authors hypothesize that active learning contributed to this improvement.",
    "atom_type": "Hypothesis"
  },
  "id7": {
    "text": "This study demonstrates the effectiveness of interactive teaching methods.",
    "atom_type": "Conclusion"
  },
  "id8": {
    "text": "Section 4.2 describes the scoring methodology.",
    "atom_type": "Other"
  }
}
```

Example 5:
TEXT:
Previous work suggests that methane emissions can be detected using hyperspectral imaging (Jacob et al., 2016). We processed satellite data using a convolutional neural network. Emission detection accuracy reached 87%. These findings indicate that machine learning improves detection performance. Users should calibrate sensors before deployment. For details, see Jacob et al. (2016), Atmospheric Chemistry and Physics.
OUTPUT:
```json
{
  "id1": {
    "text": "Previous work suggests that methane emissions can be detected using hyperspectral imaging.",
    "atom_type": "Attribution"
  },
  "id2": {
    "text": "(Jacob et al., 2016)",
    "atom_type": "Citation"
  },
  "id3": {
    "text": "Satellite data was processed using a convolutional neural network.",
    "atom_type": "Method"
  },
  "id4": {
    "text": "Emission detection accuracy reached 87%.",
    "atom_type": "Result"
  },
  "id5": {
    "text": "87% is the detection accuracy.",
    "atom_type": "Data"
  },
  "id6": {
    "text": "These findings indicate that machine learning improves detection performance.",
    "atom_type": "Conclusion"
  },
  "id7": {
    "text": "Users should calibrate sensors before deployment.",
    "atom_type": "Instruction"
  },
  "id8": {
    "text": "Jacob et al. (2016), Atmospheric Chemistry and Physics.",
    "atom_type": "Reference"
  }
}
```

Example 6:
TEXT:
The reviewer described the visualization framework as elegant and highly practical. The framework reduced annotation time by 24% in our benchmark experiments. The authors believe the interface is more intuitive than prior tools. However, usability was not formally measured.
OUTPUT:
```json
{
  "id1": {
    "text": "The reviewer described the visualization framework as elegant and highly practical.",
    "atom_type": "Opinion"
  },
  "id2": {
    "text": "The framework reduced annotation time by 24% in our benchmark experiments.",
    "atom_type": "Result"
  },
  "id3": {
    "text": "The authors believe the interface is more intuitive than prior tools.",
    "atom_type": "Opinion"
  },
  "id4": {
    "text": "Usability was not formally measured.",
    "atom_type": "Limitation"
  }
}
```

Example 7:
TEXT:
Marie Curie conducted research on radioactivity in Paris between 1898 and 1902 using 2 laboratories.
OUTPUT:
```json
{
  "id1": {
    "text": "Marie Curie conducted research on radioactivity.",
    "atom_type": "Background Fact"
  },
  "id2": {
    "text": "The research took place in Paris.",
    "atom_type": "Data"
  },
  "id3": {
    "text": "The research started in 1898.",
    "atom_type": "Data"
  },
  "id4": {
    "text": "The research ended in 1902.",
    "atom_type": "Data"
  },
  "id5": {
    "text": "The number of laboratories used was 2.",
    "atom_type": "Data"
  }
}
```

Your task:
TEXT: {{response}}
OUTPUT:
"""

def validate_science_atom_json_code_block(s: str) -> bool:
    """
    Validate that the model output is a fenced JSON code block with schema:

    {
        "id1": {
            "text": "...",
            "atom_type": "Result"
        },
        ...
    }
    """

    if not validate_json_code_block(s):
        return False

    try:
        cleaned = strip_code_fences(s)
        data = json.loads(cleaned)
    except Exception:
        return False

    if not isinstance(data, dict):
        return False

    if len(data) == 0:
        return True

    for key, value in data.items():
        if not isinstance(key, str):
            return False

        if not key.startswith("id"):
            return False

        if not isinstance(value, dict):
            return False

        required_keys = {"text", "atom_type"}
        if set(value.keys()) != required_keys:
            return False

        text = value.get("text")
        atom_type = value.get("atom_type")

        if not isinstance(text, str) or not text.strip():
            return False

        if not isinstance(atom_type, str) or not atom_type.strip():
            return False

        if atom_type not in ALLOWED_SCIENCE_ATOM_TYPES:
            return False

    return True


def filter_atoms(
    atoms: Dict[str, Dict[str, str]]
) -> Dict[str, Dict[str, str]]:
    """
    Remove unwanted atom types and reindex ids.
    """

    filtered = [
        atom
        for atom in atoms.values()
        if atom.get("atom_type") not in FILTERED_OUT_TYPES
    ]

    # Re-index to id1, id2, ...
    return {
        f"id{i+1}": atom
        for i, atom in enumerate(filtered)
    }


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

        # Disable Mellea logging
        FancyLogger.get_logger().setLevel(FancyLogger.ERROR)

    def run(self, 
        response: str,
        science_mode: bool
    ) -> Dict[str, Dict[str, str]]:
        """
        Extract atomic units from a single response.
        
        Args:
            response: str
                The response from which to extract atomic units.
            science_mode: bool
                The scientific mode for the prompts.
        Returns:
            Dict[str, Dict[str, str]]: A dictionary containing the atomic units, each with
            a unique identifier, text and atom type.
        """
        
        prompt = INSTRUCTION_ATOMIZER_SCIENCE if science_mode else INSTRUCTION_ATOMIZER 
        validator = (
            validate_science_atom_json_code_block
            if science_mode
            else validate_json_code_block
        )

        # Perform the instruction with validation
        output = mfuncs.instruct(
            prompt,
            context=SimpleContext(),
            backend=self.backend,
            requirements=[
                check(
                    "The output must be a valid JSON dictionary with markdown code fences",
                    validation_fn=simple_validate(validator),
                )
            ],
            user_variables={"response": response},
            strategy=RejectionSamplingStrategy(loop_budget=LOOP_BUDGET),
            return_sampling_results=True,
        )

        # The output is a validated JSON string; parse it
        if output.success:
            cleaned = strip_code_fences(str(output))
            parsed = json.loads(cleaned)
            if science_mode:
                parsed = filter_atoms(parsed)
            return parsed
        else:
            return {} # empty dict on failure
        
    async def run_batch(
        self, 
        responses: List[str],
        science_mode: bool = False
    ) -> Dict[str, Dict[str, str]]:
        """
        Extract atomic units from a list of responses.
        
        Args:
            responses: List[str]
                The list of response from which to extract atomic units.
            science_mode: bool
                The scientific mode for the prompts.
        Returns:
            dict: A dictionary containing the number of atomic units, the units themselves,
            all atomic units as dictionaries, and all atoms and atom types as dictionaries.
        """

        prompt = INSTRUCTION_ATOMIZER_SCIENCE if science_mode else INSTRUCTION_ATOMIZER
        validator = (
            validate_science_atom_json_code_block
            if science_mode
            else validate_json_code_block
        )
        
        # Perform the instruction with validation
        corutines = []
        for response in responses:
            corutine = mfuncs.ainstruct(
                prompt,
                context=SimpleContext(),
                backend=self.backend,
                requirements=[
                    check(
                        "The output must be a valid JSON dictionary with markdown code fences",
                        validation_fn=simple_validate(validator),
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
                parsed = json.loads(cleaned)
                if science_mode:
                    parsed = filter_atoms(parsed)
                results.append(parsed)
            else:
                results.append({}) # empty dict on failure

        return results


    def __str__(self) -> str:
        return "This is the atomizer"    
