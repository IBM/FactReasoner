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

# Atomic fact decontextualization using LLMs

import json
import asyncio
import mellea.stdlib.functional as mfuncs

from typing import Any, Dict, List
from mellea.backends import Backend
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.requirements import check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.core import FancyLogger

# Local imports
from fact_reasoner.utils import validate_json_code_block, strip_code_fences, LOOP_BUDGET

INSTRUCTION_REVISER = """
Instructions:
You task is to decontextualize a UNIT to make it standalone. Each UNIT is an independent content unit or atomic unit extracted from the broader context of a RESPONSE.   

Vague References:
- Pronouns (e.g., "he", "she", "they", "it")
- Demonstrative pronouns (e.g., "this", "that", "these", "those")
- Unknown entities (e.g., "the event", "the research", "the invention")
- Incomplete names (e.g., "Jeff..." or "Bezos..." when referring to Jeff Bezos)

Follow the steps below for unit decontextualization:
1. If the UNIT contains vague references, minimally revise them with respect to the specific subjects they refer to in the RESPONSE.
2. The decontextualized UNIT should be minimally revised by ONLY resolving vague references. No additional information must be added.
3. UNIT extraction might decompose a conjunctive statement into multiple units (e.g. Democracy treats citizens as equals regardless of their race or religion -> (1) Democracy treats citizens as equals regardless of their race, (2) Democracy treats citizens as equals regardless of their religion). Avoid adding what is potentially part of another UNIT.
4. Provide a reasoning of the revisions you made to the UNIT, justifying each decision.
5. The output must be in the following JSON format with a markdown code block:

```json
{
  "revised_unit": "<REVISED_UNIT>",
  "rationale": "<YOUR_REASONING>"
}
```
Where <REVISED_UNIT> is the decontextualized UNIT after resolving vague references, and <YOUR_REASONING> is your reasoning for the revisions made.

Use the provided examples to learn your task.

Example 1: 
UNIT: 
Acorns is a financial technology company

RESPONSE:
Acorns is a financial technology company founded in 2012 by Walter Cruttenden, \
Jeff Cruttenden, and Mark Dru that provides micro-investing services. The \
company is headquartered in Irvine, California.

OUTPUT:
```json
{
    "revised_unit": "Acorns is a financial technology company.",
    "rationale": "This UNIT does not contain any vague references. Thus, the unit does not require any further decontextualization."
}
```

Example 2: 
UNIT:
The victim had previously suffered a broken wrist.

RESPONSE:
The clip shows the victim, with his arm in a cast, being dragged to the floor \
by his neck as his attacker says "I'll drown you" on a school playing field, while forcing water from a bottle into the victim's mouth, \
simulating waterboarding. The video was filmed in a lunch break. The clip shows the victim walking away, without reacting, as the attacker \
and others can be heard continuing to verbally abuse him. The victim, a Syrian refugee, had previously suffered a broken wrist; this had also been \
investigated by the police, who had interviewed three youths but took no further action.

OUTPUT:
```json
{
    "revised_unit": "The Syrian refugee victim had previously suffered a broken wrist.",
    "rationale": "The UNIT contains a vague reference, 'the victim.' This is a reference to an unknown entity, since it is unclear who the victim is. From the RESPONSE, we can see that the victim is a Syrian refugee. Thus, the vague reference 'the victim' should be replaced with 'the Syrian refugee victim.'"
}
```

Example 3:
UNIT:
The difference is relatively small.

RESPONSE:
Both the RTX 3060 Ti and RTX 3060 are powerful GPUs, and the difference between them lies in their performance. \
The RTX 3060 Ti has more CUDA cores (4864 vs 3584) but a lower boost clock speed (1665 MHz vs 1777 MHz) compared to the RTX 3060. \
In terms of memory bandwidth, the RTX 3060 Ti has a slight edge over the RTX 3060 with a bandwidth of 448 GB/s compared to 360 GB/s. \
However, the difference is relatively small and may not be noticeable in real-world applications.

OUTPUT:
```json
{
    "revised_unit": "The difference in memory bandwidth between the RTX 3060 Ti and RTX 3060 is relatively small.",
    "rationale": "The UNIT contains a vague reference, 'The difference.' From the RESPONSE, we can see that the difference is in memory bandwidth between the RTX 3060 Ti and RTX 3060. Thus, the vague reference 'The difference' should be replaced with 'The difference in memory bandwidth between the RTX 3060 Ti and RTX 3060'. The sentence from which the UNIT is extracted includes coordinating conjunctions that potentially decompose the statement into multiple units. Thus, adding more context to the UNIT is not necessary."
}
```

Your task:
UNIT:
{{atomic_unit}}

RESPONSE:
{{response}}

OUTPUT:
"""


INSTRUCTION_REVISER_SCIENCE = """
Instructions:
Your task is to decontextualize a UNIT to make it standalone.
Each UNIT is an independent content unit extracted from the broader context of a RESPONSE drawn from scientific literature (e.g., research papers, reports, or technical documents).

Vague References:
- Pronouns (e.g., "it", "they", "this", "these", "those")
- Demonstrative phrases (e.g., "this method", "this model", "these results")
- Unknown entities (e.g., "the study", "the dataset", "the experiment", "the variable")
- Missing quantities or variables (e.g., "increased by 12%" without specifying what increased)
- Missing comparison targets (e.g., "higher than baseline")
- Missing experimental scope (e.g., region, time period, population, conditions)
- Missing attribution (e.g., "previous studies show...")
- Citations referring to unclear or implicit statements

Follow the steps below for unit decontextualization:
1. If the UNIT contains vague references, minimally revise them with respect to the specific subjects they refer to in the RESPONSE.
2. The decontextualized UNIT should be minimally revised by ONLY resolving vague references. No additional information must be added.
3. UNIT extraction might decompose a conjunctive statement into multiple units (e.g. Democracy treats citizens as equals regardless of their race or religion -> (1) Democracy treats citizens as equals regardless of their race, (2) Democracy treats citizens as equals regardless of their religion). Avoid adding what is potentially part of another UNIT.
4. Preserve all numerical values, units, variables, qualifiers, and scientific precision exactly as written.
5. Preserve necessary scope (e.g., dataset, region, time period, experimental conditions) only when required to resolve ambiguity.
6. Do not mention "can be rephrased to" or describe the decontextualization process in the revised unit.
7. Keep citations intact and ensure they refer to explicit content in the revised UNIT.
8. Provide a reasoning of the revisions you made to the UNIT, justifying each decision.

The output must be in the following JSON format with a markdown code block:

```json
{
  "revised_unit": "<REVISED_UNIT>",
  "rationale": "<YOUR_REASONING>"
}
```
Where:
<REVISED_UNIT> is the decontextualized UNIT after resolving vague references.
<YOUR_REASONING> explains whether any vague references were found and justifies each revision.
Use the provided examples to learn your task.

Example 1:
UNIT:
NDVI values increased by 12%.

RESPONSE:
NDVI values were measured using Sentinel-2 imagery across the region between 2018 and 2022. NDVI values increased by 12% over this period.

OUTPUT:
```json
{
  "revised_unit": "NDVI values increased by 12%.",
  "rationale": "This UNIT does not contain any vague references. Thus, the unit does not require any further decontextualization."
}
```

Example 2:
UNIT:
It increased by 12%.

RESPONSE:
NDVI values were measured using Sentinel-2 imagery across the region between 2018 and 2022. NDVI increased by 12% over this period.

OUTPUT:
```json
{
  "revised_unit": "NDVI values increased by 12%.",
  "rationale": "The UNIT contains a vague reference, 'It.' This is a reference to an unknown variable, since it is unclear what increased. From the RESPONSE, we can see that 'It' refers to NDVI values. Thus, the vague reference 'It' should be replaced with 'NDVI values.'"
}
```

Example 3:
UNIT:
The model performed better under these conditions compared to baseline.

RESPONSE:
We evaluated a convolutional neural network under low-light conditions and high-noise environments. The model performed better under these conditions compared to baseline.

OUTPUT:
OUTPUT:
```json
{
  "revised_unit": "The convolutional neural network performed better under low-light conditions and high-noise environments compared to baseline.",
  "rationale": "The UNIT contains vague references, 'The model' and 'these conditions.' From the RESPONSE, we can see that 'The model' refers to 'a convolutional neural network' and 'these conditions' refers to 'low-light conditions and high-noise environments.' Thus, both vague references should be replaced accordingly."
}
```

Example 4:
UNIT:
The dataset contains 10,000 samples.

RESPONSE:
We used a satellite imagery dataset collected over Europe between 2015 and 2020. The dataset contains 10,000 samples.

OUTPUT:
```json
{
  "revised_unit": "The satellite imagery dataset collected over Europe between 2015 and 2020 contains 10,000 samples.",
  "rationale": "The UNIT contains a vague reference, 'The dataset.' This is a reference to an unspecified dataset. From the RESPONSE, we can see that the dataset is 'a satellite imagery dataset collected over Europe between 2015 and 2020.' Thus, the vague reference should be replaced accordingly."
}
```

Example 5:
UNIT:
Previous studies have shown that this increases temperature (Smith et al., 2020).

RESPONSE:
Urban heat islands increase local temperatures. Previous studies have shown that this increases temperature (Smith et al., 2020).

OUTPUT:
```json
{
  "revised_unit": "Previous studies have shown that urban heat islands increase temperature (Smith et al., 2020).",
  "rationale": "The UNIT contains a vague reference, 'this.' From the RESPONSE, we can see that 'this' refers to 'urban heat islands.' Thus, the vague reference 'this' should be replaced while preserving the citation."
}
```

Example 6:
UNIT:
No significant difference was observed.

RESPONSE:
We compared accuracy between Model A and Model B. No significant difference in accuracy was observed between the two models.

OUTPUT:
```json
{
  "revised_unit": "No significant difference in accuracy was observed between Model A and Model B.",
  "rationale": "The UNIT contains missing context about what was compared and which variable was measured. From the RESPONSE, we can see that the comparison is between Model A and Model B and refers to accuracy. Thus, the UNIT should be revised accordingly."
}
```

Your task:
UNIT:
{{atomic_unit}}

RESPONSE:
{{response}}

OUTPUT:
"""

class Reviser:
    """
    Atomic unit decontextualization using LLMs.
    
    """

    def __init__(
            self,
            backend: Backend,
    ):
        """
        Initialize the Reviser.

        Args:
            backend: Backend
                The Mellea backend to use for LLM interactions.
        """ 
        
        # Safety checks        
        if backend is None:
            raise ValueError("Mellea session is None. Please provide a valid Mellea session.")

        # Initialize the reviser
        self.backend = backend
        
        # Print backend info
        print(f"[Reviser] Using Mellea backend: {self.backend.model_id}")

        # Disable Mellea logging
        FancyLogger.get_logger().setLevel(FancyLogger.ERROR)


    def run(
        self, 
        units: List[str], 
        response: str,
        science_mode: bool
    ) -> List[Dict[str, Any]]:
        """
        Decontextualize the input atomic units using the response as context.
        
        Args:
            units: List[str]
                The atomic units to be decontextualized.
            response: str
                The response from which the atomic unit is decontextualized.
            science_mode: bool
                The science mode in the prompt.
        Returns:
            List[str]: A dictionary containing the revised atomic unit.
        """

        prompt = INSTRUCTION_REVISER_SCIENCE if science_mode else INSTRUCTION_REVISER

        # Perform the instruction with validation
        results = []
        for atom_text in units:
            output = mfuncs.instruct(
                prompt,
                context=SimpleContext(),
                backend=self.backend,
                requirements=[
                    check(
                        "The output must be a valid JSON code block.",
                        validation_fn=simple_validate(
                            lambda s: validate_json_code_block(s, required_keys=["revised_unit", "rationale"])
                        )
                    )
                ],
                user_variables={"atomic_unit": atom_text, "response": response},
                strategy=RejectionSamplingStrategy(loop_budget=LOOP_BUDGET),
                return_sampling_results=True
            )

            if output.success:
                cleaned = strip_code_fences(str(output))
                revised_unit = json.loads(cleaned)
                revised_unit.update({"text": atom_text})
                results.append(revised_unit)
        
        return results

    async def run_batch(
        self, 
        units: List[str], 
        response: str,
        science_mode: bool
    ) -> List[Dict[str, Any]]:
        """
        Decontextualize the input atomic units using the response as context.
        
        Args:
            units: List[str]
                The atomic units to be decontextualized.
            response: str
                The response from which the atomic unit is decontextualized.
            science_mode: bool
                The science mode in the prompt.
        Returns:
            List[str]: A dictionary containing the revised atomic unit.
        """

        # Perform the instruction with validation
        
        corutines = []
        for atom_text in units:
            corutine = mfuncs.ainstruct(
                INSTRUCTION_REVISER,
                context=SimpleContext(),
                backend=self.backend,
                requirements=[
                    check(
                        "The output must be a valid JSON code block.",
                        validation_fn=simple_validate(
                            lambda s: validate_json_code_block(s, required_keys=["revised_unit", "rationale"])
                        )
                    )
                ],
                user_variables={"atomic_unit": atom_text, "response": response},
                strategy=RejectionSamplingStrategy(loop_budget=LOOP_BUDGET),
                return_sampling_results=True
            )
            corutines.append(corutine)

        results = []
        print(f"[Reviser] Awaiting for async execution ...")
        outputs = await asyncio.gather(*(corutines[i] for i in range(len(corutines))))

        # for output in outputs:
        #     if output.success:
        #         cleaned = strip_code_fences(str(output))
        #         revised_unit = json.loads(cleaned)
        #         revised_unit.update({"text": atom_text})
        #         results.append(revised_unit)

        for atom_text, output in zip(units, outputs):
            if output.success:
                cleaned = strip_code_fences(str(output))
                revised_unit = json.loads(cleaned)
                revised_unit.update({"text": atom_text})
                results.append(revised_unit)
        
        return results
    
