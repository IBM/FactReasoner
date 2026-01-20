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

from typing import Any, Dict
from mellea.backends import Backend
from mellea.backends.types import ModelOption
from mellea.stdlib.base import SimpleContext, CBlock
from mellea.stdlib.requirement import req, check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

# Local imports
from src.fact_reasoner.utils import validate_json_code_block, strip_code_fences

INSTRUCTION = """
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

        # In-context learning examples
        self.icl_examples = [
            """Example 1: 
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
            ```""",
            """Example 2: 
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
            ```""",
            """Example 3:
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
            ```"""
        ]

    def run(self, units: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Decontextualize the input atomic units using the response as context.
        
        Args:
            units: Dict[str, Any]
                The atomic units to be decontextualized.
            response: str
                The response from which the atomic unit is decontextualized.
        Returns:
            Dict[str, Any]: A dictionary containing the revised atomic unit.
        """

        # Safety checks
        assert "atomic_units" in units, "Input units must contain 'atomic_units' key."

        # Perform the instruction with validation
        results = {"revised_units": []}
        for unit in units["atomic_units"]:
            atom_id = unit["id"]
            atom_text = unit["text"]
            output = mfuncs.instruct(
                INSTRUCTION,
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
                user_variables={"unit": atom_text, "response": response},
                icl_examples=self.icl_examples,
                strategy=RejectionSamplingStrategy(loop_budget=3),
                return_sampling_results=True
            )

            if output.success:
                cleaned = strip_code_fences(str(output))
                revised_unit = json.loads(cleaned)
                revised_unit.update({"id": atom_id, "text": atom_text})
                results["revised_units"].append(revised_unit)
        
        return results


    async def arun(self, units: Dict[str, Any], response: str) -> Dict[str, Any]:
        """
        Decontextualize the input atomic units using the response as context.
        
        Args:
            units: Dict[str, Any]
                The atomic units to be decontextualized.
            response: str
                The response from which the atomic unit is decontextualized.
        Returns:
            Dict[str, Any]: A dictionary containing the revised atomic unit.
        """

        # Safety checks
        assert "atomic_units" in units, "Input units must contain 'atomic_units' key."

        # Perform the instruction with validation
        results = {"revised_units": []}
        for unit in units["atomic_units"]:
            atom_id = unit["id"]
            atom_text = unit["text"]
            output = await mfuncs.ainstruct(
                INSTRUCTION,
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
                user_variables={"unit": atom_text, "response": response},
                icl_examples=self.icl_examples,
                strategy=RejectionSamplingStrategy(loop_budget=3),
                return_sampling_results=True
            )

            if output.success:
                cleaned = strip_code_fences(str(output))
                revised_unit = json.loads(cleaned)
                revised_unit.update({"id": atom_id, "text": atom_text})
                results["revised_units"].append(revised_unit)
        
        return results
    
if __name__ == "__main__":
    
    use_async = False

    # Create a Mellea RITS backend
    from mellea_ibm.rits import RITSBackend, RITS
    backend = RITSBackend(
        RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 500}
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

    atoms = {
        "atomic_units": [
            {"id": 1, "text": "He has appeared in numerous films."},
            {"id": 2, "text": "He has appeared in numerous television shows."},
            {"id": 3, "text": "He has appeared in numerous theater productions."},
            {"id": 4, "text": "His career began in the late 1970s."}
        ]
    }

    # Process the atoms (sync or async)
    if not use_async:
        result = reviser.run(atoms, response)
        print(f"Reviser result: {result}")
    else:
        result = asyncio.run(reviser.arun(atoms, response))
        print(f"Reviser result: {result}")

    # Print the revised atomic units
    atoms = result.get("revised_units", [])
    print(f"Number of revised atomic units: {len(atoms)}")
    for atom in atoms:
        print(f"Original Atom {atom['id']}: {atom['text']}")
        print(f"Revised Atom {atom['id']}: {atom['revised_unit']}")
        print(f"Rationale: {atom['rationale']}")
        print("-----")

    print("Done.")
