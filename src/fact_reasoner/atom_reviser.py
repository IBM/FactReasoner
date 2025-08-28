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

import string

from typing import List
from tqdm import tqdm

from src.fact_reasoner.utils import (
    RITS_MODELS, 
    DEFAULT_PROMPT_BEGIN, 
    DEFAULT_PROMPT_END, 
    strip_string, 
    extract_first_code_block, 
    extract_last_wrapped_response
)
from src.fact_reasoner.llm_handler import LLMHandler

_SYMBOL = 'Foo'
_NOT_SYMBOL = 'Not Foo'


# v1
ATOM_DECONTEXT_PROMPT1 = """{_PROMPT_BEGIN_PLACEHOLDER}

You task is to decontextualize a UNIT to make it standalone. \
Each UNIT is an independent content unit extracted from the broader context of a RESPONSE.

Vague References:
- Pronouns (e.g., "he", "she", "they", "it")
- Demonstrative pronouns (e.g., "this", "that", "these", "those")
- Unknown entities (e.g., "the event", "the research", "the invention")
- Incomplete names (e.g., "Jeff..." or "Bezos..." when referring to Jeff Bezos)

Instructions: 
Follow the steps below for unit decontextualization:
1. If the UNIT contains vague references, minimally revise them with respect to the specific subjects they refer to in the RESPONSE.
2. The decontextualized UNIT should be minimally revised by ONLY resolving vague references. No additional information must be added.
3. UNIT extraction might decompose a conjunctive statement into multiple units (e.g. Democracy treats citizens as equals regardless of their race or religion -> (1) Democracy treats citizens as equals regardless of their race, (2) Democracy treats citizens as equals regardless of their religion). Avoid adding what is potentially part of another UNIT.
4. Provide a reasoning of the revisions you made to the UNIT, justifying each decision.
5. After showing your reasoning, provide the revised unit and wrap it in a markdown code block.

Example 1: 
UNIT:
Acorns is a financial technology company

RESPONSE:
Acorns is a financial technology company founded in 2012 by Walter Cruttenden, \
Jeff Cruttenden, and Mark Dru that provides micro-investing services. The \
company is headquartered in Irvine, California.

REVISED UNIT:
This UNIT does not contain any vague references. Thus, the unit does not require any further decontextualization.
```
Acorns is a financial technology company
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

REVISED UNIT:
The UNIT contains a vague reference, "the victim." This is a reference to an unknown entity, \
since it is unclear who the victim is. From the RESPONSE, we can see that the victim is a Syrian refugee. \
Thus, the vague reference "the victim" should be replaced with "the Syrian refugee victim."
```
The Syrian refugee victim had previously suffered a broken wrist.
```

Example 3:
UNIT:
The difference is relatively small.

RESPONSE:
Both the RTX 3060 Ti and RTX 3060 are powerful GPUs, and the difference between them lies in their performance. \
The RTX 3060 Ti has more CUDA cores (4864 vs 3584) but a lower boost clock speed (1665 MHz vs 1777 MHz) compared to the RTX 3060. \
In terms of memory bandwidth, the RTX 3060 Ti has a slight edge over the RTX 3060 with a bandwidth of 448 GB/s compared to 360 GB/s. \
However, the difference is relatively small and may not be noticeable in real-world applications.

REVISED UNIT:
The UNIT contains a vague reference, "The difference." From the RESPONSE, we can see that the difference is in memory bandwidth between the RTX 3060 Ti and RTX 3060. \
Thus, the vague reference "The difference" should be replaced with "The difference in memory bandwidth between the RTX 3060 Ti and RTX 3060." \
The sentence from which the UNIT is extracted includes coordinating conjunctions that potentially decompose the statement into multiple units. Thus, adding more context to the UNIT is not necessary.
```
The difference in memory bandwidth between the RTX 3060 Ti and RTX 3060 is relatively small.
```

YOUR TASK:
UNIT:
{_UNIT_PLACEHOLDER}

RESPONSE:
{_RESPONSE_PLACEHOLDER}

REVISED UNIT:{_PROMPT_END_PLACEHOLDER}
"""

ATOM_DECONTEXT_PROMPT2 = """{_PROMPT_BEGIN_PLACEHOLDER}\
Instructions:
1. You are given a statement and a context that the statement belongs to. Your task is to modify the \
statement so that any pronouns or anaphora (words like "it," "they," "this") are replaced with the noun \
or proper noun that they refer to, such that the sentence remains clear without referring to the \
original context.
2. Return only the revised, standalone version of the statement without adding any information that is not \
already contained within the original statement.
3. If the statement requires no changes, return the original statement as-is without any explanation.  
4. The statement that you return must start with ### and finish with ### as follows: ###<statement>###.
5. Do not include any explanation or any additional formatting including any lead-in or sign-off text.
6. Learn from the provided examples below and use that knowledge to amend the last example yourself.

Example 1:
Context: John went to the store.
Statement: He bought some apples.
Standalone: ###John bought some apples.###

Example 2:
Context: The presentation covered various aspects of climate change, including sea level rise.
Statement: This was a key part of the discussion.
Standalone: ###Sea level rise was a key part of the discussion.###

Example 3:
Context: Maria Sanchez is a renowned marine biologist known for her groundbreaking research on coral reef ecosystems. \
Her work has contributed to the preservation of many endangered coral species, and she is often invited to speak at \
international conferences on environmental conservation.
Statement: She presented her findings at the conference last year.
Standalone: ###Maria Sanchez presented her findings at the conference last year.###

Example 4:
Context: Nathan Carter is a best-selling science fiction author famous for his dystopian novels that explore the \
intersection of technology and society. His latest book, The Edge of Something, received widespread critical acclaim \
for its imaginative world-building and its poignant commentary on artificial cacti.
Statement: It was praised for its thought-provoking themes.
Standalone: ###The Edge of Tomorrow was praised for its thought-provoking themes.###

Now perform the task for the following example:
Context: {_RESPONSE_PLACEHOLDER}
Statement: {_UNIT_PLACEHOLDER}
Standalone:{_PROMPT_END_PLACEHOLDER}        
"""

class AtomReviser:
    """
    Atomic unit/fact decontextualization given the response.
    """
    def __init__(
            self,
            model: str = "llama-3.1-70b-instruct",
            prompt_version: str = "v1",
            RITS: bool = True
    ):
        """
        Args:
            mode: str
                The name/id of the model.
            prompt_version: str
                The prompt version used. Allowed values are v1 - newer, v2 - original.
        """
        
        self.model = model
        self.prompt_version = prompt_version

        self.rits_model_info = RITS_MODELS[model] #only used to ger prompt begin/end
        self.prompt_begin = self.rits_model_info.get("prompt_begin", DEFAULT_PROMPT_BEGIN)
        self.prompt_end = self.rits_model_info.get("prompt_end", DEFAULT_PROMPT_END)
    
        self.llm_handler = LLMHandler(self.model, RITS=RITS)
        
        print(f"[AtomReviser] Using LLM on {RITS*'RITS'}{(not RITS)*'vLLM'}: {self.model}")
        print(f"[AtomReviser] Using prompt version: {self.prompt_version}")

    def make_prompt(self, unit: str, response: str):
        if self.prompt_version == "v1":
            prompt = ATOM_DECONTEXT_PROMPT1.format(
                _UNIT_PLACEHOLDER=unit,
                _RESPONSE_PLACEHOLDER=response,
                _PROMPT_BEGIN_PLACEHOLDER=self.prompt_begin,
                _PROMPT_END_PLACEHOLDER=self.prompt_end
            )
        elif self.prompt_version == "v2":
            prompt = ATOM_DECONTEXT_PROMPT2.format(
                _UNIT_PLACEHOLDER=unit,
                _RESPONSE_PLACEHOLDER=response,
                _PROMPT_BEGIN_PLACEHOLDER=self.prompt_begin,
                _PROMPT_END_PLACEHOLDER=self.prompt_end
            )
        else:
            raise ValueError(f"AtomReviser: Unknow prompt version {self.prompt_version}")
        prompt = strip_string(prompt)
        return prompt
        
    def run(self, atoms: List[str], response: str):

        results = []
        prompts = [self.make_prompt(atom, response) for atom in atoms]
        print(f"[AtomReviser] Prompts created: {len(prompts)}")

        for _, response in tqdm(
            enumerate(
                self.llm_handler.batch_completion(
                    prompts    
                )
            ),
            total=len(prompts),
            desc="Decontextualization",
            unit="prompts",
            ):
                results.append(response.choices[0].message.content)

        revised_atoms = []
        if self.prompt_version == "v1":
            revised_atoms = [extract_first_code_block(output, ignore_language=True) for output in results]
        elif self.prompt_version == "v2":
            revised_atoms = [extract_last_wrapped_response(output) for output in results]

        for revised_atom in revised_atoms:
            if len(revised_atom) > 0:
                if not revised_atom[-1] in string.punctuation:
                    revised_atom += "."

        final_revised_atoms = []
        for i in range(len(atoms)):
            if len(revised_atoms[i]) > 0:
                final_revised_atoms.append(dict(revised_atom=revised_atoms[i], atom=atoms[i]))
            else:
                final_revised_atoms.append(dict(revised_atom=atoms[i], atom=atoms[i]))

        return final_revised_atoms
    
    def runall(self, atoms: List[List[str]], responses: List[str]):

        n = len(responses)
        results = []
        prompts = [self.make_prompt(atom, response) for i, response in enumerate(responses) for atom in atoms[i]]
        print(f"[AtomReviser] Prompts created: {len(prompts)}")

        for _, response in tqdm(
            enumerate(
                self.llm_handler.batch_completion(
                    prompts
                )
            ),
            total=len(prompts),
            desc="Decontextualization",
            unit="prompts",
            ):
                results.append(response.choices[0].message.content)

        revised_atoms = []
        if self.prompt_version == "v1":
            revised_atoms = [extract_first_code_block(output, ignore_language=True) for output in results]
        elif self.prompt_version == "v2":
            revised_atoms = [extract_last_wrapped_response(output) for output in results]

        # TODO: need to fix the problematic revised atoms!!
        for revised_atom in revised_atoms:
            if len(revised_atom) > 0 and not revised_atom[-1] in string.punctuation:
                revised_atom += "."

        k = 0        
        outputs = []
        for j in range(n):
            output = [{'revised_atom': revised_atoms[k + i], 'atom': atoms[j][i]} for i in range(len(atoms[j]))]
            outputs.append(output)
            k += len(atoms[j])
        return outputs
        
if __name__ == "__main__":
    
    model = "granite-3.1-8b-instruct"
    prompt_version = "v2"

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

    atoms = [
        "He has appeared in numerous films.",
        "He has appeared in numerous television shows.",
        "He has appeared in numerous theater productions.",
        "His career began in the late 1970s."
    ]

    decontextualizer = AtomReviser(model=model, prompt_version=prompt_version)
    results = decontextualizer.run(atoms, response)
    for elem in results:
        orig_atom = elem["atom"]
        revised_atom = elem["revised_atom"]
        print(f"{orig_atom} --> {revised_atom}")

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

    atoms = [
        [
            "He was born on July 23, 1904.",
            "He was born in Frankfurt, Germany.",
            "He moved to the United States in 1929.",
            "He became a citizen in the United States in 1941.",
        ],
        [
            "He has appeared in numerous films.",
            "He has appeared in numerous television shows.",
            "He has appeared in numerous theater productions.",
            "His career began in the late 1970s."
        ] 
    ]

    results = decontextualizer.runall(atoms, responses)
    print(f"Number of results: {len(results)}")
    for result in results:
        for elem in result:
            orig_atom = elem["atom"]
            revised_atom = elem["revised_atom"]
            print(f"{orig_atom} --> {revised_atom}")


    print("Done.")
