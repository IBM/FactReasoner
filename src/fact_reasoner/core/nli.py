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

# NLI extractor using LLMs.

import asyncio
import math
import mellea.stdlib.functional as mfuncs

from typing import Any, Dict, List

from mellea.backends import Backend
from mellea.backends.types import ModelOption
from mellea.stdlib.base import SimpleContext
from mellea.stdlib.base import ModelOutputThunk
from mellea.stdlib.requirement import check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

# Local imports
from src.fact_reasoner.utils import extract_last_square_brackets

INSTRUCTION_NLI = """

Instructions:
You are provided with a PREMISE and a HYPOTHESIS. 
Your task is to evaluate the relationship between the PREMISE and the HYPOTHESIS, following the steps outlined below:

1. Evaluate Relationship:
- If the PREMISE strongly implies or directly supports the HYPOTHESIS, explain the supporting evidence.
- If the PREMISE contradicts the HYPOTHESIS, identify and explain the conflicting evidence.
- If the PREMISE is insufficient to confirm or deny the HYPOTHESIS, explain why the evidence is inconclusive.
2. Provide the reasoning behind your evaluation of the relationship between PREMISE and HYPOTHESIS, justifying each decision.
3. Final Answer: Based on your reasoning, the HYPOTHESIS and the PREMISE, determine your final answer. \
Your final answer must be one of the following, wrapped in square brackets:
- [entailment] if the PREMISE strongly implies, directly supports or entails the HYPOTHESIS.
- [contradiction] if the PREMISE contradicts the HYPOTHESIS.
- [neutral] if the PREMISE and the HYPOTHESIS neither entail nor contradict each other.

Use the following examples to better understand your task.

Example 1:
PREMISE: Robert Haldane Smith, Baron Smith of Kelvin, KT, CH, FRSGS is a British businessman and former Governor of the British Broadcasting Corporation. Smith was knighted in 1999, appointed to the House of Lords as an independent crossbench peer in 2008, and appointed Knight of the Thistle in the 2014 New Year Honours.
HYPOTHESIS: Robert Smith holds the title of Baron Smith of Kelvin.
1. Evaluate Relationship:
The PREMISE states that Robert Haldane Smith, Baron Smith of Kelvin, KT, CH, FRSGS is a British businessman and former Governor of the British Broadcasting Corporation. It also mentions that Smith was appointed to the House of Lords as an independent crossbench peer in 2008. This information directly supports the HYPOTHESIS that Robert Smith holds the title of Baron Smith of Kelvin.
2: Reasoning:
The PREMISE explicitly mentions that Robert Smith is Baron Smith of Kelvin, which directly supports the HYPOTHESIS. The additional information about his knighthood, appointment to the House of Lords, and other titles further confirms his status as a peer, but it is not necessary to support the specific HYPOTHESIS about him holding the title of Baron Smith of Kelvin.
3. Final Answer: 
[entailment]

Example 2:
PREMISE: In 2022, Passover begins in Israel at sunset on Friday, 15 April, and ends at sunset on Friday, 22 April 2022.
HYPOTHESIS: Passover in 2022 begins at sundown on March 27.
1. Evaluate Relationship:
The PREMISE states that Passover in 2022 begins at sunset on Friday, 15 April, and ends at sunset on Friday, 22 April 2022. The HYPOTHESIS claims that Passover in 2022 begins at sundown on March 27. 
Upon analyzing the information, I found that the dates mentioned in the PREMISE and the HYPOTHESIS do not match. Since the dates provided in the PREMISE and the HYPOTHESIS are different, the HYPOTHESIS is contradicted by the PREMISE.
2. Reasoning:
The PREMISE provides specific information about the start date of Passover in 2022, which is April 15. The HYPOTHESIS, on the other hand, claims a different start date, March 27. This discrepancy indicates that the PREMISE and the HYPOTHESIS cannot both be true.
3. Final Answer:
[contradiction]

Example 3:
PREMISE: Little India in the East Village: Two restaurants ablaze with tiny colored lights stand at the top of a steep staircase.
HYPOTHESIS: The village had colorful decorations on every street corner.
1. Evaluate Relationship:
The PREMISE describes a specific scene in Little India in the East Village, where two restaurants are decorated with tiny colored lights at the top of a steep staircase. The HYPOTHESIS makes a broader claim that the village had colorful decorations on every street corner.
The PREMISE provides evidence of colorful decorations in one specific location, but it does not provide information about the decorations on every street corner in the village. The PREMISE is insufficient to confirm or deny the HYPOTHESIS, as it only describes a small part of the village.
2. Reasoning:
The PREMISE and HYPOTHESIS are related in that they both mention colorful decorations, but the scope of the HYPOTHESIS is much broader than the PREMISE. The PREMISE only provides a glimpse into one specific location, whereas the HYPOTHESIS makes a general claim about the entire village. Without more information, it is impossible to determine whether the village had colorful decorations on every street corner.
3. Final Answer:
[neutral]

Your task:
PREMISE: {{premise_text}}
HYPOTHESIS: {{hypothesis_text}}
"""

class NLIExtractor:
    """
    Predict the NLI relationship between a premise and a hypothesis, optionally
    given a context (or response). The considered relationships are: entailment,
    contradiction and neutrality. We use few-shot prompting for LLMs.

    v1 - original
    v2 - more recent (with reasoning)
    v3 - only for Google search results
    """
    
    def __init__(
            self,
            backend: Backend,
    ):
        """
        Initialize the NLIExtractor.

        Args:
            backend: Backend
                The Mellea backend to use for LLM interaction.
        """

        # Safety checks        
        if backend is None:
            raise ValueError("Mellea backend is None. Please provide a valid Mellea backend.")

        self.method = "logprobs"
        self.backend = backend

        # Print info
        print(f"[NLI] Using Mellea backend: {self.backend.model_id}")

    def _get_probability(self, output: ModelOutputThunk) -> float:
        """
        Compute the average log probability of the generated tokens.

        Args:
            output: ModelOutputThunk
                The model raw output (via Mellea).

        Returns:
            float: The average log probability of the generated tokens.
        """

        assert output._meta["oai_chat_response"]["logprobs"] is not None
        logprobs = output._meta["oai_chat_response"]["logprobs"]["content"][:-1] # last token is EOS

        # Go backwards and collect the logprobs of the tokens between ']' and ']'
        avg_logprob = 0
        count = 0
        for item in reversed(logprobs):
            if item['token'] == '[':
                break
            elif item['token'] == ']':
                continue
            else:
                avg_logprob += item['logprob']
                count += 1

        # Compute the probability
        avg_logprob = avg_logprob / count if count > 0 else math.inf
        return math.exp(avg_logprob) if not math.isinf(avg_logprob) else 0.0 

    def _get_label(self, output: ModelOutputThunk) -> str:
        """
        Extract the NLI label from the model output.

        Args:
            output: ModelOutputThunk
                The model raw output (via Mellea)

        Returns:
            str: The string representing the NLI label (entailment, contradiction, neutral).        
        """

        return extract_last_square_brackets(str(output))

    def run(self, premise: str, hypothesis: str) -> Dict[str, Any]:
        """
        Extract the NLI relationship between premise and hypothesis. The 
        following relationships are allowed: entailment, contradiction, neutral.
        
        Args:
            premise: str
                The premise text (e.g., context).
            hypothesis: str
                The hypothesis text (e.g., atom).

        Returns:
            Dict[str, Any]: A dictionary containing the relationship and its probability.
        """
        
        # Perform the instruction with validation
        output = mfuncs.instruct(
            INSTRUCTION_NLI,
            context=SimpleContext(),
            backend=self.backend,
            requirements=[
                check(
                    "The output must be a wrapped in square brackets",
                    validation_fn=simple_validate(
                        lambda s: extract_last_square_brackets(s) != ''
                    ),
                )
            ],
            user_variables={"premise_text": premise, "hypothesis_text": hypothesis},
            strategy=RejectionSamplingStrategy(loop_budget=3),
            return_sampling_results=True,
            model_options=dict(logprobs=True),
        )

        if output.success:
            return dict(
                label=self._get_label(output.result), 
                probability=self._get_probability(output.result)
            )
        else:
            return dict(
                label="neutral", 
                probability=1.0
            )

    async def run_batch(self, premises: List[str], hypotheses: List[str]) -> List[Dict[str, Any]]:
        """
        Extract the NLI relationships between premises and hypotheses. The 
        following relationships are allowed: entailment, contradiction, neutral.
        
        Args:
            premises: List[str]
                The list of premise texts (e.g., context).
            hypotheses: List[str]
                The list of hypothesis texts (e.g., atom).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the 
            relationships and their probabilities.
        """

        corutines = []
        for premise, hypothesis in zip(premises, hypotheses):
            corutine = mfuncs.ainstruct(
                INSTRUCTION_NLI,
                context=SimpleContext(),
                backend=self.backend,
                requirements=[
                    check(
                        "The output must be a wrapped in square brackets",
                        validation_fn=simple_validate(
                            lambda s: extract_last_square_brackets(s) != ''
                        ),
                    )
                ],
                user_variables={"premise_text": premise, "hypothesis_text": hypothesis},
                strategy=RejectionSamplingStrategy(loop_budget=3),
                return_sampling_results=True,
                model_options=dict(logprobs=True),
            )
            corutines.append(corutine)

        results = []
        print(f"[NLI] Awaiting for async execution ...")
        outputs = await asyncio.gather(*(corutines[i] for i in range(len(corutines))))
        for output in outputs:
            if output.success:
                label = self._get_label(output.result)
                probability = self._get_probability(output.result)
                results.append(dict(
                    label=label,
                    probability=probability
                ))
            else:
                results.append(dict(
                    label="neutral",
                    probability=1.0
                ))

        return results

if __name__ == "__main__":

    # Create a Mellea RITS backend
    from mellea_ibm.rits import RITSBackend, RITS
    backend = RITSBackend(
        RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 500},
    )

    # Example premise and hypothesis
    premise = "natural born killers is a 1994 american romantic crime action film directed by oliver stone and starring woody harrelson, juliette lewis, robert downey jr., tommy lee jones, and tom sizemore. the film tells the story of two victims of traumatic childhoods who become lovers and mass murderers, and are irresponsibly glorified by the mass media. the film is based on an original screenplay by quentin tarantino that was heavily revised by stone, writer david veloz, and associate producer richard rutowski. tarantino received a story credit though he subsequently disowned the film. jane hamsher, don murphy, and clayton townsend produced the film, with arnon milchan, thom mount, and stone as executive producers. natural born killers was released on august 26, 1994 in the united states, and screened at the venice film festival on august 29, 1994. it was a box office success, grossing $ 110 million against a production budget of $ 34 million, but received polarized reviews. some critics praised the plot, acting, humor, and combination of action and romance, while others found the film overly violent and graphic. notorious for its violent content and inspiring \" copycat \" crimes, the film was named the eighth most controversial film in history by entertainment weekly in 2006. = = plot = = mickey knox and his wife mallory stop at a diner in the new mexico desert. a duo of rednecks arrive and begin sexually harassing mallory as she dances by a jukebox. she initially encourages it before beating one of the men viciously. mickey joins her, and the couple murder everyone in the diner, save one customer, to whom they proudly declare their names before leaving. the couple camp in the desert, and mallory reminisces about how she met mickey, a meat deliveryman who serviced her family ' s household. after a whirlwind romance, mickey is arrested for grand theft auto and sent to prison ; he escapes and returns to mallory ' s home. the couple murders mallory ' s sexually abusive father and neglectful mother, but spare the life of mallory ' s little brother, kevin. the couple then have an unofficial marriage ceremony on a bridge. later, mickey and mallory hold a woman hostage in their hotel room. angered by mickey ' s desire for a threesome, mallory leaves, and mickey rapes the hostage. mallory drives to a nearby gas station, where she flirts with a mechanic. they begin to have sex on the hood of a car, but after mallory suffers a flashback of being raped by her father, and the mechanic recognizes her as a wanted murderer, mallory kills him. the pair continue their killing spree, ultimately claiming 52 victims in new mexico, arizona and nevada. pursuing them is detective jack scagnetti, who became obsessed with mass murderers at the age of eight after having witnessed the murder of his mother at the hand of charles whitman. beneath his heroic facade, he is also a violent psychopath and has murdered prostitutes in his past. following the pair ' s murder spree is self - serving tabloid journalist wayne gale, who profiles them on his show american maniacs, soon elevating them to cult - hero status. mickey and mallory become lost in the desert after taking psychedelic mushrooms, and they stumble upon a ranch owned by warren red cloud, a navajo man who provides them food and shelter. as mickey and mallory sleep, warren, sensing evil in the couple, attempts to exorcise the demon that he perceives in mickey, chanting over him as he sleeps. mickey, who has nightmares of his abusive parents, awakens during the exorcism and shoots warren to death. as the couple flee, they feel inexplicably guilty and come across a giant field of rattlesnakes, where they are badly bitten. they reach a drugstore to purchase snakebite antidote, but the store is sold out. a pharmacist recognizes the couple and triggers an alarm before mickey kills him. police arrive shortly after and accost the couple and a shootout ensues. the police end the showdown by beating the couple while a "
    hypothesis = "Lanny Flaherty has appeared in numerous films."

    # Create the extractor
    extractor = NLIExtractor(backend)    
    result = extractor.run(premise=premise, hypothesis=hypothesis)

    # Output results
    print(f"H -> P: {result}")
    print("Done.")