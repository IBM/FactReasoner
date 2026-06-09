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
import re
import mellea.stdlib.functional as mfuncs

from typing import Any, Dict, List, Optional

from mellea.backends import Backend
from mellea.stdlib.context import SimpleContext
from mellea.core import ModelOutputThunk 
from mellea.stdlib.requirements import check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.core import FancyLogger

# Local imports
from fact_reasoner.utils import extract_last_square_brackets

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
Your final answer must be one of the following: entailment, contradiction or neutral, wrapped in square brackets:
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


INSTRUCTION_NLI_SCIENCE = """
Instructions:
You are provided with a CONTEXT, an ATOM, and optionally an ATOM_TYPE.

Your task is to evaluate the relationship between the CONTEXT and the ATOM as they appear in scientific literature. Use careful scientific reasoning and rely only on the information in the CONTEXT.

Definitions:
- [entailment]: The CONTEXT provides explicit or directly inferable support for the ATOM within the same scope, without introducing new assumptions. All key elements (entities, variables, values, direction, units, comparisons, and conditions) must match or be clearly implied.
- [contradiction]: The CONTEXT provides information that is incompatible with the ATOM, such that both cannot be true under the same scope. This includes conflicts in values, direction of effect, statistical significance, comparisons, definitions, assumptions, or experimental conditions.
- [neutral]: The CONTEXT does not provide sufficient or appropriate evidence to confirm or deny the ATOM. This includes missing information, partial overlap, scope mismatch, weaker evidence than the ATOM, attribution without validation, or unstated assumptions required by the ATOM.

Important rule: When uncertain between entailment and neutral, choose neutral.

Follow the steps below:

1. Identify claim type and scope
- If ATOM_TYPE is provided, use it as guidance.
- Otherwise, infer the type (e.g., Result, Data, Statistical Statement, Comparison, Method, Dataset, Experimental Condition, Assumption, Definition, Conclusion, Claim, Hypothesis, Background Fact, Limitation, Negative Result, Attribution, Citation, Reference, etc.).
- Identify the scope: population, dataset, region, time period, variables, conditions, intervention, and comparison.

2. Extract evidence from CONTEXT
- Focus on exact wording related to:
  - values, units, and ranges
  - direction of effect
  - statistical significance
  - comparisons
  - methods, datasets, and conditions
  - definitions and assumptions
  - attribution to prior work

3. Evaluate relationship
- Entailment: direct support with matching scope and meaning
- Contradiction: incompatible information
- Neutral: insufficient, partial, weaker, or mismatched scope

4. Apply scientific reasoning rules
- Do not use external knowledge
- Do not infer causation from correlation
- Do not generalize beyond scope
- Distinguish:
  - result vs interpretation
  - lack of evidence vs evidence of absence
  - study findings vs cited prior work

5. Provide reasoning
Use exactly this format:

Scope: <brief description>

Evidence from context: <brief description>

Reasoning: <brief explanation>

Final Answer: <one label only>

6. Final answer
Output exactly one:
[entailment]
[contradiction]
[neutral]

Example 1:
CONTEXT: In a randomized controlled trial of 240 adults with type 2 diabetes, participants receiving Drug A showed a statistically significant reduction in HbA1c after 12 weeks compared with placebo.
ATOM: Drug A reduced HbA1c levels in adults with type 2 diabetes over 12 weeks.
ATOM_TYPE: Result

Scope: Treatment effect in adults with type 2 diabetes over 12 weeks.
Evidence from context: The CONTEXT states a statistically significant reduction in HbA1c after 12 weeks with Drug A.
Reasoning: The population, intervention, outcome, and timeframe match exactly.
Final Answer: [entailment]

Example 2:
CONTEXT: In a cohort study, higher coffee intake was associated with lower all-cause mortality, but causality cannot be inferred.
ATOM: Drinking more coffee causes lower mortality.
ATOM_TYPE: Claim

Scope: Causal claim about coffee intake and mortality.
Evidence from context: The CONTEXT reports association but explicitly states causality cannot be inferred.
Reasoning: The ATOM overstates association as causation.
Final Answer: [neutral]

Example 3:
CONTEXT: The intervention reduced systolic blood pressure from 150 mmHg to 135 mmHg, a decrease of 15 mmHg.
ATOM: The intervention reduced systolic blood pressure by 10 mmHg.
ATOM_TYPE: Data

Scope: Quantitative reduction in blood pressure.
Evidence from context: The CONTEXT reports a reduction of 15 mmHg.
Reasoning: The ATOM provides an incorrect numerical value.
Final Answer: [contradiction]

Example 4:
CONTEXT: The difference between groups was not statistically significant (p = 0.18).
ATOM: The groups differed significantly.
ATOM_TYPE: Statistical Statement

Scope: Statistical significance of group difference.
Evidence from context: The CONTEXT explicitly states no statistical significance.
Reasoning: The ATOM asserts the opposite conclusion.
Final Answer: [contradiction]

Example 5:
CONTEXT: Sentinel-2 imagery shows NDVI increased by 12% across the region between 2018 and 2022 during the growing season.
ATOM: NDVI increased by 12% between 2018 and 2022 during the growing season.
ATOM_TYPE: Result

Scope: Vegetation change over time under specific conditions.
Evidence from context: The CONTEXT reports a 12% NDVI increase over the same period and condition.
Reasoning: The variable, magnitude, and timeframe match.
Final Answer: [entailment]

Example 6:
CONTEXT: The model was trained on satellite imagery collected over Europe between 2015 and 2020 under clear-sky conditions.
ATOM: The model was trained under clear-sky conditions using satellite imagery from Europe between 2015 and 2020.
ATOM_TYPE: Method

Scope: Model training conditions and dataset.
Evidence from context: The CONTEXT states the same training data, region, timeframe, and condition.
Reasoning: All elements match exactly.
Final Answer: [entailment]

YOUR TASK:
CONTEXT: {{premise_text}}
ATOM: {{hypothesis_text}}
ATOM_TYPE: {{hypothesis_type}}
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

        # Disable Mellea logging
        FancyLogger.get_logger().setLevel(FancyLogger.ERROR)


    def _get_probability(self, output: ModelOutputThunk) -> float:
        """
        Compute the average log probability of the generated tokens.

        Args:
            output: ModelOutputThunk
                The model raw output (via Mellea).

        Returns:
            float: The average log probability of the generated tokens.
        """
        # assert output._meta["oai_chat_response"]["logprobs"] is not None
        # logprobs = output._meta["oai_chat_response"]["logprobs"]["content"][:-1] # last token is EOS

        assert output._meta["oai_chat_response"]['choices'][0]['logprobs'] is not None
        logprobs = output._meta["oai_chat_response"]['choices'][0]['logprobs']["content"][:-1] # last token is EOS


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

    def _get_reasoning(self, output: ModelOutputThunk) -> str:
        text = str(output)

        match = re.search(
            r"Reasoning:\s*(.*?)\s*(?:\d+\.\s*)?Final Answer:",
            text,
            re.DOTALL
        )

        return match.group(1).strip() if match else ""

    def run(
        self, 
        premise: str, 
        hypothesis: str,
        hypothesis_type: str = "",
        science_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Extract the NLI relationship between premise and hypothesis. The 
        following relationships are allowed: entailment, contradiction, neutral.
        
        Args:
            premise: str
                The premise text (e.g., context).
            hypothesis: str
                The hypothesis text (e.g., atom).
            science_mode: bool
                 The scientific mode for the prompts.

        Returns:
            Dict[str, Any]: A dictionary containing the relationship and its probability.
        """

        prompt = INSTRUCTION_NLI_SCIENCE if science_mode else INSTRUCTION_NLI

        if hypothesis_type is None:
            hypothesis_type = ""
        
        # Perform the instruction with validation
        output = mfuncs.instruct(
            prompt,
            context=SimpleContext(),
            backend=self.backend,
            requirements=[
                check(
                    "The output must be a wrapped in square brackets",
                    validation_fn=simple_validate(
                        lambda s: extract_last_square_brackets(s) in {"entailment", "contradiction", "neutral"}
                    ),
                )
            ],
            user_variables={
                "premise_text": premise, 
                "hypothesis_text": hypothesis,
                "hypothesis_type": hypothesis_type
            },
            strategy=RejectionSamplingStrategy(loop_budget=3),
            return_sampling_results=True,
            model_options=dict(logprobs=True),
        )

        if output.success:
            label = self._get_label(output.result)
            probability = self._get_probability(output.result)
            
            reasoning = self._get_reasoning(output.result) if science_mode else ""

            if label not in ["entailment", "contradiction", "neutral"]:
                label = "neutral"

            return dict(
                label=label, 
                probability=probability,
                reasoning=reasoning
            )
        else:
            return dict(
                label="neutral", 
                probability=1.0,
                reasoning=""
            )

    async def run_batch(
        self, 
        premises: List[str], 
        hypotheses: List[str],
        hypotheses_types: Optional[List[str]] = None,
        science_mode: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Extract the NLI relationships between premises and hypotheses. The 
        following relationships are allowed: entailment, contradiction, neutral.
        
        Args:
            premises: List[str]
                The list of premise texts (e.g., context).
            hypotheses: List[str]
                The list of hypothesis texts (e.g., atom).
            science_mode: bool
                The scientific mode for the prompts.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the 
            relationships and their probabilities.
        """

        prompt = INSTRUCTION_NLI_SCIENCE if science_mode else INSTRUCTION_NLI

        if hypotheses_types is None:
            hypotheses_types = [""] * len(hypotheses)

        corutines = []
        for premise, hypothesis, hypothesis_type in zip(premises, hypotheses, hypotheses_types):
            corutine = mfuncs.ainstruct(
                prompt,
                context=SimpleContext(),
                backend=self.backend,
                requirements=[
                    check(
                        "The output must be a wrapped in square brackets",
                        validation_fn=simple_validate(
                            lambda s: extract_last_square_brackets(s) in {"entailment", "contradiction", "neutral"}
                        ),
                    )
                ],
                user_variables={
                    "premise_text": premise, 
                    "hypothesis_text": hypothesis,
                    "hypothesis_type": hypothesis_type
                },
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
                reasoning = self._get_reasoning(output.result) if science_mode else ""
                results.append(dict(
                    label=label,
                    probability=probability,
                    reasoning=reasoning
                ))
            else:
                results.append(dict(
                    label="neutral",
                    probability=1.0,
                    reasoning=""
                ))

        return results

