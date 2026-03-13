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
            raise ValueError(
                "Mellea backend is None. Please provide a valid Mellea backend."
            )

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

        # Supports new direct path (updated mellea, incl. Bedrock converse),
        # OpenAIBackend ("oai_chat_response"), and LiteLLMBackend
        # ("litellm_chat_response").
        logprobs_object = (
            output._meta.get("logprobs")
            or output._meta.get("oai_chat_response", {}).get("logprobs")
            or output._meta.get("litellm_chat_response", {}).get("logprobs")
        )
        assert (
            logprobs_object is not None
        ), "logprobs missing from response. Ensure the backend supports logprobs."

        # last token is EOS
        logprobs = logprobs_object["content"][:-1]

        # OpenAI-compatible backends return string tokens (e.g. "[", "]").
        # The native Bedrock InvokeModel API returns numeric token IDs as
        # strings (e.g. "58"). Detect which format we have.
        has_string_tokens = any(item["token"] in ("[", "]") for item in logprobs)

        avg_logprob = 0
        count = 0

        if has_string_tokens:
            # Original logic: walk backwards, collect logprobs of tokens
            # between the last ']' and the matching '['.
            for item in reversed(logprobs):
                if item["token"] == "[":
                    break
                elif item["token"] == "]":
                    continue
                else:
                    avg_logprob += item["logprob"]
                    count += 1
        else:
            # Bedrock native: numeric token IDs — can't identify '['/']'
            # without the tokenizer. Proxy confidence via the last few
            # tokens, which correspond to the label at end of generation
            # (e.g. "[entailment]" tokenises to ~4 tokens).
            label_window = logprobs[-5:] if len(logprobs) >= 5 else logprobs
            for item in label_window:
                avg_logprob += item["logprob"]
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
                        lambda s: extract_last_square_brackets(s) != ""
                    ),
                )
            ],
            user_variables={"premise_text": premise, "hypothesis_text": hypothesis},
            strategy=RejectionSamplingStrategy(loop_budget=3),
            return_sampling_results=True,
            model_options={
                "logprobs": True,
                "top_logprobs": 5,
            },
        )

        if output.success:
            label = self._get_label(output.result)
            probability = self._get_probability(output.result)
            if label not in ["entailment", "contradiction", "neutral"]:
                label = "neutral"

            return dict(label=label, probability=probability)
        else:
            return dict(label="neutral", probability=1.0)

    async def run_batch(
        self, premises: List[str], hypotheses: List[str]
    ) -> List[Dict[str, Any]]:
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
                            lambda s: extract_last_square_brackets(s) != ""
                        ),
                    )
                ],
                user_variables={"premise_text": premise, "hypothesis_text": hypothesis},
                strategy=RejectionSamplingStrategy(loop_budget=3),
                return_sampling_results=True,
                model_options={
                    "logprobs": True,
                    "top_logprobs": 5,
                },
            )
            corutines.append(corutine)

        results = []
        print(f"[NLI] Awaiting for async execution ...")
        outputs = await asyncio.gather(*(corutines[i] for i in range(len(corutines))))
        for output in outputs:
            if output.success:
                label = self._get_label(output.result)
                probability = self._get_probability(output.result)
                results.append(dict(label=label, probability=probability))
            else:
                results.append(dict(label="neutral", probability=1.0))

        return results
