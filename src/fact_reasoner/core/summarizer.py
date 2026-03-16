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

# Context summarization using LLMs

import math
import asyncio
import mellea.stdlib.functional as mfuncs

from typing import Any, Dict, List
from mellea.backends import Backend
from mellea.backends import ModelOption
from mellea.stdlib.context import SimpleContext
from mellea.core import ModelOutputThunk
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.core import FancyLogger

from fact_reasoner.utils import LOOP_BUDGET

INSTRUCTION_WITHOUT_REF = """
You are tasked with summarising a long paragraph into a shorter, more concise version. 
Follow these rules strictly:

Rules:
1. Do NOT add any new information.  
2. Do NOT remove any information or meaning.  
3. Preserve all facts, relationships, and intent.  
4. The summary must be significantly more concise while remaining fully accurate.  
5. Maintain the original tone and perspective.

Use the provided examples to learn the task better.

EXAMPLE 1:
Input:
The research team conducted a six-month study to evaluate the effectiveness of the new machine learning model. They compared its performance against three existing baseline models and found that the new approach improved prediction accuracy by 15%. However, they also noted that training time increased significantly, which might limit its applicability in real-time systems."

Summary:
The six-month study found the new machine learning model improved accuracy by 15% over three baselines but required significantly longer training, limiting its real-time use.

EXAMPLE 2:
Input:
During the conference, several speakers emphasized the importance of transparent AI systems. They argued that without clear explanations of how models make decisions, user trust will remain low. Some presenters showcased tools designed to visualize model reasoning, which they claimed could help bridge the trust gap.

Summary:
Conference speakers stressed that transparent AI is essential for user trust and presented visualization tools to explain model reasoning and reduce the trust gap.

EXAMPLE 3:
Input:
The city's transportation department announced a new initiative to reduce traffic congestion by expanding bike lanes and increasing public transit frequency. Officials believe this will encourage more residents to use alternatives to driving, ultimately decreasing emissions and improving overall air quality.

Summary:
The transportation department plans to reduce congestion by expanding bike lanes and increasing transit frequency, aiming to shift residents from driving and improve emissions and air quality.

Your task:
Input:
{{context}}

Summary:
"""

INSTRUCTION_WITH_REF = """

Your task is to summarize the CONTEXT with respect to the ATOM.

Instructions: 
Follow the steps below for CONTEXT summarization:
1. The ATOM can be true, false or not verifiable according to the SUMMARY.
2. It is very possible that no relevant information about the ATOM or related to the ATOM can be found in the CONTEXT. In this case, the SUMMARY must be: "None".
3. If the CONTEXT does not provide information about the ATOM, or if the CONTEXT does not mention anything related to the ATOM, the SUMMARY must be: "None".
4. If the CONTEXT provides information about the ATOM, the SUMMARY must contain the most relevant information of the CONTEXT and be such that we can fact-check the ATOM using this SUMMARY. 
5. The SUMMARY must not use reported speech to refer to the CONTEXT, for instance the SUMMARY must NOT state: "according to the context", "this context mentions", or "this article outlines", but instead the SUMMARY must only summarize the CONTEXT.
6. If the CONTEXT provides information about the ATOM, provide the SUMMARY.
7. If the CONTEXT does not provide information about the ATOM, the SUMMARY must only provide "None". Do not mention that the context does not provide any information about the atom. Do not provide anything else.

Use the provided examples to learn the task better.

Example 1:
ATOM: Sense and Sensibility was published in the summer of 1811.

CONTEXT: + Sense and Sensibility + Sense and Sensibility is a novel by Jane \
Austen , published in 1811 . + Jane Austen + Jane Austen ( 16 December 1775 - 18 July \
1817 ) was an English novelist known primarily for her six major novels , which interpret , \
critique and comment upon the British landed gentry at the end of the 18th century .

SUMMARY:
Sense and Sensibility was published in 1811, however it is not known whether it \
has been published in summer.

Example 2:
ATOM: Filmfare is about cheese.

CONTEXT: + Filmfare + Filmfare is an English-language , tabloid-sized magazine \
about Hindi-language cinema , popularly known as Bollywood . + Bollywood + Bollywood \
is the sobriquet for India 's Hindi language film industry , based in the city of Mumbai , \
Maharashtra .

SUMMARY: 
Filmfare is about Hindi-language cinema, not about cheese.

Example 3:
ATOM: The 19th G7 summit only included Russia.

CONTEXT:
+ 19th G7 summit + The Group of Seven ( G7 ) was an unofficial forum \
which brought together the heads of the richest industrialized countries : France , Germany \
, Italy , Japan , the United Kingdom , the United States , Canada ( since 1976 ) and the \
President of the European Commission ( starting officially in 1981 ) .

SUMMARY: 
The 19th G7 summit did not only include Russia, but also the heads of the six \
other richest industrialized countries and the President of the European Commission.

Example 4:
ATOM: Quantum mechanics describes the behavior of particles at the smallest scales, where classical physics no longer applies.

CONTEXT:
The Amazon rainforest, often referred to as the "lungs of the Earth," spans over 5.5 million square kilometers across nine countries. \
It is home to millions of species, many of which are yet to be discovered. The rainforest plays a crucial role in global oxygen production \
and carbon dioxide absorption. However, it faces severe threats from deforestation, illegal mining, and climate change. Conservation efforts \
are ongoing, with governments, environmental organizations, and indigenous communities working together to protect this vital ecosystem. 

SUMMARY:
None


Example 5:
ATOM: Zeus was the creator of Nazgul.

CONTEXT:
+ Artemis + She was the Hellenic goddess of the hunt , wild animals , \
wilderness , childbirth , virginity and protector of young girls , bringing and relieving \
disease in women ; she often was depicted as a huntress carrying a bow and arrows .

SUMMARY:
None

Your task:
ATOM: {{atom_text}}

CONTEXT: {{context}}

SUMMARY:
"""


class ContextSummarizer:
    """
    Context summarization given the atom.
    """

    def __init__(
        self,
        backend: Backend,
    ):
        """
        Initialize the ContextSummarizer.

        Args:
            backend: str
                The Mellea backend to use for LLM interaction.
            with_reference: str
                The reference paragraph that the context will be summarized to.
        """

        # Safety checks
        if backend is None:
            raise ValueError(
                "Mellea backend is None. Please provide a valid Mellea backend."
            )

        # Initialize the extractor
        self.backend = backend

        # Print info
        print(f"[Summarizer] Using Mellea backend: {self.backend.model_id}")

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

        logprobs_object = (
            output._meta.get("logprobs")
            or output._meta.get("oai_chat_response", {}).get("logprobs")
            or output._meta.get("litellm_chat_response", {}).get("logprobs")
        )
        assert (
            logprobs_object is not None
        ), "logprobs missing from response. Ensure the backend supports logprobs."
        logprobs = logprobs_object["content"][:-1]  # last token is EOS
        avg_logprob = (
            sum(lp["logprob"] for lp in logprobs) / len(logprobs)
            if len(logprobs) > 0
            else math.inf
        )

        return math.exp(avg_logprob) if not math.isinf(avg_logprob) else 0.0

    def run(self, contexts: List[str], atom_text: str = None) -> List[Dict[str, Any]]:
        """
        Summarize a list of contexts with respect to an atomic unit.

        Args:
            contexts: List[str]
                The list of contexts to be summarized.
            atom_text: str
                The reference atomic unit text
        Returns:
            List[Dict[str, Any]]: A list of summarized contexts.
        """

        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, self.run_batch(contexts, atom_text))
            return future.result()

    async def run_batch(
        self, contexts: List[str], atom_text: str = None
    ) -> List[Dict[str, Any]]:
        """
        Summarize a list of contexts with respect to an atomic unit.

        Args:
            contexts: List[str]
                The list of contexts to be summarized.
            atom_text: str
                The reference atomic unit text
        Returns:
            List[Dict[str, Any]]: A list of summarized contexts.
        """

        # Initialize the instruction
        instruction = (
            INSTRUCTION_WITH_REF if atom_text is not None else INSTRUCTION_WITHOUT_REF
        )

        # Perform the instruction with validation
        results = []
        coroutines = []
        for context in contexts:
            coroutine = mfuncs.ainstruct(
                instruction,
                context=SimpleContext(),
                backend=self.backend,
                requirements=[],
                strategy=RejectionSamplingStrategy(loop_budget=LOOP_BUDGET),
                return_sampling_results=True,
                user_variables={"context": context, "atom_text": atom_text},
                model_options={
                    "logprobs": True,
                    "top_logprobs": 5,
                },
            )
            coroutines.append(coroutine)

        results = []
        outputs = await asyncio.gather(*(coroutines[i] for i in range(len(coroutines))))
        for output in outputs:
            cleaned = str(output).strip()
            results.append(
                {
                    "context": context,
                    "summary": cleaned if cleaned != "None" else "",
                    "probability": self._get_probability(output.result),
                }
            )

        return results
