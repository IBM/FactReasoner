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

import math
import mellea.stdlib.functional as mfuncs

from typing import Any, Dict, List, Optional

from mellea.backends import Backend
from mellea.stdlib.context import SimpleContext
from mellea.core import ModelOutputThunk
from mellea.stdlib.requirements import check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.core import MelleaLogger

# Local imports
from fact_reasoner.uncertainty import ProbabilisticClassifier, SIMBAUQSamplingStrategy
from fact_reasoner.utils import (
    extract_last_square_brackets,
    extract_logprobs_from_output,
    run_throttled,
)

# Supported methods for estimating the NLI relationship probability.
NLI_METHODS = ("logprobs", "simbauq")

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
- [entailment] if the PREMISE strongly implies, directly supports or entails the HYPOTHESIS
- [contradiction] if the PREMISE contradicts the HYPOTHESIS
- [neutral] if the PREMISE and the HYPOTHESIS neither entail nor contradict each other

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
        nli_method: str = "logprobs",
        *,
        simbauq_temperatures: Optional[List[float]] = None,
        simbauq_n_per_temp: int = 4,
        simbauq_similarity_metric: str = "rouge",
        simbauq_confidence_method: str = "aggregation",
        simbauq_aggregation: str = "mean",
        simbauq_classifier: Optional[ProbabilisticClassifier] = None,
        simbauq_training_samples: Optional[List[List[str]]] = None,
        simbauq_training_labels: Optional[List[List[int]]] = None,
    ):
        """
        Initialize the NLIExtractor.

        Args:
            backend: Backend
                The Mellea backend to use for LLM interaction.
            nli_method: str
                How to estimate the probability of the predicted NLI label.
                - "logprobs" (default): derive the probability from the token
                  logprobs of the generated label. Requires a backend that
                  exposes logprobs (RITS / vLLM); does NOT work with Ollama.
                - "simbauq": estimate the probability via SIMBA-UQ
                  self-consistency (samples across temperatures and scores by
                  consensus). Backend-agnostic; use this for Ollama.
            simbauq_*:
                SIMBA-UQ configuration, only used when nli_method="simbauq".
                See SIMBAUQSamplingStrategy for details.
        """

        # Safety checks
        if backend is None:
            raise ValueError(
                "Mellea backend is None. Please provide a valid Mellea backend."
            )
        if nli_method not in NLI_METHODS:
            raise ValueError(
                f"Unknown nli_method: {nli_method!r} (expected one of {list(NLI_METHODS)})."
            )

        self.method = nli_method
        self.backend = backend

        # Build the sampling strategy once. The SIMBA-UQ strategy is what makes
        # the probability estimate backend-agnostic (no logprobs required).
        if nli_method == "simbauq":
            self._strategy = SIMBAUQSamplingStrategy(
                temperatures=simbauq_temperatures,
                n_per_temp=simbauq_n_per_temp,
                similarity_metric=simbauq_similarity_metric,
                confidence_method=simbauq_confidence_method,
                aggregation=simbauq_aggregation,
                classifier=simbauq_classifier,
                training_samples=simbauq_training_samples,
                training_labels=simbauq_training_labels,
            )
        else:
            self._strategy = RejectionSamplingStrategy(loop_budget=3)

        # Print info
        print(
            f"[NLI] Using Mellea backend: {self.backend.model_id} "
            f"(method: {self.method})"
        )

        # Disable Mellea logging
        MelleaLogger.get_logger().setLevel(MelleaLogger.ERROR)

    def _uses_logprobs(self) -> bool:
        """Whether the current method requires the backend to return logprobs."""
        return self.method == "logprobs"

    def _logprobs_model_options(self) -> Optional[Dict[str, Any]]:
        """Model options for the current method.

        The logprobs method must request logprobs from the backend; the
        SIMBA-UQ method must NOT (Ollama rejects the option, and SIMBA-UQ
        drives its own per-temperature model_options internally).
        """
        if self._uses_logprobs():
            return {"logprobs": True, "top_logprobs": 5}
        return None

    def _get_probability(self, output: ModelOutputThunk) -> float:
        """
        Compute the average log probability of the generated tokens.

        Args:
            output: ModelOutputThunk
                The model raw output (via Mellea).

        Returns:
            float: The average log probability of the generated tokens.
        """
        logprobs = extract_logprobs_from_output(output)

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

    @staticmethod
    def _get_simbauq_confidence(output: ModelOutputThunk) -> Optional[float]:
        """
        Read the SIMBA-UQ confidence of the selected sample.

        The SIMBA-UQ sampling strategy stores its metadata on the winning
        thunk's ``_meta`` dict under the ``"simba_uq"`` key. The confidence is
        the probability of the predicted label. Returns None in the degraded
        single-sample case (where SIMBA-UQ cannot estimate a confidence).

        Args:
            output: ModelOutputThunk
                The model raw output (via Mellea).

        Returns:
            Optional[float]: The SIMBA-UQ confidence in [0, 1], or None.
        """
        meta = getattr(output, "_meta", None) or {}
        simba_uq = meta.get("simba_uq", {})
        return simba_uq.get("confidence")

    def _get_label(self, output: ModelOutputThunk) -> str:
        """
        Extract the NLI label from the model output.

        Args:
            output: ModelOutputThunk
                The model raw output (via Mellea)

        Returns:
            str: The string representing the NLI label (entailment, contradiction, neutral).
        """

        # Normalize to lowercase so label matching in _parse_output is
        # case-insensitive (the LLM may emit e.g. "[Entailment]").
        return extract_last_square_brackets(str(output)).lower()

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

        # Perform the instruction with validation. A backend/network error is
        # raised out of mfuncs.instruct (validation failures instead come back
        # as a result with success=False), so guard the whole generation.
        try:
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
                strategy=self._strategy,
                return_sampling_results=True,
                model_options=self._logprobs_model_options(),
            )
        except Exception as e:
            print(f"[NLI] Generation failed: {e}")
            return self._fallback()

        return self._parse_output(output)

    @staticmethod
    def _fallback() -> Dict[str, Any]:
        """Neutral relationship used when generation or parsing fails."""
        return dict(label="neutral", probability=1.0)

    def _parse_output(self, output: Any) -> Dict[str, Any]:
        """Map a single sampling result to a label/probability dict.

        Any failure (unsuccessful sampling or an error while extracting the
        label/probability) falls back to a neutral relationship.
        """
        if not getattr(output, "success", False):
            return self._fallback()
        try:
            label = self._get_label(output.result)
            if self.method == "simbauq":
                # The winning sample's label is the predicted NLI label, and its
                # SIMBA-UQ confidence is the probability of that label.
                confidence = self._get_simbauq_confidence(output.result)
                if confidence is None:
                    # Degraded single-sample case: no reliable confidence.
                    return self._fallback()
                probability = float(confidence)
            else:
                probability = self._get_probability(output.result)
        except Exception as e:
            print(f"[NLI] Failed to parse output: {e}")
            return self._fallback()

        if label not in ["entailment", "contradiction", "neutral"]:
            label = "neutral"
        return dict(label=label, probability=probability)

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

        # Build a fresh coroutine per (premise, hypothesis) pair. run_throttled
        # applies bounded concurrency plus a per-minute rate limit, and captures
        # per-item exceptions so a single backend failure does not drop the rest.
        def factory(pair):
            premise, hypothesis = pair
            return mfuncs.ainstruct(
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
                strategy=self._strategy,
                return_sampling_results=True,
                model_options=self._logprobs_model_options(),
            )

        pairs = list(zip(premises, hypotheses))
        print(f"[NLI] Running throttled batch of {len(pairs)} requests ...")
        outputs = await run_throttled(factory, pairs)

        # Results are positionally aligned with the input pairs; failures map to
        # a neutral relationship so callers can index result[i].
        results: List[Dict[str, Any]] = []
        for output in outputs:
            if isinstance(output, Exception):
                print(f"[NLI] Batch item failed: {output}")
                results.append(self._fallback())
                continue
            results.append(self._parse_output(output))

        return results
