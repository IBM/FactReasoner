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
from typing import Any

import mellea.stdlib.functional as mfuncs
from mellea.backends import Backend
from mellea.core import MelleaLogger, ModelOutputThunk
from mellea.stdlib.context import SimpleContext
from mellea.stdlib.requirements import check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

# Local imports
from fact_reasoner.uncertainty import ProbabilisticClassifier, SIMBAUQSamplingStrategy
from fact_reasoner.utils import (
    extract_logprobs_from_output,
    extract_nli_label_and_span,
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
A JSON object of the form {"label": "entailment"}, {"label": "contradiction"} or {"label": "neutral"} is also an acceptable final answer.

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
        simbauq_temperatures: list[float] | None = None,
        simbauq_n_per_temp: int = 5,
        simbauq_similarity_metric: str = "rouge",
        simbauq_confidence_method: str = "aggregation",
        simbauq_aggregation: str = "mean",
        simbauq_classifier: ProbabilisticClassifier | None = None,
        simbauq_classifier_path: str | None = None,
        simbauq_training_samples: list[list[str]] | None = None,
        simbauq_training_labels: list[list[int]] | None = None,
        show_progress: bool = False,
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
            simbauq_classifier_path: str, optional
                Path to a classifier saved by
                ``fact_reasoner.uncertainty.save_classifier`` (see
                ``scripts/train_simbauq_nli.py``). When provided (and no explicit
                ``simbauq_classifier`` object is given), the classifier is loaded,
                its feature dimension is validated against
                ``len(temperatures) * n_per_temp - 1``, and the SIMBA-UQ
                confidence method is set to "classifier". Precedence:
                ``simbauq_classifier`` object > ``simbauq_classifier_path`` >
                ``simbauq_training_samples``/``simbauq_training_labels`` >
                the default "aggregation" method.
            show_progress: bool
                If True, ``run_batch`` shows a ``tqdm`` progress bar that
                advances as each NLI relation is resolved. Default False.
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
        self.show_progress = show_progress
        # Recorded for the preamble when a classifier is loaded from disk.
        self._classifier_path: str | None = None

        # Build the sampling strategy once. The SIMBA-UQ strategy is what makes
        # the probability estimate backend-agnostic (no logprobs required).
        if nli_method == "simbauq":
            confidence_method = simbauq_confidence_method
            classifier = simbauq_classifier

            # Load a saved classifier when a path is given and no in-memory
            # classifier object was passed. Loading it here (rather than in the
            # strategy) keeps the strategy free of I/O and lets us validate the
            # feature dimension against this extractor's temperature schedule.
            if classifier is None and simbauq_classifier_path is not None:
                classifier = self._load_simbauq_classifier(
                    simbauq_classifier_path,
                    temperatures=simbauq_temperatures,
                    n_per_temp=simbauq_n_per_temp,
                )
                confidence_method = "classifier"
                self._classifier_path = simbauq_classifier_path

            self._strategy = SIMBAUQSamplingStrategy(
                temperatures=simbauq_temperatures,
                n_per_temp=simbauq_n_per_temp,
                similarity_metric=simbauq_similarity_metric,
                confidence_method=confidence_method,
                aggregation=simbauq_aggregation,
                classifier=classifier,
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
        if self.method == "simbauq":
            self._print_simbauq_preamble()

        # Disable Mellea logging
        MelleaLogger.get_logger().setLevel(MelleaLogger.ERROR)

    @staticmethod
    def _load_simbauq_classifier(
        path: str,
        *,
        temperatures: list[float] | None,
        n_per_temp: int,
    ) -> ProbabilisticClassifier:
        """Load and validate a saved SIMBA-UQ classifier.

        Validates that the classifier's input feature dimension matches this
        extractor's configuration (``len(temperatures) * n_per_temp - 1``), so a
        classifier trained under a different temperature schedule fails fast with
        a clear error rather than at first inference.

        Args:
            path: Path to a classifier saved by
                ``fact_reasoner.uncertainty.save_classifier``.
            temperatures: The extractor's temperature schedule (None → the
                SIMBAUQSamplingStrategy default of [0.3, 0.5, 0.7, 1.0]).
            n_per_temp: Samples per temperature.

        Returns:
            The loaded classifier estimator.

        Raises:
            ValueError: If the classifier's feature dimension does not match.
        """
        # Imported here (not at module top) to avoid importing joblib/sklearn
        # unless a classifier is actually being loaded.
        from fact_reasoner.uncertainty import load_classifier

        clf, metadata = load_classifier(path)

        effective_temps = temperatures if temperatures is not None else [0.3, 0.5, 0.7, 1.0]
        expected_features = len(effective_temps) * n_per_temp - 1
        n_features = getattr(clf, "n_features_in_", metadata.get("n_features_in"))
        if n_features is not None and n_features != expected_features:
            raise ValueError(
                f"Classifier at {path!r} expects {n_features} features, but this "
                f"NLIExtractor configuration produces {expected_features} "
                "(len(temperatures) * n_per_temp - 1). Retrain the classifier with "
                "a matching temperature schedule / n_per_temp, or configure the "
                "extractor to match the classifier."
            )
        return clf

    def _print_simbauq_preamble(self) -> None:
        """Print a short summary of the active SIMBA-UQ strategy configuration.

        Reads the settings off the constructed strategy (rather than the
        constructor arguments) so the printout reflects the actual values in
        effect, including defaults filled in by SIMBAUQSamplingStrategy.
        """
        s = self._strategy
        total = len(s.temperatures) * s.n_per_temp

        # Metric-specific detail (e.g. the rouge variant or sbert model).
        if s.similarity_metric == "rouge":
            metric = f"{s.similarity_metric} ({s.rouge_type})"
        elif s.similarity_metric == "sbert":
            metric = f"{s.similarity_metric} ({s.sbert_model})"
        else:
            metric = s.similarity_metric

        # Confidence-method-specific detail.
        if s.confidence_method == "aggregation":
            confidence = f"{s.confidence_method} ({s.aggregation})"
        elif self._classifier_path is not None:
            confidence = f"{s.confidence_method} (loaded from {self._classifier_path})"
        else:
            confidence = f"{s.confidence_method} (max_depth={s.clf_max_depth})"

        print("[NLI] SIMBA-UQ strategy configuration:")
        print(f"[NLI]   temperatures       : {s.temperatures}")
        print(f"[NLI]   samples/temperature: {s.n_per_temp}")
        print(f"[NLI]   total samples      : {total}   (len(temperatures) * n_per_temp)")
        print(f"[NLI]   similarity metric  : {metric}")
        print(f"[NLI]   confidence method  : {confidence}")

    def _uses_logprobs(self) -> bool:
        """Whether the current method requires the backend to return logprobs."""
        return self.method == "logprobs"

    def _logprobs_model_options(self) -> dict[str, Any] | None:
        """Model options for the current method.

        The logprobs method must request logprobs from the backend; the
        SIMBA-UQ method must NOT (Ollama rejects the option, and SIMBA-UQ
        drives its own per-temperature model_options internally).
        """
        if self._uses_logprobs():
            return {"logprobs": True, "top_logprobs": 5}
        return None

    # Confidence used when the label tokens cannot be located in the logprobs
    # (empty logprobs, no ``[...]`` span, or no overlapping tokens). 0.5 signals
    # "unknown confidence"; returning 0.0 would be read downstream as a
    # degenerate/impossible relation for a label the model actually generated.
    _UNKNOWN_PROBABILITY = 0.5

    def _get_probability(self, output: ModelOutputThunk) -> float:
        """
        Estimate the probability of the predicted NLI label from token logprobs.

        Aligns the token-level logprobs to the **same** label span that
        ``_get_label`` extracts — the JSON ``"label": "<value>"`` value for JSON
        output, or the ``[...]`` interior for bracket output (see
        ``extract_nli_label_and_span``) — then returns the per-token geometric
        mean probability (``exp(mean(logprob))``) of the tokens covering that
        span.

        This is robust to subword tokenization: it does not require standalone
        delimiter tokens (``"["`` / ``"]"`` / ``'"'`` are frequently fused into
        subwords), and it measures exactly the label the label path reports, so
        the two can never disagree.

        Args:
            output: ModelOutputThunk
                The model raw output (via Mellea).

        Returns:
            float: The label probability in ``(0, 1]``, or ``0.5`` when the label
            tokens cannot be located (see ``_UNKNOWN_PROBABILITY``).
        """
        logprobs = extract_logprobs_from_output(output)
        if not logprobs:
            print("[NLI] No logprobs available; using default label probability.")
            return self._UNKNOWN_PROBABILITY

        # Reconstruct the decoded text from the token strings, tracking each
        # token's [start, end) character offset in that reconstruction.
        spans = []  # (start, end, logprob) per token
        pos = 0
        for item in logprobs:
            tok = str(item["token"])
            spans.append((pos, pos + len(tok), item["logprob"]))
            pos += len(tok)
        text = "".join(str(item["token"]) for item in logprobs)

        # Locate the label text span (JSON value or bracket interior) — the same
        # span the label extraction uses.
        _, span = extract_nli_label_and_span(text)
        if span is None:
            print("[NLI] No label span in logprobs; using default probability.")
            return self._UNKNOWN_PROBABILITY
        span_start, span_end = span

        # Average the logprobs of every token overlapping the label span.
        label_logprobs = [
            lp for (t0, t1, lp) in spans if t1 > span_start and t0 < span_end
        ]
        if not label_logprobs:
            print("[NLI] Could not align label tokens; using default probability.")
            return self._UNKNOWN_PROBABILITY

        avg_logprob = sum(label_logprobs) / len(label_logprobs)
        return math.exp(avg_logprob)

    @staticmethod
    def _get_simbauq_confidence(output: ModelOutputThunk) -> float | None:
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

        Auto-detects both supported output formats: a JSON verdict
        ``{"label": "..."}`` and a bracketed label ``[...]`` (see
        ``extract_nli_label_and_span``). The label is lower-cased so matching in
        ``_parse_output`` is case-insensitive.

        Args:
            output: ModelOutputThunk
                The model raw output (via Mellea)

        Returns:
            str: The string representing the NLI label (entailment, contradiction, neutral).
        """
        label, _ = extract_nli_label_and_span(str(output))
        return label

    def run(self, premise: str, hypothesis: str) -> dict[str, Any]:
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
                        "The output must contain an NLI label, either as a JSON "
                        'object {"label": "..."} or wrapped in square brackets.',
                        validation_fn=simple_validate(
                            lambda s: extract_nli_label_and_span(s)[0] != ""
                        ),
                    )
                ],
                user_variables={"premise_text": premise, "hypothesis_text": hypothesis},
                strategy=self._strategy,
                return_sampling_results=True,
                model_options=self._logprobs_model_options(),
            )
        except Exception as e:  # noqa: BLE001
            print(f"[NLI] Generation failed: {e}")
            return self._fallback()

        return self._parse_output(output)

    @staticmethod
    def _fallback() -> dict[str, Any]:
        """Neutral relationship used when generation or parsing fails."""
        return {"label": "neutral", "probability": 1.0}

    def _parse_output(self, output: Any) -> dict[str, Any]:
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
        except Exception as e:  # noqa: BLE001
            print(f"[NLI] Failed to parse output: {e}")
            return self._fallback()

        if label not in ["entailment", "contradiction", "neutral"]:
            label = "neutral"
        return {"label": label, "probability": probability}

    async def run_batch(
        self, premises: list[str], hypotheses: list[str]
    ) -> list[dict[str, Any]]:
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
                        "The output must contain an NLI label, either as a JSON "
                        'object {"label": "..."} or wrapped in square brackets.',
                        validation_fn=simple_validate(
                            lambda s: extract_nli_label_and_span(s)[0] != ""
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

        # Optional progress bar, advanced from run_throttled's per-completion
        # callback so it ticks as each relation is resolved (not all at once).
        bar = None
        on_progress = None
        if self.show_progress and pairs:
            from tqdm import tqdm

            bar = tqdm(total=len(pairs), desc="NLI relations", unit="rel")
            on_progress = bar.update  # update() advances by 1 when called with no args
        try:
            outputs = await run_throttled(factory, pairs, on_progress=on_progress)
        finally:
            if bar is not None:
                bar.close()

        # Results are positionally aligned with the input pairs; failures map to
        # a neutral relationship so callers can index result[i].
        results: list[dict[str, Any]] = []
        for output in outputs:
            if isinstance(output, Exception):
                print(f"[NLI] Batch item failed: {output}")
                results.append(self._fallback())
                continue
            results.append(self._parse_output(output))

        return results
