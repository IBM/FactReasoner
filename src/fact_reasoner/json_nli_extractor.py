from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import traceback
from typing import Literal, cast
import regex

from itertools import batched
from litellm.types.utils import Choices, ChatCompletionTokenLogprob, ModelResponse
from tqdm.auto import tqdm
from scipy.special import logsumexp

from src.fact_reasoner.llm_handler import LLMHandler
from src.fact_reasoner.utils import RITS_MODELS

EXAMPLE_INPUT = """```json
{
  "The company hired three new software engineers this month.": [
    {
      "hypothesis": "The company did not hire any new employees."
    },
    {
      "hypothesis": "The director of the company recently announced a new expansion."
    }
  ],
  "Sarah bought a new book and has been reading it every night.": [
    {
      "hypothesis": "Sarah reads her new book in the evenings."
    },
    {
      "hypothesis": "Sarah can recognise different letters."
    }
  ],
  "The museum is open from 9 AM to 5 PM on weekdays.": [
    {
      "hypothesis": "The museum is open until 5 PM on Saturdays."
    },
    {
      "hypothesis": "The museum gets more visitors on weekends."
    },
    {
      "hypothesis": "Museum visitors especially appreciate the interactive exhibition on space exploration."
    }
  ],
  "The company announced a new product line featuring eco-friendly materials in their last press release.": [
    {
      "hypothesis": "The company has expanded its product offerings with a focus on sustainability."
    },
    {
      "hypothesis": "The company has achieved net-zero carbon emissions."
    },
    {
      "hypothesis": "New information about the company was released last week."
    }
  ],
  "The AI symposium was canceled due to the severe storm that hit Tokyo.": [
    {
      "hypothesis": "The weather is usually nice in Tokyo."
    },
    {
      "hypothesis": "No disruption was experienced in Tokyo."
    },
    {
      "hypothesis": "Tokyo is a major hub for scientific and technological research."
    }
  ],
  "The AI symposium in Tokyo lasted for three days and involved over 50 sessions.": [
    {
      "hypothesis": "At least 50 speakers attended the AI symposium in Tokyo."
    },
    {
      "hypothesis": "The AI symposium in Tokyo didn't take place due to weather."
    }
  ]
}
```"""

EXAMPLE_OUTPUT = """```json
{
  "The company hired three new software engineers this month.": [
    {
      "hypothesis": "The company did not hire any new employees.",
      "label": "contradiction"
    },
    {
      "hypothesis": "The director of the company recently announced a new expansion.",
      "label": "neutral"
    }
  ],
  "Sarah bought a new book and has been reading it every night.": [
    {
      "hypothesis": "Sarah reads her new book in the evenings.",
      "label": "entailment"
    },
    {
      "hypothesis": "Sarah can recognise different letters.",
      "label": "entailment"
    }
  ],
  "The museum is open from 9 AM to 5 PM on weekdays.": [
    {
      "hypothesis": "The museum is open until 5 PM on Saturdays.",
      "label": "neutral"
    },
    {
      "hypothesis": "The museum gets more visitors on weekends.",
      "label": "neutral"
    },
    {
      "hypothesis": "Museum visitors especially appreciate the interactive exhibition on space exploration.",
      "label": "neutral"
    }
  ],
  "The company announced a new product line featuring eco-friendly materials in their last press release.": [
    {
      "hypothesis": "The company has expanded its product offerings with a focus on sustainability.",
      "label": "entailment"
    },
    {
      "hypothesis": "The company has achieved net-zero carbon emissions.",
      "label": "neutral"
    },
    {
      "hypothesis": "New information about the company was released last week.",
      "label": "neutral"
    }
  ],
  "The AI symposium was canceled due to the severe storm that hit Tokyo.": [
    {
      "hypothesis": "The weather is usually nice in Tokyo.",
      "label": "neutral"
    },
    {
      "hypothesis": "No disruption was experienced in Tokyo.",
      "label": "contradiction"
    },
    {
      "hypothesis": "Tokyo is a major hub for scientific and technological research.",
      "label": "neutral"
    }
  ],
  "The AI symposium in Tokyo lasted for three days and involved over 50 sessions.": [
    {
      "hypothesis": "At least 50 speakers attended the AI symposium in Tokyo.",
      "label": "neutral"
    },
    {
      "hypothesis": "The AI symposium in Tokyo didn't take place due to weather.",
      "label": "contradiction"
    }
  ]
}
```"""

EXAMPLE_OUTPUT_REASONING = """```json
{
  "The company hired three new software engineers this month.": [
    {
      "hypothesis": "The company did not hire any new employees.",
      "reasoning": "The premise explicitly states that three new employees were hired, contradicting the hypothesis.",
      "label": "contradiction"
    },
    {
      "hypothesis": "The director of the company recently announced a new expansion.",
      "reasoning": "The premise does not mention any announcement of a new expansion, so the hypothesis introduces unrelated information.",
      "label": "neutral"
    }
  ],
  "Sarah bought a new book and has been reading it every night.": [
    {
      "hypothesis": "Sarah reads her new book in the evenings.",
      "reasoning": "Reading every night implies reading in the evenings, so the hypothesis logically follows from the premise.",
      "label": "entailment"
    },
    {
      "hypothesis": "Sarah can recognise different letters.",
      "reasoning": "Reading a book requires the ability to recognize letters, so the hypothesis necessarily follows from the premise.",
      "label": "entailment"
    }
  ],
  "The museum is open from 9 AM to 5 PM on weekdays.": [
    {
      "hypothesis": "The museum is open until 5 PM on Saturdays.",
      "reasoning": "The premise only specifies weekday hours, while no information is provided about opening hours on Saturdays.",
      "label": "neutral"
    },
    {
      "hypothesis": "The museum gets more visitors on weekends.",
      "reasoning": "Visitor numbers are not mentioned in the premise, so the hypothesis is neither confirmed nor contradicted.",
      "label": "neutral"
    },
    {
      "hypothesis": "Museum visitors especially appreciate the interactive exhibition on space exploration.",
      "reasoning": "The premise does not mention any exhibitions or visitor preferences.",
      "label": "neutral"
    }
  ],
  "The company announced a new product line featuring eco-friendly materials in their last press release.": [
    {
      "hypothesis": "The company has expanded its product offerings with a focus on sustainability.",
      "reasoning": "A new product line with eco-friendly materials implies expansion with a sustainability focus.",
      "label": "entailment"
    },
    {
      "hypothesis": "The company has achieved net-zero carbon emissions.",
      "reasoning": "The premise mentions eco-friendly materials, but does not provde information about overall carbon emissions.",
      "label": "neutral"
    },
    {
      "hypothesis": "New information about the company was released last week.",
      "reasoning": "The premise mentions a press release but does not specify the timing as last week.",
      "label": "neutral"
    }
  ],
  "The AI symposium was canceled due to the severe storm that hit Tokyo.": [
    {
      "hypothesis": "The weather is usually nice in Tokyo.",
      "reasoning": "The premise discusses a specific storm, not general weather patterns.",
      "label": "neutral"
    },
    {
      "hypothesis": "No disruption was experienced in Tokyo.",
      "reasoning": "The cancellation of the symposium due to a storm is a form of disruption, so the hypothesis is contradicted by the premise.",
      "label": "contradiction"
    },
    {
      "hypothesis": "Tokyo is a major hub for scientific and technological research.",
      "reasoning": "The premise does not address Tokyo's general status as a research hub, only a single event.",
      "label": "neutral"
    }
  ],
  "The AI symposium in Tokyo lasted for three days and involved over 50 sessions.": [
    {
      "hypothesis": "At least 50 speakers attended the AI symposium in Tokyo.",
      "reasoning": "Multiple sessions may have been delivered by the same speaker, so the premise does not necessarily imply the number of speakers.",
      "label": "neutral"
    },
    {
      "hypothesis": "The AI symposium in Tokyo didn't take place due to weather.",
      "reasoning": "The premise confirms the symposium occurred, contradicting the claim that it didn't take place.",
      "label": "contradiction"
    }
  ]
}
```"""

NLI_EXTRACTION_PROMPT = """{prompt_start}You will be provided with a JSON file with a \
set of premises, each followed by one or more associated hypotheses. Your task is to \
determine the type of their relationship. Specifically, you should classify each pair \
as one of the following:
* Entailment — the hypothesis logically follows from the premise.
* Contradiction — the hypothesis is logically inconsistent with the premise.
* Neutral — the hypothesis is neither entailed nor contradicted by the premise.
You should consider each pair separately, without reference to any other hypotheses \
or premises in the same batch.

For each premise and its corresponding hypotheses, output your classification using \
one of the three possible labels: "entailment", "contradiction" or "neutral". \
All classifications should be added as "label" fields to the corresponding hypotheses. \
{reasoning_prompt}Do not include any additional commentary or explanation, just the output \
JSON in the specified format. Use the following example to guide you:

Example input:
{example_input}

Example output:
{example_output}

Now, please consider each premise-hypothesis pair below and provide your classifications \
in the JSON format illustrated above. Remember to consider each premise-hypothesis pair \
in isolation.

Input:
{input_json}
{prompt_end}"""

REASONING_PROMPT = """Before you predict each label, please provide concise reasoning \
for your classification in the "reasoning" field."""

VALID_LABELS = {"entailment", "contradiction", "neutral"}
type NliLabel = Literal["entailment", "contradiction", "neutral"]


@dataclass
class NliExtractionResult:
    hypothesis: str
    label: NliLabel
    probability: float
    reasoning: str


class JsonNliExtractor:
    """
    Extracts NLI relations using LLMs with structured JSON outputs,
    optionally using short reasoning.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-instruct",
        use_logprobs: bool = True,
        use_reasoning: bool = False,
    ):
        """
        Initializes the JsonNliExtractor.
        
        Args:
            model: str
                The model to use for extracting the relations.
            use_logprobs: bool
                Whether to extract probabilistic labels using model logprobs.
            use_reasoning: bool
                Whether to use brief reasoning before predicting each relation.
        """
        rits_model_info = RITS_MODELS[model]
        self.prompt_begin = rits_model_info.get("prompt_begin")
        self.prompt_end = rits_model_info.get("prompt_end")
        self.llm_handler = LLMHandler(model, RITS=True)
        self.use_logprobs = use_logprobs
        self.use_reasoning = use_reasoning

        print(f"[BatchNliExtractor] Using LLM on RITS: {model}")

    def make_prompt(self, premise_hypotheses_mapping: dict[str, list[str]]) -> str:
        """
        Constructs a prompt for extracting NLI relations.
        
        Args:
            premise_hypotheses_mapping: dict[str, list[str]]
                A mapping from premises to their associated hypotheses.
        
        Returns:
            str:
                The constructed prompt.
        """
        input_json = json.dumps(
            {
                p: [{"hypothesis": h} for h in hs]
                for p, hs in premise_hypotheses_mapping.items()
            },
            indent=2,
        )
        if self.use_reasoning:
            return NLI_EXTRACTION_PROMPT.format(
                prompt_start=self.prompt_begin,
                prompt_end=self.prompt_end,
                example_input=EXAMPLE_INPUT,
                example_output=EXAMPLE_OUTPUT_REASONING,
                reasoning_prompt=REASONING_PROMPT,
                input_json=input_json,
            )

        return NLI_EXTRACTION_PROMPT.format(
            prompt_start=self.prompt_begin,
            prompt_end=self.prompt_end,
            example_input=EXAMPLE_INPUT,
            example_output=EXAMPLE_OUTPUT,
            reasoning_prompt="",
            input_json=input_json,
        )

    def _extract_probabilities(
        self, response: ModelResponse
    ) -> list[tuple[str, float]]:
        """
        Extracts probabilities for the NLI predictions in the model response.
        
        Args:
            response: ModelResponse
                The model response from which to extract the probabilities.
                
        Returns:
            list[tuple[str, float]]:
                A list of tuples of hypothesis strings and their NLI prediction
                probabilities.
        """
        all_logprobs = cast(
            list[ChatCompletionTokenLogprob],
            response.choices[0].logprobs.content,  # type: ignore
        )
        token_spans: list[tuple[int, int]] = []
        current_idx = 0
        for logprob in all_logprobs:
            token_spans.append((current_idx, current_idx + len(logprob.token)))
            current_idx += len(logprob.token)
        label_regex = r"\"hypothesis\": \"([^\"]+)\",[\s\S]+?\"label\":\s\"(neutral|entailment|contradiction)\""
        label_matches = list(
            regex.finditer(label_regex, response.choices[0].message.content)  # type: ignore
        )
        results: list[tuple[str, float]] = []
        for label_match in label_matches:
            hypothesis = label_match.group(1)
            prediction = label_match.group(2)
            start_idx, end_idx = label_match.span(2)
            label_logprobs = [
                logprob
                for logprob, (s, e) in zip(all_logprobs, token_spans)
                if not (start_idx >= e or end_idx <= s)
            ]

            assert len(label_logprobs) > 0, "No logprob tokens found for label."
            # The first logprob is sufficient for determining the label
            label_logprob = label_logprobs[0]
            log_prediction_proba = -math.inf
            log_total_proba = -math.inf
            for top_logprob in label_logprob.top_logprobs:
                normalised_candidate = top_logprob.token.strip().lower()
                if any(
                    label.startswith(normalised_candidate) for label in VALID_LABELS
                ):
                    # Only consider valid outputs
                    if prediction.startswith(normalised_candidate):
                        log_prediction_proba = logsumexp(
                            [log_prediction_proba, top_logprob.logprob]
                        )
                    log_total_proba = logsumexp([log_total_proba, top_logprob.logprob])
            results.append(
                (hypothesis, math.exp(log_prediction_proba - log_total_proba))  # type: ignore
            )
        return results

    def extract_result(
        self, response: ModelResponse, premise_hypotheses_mapping: dict[str, list[str]]
    ) -> dict[str, dict[str, NliExtractionResult]]:
        """
        Extract NLI predictions from a model response.
        
        Args:
            response: ModelResponse
                The model response from which to extract the predictions.
            premise_hypotheses_mapping: dict[str, list[str]]
                A mapping from premises to their associated hypotheses.
        
        Returns:
            dict[str, dict[str, NliExtractionResult]]:
                A nested dictionary mapping premises and hypotheses to their
                NLI predictions.
        """
        choice = response.choices[0]
        if not isinstance(choice, Choices):
            raise ValueError(
                f"Invalid response: should be Choices, but got {type(choice)}"
            )
        content = choice.message.content
        if content is None:
            raise ValueError("Invalid response: content is None")
        content = content.replace("```json", "").replace("```", "").strip()

        try:
            raw_nli_predictions = cast(
                dict[str, list[dict[str, str]]], json.loads(content)
            )
        except Exception:
            raise ValueError(f"Invalid response: cannot parse JSON from {content}")

        # Process output and verify validity
        nli_predictions: dict[str, dict[str, NliExtractionResult]] = {}
        for p, hs in raw_nli_predictions.items():
            if p not in premise_hypotheses_mapping:
                print(premise_hypotheses_mapping)
                print(content)
                raise ValueError(f"Invalid response: invalid premise {p}.")
            if not all("hypothesis" in h and "label" in h for h in hs):
                print(premise_hypotheses_mapping)
                print(content)
                raise ValueError(
                    f"Invalid response: missing fields in one of the hypotheses {hs}"
                )
            nli_predictions[p] = {}
            expected_hypotheses = premise_hypotheses_mapping[p]
            actual_hypotheses = [h["hypothesis"] for h in hs]
            if actual_hypotheses != expected_hypotheses:
                print(premise_hypotheses_mapping)
                print(content)
                raise ValueError(
                    f"Invalid response: output hypotheses {actual_hypotheses} don't match "
                    f"input hypotheses {expected_hypotheses}."
                )
            labels = set(h["label"] for h in hs)
            if not labels.issubset(VALID_LABELS):
                raise ValueError(f"Invalid response: invalid labels {labels}.")
            for h in hs:
                nli_predictions[p][h["hypothesis"]] = NliExtractionResult(
                    hypothesis=h["hypothesis"],
                    label=cast(NliLabel, h["label"]),
                    probability=1.0,
                    reasoning=h["reasoning"] if "reasoning" in h else "",
                )

        nli_predictions_copy = deepcopy(nli_predictions)
        if self.use_logprobs:
            try:
                probabilities = self._extract_probabilities(response)
                probabilities_idx = 0
                for p, hs in nli_predictions_copy.items():
                    for h, nli_result in hs.items():
                        if probabilities_idx >= len(probabilities):
                            raise ValueError(
                                f"Logprob results length mismatch (only got {len(probabilities)} elements)"
                            )
                        hypothesis, probability = probabilities[probabilities_idx]
                        if hypothesis != h:
                            raise ValueError(
                                f"Logprob hypothesis mismatch (expected {h}, but got {hypothesis})"
                            )
                        nli_result.probability = probability
                        probabilities_idx += 1
                nli_predictions = nli_predictions_copy
            except Exception as e:
                print(
                    "[BatchNliExtractor] Failed to compute label probabilities with logprobs. "
                    "Falling back to standard predictions."
                )
                traceback.print_exception(e)

        return nli_predictions

    def run(
        self,
        premise_hypotheses_mapping: dict[str, list[str]],
        max_batch_relations: int = 64,
        max_batched_completions: int = 16,
    ) -> dict[str, dict[str, NliExtractionResult]]:
        """
        Predicts NLI relations for a set of premises and hypotheses.
        
        Args:
            premise_hypotheses_mapping: dict[str, list[str]]
                A mapping of premises to a list of the associated
                hypotheses.
            max_batch_relations: int
                Maximum number of NLI relations to predict in each
                LLM request. Hypothesis-premise pairs will be split
                into requsts according to this number.
            max_batched_completions: int
                The maximum number of parallel model calls.
        """
        # Construct batches of premise-hypotheses mappings according
        # to max_batch_relations.
        input_batches: list[dict[str, list[str]]] = [defaultdict(list)]
        current_count = 0
        for p, hs in premise_hypotheses_mapping.items():
            for h in hs:
                if current_count == max_batch_relations:
                    input_batches.append(defaultdict(list))
                    current_count = 0
                input_batches[-1][p].append(h)
                current_count += 1
        if current_count == 0:
            return {}
        prompts = [self.make_prompt(batch) for batch in input_batches]
        print(f"[BatchNliExtractor] Prompt(s) created: {len(prompts)}")

        # Generate for all batches
        all_results: dict[str, dict[str, NliExtractionResult]] = {}
        for prompt_batch, input_batch in tqdm(
            zip(
                batched(prompts, n=max_batched_completions),
                batched(input_batches, n=max_batched_completions),
            ),
            total=math.ceil(len(input_batches) / max_batched_completions),
        ):
            responses = self.llm_handler.batch_completion(
                prompt_batch,
                max_tokens=9000,
                seed=42,
                logprobs=self.use_logprobs,
                top_logprobs=20,
            )
            for response, batch_input_mapping in zip(responses, input_batch):
                if not isinstance(response, ModelResponse):
                    raise ValueError(
                        f"Unexpected type for model response: {type(response)}"
                    )
                result = self.extract_result(response, batch_input_mapping)
                for p, h_results in result.items():
                    if p not in all_results:
                        all_results[p] = {}
                    for h, nli_result in h_results.items():
                        all_results[p][h] = nli_result
        return all_results
