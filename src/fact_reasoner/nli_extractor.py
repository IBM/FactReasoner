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

# NLI scorer with Prompting (llama-3-70b-instruct)

import operator
import math
import numpy as np

from itertools import batched
from typing import List
from tqdm import tqdm
from difflib import SequenceMatcher
from litellm.types.utils import ModelResponse
from operator import itemgetter
from src.fact_reasoner.llm_handler import LLMHandler

from src.fact_reasoner.utils import (
    RITS_MODELS, 
    DEFAULT_PROMPT_BEGIN, 
    DEFAULT_PROMPT_END, 
    dotdict, 
    extract_last_square_brackets
)

NLI_LABELS = ['entailment', 'contradiction', 'neutral']

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def get_label_probability(samples: list, labels: list):
    candidates = []
    for label in labels:
        distances = [similarity(label, sample) for sample in samples]
        candidates.append((label, np.average(distances)))
    candidates = sorted(candidates, key=itemgetter(1), reverse=True)
    return candidates[0]

def reverse_enum(L):
   for index in reversed(range(len(L))):
      yield index, L[index]

# v1
NLI_EXTRACTION_PROMPT1 = """{_PROMPT_BEGIN_PLACEHOLDER}

Instructions:
1. You are given a premise and a hypothesis. Your task is to identify the relationship \
between them: does the premise entail, contradict, or remain neutral toward the hypothesis?
2. Your only output must be one of: (entailment | contradiction | neutral) without any \
lead-in, sign-off, new lines or any other formatting.
3. Do not provide any explanation or rationale to your output.
4. Use the following examples to learn how to do this, and provide your output for the last \
example given.

Premise: The weather forecast said it will rain tomorrow.
Hypothesis: It will be sunny tomorrow.
Output: contradiction

Premise: The company hired three new software engineers this month.
Hypothesis: The company did not hire any new employees.
Output: contradiction

Premise: Sarah bought a new book and has been reading it every night.
Hypothesis: Sarah enjoys reading her new book in the evenings.
Output: entailment

Premise: The museum is open from 9 AM to 5 PM on weekdays.
Hypothesis: The museum is open until 6 PM on Saturdays.
Output: neutral

Premise: The company announced a new product line featuring eco-friendly materials in their \
latest press release.
Hypothesis: The company is expanding its product offerings with a focus on sustainability.
Output: Entailment

Premise: The event was canceled due to the severe storm that hit the city.
Hypothesis: The event went on as planned, with no major disruptions.
Output: Contradiction

Premise: The CEO of the tech company gave a keynote speech at the conference yesterday.
Hypothesis: The keynote speech was well-received by the audience.
Output: Neutral

Premise: {_PREMISE_PLACEHOLDER}
Hypothesis: {_HYPOTHESIS_PLACEHOLDER}
Output:{_PROMPT_END_PLACEHOLDER}
"""

# v2
NLI_EXTRACTION_PROMPT2 = """{_PROMPT_BEGIN_PLACEHOLDER}

Instructions:
You are provided with a PREMISE and a HYPOTHESIS. \
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

YOUR TASK:
PREMISE: {_PREMISE_PLACEHOLDER}
HYPOTHESIS: {_HYPOTHESIS_PLACEHOLDER}{_PROMPT_END_PLACEHOLDER}
"""

# v3
# TODO: we need a prompt specific to Google search results
NLI_EXTRACTION_PROMPT3_FEW_SHOTS = [
    {"claim": "Characters Lenny and Carl on The Simpsons are hearing but are depicted as close friends of the Simpsons family.", "search_result": "Title: Character Spotlight: Lenny Leonard and Carl Carlson (& Barflies)\nContent: Their friendship is a pretty singular aspect on the show -- save Bart and Milhouse (or to some degree, Mr. Burns and Smithers) -- they always ...\nLink: https://nohomers.net/forums/index.php?threads/character-spotlight-lenny-leonard-and-carl-carlson-barflies.23798/", "human_label": "Inconclusive"},
    {"claim": "The championship match of the FIFA World Cup 2026 will be hosted by the United States.", "search_result": "Title: World Cup 2026 | New York New Jersey to host final - FIFA\nContent: New York New Jersey Stadium has been confirmed as the location for the FIFA World Cup 26™ final on Sunday, 19 July 2026. The full match schedule for the ...\nLink: https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup/canadamexicousa2026/articles/new-york-new-jersey-stadium-host-world-cup-2026-final", "human_label": "Supported"},
    {"claim": "It is essential to understand the limitations of heating a dead battery to temporarily revive its function.", "search_result": "Title: Why do batteries come back to life if you let them rest?\nContent: By letting the battery rest, you give the reaction products a chance to dissipate. The higher the drain on the battery, the faster the products ...\nLink: https://electronics.howstuffworks.com/everyday-tech/question390.htm", "human_label": "Inconclusive"},
    {"claim": "Sarah and James were shot execution-style in the living room.", "search_result": "Title: By Taufik | There were string of armed robberies and free murders ...\nContent: handgun been in the execution-style murder but we ... quickly cleared the living room, went ...\nLink: https://www.facebook.com/Haru6789.cv/videos/watch-killer-siblings-season-3-episode-5-allridges/411826944038089/", "human_label": "Inconclusive"},
    {"claim": "Vikings used their longships to transport livestock.", "search_result": "Title: How did the Vikings transport animals on their ships? - Quora\nContent: The Vikings transported horses overseas in boats very similar to Viking longships, but with flat flooring built within the hulls, which allowed ...\nLink: https://www.quora.com/How-did-the-Vikings-transport-animals-on-their-ships", "human_label": "Contradicted"},
    {"claim": "Romário has scored a total of 92 international goals.", "search_result": "Title: Romário - Wikipedia\nContent: A prolific striker renowned for his clinical finishing, he scored over 700 goals and is one of the few players to score at least 100 goals for three clubs. He ...\nLink: https://en.wikipedia.org/wiki/Rom%C3%A1rio", "human_label": "Contradicted"},
    {"claim": "Utopia portrays a society that values education and learning.", "search_result": "Title: Utopia Education, Science, Philosophy Summary & Analysis\nContent: The Utopians believe that it is through education that the values and dispositions of citizens are molded. The success of the Utopian educational system is ...\nLink: https://www.sparknotes.com/philosophy/utopia/section10/", "human_label": "Supported"},
    {"claim": "The higher density of water can cause sound waves to be reflected or refracted differently.", "search_result": "Title: How does sound in air differ from sound in water?\nContent: Sounds in water and sounds in air that have the same pressures have very different intensities because the density of water is much greater than ...\nLink: https://dosits.org/science/sounds-in-the-sea/how-does-sound-in-air-differ-from-sound-in-water/", "human_label": "Supported"},
    {"claim": "Mount Katahdin is 6,288.2 feet (1,917.6 meters) tall.", "search_result": "Title: Mount Katahdin - Wikipedia\nContent: Mount Katahdin is the highest mountain in the U.S. state of Maine at 5,269 feet (1,606 m). Named Katahdin, which means \"Great Mountain\", by the Penobscot ...\nLink: https://en.wikipedia.org/wiki/Mount_Katahdin", "human_label": "Contradicted"}
]

NLI_EXTRACTION_PROMPT3 = """{_PROMPT_BEGIN_PLACEHOLDER}

You need to judge whether a claim is supported or contradicted by a Google search result, or whether there is no enough information to make the judgement. When doing the task, take into consideration whether the link of the search result is of a trustworthy source. Place your answer in square brackets.

Below are the definitions of the three categories:

Supported: A claim is supported by the search results if everything in the claim is supported and nothing is contradicted by the search results. There can be some search results that are not fully related to the claim.
Contradicted: A claim is contradicted by the search results if something in the claim is contradicted by some search results. There should be no search result that supports the same part.
Inconclusive: A claim is inconclusive based on the search results if:
- a part of a claim cannot be verified by the search results,
- a part of a claim is supported and contradicted by different pieces of evidence,
- the entity/person mentioned in the claim has no clear referent (e.g., "the approach", "Emily", "a book").

Here are some examples:

Claim: {}

{}

Your decision: [{}]

Claim: {}

{}

Your decision: [{}]

Claim: {}

{}

Your decision: [{}]

Claim: {}

{}

Your decision: [{}]

Claim: {}

{}

Your decision: [{}]

Claim: {}

{}

Your decision: [{}]

Claim: {}

{}

Your decision: [{}]

Claim: {}

{}

Your decision: [{}]

Claim: {}

{}

Your decision: [{}]

Your task:
Claim: {_CLAIM_PLACEHOLDER}

{_SEARCH_RESULTS_PLACEHOLDER}

Your decision:{_PROMPT_END_PLACEHOLDER}
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
            model: str = "llama-3.1-70b-instruct",
            method: str = "logprobs",
            prompt_version: str = "v1",
            debug: bool = False,
            is_bert: bool = False,
            RITS: bool = True
    ):
        self.model = model
        self.method = method
        self.prompt_version = prompt_version
        self.debug = debug
        self.is_bert = is_bert

        if not self.is_bert: # check if LLM based NLI extraction


            self.rits_model_info = RITS_MODELS[model] #only used to ger prompt begin/end
            self.prompt_begin = self.rits_model_info.get("prompt_begin", DEFAULT_PROMPT_BEGIN)
            self.prompt_end = self.rits_model_info.get("prompt_end", DEFAULT_PROMPT_END)
        
            self.llm_handler = LLMHandler(self.model, RITS=RITS)
            
            assert self.prompt_version in ["v1", "v2", "v3"], f"Unknown prompt version for NLIExtractor."

            print(f"[NLIExtractor] Using LLM on {RITS*'RITS'}{(not RITS)*'vLLM'}: {self.model}")
            print(f"[NLIExtractor] Prompt version: {self.prompt_version}")
            self.bert_model = None
        else:
            raise ValueError("BERT models are not supported anymore.")

    def make_prompt(self, premise: str, hypothesis: str) -> str:
        if self.prompt_version == "v1":
            prompt = NLI_EXTRACTION_PROMPT1.format(
                _PREMISE_PLACEHOLDER=premise,
                _HYPOTHESIS_PLACEHOLDER=hypothesis,
                _PROMPT_BEGIN_PLACEHOLDER=self.prompt_begin,
                _PROMPT_END_PLACEHOLDER=self.prompt_end
            )
        elif self.prompt_version == "v2":
            prompt = NLI_EXTRACTION_PROMPT2.format(
                _PREMISE_PLACEHOLDER=premise,
                _HYPOTHESIS_PLACEHOLDER=hypothesis,
                _PROMPT_BEGIN_PLACEHOLDER=self.prompt_begin,
                _PROMPT_END_PLACEHOLDER=self.prompt_end
            )
        elif self.prompt_version == "v3": # specific to Google search results (links)
            # Set the few-shots section
            few_shots_lst = []
            for dict_item in NLI_EXTRACTION_PROMPT3_FEW_SHOTS:
                claim = dict_item["claim"]
                search_result_str = dict_item["search_result"]
                human_label = dict_item["human_label"]
                few_shots_lst.extend([claim, search_result_str, human_label])

            prompt = NLI_EXTRACTION_PROMPT3.format(
                _CLAIM_PLACEHOLDER=hypothesis,
                _SEARCH_RESULTS_PLACEHOLDER=premise,
                _PROMPT_BEGIN_PLACEHOLDER=self.prompt_begin,
                _PROMPT_END_PLACEHOLDER=self.prompt_end,
                *few_shots_lst
            )
        return prompt

    def extract_relationship(self, text: str, logprobs: List[dict]):
        """Extract the relationship and probability. The relationship should be on
        the last line of the generated text and one of the following: 
            [entailment], [contradiction] or [neutral]. Anything
        else will be assumed to be neutral with probability 1. The probability
        of the relationship is the exp of the average logprob of the corresponding
        tokens.
        """
        if self.prompt_version == "v1":
            label = text.strip().lower()
            if label not in ['entailment', 'contradiction', 'neutral']:
                label = 'neutral' #'invalid_label'
                probability = 1.0
            else:
                logprob_sum = 0.0
                generated_tokens = logprobs[:-1]
                for token in generated_tokens: #last token is just <|eot_id|>
                    token = dotdict(token)
                    logprob_sum +=token.logprob

                probability = np.exp(logprob_sum/len(generated_tokens))
        elif self.prompt_version == "v2":
            label = extract_last_square_brackets(text).lower()
            probability = 1.0
            if len(label) == 0 or label not in ['entailment', 'contradiction', 'neutral']:
                label = 'neutral'
            else:
                # Look for the tokens corresponding to the label [label]
                logits = []
                for _, elem in reverse_enum(logprobs):
                    elem = dotdict(elem)
                    if elem.token in ['', '\n', ']']:
                        continue
                    if elem.token in ['[']:
                        break
                    logits.append(elem.logprob)

                if len(logits) > 0:
                    probability = np.exp(np.mean(logits))
        elif self.prompt_version == "v3":
            label = extract_last_square_brackets(text).lower()
            probability = 1.0
            if len(label) == 0 or label not in ['supported', 'contradicted', 'inconclusive']:
                label = 'neutral'
            else:
                # Look for the tokens corresponding to the label [label]
                logits = []
                for elem in reverse_enum(logprobs): # loop from the end
                    elem = dotdict(elem)
                    if elem.token in ['', '\n', ']']:
                        continue
                    if elem.token in ['[']:
                        break
                    logits.append(elem.logprob)

                if len(logits) > 0:
                    probability = np.exp(np.mean(logits))

                if label == "supported":
                    label = "entailment"
                elif label == "contradicted":
                    label = "contradiction"
                elif label == "inconclusive":
                    label = "neutral"

        return label, probability

    def extract_relationship_dict(self, response: dict):
        """
        The input is a dictionary: {'entailment': 0.9952232241630554, 
                                    'contradiction': 0.00199194741435349, 
                                    'neutral': 0.002784877549856901}
        """
        label = max(response.items(), key=operator.itemgetter(1))[0]
        probability = response[label]

        return label, probability


    def run(self, premise: str, hypothesis: str):

        if not self.is_bert: # check if LLM is used
            prompt = self.make_prompt(premise, hypothesis)
            print(f"[NLIExtractor] Prompt created ({len(prompt)}).")
            response = self.llm_handler.completion(
                prompt,
                logprobs=True,
                expect_content=True,
            )

            text = response.choices[0].message.content
            if self.debug:
                print(f"Generate response:\n{text}")
            logprobs = response.choices[0].logprobs['content']
            label, probability = self.extract_relationship(text, logprobs)
            result = {'label': label, 'probability': probability}
        else:
            print(f"[NLIExtractor] Extracting NLI relationship with BERT model.")
            response = self.bert_model.score(
                premise=premise, 
                hypothesis=hypothesis, 
                op1="mean", 
                op2="mean"
            )
            label, probability = self.extract_relationship_dict(response)
            result = {'label': label, 'probability': probability}

        return result

    def runall(
        self,
        premises: List[str],
        hypotheses: List[str],
        max_batched_completions: int = 64,
    ):
        if not self.is_bert:  # check if LLM is used
            generated_texts = []
            generated_logprobs = []
            prompts = [
                self.make_prompt(premise, hypothesis)
                for premise, hypothesis in zip(premises, hypotheses)
            ]
            print(f"[NLIExtractor] Prompts created: {len(prompts)}")

            for prompt_batch in tqdm(
                batched(prompts, n=max_batched_completions),
                total=math.ceil(len(prompts) / max_batched_completions),
            ):
                responses = self.llm_handler.batch_completion(
                    prompt_batch, logprobs=True, seed=12345, expect_content=True,
                )
                for response in responses:
                    generated_texts.append(response.choices[0].message.content)
                    generated_logprobs.append(response.choices[0].logprobs["content"])

            results = []
            for text, logprobs in zip(generated_texts, generated_logprobs):
                label, probability = self.extract_relationship(text, logprobs)
                results.append({"label": label, "probability": probability})
        else:  # BERT model is used
            print(
                f"[NLIExtractor] Extracting batched NLI relationships with BERT model: {len(premises)}"
            )
            results = []
            responses = self.bert_model.score_all(
                premises=premises, hypotheses=hypotheses, op1="mean", op2="mean"
            )
            for response in responses:
                label, probability = self.extract_relationship_dict(response)
                results.append({"label": label, "probability": probability})
        return results


class NLIExtractorOld:
    def __init__(
            self,
            model: str = "llama-3.1-70b-instruct",
            scoring_method: str = "logprobs",
            prompt_version: str = "v1",
            RITS=True
    ):
        self.model = model
        self.scoring_method = scoring_method
        self.prompt_version = prompt_version

        self.rits_model_info = RITS_MODELS[model] #only used to ger prompt begin/end
        self.prompt_begin = self.rits_model_info.get("prompt_begin", DEFAULT_PROMPT_BEGIN)
        self.prompt_end = self.rits_model_info.get("prompt_end", DEFAULT_PROMPT_END)
    
        self.llm_handler = LLMHandler(self.model, RITS=RITS)
        
        self.parameters = dict(
            max_new_tokens=1000,
            min_new_tokens=1,
            decoding_method="greedy",
        )

    def score(
            self, 
            premises: list,
            hypthotheses: list, 
            temperature: float = 1.0,
            n_trials: int = 10,
    ):
        assert self.scoring_method in ['temperature', 'logprobs', 'consistency'], f"invlalid scoring method"
        
        self.create_prompts(premises, hypthotheses)

        if self.scoring_method == 'temperature':
            self.n_trials = n_trials
            self.prompts = [item for sublist in [[prompt]*self.n_trials for prompt in self.prompts] for item in sublist]
            results = []
            for idx, response in tqdm(
                enumerate(
                    self.llm_handler.batch_completion(
                        self.prompts,
                        temperature=temperature    
                    )
                ),
                total=len(self.prompts),
                desc="NLI",
                unit="prompts",
                ):
                    results.append(response.choices[0].message.content.strip().lower())
                
            results_labels = []
            start_ind = 0  
            for ii in range(len(premises)):
                result = results[start_ind:start_ind+self.n_trials]
                start_ind+=self.n_trials

                res = {}
                for label in ['entailment', 'contradiction', 'neutral']:
                    res[label] = result.count(label)
                    
                results_labels.append(res)

            results_labels = [(max(x, key=x.get),x[max(x, key=x.get)]/self.n_trials) for x in results_labels]
        elif self.scoring_method == 'logprobs': # white-box UQ
            results = []
            for idx, response in tqdm(
                enumerate(
                    self.llm_handler.batch_completion(
                        self.prompts,
                        logprobs=True,
                        seed=12345
                    )
                ),
                total=len(self.prompts),
                desc="NLI",
                unit="prompts",
                ):
                    results.append(response.choices[0])

            results_labels = []
            for result in results:
                if self.prompt_version == "v1":
                    label = result.message.content.strip().lower()
                    if label not in ['entailment', 'contradiction', 'neutral']:
                        label = 'neutral' #'invalid_label'
                        prob = 1.0
                    else:
                        logprob_sum = 0.0
                        generated_tokens = result.logprobs['content'][:-1]
                        for token in generated_tokens: #last token is just <|eot_id|>
                            token = dotdict(token)
                            logprob_sum +=token.logprob

                        prob = np.exp(logprob_sum/len(generated_tokens))

                    results_labels.append((label, prob))
                elif self.prompt_version == "v2":
                    text = result.message.content.strip()
                    logprobs = result.logprobs['content']
                    label = extract_last_square_brackets(text).lower()
                    prob = 1.0
                    if len(label) == 0 or label not in ['entailment', 'contradiction', 'neutral']:
                        label = 'neutral'
                    else:
                        # Look for the tokens corresponding to the label [label]
                        logits = []
                        for _, elem in reverse_enum(logprobs): # loop from the end
                            elem = dotdict(elem)
                            if elem.token in ['', '\n', ']']:
                                continue
                            if elem.token in ['[']:
                                break
                            logits.append(elem.logprob)

                        if len(logits) > 0:
                            prob = np.exp(np.mean(logits))
                    results_labels.append((label, prob))

        elif self.scoring_method == "consistency": # black-box UQ
            # Consistency-based UQ based on similarity aggregation.
            samples = {}
            temperatures = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
            for t in temperatures:
                self.n_trials = n_trials
                prompts = [item for sublist in [[prompt]*self.n_trials for prompt in self.prompts] for item in sublist]
                results = []

                print(f"Generating {n_trials} samples for temperature {t}")
                for idx, response in tqdm(
                    enumerate(
                        self.llm_handler.batch_completion(
                            self.prompts,
                            temperature=t
                        )
                    ),
                    total=len(self.prompts),
                    desc="NLI",
                    unit="prompts",
                    ):
                        results.append(response.choices[0].message.content.strip().lower())
                
                samples[t] = results

            # Compute pairwise similarities between samples and aggregate them
            results_labels = []
            start_ind = 0  
            for ii in range(len(premises)): # for each (premise, hypothesis)
                sampled_labels = []
                for t in temperatures: # collect the samples at different temperatures
                    result = samples[t][start_ind:start_ind+self.n_trials]
                    sampled_labels.extend(result)
                results_labels.append(get_label_probability(sampled_labels, NLI_LABELS))                    
                start_ind += self.n_trials

        return results_labels

    def create_prompts(self, premises, hypthotheses):
        
        self.prompts = []
        for premise, hypothesis in zip(premises,hypthotheses):
            if "granite-3.0" in self.model:
                premise = premise[:2000]
                hypothesis = hypothesis[:2000]

            if self.prompt_version == "v1":
                prompt = NLI_EXTRACTION_PROMPT1.format(
                    _PREMISE_PLACEHOLDER=premise,
                    _HYPOTHESIS_PLACEHOLDER=hypothesis,
                    _PROMPT_BEGIN_PLACEHOLDER=self.prompt_begin,
                    _PROMPT_END_PLACEHOLDER=self.prompt_end
                )
            elif self.prompt_version == "v2":
                prompt = NLI_EXTRACTION_PROMPT2.format(
                    _PREMISE_PLACEHOLDER=premise,
                    _HYPOTHESIS_PLACEHOLDER=hypothesis,
                    _PROMPT_BEGIN_PLACEHOLDER=self.prompt_begin,
                    _PROMPT_END_PLACEHOLDER=self.prompt_end
                )
            else:
                raise ValueError(f"Unknown NLI prompt version: {self.prompt_version}")
            # print(f"prompt length: {len(prompt)} chars, {len(nltk.word_tokenize(prompt))} words")

            self.prompts.append(prompt)



if __name__ == "__main__":

    model = "llama-3.1-70b-instruct"
    #model = "roberta"

    # import pickle
    # with open('examples/premises.pkl','rb') as f:premises = pickle.load(f)
    # with open('examples/hypothesis.pkl','rb') as f:hypotheses = pickle.load(f)    
    # premise = hypotheses[0] #premises[0]
    # hypothesis = premises[0] #hypotheses[0]

    # premise = "natural born killers is a 1994 american romantic crime action film directed by oliver stone and starring woody harrelson, juliette lewis, robert downey jr., tommy lee jones, and tom sizemore. the film tells the story of two victims of traumatic childhoods who become lovers and mass murderers, and are irresponsibly glorified by the mass media. the film is based on an original screenplay by quentin tarantino that was heavily revised by stone, writer david veloz, and associate producer richard rutowski. tarantino received a story credit though he subsequently disowned the film. jane hamsher, don murphy, and clayton townsend produced the film, with arnon milchan, thom mount, and stone as executive producers. natural born killers was released on august 26, 1994 in the united states, and screened at the venice film festival on august 29, 1994. it was a box office success, grossing $ 110 million against a production budget of $ 34 million, but received polarized reviews. some critics praised the plot, acting, humor, and combination of action and romance, while others found the film overly violent and graphic. notorious for its violent content and inspiring \" copycat \" crimes, the film was named the eighth most controversial film in history by entertainment weekly in 2006. = = plot = = mickey knox and his wife mallory stop at a diner in the new mexico desert. a duo of rednecks arrive and begin sexually harassing mallory as she dances by a jukebox. she initially encourages it before beating one of the men viciously. mickey joins her, and the couple murder everyone in the diner, save one customer, to whom they proudly declare their names before leaving. the couple camp in the desert, and mallory reminisces about how she met mickey, a meat deliveryman who serviced her family ' s household. after a whirlwind romance, mickey is arrested for grand theft auto and sent to prison ; he escapes and returns to mallory ' s home. the couple murders mallory ' s sexually abusive father and neglectful mother, but spare the life of mallory ' s little brother, kevin. the couple then have an unofficial marriage ceremony on a bridge. later, mickey and mallory hold a woman hostage in their hotel room. angered by mickey ' s desire for a threesome, mallory leaves, and mickey rapes the hostage. mallory drives to a nearby gas station, where she flirts with a mechanic. they begin to have sex on the hood of a car, but after mallory suffers a flashback of being raped by her father, and the mechanic recognizes her as a wanted murderer, mallory kills him. the pair continue their killing spree, ultimately claiming 52 victims in new mexico, arizona and nevada. pursuing them is detective jack scagnetti, who became obsessed with mass murderers at the age of eight after having witnessed the murder of his mother at the hand of charles whitman. beneath his heroic facade, he is also a violent psychopath and has murdered prostitutes in his past. following the pair ' s murder spree is self - serving tabloid journalist wayne gale, who profiles them on his show american maniacs, soon elevating them to cult - hero status. mickey and mallory become lost in the desert after taking psychedelic mushrooms, and they stumble upon a ranch owned by warren red cloud, a navajo man who provides them food and shelter. as mickey and mallory sleep, warren, sensing evil in the couple, attempts to exorcise the demon that he perceives in mickey, chanting over him as he sleeps. mickey, who has nightmares of his abusive parents, awakens during the exorcism and shoots warren to death. as the couple flee, they feel inexplicably guilty and come across a giant field of rattlesnakes, where they are badly bitten. they reach a drugstore to purchase snakebite antidote, but the store is sold out. a pharmacist recognizes the couple and triggers an alarm before mickey kills him. police arrive shortly after and accost the couple and a shootout ensues. the police end the showdown by beating the couple while a "
    # hypothesis = "Lanny Flaherty has appeared in numerous films."

    # premise = "Title: Lanny Flaherty - Biography - IMDb\nContent: Lanny Flaherty was born on July 27, 1942 in Pontotoc, Mississippi, USA. He was an actor, known for Signs (2002), Miller's Crossing (1990) and Men in Black\u00b3 ...\nLink: https://www.imdb.com/name/nm0280890/bio/"
    # hypothesis = "Lanny Flaherty is an American."

    # premise = "Betalains (betacyanins) were first isolated and its chemical structure discovered in 1960 at the University of Zurich by Dr. Tom Mabry. It was once thought that betalains were related to anthocyanins, the reddish pigments found in most plants. Both betalains and anthocyanins are water-soluble pigments found in the vacuoles of plant cells. However, betalains are structurally and chemically unlike anthocyanins and the two have never been found in the same plant together. For example, betalains contain nitrogen whereas anthocyanins do not."
    # hypothesis = "Betacyanins are a type of anthocyanin."
    # extractor = NLIExtractor(model=model, prompt_version="v2", debug=True)
    # result = extractor.run(premise=premise, hypothesis=hypothesis)
    # print(f"P -> H: {result}")

    c0 = "Tim Fischer served as chairman of Tourism Australia from 2004 to 2007, and was later Ambassador to the Holy See from 2009 to 2012."
    # c1 = "Tim Fischer started serving as the Ambassador to the Holy See on January 30, 2009. He was the first permanent resident ambassador from Australia to the Holy See. His term ended on January 20, 2012."
    # c2 = "On 21 July 2008, Fischer was nominated by Prime Minister Kevin Rudd as the first resident Australian Ambassador to the Holy See. Fischer worked closely with the Vatican on all aspects of the canonisation of Australia's first Roman Catholic, Mary MacKillop. He retired from the post on 20 January 2012."
    c2 = "On 21 July 2008, Fischer was nominated by Prime Minister Kevin Rudd as the first resident Australian Ambassador to the Holy See."
    c3 = "Tim Fischer did not serve as Chairman of Tourism from 2004 to 2007."
    a0 = "Tim Fischer started serving as the Ambassador to the Holy See in 2008."

    extractor = NLIExtractor(model=model, prompt_version="v2", debug=False, is_bert=False)
    
    result1 = extractor.run(premise=c0, hypothesis=a0)
    # result2 = extractor.run(premise=c1, hypothesis=a0)
    result3 = extractor.run(premise=c2, hypothesis=a0)
    result4 = extractor.run(premise=c3, hypothesis=c0)
    result5 = extractor.run(premise=c3, hypothesis=a0)
    # result6 = extractor.run(premise=c0, hypothesis=c2)
    # result7 = extractor.run(premise=c2, hypothesis=c0)
    # result8 = extractor.run(premise=c1, hypothesis=c2)
    # result9 = extractor.run(premise=c2, hypothesis=c1)

    print(f"c0 -> a0: {result1}")
    # print(f"c1 -> a0: {result2}")
    print(f"c2 -> a0: {result3}")
    print(f"c3 -> c0: {result4}")
    print(f"c3 -> a0: {result5}")
    # print(f"c0 -> c2: {result6}")
    # print(f"c2 -> c0: {result7}")
    # print(f"c1 -> c2: {result8}")
    # print(f"c2 -> c1: {result9}")

    import pickle
    with open('examples/premises.pkl','rb') as f:premises = pickle.load(f)
    with open('examples/hypothesis.pkl','rb') as f:hypotheses = pickle.load(f)    
    results = extractor.runall(premises, hypotheses)
    for result in results:
        print(result)

    print("Done.")