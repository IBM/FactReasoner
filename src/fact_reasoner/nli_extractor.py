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

import operator
from difflib import SequenceMatcher
from operator import itemgetter
from typing import List

import numpy as np
from tqdm import tqdm

# Local imports
from src.fact_reasoner.llm_handler import LLMHandler
from src.fact_reasoner.prompts import (
    NLI_EXTRACTION_PROMPT_V1,
    NLI_EXTRACTION_PROMPT_V2,
    NLI_EXTRACTION_PROMPT_V3,
    NLI_EXTRACTION_PROMPT_V3_FEW_SHOTS,
)
from src.fact_reasoner.utils import dotdict, extract_last_square_brackets

# Define the NLI relationships (labels)
NLI_LABELS = ['entailment', 'contradiction', 'neutral']

def similarity(a, b):
    """Calculate the similarity ratio between two strings using SequenceMatcher.
    
    Args:
        a: str 
            The first string.
        b: str
            The second string.

    Returns:
        float: The similarity ratio between the two strings, ranging from 0 to 1.
    """
    return SequenceMatcher(None, a, b).ratio()

def get_label_probability(samples: list, labels: list):
    """Get the label with the highest average similarity to the samples.
    Args:
        samples: list
            A list of strings representing the samples.
        labels: list
            A list of strings representing the labels.
    Returns:
        tuple: A tuple containing the label with the highest average similarity and its average similarity score.
    """

    candidates = []
    for label in labels:
        distances = [similarity(label, sample) for sample in samples]
        candidates.append((label, np.average(distances)))
    candidates = sorted(candidates, key=itemgetter(1), reverse=True)
    return candidates[0]

def reverse_enum(L):
    """
    Reverse enumerate a list, yielding index and value pairs in reverse order.
    
    Args:
        L: list
            The list to reverse enumerate.
    Yields:
        tuple: A tuple containing the index and value of each element in reverse order.
    """

    # Reverse enumerate the list
    for index in reversed(range(len(L))):
        yield index, L[index]


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
            model_id: str = "llama-3.1-70b-instruct",
            method: str = "logprobs",
            prompt_version: str = "v1",
            debug: bool = False,
            backend: str = "rits",
            inference_batch_size = 8,
    ):
        """
        Initialize the NLIExtractor.

        Args:
            model_id: str
                The name of the LLM model to use for NLI extraction.
            method: str
                The method to computing the probabilities of the NLI relationships, e.g., "logprobs".
            prompt_version: str
                The version of the prompt to use for NLI extraction.
            debug: bool
                Whether to enable debug mode (prints additional information).
            backend: rits
                The model's backend (rits, hf or wx).
        """

        self.model_id = model_id
        self.method = method
        self.prompt_version = prompt_version
        self.debug = debug
        self.backend = backend
        self.inference_batch_size = inference_batch_size

        self.llm_handler = LLMHandler(model_id, backend)
        self.prompt_begin = self.llm_handler.get_prompt_begin()
        self.prompt_end = self.llm_handler.get_prompt_end()
    
        if self.prompt_version not in ["v1", "v2", "v3"]:
            raise ValueError(f"Unknown prompt version: {self.prompt_version}. "
                                f"Supported versions are: 'v1', 'v2', 'v3'.")

        print(f"[NLIExtractor] Using LLM on {self.backend}: {self.model_id}")
        print(f"[NLIExtractor] Prompt version: {self.prompt_version}")

    def make_prompt(self, premise: str, hypothesis: str) -> str:
        """
        Create the prompt for NLI extraction based on the premise and hypothesis.
        
        Args:
            premise: str
                The premise text.
            hypothesis: str
                The hypothesis text.
        Returns:
            str: The formatted prompt string.
        """
        
        if self.prompt_version == "v1":
            prompt = NLI_EXTRACTION_PROMPT_V1.format(
                _PREMISE_PLACEHOLDER=premise,
                _HYPOTHESIS_PLACEHOLDER=hypothesis,
                _PROMPT_BEGIN_PLACEHOLDER=self.prompt_begin,
                _PROMPT_END_PLACEHOLDER=self.prompt_end
            )
        elif self.prompt_version == "v2":
            prompt = NLI_EXTRACTION_PROMPT_V2.format(
                _PREMISE_PLACEHOLDER=premise,
                _HYPOTHESIS_PLACEHOLDER=hypothesis,
                _PROMPT_BEGIN_PLACEHOLDER=self.prompt_begin,
                _PROMPT_END_PLACEHOLDER=self.prompt_end
            )
        elif self.prompt_version == "v3": # specific to Google search results (links)
            # Set the few-shots section
            few_shots_lst = []
            for dict_item in NLI_EXTRACTION_PROMPT_V3_FEW_SHOTS:
                claim = dict_item["claim"]
                search_result_str = dict_item["search_result"]
                human_label = dict_item["human_label"]
                few_shots_lst.extend([claim, search_result_str, human_label])

            prompt = NLI_EXTRACTION_PROMPT_V3.format(
                _CLAIM_PLACEHOLDER=hypothesis,
                _SEARCH_RESULTS_PLACEHOLDER=premise,
                _PROMPT_BEGIN_PLACEHOLDER=self.prompt_begin,
                _PROMPT_END_PLACEHOLDER=self.prompt_end,
                *few_shots_lst
            )

        return prompt

    def extract_relationship(self, text: str, logprobs: List[dict]):
        """
        Extract the relationship and probability. The relationship should be on
        the last line of the generated text and one of the following: 
            [entailment], [contradiction] or [neutral]. Anything
        else will be assumed to be neutral with probability 1. The probability
        of the relationship is the exp of the average logprob of the corresponding
        tokens.

        Args:
            text: str
                The generated text from the LLM.
            logprobs: List[dict]
                The log probabilities of the generated tokens.
        Returns:
            tuple: A tuple containing the label (str) and its probability (float).
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

                probability = float(np.exp(logprob_sum/len(generated_tokens)))
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
                    probability = float(np.exp(np.mean(logits)))
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
                    probability = float(np.exp(np.mean(logits)))

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
        """
        Extract the NLI relationship between a premise and a hypothesis.
        
        Args:
            premise: str
                The premise text.
            hypothesis: str
                The hypothesis text.
        
        Returns:
            dict: A dictionary containing the label and its probability.
        """
        
        prompt = self.make_prompt(premise, hypothesis)
        print(f"[NLIExtractor] Prompt created ({len(prompt)}).")
        response = self.llm_handler.completion(
            prompt,
            logprobs=True
        )

        text = response.choices[0].message.content
        if self.debug:
            print(f"Generate response:\n{text}")
        logprobs = response.choices[0].logprobs['content']
        label, probability = self.extract_relationship(text, logprobs)
        result = {'label': label, 'probability': probability}

        return result
    
    def runall(self, premises: List[str], hypotheses: List[str]):
        """
        Extract the NLI relationships for a list of premises and hypotheses.
        
        Args:
            premises: List[str]
                A list of premise texts.
            hypotheses: List[str]
                A list of hypothesis texts.
        Returns:
            List[dict]: A list of dictionaries, each containing the label and 
            its probability for each premise-hypothesis pair.
        """
        
        # Safety checks
        assert len(premises) == len(hypotheses), "Premises and hypotheses must have the same length."

        generated_texts = []
        generated_logprobs = []
        prompts = [self.make_prompt(premise, hypothesis) for premise, hypothesis in zip(premises, hypotheses)]
        print(f"[NLIExtractor] Prompts created: {len(prompts)}")

        batched_prompts = [prompts[i:i + self.inference_batch_size] for i in range(0, len(prompts), self.inference_batch_size)]

        for batch_idx, current_batch in tqdm(
            enumerate(batched_prompts),
            total=len(batched_prompts),
            desc="NLI (batches)",
            unit="batches"
            ):
                #responses = self.llm_handler.batch_completion(current_batch, logprobs=True, seed=12345)
                responses = self.llm_handler.batch_completion(
                    current_batch,
                    logprobs=True,
                    seed=12345,
                )

                for response in responses:
                    generated_texts.append(response.choices[0].message.content)
                    generated_logprobs.append(response.choices[0].logprobs['content'])

        results = []
        for text, logprobs in zip(generated_texts, generated_logprobs):
            label, probability = self.extract_relationship(text, logprobs)
            results.append({"label": label, "probability": probability})

        return results


if __name__ == "__main__":

    model_id = "llama-3.3-70b-instruct"
    backend = "rits"

    # Example premise and hypothesis
    premise = "natural born killers is a 1994 american romantic crime action film directed by oliver stone and starring woody harrelson, juliette lewis, robert downey jr., tommy lee jones, and tom sizemore. the film tells the story of two victims of traumatic childhoods who become lovers and mass murderers, and are irresponsibly glorified by the mass media. the film is based on an original screenplay by quentin tarantino that was heavily revised by stone, writer david veloz, and associate producer richard rutowski. tarantino received a story credit though he subsequently disowned the film. jane hamsher, don murphy, and clayton townsend produced the film, with arnon milchan, thom mount, and stone as executive producers. natural born killers was released on august 26, 1994 in the united states, and screened at the venice film festival on august 29, 1994. it was a box office success, grossing $ 110 million against a production budget of $ 34 million, but received polarized reviews. some critics praised the plot, acting, humor, and combination of action and romance, while others found the film overly violent and graphic. notorious for its violent content and inspiring \" copycat \" crimes, the film was named the eighth most controversial film in history by entertainment weekly in 2006. = = plot = = mickey knox and his wife mallory stop at a diner in the new mexico desert. a duo of rednecks arrive and begin sexually harassing mallory as she dances by a jukebox. she initially encourages it before beating one of the men viciously. mickey joins her, and the couple murder everyone in the diner, save one customer, to whom they proudly declare their names before leaving. the couple camp in the desert, and mallory reminisces about how she met mickey, a meat deliveryman who serviced her family ' s household. after a whirlwind romance, mickey is arrested for grand theft auto and sent to prison ; he escapes and returns to mallory ' s home. the couple murders mallory ' s sexually abusive father and neglectful mother, but spare the life of mallory ' s little brother, kevin. the couple then have an unofficial marriage ceremony on a bridge. later, mickey and mallory hold a woman hostage in their hotel room. angered by mickey ' s desire for a threesome, mallory leaves, and mickey rapes the hostage. mallory drives to a nearby gas station, where she flirts with a mechanic. they begin to have sex on the hood of a car, but after mallory suffers a flashback of being raped by her father, and the mechanic recognizes her as a wanted murderer, mallory kills him. the pair continue their killing spree, ultimately claiming 52 victims in new mexico, arizona and nevada. pursuing them is detective jack scagnetti, who became obsessed with mass murderers at the age of eight after having witnessed the murder of his mother at the hand of charles whitman. beneath his heroic facade, he is also a violent psychopath and has murdered prostitutes in his past. following the pair ' s murder spree is self - serving tabloid journalist wayne gale, who profiles them on his show american maniacs, soon elevating them to cult - hero status. mickey and mallory become lost in the desert after taking psychedelic mushrooms, and they stumble upon a ranch owned by warren red cloud, a navajo man who provides them food and shelter. as mickey and mallory sleep, warren, sensing evil in the couple, attempts to exorcise the demon that he perceives in mickey, chanting over him as he sleeps. mickey, who has nightmares of his abusive parents, awakens during the exorcism and shoots warren to death. as the couple flee, they feel inexplicably guilty and come across a giant field of rattlesnakes, where they are badly bitten. they reach a drugstore to purchase snakebite antidote, but the store is sold out. a pharmacist recognizes the couple and triggers an alarm before mickey kills him. police arrive shortly after and accost the couple and a shootout ensues. the police end the showdown by beating the couple while a "
    hypothesis = "Lanny Flaherty has appeared in numerous films."

    # Create the extractor
    extractor = NLIExtractor(model_id=model_id, prompt_version="v2", backend=backend)    
    result = extractor.run(premise=premise, hypothesis=hypothesis)

    # Output results
    print(f"H -> P: {result}")
    print("Done.")