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

# NLI extractor using LLMs or BERT models.

import operator
import numpy as np

from typing import List
from tqdm import tqdm
from difflib import SequenceMatcher
from operator import itemgetter

# Local imports
from nli.align_scorer import AlignScorer
from llm_handler import LLMHandler
from utils import dotdict, extract_last_square_brackets
from prompts import NLI_EXTRACTION_PROMPT_V1, NLI_EXTRACTION_PROMPT_V2, NLI_EXTRACTION_PROMPT_V3, NLI_EXTRACTION_PROMPT_V3_FEW_SHOTS


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
            is_bert: bool = False,
            use_rits: bool = True
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
            is_bert: bool
                Whether to use a BERT model for NLI extraction instead of an LLM.
            use_rits: bool
                Whether to use RITS for LLM inference.
        """

        self.model_id = model_id
        self.method = method
        self.prompt_version = prompt_version
        self.debug = debug
        self.is_bert = is_bert
        self.use_rits = use_rits

        if not self.is_bert: # LLM based NLI extraction

            self.llm_handler = LLMHandler(model_id, use_rits)
            self.prompt_begin = self.llm_handler.get_prompt_begin()
            self.prompt_end = self.llm_handler.get_prompt_end()
        
            if self.prompt_version not in ["v1", "v2", "v3"]:
                raise ValueError(f"Unknown prompt version: {self.prompt_version}. "
                                 f"Supported versions are: 'v1', 'v2', 'v3'.")

            print(f"[NLIExtractor] Using LLM on {use_rits*'RITS'}{(not use_rits)*'vLLM'}: {self.model}")
            print(f"[NLIExtractor] Prompt version: {self.prompt_version}")
            self.bert_model = None
        else: # BERT model based NLI extraction
            self.prompt_begin = None
            self.prompt_end = None
            print(f"[NLIExtractor] Using BERT model: {self.model_id}")
            self.bert_model = AlignScorer(
                model="roberta-large", 
                ckpt_path="/home/radu/ckpts/AlignScore-large.ckpt",
                evaluation_mode="nli_sp",
                granularity="paragraph"
            ) 

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
        
        if not self.is_bert: # check if LLM is used
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

        if not self.is_bert: # LLM is used
            generated_texts = []
            generated_logprobs = []
            prompts = [self.make_prompt(premise, hypothesis) for premise, hypothesis in zip(premises, hypotheses)]
            print(f"[NLIExtractor] Prompts created: {len(prompts)}")

            for _, response in tqdm(
                enumerate(
                    self.llm_handler.batch_completion(
                        prompts,
                        logprobs=True,
                        seed=12345  
                    )
                ),
                total=len(prompts),
                desc="NLI",
                unit="prompts",
                ):
                    generated_texts.append(response.choices[0].message.content)
                    generated_logprobs.append(response.choices[0].logprobs['content'])

            results = []
            for text, logprobs in zip(generated_texts, generated_logprobs):
                label, probability = self.extract_relationship(text, logprobs)
                results.append({"label": label, "probability": probability})
        else: # BERT model is used
            print(f"[NLIExtractor] Extracting batched NLI relationships with BERT model: {len(premises)}")
            results = []
            responses = self.bert_model.score_all(
                premises=premises,
                hypotheses=hypotheses,
                op1="mean",
                op2="mean"
            )
            for response in responses:
                label, probability = self.extract_relationship_dict(response)
                results.append({"label": label, "probability": probability})

        return results


class NLIExtractorOld:
    def __init__(
            self,
            model_id: str = "llama-3.1-70b-instruct",
            scoring_method: str = "logprobs",
            prompt_version: str = "v1",
            use_rits=True
    ):
        self.model_id = model_id
        self.scoring_method = scoring_method
        self.prompt_version = prompt_version
        self.llm_handler = LLMHandler(model_id, use_rits)
        self.prompt_begin = self.llm_handler.get_prompt_begin()
        self.prompt_end = self.llm_handler.get_prompt_end()
            
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
            else:
                raise ValueError(f"Unknown NLI prompt version: {self.prompt_version}")
            # print(f"prompt length: {len(prompt)} chars, {len(nltk.word_tokenize(prompt))} words")

            self.prompts.append(prompt)



if __name__ == "__main__":

    model_id = "llama-3.1-70b-instruct"
    # model_id = "roberta"

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

    extractor = NLIExtractor(model_id=model_id, prompt_version="v2", debug=False, is_bert=False)
    
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