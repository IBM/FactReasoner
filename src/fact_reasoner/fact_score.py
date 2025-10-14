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

# Our implementation of the FactScore paper using LLAMA3 models

import argparse
import json
import os
import string
import sys
from typing import List

import nltk
import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm

import litellm

# Local
from fm_factual.atom_extractor import AtomExtractor
from fm_factual.atom_reviser import AtomReviser
from fm_factual.context_retriever import ContextRetriever
from fm_factual.fact_utils import Atom, Context, build_atoms, build_contexts
from fm_factual.utils import (
    DEFAULT_PROMPT_BEGIN,
    DEFAULT_PROMPT_END,
    MODELS,
    extract_last_square_brackets,
)

# litellm.set_verbose = True



# os.environ['LITELLM_LOG'] = 'DEBUG'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Version 1 of the prompt (from the original FactScore paper)
FACTSCORE_PROMPT = """{_PROMPT_BEGIN_PLACEHOLDER}
Answer the question about {_TOPIC_PLACEHOLDER} based on the given context.
 
{_KNOWLEDGE_PLACEHOLDER}

Input: {_STATEMENT_PLACEHOLDER} True or False?
Output:{_PROMPT_END_PLACEHOLDER}
"""

FACTSCORE_PROMPT_NOTOPIC = """{_PROMPT_BEGIN_PLACEHOLDER}
Answer the input question based on the given context.
 
{_KNOWLEDGE_PLACEHOLDER}

Input: {_STATEMENT_PLACEHOLDER} True or False?
Output:{_PROMPT_END_PLACEHOLDER}
"""

# Version 2 of the prompt (based on more recent work VeriScore, FactBench)
FACTBENCH_PROMPT = """{_PROMPT_BEGIN_PLACEHOLDER}

Instructions:
You are provided with a STATEMENT and several KNOWLEDGE points. \
Your task is to evaluate the relationship between the STATEMENT and the KNOWLEDGE, following the steps outlined below:

1. Summarize KNOWLEDGE Points: Carefully analyze the KNOWLEDGE points one by one and assess their relevance to the STATEMENT. \
Summarize the main points of the KNOWLEDGE.
2. Evaluate Evidence: Based on your reasoning:
- If the KNOWLEDGE strongly implies or directly supports the STATEMENT, explain the supporting evidence.
- If the KNOWLEDGE contradicts the STATEMENT, identify and explain the conflicting evidence.
- If the KNOWLEDGE is insufficient to confirm or deny the STATEMENT, explain why the evidence is inconclusive.
3. Restate the STATEMENT: After considering the evidence, restate the STATEMENT to maintain clarity.
4. Final Answer: Based on your reasoning and the STATEMENT, determine your final answer. \
Your final answer must be one of the following, wrapped in square brackets:
- [Supported] if the STATEMENT is supported by the KNOWLEDGE.
- [Contradicted] if the STATEMENT is contradicted by the KNOWLEDGE.
- [Unverifiable] if the KNOWLEDGE is insufficient to verify the STATEMENT.

Your task:

KNOWLEDGE: 
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}{_PROMPT_END_PLACEHOLDER}
"""

class FactScore:
    """
    Our implementation of the FactScore paper. We implement both the original
    FactScore pipeline that works with contexts retrieved from wikipedia (texts)
    as well as the more recent version presented in the FactBench paper.

        v1 - original FactScore paper
        v2 - recent FactBench paper
    """

    def __init__(
            self,
            context_retriever: ContextRetriever = None,
            atom_extractor: AtomReviser = None,
            atom_reviser: AtomReviser = None,
            model: str = "llama-3.3-70b-instruct",
            prompt_version: str = "v1",
            debug_mode: bool = False,
            binary_output: bool = True,
            add_topic: bool = False,
            llm_inference_provider: str = "rits"
    ):
        """
        Construct the FactScore pipeline instance.

        Args:
            context_retriever: ContextRetriever
                The service used for retrieving external contexts.
            atom_extractor: AtomExtractor
                The service used for extracting atoms from the response.
            atom_reviser: AtomReviser
                The service used for decontextualizing the atoms.
            model: str
                The name of the model used by FactScore.
            prompt_version: str
                The prompt version: v1 - FactScore, v2 - FactBench
            debug_mode: bool
                Flaf indicating debug mode (default is False)
            binary_output: bool
                If true, the output labels are [S - Supported, NS - NotSupported].
                Otherwise, the output labels are [S - Supported, C - Contradicted, U - Unverifiable or Undediced]
            add_topic: bool
                If True, then the topic is added (relevant only for v1 and Biographies).
        """

        self.query = None
        self.response = None
        self.topic = None
        self.debug_mode = debug_mode
        self.add_topic = add_topic # default is False

        self.model = model

        self.context_retriever = context_retriever
        self.atom_extractor = atom_extractor
        self.atom_reviser = atom_reviser
        self.prompt_version = prompt_version
        self.binary_output = binary_output # default is True
    
        assert self.prompt_version in ["v1", "v2"], f"Unknown prompt version: {self.prompt_version}"
        if self.prompt_version == "v1": # for FactScore force binary output
            self.binary_output = True

        self.model_info = MODELS[llm_inference_provider][model]
        self.prompt_template = self.model_info.get("prompt_template", None)
        self.max_new_tokens = self.model_info.get("max_new_tokens", None)
        self.api_base = self.model_info.get("api_base", None)
        self.model_id = self.model_info.get("model_id", None)
        self.prompt_begin = self.model_info.get("prompt_begin", DEFAULT_PROMPT_BEGIN)
        self.prompt_end = self.model_info.get("prompt_end", DEFAULT_PROMPT_END)
        self.use_short_prompt = True if self.max_new_tokens <= 4096 else False

        assert self.prompt_template is not None \
            and self.max_new_tokens is not None \
            and self.api_base is not None \
            and self.model_id is not None

        if not os.environ.get("_DOTENV_LOADED"):
            load_dotenv(override=True) 
            os.environ["_DOTENV_LOADED"] = "1"
         
        if self.llm_inference_provider == "rits":
            self.API_KEY = os.getenv("RITS_API_KEY")
            self.RITS_API_KEY = self.API_KEY
        elif llm_inference_provider == "watsonx":
            self.API_KEY = os.getenv("WATSONX_API_KEY")
            self.RITS_API_KEY = None
        elif llm_inference_provider == "openai":
            self.API_KEY  = os.getenv("OPENAI_API_KEY")
            self.RITS_API_KEY = None
        
        print(f"[FactScore] Using LLM on {llm_inference_provider}: {self.model_id}")
        print(f"[FactScore] Using short prompt: {self.use_short_prompt}")
        print(f"[FactScore] Prompt version: {self.prompt_version}")
        print(f"[FactScore] Binary output: {self.binary_output}")

        self.atoms = {} # indexed by atom id
        self.contexts = {} # indexed by context id

        self.labels_human = None
        self.labels_chatgpt = None
        self.labels_llamanp = None

    def from_json(self, json_file: str):
        """
        Create a problem instance from a json file containing both atoms and contexts.

        Args:
            json_file: str
                The path to the json file containing the problem instance.
        """
        
        print(f"[FactScore] Reading JSON instance from: {json_file}")
        with open(json_file) as f:
            data = json.load(f)
            f.close()

        self.query = data["query"]
        self.response = data["response"]
        if self.add_topic:
            self.topic = data["topic"]

        for atom_dict in data["atoms"]:
            aid = atom_dict["id"]
            text = atom_dict["text"]
            a = Atom(id=aid, text=text)
            self.atoms[aid] = a
        
        print(f"[FactScore] Atoms found: {len(self.atoms)}")

        for context_dict in data["contexts"]:
            cid = context_dict["id"]
            aid = context_dict["atom_id"]
            text = context_dict["text"]

            a = self.atoms[aid]
            ctxt = Context(id=cid, atom=a, text=text, title="", snippet="", link="")
            a.add_context(ctxt)
            self.contexts[cid] = ctxt

        print(f"[FactScore] Contexts found: {len(self.contexts)}")

    def from_dict_with_contexts(
            self,
            data: dict,
    ):
        """
        Create a problem instance from a dict containing both atoms and contexts.

        Args:
            data: dict
                The dict containing the problem instance.
        """

        self.query = data["input"]
        self.response = data["output"]
        if self.topic:
            self.topic = data["topic"]
        
        print(f"[FactScore] Reading the human annotated atoms ...")                
        gold_labels = []
        atom_ids = []
        self.atoms = {}
        self.contexts = {}
        atom2contexts = {}
        for atom_dict in data["atoms"]:
            aid = atom_dict["id"]
            text = atom_dict["text"]
            original = atom_dict["original"]
            label = atom_dict.get("label", None)
            contexts = atom_dict["contexts"]
            a = Atom(id=aid, text=text, label=label)
            a.set_original(original)
            atom_ids.append(aid)
            gold_labels.append(label)
            self.atoms[aid] = a
            atom2contexts[aid] = contexts

        print(f"[FactScore] Atoms found: {len(self.atoms)}")
        for _, atom in self.atoms.items():
            print(atom)
        
        self.labels_human = dict(zip(atom_ids, gold_labels))
        print(f"[FactScore] Lables found: {self.labels_human}")

        print(f"[FactScore] Reading the contexts ...")
        for context_dict in data["contexts"]:
            cid = context_dict["id"]
            title = context_dict["title"]
            text = context_dict["text"]
            snippet = context_dict.get("snippet", "")
            link = context_dict.get("link", "")
            ctxt = Context(id=cid, atom=None, text=text, title=title, snippet=snippet, link=link)
            self.contexts[cid] = ctxt

        print(f"[FactScore] Contexts found: {len(self.contexts)}")
        for aid, atom in self.atoms.items():
            ctxts = []
            for c in atom2contexts[aid]:
                ctxts.append(self.contexts[c])
                self.contexts[c].set_atom(atom)
            atom.add_contexts(ctxts)
        return True

    def build(
            self,
            debug_mode: bool = False,
            has_atoms: bool = False,
            has_contexts: bool = False,
            decontextualize_atoms: bool = True,
            no_contexts: bool = False
    ):
        """
        Build the atoms and contexts using the retrieval service.

        Args:
            debug_mode: bool
                Boolean flag indicating debugging mode (default False)
            has_atoms: bool
                A boolean flag indicating if the atoms have already been created.
            has_contexts: bool
                A boolean flag indicating if the contexts have already been created.
            decontextualize_atoms: bool
                A boolean flag indicating that the atoms need to be decontextualized
                (i.e., pronouns he, she, it, ... replaced by the actual entity)
            no_contexts: bool
                A boolean flag indicating if contexts are to be retrieved or not.
                If True, then we run a version that only leverages the internal
                knowledge of the language model.
        """

        # Initialize the scorer
        self.debug_mode = debug_mode
        self.no_contexts = no_contexts

        # Create the atomizer (for the response)
        assert self.atom_extractor is not None, f"Atom extractor must be created."
        assert self.atom_reviser is not None, f"Atom reviser must be created."

        print(f"[FactScore] Building the pipeline instance ...]")
        print(f"[FactScore] Using contexts: {not no_contexts}")
        
        # Build the atoms 
        if has_atoms == False:
            self.atoms = build_atoms(
                response=self.response,
                atom_extractor=self.atom_extractor
            )

        assert len(self.atoms) > 0, f"Atoms must be initialized if `has_atoms` is True!"

        # Decontextualize the atoms
        if decontextualize_atoms:
            print(f"[FactScore] Decontextualize the atoms ...")
            atom_ids = [aid for aid in sorted(self.atoms.keys())]
            old_atoms = [self.atoms[aid].get_text() for aid in atom_ids]
            result = self.atom_reviser.run(old_atoms, self.response)
            for i, aid in enumerate(atom_ids):
                elem = result[i]
                self.atoms[aid].set_text(elem["revised_atom"])
                print(self.atoms[aid])

        # Build the contexts (per atom)
        if no_contexts:
            self.contexts = {}
        else:
            if has_contexts == False: # check if contexts already in file
                self.contexts = build_contexts(
                    atoms=self.atoms,
                    retriever=self.context_retriever,
                )

    def make_prompt(
            self,
            atom: str,
            topic: str,
            passages: List[dict],
    ):
        """
        Create the prompt for predicting the label of the atom given contexts.

        Args:
            atom: str
                The string representing the atom.
            topic: str
                The topic (str) associated with the atom.
            passages: List[dict]
                A list of dictionaries representing the retrieved passages 
                relevant to the atom. Each passage is a dict with two keys:
                title - title of the article and text - passage in that article.
            model_id: str
                The model id used for prediction.

        Returns:
            A string representing the prompt (follow the FactScore paper instructions).
        """

        knowledge = ""
        for _, psg in enumerate(passages):
            title = psg["title"]
            text = psg["text"]
            snippet = psg.get("snippet", "")
            if self.use_short_prompt: # check for small context (e.g., granite-3.0)
                knowledge += "Title: {}\nSummary: {}\nText: {}\n\n".format(title, snippet, text[:2000])
            else:
                knowledge += "Title: {}\nSummary: {}\nText: {}\n\n".format(title, snippet, text)

        if self.prompt_version == "v1":
            if topic is not None:
                prompt = FACTSCORE_PROMPT.format(
                    _PROMPT_BEGIN_PLACEHOLDER=self.prompt_begin,
                    _PROMPT_END_PLACEHOLDER=self.prompt_end,
                    _TOPIC_PLACEHOLDER=topic,
                    _STATEMENT_PLACEHOLDER=atom,
                    _KNOWLEDGE_PLACEHOLDER=knowledge,
                )
            else:
                prompt = FACTSCORE_PROMPT_NOTOPIC.format(
                    _PROMPT_BEGIN_PLACEHOLDER=self.prompt_begin,
                    _PROMPT_END_PLACEHOLDER=self.prompt_end,
                    _STATEMENT_PLACEHOLDER=atom,
                    _KNOWLEDGE_PLACEHOLDER=knowledge,
                )
        elif self.prompt_version == "v2":
            prompt = FACTBENCH_PROMPT.format(
                _PROMPT_BEGIN_PLACEHOLDER=self.prompt_begin,
                _PROMPT_END_PLACEHOLDER=self.prompt_end,
                _STATEMENT_PLACEHOLDER=atom,
                _KNOWLEDGE_PLACEHOLDER=knowledge,
            )
        else:
            raise ValueError(f"FactScore: Unknown prompt version {self.prompt_version}.")

        # print(f"prompt length: {len(prompt)} chars, {len(nltk.word_tokenize(prompt))} words")        
        return prompt

    def extract_label(self, text: str) -> str:
        """
        Extract the atom label from the generated text. We expect the label to
        be on the last line of the response, and be one of the following:
            [Supported], [Contradicted], [Unverifiable].
        We only consider [Supported]/S atoms, the others will be [NotSupported]/NS.
        """
        if self.prompt_version == "v1": # only binary output supported
            generated_answer = text.lower()
            if "true" in generated_answer or "false" in generated_answer:
                if "true" in generated_answer and "false" not in generated_answer:
                    is_supported = True
                elif "false" in generated_answer and "true" not in generated_answer:
                    is_supported = False
                else:
                    is_supported = generated_answer.index("true") > generated_answer.index("false")
            else:
                is_supported = all([keyword not in generated_answer.lower().translate(str.maketrans("", "", string.punctuation)).split() for keyword in ["not", "cannot", "unknown", "information"]])

            label = "S" if is_supported else "NS"
            return label
        elif self.prompt_version == "v2":
            label = extract_last_square_brackets(text)
            if self.binary_output:
                if len(label) > 0 and label.lower() in ['supported']:
                    return "S"
                else:
                    return "NS"
            else:
                if len(label) > 0 and label.lower() in ['supported']:
                    return "S"
                elif len(label) > 0 and label.lower() in ['contradicted']:
                    return "C"
                else:
                    return "U"
                
    def predict_atom_labels(self) -> dict:
        """
        Use a strong LLM to predict the label S or NS of an atom given its contexts.
        """

        assert len(self.atoms) > 0

        # Use the LLM to label the atom
        print(f"[FactScore] Labeling atoms with {self.model_id} ...")
        prompts = []
        atom_ids = []

        # Create the prompts for each of the atoms
        for aid, atom in self.atoms.items():
            atom_ids.append(aid)
            contexts = atom.get_contexts()
            if contexts is not None and len(contexts) > 0:
                passages = []
                for c in contexts:
                    if len(c.get_text()) == 0:
                        passages.append(dict(title=c.get_title(), text=c.get_snippet()))
                    else:
                        passages.append(dict(title=c.get_title(), text=c.get_text()))
            else:
                passages = [] # no passages retrieved for the atom

            prompt = self.make_prompt(
                atom=atom.get_text(),
                topic=self.topic,
                passages=passages
            )

            # print(f"prompt length: {len(prompt)} chars, {len(nltk.word_tokenize(prompt))} tokens")
            prompts.append(prompt)

        print(f"[FactScore] Prompts created: {len(prompts)}")

        # Prepare the LLM call
        results = []
        messages = [[dict(role="user", content=prompt)] for prompt in prompts]
        for _, response in tqdm(
            enumerate(
                litellm.batch_completion(
                    model=self.model_id,
                    api_base=self.api_base,
                    messages=messages,
                    temperature=0,
                    seed=42,
                    api_key=self.RITS_API_KEY,
                    extra_headers={
                        "RITS_API_KEY": self.RITS_API_KEY
                    }
                )
            ),
            total=len(messages),
            desc="Prediction",
            unit="prompts",
            ):
                results.append(response.choices[0].message.content)

        if self.debug_mode:
            for i, response in enumerate(results):
                print(f"PROMPT:\n{prompts[i]}")
                print(f"RESPONSE:\n{response}")

        # Postprocess the generated answers
        atom_labels = [self.extract_label(text) for text in results]
        return dict(zip(atom_ids, atom_labels))
    
    def score(self):
        """
        Compute the factuality score taking into consideration the contexts 
        retrieved for each of the atom in the answer.

        Factuality score = # atoms(true) / # atoms

        Intuitively, a score of 100% means that all atoms in the answer are
        factually correct. If none of them are correct, then the score is 0%. If
        only half of the atoms are correct, then the score is 50%.

        Returns:
            dict
                The results dictionary containing the factuality score i.e., a real value in [0, 1]
        """

        # Safety checks
        # assert len(self.atoms) > 0
        # assert len(self.contexts) > 0

        # Compute the FactScore
        num_true_atoms = 0
        num_false_atoms = 0
        num_uniform_atoms = 0
        labels = self.predict_atom_labels()
        for _, label in labels.items():
            if self.binary_output:
                if label == "S":
                    num_true_atoms += 1
                else:
                    num_false_atoms += 1
            else:
                if label == "S":
                    num_true_atoms += 1
                elif label == "C":
                    num_false_atoms += 1
                else:
                    num_uniform_atoms += 1
      
        # Precision
        fscore = float(num_true_atoms)/float(len(self.atoms))

        results = {}
        results["factuality_score"] = fscore
        results["num_atoms"] = len(self.atoms)
        results["num_contexts"] = len(self.contexts)
        results["num_true_atoms"] = num_true_atoms
        results["num_false_atoms"] = num_false_atoms
        results["num_uniform_atoms"] = num_uniform_atoms
        results["entropy"] = None
        results["avg_entropy"] = None

        print(f"[FactScore] Predictions: {labels}")
        if self.labels_human is not None and self.binary_output is True:
            true_atoms = 0
            false_atoms = 0
            num_true_positive = 0
            num_true_negative = 0
            num_false_positive = 0
            num_false_negative = 0
            for aid, l in self.labels_human.items():
                if l == "S":
                    true_atoms += 1
                    if labels[aid] == "S":
                        num_true_positive += 1
                    else:
                        num_false_negative += 1
                else:
                    false_atoms += 1
                    if labels[aid] == "NS":
                        num_true_negative += 1
                    else:
                        num_false_positive += 1                    
            fscore_gold = true_atoms/len(self.labels_human)
            print(f"[FactScore] Gold labels: {self.labels_human}")
            print(f"[FactScore] Predictions: {labels}")
            print(f"[FactScore] Gold fscore: {fscore_gold} ({true_atoms}/{len(self.labels_human)})")
            results["gold_factuality_score"] = fscore_gold
            results["gold_true_atoms"] = true_atoms
            results["true_positive"] = num_true_positive
            results["true_negative"] = num_true_negative
            results["false_positive"] = num_false_positive
            results["false_negative"] = num_false_negative
        elif self.labels_human is not None and self.binary_output is False:
            true_atoms = 0
            false_atoms = 0
            num_true_positive = 0
            num_true_negative = 0
            num_false_positive = 0
            num_false_negative = 0
            for aid, l in self.labels_human.items(): # true labels are either S or NS
                if l == "S":
                    true_atoms += 1
                    if labels[aid] == "S":
                        num_true_positive += 1
                    else:
                        num_false_negative += 1
                else:
                    false_atoms += 1
                    if labels[aid] in ["C", "U"]:
                        num_true_negative += 1
                    else:
                        num_false_positive += 1     

            fscore_gold = true_atoms/len(self.labels_human)
            print(f"[FactScore] Gold labels: {self.labels_human}")
            print(f"[FactScore] Predictions: {labels}")
            print(f"[FactScore] Gold fscore: {fscore_gold} ({true_atoms}/{len(self.labels_human)})")
            results["gold_factuality_score"] = fscore_gold
            results["gold_true_atoms"] = true_atoms
            results["true_positive"] = num_true_positive
            results["true_negative"] = num_true_negative
            results["false_positive"] = num_false_positive
            results["false_negative"] = num_false_negative

        if self.topic is not None and len(self.topic) > 0:
            results["topic"] = self.topic
        results["input"] = self.query

        return results

def test():

    model = "granite-3.1-8b-instruct"
    prompt_version = "v2"
    cache_dir = "/home/radu/data/cache"

    context_retriever = ContextRetriever(service_type="google", top_k=5, cache_dir=cache_dir)
    atom_extractor = AtomExtractor(model)
    atom_reviser = AtomReviser(model)

    # Create the FactScore pipeline
    pipeline = FactScore(
        context_retriever=context_retriever,
        atom_extractor=atom_extractor,
        atom_reviser=atom_reviser,
        model=model,
        prompt_version=prompt_version,
        binary_output=True,
        add_topic=True
    )

    # Load the problem instance from a file
    json_file = "/home/radu/git/fm-factual/examples/test4.json"
    with open(json_file, "r") as f:
        data = json.load(f)
    
    pipeline.from_dict_with_contexts(data)

    # Build the scorer
    pipeline.build(
        has_atoms=True,
        has_contexts=True,
        decontextualize_atoms=False
    )

    results = pipeline.score()
    print(f"[FactScore] Results: {results}")
    print(f"Done.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_file', 
        type=str, 
        default=None, 
        help="Path to the labeled dataset (gold)."
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=None, 
        help="Path to the output directory."
    )

    parser.add_argument(
        '--cache_dir', 
        type=str, 
        default=None, 
        help="Path to the cache directory."
    )

    parser.add_argument(
        '--dataset_name', 
        type=str, 
        default=None, 
        help="Name of the dataset."
    )

    parser.add_argument(
        '--model', 
        type=str, 
        default="llama-3.3-70b-instruct", 
        help="Name of the underlying LLM."
    )

    parser.add_argument(
        '--binary_output', 
        default=False, 
        action='store_true', 
        help="Ensure binary output for the atomic unit label prediction."
    )

    parser.add_argument(
        '--add_topic', 
        default=False, 
        action='store_true', 
        help="Ensure the the topic is added (relevant only for Biographies)."
    )

    parser.add_argument(
        '--test', 
        default=False, 
        action='store_true', 
        help="Debugging mode."
    )

    parser.add_argument(
        '--service_type', 
        type=str,
        default="google", 
        help="Retriever type (chromadb, langchain, google)."
    )

    parser.add_argument(
        '--no_contexts', 
        default=False, 
        action='store_true', 
        help="Flag for enabling FactScore Zero, without contexts."
    )

    parser.add_argument(
        '--prompt_version', 
        type=str,
        default="v2", 
        help="Prompt version (v1 - original, v2 - enhanced)."
    )

    args = parser.parse_args()

    if args.test == True:
        test()
        sys.exit(0)

    option = "1" if args.prompt_version == "v1" else "2"

    # Create the atom extractor, atom reviser and context retriever
    context_retriever = ContextRetriever(service_type=args.service_type, top_k=5, cache_dir=args.cache_dir)
    atom_extractor = AtomExtractor(model=args.model)
    atom_reviser = AtomReviser(model=args.model)

    print(f"[FactScore] Processing input dataset: {args.input_file}")
    filename = args.input_file # a jsonl file

    with open(filename) as f:
        lines = f.read().splitlines()
    df_inter = pd.DataFrame(lines)
    df_inter.columns = ['json_element']
    df_inter['json_element'].apply(json.loads)
    df = pd.json_normalize(df_inter['json_element'].apply(json.loads))
    labeled_dataset = df.to_dict('records')
    f.close()

    print(f"[FactScore] Loading data from: {filename}")
    print(f"[FactScore] Found {len(labeled_dataset)} elements")

    # Check if previous results exist. If yes, load them and skip over them
    # when processing the input dataset.
    filename = "eval_results_factscore{}_{}_{}_{}.jsonl".format(
        option,
        args.service_type,
        args.dataset_name,
        args.model
    )
    output_filename = os.path.join(args.output_dir, filename)
    print(f"[FactScore] Reading previous results from: {output_filename}")
    evaluation_data = []
    if os.path.isfile(output_filename):
        with open(output_filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                evaluation_data.append(json.loads(line))

    print(f"[FactScore] Found {len(evaluation_data)} existing evaluations data.")
    for data in labeled_dataset: # 183 labeled bios (needs decontextualization)

        # Check if current data has been processed already
        processed = False
        for eval_data in evaluation_data:
            if eval_data["input"] == data["input"]:
                processed = True
                break
        if processed:
            print(f"[FactScore] Input {data} already processed.")
            continue

        # Process the data point with the FactScore pipeline
        pipeline = FactScore(
            context_retriever=context_retriever,
            atom_extractor=atom_extractor,
            atom_reviser=atom_reviser,
            model=args.model,
            prompt_version=args.prompt_version,
            add_topic=args.add_topic,
            binary_output=args.binary_output
        )

        # Load the problem instance from a file
        ok = pipeline.from_dict_with_contexts(data)
        if not ok:
            continue # annotations are null (ignore)

        # Build the FactScore pipeline 
        pipeline.build(
            has_atoms=True,
            has_contexts=True,
            decontextualize_atoms=False,
            no_contexts=args.no_contexts
        )

        results = pipeline.score()
        results["model_name"] = args.model
        evaluation_data.append(results)
        print(f"[FactScore] Results: {results}")

        # Save results to a file (progressively)
        filename = "eval_results_factscore{}_{}_{}_{}.jsonl".format(
            option,
            args.service_type,
            args.dataset_name,
            args.model
        )
        output_filename = os.path.join(args.output_dir, filename)
        print(f"[FactScore] Writing results to: {output_filename}")
        with open(output_filename, "w") as f:
            for res in evaluation_data:
                f.write(f"{json.dumps(res)}\n")
        f.close()

    print("Done.")
