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

import os
import json
import string
import argparse
import mellea.stdlib.functional as mfuncs

from typing import List, Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv

from mellea.backends import Backend
from mellea.backends.types import ModelOption
from mellea.stdlib.base import SimpleContext, CBlock
from mellea.stdlib.requirement import req, check, simple_validate
from mellea.stdlib.sampling import RejectionSamplingStrategy

# Local imports
from src.fact_reasoner.core.atomizer import Atomizer
from src.fact_reasoner.core.reviser import Reviser
from src.fact_reasoner.core.retriever import ContextRetriever
from src.fact_reasoner.fact_utils import Atom, Context, build_atoms, build_contexts

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

class FactScore:
    """
    Our implementation of the FactScore factuality assessor. 
    """

    def __init__(
            self,
            backend: Backend,
            context_retriever: ContextRetriever = None,
            atom_extractor: Atomizer = None,
            atom_reviser: Reviser = None,
            add_topic: bool = False,
    ):
        """
        Construct the FactScore pipeline instance.

        Args:
            backend: Backend
                The Mellea backend to use for LLM interactions.
            context_retriever: ContextRetriever
                The service used for retrieving external contexts.
            atom_extractor: Atomizer
                The atom decomposition component.
            atom_reviser: Reviser
                The atom reviser component.
            add_topic: bool
                If True, then the topic is added (relevant only for v1 and Biographies).
        """

        self.backend = backend
        self.query = None
        self.response = None
        self.topic = None
        self.add_topic = add_topic # default is False

        self.context_retriever = context_retriever
        self.atom_extractor = atom_extractor
        self.atom_reviser = atom_reviser
        self.binary_output = True # default is True
    
        if not os.environ.get("_DOTENV_LOADED"):
            load_dotenv(override=True) 
            os.environ["_DOTENV_LOADED"] = "1"
         
        print(f"[FactScore] Using Mellea backend: {self.backend.model_id}")
        print(f"[FactScore] Binary output: {self.binary_output}")

        self.atoms = {} # indexed by atom id
        self.contexts = {} # indexed by context id

        # Ground truth labels (if any)
        self.labels_human = None

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
            data: Dict[str, Any],
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
            query: str = None,
            response: str = None,
            has_atoms: bool = False,
            has_contexts: bool = False,
            revise_atoms: bool = True,
            no_contexts: bool = False
    ):
        """
        Build the atoms and contexts using the retrieval service.

        Args:
            query: str
                The input user query.
            response: str
                The LLM generated response to the input query.
            has_atoms: bool
                A boolean flag indicating if the atoms have already been created.
            has_contexts: bool
                A boolean flag indicating if the contexts have already been created.
            revise_atoms: bool
                A boolean flag indicating that the atoms need to be decontextualized
                (i.e., pronouns he, she, it, ... replaced by the actual entity)
            no_contexts: bool
                A boolean flag indicating if contexts are to be retrieved or not.
                If True, then we run a version that only leverages the internal
                knowledge of the language model.
        """

        # Initialize the scorer
        self.query = query
        self.response = response
        self.no_contexts = no_contexts
        self.revise_atoms = revise_atoms

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
        if revise_atoms:
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
            passages: List[Dict[str, Any]],
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
            knowledge += "Title: {}\nSummary: {}\nText: {}\n\n".format(title, snippet, text)

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

        return prompt

    def extract_label(self, text: str) -> str:
        """
        Extract the atom label from the generated text. We expect the label to
        be on the last line of the response, and be one of the following:
            [Supported], [Contradicted], [Unverifiable].
        We only consider [Supported]/S atoms, the others will be [NotSupported]/NS.
        """
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
                
    def predict_atom_labels(self) -> dict:
        """
        Use a strong LLM to predict the label S or NS of an atom given its contexts.
        """

        assert len(self.atoms) > 0

        # Use the LLM to label the atom
        print(f"[FactScore] Labeling atoms with {self.backend.model_id} ...")
        atom_ids = []

        # Label each atom given its retrieved contexts
        results = []
        for aid, atom in self.atoms.items():
            atom_ids.append(aid)
            atom_text = atom.get_text()
            contexts = atom.get_contexts()
            if contexts is not None and len(contexts) > 0:
                passages = []
                for cid, c in contexts.items():
                    if len(c.get_text()) == 0:
                        passages.append(dict(title=c.get_title(), text=c.get_snippet()))
                    else:
                        passages.append(dict(title=c.get_title(), text=c.get_text()))
            else:
                passages = [] # no passages retrieved for the atom

            output = mfuncs.instruct(
                INSTRUCTION_FACTSCORE,
                context=SimpleContext(),
                backend=self.backend,
                requirements=[
                    check(
                        "The output must be True or False"
                    )
                ],
                user_variables={"atom": atom_text, "knowledge": knowledge},
                strategy=RejectionSamplingStrategy(loop_budget=3),
                return_sampling_results=True
            )

            if output.success:
                label = extract_label(str(output))
                atom_labels.append(label)
    
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

if __name__ == "__main__":

    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_file',
        type=str,
        default=None,
        required=True,
        help="Path to the input test file (json)."
    )

    # Parse CLI arguments
    args = parser.parse_args()

    # Define the model and backend
    model_id = "llama-3.3-70b-instruct"
    backend = "rits"
    cache_dir = None # "/home/radu/data/cache"

    # Create the retriever, atomizer and reviser.
    context_retriever = ContextRetriever(service_type="google", top_k=5, cache_dir=cache_dir)
    atom_extractor = AtomExtractor(model_id=model_id, backend=backend)
    atom_reviser = AtomReviser(model_id=model_id, backend=backend)

    # Create the FactScore pipeline
    pipeline = FactScore(
        context_retriever=context_retriever,
        atom_extractor=atom_extractor,
        atom_reviser=atom_reviser,
        model_id=model_id,
        add_topic=True,
        backend=backend,  # Use RITS for the LLM
    )

    # Load the problem instance from a file
    assert args.input_file is not None, f"Input file cannot be None. Aborting."
    json_file = args.input_file
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Load the file (json)
    print(f"[FactScore] Initializing pipeline from: {json_file}")
    pipeline.from_dict_with_contexts(data)

    # Build the scorer
    pipeline.build(
        has_atoms=True,
        has_contexts=True,
        decontextualize_atoms=False
    )

    # Print the results
    results = pipeline.score()
    print(f"[FactScore] Results: {results}")
    print(f"Done.")

