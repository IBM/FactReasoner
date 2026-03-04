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

import json
import asyncio
import string
import time
import mellea.stdlib.functional as mfuncs

from typing import List, Dict, Any, Tuple
from mellea.backends import Backend
from mellea.stdlib.context import SimpleContext
from mellea.core import ModelOutputThunk
from mellea.stdlib.requirements import check
from mellea.stdlib.sampling import RejectionSamplingStrategy
from mellea.core import FancyLogger

# Local imports
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.core.retriever import ContextRetrieverFast
from fact_reasoner.core.base import Atom, Context
from fact_reasoner.core.utils import build_atoms, build_contexts, remove_duplicated_atoms
from fact_reasoner.utils import LOOP_BUDGET

# Version 1 of the prompt (from the original FactScore paper)
INSTRUCTION_FACTSCORE = """
Answer the question about {{topic_text}} based on the given context.
Your answer must be either True or False. Do not add any other information.

{{knowledge_text}}

Input: {{atom_text}} True or False?
Output:
"""

INSTRUCTION_FACTSCORE_NOTOPIC = """
Answer the input question based on the given context.
Your answer must be either True or False. Do not add any other information.

{{knowledge_text}}

Input: {{atom_text}} True or False?
Output:
"""

class FactScore:
    """
    Implementation of the FactScore factuality assessor.

    Source:
        @misc{min2023factscore,
        title={FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation}, 
        author={Sewon Min and Kalpesh Krishna and Xinxi Lyu and Mike Lewis and Wen-tau Yih and Pang Wei Koh and Mohit Iyyer and Luke Zettlemoyer and Hannaneh Hajishirzi},
        year={2023},
        eprint={2305.14251},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2305.14251}, 
        }
    """

    def __init__(
            self,
            backend: Backend,
            atom_extractor: Atomizer = None,
            atom_reviser: Reviser = None,
            context_retriever: ContextRetrieverFast = None,
    ):
        """
        Initialize the FactScore pipeline.

        Args:
            backend: Backend
                The Mellea backend to use for LLM interactions.
            atom_extractor: Atomizer
                The atom decomposition component.
            atom_reviser: Reviser
                The atom reviser component.
            context_retriever: ContextRetriever or ContextRetrieverFast
                The service used for retrieving external contexts.
        """

        self.backend = backend
        self.query = None
        self.response = None
        self.topic = None
        self.start_time = time.perf_counter() # get the start time

        self.context_retriever = context_retriever
        self.atom_extractor = atom_extractor
        self.atom_reviser = atom_reviser
        self.binary_output = True # default is True
    
        print(f"[FactScore] Using Mellea backend: {self.backend.model_id}")
        print(f"[FactScore] Binary output: {self.binary_output}")

        self.atoms = {} # indexed by atom id
        self.contexts = {} # indexed by context id

        # Ground truth labels (if any)
        self.labels_human = None

        # Disable Mellea logging
        FancyLogger.get_logger().setLevel(FancyLogger.ERROR)

    def from_dict_with_contexts(
            self,
            data: Dict[str, Any],
    ):
        """
        Initialize FactScore from a dict containing both atoms and contexts.

        Args:
            data: Dict[str, Any]
                The dict containing the problem instance.
        """

        self.query = data["input"]
        self.response = data["output"]
        self.topic = data.get("topic", None)
        
        print(f"[FactScore] Reading the atoms ...")                
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
            print(f"[FactScore] {atom}")
        
        self.labels_human = dict(zip(atom_ids, gold_labels))
        print(f"[FactScore] Lables found: {self.labels_human}")

        print(f"[FactScore] Reading the contexts ...")
        for context_dict in data["contexts"]:
            cid = context_dict["id"]
            title = context_dict["title"]
            text = context_dict["text"]
            snippet = context_dict.get("snippet", "")
            link = context_dict.get("link", "")
            ctxt = Context(
                id=cid, 
                atom=None, 
                text=text, 
                title=title, 
                snippet=snippet, 
                link=link
            )
            self.contexts[cid] = ctxt

        print(f"[FactScore] Contexts found: {len(self.contexts)}")
        for aid, atom in self.atoms.items():
            ctxts = []
            for c in atom2contexts[aid]:
                ctxts.append(self.contexts[c])
                self.contexts[c].set_atom(atom)
            atom.add_contexts(ctxts)
        print(f"[FactScore] Pipeline initialized with {len(self.atoms)} atoms and {len(self.contexts)} contexts.")

    def to_json(self, json_file_path: str = None) -> Dict[str, Any]:
        """
        Save the FactScore instance to a JSON file.

        Args:
            json_file: str
                The path to the output JSON file.
        """

        data = {}
        data["input"] = self.query
        data["output"] = self.response.strip()
        data["topic"] = self.topic
        data["atoms"] = []
        data["contexts"] = []

        # Write the atoms
        for aid, atom in self.atoms.items():
            atom_data = dict(
                id=aid, text=atom.get_text(), contexts=list(atom.get_contexts().keys())
            )
            if atom.get_label() is not None:
                atom_data["label"] = atom.get_label()
            data["atoms"].append(atom_data)

        # Write the contexts
        data["contexts"] = [context.to_json() for context in self.contexts.values()]

        # Write to a JSON file (if any)
        if json_file_path:
            with open(json_file_path, "w") as f:
                json.dump(data, f, indent=4)
            f.close()
            print(f"[FactScore] Pipeline instance written to: {json_file_path}")

        return data

    def build(
            self,
            query: str = None,
            response: str = None,
            topic: str = None,
            has_atoms: bool = False,
            has_contexts: bool = False,
            revise_atoms: bool = False,
            use_fast_retriever: bool = True
    ):
        """
        Build the atoms and contexts using the retrieval service.

        Args:
            query: str
                The input user query.
            response: str
                The LLM generated response to the input query.
            topic: str
                The topic of the input query/response.
            has_atoms: bool
                A boolean flag indicating if the atoms have already been created.
            has_contexts: bool
                A boolean flag indicating if the contexts have already been created.
            revise_atoms: bool
                A boolean flag indicating that the atoms need to be decontextualized
                (i.e., pronouns he, she, it, ... replaced by the actual entity)
            use_fast_retriever: bool
                Use a fast multi-threaded context retriever if True, else the
                standard sequential retriever.
        """

        # Initialize the scorer
        if query is not None: 
            self.query = query
        if response is not None:
            self.response = response
        if topic is not None:
            self.topic = topic

        self.revise_atoms = revise_atoms

        # Safety checks
        assert self.atom_extractor is not None, \
            f"The atom extractor must be created."
        assert self.atom_reviser is not None, \
            f"The atom reviser must be created."

        print(f"[FactScore] Building the pipeline ...")
        
        # Build the atoms 
        if has_atoms == False:
            self.atoms = build_atoms(
                response=self.response,
                atom_extractor=self.atom_extractor
            )
            self.revise_atoms = True # revise atoms is newly created
            print(f"[FactScore] Extracted {len(self.atoms)} atoms.")
            for aid in self.atoms.keys():
                print(f"[FactScore] {self.atoms[aid]}")

        # Safety checks
        assert len(self.atoms) > 0, \
            f"The atoms must be initialized before running the pipeline."

        # Revise the atoms
        if self.revise_atoms:
            print(f"[FactScore] Revising the atoms ...")
            assert self.response is not None, f"The atom reviser requires a response."
            atom_ids = [aid for aid in sorted(self.atoms.keys())]
            old_atoms = [self.atoms[aid].get_text() for aid in atom_ids]
            result = asyncio.run(self.atom_reviser.run_batch(old_atoms, self.response))
            for i, aid in enumerate(atom_ids):
                elem = result[i]
                self.atoms[aid].set_text(elem["revised_unit"])
                print(f"[FactScore] {self.atoms[aid]}")

        # Remove duplicated atoms (if any)
        self.atoms = remove_duplicated_atoms(self.atoms)
        print(f"[FactScore] Created {len(self.atoms)} unique atoms.")

        # Build the contexts (per atom)
        if has_contexts == False: # check if contexts already in file
            self.contexts = build_contexts(
                atoms=self.atoms,
                retriever=self.context_retriever,
                use_fast_retriever=use_fast_retriever,
            )
        print(f"[FactScore] Retrieved {len(self.contexts)} contexts.")
        print(f"[FactScore] Pipeline building completed.")

    def _get_label(self, output: ModelOutputThunk) -> str:
        """
        Extract the atom label from the generated text. We expect the label to
        be on the last line of the response, and be one of the following:
            [Supported], [Contradicted], [Unverifiable].
        We only consider [Supported]/S atoms, the others will be [NotSupported]/NS.
        """

        text = str(output)
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
                
    async def predict_atom_labels(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        For each atom predict its label (S or NS) given the corresponding
        retrieved contexts.
        """

        # Safety checks
        assert len(self.atoms) > 0

        # Utility function to assemble the context of an atom
        def make_knowledge(passages: List[Dict[str, Any]]) -> str:
            knowledge = ""
            for _, psg in enumerate(passages):
                title = psg["title"]
                text = psg["text"]
                snippet = psg.get("snippet", "")
                knowledge += "Title: {}\nSummary: {}\nText: {}\n\n".format(title, snippet, text)

            return knowledge
        
        # Use the LLM to label the atom
        print(f"[FactScore] Labeling atoms with {self.backend.model_id} ...")

        # Label each atom given its retrieved contexts
        atom_ids = []
        atom_labels = []
        atom_outputs = []
        corutines = []
        for aid, atom in self.atoms.items():
            atom_ids.append(aid)
            atom_text = atom.get_text()
            contexts = atom.get_contexts()
            passages = []
            if contexts is not None and len(contexts) > 0:
                for _, c in contexts.items():
                    passages.append(dict(title=c.get_title(), text=c.get_text()))
            
            # Prepare the context
            knowledge_text = make_knowledge(passages)
            print(f"[FactScore] Processing atom: ({aid}) {atom_text}")

            # Prepare the instruction
            if self.topic is not None:
                instruction = INSTRUCTION_FACTSCORE
                user_variables = {
                    "topic_text": self.topic, 
                    "atom_text": atom_text, 
                    "knowledge_text": knowledge_text
                }
            else:
                instruction = INSTRUCTION_FACTSCORE_NOTOPIC
                user_variables = {
                    "atom_text": atom_text, 
                    "knowledge_text": knowledge_text
                }

            # Execute the instruction
            corutine = mfuncs.ainstruct(
                instruction,
                context=SimpleContext(),
                backend=self.backend,
                requirements=[
                    check(
                        "The output must contain the tokens True or False"
                    )
                ],
                user_variables=user_variables,
                strategy=RejectionSamplingStrategy(loop_budget=LOOP_BUDGET),
                return_sampling_results=True
            )
            corutines.append(corutine)
        
        print(f"[FactScore] Awaiting for the async execution ...")
        outputs = await asyncio.gather(*(corutines[i] for i in range(len(corutines))))
        for output in outputs:
            label = self._get_label(output.result)
            atom_labels.append(label)
            atom_outputs.append(str(output))

        # Return the labeled atoms (and also the outputs)
        return dict(zip(atom_ids, atom_labels)), dict(zip(atom_ids, atom_outputs))

    def score(self) -> Dict[str, Any]:
        """
        Compute the factuality score taking into consideration the contexts 
        retrieved for each of the atom in the answer.

        Factuality score = # atoms(true) / # atoms

        Intuitively, a score of 100% means that all atoms in the answer are
        factually correct. If none of them are correct, then the score is 0%. If
        only half of the atoms are correct, then the score is 50%.

        Returns:
            Dict[str, Any]: The results dictionary containing the factuality score i.e., a real value in [0, 1]
        """

        # Run the FactScore pipeline
        num_true_atoms = 0
        num_false_atoms = 0
        num_uniform_atoms = 0
        labels, raw_outputs = asyncio.run(self.predict_atom_labels())
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
      
        # Precision, R@K and F1@K
        fscore = float(num_true_atoms)/float(len(self.atoms))
        K = int(len(self.atoms) / 2) # K is assumed to be half
        recall_k = min(float(num_true_atoms)/K, 1.0)
        try:
            f1k = 2 * fscore * recall_k / (fscore + recall_k)
        except Exception as _:
            f1k = 0.0

        # Elapsed time
        elapsed_time = time.perf_counter() - self.start_time # elapsed time

        results = {}
        results["factuality_score"] = fscore
        results["recall_k"] = recall_k
        results["f1_k"] = f1k
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

        results["topic"] = self.topic
        results["query"] = self.query
        results["response"] = self.response
        results["elapsed_time"] = elapsed_time
        results["predictions"] = labels
        results["raw_outputs"] = raw_outputs
        print(f"[FactScore] Elapsed time: {elapsed_time:.4f} seconds.")

        return results

