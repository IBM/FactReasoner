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

# Our implementation of the VeriScore paper with open-source models (LLaMA)
# where facts/claims are checked with respect to contexts retrieved from an
# external source such as Wikipedia or Google Search (for the latter we consider
# the passage retrieved from the corresponding link).

import json
import asyncio
import time
import mellea.stdlib.functional as mfuncs

from typing import Any, Dict, List, Tuple
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
from fact_reasoner.utils import extract_last_square_brackets, LOOP_BUDGET

# Version 2 of the prompt (based on more recent work VeriScore, FactBench)
INSTRUCTION_VERISCORE = """

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
{{knowledge_text}}

STATEMENT:
{{atom_text}}
"""

class VeriScore:
    """
    Implementation of the VeriScore/FactBench paper. 

    Source:
        @misc{bayat2025factbench,
            title={FactBench: A Dynamic Benchmark for In-the-Wild Language Model Factuality Evaluation}, 
            author={Farima Fatahi Bayat and Lechen Zhang and Sheza Munir and Lu Wang},
            year={2025},
            eprint={2410.22257},
            archivePrefix={arXiv},
            primaryClass={cs.CL},
            url={https://arxiv.org/abs/2410.22257}, 
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
        Initialize the VeriScore pipeline.

        Args:
            backend: Backend
                The Mellea backend to use for LLM interactions.
            atom_extractor: Atomizer
                The atom decomposition component.
            atom_reviser: Reviser
                The atom reviser component.
            context_retriever: ContextRetriever
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
        self.binary_output = False # default is False
    
        print(f"[VeriScore] Using Mellea backend: {self.backend.model_id}")
        print(f"[VeriScore] Binary output: {self.binary_output}")

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
        Initialize VeriScore from a dict containing both atoms and contexts.

        Args:
            data: Dict[str, Any]
                The dict containing the problem instance.
        """

        self.query = data["input"]
        self.response = data["output"]
        self.topic = data.get("topic", None)
        
        print(f"[VeriScore] Reading the atoms ...")                
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

        print(f"[VeriScore] Atoms found: {len(self.atoms)}")
        for _, atom in self.atoms.items():
            print(f"[VeriScore] {atom}")
        
        self.labels_human = dict(zip(atom_ids, gold_labels))
        print(f"[VeriScore] Lables found: {self.labels_human}")

        print(f"[VeriScore] Reading the contexts ...")
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

        print(f"[VeriScore] Contexts found: {len(self.contexts)}")
        for aid, atom in self.atoms.items():
            ctxts = []
            for c in atom2contexts[aid]:
                ctxts.append(self.contexts[c])
                self.contexts[c].set_atom(atom)
            atom.add_contexts(ctxts)
        
        print(f"[VeriScore] Pipeline initialized with {len(self.atoms)} atoms and {len(self.contexts)} contexts.")

    def to_json(self, json_file_path: str = None) -> Dict[str, Any]:
        """
        Save the VeriScore instance to a JSON file.

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
            print(f"[VeriScore] Pipeline instance written to: {json_file_path}")

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

        print(f"[VeriScore] Building the pipeline ...")
        
        # Build the atoms 
        if has_atoms == False:
            self.atoms = build_atoms(
                response=self.response,
                atom_extractor=self.atom_extractor
            )
            self.revise_atoms = True # revise atoms if newly created
            print(f"[VeriScore] Extracted {len(self.atoms)} atoms.")
            for aid in self.atoms.keys():
                print(f"[VeriScore] {self.atoms[aid]}")

        assert len(self.atoms) > 0, \
            f"The atoms must be initialized before running the pipeline."

        # Revise the atoms
        if self.revise_atoms:
            print(f"[VeriScore] Revise the atoms ...")
            assert self.response is not None, f"The atom reviser requires a response."
            atom_ids = [aid for aid in sorted(self.atoms.keys())]
            old_atoms = [self.atoms[aid].get_text() for aid in atom_ids]
            result = asyncio.run(self.atom_reviser.run_batch(old_atoms, self.response))
            for i, aid in enumerate(atom_ids):
                elem = result[i]
                self.atoms[aid].set_text(elem["revised_unit"])
                print(f"[VeriScore] {self.atoms[aid]}")

        # Remove duplicated atoms (if any)
        self.atoms = remove_duplicated_atoms(self.atoms)
        print(f"[VeriScore] Created {len(self.atoms)} unique atoms.")

        # Build the contexts (per atom)
        if has_contexts == False: # check if contexts already in file
            self.contexts = build_contexts(
                atoms=self.atoms,
                retriever=self.context_retriever,
                use_fast_retriever=use_fast_retriever
            )

        print(f"[VeriScore] Retrieved {len(self.contexts)} contexts.")
        print(f"[VeriScore] Pipeline building completed.")

    def _get_label(self, output: ModelOutputThunk) -> str:
        """
        Extract the atom label from the generated text. We expect the label to
        be on the last line of the response, and be one of the following:
            [Supported], [Contradicted], [Unverifiable].
        We only consider [Supported]/S atoms, the others will be [NotSupported]/NS.
        """

        text = str(output)
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
                
    async def predict_atom_labels(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        For each atom predict its label given the corresponding retrieved contexts.
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
        print(f"[VeriScore] Labeling atoms with {self.backend.model_id} ...")

        # Create the prompts for each of the atoms
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

            # prepare the context
            knowledge_text = make_knowledge(passages)
            print(f"[VeriScore] Processing atom: ({aid}) {atom_text}")

            # Execute the instruction
            corutine = mfuncs.ainstruct(
                INSTRUCTION_VERISCORE,
                context=SimpleContext(),
                backend=self.backend,
                requirements=[
                    check(
                        "The output must contain [Supported], [Contradicted] or [Unverifiable]"
                    )
                ],
                user_variables={"atom_text": atom_text, "knowledge_text": knowledge_text},
                strategy=RejectionSamplingStrategy(loop_budget=LOOP_BUDGET),
                return_sampling_results=True
            )
            corutines.append(corutine)

        print(f"[VeriScore] Awaiting for the async execution ...")
        outputs = await asyncio.gather(*(corutines[i] for i in range(len(corutines))))
        for output in outputs:
            label = self._get_label(output.result)
            atom_labels.append(label)
            atom_outputs.append(str(output))

        # Return the labeled atoms
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

        # Run the VeriScore pipeline
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

        print(f"[VeriScore] Predictions: {labels}")
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
            print(f"[VeriScore] Gold labels: {self.labels_human}")
            print(f"[VeriScore] Predictions: {labels}")
            print(f"[VeriScore] Gold fscore: {fscore_gold} ({true_atoms}/{len(self.labels_human)})")
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
            print(f"[VeriScore] Gold labels: {self.labels_human}")
            print(f"[VeriScore] Predictions: {labels}")
            print(f"[VeriScore] Gold fscore: {fscore_gold} ({true_atoms}/{len(self.labels_human)})")
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
        print(f"[VeriScore] Elapsed time: {elapsed_time:.4f} seconds.")

        return results

