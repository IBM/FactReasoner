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
import string
import mellea.stdlib.functional as mfuncs

from typing import List, Dict, Any, Tuple
from mellea.backends import Backend
from mellea.backends.types import ModelOption
from mellea.stdlib.base import SimpleContext, ModelOutputThunk
from mellea.stdlib.requirement import check
from mellea.stdlib.sampling import RejectionSamplingStrategy

# Local imports
from src.fact_reasoner.core.atomizer import Atomizer
from src.fact_reasoner.core.reviser import Reviser
from src.fact_reasoner.core.retriever import ContextRetriever
from src.fact_reasoner.core.query_builder import QueryBuilder
from src.fact_reasoner.core.utils import Atom, Context, build_atoms, build_contexts, remove_duplicated_atoms
from src.fact_reasoner.utils import LOOP_BUDGET

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
            context_retriever: ContextRetriever = None,
            atom_extractor: Atomizer = None,
            atom_reviser: Reviser = None,
    ):
        """
        Initialize the FactScore pipeline.

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
                If True, then the topic is added (relevant only for Biographies).
        """

        self.backend = backend
        self.query = None
        self.response = None
        self.topic = None

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

    def from_json(self, json_file: str):
        """
        Initialize FactScore from a json file containing both atoms and contexts.

        Args:
            json_file: str
                The path to the json file containing the problem instance.
        """
        
        print(f"[FactScore] Reading JSON instance from: {json_file}")
        with open(json_file) as f:
            data = json.load(f)
            f.close()

        # Get the query, response and topic
        self.query = data["query"]
        self.response = data["response"]
        self.topic = data.get("topic", None)

        # Get the atoms
        for atom_dict in data["atoms"]:
            aid = atom_dict["id"]
            text = atom_dict["text"]
            a = Atom(id=aid, text=text)
            self.atoms[aid] = a
        
        print(f"[FactScore] Atoms found: {len(self.atoms)}")

        # Get the contexts
        for context_dict in data["contexts"]:
            cid = context_dict["id"]
            aid = context_dict["atom_id"]
            text = context_dict["text"]

            a = self.atoms[aid]
            ctxt = Context(
                id=cid, 
                atom=a, 
                text=text, 
                title="", 
                snippet="", 
                link=""
            )
            a.add_context(ctxt)
            self.contexts[cid] = ctxt

        print(f"[FactScore] Contexts found: {len(self.contexts)}")

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
            revise_atoms: bool = False
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
        self.query = query
        self.response = response
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
            result = self.atom_reviser.run(old_atoms, self.response)
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
                
    def predict_atom_labels(self) -> Tuple[Dict[str, str], Dict[str, str]]:
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
            output = mfuncs.instruct(
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
        labels, raw_outputs = self.predict_atom_labels()
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
      
        # Precision i.e., factuality score
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
        results["predictions"] = labels
        results["raw_outputs"] = raw_outputs

        return results

if __name__ == "__main__":

    # Example query and response
    query = "Tell me a biography of Lanny Flaherty"
    response = "Lanny Flaherty is an American actor born on December 18, 1949, in Pensacola, Florida. He has appeared in numerous films, television shows, and theater productions throughout his career, which began in the late 1970s. Some of his notable film credits include \"King of New York,\" \"The Abyss,\" \"Natural Born Killers,\" \"The Game,\" and \"The Straight Story.\" On television, he has appeared in shows such as \"Law & Order,\" \"The Sopranos,\" \"Boardwalk Empire,\" and \"The Leftovers.\" Flaherty has also worked extensively in theater, including productions at the Public Theater and the New York Shakespeare Festival. He is known for his distinctive looks and deep gravelly voice, which have made him a memorable character actor in the industry."
    topic = "Lanny Flaherty"
    init_from_file = False

    # Create a Mellea RITS backend
    from mellea_ibm.rits import RITSBackend, RITS
    backend = RITSBackend(
        RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 4096},
    )

    # Set cache dir for context retriever
    cache_dir = None # "/home/radu/data/cache"

    # Create the retriever, atomizer and reviser.
    qb = QueryBuilder(backend)
    atom_extractor = Atomizer(backend)
    atom_reviser = Reviser(backend)
    context_retriever = ContextRetriever(
        service_type="google", 
        top_k=5, 
        cache_dir=cache_dir, 
        fetch_text=True, 
        query_builder=qb
    )

    # Create the FactScore pipeline
    pipeline = FactScore(
        backend=backend,
        context_retriever=context_retriever,
        atom_extractor=atom_extractor,
        atom_reviser=atom_reviser,
    )

    # Load the problem instance from a file
    if init_from_file:
        json_file = "/home/radu/storage/git/FactReasoner/examples/flaherty_wikipedia.json"
        with open(json_file, "r") as f:
            data = json.load(f)
        
        # Load the file (json)
        print(f"[FactScore] Initializing pipeline from: {json_file}")
        pipeline.from_dict_with_contexts(data)

        # Build the scorer
        pipeline.build(
            has_atoms=True,
            has_contexts=True,
            revise_atoms=False
        )
    else:
        pipeline.build(
            query=query,
            response=response,
            topic=topic,
            has_atoms=False,
            has_contexts=False,
            revise_atoms=True
        )

    # Print the results
    results = pipeline.score()
    print(f"[FactScore] Results: {results}")

    # Save the pipeline to a JSON file
    output_file = "/home/radu/storage/git/FactReasoner/examples/test.json"
    output = pipeline.to_json()
    output["results"] = results
    with open(output_file, "w") as fp:
        json.dump(output, fp, indent=4)
    print(f"Done.")

