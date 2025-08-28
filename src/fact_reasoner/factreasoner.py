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

# Factuality reasoner
import sys
import os
import json
import math
import torch
import argparse
import subprocess
import logging
import uuid
import pandas as pd

from pgmpy.models import MarkovNetwork
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.readwrite import UAIWriter
from pgmpy.global_vars import logger

logger.setLevel(logging.WARNING)

from collections import defaultdict
from dotenv import load_dotenv
from typing import Literal, cast

# Local
from src.fact_reasoner.argumentation_framework import ArgumentationFramework
from src.fact_reasoner.atom_extractor import AtomExtractor
from src.fact_reasoner.atom_reviser import AtomReviser
from src.fact_reasoner.context_retriever import ContextRetriever
from src.fact_reasoner.query_builder import QueryBuilder

from src.fact_reasoner.context_summarizer import ContextSummarizer
from src.fact_reasoner.nli_extractor import NLIExtractor, NLIExtractorOld

from src.fact_reasoner.fact_components import Atom, Context, Relation
from src.fact_reasoner.fact_graph import FactGraph
from src.fact_reasoner.fact_utils import (
    build_atoms,
    build_contexts,
    build_relations,
    build_markov_network,
    compute_factuality_results,
    is_relevant_context,
    remove_duplicated_contexts,
    remove_duplicated_atoms,
    run_merlin,
)

from src.fact_reasoner.utils import RITS_MODELS, DEFAULT_PROMPT_BEGIN, DEFAULT_PROMPT_END

# pgmpy set the root logger to INFO -- changed it to WARNING
import logging

os.environ["LITELLM_LOG"] = 'ERROR'
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("pgmpy").setLevel(logging.WARNING)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FactReasoner:
    def __init__(
            self,
            context_retriever: ContextRetriever = None,
            context_summarizer: ContextSummarizer = None,
            atom_extractor: AtomExtractor = None,
            atom_reviser: AtomReviser = None,
            nli_extractor: NLIExtractor = None,
            query_builder: QueryBuilder = None,
            merlin_path: str = None,
            debug_mode: bool = False,
            use_priors: bool = True,
    ):
        """
        Construct the FactReasoner pipeline.

        Args:
            context_retriever: ContextRetriever
                The service used for retrieving external contexts.
            context_summarizer: ContextSummarizer
                The service used for summarizing contexts.
            atom_extractor: AtomExtractor
                The service used for extracting atoms from the response.
            atom_reviser: AtomReviser
                The service used for decontextualizing the atoms.
            nli_extractor: NLIExtractor
                The service used for NLI relationship extraction.
            query_builder: QueryBuilder
                The query builder used to generating search queries for atoms.
            merlin_path: str
                Path to the Merlin probabilistic reasoning engine (c++ implementation).
            debug_mode: bool
                Flag indicating the debug mode (default is False).
            use_priors: bool
                Flag indicating that atom and context priors are used in the factor definition.
        """

        self.query = None
        self.response = None
        self.topic = None
        self.debug_mode = debug_mode
        self.use_priors = use_priors

        self.context_retriever = context_retriever
        self.context_summarizer = context_summarizer
        self.atom_extractor = atom_extractor
        self.atom_reviser = atom_reviser
        self.nli_extractor = nli_extractor
        self.merlin_path = merlin_path
        self.query_builder = query_builder

        # Inject the query builder into the context retriever
        if self.context_retriever is not None:
            self.context_retriever.set_query_builder(self.query_builder)

        # Safety checks
        assert self.merlin_path is not None, f"Path to `merlin` cannot be None."

        print(f"[FactReasoner] Using merlin at: {self.merlin_path}")
        print(f"[FactReasoner] Using atom/context priors: {self.use_priors}")

        self.atoms = {}  # indexed by atom id
        self.contexts = {}  # indexed by context id
        self.relations = []

        self.num_retrieved_contexts = 0
        self.num_summarized_contexts = 0

        # The fact graph and probabilistic model (Markov Network)
        self.fact_graph = None
        self.markov_network = None

        self.labels_human = None
        self.labels_chatgpt = None
        self.labels_llamanp = None

    def from_fact_graph(
            self,
            fact_graph: FactGraph
    ):
        """
        Create a FactReasoner instance from a FactGraph instance.

        Args:
            fact_graph: FactGraph
                A FactGraph instance.
        """

        # Create the atoms, contexts and relations
        self.atoms = {}
        self.contexts = {}
        self.relations = []

        for node_id, node in fact_graph.nodes.items():
            if node.type == "atom":
                self.atoms[node_id] = Atom(
                    id=node_id,
                    text=""
                )
            elif node.type == "context":
                self.contexts[node_id] = Context(
                    id=node_id,
                    atom=None,
                    text=""
                )

        for edge in fact_graph.edges:
            id_source = edge.source
            id_target = edge.target
            if id_source in self.atoms:
                src = self.atoms[id_source]
            elif id_source in self.contexts:
                src = self.contexts[id_source]
            if id_target in self.atoms:
                trg = self.atoms[id_target]
            elif id_target in self.contexts:
                trg = self.contexts[id_target]

            rel = Relation(
                source=src,
                target=trg,
                type=edge.type,
                probability=edge.probability,
                link=edge.link
            )

            self.relations.append(rel)

        # Create the corresponding fact graph
        self.fact_graph = fact_graph

        # Create the corresponding probabilistic model (Markov Network)
        assert self.fact_graph is not None, "The fact must be built before building the Markov network."
        self.markov_network = build_markov_network(
            self.fact_graph,
            use_priors=self.use_priors,
            debug_mode=self.debug_mode
        )

    def from_dict_with_contexts(
            self,
            data: dict,
    ):
        """
        Create a problem instance from a dict containing both atoms and contexts.

        Args:
            data: str
                The path to the json file containing the problem instance.
        """

        self.query = data["input"]
        self.response = data["output"]
        self.topic = data["topic"]

        print(f"[FactReasoner] Loading the atoms ...")
        gold_labels = []
        atom_ids = []
        self.atoms = {}
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
        print(f"[FactReasoner] Atoms found: {len(self.atoms.keys())}")
        for _, atom in self.atoms.items():
            print(atom)

        self.labels_human = dict(zip(atom_ids, gold_labels))
        print(f"[FactReasoner] Labels found: {self.labels_human}")

        print(f"[FactReasoner] Loading the contexts ...")
        for context_dict in data["contexts"]:
            cid = context_dict["id"]
            title = context_dict["title"]
            text = context_dict["text"]
            snippet = context_dict.get("snippet", "")
            link = context_dict.get("link", "")
            ctxt = Context(id=cid, atom=None, text=text, title=title, snippet=snippet, link=link)
            self.contexts[cid] = ctxt

        print(f"[FactReasoner] Contexts retrieved: {len(self.contexts.keys())}")
        for aid, atom in self.atoms.items():
            ctxts = []
            for c in atom2contexts[aid]:
                ctxts.append(self.contexts[c])
                self.contexts[c].set_atom(atom)
            atom.add_contexts(ctxts)
        return True

    def build(
            self,
            response: str = None,
            debug_mode: bool = False,
            has_atoms: bool = False,
            has_contexts: bool = False,
            revise_atoms: bool = True,
            remove_duplicates: bool = False,
            summarize_contexts: bool = False,
            contexts_per_atom_only: bool = False,
            rel_atom_context: bool = True,
            rel_context_context: bool = True,
            question: str = None,
            text_only: bool = True
    ):
        """
        Build the atoms and contexts using the retrieval service.

        Args:
            response: str
                The input LLM generated response
            debug_mode: bool
                Boolean flag indicating debugging mode (default False)
            has_atoms: bool
                Flag indicating if the atoms were previously initialized.
            has_contexts: bool
                Flag indicating is the contexts were previously initialized.
            revise_atoms: bool
                Flag indicating that the atoms will be revised (decontextualized).
            remove_duplicates: bool
                Flag indicating if duplicated contexts are to be removed.
            summarize_contexts: bool
                Flag indicating if contexts are to be summarized.
            contexts_per_atom_only: bool
                Flag indicating that only the contexts retrieved per atom will be used.
            rel_atom_context: bool (default is True)
                Flag indicating the presence of atom-to-context relationships.
            rel_context_context: bool (default is False)
                Flag indicating the presence of context-to-context relationships.
            question: str (default is None)
                If it is not None, it is a string used to retrieve contexts related to that string.
            text_only: bool (default is True)
                Flag indicating that contexts are text only. If False, then the
                contexts are (Title, Snippet, Link, Text).
        """

        # Initialize the reasoner
        self.fact_graph = None
        self.markov_network = None
        self.debug_mode = debug_mode
        self.response = response

        # Safety checks
        assert self.atom_extractor is not None, f"Atom extractor must be created."
        assert self.atom_reviser is not None, f"Atom reviser must be created."
        assert self.nli_extractor is not None, f"NLI extractor must be created."

        # Output some info
        print(f"[FactReasoner] Building the pipeline instance ...")
        print(f"[FactReasoner] Using text only contexts: {text_only}")
        
        # Stage 1: decompose the response into atomic units (Atomizer)
        if has_atoms == False:
            assert self.response is not None, f"Response cannot be None for decomposition!"
            self.atoms = build_atoms(
                response=self.response,
                atom_extractor=self.atom_extractor
            )

        # Safety checks
        assert (len(self.atoms.keys()) > 0 or not has_atoms), \
            f"Atoms must be initialized if `has_atoms` is True!"

        # Stage 2: revise the atomic units to be self-contained (Reviser)
        if revise_atoms:
            print(f"[FactReasoner] Revise the atoms ...")
            atom_ids = [aid for aid in sorted(self.atoms.keys())]
            old_atoms = [self.atoms[aid].get_text() for aid in atom_ids]
            result = self.atom_reviser.run(old_atoms, self.response)
            for i, aid in enumerate(atom_ids):
                elem = result[i]
                self.atoms[aid].set_text(elem["revised_atom"])
                print(self.atoms[aid])

        self.atoms = remove_duplicated_atoms(self.atoms)
        
        # Stage 3: Build contexts (ContextRtriever)
        if not has_contexts:
            self.contexts = build_contexts(
                atoms=self.atoms,
                question=question,
                retriever=self.context_retriever
            )

        # for tracking purposes
        self.num_retrieved_contexts = len(self.contexts.keys()) 

        # Safety checks
        assert (len(self.contexts.keys()) > 0 or not has_contexts), \
            f"Contexts must be initialized if `has_contexts` is True!"
        
        if remove_duplicates:
            self.contexts, self.atoms = remove_duplicated_contexts(self.contexts, self.atoms)
            print(f"[FactReasoner] Found {len(self.contexts.keys())} unique contexts.")

        # Summarize contexts given atoms       
        if summarize_contexts:
            print(f"[FactReasoner] Summarizing the contexts ...")
            for atom_id, atom in self.atoms.items():
                if len(atom.contexts.keys()) > 0:
                    contexts_ids, contexts =  zip(*atom.contexts.items()) 
                    # 1 round of summarization instead of 2 rounds
                    results = self.context_summarizer.run([context.get_snippet_and_text() for context in contexts], atom.text) 
                    # results2 = self.context_summarizer.run([result["summary"] for result in results], atom.text) 

                    # for context_id, result, result2 in zip(contexts_ids, results, results2):
                    for context_id, result in zip(contexts_ids, results):
                        is_relevant = is_relevant_context(result["summary"])
                        # if result2["summary"] != "":
                        if result["summary"] != "" and is_relevant:
                            # self.contexts[context_id].set_synthetic_summary(result2["summary"])
                            self.contexts[context_id].set_synthetic_summary(result["summary"])
                            # update prior probability of context based on the confidence estimation of the summary
                            # self.contexts[context_id].set_probability(result["probability"] * result2["probability"] * self.contexts[context_id].get_probability())
                            self.contexts[context_id].set_probability(result["probability"] * self.contexts[context_id].get_probability())
                        else:
                            # we remove the context because it is not related to the atom
                            del self.contexts[context_id]
                            del self.atoms[atom_id].contexts[context_id]

            # summarize contexts for question
            
            # first get contexts for the question
            c_qs = {c_id: context for c_id, context in self.contexts.items() if c_id.startswith("c_q")} 
            if len(c_qs.keys()) > 0:
                contexts_ids, contexts =  zip(*c_qs.items()) 
                # 1 round of summarization instead of 2 rounds
                results = self.context_summarizer.run([context.get_snippet_and_text() for context in contexts], question) 
                # results2 = self.context_summarizer.run([result["summary"] for result in results], question) 

                # for context_id, result, result2 in zip(contexts_ids, results, results2):
                for context_id, result in zip(contexts_ids, results):
                    is_relevant = is_relevant_context(result["summary"])
                    # if result2["summary"] != "":
                    if result["summary"] != "" and is_relevant:
                        # self.contexts[context_id].set_synthetic_summary(result2["summary"])
                        self.contexts[context_id].set_synthetic_summary(result["summary"])
                        # update prior probability of context based on the confidence estimation of the summary
                        # self.contexts[context_id].set_probability(result["probability"] * result2["probability"] * self.contexts[context_id].get_probability())
                        self.contexts[context_id].set_probability(result["probability"] * self.contexts[context_id].get_probability())
                    else:
                        # we remove the context because it is not related to the atom
                        del self.contexts[context_id]
                              
        else:
            for context_id in self.contexts.keys():
                self.contexts[context_id].set_synthetic_summary(self.contexts[context_id].get_snippet_and_text()) 

        # for tracking purposes
        self.num_summarized_contexts = len(self.contexts.keys()) 

        # Stage 4: Extract NLI relationships (Evaluator)
        if self.num_summarized_contexts > 0 and len(self.atoms.keys()) > 0:
            # Build the NLI relationships
            self.relations = build_relations(
                atoms=self.atoms,
                contexts=self.contexts,
                rel_atom_context=rel_atom_context,
                rel_context_context=rel_context_context,
                contexts_per_atom_only=contexts_per_atom_only,
                nli_extractor=self.nli_extractor,
                text_only=text_only,
            )
    
            # Build the fact graph and Markov network
            print(f"[FactReasoner] Building the graphical model ...")
            self._build_fact_graph()
            assert self.fact_graph is not None, "The fact must be built before building the Markov network."
            self.markov_network = build_markov_network(
                self.fact_graph,
                use_priors=self.use_priors,
                debug_mode=self.debug_mode
            )

            print(f"[FactReasoner] Pipeline instance created.")
        elif self.num_summarized_contexts == 0 and len(self.atoms.keys()) == 0:
            print(f"[FactReasoner] Could not create fact graph because no relevant contexts were retrieved and no atoms are available.")
        elif self.num_summarized_contexts == 0:
            print(f"[FactReasoner] Could not create fact graph because no relevant contexts were retrieved.")
        else:
            print(f"[FactReasoner] Could not create fact graph because no atoms are available.")

    def dump(self):
        """
        Dump the content of the fact reasoner for debugging purposes.
        """

        print("Atoms:")
        for _, atom in self.atoms.items():
            print(atom)
        print("Contexts:")
        for _, context in self.contexts.items():
            print(context)
        print("Relations:")
        for rel in self.relations:
            print(rel)

    def _build_fact_graph(self):
        """
        Create the fact graph representation from atoms, contexts and relations.
        """
        self.fact_graph = FactGraph(
            atoms=list(self.atoms.values()),
            contexts=list(self.contexts.values()),
            relations=self.relations
        )

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
                The results dictionary containing the marginals, factuality score i.e., a real value in [0, 1]
        """
        # Safety checks
        if len(self.atoms.keys()) == 0:
            print("WARNING: no atoms have been identified!")
        if len(self.contexts.keys()) == 0:
            print("WARNING: no contexts have been retrieved!")
        if len(self.relations) == 0:
            print("WARNING: no relationships have been identified!")

        assert self.fact_graph is not None
        assert self.markov_network is not None

        marginals = run_merlin(list(self.atoms.keys()), self.markov_network, self.merlin_path)
        scores = {
            m["variable"]: (m["probabilities"][0], m["probabilities"][1]) for m in marginals
        }
        results = compute_factuality_results(
            scores=scores,
            fact_graph=self.fact_graph,
            atoms=self.atoms,
            contexts=self.contexts,
            query=self.query,
            labels_human=self.labels_human
        )
        return results, marginals
    
    def score_arg(self, semantics: Literal["qe", "dfquad"]):
        """
        Compute the factuality score and other results using computational argumentation.
        
        Returns:
            dict
                The results dictionary containing the factuality score (real values in [0, 1]) and other measures.
        """
        assert self.fact_graph is not None

        af = ArgumentationFramework(self.fact_graph)
        strengths = af.evaluate_strengths(semantics=semantics)
        scores = {
            v: (1 - s, s) for v, s in strengths.items()
        }
        results = compute_factuality_results(
            scores=scores,
            fact_graph=self.fact_graph,
            atoms=self.atoms,
            contexts=self.contexts,
            query=self.query,
            labels_human=self.labels_human
        )
        return results, strengths


def test():
    model = "llama-3.1-70b-instruct"
    cache_dir = "my_database.db"

    context_retriever = ContextRetriever(service_type="google", top_k=5, cache_dir=cache_dir)
    context_summarizer = ContextSummarizer(model=model, prompt_version="v1")
    query_builder = QueryBuilder(model=model, prompt_version="v1")
    atom_extractor = AtomExtractor(model)
    atom_reviser = AtomReviser(model)
    # nli_extractor = NLIExtractor(model, prompt_version="v1")
    nli_extractor = NLIExtractorOld(model, prompt_version="v2")

    merlin_path = "/home/radu/git/fm-factual/lib/merlin"

    # Create the FactReasoner pipeline
    pipeline = FactReasoner(
        context_retriever=context_retriever,
        context_summarizer=context_summarizer,
        atom_extractor=atom_extractor,
        atom_reviser=atom_reviser,
        nli_extractor=nli_extractor,
        query_builder=query_builder,
        merlin_path=merlin_path,
    )

    # Load the problem instance from a file
    json_file = "/home/radu/git/fm-factual/examples/test.json"
    with open(json_file, "r") as f:
        data = json.load(f)

    pipeline.from_dict_with_contexts(data)

    # Build the FactReasoner pipeline
    pipeline.build(
        has_atoms=True,
        has_contexts=True,
        revise_atoms=False,
        remove_duplicates=True,
        contexts_per_atom_only=False,
        rel_atom_context=True, 
        rel_context_context=True,
        text_only=False
    )

    results, marginals = pipeline.score()
    print(f"[FactReasoner] Marginals: {marginals}")
    print(f"[FactReasoner] Results: {results}")
    print(f"Done.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_file',
        type=str,
        default=None,
        help="Path to the input dataset (jsonl)."
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
        '--service_type',
        type=str,
        default="google",
        help="Service type (langchain, chromadb, google)."
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help="Name of the RITS model used internally"
    )

    parser.add_argument(
        '--version',
        type=int,
        default=1,
        help="FactReasoner version: 1, 2 or 3"
    )

    parser.add_argument(
        '--top_k', 
        type=int, 
        default=3, 
        help="Top k results retrieved as contexts per atom."
    )

    parser.add_argument(
        '--use_priors', 
        default=False, 
        action='store_true', 
        help="Use the atom and context priors in the factor definition."
    )

    parser.add_argument(
        '--use_query_builder', 
        default=False, 
        action='store_true', 
        help="Use the QueryBuilder to generate queries for Google search."
    )

    parser.add_argument(
        '--text_only', 
        default=False, 
        action='store_true', 
        help="Contexts are considered text only."
    )

    parser.add_argument(
        '--nli_prompt_version', 
        type=str, 
        default="v1", 
        help="NLI prompt version: v1 (original) or v2 (more recent - some reasoning)"
    )

    parser.add_argument(
        '--atomizer_prompt_version', 
        type=str, 
        default="v2", 
        help="Atomizer prompt version: v1 (original) or v2 (newer)"
    )

    parser.add_argument(
        '--reviser_prompt_version', 
        type=str, 
        default="v1", 
        help="Reviser prompt version: v1 (newer) or v2 (original)"
    )

    parser.add_argument(
        '--test', 
        default=False, 
        action='store_true', 
        help="Debugging mode."
    )

    parser.add_argument(
        '--bert_nli',
        default=False,
        action='store_true',
        help="A BERT model (roberta) is used for NLI extraction."
    )

    parser.add_argument(
        '--merlin_path',
        type=str,
        default="/home/radu/git/fm-factual/lib/merlin",
        help="Path to the probabilistic inference merlin."
    )

    parser.add_argument(
        '--arg-semantics',
        type=str,
        nargs="+",
        required=False,
        choices=["qe", "dfquad"],
        default=[],
    )

    args = parser.parse_args()

    if args.test:
        test()
        sys.exit(0)

    # FactReasoner versions:
    if args.version == 1:  # 1 - context-atom relationships only, allow duplicated contexts
        rel_context_context = False
        remove_duplicates = False
        contexts_per_atom_only = True
        option = "1"
    elif args.version == 2:  # 2 - context-atom relationships only, no duplicated contexts
        rel_context_context = False
        remove_duplicates = True
        contexts_per_atom_only = False
        option = "2"
    elif args.version == 3:  # 3 - context-atom and context-context relationships, no duplicated contexts
        rel_context_context = True
        remove_duplicates = True
        contexts_per_atom_only = False
        option = "3"
    else:
        raise ValueError(f"Unknown FactReasoner version: {args.version}")

    # Get the NLI prompt version
    nli_prompt_version = args.nli_prompt_version

    # Create context retriever
    context_retriever = None

    # Create the atom extractor
    atom_extractor = AtomExtractor(
        model=args.model, 
        prompt_version=args.atomizer_prompt_version
    )
    
    # Create the atom reviser
    atom_reviser = AtomReviser(
        model=args.model, 
        prompt_version=args.reviser_prompt_version
    )

    # Create the NLI extractor
    if not args.bert_nli:
        nli_extractor = NLIExtractor(model=args.model, prompt_version=nli_prompt_version)
        nli_model_name = args.model
    else:  # BERT based NLI extraction
        nli_extractor = NLIExtractor(model="roberta", is_bert=True)
        nli_model_name = "roberta"

    # Create the query builder
    if args.use_query_builder:
        query_builder = QueryBuilder(model=args.model)
    else:
        query_builder = None

    arg_semantics = list(set(args.arg_semantics))
    arg_data = defaultdict(list)
    arg_filenames = {
        s: "eval_results_factreasoner{}_{}_{}_{}_arg_{}.jsonl".format(
            option,
            args.service_type,
            args.dataset_name,
            nli_model_name,
            s
        ) for s in arg_semantics
    }

    print(f"[FactReasoner] Processing input dataset: {args.input_file}")
    filename = args.input_file  # a jsonl file

    with open(filename) as f:
        lines = f.read().splitlines()
    df_inter = pd.DataFrame(lines)
    df_inter.columns = ['json_element']
    df_inter['json_element'].apply(json.loads)
    df = pd.json_normalize(df_inter['json_element'].apply(json.loads))
    dataset = df.to_dict('records')

    print(f"[FactReasoner] Loading data from: {filename}")
    print(f"[FactReasoner] Found {len(dataset)} elements")

    # Check if previous results exist. If yes, load them and skip over them
    # when processing the input dataset.
    filename = "eval_results_factreasoner{}_{}_{}_{}.jsonl".format(
        option,
        args.service_type,
        args.dataset_name,
        nli_model_name
    )
    output_filename = os.path.join(args.output_dir, filename)
    print(f"[FactReasoner] Reading previous results from: {output_filename}")
    evaluation_data = []
    if os.path.isfile(output_filename):
        with open(output_filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                evaluation_data.append(json.loads(line))

    print(f"[FactReasoner] Found {len(evaluation_data)} existing evaluations data.")

    for semantics in arg_semantics:
        output_path = os.path.join(args.output_dir, arg_filenames[semantics])
        if os.path.isfile(output_path):
            with open(output_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    arg_data[semantics].append(json.loads(line))

        print(f"[FactReasoner] Found {len(arg_data[semantics])} existing arg evaluations data.")

    # Loop over the data points in the dataset
    for input_data in dataset:
        # Check if current data has been processed already
        processed = False
        for eval_data in evaluation_data:
            if eval_data["input"] == input_data["input"]:
                processed = True
                break
        arg_processed = defaultdict(bool)
        for semantics in arg_semantics:
            for eval_data in arg_data[semantics]:
                if eval_data["input"] == input_data["input"]:
                    arg_processed[semantics] = True
                    break
        if processed and all(arg_processed.values()):
            prompt = input_data["input"]
            print(f"[FactReasoner] Input: {prompt} already processed.")
            continue

        # Process the data point with the FactReasoner pipeline
        pipeline = FactReasoner(
            context_retriever=context_retriever,
            atom_extractor=atom_extractor,
            atom_reviser=atom_reviser,
            nli_extractor=nli_extractor,
            query_builder=query_builder,
            merlin_path=args.merlin_path,
            use_priors=args.use_priors
        )

        # Load the problem instance from a file or dict
        ok = pipeline.from_dict_with_contexts(input_data)
        if not ok:
            continue  # annotations are null (ignore)

        # Build the FactReasoner pipeline
        pipeline.build(
            remove_duplicates=remove_duplicates,
            contexts_per_atom_only=contexts_per_atom_only,
            has_atoms=True,
            has_contexts=True,
            revise_atoms=False,
            rel_atom_context=True,
            rel_context_context=rel_context_context,
            text_only=args.text_only
        )

        if not processed:
            results, marginals = pipeline.score()
            results["model_name"] = args.model
            evaluation_data.append(results)
            print(f"[FactReasoner] Marginals: {marginals}")
            print(f"[FactReasoner] Results: {results}")

            # Save results to a file
            filename = "eval_results_factreasoner{}_{}_{}_{}.jsonl".format(
                option,
                args.service_type,
                args.dataset_name,
                nli_model_name
            )
            output_filename = os.path.join(args.output_dir, filename)
            print(f"[FactReasoner] Writing results to: {output_filename}")
            with open(output_filename, "w") as f:
                for res in evaluation_data:
                    f.write(f"{json.dumps(res)}\n")

        for semantics in arg_semantics:
            if not arg_processed[semantics]:
                arg_results, arg_scores = pipeline.score_arg(semantics=semantics)
                arg_results["model_name"] = args.model
                arg_data[semantics].append(arg_results)
                print(f"[FactReasoner] Arg Scores ({semantics}): {arg_scores}")
                print(f"[FactReasoner] Arg Results ({semantics}): {arg_results}")

                # Save results to a file
                output_path = os.path.join(args.output_dir, arg_filenames[semantics])
                print(f"[FactReasoner] Writing arg results ({semantics}) to: {output_path}")
                with open(output_path, "w") as f:
                    for res in arg_data[semantics]:
                        f.write(f"{json.dumps(res)}\n")

    print("Done.")
