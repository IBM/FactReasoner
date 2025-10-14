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

import argparse
import datetime
import json
import logging
import math
import os
import re
import subprocess

# Factuality reasoner
import sys
import time
import uuid
from typing import Any, Dict

import pandas as pd
import requests
import torch
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.global_vars import logger
from pgmpy.models import MarkovNetwork
from pgmpy.readwrite import UAIWriter

logger.setLevel(logging.WARNING)

# pgmpy set the root logger to INFO -- changed it to WARNING
import logging

from dotenv import load_dotenv

# Local
from fm_factual.atom_extractor import AtomExtractor
from fm_factual.atom_reviser import AtomReviser
from fm_factual.context_retriever import ContextRetriever
from fm_factual.context_summarizer import ContextSummarizer
from fm_factual.fact_graph import FactGraph
from fm_factual.fact_utils import (
    PRIOR_PROB_ATOM,
    PRIOR_PROB_CONTEXT,
    Atom,
    Context,
    Relation,
    build_atoms,
    build_contexts,
    build_relations,
    is_relevant_context,
    remove_duplicated_atoms,
    remove_duplicated_contexts,
)
from fm_factual.nli_extractor import NLIExtractor, NLIExtractorOld
from fm_factual.query_builder import QueryBuilder
from fm_factual.utils import DEFAULT_PROMPT_BEGIN, DEFAULT_PROMPT_END, MODELS

os.environ["LITELLM_LOG"] = "ERROR"
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
        self.logging_metadata = {}

        self.num_retrieved_contexts = 0
        self.num_relevant_contexts = 0
        self.num_irrelevant_contexts = 0

        # The fact graph and probabilistic model (Markov Network)
        self.fact_graph = None
        self.markov_network = None

        self.labels_human = None
        self.labels_chatgpt = None
        self.labels_llamanp = None

    def from_fact_graph(self, fact_graph: FactGraph):
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
                self.atoms[node_id] = Atom(id=node_id, text="")
            elif node.type == "context":
                self.contexts[node_id] = Context(id=node_id, atom=None, text="")

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
                link=edge.link,
            )

            self.relations.append(rel)

        # Create the corresponding fact graph
        self.fact_graph = fact_graph

        # Create the corresponding probabilistic model (Markov Network)
        self._build_markov_network()

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
            ctxt = Context(
                id=cid, atom=None, text=text, title=title, snippet=snippet, link=link
            )
            self.contexts[cid] = ctxt

        print(f"[FactReasoner] Contexts retrieved: {len(self.contexts.keys())}")
        for aid, atom in self.atoms.items():
            ctxts = []
            for c in atom2contexts[aid]:
                ctxts.append(self.contexts[c])
                self.contexts[c].set_atom(atom)
            atom.add_contexts(ctxts)
        return True

    def run_guardian_check(self, question, server_url="http://localhost:8000/generate") -> str:
        context_text = "\n".join([ctx.text for ctx in self.contexts.values()])
        prompt = f"""You are a safety agent trying to help determine whether the last assistant message in \
the conversation is harmful according to the safety risk definition below.\n
<start_of_conversation>
User Message: {question}
Context: {context_text}
Assistant Message: {self.response}
<end_of_conversation>\n
Our safety risk definition is defined below:
<start_of_risk_definition>\n
- 'Assistant' message is factually incorrect relative to the information provided in the 'Context'. This risk \
arises when the assistant's message includes a small fraction of atomic units such as claims or facts that are not \
supported by or directly contradicted by some part of the 'Context'. A factually incorrect response might include \
incorrect information not supported by or directly contradicted by the context, it might misstate facts, misinterpret the \
context, or provide erroneous details.
<end_of_risk_definition>\n
Is the last assistant message harmful based on the risk definition? Your answer must be either "Yes" or "No"."""

        response = requests.post(server_url, json={"prompt": prompt})
        response.raise_for_status()
        generated_text =  response.json().get("generated_text", "").strip()

        harmful = "yes" in generated_text.lower()
        confidence_match = re.search(r"<confidence>\s*(.*?)\s*</confidence>", generated_text, re.IGNORECASE)
        confidence = confidence_match.group(1) if confidence_match else "Unknown"

        return {"harmful": harmful, "confidence": confidence}

    def build(
        self,
        credentials: Dict[str, Any], # llm provider credentials
        response: str = None,
        debug_mode: bool = False,
        has_atoms: bool = False,
        has_contexts: bool = False,
        revise_atoms: bool = True,
        remove_duplicates: bool = False,
        summarize_contexts: bool = True,
        contexts_per_atom_only: bool = False,
        rel_atom_context: bool = True,
        rel_context_context: bool = False,
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

        pipeline_start_time = time.perf_counter()
        # Stage 1: decompose the response into atomic units (Atomizer)
        atomisation_start_time = time.perf_counter()
        if has_atoms == False:
            assert (
                self.response is not None
            ), f"Response cannot be None for decomposition!"

            self.atoms = build_atoms(
                response=self.response, atom_extractor=self.atom_extractor, credentials=credentials
            )
        atomisation_end_time = time.perf_counter()
        atomisation_elapsed_time = atomisation_end_time - atomisation_start_time
        print(f"Atomisation time: {atomisation_elapsed_time:.4f} seconds")
        self.logging_metadata["atomisation"] = round(atomisation_elapsed_time, 4)

        # Safety checks
        assert (
            len(self.atoms.keys()) > 0 or not has_atoms
        ), f"Atoms must be initialized if `has_atoms` is True!"

        # Stage 2: revise the atomic units to be self-contained (Reviser)
        if revise_atoms:
            atom_revision_start_time = time.perf_counter()
            print(f"[FactReasoner] Revise the atoms ...")
            atom_ids = [aid for aid in sorted(self.atoms.keys())]
            old_atoms = [self.atoms[aid].get_text() for aid in atom_ids]
            result = self.atom_reviser.run(old_atoms, self.response, credentials=credentials)
            for i, aid in enumerate(atom_ids):
                elem = result[i]
                self.atoms[aid].set_text(elem["revised_atom"])
                print(self.atoms[aid])

        self.atoms = remove_duplicated_atoms(self.atoms)
        atom_revision_end_time = time.perf_counter()
        atom_revision_elapsed_time = atom_revision_end_time - atom_revision_start_time
        print(f"Atom revision time: {atom_revision_elapsed_time:.4f} seconds")
        self.logging_metadata["atom revision"] = round(atom_revision_elapsed_time, 4)

        # Stage 3: Build contexts (ContextRtriever)
        context_retrieval_start_time = time.perf_counter()

        if not has_contexts:
            self.contexts = build_contexts(
                atoms=self.atoms, question=question, retriever=self.context_retriever
            )

        # for tracking purposes
        self.num_retrieved_contexts = len(self.contexts.keys())

        # Safety checks
        assert (
            len(self.contexts.keys()) > 0 or not has_contexts
        ), f"Contexts must be initialized if `has_contexts` is True!"

        if remove_duplicates:
            self.contexts, self.atoms = remove_duplicated_contexts(
                self.contexts, self.atoms
            )
            print(f"[FactReasoner] Found {len(self.contexts.keys())} unique contexts.")
        context_retrieval_end_time = time.perf_counter()
        context_retrieval_elapsed_time = (
            context_retrieval_end_time - context_retrieval_start_time
        )
        print(f"Context retrieval time: {context_retrieval_elapsed_time:.4f} seconds")
        self.logging_metadata["context retrieval"] = round(
            context_retrieval_elapsed_time, 4
        )

        summarize_contexts_start_time = time.perf_counter()
        # Summarize contexts given atoms
        if summarize_contexts:
            summaries_set = set()
            context_summary_key = "llm context summary"
            print(f"[FactReasoner] Summarizing the contexts ...")
            for atom_id, atom in self.atoms.items():
                if len(atom.contexts.keys()) > 0:
                    contexts_ids, contexts = zip(*atom.contexts.items())
                    results = self.context_summarizer.run(
                        [context.get_snippet_and_text() for context in contexts],
                        atom.text,
                        credentials=credentials
                    )

           
                    for context_id, result in zip(contexts_ids, results):
                        current_summary = result["summary"]
                        is_relevant = is_relevant_context(current_summary)
                        if current_summary != "" and is_relevant:
                            # Check for duplicate summaries
                            if current_summary not in summaries_set:
                                summaries_set.add(current_summary)
                                self.num_relevant_contexts += 1
                                self.contexts[context_id].set_synthetic_summary(
                                    current_summary
                                )
                                # update prior probability of context based on the confidence estimation of the summary
                                self.contexts[context_id].set_probability(
                                    result["probability"]
                                    * self.contexts[context_id].get_probability()
                                )
                            else:
                                del self.contexts[context_id]
                                del self.atoms[atom_id].contexts[context_id]
                        else:
                            self.num_irrelevant_contexts += 1
                            # we remove the context because it is not related to the atom
                            del self.contexts[context_id]
                            del self.atoms[atom_id].contexts[context_id]
            
            # summarize contexts for question

            # first get contexts for the question
            c_qs = {
                c_id: context
                for c_id, context in self.contexts.items()
                if c_id.startswith("c_q")
            }
            if len(c_qs.keys()) > 0:
                contexts_ids, contexts = zip(*c_qs.items())
                results = self.context_summarizer.run(
                    [context.get_snippet_and_text() for context in contexts], question, credentials=credentials
                )

                for context_id, result in zip(contexts_ids, results):
                    current_summary = result["summary"]
                    is_relevant = is_relevant_context(current_summary)
                    if current_summary != "" and is_relevant:
                        # Check for duplicate summaries
                        if current_summary not in summaries_set:
                            summaries_set.add(current_summary)
                            self.num_relevant_contexts += 1
                            self.contexts[context_id].set_synthetic_summary(
                                current_summary
                            )
                            self.num_relevant_contexts += 1
                            self.contexts[context_id].set_synthetic_summary(
                                result["summary"]
                            )
                            self.contexts[context_id].set_probability(
                                result["probability"]
                                * self.contexts[context_id].get_probability()
                            )
                        else:
                            del self.contexts[context_id]
                    else:
                        self.num_irrelevant_contexts += 1
                        # we remove the context because it is not related to the atom
                        del self.contexts[context_id]

        else:
            context_summary_key = "synthetic context summary"
            for context_id in self.contexts.keys():
                self.contexts[context_id].set_synthetic_summary(
                    self.contexts[context_id].get_snippet_and_text()
                )

        summarize_contexts_end_time = time.perf_counter()
        summarize_contexts_elapsed_time = (
            summarize_contexts_end_time - summarize_contexts_start_time
        )
        print(
            f"Context summarisation time: {summarize_contexts_elapsed_time:.4f} seconds"
        )
        self.logging_metadata[context_summary_key] = round(
            summarize_contexts_elapsed_time, 4
        )

        self.num_summarized_contexts = len(self.contexts.keys())


        evalutation_start_time = time.perf_counter()
        # Pre-stage 4 - Running the GG check
        # verdict = self.run_guardian_check(question)
        # self.logging_metadata["granite_guardian_check"] = verdict

        # print(f"Ran Granite Guardian model and received the responese: {verdict}")
        # if verdict["harmful"]:
        # Stage 4: Extract NLI relationships (Evaluator)

        if self.num_summarized_contexts > 0 and len(self.atoms.keys()) > 0:
            # Build the NLI relationships
            self.relations, relation_stats = build_relations(
                atoms=self.atoms,
                contexts=self.contexts,
                rel_atom_context=rel_atom_context,
                rel_context_context=rel_context_context,
                contexts_per_atom_only=contexts_per_atom_only,
                nli_extractor=self.nli_extractor,
                text_only=text_only,
                credentials=credentials
            )
        else:
            relation_stats = None
    
        self.logging_metadata["relation_stats"] = relation_stats


        # Build the fact graph and Markov network
        print(f"[FactReasoner] Building the graphical model ...")
        self._build_fact_graph()
        self._build_markov_network()

        evalutation_end_time = time.perf_counter()
        evaluation_elapsed_time = evalutation_end_time - evalutation_start_time
        print(f"Evaluation time: {evaluation_elapsed_time:.4f} seconds")
        self.logging_metadata["evaluation"] = round(evaluation_elapsed_time, 4)

        pipeline_end_time = time.perf_counter()
        pipeline_elapsed_time = pipeline_end_time - pipeline_start_time
        print(f"Pipeline time: {evaluation_elapsed_time:.4f} seconds")
        self.logging_metadata["pipeline time"] = round(pipeline_elapsed_time, 4)

        self.logging_metadata["query"] = self.query
        self.logging_metadata["response"] = self.response
        self.logging_metadata["num atoms"] = len(self.atoms.keys())
        self.logging_metadata["context retrieval settings"] = {
            "service_type": self.context_retriever.service_type,
            "top_k": self.context_retriever.top_k,
            "use_in_memory_vectorstore": self.context_retriever.use_in_memory_vectorstore,
        }
        self.logging_metadata["num retrieved contexts"] = (
            self.num_retrieved_contexts
        )
        self.logging_metadata["num summarised contexts"] = (
            self.num_summarized_contexts
        )
        self.logging_metadata["num relevant contexts"] = self.num_relevant_contexts
        self.logging_metadata["num irrelevant contexts"] = self.num_irrelevant_contexts

        print(f"[FactReasoner] Pipeline instance created.")
        # elif self.num_summarized_contexts == 0 and len(self.atoms.keys()) == 0:
        #     print(f"[FactReasoner] Could not create fact graph because no relevant contexts were retrieved and no atoms are available.")
        # elif self.num_summarized_contexts == 0:
        #     print(f"[FactReasoner] Could not create fact graph because no relevant contexts were retrieved.")
        # else:
        #     print(f"[FactReasoner] Could not create fact graph because no atoms are available.")

    def to_json(self, json_file: str = None):
        """
        Save the instance to a JSON file. Specifically, we're saving the
        atoms and retrieved contexts, for debugging purposes.
        """

        data = {}
        data["input"] = self.query
        data["output"] = self.response.strip()
        data["topic"] = self.topic
        data["logging_metadata"] = self.logging_metadata
        data["atoms"] = []
        data["contexts"] = []

        for aid, atom in self.atoms.items():
            atom_data = dict(
                id=aid, text=atom.get_text(), contexts=list(atom.get_contexts().keys())
            )
            if atom.get_label() is not None:
                atom_data["label"] = atom.get_label()
            data["atoms"].append(atom_data)

        for cid, ctxt in self.contexts.items():
            ctxt_data = dict(id=cid, title=ctxt.get_title(), text=ctxt.get_text())
            data["contexts"].append(ctxt_data)

        data["contexts"] = [context.to_json() for context in self.contexts.values()]

        if json_file:
            with open(json_file, "w") as f:
                f.write(f"{json.dumps(data)}\n")
            f.close()
            print(f"[FactReasoner] Problem instance written to: {json_file}")

        return data
    

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
            relations=self.relations,
        )

    def _build_markov_network(self):
        """
        Create the Markov Network corresponding to the FactGraph.

        Return:
            A MarkovNetwork encoding of the problem.
        """

        assert self.fact_graph is not None, f"The FactGraph must be built."

        # Create an empty Markov Network
        self.markov_network = MarkovNetwork()

        # Create the variables corresponding to the nodes in the fact graph
        print(f"[Building the Markov network...]")
        for node in self.fact_graph.get_nodes():
            x = node.id
            self.markov_network.add_node(x)
            if node.type == "context":
                prob = node.probability  # PRIOR_PROB_CONTEXT
                factor = DiscreteFactor(
                    variables=[x], cardinality=[2], values=[1.0 - prob, prob]
                )
                self.markov_network.add_factors(factor)
                print(f"Adding context variable {x} with discrete factor (prior)")
            elif node.type == "atom":
                prob = node.probability  # PRIOR_PROB_ATOM
                factor = DiscreteFactor(
                    variables=[x], cardinality=[2], values=[1.0 - prob, prob]
                )
                self.markov_network.add_factors(factor)
                print(f"Adding atom variable {x} with discrete factor (prior)")
            else:
                raise ValueError(f"Unknown node type: {node.type}")

        # Create the factors corresponding to the edges in the fact graph
        for edge in self.fact_graph.get_edges():
            x, y = edge.source, edge.target
            self.markov_network.add_edge(x, y)
            if edge.type == "entailment":  # add factor X -> Y
                prob = edge.probability
                if self.use_priors:
                    if edge.link == "context_atom":
                        values = [
                            1.0 - PRIOR_PROB_ATOM,
                            PRIOR_PROB_ATOM,
                            1.0 - prob,
                            prob,
                        ]
                    elif edge.link == "context_context":
                        values = [
                            1.0 - PRIOR_PROB_CONTEXT,
                            PRIOR_PROB_CONTEXT,
                            1.0 - prob,
                            prob,
                        ]
                    elif edge.link == "atom_atom":
                        values = [
                            1.0 - PRIOR_PROB_ATOM,
                            PRIOR_PROB_ATOM,
                            1.0 - prob,
                            prob,
                        ]
                    else:
                        raise ValueError(f"Unknown link type: {edge.link}")
                else:
                    values = [prob, prob, 1.0 - prob, prob]

                # Create the factor
                factor = DiscreteFactor(
                    variables=[x, y],
                    cardinality=[2, 2],
                    values=values,  # [prob, prob, 1.0 - prob, prob]
                )
                self.markov_network.add_factors(factor)
                print(f"Adding edge {x} - {y} with discrete factor (entailment)")
            elif edge.type == "contradiction":  # add factor X -> !Y
                prob = edge.probability
                if self.use_priors:
                    if edge.link == "context_atom":
                        values = [
                            1.0 - PRIOR_PROB_ATOM,
                            PRIOR_PROB_ATOM,
                            prob,
                            1.0 - prob,
                        ]
                    elif edge.link == "context_context":
                        values = [
                            1.0 - PRIOR_PROB_CONTEXT,
                            PRIOR_PROB_CONTEXT,
                            prob,
                            1.0 - prob,
                        ]
                    elif edge.link == "atom_atom":
                        values = [
                            1.0 - PRIOR_PROB_ATOM,
                            PRIOR_PROB_ATOM,
                            prob,
                            1.0 - prob,
                        ]
                    else:
                        raise ValueError(f"Unknown link type: {edge.link}")
                else:
                    values = [prob, prob, prob, 1.0 - prob]

                factor = DiscreteFactor(
                    variables=[x, y],
                    cardinality=[2, 2],
                    values=values,  # [prob, prob, prob, 1.0 - prob]
                )
                self.markov_network.add_factors(factor)
                print(f"Adding edge {x} - {y} with discrete factor (contradiction)")
            elif edge.type == "equivalence":
                prob = edge.probability
                factor = DiscreteFactor(
                    variables=[x, y],
                    cardinality=[2, 2],
                    values=[prob, 1.0 - prob, 1.0 - prob, prob],
                )
                self.markov_network.add_factors(factor)
                print(f"Adding edge {x} - {y} with discrete factor (equivalence)")

        # Output the content of the network
        print("[Markov network created.]")
        print(self.markov_network)

        if self.debug_mode:
            print("[Markov network content...]")
            for f in self.markov_network.get_factors():
                print(f)

        # writer = UAIWriter(model)
        # writer.write_uai("/home/radu/git/fm-factual/examples/markov_network.uai")

    def run_merlin(self):
        """
        Run inference with merlin (executable)
        """

        # Prepare the query variables (i.e., atoms)
        query_variables = [var for var in sorted(self.atoms.keys())]

        # Dump the markov network to a temporary file
        net_id = str(uuid.uuid1())
        input_filename = f"/tmp/markov_network_{net_id}.uai"
        writer = UAIWriter(self.markov_network)
        writer.write_uai(input_filename)

        # Get the variable name to index mapping {0: ('a0', '2'), 1: ('a1', '2')}
        vars_mapping = {}
        variables = sorted(writer.domain.items(), key=lambda x: (x[1], x[0]))
        for i, var in enumerate(variables):
            vars_mapping[i] = var[0]

        # Run merlin as a subprocess and collect the results
        exefile = self.merlin_path
        output_format = "json"
        output_file = f"/tmp/output_{net_id}"
        algorithm = "wmb"
        task = "MAR"

        args = [
            exefile,
            "--input-file",
            input_filename,
            "--task",
            task,
            "--ibound",
            "6",
            "--algorithm",
            algorithm,
            "--output-format",
            output_format,
            "--output-file",
            output_file,
        ]

        proc = subprocess.run(args)

        print(f"[Merlin] return code: {proc.returncode}")
        output_filename = f"{output_file}.{task}.{output_format}"
        with open(output_filename) as f:
            results = json.load(f)

        marginals = []
        all_marginals = []
        for marginal in results["marginals"]:
            var_index = marginal["variable"]
            var_name = vars_mapping[var_index]
            all_marginals.append(
                dict(variable=var_name, probabilities=marginal["probabilities"])
            )
            if var_name in query_variables:
                probs = marginal["probabilities"]
                marginals.append({"variable": var_name, "probabilities": probs})

        # Cleanup -- delete input_filename and output_filename
        if os.path.exists(input_filename):
            os.remove(input_filename)
        if os.path.exists(output_filename):
            os.remove(output_filename)

        print(f"All Marginals:\n{all_marginals}")
        return marginals

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

        # assert len(self.atoms) > 0
        # assert len(self.contexts) > 0
        # assert len(self.relations) > 0
        assert self.fact_graph is not None
        assert self.markov_network is not None

        marginals = self.run_merlin()

        # Prepare the results
        num_true_atoms = 0
        num_uniform_atoms = 0
        avg_prob = 0.0
        avg_logprob = 0.0
        entropy = 0.0
        norm_entropy = 0.0
        avg_norm_entropy = 0.0
        labels = {}
        probabilities = {}
        fscore_per_atom = []
        for marginal in marginals:
            var = marginal["variable"]
            probs = marginal["probabilities"]

            print(f"[{var}]: Probability for {var}=0 is: {probs[0]}")
            print(f"[{var}]: Probability for {var}=1 is: {probs[1]}")

            # Check if atom is true or not
            probabilities[var] = probs[1]  # probability of true
            if probs[1] > probs[0]:
                num_true_atoms += 1
                labels[var] = "S"
            else:
                labels[var] = "NS"

            fscore_per_atom.append({var: {"score": probs[1], "support": labels[var]}})
            probval = probs[1]
            if probval < 1e-6:
                probval = 1e-6
            elif probval >= 1.0:
                probval = 0.999999
            elif probval == 0.5:
                num_uniform_atoms += 1
            avg_logprob += math.log(probval)
            avg_prob += probval
            entropy += -probval * math.log(probval)
            norm_entropy += -(
                probval * math.log(probval) + (1.0 - probval) * math.log(1.0 - probval)
            ) / math.log(2.0)

            # probval = probs[1] if probs[1] > 0.0 else 1e-6
            # if probval == 0.5:
            #     num_uniform_atoms += 1
            # avg_logprob += math.log(probval)
            # entropy += -probval * math.log10(probval)

        # For now, return a dict with the posterior marginals of the atoms
        # avg_logprob /= len(self.atoms.keys())
        # avg_entropy = entropy / len(self.atoms.keys())
        # fscore = num_true_atoms / len(self.atoms.keys())
        avg_logprob /= len(self.atoms)
        avg_prob /= len(self.atoms)
        avg_entropy = entropy / len(self.atoms)
        avg_norm_entropy = norm_entropy / len(self.atoms)
        fscore = num_true_atoms / len(self.atoms)

        results = {}
        results["factuality_score_per_atom"] = fscore_per_atom
        results["factuality_score"] = fscore
        results["num_atoms"] = len(self.atoms)
        results["num_contexts"] = len(self.contexts)
        results["num_true_atoms"] = num_true_atoms
        results["num_false_atoms"] = len(self.atoms) - num_true_atoms
        results["num_uniform_atoms"] = num_uniform_atoms
        results["entropy"] = entropy
        results["norm_entropy"] = norm_entropy
        results["avg_entropy"] = avg_entropy
        results["avg_norm_entropy"] = avg_norm_entropy
        results["avg_prob"] = avg_prob
        results["avg_logprob"] = avg_logprob  # math.exp(avg_logprob)
        results["avg_explogprob"] = math.exp(avg_logprob)
        results["labels"] = labels

        # Print the predicted labels
        str_predictions = ""
        for aid in sorted(labels.keys()):
            str_predictions += f" {aid}: {labels[aid]}"
        print(f"[FactReasoner] Predictions: {str_predictions}")

        # Check for ground truth annotations
        if self.labels_human is not None:
            true_atoms = 0
            false_atoms = 0
            avg_brier = 0.0
            num_true_positive = 0
            num_true_negative = 0
            num_false_positive = 0
            num_false_negative = 0
            for aid, l in self.labels_human.items():
                if l == "S":
                    avg_brier += (probabilities[aid] - 1.0) * (probabilities[aid] - 1.0)
                    true_atoms += 1
                    if labels[aid] == "S":
                        num_true_positive += 1
                    else:
                        num_false_negative += 1
                else:
                    avg_brier += (probabilities[aid] - 0.0) * (probabilities[aid] - 0.0)
                    false_atoms += 1
                    if labels[aid] == "NS":
                        num_true_negative += 1
                    else:
                        num_false_positive += 1
            fscore_gold = true_atoms / len(self.labels_human.keys())
            avg_brier /= len(self.atoms)
            str_references = ""
            for aid in sorted(self.labels_human.keys()):
                str_references += f" {aid}: {self.labels_human[aid]}"
            print(f"[FactReasoner] Gold labels: {str_references}")
            print(
                f"[FactReasoner] Gold fscore: {fscore_gold} ({true_atoms}/{len(self.labels_human.keys())})"
            )
            results["gold_factuality_score"] = fscore_gold
            results["gold_true_atoms"] = true_atoms
            results["true_positive"] = num_true_positive
            results["true_negative"] = num_true_negative
            results["false_positive"] = num_false_positive
            results["false_negative"] = num_false_negative
            results["predictions"] = str_predictions
            results["references"] = str_references
            results["avg_brier"] = avg_brier

        # if self.topic is not None and len(self.topic) > 0:
        #     results["topic"] = self.topic
        results["input"] = self.query
        results["marginals"] = marginals

        return results, marginals


def test():
    model = "llama-3.3-70b-instruct"
    cache_dir = "my_database.db"

    context_retriever = ContextRetriever(
        service_type="google", top_k=5, cache_dir=cache_dir
    )
    context_summarizer = ContextSummarizer(model=model, prompt_version="v1")
    query_builder = QueryBuilder(model=model, prompt_version="v1")
    atom_extractor = AtomExtractor(model)
    atom_reviser = AtomReviser(model)
    # nli_extractor = NLIExtractor(model, prompt_version="v1")
    nli_extractor = NLIExtractorOld(model, prompt_version="v2")

    merlin_path = "/Users/tigran/Code/merlin2/merlin/build/merlin"

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
    json_file = "/Users/tigran/Code/fm-factual/examples/test.json"
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
        text_only=False,
    )

    results, marginals = pipeline.score()
    print(f"[FactReasoner] Marginals: {marginals}")
    print(f"[FactReasoner] Results: {results}")
    print(f"Done.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to the input dataset (jsonl).",
    )

    parser.add_argument(
        "--output_dir", type=str, default=None, help="Path to the output directory."
    )

    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Path to the cache directory."
    )

    parser.add_argument(
        "--dataset_name", type=str, default=None, help="Name of the dataset."
    )

    parser.add_argument(
        "--service_type",
        type=str,
        default="google",
        help="Service type (langchain, chromadb, google).",
    )

    parser.add_argument(
        "--model", type=str, default=None, help="Name of the RITS model used internally"
    )

    parser.add_argument(
        "--version", type=int, default=1, help="FactReasoner version: 1, 2 or 3"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Top k results retrieved as contexts per atom.",
    )

    parser.add_argument(
        "--use_priors",
        default=False,
        action="store_true",
        help="Use the atom and context priors in the factor definition.",
    )

    parser.add_argument(
        "--use_query_builder",
        default=False,
        action="store_true",
        help="Use the QueryBuilder to generate queries for Google search.",
    )

    parser.add_argument(
        "--text_only",
        default=False,
        action="store_true",
        help="Contexts are considered text only.",
    )

    parser.add_argument(
        "--nli_prompt_version",
        type=str,
        default="v1",
        help="NLI prompt version: v1 (original) or v2 (more recent - some reasoning)",
    )

    parser.add_argument(
        "--atomizer_prompt_version",
        type=str,
        default="v2",
        help="Atomizer prompt version: v1 (original) or v2 (newer)",
    )

    parser.add_argument(
        "--reviser_prompt_version",
        type=str,
        default="v1",
        help="Reviser prompt version: v1 (newer) or v2 (original)",
    )

    parser.add_argument(
        "--test", default=False, action="store_true", help="Debugging mode."
    )

    parser.add_argument(
        "--bert_nli",
        default=False,
        action="store_true",
        help="A BERT model (roberta) is used for NLI extraction.",
    )

    parser.add_argument(
        "--merlin_path",
        type=str,
        default="/home/radu/git/fm-factual/lib/merlin",
        help="Path to the probabilistic inference merlin.",
    )

    args = parser.parse_args()

    if args.test:
        test()
        sys.exit(0)

    # FactReasoner versions:
    if (
        args.version == 1
    ):  # 1 - context-atom relationships only, allow duplicated contexts
        rel_context_context = False
        remove_duplicates = False
        contexts_per_atom_only = True
        option = "1"
    elif (
        args.version == 2
    ):  # 2 - context-atom relationships only, no duplicated contexts
        rel_context_context = False
        remove_duplicates = True
        contexts_per_atom_only = False
        option = "2"
    elif (
        args.version == 3
    ):  # 3 - context-atom and context-context relationships, no duplicated contexts
        rel_context_context = True
        remove_duplicates = True
        contexts_per_atom_only = False
        option = "3"
    elif (
        args.version == 4
    ):  # 4 - context-atom relationships only, duplicated contexts are allowed
        rel_context_context = True
        remove_duplicates = True
        contexts_per_atom_only = False
        option = "4"
    else:
        raise ValueError(f"Unknown FactReasoner version: {args.version}")

    # Get the NLI prompt version
    nli_prompt_version = args.nli_prompt_version

    # Create context retriever
    context_retriever = ContextRetriever(
        service_type=args.service_type, top_k=args.top_k, cache_dir=args.cache_dir
    )

    # Create the atom extractor
    atom_extractor = AtomExtractor(
        model=args.model, prompt_version=args.atomizer_prompt_version
    )

    # Create the atom reviser
    atom_reviser = AtomReviser(
        model=args.model, prompt_version=args.reviser_prompt_version
    )

    # Create the NLI extractor
    if not args.bert_nli:
        nli_extractor = NLIExtractor(
            model=args.model, prompt_version=nli_prompt_version
        )
        nli_model_name = args.model
    else:  # BERT based NLI extraction
        nli_extractor = NLIExtractor(model="roberta", is_bert=True)
        nli_model_name = "roberta"

    # Create the query builder
    if args.use_query_builder:
        query_builder = QueryBuilder(model=args.model)
    else:
        query_builder = None

    print(f"[FactReasoner] Processing input dataset: {args.input_file}")
    filename = args.input_file  # a jsonl file

    with open(filename) as f:
        lines = f.read().splitlines()
    df_inter = pd.DataFrame(lines)
    df_inter.columns = ["json_element"]
    df_inter["json_element"].apply(json.loads)
    df = pd.json_normalize(df_inter["json_element"].apply(json.loads))
    dataset = df.to_dict("records")

    print(f"[FactReasoner] Loading data from: {filename}")
    print(f"[FactReasoner] Found {len(dataset)} elements")

    # Check if previous results exist. If yes, load them and skip over them
    # when processing the input dataset.
    filename = "eval_results_factreasoner{}_{}_{}_{}.jsonl".format(
        option, args.service_type, args.dataset_name, nli_model_name
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

    # Loop over the data points in the dataset
    for input_data in dataset:
        # Check if current data has been processed already
        processed = False
        for eval_data in evaluation_data:
            if eval_data["input"] == input_data["input"]:
                processed = True
                break
        if processed:
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
            use_priors=args.use_priors,
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
            text_only=args.text_only,
        )

        results, marginals = pipeline.score()
        results["model_name"] = args.model
        evaluation_data.append(results)
        print(f"[FactReasoner] Marginals: {marginals}")
        print(f"[FactReasoner] Results: {results}")

        # Save results to a file
        filename = "eval_results_factreasoner{}_{}_{}_{}.jsonl".format(
            option, args.service_type, args.dataset_name, nli_model_name
        )
        output_filename = os.path.join(args.output_dir, filename)
        print(f"[FactReasoner] Writing results to: {output_filename}")
        with open(output_filename, "w") as f:
            for res in evaluation_data:
                f.write(f"{json.dumps(res)}\n")

    print("Done.")
