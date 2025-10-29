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

# FactReasoner pipeline

import argparse
import json
import logging
import math
import os
import subprocess
import uuid

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.global_vars import logger
from pgmpy.models import MarkovNetwork
from pgmpy.readwrite import UAIWriter

# Local imports
from src.fact_reasoner.atom_extractor import AtomExtractor
from src.fact_reasoner.atom_reviser import AtomReviser
from src.fact_reasoner.context_retriever import ContextRetriever
from src.fact_reasoner.context_summarizer import ContextSummarizer
from src.fact_reasoner.fact_graph import FactGraph
from src.fact_reasoner.fact_utils import (
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
from src.fact_reasoner.nli_extractor import NLIExtractor
from src.fact_reasoner.query_builder import QueryBuilder

# Set logging levels 
# pgmpy set the root logger to INFO -- changed it to WARNING
os.environ["LITELLM_LOG"] = 'ERROR'
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("pgmpy").setLevel(logging.WARNING)


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
                The input LLM generated long-form response.
            debug_mode: bool
                Boolean flag indicating debugging mode (default False).
            has_atoms: bool
                Flag indicating if the atoms were previously initialized.
            has_contexts: bool
                Flag indicating if the contexts were previously initialized.
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

        if not self.query:
            self.query = question

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
            self._build_markov_network()

            print(f"[FactReasoner] Pipeline instance created.")
        elif self.num_summarized_contexts == 0 and len(self.atoms.keys()) == 0:
            print(f"[FactReasoner] Could not create fact graph because no relevant contexts were retrieved and no atoms are available.")
        elif self.num_summarized_contexts == 0:
            print(f"[FactReasoner] Could not create fact graph because no relevant contexts were retrieved.")
        else:
            print(f"[FactReasoner] Could not create fact graph because no atoms are available.")


    def pipeline_to_json(self, json_file_path: str = None):
        """
        Save the pipeline instance to a JSON file.
        """

        data = {}
        data["input"] = self.query
        data["output"] = self.response.strip()
        data["topic"] = self.topic
        data["atoms"] = []
        data["contexts"] = []

        for aid, atom in self.atoms.items():
            atom_data = dict(
                id=aid, text=atom.get_text(), contexts=list(atom.get_contexts().keys())
            )
            if atom.get_label() is not None:
                atom_data["label"] = atom.get_label()
            data["atoms"].append(atom_data)

        data["contexts"] = [context.context_to_json() for context in self.contexts.values()]

        if json_file_path:
            with open(json_file_path, "w") as f:
                f.write(f"{json.dumps(data)}\n")
            f.close()
            print(f"[FactReasoner] Pipeline instance written to: {json_file_path}")

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
            relations=self.relations
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
                prob = node.probability  #PRIOR_PROB_CONTEXT
                factor = DiscreteFactor(
                    variables=[x],
                    cardinality=[2],
                    values=[1.0 - prob, prob]
                )
                self.markov_network.add_factors(factor)
                print(f"Adding context variable {x} with discrete factor (prior)")
            elif node.type == "atom":
                prob = node.probability  # PRIOR_PROB_ATOM
                factor = DiscreteFactor(
                    variables=[x],
                    cardinality=[2],
                    values=[1.0 - prob, prob]
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
                        values = [1.0 - PRIOR_PROB_ATOM, PRIOR_PROB_ATOM, 1.0 - prob, prob]
                    elif edge.link == "context_context":
                        values = [1.0 - PRIOR_PROB_CONTEXT, PRIOR_PROB_CONTEXT, 1.0 - prob, prob]
                    elif edge.link == "atom_atom":
                        values = [1.0 - PRIOR_PROB_ATOM, PRIOR_PROB_ATOM, 1.0 - prob, prob]
                    else:
                        raise ValueError(f"Unknown link type: {edge.link}")
                else:
                    values = [prob, prob, 1.0 - prob, prob]

                # Create the factor    
                factor = DiscreteFactor(
                    variables=[x, y],
                    cardinality=[2, 2],
                    values=values  #[prob, prob, 1.0 - prob, prob]
                )
                self.markov_network.add_factors(factor)
                print(f"Adding edge {x} - {y} with discrete factor (entailment)")
            elif edge.type == "contradiction":  # add factor X -> !Y
                prob = edge.probability
                if self.use_priors:
                    if edge.link == "context_atom":
                        values = [1.0 - PRIOR_PROB_ATOM, PRIOR_PROB_ATOM, prob, 1.0 - prob]
                    elif edge.link == "context_context":
                        values = [1.0 - PRIOR_PROB_CONTEXT, PRIOR_PROB_CONTEXT, prob, 1.0 - prob]
                    elif edge.link == "atom_atom":
                        values = [1.0 - PRIOR_PROB_ATOM, PRIOR_PROB_ATOM, prob, 1.0 - prob]
                    else:
                        raise ValueError(f"Unknown link type: {edge.link}")
                else:
                    values = [prob, prob, prob, 1.0 - prob]

                factor = DiscreteFactor(
                    variables=[x, y],
                    cardinality=[2, 2],
                    values=values  #[prob, prob, prob, 1.0 - prob]
                )
                self.markov_network.add_factors(factor)
                print(f"Adding edge {x} - {y} with discrete factor (contradiction)")
            elif edge.type == "equivalence":
                prob = edge.probability
                factor = DiscreteFactor(
                    variables=[x, y],
                    cardinality=[2, 2],
                    values=[prob, 1.0 - prob, 1.0 - prob, prob]
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
            "--input-file", input_filename,
            "--task", task,
            "--ibound", "6",
            "--algorithm", algorithm,
            "--output-format", output_format,
            "--output-file", output_file
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
            all_marginals.append(dict(variable=var_name, probabilities=marginal["probabilities"]))
            if var_name in query_variables:
                probs = marginal["probabilities"]
                marginals.append({"variable": var_name, "probabilities": probs})

        # Cleanup -- delete input_filename and output_filename
        if os.path.exists(input_filename): os.remove(input_filename)
        if os.path.exists(output_filename): os.remove(output_filename)

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
            probabilities[var] = probs[1] # probability of true
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
            norm_entropy += -(probval * math.log(probval) + (1.0 - probval) * math.log(1.0 - probval)) / math.log(2.0)

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
        results["avg_logprob"] = avg_logprob # math.exp(avg_logprob)
        results["avg_explogprob"] = math.exp(avg_logprob)

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
            print(f"[FactReasoner] Gold fscore: {fscore_gold} ({true_atoms}/{len(self.labels_human.keys())})")
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

    parser.add_argument(
        '--merlin_path',
        type=str,
        default=None,
        required=True,
        help="Path to the probabilistic inference engine merlin."
    )

    # Parse CLI arguments
    args = parser.parse_args()

    # Define the model and backend
    model_id = "llama-3.3-70b-instruct"
    cache_dir = None # "my_database.db"
    backend = "rits"

    # Create the pipeline modules
    query_builder = QueryBuilder(model_id=model_id, prompt_version="v1", backend=backend)
    context_retriever = ContextRetriever(service_type="google", top_k=5, cache_dir=cache_dir)
    context_summarizer = ContextSummarizer(model_id=model_id, prompt_version="v1", backend=backend)
    atom_extractor = AtomExtractor(model_id=model_id, backend=backend)
    atom_reviser = AtomReviser(model_id=model_id, backend=backend)
    nli_extractor = NLIExtractor(model_id=model_id, prompt_version="v1", backend=backend)

    # Path to merlin (probabilistic inference engine)
    assert args.merlin_path is not None, f"Path to merlin cannot be None. Aborting."
    merlin_path = args.merlin_path

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
    assert args.input_file is not None, f"Path to the input file cannot be None. Aborting."
    json_file = args.input_file
    with open(json_file, "r") as f:
        data = json.load(f)

    # Instantiate the pipeline from a data file
    pipeline.from_dict_with_contexts(data)

    # Build the FactReasoner pipeline (FR2 version)
    pipeline.build(
        has_atoms=True,
        has_contexts=True,
        revise_atoms=False,
        remove_duplicates=True,
        contexts_per_atom_only=False,
        rel_atom_context=True, 
        rel_context_context=False,
        text_only=False
    )

    # Compute the marginals
    results, marginals = pipeline.score()
    print(f"[FactReasoner] Marginals: {marginals}")
    print(f"[FactReasoner] Results: {results}")
    print(f"Done.")

