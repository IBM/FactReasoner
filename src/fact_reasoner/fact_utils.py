# coding=utf-8
# Copyright 2023-present the International Business Machines.g
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

# Atom and Context classes
from collections import defaultdict
import json
from typing import Callable, Tuple, Union, List, TypedDict, cast
from operator import itemgetter
import os
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.models import MarkovNetwork
from pgmpy.readwrite import UAIWriter
from itertools import combinations
from litellm.types.utils import Choices, ChatCompletionTokenLogprob, ModelResponse, TopLogprob
import math
from nltk.tokenize import sent_tokenize
import nltk
import regex
from scipy.special import logsumexp
import subprocess
import traceback
import uuid

# Local
from src.fact_reasoner.atom_extractor import AtomExtractor
from src.fact_reasoner.json_nli_extractor import JsonNliExtractor
from src.fact_reasoner.context_retriever import ContextRetriever
from src.fact_reasoner.fact_components import (
    PRIOR_PROB_ATOM,
    PRIOR_PROB_CONTEXT,
    Atom,
    Context,
    Relation,
)
from src.fact_reasoner.fact_graph import FactGraph
from src.fact_reasoner.nli_extractor import NLIExtractor, NLIExtractorOld
from src.fact_reasoner.utils import punctuation_only_inside_quotes

import logging
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


def predict_nli_relationships(
        object_pairs: List[Tuple[Union[Atom, Context], Union[Atom, Context]]],
        nli_extractor: NLIExtractor,
        links_type: str = "context_atom",
        text_only: bool = True,
    ) -> list[Relation]:
    """
    Predict the NLI relationship between two objects using an model based NLI extractor.

    Args:
        object_pairs: List
            A list of object pairs e.g., (atom, context) or (context, context)
        nli_extractor: NLIExtractor
            The model based NLI extractor
        top_k_per_atom: int
            The top k relationships considered for each atom.
        links_type: str
            The type of links represented by the object pairs (context_atom, context_context).
    """

    assert (nli_extractor is not None), "NLI extractor cannot be None."
    assert isinstance(nli_extractor, NLIExtractor), "NLI extractor must be NLIExtractor."

    premises = [pair[0] if isinstance(pair[0], str) else pair[0].get_synthetic_summary(text_only) for pair in object_pairs]
    hypotheses = [pair[1] if isinstance(pair[1], str) else pair[1].get_synthetic_summary(text_only) for pair in object_pairs]

    # premises = [pair[0] if isinstance(pair[0],str) else pair[0].get_text(text_only) for pair in object_pairs]
    # hypotheses = [pair[1] if isinstance(pair[1],str) else pair[1].get_text(text_only) for pair in object_pairs]

    results = nli_extractor.runall(premises, hypotheses)

    # print(f"Found: {len(results)} relationships")
    # print(results)

    relations = []
    for ii, result in enumerate(results):
        label = result["label"]
        probability = result["probability"]
        link_type = links_type if links_type is not None else "unknown"
        rel = Relation(
            source=object_pairs[ii][0],
            target=object_pairs[ii][1],
            type=label,
            probability=probability,
            link=link_type
        )
        relations.append(rel)
   
    return relations

def get_nli_relations_prompting(
        atom_context_pairs: List[Tuple[Union[Atom, Context], Union[Atom, Context]]],
        nli_scorer = None,
        top_k_per_atom = None,
        links_type: str = "context_atom",
        text_only: bool = True
    ) -> list[Relation]:
        
    assert (nli_scorer is not None), "NLI extractor cannot be None."
    assert isinstance(nli_scorer, NLIExtractorOld), "NLI extractor must be NLIExtractorOld."

    premises = [pair[0] if isinstance(pair[0],str) else pair[0].get_text(text_only) for pair in atom_context_pairs]
    hypotheses = [pair[1] if isinstance(pair[1],str) else pair[1].get_text(text_only) for pair in atom_context_pairs]

    results_labels = nli_scorer.score(
        premises, 
        hypotheses, 
    )

    # print(f"Found: {len(results_labels)} relationships")
    # print(results_labels)

    relations = []
    if top_k_per_atom is None:
        for ii, (label, score) in enumerate(results_labels):
            #if label == 'neutral':continue
            link = links_type if links_type is not None else "unknown"
            relations.append(Relation(
                        source=atom_context_pairs[ii][0],
                        target=atom_context_pairs[ii][1],
                        type=label,
                        probability=score,
                        link=link
                    )
                )
    else: # TODO: I'm not sure it works correctly (debug)
        candidates_per_atom = [[]]
        for ii, (label, score) in enumerate(results_labels):
            atom = atom_context_pairs[ii][0]
            if ii==0:
                previous_atom=atom
            else:
                if atom!=previous_atom:
                    candidates_per_atom.append([])

            #if label == 'neutral':continue
            link = links_type if links_type is not None else "unknown"
            rel = Relation(
                    source=atom_context_pairs[ii][0],
                    target=atom_context_pairs[ii][1],
                    type=label,
                    probability=score,
                    link=link                   
                )
            candidates_per_atom[-1].append((rel.get_probability(), rel))

        for candidates in candidates_per_atom:
            k = min(top_k_per_atom, len(candidates))
            candidates = sorted(candidates, key=itemgetter(0), reverse=True)
            for i in range(k):
                rel = candidates[i][1]
                relations.append(rel)

    return relations


def build_atoms(response: str, atom_extractor: AtomExtractor) -> dict:
    """
    Decompose the given response into atomic units (i.e., atoms).

    Args:
        response: str
            The string representing the LLM response.
        atom_extractor: AtomExtractor 
            The model based atom extractor.
    Returns:
        A dict containing the atoms of the response.
    """

    assert (response is not None and len(response) > 0), \
        f"Make sure that the response is not empty."

    print(f"[Building atoms ...]")
    result = atom_extractor.run(response)
    candidates = [
        Atom(
            id="a" + str(i),
            text=elem["atom"]
        ) for i, elem in enumerate(result["all_facts"])
    ]

    atoms = {}
    for atom in candidates:
        print(atom)
        atoms[atom.id] = atom

    print(f"[Atoms built: {len(atoms)}]")

    return atoms


def build_contexts(
        atoms: dict = {},
        question: str = None,
        retriever: ContextRetriever = None,
):
    """
    Retrieve the relevant contexts for the input atoms.

    Args:
        atoms: dict
            A dict containing the atoms in the response.
        retriever: ContextRetriever 
            The context retriever (chromadb, langchain, google).
    """

    assert (len(atoms) > 0), \
        "Please ensure a non-empty list of atoms."
    assert (retriever is not None), \
        "Please ensure an existing context retriever instance."

    # Building the contexts
    print(f"[Building contexts...]")
    contexts = {}

    for aid, atom in atoms.items():
        retrieved_contexts = retriever.query(
            text=atom.text,
        )
        
        if len(retrieved_contexts) > 0:
            contexts_per_atom = [
                Context(
                    id="c_" + aid + "_" + str(j),
                    atom=atom,
                    text=context["text"],
                    title=context["title"],
                    link=context["link"],
                    snippet=context["snippet"]
                    # An empty summary means that the context is not relevant, therefore we do not add it to the list of contexts for the pipeline
                ) for j, context in enumerate(retrieved_contexts) 
            ]

            for ctxt in contexts_per_atom:
                contexts[ctxt.id] = ctxt
            atoms[aid].add_contexts(contexts_per_atom)

    # we retrieve the contexts for the question
    retrieved_contexts = retriever.query(
        text=question,
    )
    
    if len(retrieved_contexts) > 0:
        contexts_per_atom = [
            Context(
                id="c_q_" + str(j),
                atom=None,
                text=context["text"],
                title=context["title"],
                link=context["link"],
                snippet=context["snippet"]
                # An empty summary means that the context is not relevant, therefore we do not add it to the list of contexts for the pipeline
            ) for j, context in enumerate(retrieved_contexts) 
        ]

        for ctxt in contexts_per_atom:
            contexts[ctxt.id] = ctxt
    
    print(f"[Contexts built: {len(contexts)}]")
    return contexts

def remove_duplicated_atoms(atoms: dict) -> dict:
    """
    Remove the duplicated atoms.
    """
    duplicates = {}
    filtered_atoms = {}
    for aid, atom in atoms.items():
        text = atom.get_text(text_only=False)
        if text not in duplicates:
            duplicates[text] = aid
            filtered_atoms[aid] = atom
    
    return filtered_atoms


def remove_duplicated_contexts(contexts: dict, atoms: dict) -> dict:
    """
    Remove the duplicated contexts.
    """
    duplicates = {}
    filtered_contexts = {}
    for cid, context in contexts.items():
        text = context.get_text(text_only=False)
        if text not in duplicates:
            duplicates[text] = cid
            filtered_contexts[cid] = context
        elif context.atom and context.atom.id in atoms:
            del atoms[context.atom.id].contexts[cid]
    
    return filtered_contexts, atoms


def is_relevant_context(context: str) -> dict:
    """
    Check if context is relevant.
    """

    keywords = [
        "not provide information about the atom",
        "not provide any information about the atom",
        "not provide specific information about the atom",
        "not contain information about the atom",
        "not provide any information related to the atom",
        "not provide specific information related to the atom",
        "not provide information related to the atom",
        
        "not contain information about the atom",   
        "not contain any information about the atom",
        "not contain specific information about the atom",
        "not provide information on the atom",
        "not provide any information on the atom",
        "not provide specific information on the atom",
        "insufficient to make a conclusion about the atom",
        "not provide enough information to make a conclusion about the atom",
        "not contain enough information to make a conclusion about the atom",
        "not provide any relevant information about the atom",
        "information about the atom cannot be found",
        "information is not about the atom",
        "information is not related to the atom",
        "is known that",
        "is generally known that",
        "is believed that",
        "don't have permission to view this page",
        "due to a 403 forbidden error",
        "shows a 403 forbidden error",
        "is a 403 forbidden error",
        "not have permission to view",
        "not have permission to access",
        "access to the page is forbidden",
        "context is not available",
        "context is not accessible",
        "not possible to summarize the context",
        "verify the given atom",
        "atom statement",
        "atom states",
    ]

    context_lower = context.lower()
    if not all(keyword.lower() not in context_lower for keyword in keywords):
        return False
            
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("'punkt' not found. Downloading...")
        nltk.download('punkt')

    sentences = sent_tokenize(context)
    num_sentences = len(sentences)
    # we filter out summaries of only one sentence of the form: "the context does not..."
    if (num_sentences == 1 
        and punctuation_only_inside_quotes(sentences[0]) 
        and ("the context does not" in sentences[0].lower())
       ):
            return False
    
    return True


def batch_build_relations(
    atoms: dict[str, Atom],
    contexts: dict[str, Context],
    nli_extractor: JsonNliExtractor,
    contexts_per_atom_only: bool = False,
    rel_atom_context: bool = True,
    rel_context_context: bool = True,
    text_only: bool = True,
    max_batch_relations: int = 64,
    max_batched_completions: int = 16,
) -> List[Relation]:
    """
    Create the NLI relations between atoms and contexts. The following
    pairwise relations are considered: atom-context and context-context.
    Optionally, atom-atom relations can also be considered.

    Args:
        atoms: dict
            A dict containing the atoms in the response.
        contexts: dict
            A dict containing the contexts retrived from the vector store.
        nli_extractor: JsonNliExtractor
            The JsonNliExtractor instance to use for extracting the relations.
        contexts_per_atom_only: bool
            Flag indicating that for each atom only its corresponding contexts are considered. 
        rel_atom_context: bool (default is True)
            Flag indicating the presence of atom-to-context relationships.
        rel_context_context: bool (default is True)
            Flag indicating the presence of context-to-context relationships.
        text_only: bool
            Flag indicating that contexts are text only. If False, then the
            contexts include (Title, Snippet, Link, Text).
        max_batch_relations: int
            Maximum number of NLI relations to predict in each LLM request.
            Hypothesis-premise pairs will be split into requsts according
            to this number.
        max_batched_completions: int
            The maximum number of parallel model calls.
    Returns:
        A list of Relations.  
    """
    relations: list[Relation] = []
    text_to_atom: dict[str, Atom] = {
        a.get_synthetic_summary(text_only): a for a in atoms.values()
    }
    text_to_context: dict[str, Context] = {
        c.get_synthetic_summary(text_only): c for c in contexts.values()
    }

    if rel_atom_context:
        print("[Building atom-context relations...]")
        if not contexts_per_atom_only:  # use all contexts for each atom
            # Create the (context, atom) pairs
            print("Using all contexts retrieved per atom.")
            nli_inputs: dict[str, list[str]] = {
                c.get_synthetic_summary(text_only): [
                    a.get_synthetic_summary(text_only) for a in atoms.values()
                ]
                for c in contexts.values()
            }
        else:
            print("Using only the contexts retrieved per atom.")
            nli_inputs: dict[str, list[str]] = {
                c.get_synthetic_summary(text_only): [
                    c.atom.get_synthetic_summary(text_only)
                ]
                for c in contexts.values()
                if c.atom is not None
            }
        nli_results = nli_extractor.run(
            nli_inputs,
            max_batch_relations=max_batch_relations,
            max_batched_completions=max_batched_completions,
        )
        for premise, hypotheses in nli_results.items():
            for hypothesis, nli_result in hypotheses.items():
                if nli_result.label != "neutral":
                    relation = Relation(
                        source=text_to_context[premise],
                        target=text_to_atom[hypothesis],
                        type=nli_result.label,
                        probability=nli_result.probability,
                        link="context_atom",
                        reasoning=nli_result.reasoning,
                    )
                    print(relation)
                    relations.append(relation)

    if rel_context_context:
        print("[Building context-context relations...]")
        nli_inputs: dict[str, list[str]] = {
            c1.get_synthetic_summary(text_only): [
                c2.get_synthetic_summary(text_only)
                for c2 in contexts.values()
                if c1 != c2
            ]
            for c1 in contexts.values()
        }
        nli_results = nli_extractor.run(
            nli_inputs,
            max_batch_relations=max_batch_relations,
            max_batched_completions=max_batched_completions,
        )
        for premise, hypotheses in nli_results.items():
            for hypothesis, nli_result in hypotheses.items():
                # TODO: Will need to change this when only retrieving relevant items
                #       due to potential assymetry.
                inverse_nli_result = nli_results[hypothesis][premise]
                if (
                    inverse_nli_result.probability > nli_result.probability
                    and inverse_nli_result.label != "neutral"
                ):
                    # Should add the inverse NLI relation rather than this one
                    continue
                if nli_result.label == "neutral":
                    # Skip neutral relations
                    continue
                relation_type = (
                    "equivalence"
                    if (nli_result.label == inverse_nli_result.label == "entailment")
                    else nli_result.label
                )
                relation = Relation(
                    source=text_to_context[premise],
                    target=text_to_context[hypothesis],
                    type=relation_type,
                    probability=nli_result.probability,
                    link="context_context",
                    reasoning=nli_result.reasoning,
                )
                print(relation)
                relations.append(relation)

    return relations


def build_relations(
        atoms: dict = {},
        contexts: dict = {},
        contexts_per_atom_only: bool = False,
        rel_atom_context: bool = True,
        rel_context_atom: bool = False,
        rel_context_context: bool = True,
        nli_extractor: Union[NLIExtractor, NLIExtractorOld, None] = None,
        text_only: bool = True
) -> List[Relation]:
    """
    Create the NLI relations between atoms and contexts. The following
    pairwise relations are considered: atom-context and context-context.
    Optionally, atom-atom relations can also be considered.

    Args:
        atoms: dict
            A dict containing the atoms in the response.
        contexts: dict
            A dict containing the contexts retrived from the vector store.
        contexts_per_atom_only: bool
            Flag indicating that for each atom only its corresponding contexts are considered. 
        rel_atom_atom: bool (default is False)
            Flag indicating the presence of atom-to-atom relationships.
        rel_atom_context: bool (default is True)
            Flag indicating the presence of context-to-atom relationships.
            NOTE: Notice the flipped order in the argument name for legacy
            compatibility reasons.
        rel_context_atom: bool (default is False)
            Flag indicating the presence of atom-to-context relationships.
            NOTE: Notice the flipped order in the argument name for legacy
            compatibility reasons.
        rel_context_context: bool (default is False)
            Flag indicating the presence of context-to-context relationships.
        nli_extractor: NLIExtractor or NLIExtractorOld
            The NLI model used for predicting the relationships.
        text_only: bool
            Flag indicating that contexts are text only. If False, then the
            contexts include (Title, Snippet, Link, Text).
    Returns:
        A list of Relations.  
    """

    assert (nli_extractor is not None), f"The NLI extractor must exist!"
    
    context_context_pairs1 = []
    context_context_pairs2 = []

    relations = []

    if rel_atom_context or rel_context_atom:
        context_to_atom_relations: list[Relation] = []
        atom_to_context_relations: list[Relation] = []

        if not contexts_per_atom_only:
            atom_context_pairs = [
                (context, atom) for atom in atoms.values() for context in contexts.values()
            ]
        else:
            atom_context_pairs = [
                (context, atom) for atom in atoms.values() for context in atom.get_contexts()
            ]

        # Create atom-context relations (i.e., Context -> Atom)
        if rel_atom_context:
            if isinstance(nli_extractor, NLIExtractorOld):
                # Get all relationships (NLI-prompt)
                context_to_atom_relations = get_nli_relations_prompting(
                    atom_context_pairs,
                    nli_scorer=nli_extractor,
                    links_type="context_atom",
                    text_only=text_only
                )
            else:
                # Get all relationships (NLI-prompt)
                context_to_atom_relations = predict_nli_relationships(
                    atom_context_pairs,
                    nli_extractor=nli_extractor,
                    links_type="context_atom",
                    text_only=text_only
                )

        if rel_context_atom:
            if not isinstance(nli_extractor, NLIExtractor):
                raise ValueError("Context-atom relation mining is only supported for NLIExtractor.")
            
            atom_to_context_relations = predict_nli_relationships(
                [(a, c) for c, a in atom_context_pairs],
                nli_extractor=nli_extractor,
                links_type="atom_context",
                text_only=text_only
            )

        relations_tmp: list[Relation] = []
        if rel_atom_context and rel_context_atom:
            assert [r1.source.id for r1 in context_to_atom_relations] == [r2.target.id for r2 in atom_to_context_relations]
            assert [r1.target.id for r1 in context_to_atom_relations] == [r2.source.id for r2 in atom_to_context_relations]
            relations_tmp = [
                r1 if r1.get_probability() > r2.get_probability() else r2
                for r1, r2 in zip(context_to_atom_relations, atom_to_context_relations)
            ]
            for i, rel in enumerate(relations_tmp):
                if context_to_atom_relations[i].type == "entailment" and atom_to_context_relations[i].type == "entailment":
                    rel.type = "equivalence"
        elif rel_atom_context:
            relations_tmp = context_to_atom_relations
        elif rel_context_atom:
            relations_tmp = atom_to_context_relations
        
        for rel in relations_tmp:
            if rel.type != "neutral":
                print(rel)
                relations.append(rel)

    # Create context-context relations
    if rel_context_context:
        print(f"[Building context-context relations...]")
        clist = [ci for ci in sorted(contexts.keys())]
        all_pairs = list(combinations(clist, 2))
        # Create all (context, context) pairs
        for ci, cj in all_pairs:
            context_i = contexts[ci]
            context_j = contexts[cj]
            context_context_pairs1.append((context_i, context_j))
            context_context_pairs2.append((context_j, context_i))

        if isinstance(nli_extractor, NLIExtractorOld):
            # Get relationships (c_i, c_j)
            relations1 = get_nli_relations_prompting(
                context_context_pairs1,
                nli_scorer=nli_extractor,
                links_type="context_context",
                text_only=text_only
            )

            # Get relationships (c_j, c_i)
            relations2 = get_nli_relations_prompting(
                context_context_pairs2,
                nli_scorer=nli_extractor,
                links_type="context_context",
                text_only=text_only
            )
        else:
            # Get relationships (c_i, c_j)
            relations1 = predict_nli_relationships(
                context_context_pairs1,
                nli_extractor=nli_extractor,
                links_type="context_context",
                text_only=text_only
            )

            # Get relationships (c_j, c_i)
            relations2 = predict_nli_relationships(
                context_context_pairs2,
                nli_extractor=nli_extractor,
                links_type="context_context",
                text_only=text_only
            )

        relations_tmp = [pair[0] if pair[0].get_probability()>pair[1].get_probability() else pair[1] for pair in zip(relations1,relations2)]
        assert len(relations_tmp) == len(relations1) # safety checks

        for rel_ind in range(len(relations_tmp)):
            if not (relations1[rel_ind].get_type() == "entailment" and relations2[
                rel_ind].get_type() == "entailment"): continue
            relations_tmp[rel_ind].type = "equivalence"
        for rel in relations_tmp:
            if rel.get_type() != "neutral":
                print(rel)
                relations.append(rel)

    print(f"[Relations built: {len(relations)}]")
    return relations


def build_markov_network(
    fact_graph: FactGraph, use_priors: bool = True, debug_mode: bool = False
) -> MarkovNetwork:
    """
    Create the Markov Network corresponding to the FactGraph.

    Args:
        fact_graph: FactGraph
            The FactGraph to use for building the Markov Network.
        use_priors: bool
            Flag indicating that atom and context priors are used in the factor definition.
        debug_mode: bool
            Enables debugging output.

    Return:
        A MarkovNetwork encoding of the problem.
    """
    # Create an empty Markov Network
    markov_network = MarkovNetwork()

    # Create the variables corresponding to the nodes in the fact graph
    print("[Building the Markov network...]")
    for node in fact_graph.get_nodes():
        x = node.id
        markov_network.add_node(x)
        if node.type == "context":
            prob = node.probability  # PRIOR_PROB_CONTEXT
            factor = DiscreteFactor(
                variables=[x], cardinality=[2], values=[1.0 - prob, prob]
            )
            markov_network.add_factors(factor)
            print(f"Adding context variable {x} with discrete factor (prior)")
        elif node.type == "atom":
            prob = node.probability  # PRIOR_PROB_ATOM
            factor = DiscreteFactor(
                variables=[x], cardinality=[2], values=[1.0 - prob, prob]
            )
            markov_network.add_factors(factor)
            print(f"Adding atom variable {x} with discrete factor (prior)")
        else:
            raise ValueError(f"Unknown node type: {node.type}")

    # Create the factors corresponding to the edges in the fact graph
    for edge in fact_graph.get_edges():
        x, y = edge.source, edge.target
        markov_network.add_edge(x, y)
        if edge.type == "entailment":  # add factor X -> Y
            prob = edge.probability
            if use_priors:
                if edge.link == "context_atom":
                    values = [1.0 - PRIOR_PROB_ATOM, PRIOR_PROB_ATOM, 1.0 - prob, prob]
                elif edge.link == "context_context":
                    values = [
                        1.0 - PRIOR_PROB_CONTEXT,
                        PRIOR_PROB_CONTEXT,
                        1.0 - prob,
                        prob,
                    ]
                elif edge.link == "atom_atom":
                    values = [1.0 - PRIOR_PROB_ATOM, PRIOR_PROB_ATOM, 1.0 - prob, prob]
                elif edge.link == "atom_context":
                    values = [
                        1.0 - PRIOR_PROB_CONTEXT,
                        PRIOR_PROB_CONTEXT,
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
            markov_network.add_factors(factor)
            print(f"Adding edge {x} - {y} with discrete factor (entailment)")
        elif edge.type == "contradiction":  # add factor X -> !Y
            prob = edge.probability
            if use_priors:
                if edge.link == "context_atom":
                    values = [1.0 - PRIOR_PROB_ATOM, PRIOR_PROB_ATOM, prob, 1.0 - prob]
                elif edge.link == "context_context":
                    values = [
                        1.0 - PRIOR_PROB_CONTEXT,
                        PRIOR_PROB_CONTEXT,
                        prob,
                        1.0 - prob,
                    ]
                elif edge.link == "atom_atom":
                    values = [1.0 - PRIOR_PROB_ATOM, PRIOR_PROB_ATOM, prob, 1.0 - prob]
                elif edge.link == "atom_context":
                    values = [
                        1.0 - PRIOR_PROB_CONTEXT,
                        PRIOR_PROB_CONTEXT,
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
            markov_network.add_factors(factor)
            print(f"Adding edge {x} - {y} with discrete factor (contradiction)")
        elif edge.type == "equivalence":
            prob = edge.probability
            factor = DiscreteFactor(
                variables=[x, y],
                cardinality=[2, 2],
                values=[prob, 1.0 - prob, 1.0 - prob, prob],
            )
            markov_network.add_factors(factor)
            print(f"Adding edge {x} - {y} with discrete factor (equivalence)")

    # Output the content of the network
    print("[Markov network created.]")
    print(markov_network)

    if debug_mode:
        print("[Markov network content...]")
        for f in markov_network.get_factors():
            print(f)

    return markov_network

class Marginal(TypedDict):
    variable: str
    probabilities: tuple[float, float]

def run_merlin(
    variable_names: list[str], markov_network: MarkovNetwork, merlin_path: str
) -> list[Marginal]:
    """
    Run inference with merlin (executable)
    """

    # Prepare the query variables (i.e., atoms)
    query_variables = [var for var in sorted(variable_names)]

    # Dump the markov network to a temporary file
    net_id = str(uuid.uuid1())
    input_filename = f"/tmp/markov_network_{net_id}.uai"
    writer = UAIWriter(markov_network)
    writer.write_uai(input_filename)

    # Get the variable name to index mapping {0: ('a0', '2'), 1: ('a1', '2')}
    vars_mapping = {}
    variables = sorted(writer.domain.items(), key=lambda x: (x[1], x[0]))
    for i, var in enumerate(variables):
        vars_mapping[i] = var[0]

    # Run merlin as a subprocess and collect the results
    exefile = merlin_path
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

    marginals: list[Marginal] = []
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

def compute_factuality_results(
    scores: dict[str, tuple[float, float]],
    fact_graph: FactGraph,
    atoms: dict[str, Atom],
    contexts: dict[str, Context],
    query: str | None = None,
    labels_human: dict[str, str] | None = None
) -> dict:
    """
    Computes the factuality score and other results, taking into consideration
    the contexts retrieved for each of the atom in the answer.
    
    Factuality score = # atoms(true) / # atoms

    Intuitively, a score of 100% means that all atoms in the answer are
    factually correct. If none of them are correct, then the score is 0%. If
    only half of the atoms are correct, then the score is 50%.

    Args:
        scores: dict
            The scores for each atom — a dictionary mapping atom IDs to scores for the atom being false/true.
        fact_graph: FactGraph
            The constructed fact graph.
        atoms: dict[str, Atom]
            The dictionary of atoms in the scored answer.
        contexts: dict[str, Context]
            The retrieved contexts.
        query: str
            The original user query if available.
        labels_human: dict[str, str] | None
            Ground-truth human annotations, mapping each supported atom to "S" and each unsupported atom to "NS".

    Returns:
        dict
            The results dictionary containing the factuality score (real values in [0, 1]) and other measures.
    """
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
    for var, (score_false, score_true) in scores.items():
        if not var.startswith("a"):
            continue

        print(f"[{var}]: Probability for {var}=0 is: {score_false}")
        print(f"[{var}]: Probability for {var}=1 is: {score_true}")

        # Check if atom is true or not
        probabilities[var] = score_true # probability of true
        if score_true > score_false:
            num_true_atoms += 1
            labels[var] = "S"
        else:
            labels[var] = "NS"

        fscore_per_atom.append({var: {"score": score_true, "support": labels[var]}})
        probval = score_true
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

    avg_logprob /= max(len(atoms), 1)
    avg_prob /= max(len(atoms), 1)
    avg_entropy = entropy / max(len(atoms), 1)
    avg_norm_entropy = norm_entropy / max(len(atoms), 1)
    fscore = num_true_atoms / max(len(atoms), 1)

    results = {}
    results["factuality_score_per_atom"] = fscore_per_atom
    results["factuality_score"] = fscore
    results["num_atoms"] = len(atoms)
    results["num_contexts"] = len(contexts)
    results["num_true_atoms"] = num_true_atoms
    results["num_false_atoms"] = len(atoms) - num_true_atoms
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
    if labels_human is not None:
        true_atoms = 0
        false_atoms = 0
        avg_brier = 0.0
        num_true_positive = 0
        num_true_negative = 0
        num_false_positive = 0
        num_false_negative = 0
        for aid, label in labels_human.items():
            if aid not in probabilities:
                # Skip atoms removed during deduplication
                continue

            if label == "S":
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
        fscore_gold = true_atoms / len(labels_human.keys())
        avg_brier /= max(len(atoms), 1)
        str_references = ""
        for aid in sorted(labels_human.keys()):
            str_references += f" {aid}: {labels_human[aid]}"
        print(f"[FactReasoner] Gold labels: {str_references}")
        print(f"[FactReasoner] Gold fscore: {fscore_gold} ({true_atoms}/{len(labels_human.keys())})")
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
    results["input"] = query
    results["scores"] = scores
    results["fact_graph_json"] = fact_graph.to_json()
    results["atoms"] = atoms
    results["contexts"] = contexts

    return results


def _extract_probabilistic_predictions(
    response: ModelResponse,
    extraction_regex: str,
    logprobs_processing_fun: Callable[[str, list[ChatCompletionTokenLogprob]], None],
) -> None:
    """
    Helper function for extracting probabilistic predictions.
    Doesn't return any value — it is the responsibility of logprobs_processing_fun
    to aggregate the results.

    Args:
        response: ModelResponse
            The model response.
        extraction_regex: str
            The regular expression used to extract the results from the model response.
            Should include one group for capturing the context and one group for
            the probabilistic result.
        logprobs_processing_fun: Callable[[str, list[ChatCompletionTokenLogprob]], None]
            A function that takes the context value and a list of ChatCompletionTokenLogprob
            objects and aggregates them into the desired result.
    """
    all_logprobs = cast(
        list[ChatCompletionTokenLogprob],
        response.choices[0].logprobs.content,  # type: ignore
    )
    choice = response.choices[0]
    if not isinstance(choice, Choices):
        raise ValueError(f"Unexpected type for model choice: {type(choice)}")
    message_content = cast(str, choice.message.content)

    # Try to trim excess tokens
    fallback_active = False
    joined_tokens = "".join([lp.token for lp in all_logprobs])
    filtered_logprobs = list(all_logprobs)
    while filtered_logprobs and not joined_tokens.startswith(message_content):
        filtered_logprobs.pop(0)
        joined_tokens = "".join([lp.token for lp in filtered_logprobs])
    if not joined_tokens.startswith(message_content):
        fallback_active = True
    token_spans: list[tuple[int, int]] = []
    current_idx = 0
    for logprob in filtered_logprobs:
        token_spans.append((current_idx, current_idx + len(logprob.token)))
        current_idx += len(logprob.token)

    result_matches = list(regex.finditer(extraction_regex, message_content))
    for result_match in result_matches:
        context = result_match.group(1)
        start_idx, end_idx = result_match.span(2)
        result_logprobs = [
            logprob
            for logprob, (s, e) in zip(filtered_logprobs, token_spans)
            if not (start_idx >= e or end_idx <= s)
        ]
        try:
            if fallback_active or not result_logprobs:
                print(
                    f"WARNING: Failed to match message content with logprob tokens for comparison {result_match.group(0).strip()}. "
                    "Falling back to nonprobabilistic model output."
                )
                result_value = result_match.group(2)
                logprobs_processing_fun(
                    context,
                    [
                        ChatCompletionTokenLogprob(
                            token=result_value,
                            logprob=1e-10,
                            top_logprobs=[TopLogprob(token=result_value, logprob=1e-10)],
                        )
                    ],
                )
            else:
                logprobs_processing_fun(context, result_logprobs)
        except Exception:
            print("WARNING: Failed to process logprobs! See details below:")
            print("Response:")
            print(message_content)
            print("Result logprobs:")
            print(result_logprobs)
            print("Match:")
            print(result_match.group(0).strip())
            print("Result value:")
            print(result_match.group(2))
            print("Exception:")
            traceback.print_exc()
            print("Continuing iteration over matches...")


def extract_weighted_scores(
    response: ModelResponse, extraction_regex: str
) -> list[tuple[str, float]]:
    """
    Extracts the results from the model responses, with scores weighted by their probability. Assumes each score is a single token

    Args:
        response: ModelResponse
            The model response.
        extraction_regex: str
            The regular expression used to extract the results from the model response.
            Should include one group for capturing the context and one group for capturing the score.

    Returns:
        list[tuple[str, float]]:
            A list of tuples containing the extracted contexts and the corresponding scores,
            weighted by the probability of the context being true.
    """
    results: list[tuple[str, float]] = []

    def logprobs_processing_fun(
        context: str, logprobs: list[ChatCompletionTokenLogprob]
    ) -> None:
        score_logprob = logprobs[0]
        aggregate_score = 0
        total_proba = 0
        for top_logprob in score_logprob.top_logprobs:
            normalised_candidate = top_logprob.token.strip().lower()
            try:
                candidate_score = int(normalised_candidate)
            except ValueError:
                continue

            candidate_proba = math.exp(top_logprob.logprob)
            aggregate_score += candidate_score * candidate_proba
            total_proba += candidate_proba
        aggregate_score /= total_proba
        results.append((context, aggregate_score))

    _extract_probabilistic_predictions(
        response, extraction_regex, logprobs_processing_fun
    )

    return results


def extract_probabilistic_labels(
    response: ModelResponse, extraction_regex: str, label_options: list[str]
) -> list[tuple[str, str, float]]:
    """
    Extracts probabilistic labels from a model response.
    Assumes all label options have distinct token prefixes.

    Args:
        response: ModelResponse
            The model response to extract labels from.
        extraction_regex: str
            The regular expression to extract labels from the response.
            Should include one group for capturing the context and one group for capturing the score.
        label_options: list[str]
            The list of possible labels values.

    Returns:
        list[tuple[str, str, float]]
            A list of tuples containing the context, the extracted label, and its probability.
    """
    results: list[tuple[str, str, float]] = []

    def logprobs_processing_fun(
        context: str, logprobs: list[ChatCompletionTokenLogprob]
    ) -> None:
        label_logprob = logprobs[0]
        label_log_probabilities: dict[str, float] = defaultdict(lambda: -math.inf)
        normalised_label_options = [label.strip().lower() for label in label_options]
        for top_logprob in label_logprob.top_logprobs:
            normalised_candidate = top_logprob.token.strip().lower()
            for label_option in normalised_label_options:
                if label_option.startswith(normalised_candidate):
                    # Found matching label
                    label_log_probabilities[label_option] = logsumexp(  # type: ignore
                        [label_log_probabilities[label_option], top_logprob.logprob]
                    )
                    break

        normalised_top_label, top_log_proba = max(
            label_log_probabilities.items(), key=lambda x: x[1]
        )
        total_log_proba = logsumexp(list(label_log_probabilities.values()))
        label_proba = math.exp(top_log_proba - total_log_proba)  # type: ignore
        label = label_options[normalised_label_options.index(normalised_top_label)]
        results.append((context, label, label_proba))

    _extract_probabilistic_predictions(
        response, extraction_regex, logprobs_processing_fun
    )

    return results