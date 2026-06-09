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

from itertools import combinations
from typing import Dict, List, Tuple, Union

import asyncio
import nltk
from nltk.tokenize import sent_tokenize
import concurrent.futures


# Local imports
from .base import Atom, Context, Relation
from .atomizer import Atomizer
from .retriever import ContextRetriever
from .nli import NLIExtractor
from fact_reasoner.utils import punctuation_only_inside_quotes


def predict_nli_relationships(
    object_pairs: List[Tuple[Union[Atom, Context], Union[Atom, Context]]],
    nli_extractor: NLIExtractor,
    links_type: str = "context_atom",
    use_summary: bool = False,
) -> List[Relation]:
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

    # Safety checks
    assert nli_extractor is not None, "NLI extractor cannot be None."
    assert isinstance(
        nli_extractor, NLIExtractor
    ), "NLI extractor must be NLIExtractor."

    # Set up the premises and hypotheses
    if use_summary:
        premises = [
            pair[0] if isinstance(pair[0], str) else pair[0].get_summary()
            for pair in object_pairs
        ]
        hypotheses = [
            pair[1] if isinstance(pair[1], str) else pair[1].get_summary()
            for pair in object_pairs
        ]
    else:
        premises = [
            pair[0] if isinstance(pair[0], str) else pair[0].get_text()
            for pair in object_pairs
        ]
        hypotheses = [
            pair[1] if isinstance(pair[1], str) else pair[1].get_text()
            for pair in object_pairs
        ]

    # Safety checks
    assert len(premises) == len(hypotheses)

    # Extract the NLI relationships between premises and hyptheses
    print(f"[NLI] Processing {len(premises)} potential relationships ...")
    # results = [nli_extractor.run(premises[i], hypotheses[i]) for i in range(len(premises))]
    try:
        loop = asyncio.get_running_loop()

        with concurrent.futures.ThreadPoolExecutor() as pool:
            results = pool.submit(
                asyncio.run, nli_extractor.run_batch(premises, hypotheses)
            ).result()
    except RuntimeError:
        results = asyncio.run(nli_extractor.run_batch(premises, hypotheses))

    relations = []
    for ii, result in enumerate(results):
        label = result.get("label") or "neutral"
        probability = result.get("probability", 0.0)
        link_type = links_type if links_type is not None else "unknown"
        rel = Relation(
            source=object_pairs[ii][0],
            target=object_pairs[ii][1],
            type=label,
            probability=probability,
            link=link_type,
        )
        relations.append(rel)

    return relations


def build_atoms(response: str, atom_extractor: Atomizer) -> Dict[str, Atom]:
    """
    Decompose the given response into atomic units (i.e., atoms).

    Args:
        response: str
            The string representing the LLM response.
        atom_extractor: Atomizer
            The atom extractor.

    Returns:
        Dict[str, Atom]: A dict containing the atoms of the response.
    """

    assert (
        response is not None and len(response) > 0
    ), f"Please ensure a non empty response."

    result = atom_extractor.run(response)

    candidates = [Atom(id="a" + str(i), text=v) for i, v in enumerate(result.values())]

    return {atom.id: atom for atom in candidates}


def build_contexts(
    atoms: Dict[str, Atom] = {},
    query: str = None,
    retriever: ContextRetriever = None,
    use_fast_retriever: bool = True,
) -> Dict[str, Context]:
    """
    Retrieve the relevant contexts for the input atoms.

    Args:
        atoms: dict
            A dict containing the atoms in the response.
        query: str
            The user query text.
        retriever: ContextRetriever
            The context retriever (chromadb, langchain, google).
        use_fast_retriever: bool
            Use the fast multi-threaded context retriever.

    Returns:
        Dict[str, Context]: A dict containing the retrieved contexts.
    """

    assert len(atoms) > 0, "Please ensure a non-empty list of atoms."
    assert (
        retriever is not None
    ), "Please ensure an existing context retriever instance."

    # Building the contexts
    contexts = {}

    if not use_fast_retriever:
        # Retrieve contexts for the atoms
        for aid, atom in atoms.items():

            # Sequential but with multi-threaded top-k retrieval
            retrieved_contexts = retriever.context_retriever.query(
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
                        snippet=context["snippet"],
                        # An empty summary means that the context is not relevant,
                        # therefore we do not add it to the list of contexts for the pipeline
                    )
                    for j, context in enumerate(retrieved_contexts)
                ]

                for ctxt in contexts_per_atom:
                    contexts[ctxt.id] = ctxt
                atoms[aid].add_contexts(contexts_per_atom)

        # Retrieve the contexts for the question
        retrieved_contexts = retriever.context_retriever.query(
            text=query,
        )

        if len(retrieved_contexts) > 0:
            contexts_per_query = [
                Context(
                    id="c_q_" + str(j),
                    atom=None,
                    text=context["text"],
                    title=context["title"],
                    link=context["link"],
                    snippet=context["snippet"],
                    # An empty summary means that the context is not relevant,
                    # therefore we do not add it to the list of contexts for the pipeline
                )
                for j, context in enumerate(retrieved_contexts)
            ]

            for ctxt in contexts_per_query:
                contexts[ctxt.id] = ctxt
    else:
        # Retrieve contexts for all atoms in parallel
        contexts = retriever.retrieve_all(atoms=atoms, query=query)

    return contexts


def remove_duplicated_atoms(atoms: Dict[str, Atom]) -> Dict[str, Atom]:
    """
    Remove the duplicated atoms.

    Args:
        atoms: Dict[str, Any]
            The dict containing the atoms.

    Returns:
        Dict[str, Any]: A dict containing the unique atoms.
    """

    seen = set()
    out = {}
    for k, v in atoms.items():
        text = v.get_text()
        if text not in seen:
            out[k] = v
            seen.add(text)
    return out


def remove_duplicated_contexts(
    contexts: Dict[str, Context], atoms: Dict[str, Atom], check_summary: bool = False
) -> dict:
    """
    Remove the duplicated contexts.

    Args:
        contexts: Dict[str, Context]
            The dict containing the contexts.
        atoms: Dict[str, Atom]
            The dict containing the atoms.
        check_summary: bool
            Whether to check the summary of the contexts.

    Returns:
        The updated dicts containing the contexts and atoms.
    """

    seen = set()
    out = {}
    for k, v in contexts.items():
        text = (
            v.get_text()
            if not check_summary or v.get_summary() is None
            else v.get_summary()
        )
        if text not in seen:
            seen.add(text)
            out[k] = v
        elif v.atom and v.atom.id in atoms:
            del atoms[v.atom.id].contexts[k]

    return out, atoms


def is_relevant_context(context: str) -> bool:
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

    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            print(f"'{resource}' not found. Downloading...")
            nltk.download(resource)

    sentences = sent_tokenize(context)
    num_sentences = len(sentences)

    # we filter out summaries of only one sentence of the form: "the context does not..."
    if (
        num_sentences == 1
        and punctuation_only_inside_quotes(sentences[0])
        and ("the context does not" in sentences[0].lower())
    ):
        return False

    return True


def build_relations(
    atoms: Dict[str, Atom] = {},
    contexts: Dict[str, Context] = {},
    contexts_per_atom_only: bool = False,
    rel_atom_context: bool = True,
    rel_context_context: bool = True,
    nli_extractor: NLIExtractor = None,
    use_summarized_contexts: bool = False,
) -> List[Relation]:
    """
    Create the NLI relations between atoms and contexts. The following
    pairwise relations are considered: atom-context and context-context.

    Args:
        atoms: dict
            A dict containing the atoms in the response.
        contexts: dict
            A dict containing the contexts retrived from the vector store.
        contexts_per_atom_only: bool
            Flag indicating that for each atom only its corresponding contexts are considered.
        rel_atom_context: bool (default is True)
            Flag indicating the presence of atom-to-context relationships.
        rel_context_context: bool (default is False)
            Flag indicating the presence of context-to-context relationships.
        nli_extractor: NLIExtractor
            The NLI model used for predicting the relationships.
        use_summarized_contexts: bool
            Flag indicating that summarized contexts are used. If False, then the
            contexts include the extracted text.
    Returns:
        A list of Relations.
    """

    assert len(atoms) > 0, f"The atoms must be initialized!"
    assert len(contexts) > 0, f"The contexts must be initialized!"
    assert nli_extractor is not None, f"The NLI extractor must exist!"

    atom_context_pairs = []
    context_context_pairs1 = []
    context_context_pairs2 = []

    relations = []

    # Create atom-context relations (i.e., Context -> Atom)
    if rel_atom_context:
        print(f"[NLI] Building atom-context relations...")
        if not contexts_per_atom_only:  # use all contexts for each atom
            # Create the (context, atom) pairs
            print(f"[NLI] Using all contexts retrieved.")
            for _, atom in atoms.items():
                for _, context in contexts.items():
                    atom_context_pairs.append((context, atom))
        else:
            print(f"[NLI] Using only the contexts retrieved per atom.")
            # Create the (context, atom) pairs
            for _, atom in atoms.items():
                for context in atom.get_contexts():
                    atom_context_pairs.append((context, atom))

        # Get all relationships (NLI-prompt)
        all_rels = predict_nli_relationships(
            atom_context_pairs,
            nli_extractor=nli_extractor,
            links_type="context_atom",
            use_summary=use_summarized_contexts,
        )

        # Filter out the neutral relationships
        for rel in all_rels:
            if rel.get_type() != "neutral":
                print(f"[NLI] Found relation: {rel}")
                relations.append(rel)

    # Create context-context relations
    if rel_context_context:
        print(f"[NLI] Building context-context relations...")
        clist = [ci for ci in sorted(contexts.keys())]
        all_pairs = list(combinations(clist, 2))
        # Create all (context, context) pairs
        for ci, cj in all_pairs:
            context_i = contexts[ci]
            context_j = contexts[cj]
            context_context_pairs1.append((context_i, context_j))
            context_context_pairs2.append((context_j, context_i))

        # Get relationships (c_i, c_j)
        relations1 = predict_nli_relationships(
            context_context_pairs1,
            nli_extractor=nli_extractor,
            links_type="context_context",
            use_summary=use_summarized_contexts,
        )

        # Get relationships (c_j, c_i)
        relations2 = predict_nli_relationships(
            context_context_pairs2,
            nli_extractor=nli_extractor,
            links_type="context_context",
            use_summary=use_summarized_contexts,
        )

        relations_tmp = [
            (
                pair[0]
                if pair[0].get_probability() > pair[1].get_probability()
                else pair[1]
            )
            for pair in zip(relations1, relations2)
        ]
        assert len(relations_tmp) == len(relations1)  # safety checks

        for rel_ind in range(len(relations_tmp)):
            if not (
                relations1[rel_ind].get_type() == "entailment"
                and relations2[rel_ind].get_type() == "entailment"
            ):
                continue
            relations_tmp[rel_ind].type = "equivalence"
        for rel in relations_tmp:
            if rel.get_type() != "neutral":
                print(f"[NLI] Found relation: {rel}")
                relations.append(rel)

    print(f"[NLI] Relations built: {len(relations)}")
    return relations
