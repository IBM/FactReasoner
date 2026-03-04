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

from typing import Any, Dict, Optional, Union

# Defaut prior probabilities for atoms and contexts
PRIOR_PROB_ATOM = 0.5
PRIOR_PROB_CONTEXT = 0.9

class Atom:
    """
    Represents an atomic unit of the model's response.
    """

    def __init__(
            self,
            id: str,
            text: str,
            label: str = None
    ):
        """
        Atom constructor.
        Args:
            id: str
                Unique ID of the atom e.g., `a1`.
            text: int
                The text associated with the atom.
            label: str
                The gold label associated with the atom (S or NS).
        """

        self.id = id
        self.text = text
        self.original = text  # keeps around the original atom
        self.label = label
        self.contexts = {}
        self.search_results = []
        self.probability = PRIOR_PROB_ATOM  # prior probability of the atom being true

    def __str__(self) -> str:
        return f"Atom {self.id}: {self.text}"

    def get_text(self):
        return self.text
    
    def get_summary(self):
        return self.text

    def set_text(self, new_text: str):
        self.text = new_text

    def get_original(self):
        return self.original

    def set_original(self, new_original: str):
        self.original = new_original

    def get_label(self):
        return self.label

    def add_context(
            self,
            context
    ):
        """
        Add a context relevat to the atom.

        Args:
            context: Context
                The context relevat to the atom.
        """
        self.contexts[context.id] = context

    def add_contexts(
            self,
            contexts
    ):
        """
        Add a list of contexts relevant to the atom.
        Args:
            context: list
                The contexts relevant to the atom.
        """
        for context in contexts:
            self.contexts[context.id] = context

    def get_contexts(self):
        """
        Return the contexts relevant to the atom.
        """
        return self.contexts


class Context:
    """
    Represents a context retrieved from an external source of knowledge.
    """

    def __init__(
            self,
            id: str,
            atom: Optional[Atom],
            text: str = "",
            synthetic_summary: Optional[str] = None,
            title: str = "",
            link: str = "",
            snippet: str = ""
    ):
        """
        Context constructor.
        Args:
            id: str
                Unique ID for the context e.g., `c1_1`.
            atom: Atom
                The reference atom (from the answer)
            text: str
                The text of the context (one or more paragraphs)
            sumary: str
                The summary of the context (one or more paragraphs)
            title: str
                The title of the context (e.g., title of the wikipedia page)
            link: str
                The link to a web page if the context is a search results. It
                is assumed to be empty if the context is a retrieved passage.
            snippet: str
                The snippet associated with a search result
        """

        self.id = id
        self.atom = atom
        self.text = text
        self.synthetic_summary = synthetic_summary
        self.title = title
        self.link = link
        self.snippet = snippet
        self.probability = PRIOR_PROB_CONTEXT  # prior probability of the context being true

    def __str__(self) -> str:
        return f"Context {self.id} [{self.title}]: {self.text}"

    def get_id(self):
        return self.id

    def get_summary(self):
        return self.synthetic_summary if self.synthetic_summary is not None else ""

    def get_text(self):
        if self.snippet != "" and self.text != "":
            return "Snippet/Summary of Text:\n\n" + self.snippet + "\n\n" + "Text:\n\n" + self.text 
        elif self.snippet == "" and self.text != "":
            return self.text
        elif self.snippet != "" and self.text == "":
            return self.snippet
        else:
            return ""
       
    def get_title(self):
        return self.title

    def get_link(self):
        return self.link

    def get_snippet(self):
        return self.snippet

    def set_atom(self, atom):
        self.atom = atom

    def set_link(self, link: str):
        self.link = link

    def set_snippet(self, snippet: str):
        self.snippet = snippet

    def set_synthetic_summary(self, synthetic_summary: str):
        self.synthetic_summary = synthetic_summary

    def get_probability(self):
        return self.probability

    def set_probability(self, probability):
        self.probability = probability

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "link": self.link,
            "snippet": self.snippet,
            "synthetic_summary": self.get_summary(),
            "probability": self.probability,
        }     


class Relation:
    """
    Represents the NLI relationship between a source text and a target text.
    """

    def __init__(
            self,
            source: Union[Atom, Context],
            target: Union[Atom, Context],
            type: str,
            probability: float,
            link: str
    ):
        """
        Relation constructor.
        Args:
            source: [Atom|Context]
                The source atom or context.
            target: [Atom|Context]
                The target atom or context.
            type: str
                The relation type: ["entailment", "contradiction", "equivalence"].
                Note that `entailment` is not symmetric while `contradiction` and
                `equivalence` are symmetric relations.
            probability: float
                The probability value associated with the NLI relation.
            link: str
                The link type: [context_atom, context_context, atom_atom]
        
        Comment: `entailment` is not symmetric, while `contradiction`, `neutral`
        and `equivalence are symmetric. Namely, if A contradicts B then B
        contradicts A (same with neutral, equivalence). However, if A entails B
        then B doesn't necessarily entails A.
        """

        assert (type in ["entailment", "contradiction", "equivalence", "neutral"]), \
            f"Unknown relation type: {type}."
        assert (link in ["context_atom", "context_context", "atom_atom"]), \
            f"Unknown link type: {link}"

        self.source = source
        self.target = target
        self.type = type
        self.probability = probability
        self.link = link

    def __str__(self) -> str:
        return f"[{self.source.id} -> {self.target.id}] : {self.type} : {self.probability}"

    def get_type(self) -> str:
        return self.type

    def get_probability(self) -> float:
        return self.probability

