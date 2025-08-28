from typing import Any, Optional, Union

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
            label: str | None = None,
            contexts: dict[str, "Context"] = {},
            metadata: dict[str, Any] = {},
    ):
        """
        Atom constructor.
        Args:
            id: str
                Unique ID of the atom e.g., `a1`.
            text: int
                The text associated with the atom.
            label: str | None
                The gold label associated with the atom (S or NS).
            contexts: dict[str, Context]
                The contexts associated with the atom.
            metadata: dict[str, Any]
                Metadata associated with the atom.
        """

        self.id = id
        self.text = text
        self.original = text  # keeps around the original atom
        self.label = label
        self.contexts = contexts
        self.search_results = []
        self.probability = PRIOR_PROB_ATOM  # prior probability of the atom being true
        self.metadata = metadata

    def __str__(self) -> str:
        return f"Atom {self.id}: {self.text}"

    def get_text(self, text_only: bool = True):
        return self.text
    
    def get_synthetic_summary(self, text_only: bool = True):
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
            snippet: str = "",
            metadata: dict[str, Any] = {}
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
            metadata: dict[str, Any]
                Any additional metadata associated with the context.
        """

        self.id = id
        self.atom = atom
        self.text = text
        self.synthetic_summary = synthetic_summary
        self.title = title
        self.link = link
        self.snippet = snippet
        self.probability = PRIOR_PROB_CONTEXT  # prior probability of the context being true
        self.metadata = metadata

    def __str__(self) -> str:
        return f"Context {self.id} [{self.title}]: {self.text}"

    def get_id(self):
        return self.id

    def get_synthetic_summary(self, text_only: bool = True):
        if self.synthetic_summary is not None:
            return self.synthetic_summary
        else:
            return self.get_text(text_only)

    # def get_text2(self):
    #    return f"{self.title}\n{self.snippet}\n{self.link}\n{self.text}"

    def get_snippet_and_text(self):
        if self.snippet != "" and self.text != "":
            return "Snippet/Summary of Text:\n\n" + self.snippet + "\n\n" + "Text:\n\n" + self.text 
        elif self.snippet == "" and self.text != "":
            return self.text
        elif self.snippet != "" and self.text == "":
            return self.snippet
        else:
            return ""

    def get_text(self, text_only: bool = True):
        if text_only:
            return self.text
        else:
            return self.get_snippet_and_text()
            # return f"Title: {self.title}\nSummary: {self.snippet}\nLink: {self.link}\nText: {self.text}"
        
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
            link: str,
            reasoning: str = "",
            metadata: dict[str, Any] = {}
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
                The link type: [context_atom, context_context, atom_atom, atom_context]
            reasoning: str
                Model's reasoning for this relation.
            metadata: dict[str, Any]
                Additional metadata associated with the relation.
        
        Comment: `entailment` is not symmetric, while `contradiction`, `neutral`
        and `equivalence are symmetric. Namely, if A contradicts B then B
        contradicts A (same with neutral, equivalence). However, if A entails B
        then B doesn't necessarily entails A.
        """

        assert (type in ["entailment", "contradiction", "equivalence", "neutral"]), \
            f"Unknown relation type: {type}."
        assert (link in ["context_atom", "context_context", "atom_atom", "atom_context"]), \
            f"Unknown link type: {link}"

        self.source = source
        self.target = target
        self.type = type
        self.probability = probability
        self.link = link
        self.reasoning = reasoning
        self.metadata = metadata

    def __str__(self) -> str:
        return f"[{self.source.id} -> {self.target.id}] : {self.type} : {self.probability}"

    def get_type(self) -> str:
        return self.type

    def get_probability(self) -> float:
        return self.probability