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

"""Unit tests for fact_reasoner.core.utils module."""

import pytest
from src.fact_reasoner.core.utils import (
    Atom,
    Context,
    Relation,
    PRIOR_PROB_ATOM,
    PRIOR_PROB_CONTEXT,
    remove_duplicated_atoms,
    remove_duplicated_contexts,
    is_relevant_context,
)


class TestAtom:
    """Tests for Atom class."""

    def test_atom_creation(self):
        atom = Atom(id="a0", text="Test atom text")
        assert atom.id == "a0"
        assert atom.text == "Test atom text"
        assert atom.original == "Test atom text"
        assert atom.label is None
        assert atom.probability == PRIOR_PROB_ATOM

    def test_atom_with_label(self):
        atom = Atom(id="a1", text="Labeled atom", label="S")
        assert atom.get_label() == "S"

    def test_atom_str(self):
        atom = Atom(id="a0", text="Test text")
        assert str(atom) == "Atom a0: Test text"

    def test_atom_get_set_text(self):
        atom = Atom(id="a0", text="Original")
        assert atom.get_text() == "Original"
        atom.set_text("Modified")
        assert atom.get_text() == "Modified"
        assert atom.get_original() == "Original"

    def test_atom_set_original(self):
        atom = Atom(id="a0", text="Text")
        atom.set_original("New original")
        assert atom.get_original() == "New original"

    def test_atom_get_summary(self):
        atom = Atom(id="a0", text="Summary text")
        assert atom.get_summary() == "Summary text"

    def test_atom_add_context(self):
        atom = Atom(id="a0", text="Test")
        context = Context(id="c0", atom=atom, text="Context text")
        atom.add_context(context)
        assert "c0" in atom.get_contexts()
        assert atom.get_contexts()["c0"] == context

    def test_atom_add_contexts(self):
        atom = Atom(id="a0", text="Test")
        contexts = [
            Context(id="c0", atom=atom, text="Context 0"),
            Context(id="c1", atom=atom, text="Context 1"),
        ]
        atom.add_contexts(contexts)
        assert len(atom.get_contexts()) == 2
        assert "c0" in atom.get_contexts()
        assert "c1" in atom.get_contexts()


class TestContext:
    """Tests for Context class."""

    def test_context_creation(self):
        context = Context(
            id="c0",
            atom=None,
            text="Context text",
            title="Title",
            link="https://example.com",
            snippet="Snippet"
        )
        assert context.id == "c0"
        assert context.text == "Context text"
        assert context.title == "Title"
        assert context.link == "https://example.com"
        assert context.snippet == "Snippet"
        assert context.probability == PRIOR_PROB_CONTEXT

    def test_context_str(self):
        context = Context(id="c0", atom=None, text="Text", title="Title")
        assert "Context c0" in str(context)
        assert "Title" in str(context)

    def test_context_get_text_with_snippet_and_text(self):
        context = Context(id="c0", atom=None, text="Main text", snippet="Snippet")
        result = context.get_text()
        assert "Snippet/Summary of Text:" in result
        assert "Main text" in result
        assert "Snippet" in result

    def test_context_get_text_only_text(self):
        context = Context(id="c0", atom=None, text="Main text", snippet="")
        assert context.get_text() == "Main text"

    def test_context_get_text_only_snippet(self):
        context = Context(id="c0", atom=None, text="", snippet="Snippet only")
        assert context.get_text() == "Snippet only"

    def test_context_get_text_empty(self):
        context = Context(id="c0", atom=None, text="", snippet="")
        assert context.get_text() == ""

    def test_context_get_summary_with_synthetic(self):
        context = Context(id="c0", atom=None, text="Text")
        context.set_synthetic_summary("Synthetic summary")
        assert context.get_summary() == "Synthetic summary"

    def test_context_get_summary_without_synthetic(self):
        context = Context(id="c0", atom=None, text="Text")
        assert context.get_summary() == ""

    def test_context_set_atom(self):
        atom = Atom(id="a0", text="Atom text")
        context = Context(id="c0", atom=None, text="Text")
        context.set_atom(atom)
        assert context.atom == atom

    def test_context_probability(self):
        context = Context(id="c0", atom=None, text="Text")
        assert context.get_probability() == PRIOR_PROB_CONTEXT
        context.set_probability(0.75)
        assert context.get_probability() == 0.75

    def test_context_to_json(self):
        context = Context(
            id="c0",
            atom=None,
            text="Text",
            title="Title",
            link="https://example.com",
            snippet="Snippet"
        )
        json_data = context.to_json()
        assert json_data["id"] == "c0"
        assert json_data["title"] == "Title"
        assert json_data["text"] == "Text"
        assert json_data["link"] == "https://example.com"
        assert json_data["snippet"] == "Snippet"
        assert "probability" in json_data


class TestRelation:
    """Tests for Relation class."""

    def test_relation_creation_entailment(self):
        atom = Atom(id="a0", text="Atom")
        context = Context(id="c0", atom=atom, text="Context")
        relation = Relation(
            source=context,
            target=atom,
            type="entailment",
            probability=0.9,
            link="context_atom"
        )
        assert relation.source == context
        assert relation.target == atom
        assert relation.type == "entailment"
        assert relation.probability == 0.9
        assert relation.link == "context_atom"

    def test_relation_creation_contradiction(self):
        atom = Atom(id="a0", text="Atom")
        context = Context(id="c0", atom=atom, text="Context")
        relation = Relation(
            source=context,
            target=atom,
            type="contradiction",
            probability=0.85,
            link="context_atom"
        )
        assert relation.get_type() == "contradiction"
        assert relation.get_probability() == 0.85

    def test_relation_creation_equivalence(self):
        context1 = Context(id="c0", atom=None, text="Context 1")
        context2 = Context(id="c1", atom=None, text="Context 2")
        relation = Relation(
            source=context1,
            target=context2,
            type="equivalence",
            probability=0.95,
            link="context_context"
        )
        assert relation.type == "equivalence"
        assert relation.link == "context_context"

    def test_relation_creation_neutral(self):
        atom = Atom(id="a0", text="Atom")
        context = Context(id="c0", atom=atom, text="Context")
        relation = Relation(
            source=context,
            target=atom,
            type="neutral",
            probability=0.5,
            link="context_atom"
        )
        assert relation.type == "neutral"

    def test_relation_invalid_type(self):
        atom = Atom(id="a0", text="Atom")
        context = Context(id="c0", atom=atom, text="Context")
        with pytest.raises(AssertionError):
            Relation(
                source=context,
                target=atom,
                type="invalid_type",
                probability=0.5,
                link="context_atom"
            )

    def test_relation_invalid_link(self):
        atom = Atom(id="a0", text="Atom")
        context = Context(id="c0", atom=atom, text="Context")
        with pytest.raises(AssertionError):
            Relation(
                source=context,
                target=atom,
                type="entailment",
                probability=0.5,
                link="invalid_link"
            )

    def test_relation_str(self):
        atom = Atom(id="a0", text="Atom")
        context = Context(id="c0", atom=atom, text="Context")
        relation = Relation(
            source=context,
            target=atom,
            type="entailment",
            probability=0.9,
            link="context_atom"
        )
        result = str(relation)
        assert "c0" in result
        assert "a0" in result
        assert "entailment" in result


class TestRemoveDuplicatedAtoms:
    """Tests for remove_duplicated_atoms function."""

    def test_no_duplicates(self):
        atoms = {
            "a0": Atom(id="a0", text="First atom"),
            "a1": Atom(id="a1", text="Second atom"),
        }
        result = remove_duplicated_atoms(atoms)
        assert len(result) == 2

    def test_with_duplicates(self):
        atoms = {
            "a0": Atom(id="a0", text="Same text"),
            "a1": Atom(id="a1", text="Same text"),
            "a2": Atom(id="a2", text="Different text"),
        }
        result = remove_duplicated_atoms(atoms)
        assert len(result) == 2
        assert "a0" in result  # First one should be kept
        assert "a2" in result

    def test_empty_dict(self):
        result = remove_duplicated_atoms({})
        assert len(result) == 0

    def test_all_duplicates(self):
        atoms = {
            "a0": Atom(id="a0", text="Same"),
            "a1": Atom(id="a1", text="Same"),
            "a2": Atom(id="a2", text="Same"),
        }
        result = remove_duplicated_atoms(atoms)
        assert len(result) == 1


class TestRemoveDuplicatedContexts:
    """Tests for remove_duplicated_contexts function."""

    def test_no_duplicates(self):
        atom = Atom(id="a0", text="Atom")
        contexts = {
            "c0": Context(id="c0", atom=atom, text="First context"),
            "c1": Context(id="c1", atom=atom, text="Second context"),
        }
        atom.add_contexts(list(contexts.values()))
        atoms = {"a0": atom}

        result_contexts, result_atoms = remove_duplicated_contexts(contexts, atoms)
        assert len(result_contexts) == 2

    def test_with_duplicates(self):
        atom = Atom(id="a0", text="Atom")
        contexts = {
            "c0": Context(id="c0", atom=atom, text="Same text"),
            "c1": Context(id="c1", atom=atom, text="Same text"),
            "c2": Context(id="c2", atom=atom, text="Different text"),
        }
        atom.add_contexts(list(contexts.values()))
        atoms = {"a0": atom}

        result_contexts, result_atoms = remove_duplicated_contexts(contexts, atoms)
        assert len(result_contexts) == 2

    def test_empty_contexts(self):
        result_contexts, result_atoms = remove_duplicated_contexts({}, {})
        assert len(result_contexts) == 0


class TestIsRelevantContext:
    """Tests for is_relevant_context function."""

    def test_relevant_context(self):
        context = "Albert Einstein was born in Ulm, Germany in 1879."
        assert is_relevant_context(context) is True

    def test_irrelevant_not_provide_information(self):
        context = "The context does not provide information about the atom."
        assert is_relevant_context(context) is False

    def test_irrelevant_403_error(self):
        context = "Due to a 403 forbidden error, the page cannot be accessed."
        assert is_relevant_context(context) is False

    def test_irrelevant_not_verifiable(self):
        context = "It is not possible to verify the given atom."
        assert is_relevant_context(context) is False

    def test_relevant_long_context(self):
        context = """
        Albert Einstein was a German-born theoretical physicist who developed
        the theory of relativity. He was born on March 14, 1879, in Ulm,
        in the Kingdom of Württemberg in the German Empire.
        """
        assert is_relevant_context(context) is True

    def test_irrelevant_single_sentence_does_not(self):
        context = "The context does not mention anything about Einstein."
        assert is_relevant_context(context) is False

    def test_irrelevant_permission_denied(self):
        context = "You don't have permission to view this page."
        assert is_relevant_context(context) is False

    def test_irrelevant_atom_statement(self):
        context = "The atom statement cannot be verified."
        assert is_relevant_context(context) is False
