from abc import abstractmethod
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dotenv import load_dotenv
import json
import logging
import math
import os
from pathlib import Path
import regex
import shutil
import statistics
import time
from typing import (
    Any,
    Callable,
    Literal,
    NotRequired,
    Protocol,
    TypedDict,
    cast,
    override,
)

import datasets
import dill
import pandas as pd
from tenacity import Retrying, stop_after_attempt, wait_random_exponential
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm.auto import tqdm

from src.fact_reasoner.argumentation_framework import ArgumentationFramework
from src.fact_reasoner.atom_extractor import AtomExtractor
from src.fact_reasoner.batch_atom_reviser import BatchAtomReviser
from src.fact_reasoner.coverage_evaluator import CoverageEvaluator
from src.fact_reasoner.fact_components import Atom, Context, Relation
from src.fact_reasoner.fact_graph import FactGraph
from src.fact_reasoner.fact_utils import (
    build_markov_network,
    build_relations,
    compute_factuality_results,
    run_merlin,
)
from src.fact_reasoner.nli_extractor import NLIExtractor
from src.fact_reasoner.output_generator import OutputGenerator
from src.fact_reasoner.qag_processor import (
    Answer,
    AnswerRelation,
    QagProcessor,
    QuestionData,
    print_answers,
)
from src.fact_reasoner.qag_utils import (
    compare_quantities_with_units_definition,
)
from src.fact_reasoner.query_context_summarizer import QueryContextSummarizer
from src.fact_reasoner.relevance_estimator import RelevanceEstimator
from src.fact_reasoner.utils import std_out_err_redirect_tqdm

type FieldId = str
type Question = str

MODEL_OUTPUT_FIELD_ID = "__new_model_output__"


class BaseFieldData(TypedDict):
    """
    Stores content and basic metadata for a context or model
    output field from a sample.
    """

    id: str
    field_type: Literal["context", "answer"]
    original_content: str
    cleaned_content: str
    summary: str
    metadata: dict[str, Any]


class AtomizedFieldData(BaseFieldData):
    """
    Stores extracted atoms for a context or model output field.
    """

    atoms: list[str]


class QagFieldData(BaseFieldData):
    """
    Stores generated summary, questions and answers for a context
    or model output field.
    """

    questions: list[str]
    answers: dict[FieldId, dict[Question, QuestionData]]


class ComprehensivenessResult(TypedDict):
    """
    Stores results of comprehensiveness evaluation for a single sample
    """

    sample: dict
    user_query: str
    answer_field: str
    field_data: dict[FieldId, BaseFieldData]
    atoms: dict[str, Atom]
    contexts: dict[str, Context]
    relations: list[Relation]
    fact_graph: FactGraph
    probabilistic_results: dict
    argumentative_results: dict[str, dict]
    comprehensiveness_score: float
    covered_contexts: list[str]
    uncovered_contexts: list[str]
    context_equivalence_clusters: dict[str, list[str]]
    uncovered_context_basis: dict[str, list[str]]
    comprehensiveness_eval_main_score: bool | float | int | None
    comprehensiveness_eval_results: NotRequired[Any]


class FactGraphMiner(Protocol):
    """
    A protocol for constructing fact graph components from unstructured
    contexts and model outputs for evaluating comprehensiveness.
    """

    @abstractmethod
    def process_sample(
        self,
        sample: dict,
        question_field: FieldId,
        base_field_data: dict[FieldId, BaseFieldData],
    ) -> dict:
        """
        Processes sample for use with the given fact graph miner and
        returns the updated sample dictionary. This enables performing
        general preprocessing on the full sample before calling __call__
        for all the individual model outputs, potentially resulting in
        efficiency improvements.

        Args:
            sample: dict
                The sample to process, including the user query, contexts
                and model outputs as fields.
            question_field: str
                The field name containing the user query.
            base_field_data: dict[FieldId, BaseFieldData]
                The processed field data.

        Returns:
            dict:
                The updated sample dictionary.
        """
        ...

    @abstractmethod
    def __call__(
        self,
        sample: dict,
        question_field: FieldId,
        answer_field: FieldId,
        base_field_data: dict[FieldId, BaseFieldData],
    ) -> tuple[dict, dict[str, Context], dict[str, Atom], list[Relation]]:
        """
        Runs the fact graph miner on a single sample and model output (answer).

        Args:
            sample: dict
                The sample to process, including the user query, contexts
                and model outputs as fields.
            question_field: str
                The field name containing the user query.
            answer_field: str
                The field name containing the evaluated model output.
            base_field_data: dict[str, BaseFieldData]
                The processed field data.

        Returns:
            tuple[dict, dict[str, Context], dict[str, Atom], list[Relation]]
                A tuple containing the sample dictionary updated with any additional
                data from graph mining, a dictionary of contexts, a dictionary of atoms
                and a list of relations to be used for constructing a fact graph.
        """
        ...


class EndToEndFactGraphMiner(FactGraphMiner):
    """
    A miner constructing the fact graph end-to-end using an LLM.
    """

    def __init__(self, coverage_evaluator: CoverageEvaluator):
        """
        Initializes the EndToEndFactGraphMiner.

        Args:
            coverage_evaluator: float
                The CoverageEvaluator instance to be used for directly
                determining covered/uncovered contexts.
        """
        self.coverage_evaluator = coverage_evaluator

    @override
    def process_sample(
        self,
        sample: dict,
        question_field: FieldId,
        base_field_data: dict[FieldId, BaseFieldData],
    ) -> dict:
        """
        Returns unchanged sample, as the end-to-end fact graph miner doesn't
        require any special preprocessing.

        Args:
            sample: dict
                The sample to process, including the user query, contexts
                and model outputs as fields.
            question_field: str
                The field name containing the user query.
            answer_fields: str
                The field name containing the evaluated model output.
            base_field_data: dict[FieldId, BaseFieldData]
                The processed field data.

        Returns:
            dict:
                The unchanged sample dictionary.
        """
        return sample

    @override
    def __call__(
        self,
        sample: dict,
        question_field: FieldId,
        answer_field: FieldId,
        base_field_data: dict[FieldId, BaseFieldData],
    ) -> tuple[dict, dict[str, Context], dict[str, Atom], list[Relation]]:
        """
        Runs the fact graph miner on a single sample and model output (answer).

        Args:
            sample: dict
                The sample to process, including the user query, contexts
                and model outputs as fields.
            question_field: str
                The field name containing the user query.
            answer_field: str
                The field name containing the evaluated model output.
            base_field_data: dict[str, BaseFieldData]
                The processed field data.

        Returns:
            tuple[dict, dict[str, Context], dict[str, Atom], list[Relation]]
                A tuple containing the sample dictionary updated with any additional
                data from graph mining, a dictionary of contexts, a dictionary of atoms
                and a list of relations to be used for constructing a fact graph.
        """
        user_query = sample[question_field]
        contexts = {
            fid: f["summary"]
            for fid, f in base_field_data.items()
            if f["field_type"] == "context"
        }
        answer = base_field_data[answer_field]["summary"]
        sample = deepcopy(sample)

        covered_contexts, uncovered_contexts = self.coverage_evaluator.run(
            query=user_query, background_texts=contexts, answer=answer
        )
        sample["covered_coverage_results"] = covered_contexts
        sample["uncovered_coverage_results"] = uncovered_contexts

        aid_counter = 0
        cid_counter = 0
        context_dict: dict[str, Context] = {}
        atoms_dict: dict[str, Atom] = {}
        relations: list[Relation] = []
        for coverage_result in covered_contexts + uncovered_contexts:
            atom = None
            if coverage_result["is_covered"]:
                # The contexts in this cluster are covered, so they should be
                # linked with the associated atom.
                aid = f"a{aid_counter}"
                aid_counter += 1
                atom = Atom(aid, text=coverage_result["statement"])
                atoms_dict[aid] = atom

            cluster_contexts: list[Context] = []
            for source_id in coverage_result["context_ids"]:
                # We create a separate context for each source ID, even though
                # they share the same text. This makes the resulting fact graph
                # closer to the graphs extracted by the other miners.
                cid = f"c{cid_counter}_{source_id}"
                cid_counter += 1
                context = Context(
                    cid,
                    atom=None,
                    text=coverage_result["statement"],
                    metadata={"source_id": source_id},
                )
                context_dict[cid] = context
                cluster_contexts.append(context)

                if coverage_result["is_covered"]:
                    assert atom is not None, "Got unexpected None atom"
                    relations.append(
                        Relation(
                            source=atom,
                            target=context,
                            type="equivalence",
                            probability=1.0,
                            link="context_atom",
                        )
                    )

            # All contexts in a cluster are equivalent
            for i, c1 in enumerate(cluster_contexts):
                for j, c2 in enumerate(cluster_contexts):
                    if i >= j:
                        continue
                    relations.append(
                        Relation(
                            source=c1,
                            target=c2,
                            type="equivalence",
                            probability=1.0,
                            link="context_context",
                        )
                    )

        for atom in atoms_dict.values():
            atom.contexts = context_dict

        return sample, context_dict, atoms_dict, relations


class NliFactGraphMiner(FactGraphMiner):
    """
    A miner constructing fact graph components from unstructured contexts
    and model outputs using NLI.
    """

    def __init__(
        self,
        relevance_threshold: float,
        atom_extractor: AtomExtractor,
        atom_reviser: BatchAtomReviser,
        relevance_estimator: RelevanceEstimator,
        nli_extractor: NLIExtractor,
        max_batch_nli_relations: int = 1,
        max_atomization_batch_workers: int = 16,
    ):
        """
        Initializes the NliFactGraphMiner.

        Args:
            relevance_threshold: float
                The threshold for considering an atomic statement to be relevant to
                the original user query. See prompts in RelevanceEstimator for
                a relevance scoring rubric.
            atom_extractor: AtomExtractor
                The AtomExtractor instance to be used for extracting the atoms from
                contexts and the evaluated text.
            atom_reviser: BatchAtomReviser
                The BatchAtomReviser instance to be used for decontextualizing atoms.
            relevance_estimator: RelevanceEstimator
                The RelevanceEstimator instance to be used for estimating
                the relevance of atomic statements.
            nli_extractor: NLIExtractor
                The NLIExtractor instance to be used for extracting NLI relations
                between atomic statements.
            max_batch_nli_relations: int
                The maximum nubmber of NLI relations to determine in a single
                batch when using JsonNliExtractor. Defaults to 1.
            max_atomization_batch_workers: int
                The maximum number of threads for perofmring parallel atomization.
        """
        self.relevance_threshold = relevance_threshold
        self.atom_extractor = atom_extractor
        self.atom_reviser = atom_reviser
        self.relevance_estimator = relevance_estimator
        self.nli_extractor = nli_extractor
        self.max_batch_nli_relations = max_batch_nli_relations
        self.max_atomization_batch_workers = max_atomization_batch_workers

    def _extract_atoms(self, user_query: str, field: BaseFieldData) -> list[str]:
        """
        Extracts atomic statements relevant to a given user query
        from the provided dataset field.

        Args:
            user_query: str
                The user query.
            field: BaseFieldData
                The dataset field from which to extract atomic statements.

        Returns:
            list[str]:
                A list of atomic statements.
        """
        extracted_atoms = self.atom_extractor.run(field["summary"])
        atomic_facts = [a["atom"] for a in extracted_atoms["all_facts"]]

        if atomic_facts:
            revised_atoms = self.atom_reviser.run(field["summary"], atomic_facts)
        else:
            revised_atoms: list[str] = []

        if field["field_type"] == "context":
            relevance_results = self.relevance_estimator.run(user_query, revised_atoms)
            revised_atoms = [
                a
                for a, relevance_score in relevance_results
                if relevance_score >= self.relevance_threshold
            ]

        return revised_atoms

    @override
    def process_sample(
        self,
        sample: dict,
        question_field: FieldId,
        base_field_data: dict[FieldId, BaseFieldData],
    ) -> dict:
        """
        Processes sample for use with the given fact graph miner and
        returns the updated sample dictionary.

        Args:
            sample: dict
                The sample to process, including the user query, contexts
                and model outputs as fields.
            question_field: str
                The field name containing the user query.
            answer_fields: str
                The field name containing the evaluated model output.
            base_field_data: dict[FieldId, BaseFieldData]
                The processed field data.

        Returns:
            dict:
                The updated sample dictionary.
        """
        user_query = sample[question_field]
        sample = deepcopy(sample)
        atomized_field_data = deepcopy(base_field_data)

        # We extract the atoms for all fields here so that they can be reused
        # when evaluating comprehensiveness for each model output/answer
        with ThreadPoolExecutor(
            max_workers=self.max_atomization_batch_workers
        ) as executor:
            all_revised_atoms = list(
                tqdm(
                    executor.map(
                        lambda f: self._extract_atoms(user_query, f),
                        atomized_field_data.values(),
                    ),
                    total=len(atomized_field_data),
                )
            )
        for field, revised_atoms in zip(
            atomized_field_data.values(), all_revised_atoms, strict=True
        ):
            field = cast(AtomizedFieldData, field)
            field["atoms"] = revised_atoms

            print(f"[Relevant atoms for field {field['id']}]")
            for a in revised_atoms:
                print(f"— {a}")

        sample["atomized_field_data"] = atomized_field_data
        return sample

    @override
    def __call__(
        self,
        sample: dict,
        question_field: str,
        answer_field: str,
        base_field_data: dict[str, BaseFieldData],
    ) -> tuple[dict, dict[str, Context], dict[str, Atom], list[Relation]]:
        """
        Runs the fact graph miner on a single sample and model output (answer).

        Args:
            sample: dict
                The sample to process, including the user query, contexts
                and model outputs as fields.
            question_field: str
                The field name containing the user query.
            answer_field: str
                The field name containing the evaluated model output.
            base_field_data: dict[str, BaseFieldData]
                The processed field data.

        Returns:
            tuple[dict, dict[str, Context], dict[str, Atom], list[Relation]]
                A tuple containing the sample dictionary updated with any additional
                data from graph mining, a dictionary of contexts, a dictionary of atoms
                and a list of relations to be used for constructing a fact graph.
        """
        atomized_field_data = cast(
            dict[FieldId, AtomizedFieldData], sample["atomized_field_data"]
        )

        atoms_dict: dict[str, Atom] = {
            f"a{i}_{answer_field}": Atom(
                id=f"a{i}_{answer_field}",
                text=atom_text,
                metadata={"source_id": answer_field},
            )
            for i, atom_text in enumerate(atomized_field_data[answer_field]["atoms"])
        }
        context_counter = 0
        context_dict: dict[str, Context] = {}
        for fid, field_data in atomized_field_data.items():
            if field_data["field_type"] == "context":
                for atom in field_data["atoms"]:
                    cid = f"c{context_counter}_{fid}"
                    context_dict[cid] = Context(
                        id=cid, atom=None, text=atom, metadata={"source_id": fid}
                    )
                    context_counter += 1
        for atom in atoms_dict.values():
            atom.contexts = context_dict

        print("[Relevant context atoms]")
        for cid, context_atom in context_dict.items():
            print(f"{cid}: {context_atom.text}")
        print(f"[Answer {answer_field} atoms]")
        for aid, answer_atom in atoms_dict.items():
            print(f"{aid}: {answer_atom.text}")

        answer_relations = build_relations(
            atoms_dict,
            context_dict,
            nli_extractor=self.nli_extractor,
            contexts_per_atom_only=False,
            rel_atom_context=True,
            rel_context_atom=True,
            rel_context_context=True,
            text_only=True,
        )

        return sample, context_dict, atoms_dict, answer_relations


class QagFactGraphMiner(FactGraphMiner):
    """
    A miner constructing fact graph components from unstructured contexts
    and model outputs using question answering.
    """

    def __init__(
        self,
        relevance_threshold: float,
        qag_processor: QagProcessor,
        confidence_threshold: float = 2.0,
        batch_size: int = 64,
    ):
        """
        Initializes the QagFactGraphMiner.

        Args:
            relevance_threshold: float
                The threshold for considering a question to be relevant to the original
                user query. See prompts in QagProcessor for a relevance scoring rubric.
            qag_processor: QagProcessor
                The QagProcessor instance to be used for extracting question and answers
                from contexts and the evaluated text.
            confidence_threshold: float
                The threshold for considering a question answer to be valid based on the
                provided context or model output.
            merlin_path: str | None
                The path to the merlin binary. If None, defaults to retrieving the
                path from the MERLIN_PATH environment variable.
            batch_size: int
                The maximum concurrent requests in a single batch for LLM calls.
        """
        self.relevance_threshold = relevance_threshold
        self.qag_processor = qag_processor
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size

    def _extract_raw_questions(
        self, user_query: str, base_field_data: dict[FieldId, BaseFieldData]
    ) -> dict[FieldId, QagFieldData]:
        """
        (Optionally) summarizes the contexts and extracts the raw, factual
        questions from each context/answer field.

        Args:
            user_query: str
                The original user query associated with the evaluated model
                output.
            base_field_data: dict[str, BaseQagFieldData]
                The processed field data generated by _process_sample_fields.

        Returns:
            dict[str, QagFieldData]
                The field data enriched with the extracted questions.
        """
        question_extraction_start = time.time()
        print("[Extracting raw context/answer questions...]")
        fields_for_extraction = {
            fid: field_data["summary"]
            for fid, field_data in base_field_data.items()
            if field_data["summary"].lower() != "none"
        }
        fids, background_texts = list(zip(*fields_for_extraction.items()))
        extracted_questions = dict(
            zip(
                fids,
                self.qag_processor.extract_all_questions(
                    user_query, list(background_texts), batch_size=self.batch_size
                ),
            )
        )
        for fid, field in base_field_data.items():
            questions = extracted_questions.get(fid, [])
            field = cast(QagFieldData, field)
            field["questions"] = questions
            field["answers"] = {}
        qa_field_data = cast(dict[FieldId, QagFieldData], base_field_data)
        print(
            f"Question extraction done in {(time.time() - question_extraction_start):.2f} seconds"
        )
        print()

        print("[Raw questions]")
        for fid, field in qa_field_data.items():
            print(f"[{fid}]")
            for question in field["questions"]:
                print(f"— {question}")

        return qa_field_data

    def _refine_questions(
        self,
        user_query: str,
        answer_field: str,
        qa_field_data: dict[FieldId, QagFieldData],
    ) -> dict[str, QuestionData]:
        """
        Refines the questions extracted in the previous step by removing duplicates and
        estimating their relevance to the original user query.

        Args:
            user_query: str
                The original user query associated with the evaluated model
                output.
            answer_field: str
                The model output field for which to refine the questions. If there
                are multiple output fields, the method should be called for each
                of these individually.
            qa_field_data: dict[FieldId, QagFieldData]
                A dictionary mapping field ids to their field data, including
                the questions extracted in the previous pipeline stage.

        Returns:
            dict[str, QuestionData]
                A dictionary mapping question strings to their structured data,
                including the relevance scores.
        """
        question_refinement_start = time.time()
        print()
        print(f"[Refining questions for answer {answer_field}]")
        all_questions = [
            q
            for field in qa_field_data.values()
            for q in field["questions"]
            if field["field_type"] == "context" or field["id"] == answer_field
        ]
        all_questions_dict = self.qag_processor.refine_questions(
            user_query, all_questions
        )
        sorted_questions = sorted(
            all_questions_dict.items(), key=lambda q: q[1].relevance, reverse=True
        )
        for q, qd in sorted_questions:
            print(f"— {q} [Relevance: {qd.relevance:.2f}]")
        print(f"Using questions with relevance >= {self.relevance_threshold}")
        print(
            f"Question refinement done in {(time.time() - question_refinement_start):.2f} seconds"
        )
        print()
        all_questions_dict = {
            q: qd
            for q, qd in sorted_questions
            if qd.relevance >= self.relevance_threshold
        }
        return all_questions_dict

    def _extract_answers(
        self,
        answer_field: str,
        qa_field_data: dict[FieldId, QagFieldData],
        all_questions_dict: dict[Question, QuestionData],
    ) -> dict[Question, QuestionData]:
        """
        Extract answers to the generated questions from the contexts and
        the evaluated model output.

        Args:
            answer_field: str
                The model output field for which to refine the questions. If there
                are multiple answer fields, the method should be called for each
                of these individually.
            qa_field_data: dict[FieldId, QagFieldData]
                A dictionary mapping field ids to their field data, including
                the questions extracted in the previous pipeline stage.
            all_questions_dict: dict[str, QuestionData]
                A dictionary with the refined questions from the previous
                pipeline stage.

        Returns:
            dict[Question, QuestionData]
                A dictionary mapping questions to their structured data,
                including the relevance scores and extracted answers.
        """
        answer_extraction_start = time.time()
        print(f"[Extracting answers for answer {answer_field}]")

        # We want to extract answers in a batch for all the contexts
        # and model outputs/answers, so we construct a dictionary with
        # the extraction arguments for every answer and pivot it before
        # passing it to extract_all_answers
        field_args_for_extraction: dict[FieldId, dict[str, str]] = {}
        answer_metadata = qa_field_data[answer_field]
        current_answer = answer_metadata["summary"]
        field_args_for_extraction[answer_field] = {
            "background_texts": current_answer,
            "source_ids": answer_field,
            "source_types": "answer",
        }
        for fid, field in qa_field_data.items():
            if field["field_type"] == "context":
                context_summary = field["summary"]
                if context_summary.strip().lower() == "none":
                    continue
                field_args_for_extraction[fid] = {
                    "background_texts": context_summary,
                    "source_ids": fid,
                    "source_types": "context",
                }
        all_answers = self.qag_processor.extract_all_answers(
            questions=all_questions_dict,
            batch_size=self.batch_size,
            # This pivots the arguments dictionary as outlined above
            **{
                arg_name: [
                    field_args[arg_name]
                    for field_args in field_args_for_extraction.values()
                ]
                for arg_name in next(iter(field_args_for_extraction.values()))
            },
        )
        print(
            f"Answer extraction done in {(time.time() - answer_extraction_start):.2f} seconds"
        )
        print()

        for fid, answers in zip(field_args_for_extraction.keys(), all_answers):
            field_data = qa_field_data[fid]
            field_data["answers"][answer_field] = answers
            print(f"[Q&As for {fid}]")
            print(field_data["summary"])
            print_answers(answers)
        print()

        # We merge the question and answer data for all the questions
        # into a single dictionary — the Answer objects in
        # QuestionData still keep track of the source of each answer,
        # so we don't lose any information.
        all_answers_dict = self.qag_processor.merge_questions(
            all_answers,
            confidence_threshold=self.confidence_threshold,
        )
        return all_answers_dict

    def _extract_answer_relations(
        self,
        all_answers_dict: dict[Question, QuestionData],
    ) -> tuple[dict[str, Context], dict[str, Atom], list[Relation]]:
        """
        Constructs the context and atom entities based on the extracted answers
        and determines the relations between them.

        Args:
            all_answers_dict: dict[Question, QuestionData]
                A dictionary with the questions and their structured data,
                including the answers extracted in the previous pipeline stage.

        Returns:
            tuple[dict[str, Context], dict[str, Atom], list[Relation]]
                A tuple containing the built contexts, atoms and relations.
        """
        relation_extraction_start = time.time()
        print("[Extracting answer relations]")

        # The answer frozensets always contain two elements and allow looking
        # up relations without regard to order.
        all_answer_relations: dict[
            Question, tuple[QuestionData, dict[frozenset[Answer], AnswerRelation]]
        ] = {}
        comparison_results = self.qag_processor.compare_all_answers(
            list(all_answers_dict.values()), batch_size=self.batch_size
        )
        for q, answer_relations in zip(all_answers_dict.keys(), comparison_results):
            all_answer_relations[q] = (all_answers_dict[q], answer_relations)

        atom_counter = 0
        context_counter = 0
        answer_atom_contexts: dict[tuple[Question, Answer], Atom | Context] = {}
        relations: list[Relation] = []
        for q, (q_data, answer_relations) in all_answer_relations.items():
            for answer in q_data.answers:
                if answer.answer.strip().lower() == "unknown":
                    continue

                if answer.source_type == "answer":
                    answer_atom_contexts[(q, answer)] = Atom(
                        id=f"a{atom_counter}_{answer.source_id}",
                        text=f"Q: {q}, A: {answer.answer}",
                        metadata={"answer": answer, "source_id": answer.source_id},
                    )
                    atom_counter += 1
                elif answer.source_type == "context":
                    answer_atom_contexts[(q, answer)] = Context(
                        id=f"c{context_counter}_{answer.source_id}",
                        atom=None,
                        text=f"Q: {q}, A: {answer.answer}",
                        metadata={"answer": answer, "source_id": answer.source_id},
                    )
                    context_counter += 1

            # This converts relations from the AnswerRelation model to
            # standard Relation objects used by FactReasoner.
            for relation in answer_relations.values():
                if relation.label == "neutral":
                    continue

                source = answer_atom_contexts[(q, relation.fst)]
                target = answer_atom_contexts[(q, relation.snd)]
                if relation.label == "second implies first":
                    source, target = target, source

                if isinstance(source, Context) and isinstance(target, Atom):
                    link = "context_atom"
                elif isinstance(source, Atom) and isinstance(target, Atom):
                    link = "atom_atom"
                elif isinstance(source, Context) and isinstance(target, Context):
                    link = "context_context"
                elif isinstance(source, Atom) and isinstance(target, Context):
                    if relation.label in ["equivalent", "contradictory"]:
                        # Flip the relation to standardize the order
                        source, target = target, source
                        link = "context_atom"
                    else:
                        link = "atom_context"
                else:
                    raise ValueError(f"Unexpected relation pair: {source} - {target}")

                relation_types = {
                    "equivalent": "equivalence",
                    "first implies second": "entailment",
                    "second implies first": "entailment",
                    "contradictory": "contradiction",
                }

                relation = Relation(
                    source=source,
                    target=target,
                    type=relation_types[relation.label],
                    probability=relation.probability,
                    link=link,
                    reasoning=relation.reasoning,
                )
                relations.append(relation)
        print(
            f"Relation extraction done in {(time.time() - relation_extraction_start):.2f} seconds"
        )
        print()

        context_dict = {
            c.id: c for c in answer_atom_contexts.values() if isinstance(c, Context)
        }
        atoms_dict = {
            a.id: Atom(a.id, a.text, contexts=context_dict)
            for a in answer_atom_contexts.values()
            if isinstance(a, Atom)
        }

        print("[Contexts]")
        for cid, context_atom in context_dict.items():
            print(f"{cid}: {context_atom.text}")
        print("[Atoms]")
        for aid, answer_atom in atoms_dict.items():
            print(f"{aid}: {answer_atom.text}")
        print("[Relations]")
        for relation in relations:
            print(relation)
        print()

        return context_dict, atoms_dict, relations

    def process_sample(
        self,
        sample: dict,
        question_field: FieldId,
        base_field_data: dict[FieldId, BaseFieldData],
    ) -> dict:
        """
        Processes sample for use with the given fact graph miner and
        returns the updated sample dictionary.

        Args:
            sample: dict
                The sample to process, including the user query, contexts
                and model outputs as fields.
            question_field: str
                The field name containing the user query.
            answer_fields: str
                The field name containing the evaluated model output.
            base_field_data: dict[FieldId, BaseFieldData]
                The processed field data.

        Returns:
            dict:
                The updated sample dictionary.
        """
        user_query = sample[question_field]
        sample = deepcopy(sample)

        # Summarise contexts and extract raw questions covering the facts in
        # the contexts and the model answers.
        qa_field_data = self._extract_raw_questions(user_query, base_field_data)
        sample["qa_field_data"] = qa_field_data

        return sample

    @override
    def __call__(
        self,
        sample: dict,
        question_field: str,
        answer_field: str,
        base_field_data: dict[str, BaseFieldData],
    ) -> tuple[dict, dict[str, Context], dict[str, Atom], list[Relation]]:
        """
        Runs the fact graph miner on a single sample and model output (answer).

        Args:
            sample: dict
                The sample to process, including the user query, contexts
                and model outputs as fields.
            question_field: str
                The field name containing the user query.
            answer_field: str
                The field name containing the evaluated model output.
            base_field_data: dict[str, BaseFieldData]
                The processed field data.

        Returns:
            tuple[dict, dict[str, Context], dict[str, Atom], list[Relation]]
                A tuple containing the sample dictionary updated with any additional
                data from graph mining, a dictionary of contexts, a dictionary of atoms
                and a list of relations to be used for constructing a fact graph.
        """
        user_query = sample[question_field]
        sample = deepcopy(sample)

        # Retrieve the preprocessed Q&A field data
        qa_field_data = cast(dict[FieldId, QagFieldData], sample["qa_field_data"])

        # Refine the full set of questions for the given model answer
        all_questions_dict = self._refine_questions(
            user_query, answer_field, qa_field_data
        )

        # Generate answers for all the refined questions
        all_answers_dict = self._extract_answers(
            answer_field, qa_field_data, all_questions_dict
        )
        sample["all_answers_dict"] = all_answers_dict

        # Compare the generated answers, determine the relationships between them
        # and construct the associated context/atom objects
        context_dict, atoms_dict, relations = self._extract_answer_relations(
            all_answers_dict
        )

        return sample, context_dict, atoms_dict, relations


class ComprehensivenessPipeline:
    """
    Pipeline for evaluating the comprehensiveness of model outputs.
    """

    def __init__(
        self,
        fact_graph_miner: FactGraphMiner,
        query_context_summarizer: QueryContextSummarizer | None = None,
        output_generator: OutputGenerator | None = None,
        merlin_path: str | None = None,
        batch_size: int = 64,
    ):
        """
        Initializes the QagComprehensivenessPipeline.

        Args:
            fact_graph_miner: FactGraphMiner
                The FactGraphMiner instance to use for mining the components of the
                fact graph.
            query_context_summarizer: QueryContextSummarizer | None
                The QueryContextSummarizer instance to use for summarising the contexts
                before mining questions and answers from them. If not provided, the full
                context texts are used.
            output_generator: OutputGenerator | None
                The OutputGenerator instance to use for generating model outputs when
                evaluating the comprehensiveness of the answers produced by a specific
                LLM.
            merlin_path: str | None
                The path to the merlin binary. If None, defaults to retrieving the
                path from the MERLIN_PATH environment variable.
            batch_size: int
                The maximum concurrent requests in a single batch for LLM calls.
        """
        self.fact_graph_miner = fact_graph_miner
        if query_context_summarizer is None:
            print(
                "[ComprehensivenessPipeline] Running without query context summarizer."
            )
        self.query_context_summarizer = query_context_summarizer
        self.output_generator = output_generator
        if merlin_path is not None:
            self.merlin_path = merlin_path
        else:
            load_dotenv()
            self.merlin_path = os.getenv("MERLIN_PATH")
        if self.merlin_path is None:
            raise ValueError("Merlin path cannot be none.")
        print(f"[ComprehensivnessPipeline] Using merlin at: {self.merlin_path}")
        self.batch_size = batch_size

    def _process_sample_fields(
        self,
        sample: dict,
        user_query: str,
        context_fields: tuple[FieldId, ...] = ("context",),
        answer_fields: tuple[FieldId, ...] = ("answer",),
        metadata_fields: dict[FieldId, dict[FieldId, FieldId]] | None = None,
    ) -> dict[FieldId, BaseFieldData]:
        """
        Preprocesses the sample fields and initialises their metadata.

        Note that additional preprocessing might be performed by
        the used fact graph miner and its process_sample method.

        Args:
            sample: dict
                The sample to process.
            context_fields: tuple[FieldId, ...]
                The fields of the sample to consider as contexts. Defaults to ("context",).
            answer_fields: tuple[FieldId, ...]
                The fields of the sample to consider as evaluated model output.
                Defaults to ("answer",).
            metadata_fields: dict[FieldId, dict[FieldId, FieldId]] | None
                A nested dictionary mapping context/answer field names to mappings
                from the corresponding metadata field keys to the alias keys under which
                they should be stored in the field metadata dictionary.

        Returns:
            dict[FieldId, BaseQagFieldData]
                A dictionary mapping field ids to their processed field data.
        """
        if metadata_fields is None:
            metadata_fields = {}

        def process_field(
            fid: FieldId, field_type: Literal["context", "answer"]
        ) -> BaseFieldData:
            return {
                "id": fid,
                "field_type": field_type,
                "original_content": sample[fid],
                "cleaned_content": sample[fid].strip(),
                "summary": sample[fid].strip(),
                "metadata": {
                    f_alias: sample[mf] for mf, f_alias in metadata_fields[fid].items()
                }
                if fid in metadata_fields
                else {},
            }

        base_field_data: dict[FieldId, BaseFieldData] = {}
        for fid in context_fields:
            base_field_data[fid] = process_field(fid, field_type="context")
        for fid in answer_fields:
            base_field_data[fid] = process_field(fid, field_type="answer")

        if self.output_generator is not None:
            # Generate the output to be evaluated
            model_output = self.output_generator.run(
                user_query,
                background_texts=[
                    base_field_data[fid]["cleaned_content"] for fid in context_fields
                ],
            )
            base_field_data[MODEL_OUTPUT_FIELD_ID] = {
                "id": MODEL_OUTPUT_FIELD_ID,
                "field_type": "answer",
                "original_content": model_output,
                "cleaned_content": model_output.strip(),
                "summary": model_output.strip(),
                "metadata": {},
            }

        print("[Sample info]")
        for fid, field in base_field_data.items():
            print(f"[{fid}]")
            print(field["cleaned_content"])
        print()

        summarisation_start = time.time()
        if self.query_context_summarizer is not None:
            print("[Summarizing context fields...]")
            fields_to_summarize = {
                fid: field["cleaned_content"]
                for fid, field in base_field_data.items()
                if field["field_type"] == "context"
            }
            fids, background_texts = list(zip(*fields_to_summarize.items()))
            summaries_list = self.query_context_summarizer.runall(
                user_query, list(background_texts), batch_size=self.batch_size
            )
            for fid, summary in zip(fids, summaries_list):
                base_field_data[fid]["summary"] = summary.strip()
        print(
            f"Summarization done in {(time.time() - summarisation_start):.2f} seconds"
        )
        print()

        print("[Summarized contexts]")
        for fid, field in base_field_data.items():
            if field["field_type"] == "context":
                print(f"[{fid}]")
                print(field["summary"])
        print()

        return base_field_data

    def _evaluate_comprehensiveness(
        self,
        context_dict: dict[str, Context],
        atoms_dict: dict[str, Atom],
        relations: list[Relation],
        current_result: dict[str, Any],
    ):
        """
        Constructs a FactGraph and computes the final comprehensiveness/factuality scores.

        Args:
            context_dict: dict[str, Context]
                A dictionary mapping context ids to their corresponding Context objects.
            atoms_dict: dict[str, Atom]
                A dictionary mapping atom ids to their corresponding Atom objects.
            relations: list[Relation]
                A list of Relation objects specifying the relations between context
                and atom nodes.
            current_result: dict[str, Any]
                A dictionary that should be used for storing the evaluation results.
                Note that the dictionary will be directly modified by this method.
        """
        print("[Evaluating comprehensiveness]")
        fact_graph = FactGraph(
            atoms=list(atoms_dict.values()),
            contexts=list(context_dict.values()),
            relations=relations,
        )
        markov_network = build_markov_network(fact_graph=fact_graph, use_priors=True)
        marginals = run_merlin(
            variable_names=list(atoms_dict.keys()),
            markov_network=markov_network,
            merlin_path=self.merlin_path,  # type: ignore
        )
        current_result["answer_relations"] = relations
        current_result["fact_graph"] = fact_graph

        scores = {
            m["variable"]: (m["probabilities"][0], m["probabilities"][1])
            for m in marginals
        }
        current_result["probabilistic_results"] = compute_factuality_results(
            scores=scores,
            fact_graph=fact_graph,
            atoms=atoms_dict,
            contexts=context_dict,
        )

        current_result["argumentative_results"] = {}
        for semantics in ["dfquad", "qe"]:
            try:
                af = ArgumentationFramework(fact_graph)
                strengths = af.evaluate_strengths(semantics="dfquad")
                scores = {v: (1 - s, s) for v, s in strengths.items()}
                current_result["argumentative_results"][semantics] = (
                    compute_factuality_results(
                        scores=scores,
                        fact_graph=fact_graph,
                        atoms=atoms_dict,
                        contexts=context_dict,
                    )
                )
            except Exception as e:
                print(f"Exception when computing results for semantics {semantics}:")
                print(e)

        covered_contexts = fact_graph.covered_context_ids
        uncovered_contexts = fact_graph.uncovered_context_ids
        context_equivalence_clusters = fact_graph.context_equivalence_clusters
        uncovered_context_basis = fact_graph.uncovered_context_basis
        assert set(covered_contexts + uncovered_contexts) == set(
            context_dict.keys()
        ), f"Unexpected covered/uncovered context results ({
            set(covered_contexts + uncovered_contexts)
        } != {set(context_dict.keys())})"
        assert set(uncovered_context_basis.keys()).issubset(
            uncovered_contexts
        ), f"Uncovered context basis is not a subset of uncovered contexts ({
            set(uncovered_context_basis.keys())
        } is not a subset of {set(uncovered_contexts)})."
        current_result["covered_contexts"] = covered_contexts
        current_result["uncovered_contexts"] = uncovered_contexts
        current_result["context_equivalence_clusters"] = context_equivalence_clusters
        current_result["uncovered_context_basis"] = uncovered_context_basis
        cluster_prototypes = set(context_equivalence_clusters.keys())
        covered_clusters = sorted(
            list(set(covered_contexts).intersection(cluster_prototypes))
        )
        uncovered_clusters = sorted(
            list(set(uncovered_contexts).intersection(cluster_prototypes))
        )
        comprehensiveness = len(covered_clusters) / max(
            len(covered_clusters) + len(uncovered_clusters), 1
        )
        current_result["comprehensiveness_score"] = comprehensiveness
        print(f"Covered contexts: {covered_contexts}")
        print(f"Uncovered contexts: {uncovered_contexts}")
        print(f"Covered cluster prototypes: {covered_clusters}")
        print(f"Uncovered cluster prototypes: {uncovered_clusters}")
        print()

        print("Context coverage details:")
        for pid, context_atom in context_dict.items():
            if pid not in (covered_clusters + uncovered_clusters):
                continue
            print(
                f"{pid}: {context_atom.text} {'✅' if pid in covered_contexts else '❌'}"
            )
            equivalent_contexts = list(context_equivalence_clusters[pid])
            equivalent_contexts.remove(pid)
            for cid in equivalent_contexts:
                print(
                    f"  - {cid}: {context_dict[cid].text} {'✅' if cid in covered_contexts else '❌'}"
                )
        print()
        print("Uncovered context basis details:")
        for bid, covered_ids in uncovered_context_basis.items():
            print(f"{bid}: {context_dict[bid].text} ❌")
            covered_ids = list(covered_ids)
            covered_ids.remove(bid)
            for pid in covered_ids:
                print(f"  - {pid}: {context_dict[pid].text} ❌")
        print()
        print(f"Comprehensiveness score: {comprehensiveness:.2f}")

    def run_sample(
        self,
        sample: dict,
        question_field: str = "question",
        context_fields: tuple[FieldId, ...] = ("context",),
        answer_fields: tuple[FieldId, ...] = ("answer",),
        metadata_fields: dict[FieldId, dict[FieldId, FieldId]] | None = None,
        comprehensiveness_evaluation_fun: Callable[
            [ComprehensivenessResult], ComprehensivenessResult
        ] = lambda r: r,
    ) -> list[ComprehensivenessResult]:
        """
        Executes the comprehensiveness pipeline on a single sample.

        Args:
            sample: dict
                The sample to process, including the user query, contexts
                and model outputs as fields.
            question_field: str
                The field name containing the user query.
            context_fields: tuple[FieldId, ...]
                The fields of the sample that contain the contexts.
            answer_fields: tuple[FieldId, ...]
                The fields of the sample that contain the model outputs.
            metadata_fields: dict[FieldId, dict[FieldId, FieldId]] | None
                A nested dictionary mapping context/answer field names to mappings
                from the corresponding metadata field keys to the alias keys under
                which they should be stored in the field metadata dictionary.
            comprehensiveness_evaluation_fun: Callable[[ComprehensivenessResult], ComprehensivenessResult]
                A function for performing dataset-specific evaluation
                of the comprehensiveness score and returns an updated
                comprehensiveness result.

        Returns:
            list[ComprehensivenessResult]
                A list of comprehensiveness result objects (one for each
                model output field).
        """
        user_query = sample[question_field]

        print("[QagComprehensivenessPipeline] Running pipeline on sample.")
        print()
        print("[User query]")
        print(user_query)
        print()

        # Preprocess the input fields
        base_field_data = self._process_sample_fields(
            sample,
            user_query=user_query,
            context_fields=context_fields,
            answer_fields=answer_fields,
            metadata_fields=metadata_fields,
        )
        if MODEL_OUTPUT_FIELD_ID in base_field_data:
            answer_fields = answer_fields + (MODEL_OUTPUT_FIELD_ID,)

        # Apply FactGraphMiner-specific preprocessing
        processed_sample = self.fact_graph_miner.process_sample(
            sample=sample,
            question_field=question_field,
            base_field_data=base_field_data,
        )

        results: list[ComprehensivenessResult] = []
        for answer_field in answer_fields:
            # Mine components needed for constructing the fact graph using
            # the desired strategy.
            updated_sample, context_dict, atoms_dict, relations = self.fact_graph_miner(
                sample=processed_sample,
                question_field=question_field,
                answer_field=answer_field,
                base_field_data=base_field_data,
            )

            current_result = {
                "sample": updated_sample,
                "user_query": user_query,
                "answer_field": answer_field,
                "field_data": base_field_data,
                "contexts": context_dict,
                "atoms": atoms_dict,
                "relations": relations,
            }

            # Construct FactReasoner fact graph and compute factuality metrics
            # including comprehensiveness.
            self._evaluate_comprehensiveness(
                context_dict, atoms_dict, relations, current_result
            )
            current_result["comprehensiveness_eval_main_score"] = None
            current_result = cast(ComprehensivenessResult, current_result)

            # Evaluate the comprehensiveness result if applicable.
            current_result = comprehensiveness_evaluation_fun(current_result)

            results.append(current_result)

        return results


class DatasetConfig(TypedDict):
    """
    Dataset-specific configuration for evaluating comprehensiveness.
    See run_sample docstring for the description of the individual
    keys.
    """

    question_field: str
    context_fields: tuple[FieldId, ...]
    answer_fields: tuple[FieldId, ...]
    metadata_fields: NotRequired[dict[FieldId, dict[FieldId, FieldId]]]
    comprehensiveness_evaluation_fun: NotRequired[
        Callable[[ComprehensivenessResult], ComprehensivenessResult]
    ]


class DataLoader(Protocol):
    """
    A protocol for loading a specific dataset.
    """

    def __call__(self) -> tuple[list[dict], DatasetConfig]: ...


def load_wikicontradict() -> tuple[list[dict], DatasetConfig]:
    """
    Loads the WikiContradict dataset for comprehensiveness evaluation.

    Returns:
        tuple[list[dict], DatasetConfig]:
            A tuple with the dataset samples and the dataset config object.
    """
    load_dotenv()
    DATA_PATH = os.environ.get("DATA_PATH")
    wiki_contradict = pd.read_json(
        f"{DATA_PATH}/wiki_contradict_simple_humaneval.jsonl", lines=True
    ).to_dict("records")

    def evaluate_comprehensiveness(
        result: ComprehensivenessResult,
    ) -> ComprehensivenessResult:
        """
        Evaluates comprehensiveness scores using human labels from
        WikiContradict.

        Args:
            result: ComprehensivenessResult
                The comprehensiveness result to evaluate.

        Returns:
            ComprehensivenessResult:
                The updated comprehensiveness result with the evaluation
                results.
        """
        answer_field = result["answer_field"]
        comprehensiveness = result["comprehensiveness_score"]
        label = result["field_data"][answer_field]["metadata"]["label"]

        if comprehensiveness > (1 - 10e-6):
            std_pred = "Correct"
        elif comprehensiveness > 10e-6:
            std_pred = "Partially correct"
        else:
            std_pred = "Incorrect"

        MODERATE_LOW_THRESHOLD = 0.2
        MODERATE_HIGH_THRESHOLD = 0.8
        if comprehensiveness > MODERATE_HIGH_THRESHOLD:
            moderate_pred = "Correct"
        elif comprehensiveness > MODERATE_LOW_THRESHOLD:
            moderate_pred = "Partially correct"
        else:
            moderate_pred = "Incorrect"

        adjusted_label = label
        if label == "Correct" and answer_field in ["answer1", "answer2"]:
            adjusted_label = "Partially correct"
        score_satisfies_criteria = std_pred == adjusted_label
        score_satisfies_moderate_criteria = moderate_pred == adjusted_label
        print(f"Label is: {label}")
        print(f"Adjusted label is: {adjusted_label}")
        print(f"Score satisfies criteria: {score_satisfies_criteria}")
        print(f"Score satisfies moderate criteria: {score_satisfies_moderate_criteria}")

        result["comprehensiveness_eval_main_score"] = score_satisfies_criteria
        result["comprehensiveness_eval_results"] = {
            "score_satisfies_criteria": score_satisfies_criteria,
            "score_satisfies_moderate_criteria": score_satisfies_moderate_criteria,
            "comprehensiveness_pred": std_pred,
            "moderate_pred": moderate_pred,
            "label": label,
            "adjusted_label": adjusted_label,
        }
        return result

    config: DatasetConfig = {
        "question_field": "question",
        "context_fields": ("context1_self_contained", "context2_self_contained"),
        # answer0 — LLM answer without any context
        # answer1 — LLM answer based on the first context
        # answer2 — LLM answer based on the second context
        # answer3 — LLM answer based on both contexts
        # answer4 — LLM answer based on both contexts, with focus on conflicts
        "answer_fields": ("answer0", "answer1", "answer2", "answer3", "answer4"),
        "metadata_fields": {f"answer{i}": {f"label{i}": "label"} for i in range(5)},
        "comprehensiveness_evaluation_fun": evaluate_comprehensiveness,
    }

    return wiki_contradict, config


def load_conflict_bank() -> tuple[list[dict], DatasetConfig]:
    """
    Loads the ConflictBank dataset for comprehensiveness evaluation.

    Returns:
        tuple[list[dict], DatasetConfig]:
            A tuple with the dataset samples and the dataset config object.
    """
    dataset = datasets.load_dataset("Warrieryes/CB_qa_v2")
    if isinstance(dataset, datasets.DatasetDict):
        dataset = dataset["train"]
    assert isinstance(dataset, datasets.Dataset)
    dataset = dataset.shuffle(seed=42)
    conflict_bank: list[dict] = dataset.to_list()
    conflict_bank = conflict_bank[:500]

    def evaluate_comprehensiveness(
        result: ComprehensivenessResult,
    ) -> ComprehensivenessResult:
        """
        Evaluates comprehensiveness scores using the metadata from ConflictBank.

        Args:
            result: ComprehensivenessResult
                The comprehensiveness result to evaluate.

        Returns:
            ComprehensivenessResult:
                The updated comprehensiveness result with the evaluation
                results.
        """
        answer_field = result["answer_field"]
        covered_contexts = result["covered_contexts"]
        uncovered_contexts = result["uncovered_contexts"]

        covered_contexts_by_source: dict[str, int] = defaultdict(int)
        total_contexts_by_source: dict[str, int] = defaultdict(int)
        for context in result["contexts"].values():
            source_id = context.metadata["source_id"]
            if context.id in covered_contexts:
                covered_contexts_by_source[source_id] += 1
            elif context.id not in uncovered_contexts:
                raise ValueError(
                    f"Context {context.id} not in either covered or uncovered contexts."
                )
            total_contexts_by_source[source_id] += 1

        MODERATE_LOW_THRESHOLD = 0.2
        MODERATE_HIGH_THRESHOLD = 0.8
        default_claim_coverage = covered_contexts_by_source["default_evidence"] / (
            total_contexts_by_source["default_evidence"] + 1e-9
        )
        satisfies_lax_criteria = 0
        satisfies_moderate_criteria = 0
        satisfies_strict_criteria = 0
        if "conflict" in answer_field:
            # Expect conflict contexts to be covered and default context
            # to be uncovered.
            for source in total_contexts_by_source.keys():
                if "conflict" in source:
                    source_coverage = covered_contexts_by_source[source] / (
                        total_contexts_by_source[source] + 1e-9
                    )
                    if source_coverage > default_claim_coverage:
                        satisfies_lax_criteria += 1
                    if source_coverage > MODERATE_HIGH_THRESHOLD:
                        satisfies_moderate_criteria += 1
                    if math.isclose(source_coverage, 1.0, rel_tol=1e-6):
                        satisfies_strict_criteria += 1
                elif "default" in source:
                    if default_claim_coverage < MODERATE_LOW_THRESHOLD:
                        satisfies_moderate_criteria += 1
                    if math.isclose(default_claim_coverage, 0.0, rel_tol=1e-6):
                        satisfies_strict_criteria += 1
        elif "default" in answer_field:
            # Expect conflict context to be uncovered and default context
            # to be covered.
            for source in total_contexts_by_source.keys():
                if "conflict" in source:
                    source_coverage = covered_contexts_by_source[source] / (
                        total_contexts_by_source[source] + 1e-9
                    )
                    if source_coverage < default_claim_coverage:
                        satisfies_lax_criteria += 1
                    if source_coverage < MODERATE_LOW_THRESHOLD:
                        satisfies_moderate_criteria += 1
                    if math.isclose(source_coverage, 0.0, rel_tol=1e-6):
                        satisfies_strict_criteria += 1
                elif "default" in source:
                    if default_claim_coverage > MODERATE_HIGH_THRESHOLD:
                        satisfies_moderate_criteria += 1
                    if math.isclose(default_claim_coverage, 1.0, rel_tol=1e-6):
                        satisfies_strict_criteria += 1
        else:
            raise ValueError(f"Unexpected answer field {answer_field}")

        # We do (n - 1) comparisons for the lax score, as we compare all conflict
        # claims to the default claim.
        lax_score = satisfies_lax_criteria / (
            len(total_contexts_by_source.keys()) - 1 + 1e-9
        )
        moderate_score = satisfies_moderate_criteria / (
            len(total_contexts_by_source.keys()) + 1e-9
        )
        strict_score = satisfies_strict_criteria / (
            len(total_contexts_by_source.keys()) + 1e-9
        )

        result["comprehensiveness_eval_main_score"] = statistics.mean(
            [lax_score, moderate_score, strict_score]
        )
        result["comprehensiveness_eval_results"] = {
            "lax_score": lax_score,
            "moderate_score": moderate_score,
            "strict_score": strict_score,
        }

        print(f"Lax score: {lax_score}")
        print(f"Moderate score: {moderate_score}")
        print(f"Strict score: {strict_score}")

        return result

    config: DatasetConfig = {
        "question_field": "question",
        "context_fields": (
            "default_evidence",
            "misinformation_conflict_evidence",
            "temporal_conflict_evidence",
            "semantic_conflict_evidence",
        ),
        # Only include one counterfactual claim
        "answer_fields": ("default_claim", "misinformation_conflict_claim"),
        "comprehensiveness_evaluation_fun": evaluate_comprehensiveness,
    }

    return conflict_bank, config


def load_eli5(version: Literal["base", "v2"]) -> tuple[list[dict], DatasetConfig]:
    """
    Loads the ELI5 dataset for comprehensiveness evaluation.

    See `eli5_data.ipynb` for details of how this data was processed.

    Args:
        version: Literal["base", "v2]
            The version of the ELI5 dataset to load.

    Returns:
        tuple[list[dict], DatasetConfig]:
            A tuple with the dataset samples and the dataset config object.
    """
    load_dotenv()
    DATA_PATH = os.environ.get("DATA_PATH")
    with open(f"{DATA_PATH}/eli5-comprehensiveness.json", "r") as f:
        data: list[dict] = json.load(f)

    def evaluate_comprehensiveness(
        result: ComprehensivenessResult,
    ) -> ComprehensivenessResult:
        """
        Evaluates comprehensiveness of model outputs on ELI5.

        Args:
            result: ComprehensivenessResult
                The comprehensiveness result to evaluate.

        Returns:
            ComprehensivenessResult:
                The updated comprehensiveness result with the evaluation
                results.
        """
        # Simply return comprehensiveness as the main score
        comprehensiveness = result["comprehensiveness_score"]
        result["comprehensiveness_eval_main_score"] = comprehensiveness
        return result

    config: DatasetConfig = {
        "question_field": "query",
        "context_fields": tuple(
            f for f in data[0].keys() if f"{version}_context_" in f
        ),
        # Given that we only use ELI5 for evaluating models rather than
        # the comprehensiveness pipeline itself, we generate answers
        # on the fly.
        "answer_fields": (),
        "comprehensiveness_evaluation_fun": evaluate_comprehensiveness,
    }

    return data, config


DATA_LOADERS: dict[str, DataLoader] = {
    "wiki_contradict_humaneval": load_wikicontradict,
    "conflict_bank": load_conflict_bank,
    "eli5_base": lambda: load_eli5(version="base"),
    "eli5_v2": lambda: load_eli5(version="v2"),
}

if __name__ == "__main__":
    load_dotenv()
    RESULTS_PATH = os.environ["RESULTS_PATH"]
    RITS_API_KEY = os.environ["RITS_API_KEY"]

    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(
        logging.ERROR
    )
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    # NOTE: If you update these arguments, please also make sure to update
    #       orchestrate.py accordingly.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="ID of the dataset to use.",
        required=True,
        choices=DATA_LOADERS.keys(),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model to use for the comprehensiveness pipeline.",
        required=True,
    )
    parser.add_argument(
        "--evaluated_model_name",
        type=str,
        help="The name of the LLM that should be used for generating outputs to be evaluated for comprehensiveness.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        help="The version of the pipeline to use",
        required=True,
        choices=["qa", "nli", "e2e", "e2e-base"],
    )
    parser.add_argument(
        "--relevance_threshold",
        type=float,
        help="The relevance threshold to use for the experiment.",
        required=True,
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=2.0,
        help="The answer confidence threshold to use for the experiment.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="comprehensiveness_qa",
        help="Name of the experiment to use for the result file.",
    )
    parser.add_argument(
        "--use_default_prompt_template",
        default=False,
        action="store_true",
        help="Uses default prompt template for the chat completion API instead of injecting it manually.",
    )
    parser.add_argument(
        "--disable_tools",
        default=False,
        action="store_true",
        help="Disable tool calls for answer comparisons.",
    )
    args = parser.parse_args()

    dataset, dataset_config = DATA_LOADERS[args.dataset]()

    model_for_filename = regex.sub(r"[^\w\s-_]", "-", args.model_name)
    eval_model_for_filename = (
        ""
        if args.evaluated_model_name is None
        else "_" + regex.sub(r"[^\w\s-_]", "-", args.evaluated_model_name)
    )
    save_filename = f"{args.dataset}_{args.experiment_name}_{model_for_filename}{eval_model_for_filename}_rt-{args.relevance_threshold}_dp-{args.use_default_prompt_template}_tools-{not args.disable_tools}"
    save_file_path = f"{RESULTS_PATH}/{save_filename}.dill"
    tmp_file_path = f"{RESULTS_PATH}/{save_filename}_tmp.dill"

    all_results: list[list[ComprehensivenessResult]] = []
    if os.path.exists(save_file_path):
        with open(save_file_path, "rb") as f:
            all_results = dill.load(f)["results"]
    todo_samples = dataset[len(all_results) :]
    print()
    print(
        f"Loaded {len(all_results)} results, {len(todo_samples)} / {len(dataset)} remaining."
    )
    print()

    query_context_summarizer = QueryContextSummarizer(
        model=args.model_name,
        inject_prompt_template=not args.use_default_prompt_template,
    )
    if args.variant == "qa":
        if args.disable_tools:
            comparison_tools = []
        else:
            comparison_tools = [
                compare_quantities_with_units_definition,
            ]
        qag_processor = QagProcessor(
            model=args.model_name,
            comparison_tools=comparison_tools,
            inject_prompt_template=not args.use_default_prompt_template,
        )
        fact_graph_miner = QagFactGraphMiner(
            relevance_threshold=args.relevance_threshold,
            confidence_threshold=args.confidence_threshold,
            qag_processor=qag_processor,
        )
    elif args.variant == "qa":
        atom_extractor = AtomExtractor(model=args.model_name, prompt_version="v2")
        atom_reviser = BatchAtomReviser(model=args.model_name)
        relevance_estimator = RelevanceEstimator(model=args.model_name)
        nli_extractor = NLIExtractor(model=args.model_name)
        fact_graph_miner = NliFactGraphMiner(
            relevance_threshold=args.relevance_threshold,
            atom_extractor=atom_extractor,
            atom_reviser=atom_reviser,
            relevance_estimator=relevance_estimator,
            nli_extractor=nli_extractor,
        )
    elif "e2e" in args.variant:
        if args.variant == "e2e-base":
            coverage_evaluator = CoverageEvaluator(
                model=args.model_name, version="base"
            )
        else:
            coverage_evaluator = CoverageEvaluator(
                model=args.model_name, version="few-shot"
            )
        fact_graph_miner = EndToEndFactGraphMiner(coverage_evaluator=coverage_evaluator)
    else:
        raise ValueError(f"Unexpected variant {args.variant}!")

    output_generator = None
    if args.evaluated_model_name is not None:
        output_generator = OutputGenerator(model=args.evaluated_model_name)
    comprehensiveness_pipeline = ComprehensivenessPipeline(
        fact_graph_miner=fact_graph_miner,
        query_context_summarizer=query_context_summarizer,
        output_generator=output_generator,
    )

    running_results = []
    for result in all_results:
        for subresult in result:
            running_results.append(subresult["comprehensiveness_eval_main_score"])
    with logging_redirect_tqdm():
        with std_out_err_redirect_tqdm() as orig_stdout:
            for sample in (pbar := tqdm(todo_samples, file=orig_stdout)):
                try:
                    running_avg = sum(running_results) / (len(running_results) + 1e-6)
                    pbar.set_postfix_str(f"Running result: {running_avg:.2f}")
                except Exception:
                    pass

                current_results: None | list[ComprehensivenessResult] = None
                for attempt in Retrying(
                    stop=stop_after_attempt(3),
                    wait=wait_random_exponential(multiplier=10, max=30),
                ):
                    with attempt:
                        current_results = comprehensiveness_pipeline.run_sample(
                            sample, **dataset_config
                        )
                assert current_results is not None, (
                    "Results should never by None after retries."
                )
                for result in current_results:
                    running_results.append(result["comprehensiveness_eval_main_score"])
                all_results.append(current_results)
                print("\n" * 5, end="")

                args_dict = vars(args)
                with open(tmp_file_path, "wb") as f:
                    dill.dump({"args": args_dict, "results": all_results}, f)
                shutil.copy2(tmp_file_path, save_file_path)
                Path(tmp_file_path).unlink()
