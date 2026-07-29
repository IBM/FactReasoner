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

import json
import logging
import math
import time
from typing import Any

from mellea.core import MelleaLogger

# Local imports
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.base import (
    Atom,
    Context,
    Relation,
)
from fact_reasoner.core.nli import NLIExtractor
from fact_reasoner.core.nli_cache import NLIVerdictCache
from fact_reasoner.core.nli_config import FAITHFUL as NLI_FAITHFUL
from fact_reasoner.core.nli_pairs import (
    dedup_contexts_near,
    prune_dangling_context_refs,
)
from fact_reasoner.core.retriever import ContextRetriever
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.core.summarizer import ContextSummarizer
from fact_reasoner.core.utils import (
    build_atoms,
    build_contexts,
    build_relations,
    is_relevant_context,
    remove_duplicated_atoms,
    remove_duplicated_contexts,
)
from fact_reasoner.fact_graph import FactGraph
from fact_reasoner.factors import (
    build_markov_network as _build_markov_network_shared,
)
from fact_reasoner.factors import (
    edge_factor_values as _edge_factor_values_shared,
)
from fact_reasoner.factors import (
    pairwise_prior as _pairwise_prior_shared,
)
from fact_reasoner.inference import run_merlin as _run_merlin_shared

# Set logging levels
logging.getLogger("httpx").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class FactReasoner:
    def __init__(
        self,
        atom_extractor: Atomizer = None,
        atom_reviser: Reviser = None,
        nli_extractor: NLIExtractor = None,
        context_retriever: ContextRetriever = None,
        context_summarizer: ContextSummarizer = None,
        merlin_path: str | None = None,
        use_priors: bool = True,
        early_exit_evaluator: callable | None = None,
        nli_pair_config=None,
        nli_cache_dir: str | None = None,
    ):
        """
        Initialize the FactReasoner pipeline.

        Args:
            atom_extractor: Atomizer
                The service used for extracting atoms from the response.
            atom_reviser: Reviser
                The service used for decontextualizing the atoms.
            context_retriever: ContextRetriever
                The service used for retrieving external contexts.
            context_summarizer: ContextSummarizer
                The service used for summarizing contexts.
            nli_extractor: NLIExtractor
                The service used for NLI relationship extraction.
            merlin_path: str
                Path to the Merlin probabilistic reasoning engine (c++ implementation).
            use_priors: bool
                Flag indicating that atom and context priors are used in the factor definition.
            early_exit_evaluator: callable
                A callable that, if provided, allows early exit from the reasoning process.
                Eg: a single call to the Granite Guardian 3.2 5b sft detect model.
                Should match the following signature:

                def early_exit_evaluator(
                    context: str,
                    response: str
                ) -> Dict{"continue_pipeline_execution": bool, **additional_items}:
                        ...
            nli_pair_config: NLIPairConfig
                Which NLI candidate pairs to score, and how. Defaults to
                `FAITHFUL`, which enumerates every pair exactly as the original
                implementation did.
            nli_cache_dir: str
                Directory for the cross-run NLI verdict cache. None disables it.
                Caching returns previously computed verdicts, so it changes cost
                and never results.
        """

        # Initialize FactReasoner
        self.query = None
        self.response = None
        self.topic = None
        self.use_priors = use_priors
        self.early_exit_evaluator = early_exit_evaluator
        self.early_exit_evaluation = None
        self.start_time = time.perf_counter()  # get the start time

        self.context_retriever = context_retriever
        self.context_summarizer = context_summarizer
        self.atom_extractor = atom_extractor
        self.atom_reviser = atom_reviser
        self.nli_extractor = nli_extractor
        self.merlin_path = merlin_path

        # Safety checks
        assert self.merlin_path is not None, "Path to `merlin` cannot be None."

        print(f"[FactReasoner] Using merlin at: {self.merlin_path}")
        print(f"[FactReasoner] Using atom/context priors: {self.use_priors}")

        self.atoms = {}  # indexed by atom id
        self.contexts = {}  # indexed by context id
        self.relations = []

        self.num_retrieved_contexts = 0
        self.num_summarized_contexts = 0
        self.timing = {}

        # NLI relation-extraction cost control. The default reproduces the
        # original all-pairs behavior; `nli_stats` records what was scored versus
        # pruned so a run reports its own saving.
        self.nli_pair_config = (
            nli_pair_config if nli_pair_config is not None else NLI_FAITHFUL
        )
        self.nli_stats = {}
        self.nli_cache = (
            NLIVerdictCache(nli_cache_dir) if nli_cache_dir else None
        )
        if self.nli_cache is not None:
            print(f"[FactReasoner] Using NLI verdict cache at: {nli_cache_dir}")
        if not self.nli_pair_config.is_faithful:
            print(
                "[FactReasoner] NLI pair policy: "
                f"{self.nli_pair_config.policy} "
                f"(cascade={self.nli_pair_config.ctx_ctx_single_direction_cascade}, "
                f"near-dup dedup={self.nli_pair_config.dedup_near_duplicates})"
            )

        # The fact graph and probabilistic model (Markov Network)
        self.fact_graph = None
        self.markov_network = None

        # Ground truth labels (if any)
        self.labels_human = None

        # Disable Mellea logging
        MelleaLogger.get_logger().setLevel(MelleaLogger.ERROR)

    def from_fact_graph(self, fact_graph: FactGraph):
        """
        Initialize FactReasoner from a FactGraph instance.

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
        data: dict[str, Any],
    ):
        """
        Initialize FactReasoner from a dict containing both atoms and contexts.

        Args:
            data: Dict[str, Any]
                The dict containing the problem instance.
        """

        self.query = data["input"]
        self.response = data["output"]
        self.topic = data.get("topic", None)

        print("[FactReasoner] Reading the atoms ...")
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
        for _, atom in self.atoms.items():  # noqa: PERF102
            print(f"[FactReasoner] {atom}")

        self.labels_human = dict(zip(atom_ids, gold_labels))
        print(f"[FactReasoner] Labels found: {self.labels_human}")

        print("[FactReasoner] Reading the contexts ...")
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

        print(f"[FactReasoner] Contexts found: {len(self.contexts)}")
        for aid, atom in self.atoms.items():
            ctxts = []
            for c in atom2contexts[aid]:
                ctxts.append(self.contexts[c])
                self.contexts[c].set_atom(atom)
            atom.add_contexts(ctxts)

        print(
            f"[FactReasoner] Pipeline initialized with {len(self.atoms)} atoms and {len(self.contexts)} contexts."
        )

    async def build(
        self,
        query: str | None = None,
        response: str | None = None,
        topic: str | None = None,
        has_atoms: bool = False,
        has_contexts: bool = False,
        revise_atoms: bool = False,
        remove_duplicates: bool = False,
        summarize_contexts: bool = True,
        contexts_per_atom_only: bool = False,
        rel_atom_context: bool = True,
        rel_context_context: bool = False,
        use_fast_retriever: bool = True,
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
        """

        # Initialize the reasoner
        if query is not None:
            self.query = query
        if response is not None:
            self.response = response
        if topic is not None:
            self.topic = topic

        self.fact_graph = None
        self.markov_network = None
        self.revise_atoms = revise_atoms
        self.summarize_contexts = summarize_contexts  # default is False

        # Safety checks
        assert self.nli_extractor is not None, "The NLI extractor must be created."

        print("[FactReasoner] Building the pipeline ...")
        _build_start = time.perf_counter()

        # Build the atoms
        if not has_atoms:
            print("[FactReasoner] Extracting the atoms ...")

            assert self.atom_extractor is not None, (
                "The atom extractor must be created."
            )

            _t = time.perf_counter()
            self.atoms = build_atoms(
                response=self.response, atom_extractor=self.atom_extractor
            )
            self.timing["atom_extraction"] = time.perf_counter() - _t
            print(
                f"[FactReasoner][TIMING] Atom extraction: {self.timing['atom_extraction']:.4f}s"
            )
            self.revise_atoms = True  # revise the atoms if newly created
            print(f"[FactReasoner] Extracted {len(self.atoms)} atoms.")
            for aid in self.atoms:
                print(f"[FactReasoner] {self.atoms[aid]}")

        # Safety checks
        assert len(self.atoms) > 0, (
            "The atoms must be initialized before running the pipeline."
        )

        # Revise the atoms
        if self.revise_atoms:
            print("[FactReasoner] Revising the atoms ...")
            assert self.atom_reviser is not None, "The atom reviser must be created."

            assert self.response is not None, "The atom reviser requires a response."
            atom_ids = [aid for aid in sorted(self.atoms.keys())]
            old_atoms = [self.atoms[aid].get_text() for aid in atom_ids]
            _t = time.perf_counter()
            result = self.atom_reviser.run(old_atoms, self.response)
            self.timing["atom_revision"] = time.perf_counter() - _t
            print(
                f"[FactReasoner][TIMING] Atom revision: {self.timing['atom_revision']:.4f}s"
            )
            for i, aid in enumerate(atom_ids):
                elem = result[i]
                self.atoms[aid].set_text(elem["revised_unit"])
                print(f"[FactReasoner] {self.atoms[aid]}")

        # Remove duplicated atoms (if any)
        self.atoms = remove_duplicated_atoms(self.atoms)
        print(f"[FactReasoner] Created {len(self.atoms)} unique atoms.")

        # Build the contexts (per atom)
        if not has_contexts:  # check if contexts already in file
            _t = time.perf_counter()
            self.contexts = build_contexts(
                atoms=self.atoms,
                query=self.query,
                retriever=self.context_retriever,
                use_fast_retriever=use_fast_retriever,
            )
            self.timing["context_retrieval"] = time.perf_counter() - _t
            print(
                f"[FactReasoner][TIMING] Context retrieval: {self.timing['context_retrieval']:.4f}s"
            )

        # For tracking purposes
        self.num_retrieved_contexts = len(self.contexts.keys())
        print(f"[FactReasoner] Retrieved {self.num_retrieved_contexts} contexts.")

        # Safety checks
        assert len(self.contexts.keys()) > 0 or not has_contexts, (
            "Contexts must be initialized if `has_contexts` is True!"
        )

        # Remove duplicated contexts
        if remove_duplicates:
            self.contexts, self.atoms = remove_duplicated_contexts(
                self.contexts, self.atoms
            )
            print(
                f"[FactReasoner] Created {len(self.contexts.keys())} unique contexts."
            )

        # Summarize the retrieved contexts (if any)
        if self.summarize_contexts:
            print("[FactReasoner] Summarizing the contexts ...")
            _t_summarize = time.perf_counter()

            # Summarize contexts for atoms
            _t = time.perf_counter()
            for atom_id, atom in self.atoms.items():
                if len(atom.contexts.keys()) > 0:
                    contexts_ids, contexts = zip(*atom.contexts.items())
                    results = await self.context_summarizer.run_batch(
                        [context.get_text() for context in contexts], atom.text
                    )

                    # Safety checks
                    assert len(results) == len(contexts), (
                        "The number of summaries must be equal to the number of contexts."
                    )

                    # Set the new syntheric summaries
                    for context_id, result in zip(contexts_ids, results):
                        is_relevant = is_relevant_context(result["summary"])
                        if result["summary"] != "" and is_relevant:
                            self.contexts[context_id].set_synthetic_summary(
                                result["summary"]
                            )
                            # update prior probability of context based on the confidence estimation of the summary
                            self.contexts[context_id].set_probability(
                                result["probability"]
                                * self.contexts[context_id].get_probability()
                            )
                        else:
                            # we remove the context because it is not related to the atom
                            del self.contexts[context_id]
                            del self.atoms[atom_id].contexts[context_id]
                    print(
                        f"[FactReasoner] Created {len(results)} summarized contexts for atom {atom_id}."
                    )
            self.timing["context_summarization_atoms"] = time.perf_counter() - _t
            print(
                f"[FactReasoner][TIMING] Context summarization (atoms): {self.timing['context_summarization_atoms']:.4f}s"
            )

            # Summarize contexts for question
            c_qs = {
                c_id: context
                for c_id, context in self.contexts.items()
                if c_id.startswith("c_q")
            }
            if len(c_qs.keys()) > 0:
                _t = time.perf_counter()
                contexts_ids, contexts = zip(*c_qs.items())
                results = await self.context_summarizer.run_batch(
                    [context.get_text() for context in contexts], self.query
                )

                assert len(results) == len(contexts), (
                    "The number of summaries must be equal to the number of contexts."
                )

                for context_id, result in zip(contexts_ids, results):
                    is_relevant = is_relevant_context(result["summary"])
                    if result["summary"] != "" and is_relevant:
                        self.contexts[context_id].set_synthetic_summary(
                            result["summary"]
                        )
                        # update prior probability of context based on the confidence estimation of the summary
                        self.contexts[context_id].set_probability(
                            result["probability"]
                            * self.contexts[context_id].get_probability()
                        )
                    else:
                        # we remove the context because it is not related to the atom
                        del self.contexts[context_id]
                print(
                    f"[FactReasoner] Created {len(results)} summarized contexts for the question."
                )
                self.timing["context_summarization_question"] = time.perf_counter() - _t
                print(
                    f"[FactReasoner][TIMING] Context summarization (question): {self.timing['context_summarization_question']:.4f}s"
                )

            # Contexts judged irrelevant are dropped above, but the atoms that
            # retrieved them are not all cleaned: the per-atom branch clears only
            # the one owning atom (so a context shared by several atoms leaves
            # references behind) and the query branch clears none. Sweep once here
            # so no atom points at a context that no longer exists.
            _dangling = prune_dangling_context_refs(self.atoms, self.contexts)
            if _dangling:
                print(
                    f"[FactReasoner] Pruned {_dangling} stale context reference(s)."
                )

            # For tracking purposes
            self.num_summarized_contexts = len(self.contexts.keys())
            print(
                f"[FactReasoner] Created {self.num_summarized_contexts} summarized contexts."
            )
            self.timing["context_summarization_total"] = (
                time.perf_counter() - _t_summarize
            )
            print(
                f"[FactReasoner][TIMING] Context summarization (total): {self.timing['context_summarization_total']:.4f}s"
            )

            # Remove duplicated contexts that have the same summary (if any)
            if remove_duplicates:
                self.contexts, self.atoms = remove_duplicated_contexts(
                    self.contexts, self.atoms, check_summary=True
                )
                print(
                    f"[FactReasoner] Created {len(self.contexts.keys())} unique summarized contexts."
                )

        # Collapse near-duplicate contexts. Run after summarization on purpose:
        # summaries are short and synthetic, so near-duplicate detection on them is
        # both cheaper and more accurate than on raw pages. Because both dominant
        # NLI cost terms are super-linear in the number of contexts, this has
        # quadratic leverage on the call count.
        if self.nli_pair_config.dedup_near_duplicates and len(self.contexts) > 1:
            _t = time.perf_counter()
            self.contexts, self.atoms, dedup_stats = dedup_contexts_near(
                self.contexts,
                self.atoms,
                threshold=self.nli_pair_config.dedup_threshold,
                use_summary=self.summarize_contexts,
                embedding_model=self.nli_pair_config.embedding_model,
            )
            self.nli_stats["dedup"] = dedup_stats
            self.timing["context_near_dedup"] = time.perf_counter() - _t
            print(
                f"[FactReasoner] Near-duplicate dedup: "
                f"{dedup_stats['contexts_before']} -> {dedup_stats['contexts_after']} "
                f"contexts ({dedup_stats['collapsed']} collapsed, "
                f"{dedup_stats['owners_merged']} owners merged)."
            )

        # If the user submits the override early exit flag then this method
        # will be set to None
        if self.early_exit_evaluator is not None:
            if not callable(self.early_exit_evaluator):
                raise ValueError(
                    f"The `early_exit_evaluator` must be a callable function that takes in input the context and response and outputs a dict with the key `continue_pipeline_execution` (boolean) and optionally other keys with additional information about the evaluation. Instead got: {type(self.early_exit_evaluator)}"
                )
            print("[FactReasoner] Evaluating early exit condition ...")
            _t = time.perf_counter()
            self.early_exit_evaluation = await self.early_exit_evaluator(
                context="\n".join(
                    [
                        c.summary if hasattr(c, "summary") else c.text
                        for c in self.contexts.values()
                    ]
                ),
                response=self.response.strip(),
            )
            self.timing["early_exit_evaluation"] = time.perf_counter() - _t
            print(
                f"[FactReasoner][TIMING] Early exit evaluation: {self.timing['early_exit_evaluation']:.4f}s"
            )

            # set default choice to `True` so that full pipeline is executed
            # if `continue_pipeline_execution` is absent from the early exit evaluation dict
            # for some reason
            if (
                self.early_exit_evaluation.get("continue_pipeline_execution", True)
                is False
            ):
                print(
                    "[FactReasoner] Early exit condition met, exiting reasoning pipeline, returning early exit evaluator output."
                )
                self.timing["build_total"] = time.perf_counter() - _build_start
                print(
                    f"[FactReasoner][TIMING] build() total (early exit): {self.timing['build_total']:.4f}s"
                )
                return

            print(
                "[FactReasoner] Early exit condition not met, continuing with full reasoning pipeline."
            )

        # Build the NLI relationships
        _t = time.perf_counter()
        self.relations = build_relations(
            atoms=self.atoms,
            contexts=self.contexts,
            rel_atom_context=rel_atom_context,
            rel_context_context=rel_context_context,
            contexts_per_atom_only=contexts_per_atom_only,
            nli_extractor=self.nli_extractor,
            use_summarized_contexts=self.summarize_contexts,
            pair_config=self.nli_pair_config,
            stats=self.nli_stats,
            cache=self.nli_cache,
        )
        self.timing["nli_relation_extraction"] = time.perf_counter() - _t
        print(
            f"[FactReasoner][TIMING] NLI relation extraction: {self.timing['nli_relation_extraction']:.4f}s"
        )
        # Per-phase sub-timings, so the two phases can be compared directly.
        for _phase in ("atom_context", "context_context"):
            if _phase in self.nli_stats:
                self.timing[f"nli_{_phase}"] = self.nli_stats[_phase].get("seconds")

        # Build the fact graph and Markov network
        print("[FactReasoner] Building the graphical model ...")
        _t = time.perf_counter()
        self._build_fact_graph()
        self._build_markov_network()
        self.timing["graphical_model_construction"] = time.perf_counter() - _t
        print(
            f"[FactReasoner][TIMING] Graphical model construction: {self.timing['graphical_model_construction']:.4f}s"
        )

        self.timing["build_total"] = time.perf_counter() - _build_start
        print(
            f"[FactReasoner][TIMING] build() total: {self.timing['build_total']:.4f}s"
        )
        print("[FactReasoner] Pipeline instance created.")

    def to_json(self, json_file_path: str | None = None) -> dict[str, Any]:
        """
        Save the FactReasoner instance to a JSON file.

        Args:
            json_file: str
                The path to the output JSON file.
        """

        data = {}
        data["input"] = self.query
        data["output"] = self.response
        data["topic"] = self.topic
        data["atoms"] = []
        data["contexts"] = []

        # Write the atoms
        for aid, atom in self.atoms.items():
            atom_data = {
                "id": aid, "text": atom.get_text(), "contexts": list(atom.get_contexts().keys())
            }
            if atom.get_label() is not None:
                atom_data["label"] = atom.get_label()
            data["atoms"].append(atom_data)

        # Write the contexts
        data["contexts"] = [context.to_json() for context in self.contexts.values()]
        if self.early_exit_evaluation is not None:
            data["early_exit_evaluation"] = self.early_exit_evaluation

        # Write to a JSON file (if any)
        if json_file_path:
            with open(json_file_path, "w") as f:
                json.dump(data, f, indent=4)
            f.close()
            print(f"[FactReasoner] Pipeline instance written to: {json_file_path}")

        return data

    def _build_fact_graph(self):
        """
        Create the fact graph representation from atoms, contexts and relations.
        """

        self.fact_graph = FactGraph(
            atoms=list(self.atoms.values()),
            contexts=list(self.contexts.values()),
            relations=self.relations,
        )

    @staticmethod
    def _pairwise_prior(link: str) -> float:
        """Return the source-node prior for a pairwise factor given its link type.

        Thin wrapper over :func:`fact_reasoner.factors.pairwise_prior`.

        Args:
            link: The edge link type ("context_atom", "context_context", or
                "atom_atom").

        Returns:
            The prior probability of the source node being true.
        """
        return _pairwise_prior_shared(link)

    def _edge_factor_values(self, edge) -> list[float]:
        """Compute the flattened pairwise factor table for a fact-graph edge.

        Thin wrapper over :func:`fact_reasoner.factors.edge_factor_values`,
        passing this assessor's ``use_priors`` setting.

        Args:
            edge: A fact-graph edge with ``type``, ``link`` and ``probability``.

        Returns:
            The four factor values for the pairwise factor.
        """
        return _edge_factor_values_shared(edge, use_priors=self.use_priors)

    def _build_markov_network(self):
        """
        Create the Markov Network corresponding to the FactGraph.

        Delegates to :func:`fact_reasoner.factors.build_markov_network`.

        Return:
            A MarkovNetwork encoding of the problem.
        """

        assert self.fact_graph is not None, "The FactGraph must be built."

        logger.debug("Building the Markov network ...")
        self.markov_network = _build_markov_network_shared(
            self.fact_graph, use_priors=self.use_priors
        )
        logger.debug("Markov network created.")

    def run_merlin(self):
        """
        Run inference with merlin (executable).

        Delegates to :func:`fact_reasoner.inference.run_merlin` (MAR task),
        restricting the returned marginals to the atom variables.
        """

        # Prepare the query variables (i.e., atoms)
        query_variables = [var for var in sorted(self.atoms.keys())]

        result = _run_merlin_shared(
            self.markov_network,
            self.merlin_path,
            task="MAR",
            query_variables=query_variables,
        )
        return result["marginals"]

    def score(self) -> dict[str, Any]:
        """
        Compute the factuality score taking into consideration the contexts
        retrieved for each of the atom in the answer.

        Factuality score = # atoms(true) / # atoms

        Intuitively, a score of 100% means that all atoms in the answer are
        factually correct. If none of them are correct, then the score is 0%. If
        only half of the atoms are correct, then the score is 50%.

        Returns:
            Dict[str, Any]: The results dictionary containing the marginals, factuality score i.e., a real value in [0, 1]
        """

        # Safety checks
        if len(self.contexts.keys()) == 0:
            print("WARNING: no contexts have been retrieved!")
        if len(self.relations) == 0:
            print("WARNING: no relationships have been identified!")

        # Without atoms all the factuality metrics are undefined (they divide by
        # the number of atoms), so return a well-defined empty result instead of
        # crashing with a ZeroDivisionError.
        if len(self.atoms) == 0:
            print("WARNING: no atoms have been identified!")
            results = {
                "factuality_score_per_atom": [],
                "factuality_score": 0.0,
                "recall_k": 0.0,
                "f1_k": 0.0,
                "num_atoms": 0,
                "num_contexts": len(self.contexts),
                "num_true_atoms": 0,
                "num_false_atoms": 0,
                "num_uniform_atoms": 0,
                "entropy": 0.0,
                "norm_entropy": 0.0,
                "avg_entropy": 0.0,
                "avg_norm_entropy": 0.0,
                "avg_prob": 0.0,
                "avg_logprob": 0.0,
                "avg_explogprob": 0.0,
                "marginals": [],
                "predictions": {},
                "topic": self.topic,
                "query": self.query,
                "response": self.response,
                "elapsed_time": time.perf_counter() - self.start_time,
            }
            return results, []

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
        labels = {}
        probabilities = {}
        fscore_per_atom = []
        for marginal in marginals:
            var = marginal["variable"]
            probs = marginal["probabilities"]

            print(f"[FactReasoner] ({var}): Probability for {var}=0 is: {probs[0]}")
            print(f"[FactReasoner] ({var}): Probability for {var}=1 is: {probs[1]}")

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

        # For now, return a dict with the posterior marginals of the atoms
        num_atoms = len(self.atoms)
        avg_logprob /= num_atoms
        avg_prob /= num_atoms
        avg_entropy = entropy / num_atoms
        avg_norm_entropy = norm_entropy / num_atoms

        # Precision, R@K and F1@K
        fscore = float(num_true_atoms) / float(num_atoms)
        K = int(num_atoms / 2)  # K is assumed to be half
        # ensure that K is at least 1 to avoid division by zero when there is only one atom (since int(0.5) would be 0)
        K = max(K, 1)
        recall_k = min(float(num_true_atoms / K), 1.0)
        # fscore + recall_k == 0 exactly when there are no true atoms.
        if fscore + recall_k > 0.0:
            f1k = 2 * fscore * recall_k / (fscore + recall_k)
        else:
            f1k = 0.0

        # Elapsed time
        elapsed_time = time.perf_counter() - self.start_time  # elapsed time

        results = {}
        results["factuality_score_per_atom"] = fscore_per_atom
        results["factuality_score"] = fscore
        results["recall_k"] = recall_k
        results["f1_k"] = f1k
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
        results["marginals"] = marginals
        results["predictions"] = labels
        # Carry the NLI cost report into the scored output, so a whole evaluation
        # sweep can be analyzed for call counts after the fact.
        if self.nli_stats:
            results["nli_stats"] = self.nli_stats
        print(f"[FactReasoner] Predictions: {labels}")

        # Remove duplicate atoms in self.labels_human
        if self.labels_human is not None:
            self.labels_human = {
                k: v
                for i, (k, v) in enumerate(self.labels_human.items())
                if k in self.atoms
                and k not in list(self.labels_human.keys())[:i]
            }

        # Check for ground truth annotations
        if self.labels_human is not None:
            true_atoms = 0
            false_atoms = 0
            avg_brier = 0.0
            num_true_positive = 0
            num_true_negative = 0
            num_false_positive = 0
            num_false_negative = 0
            for aid, gold_label in self.labels_human.items():
                if gold_label == "S":
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
            print(f"[FactReasoner] Gold labels: {self.labels_human}")
            print(
                f"[FactReasoner] Gold fscore: {fscore_gold} ({true_atoms}/{len(self.labels_human.keys())})"
            )
            results["gold_factuality_score"] = fscore_gold
            results["gold_true_atoms"] = true_atoms
            results["true_positive"] = num_true_positive
            results["true_negative"] = num_true_negative
            results["false_positive"] = num_false_positive
            results["false_negative"] = num_false_negative
            results["references"] = self.labels_human
            results["avg_brier"] = avg_brier

        results["topic"] = self.topic
        results["query"] = self.query
        results["response"] = self.response
        results["elapsed_time"] = elapsed_time
        print(f"[FactReasoner] Elapsed time: {elapsed_time:.4f} seconds.")

        return results, marginals

    def pipeline_to_json(self, json_file_path: str | None = None):
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
            atom_data = {
                "id": aid, "text": atom.get_text(), "contexts": list(atom.get_contexts().keys())
            }
            if atom.get_label() is not None:
                atom_data["label"] = atom.get_label()
            data["atoms"].append(atom_data)

        data["contexts"] = [context.to_json() for context in self.contexts.values()]
        if self.early_exit_evaluation is not None:
            data["early_exit_evaluation"] = self.early_exit_evaluation
        if self.timing:
            data["timing"] = self.timing
        if self.nli_stats:
            data["nli_stats"] = self.nli_stats

        if json_file_path:
            with open(json_file_path, "w") as f:
                f.write(f"{json.dumps(data)}\n")
            f.close()
            print(f"[FactReasoner] Pipeline instance written to: {json_file_path}")

        return data
