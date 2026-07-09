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

# Unified factuality runner.
#
# A single class that runs any factuality assessor (FactReasoner or a baseline)
# with any Mellea backend, over either a single query+response or a jsonl dataset
# of pre-annotated responses.

import asyncio
import inspect
import json
import os

from typing import Any, Dict, List, Optional

from mellea.backends import Backend

from fact_reasoner.assessor import FactReasoner
from fact_reasoner.baselines.factscore import FactScore
from fact_reasoner.baselines.factverify import FactVerify
from fact_reasoner.baselines.veriscore import VeriScore
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.core.retriever import ContextRetriever, SourceRetriever
from fact_reasoner.core.query_builder import QueryBuilder
from fact_reasoner.core.summarizer import ContextSummarizer
from fact_reasoner.core.nli import NLIExtractor

# Recognized factuality pipelines.
PIPELINES = ("factreasoner", "factscore", "veriscore", "factverify")

# FactReasoner version -> (rel_context_context, remove_duplicates, contexts_per_atom_only).
_FR_VERSIONS = {
    "v1": (False, False, True),  # context-atom only, allow duplicated contexts
    "v2": (False, True, False),  # context-atom only, no duplicated contexts
    "v3": (True, True, False),  # context-atom + context-context, no duplicates
}


def _run_build(pipeline_obj, /, **kwargs) -> None:
    """Call a pipeline's ``build`` correctly whether it is sync or async.

    ``FactReasoner.build`` is a coroutine; the baseline pipelines' ``build`` is a
    plain method. This drives either uniformly.
    """
    result = pipeline_obj.build(**kwargs)
    if inspect.isawaitable(result):
        asyncio.run(result)


class FactualityRunner:
    """Run a factuality assessor over a single item or a dataset.

    The runner owns a Mellea backend and the shared pipeline components, and
    exposes two entry points:

    * :meth:`assess` — score a single ``query`` / ``response`` pair, generating
      atoms and contexts from scratch (requires a retriever).
    * :meth:`assess_file` — score a jsonl dataset whose items already contain
      atoms and contexts (like the legacy evaluation driver), writing results
      incrementally and skipping already-processed inputs.

    Args:
        backend: The Mellea backend that drives all components.
        pipeline: Which assessor to run — one of ``PIPELINES``.
        pipeline_version: FactReasoner version (``v1``/``v2``/``v3``);
            ignored by the baselines.
        service_type: Retrieval service (``google``/``wikipedia``/``chromadb``).
        cache_dir: Optional retriever cache directory.
        top_k: Top-k contexts retrieved per atom.
        num_workers: Parallelism for context retrieval.
        use_priors: Use atom/context priors (FactReasoner only).
        use_summarizer: Summarize contexts (FactReasoner only).
        use_query_builder: Use the QueryBuilder for search queries.
        merlin_path: Path to the Merlin inference engine (required for
            FactReasoner).
        nli_method: How the NLI extractor estimates relation probabilities —
            ``logprobs`` (needs a logprobs-capable backend like RITS/vLLM) or
            ``simbauq`` (self-consistency; backend-agnostic, required for
            Ollama which does not expose logprobs).
        nli_similarity_metric: Similarity metric for the SIMBA-UQ NLI method
            (only used when ``nli_method='simbauq'``).
    """

    def __init__(
        self,
        backend: Backend,
        *,
        pipeline: str = "factreasoner",
        pipeline_version: str = "v2",
        service_type: str = "google",
        cache_dir: Optional[str] = None,
        top_k: int = 3,
        num_workers: int = 4,
        use_priors: bool = False,
        use_summarizer: bool = False,
        use_query_builder: bool = False,
        merlin_path: Optional[str] = None,
        nli_method: str = "logprobs",
        nli_similarity_metric: str = "rouge",
        nli_confidence_method: str = "aggregation",
        nli_classifier_path: Optional[str] = None,
    ) -> None:
        """Initialize the runner and its shared components."""
        if pipeline not in PIPELINES:
            raise ValueError(
                f"Unknown pipeline: {pipeline!r} (expected one of {list(PIPELINES)})."
            )
        if pipeline_version not in _FR_VERSIONS:
            raise ValueError(
                f"Unknown pipeline_version: {pipeline_version!r} "
                f"(expected one of {list(_FR_VERSIONS)})."
            )
        if pipeline == "factreasoner" and not merlin_path:
            raise ValueError("The 'factreasoner' pipeline requires a merlin_path.")

        self.backend = backend
        self.pipeline = pipeline
        self.pipeline_version = pipeline_version
        self.service_type = service_type
        self.cache_dir = cache_dir
        self.top_k = top_k
        self.num_workers = num_workers
        self.use_priors = use_priors
        self.use_summarizer = use_summarizer
        self.use_query_builder = use_query_builder
        self.merlin_path = merlin_path
        self.nli_method = nli_method
        self.nli_similarity_metric = nli_similarity_metric
        self.nli_confidence_method = nli_confidence_method
        self.nli_classifier_path = nli_classifier_path

        # Shared components.
        self.atom_extractor = Atomizer(backend)
        self.atom_reviser = Reviser(backend)
        self.nli_extractor = NLIExtractor(
            backend,
            nli_method=self.nli_method,
            simbauq_similarity_metric=self.nli_similarity_metric,
            simbauq_confidence_method=self.nli_confidence_method,
            simbauq_classifier_path=self.nli_classifier_path,
        )
        self.context_summarizer = ContextSummarizer(backend)

    def _build_context_retriever(self) -> ContextRetriever:
        """Wire a ``SourceRetriever`` into a ``ContextRetriever`` (the correct order).

        ``ContextRetriever`` wraps a ``SourceRetriever``; constructing it with the
        ``SourceRetriever`` keyword arguments directly is incorrect.
        """
        query_builder = QueryBuilder(self.backend) if self.use_query_builder else None
        retriever = SourceRetriever(
            service_type=self.service_type,
            top_k=self.top_k,
            cache_dir=self.cache_dir,
            # Baselines that check a claim against retrieved text don't fetch
            # full page content; FactReasoner and the others do.
            fetch_text=self.pipeline != "factverify",
            query_builder=query_builder,
            num_workers=self.num_workers,
        )
        return ContextRetriever(
            retriever=retriever,
            context_summarizer=self.context_summarizer,
            num_workers=self.num_workers,
        )

    def _make_pipeline(self, context_retriever: ContextRetriever):
        """Construct the selected assessor with the shared components."""
        if self.pipeline == "factreasoner":
            return FactReasoner(
                atom_extractor=self.atom_extractor,
                atom_reviser=self.atom_reviser,
                nli_extractor=self.nli_extractor,
                context_retriever=context_retriever,
                context_summarizer=self.context_summarizer,
                merlin_path=self.merlin_path,
                use_priors=self.use_priors,
            )
        elif self.pipeline == "factscore":
            return FactScore(
                backend=self.backend,
                atom_extractor=self.atom_extractor,
                atom_reviser=self.atom_reviser,
                context_retriever=context_retriever,
            )
        elif self.pipeline == "veriscore":
            return VeriScore(
                backend=self.backend,
                atom_extractor=self.atom_extractor,
                atom_reviser=self.atom_reviser,
                context_retriever=context_retriever,
            )
        else:  # factverify
            return FactVerify(
                atom_extractor=self.atom_extractor,
                atom_reviser=self.atom_reviser,
                context_retriever=context_retriever,
            )

    @staticmethod
    def _normalize_results(scored: Any) -> Dict[str, Any]:
        """Normalize the pipeline ``score()`` return to a results dict.

        ``FactReasoner.score`` returns ``(results, marginals)``; the baselines
        return just ``results``.
        """
        if isinstance(scored, tuple):
            return scored[0]
        return scored

    def assess(
        self,
        query: str,
        response: str,
        topic: Optional[str] = None,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assess a single query/response pair.

        Atoms and contexts are generated from scratch (the response is atomized,
        atoms are revised, and contexts are retrieved), so a working retriever is
        required.

        Args:
            query: The input query.
            response: The response to assess.
            topic: Optional topic hint.
            output_file: If set, write the results dict to this path as JSON.

        Returns:
            The results dictionary.
        """
        context_retriever = self._build_context_retriever()
        pipeline_obj = self._make_pipeline(context_retriever)

        rel_ctx_ctx, remove_dups, ctx_per_atom = _FR_VERSIONS[self.pipeline_version]
        _run_build(
            pipeline_obj,
            query=query,
            response=response,
            topic=topic,
            has_atoms=False,
            has_contexts=False,
            revise_atoms=True,
            remove_duplicates=remove_dups,
            contexts_per_atom_only=ctx_per_atom,
            rel_atom_context=True,
            rel_context_context=rel_ctx_ctx,
            summarize_contexts=self.use_summarizer,
        )

        results = self._normalize_results(pipeline_obj.score())

        if output_file:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
            print(f"[FactualityRunner] Results written to: {output_file}")

        return results

    def assess_file(
        self,
        input_file: str,
        output_dir: str,
        *,
        dataset_name: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Assess a jsonl dataset of pre-annotated responses.

        Each item is expected to already contain atoms and contexts (as produced
        by :meth:`FactReasoner.to_json`). Results are written incrementally to a
        jsonl file in ``output_dir``, and inputs already present in that file are
        skipped, so the run is resumable.

        Args:
            input_file: Path to the input jsonl dataset.
            output_dir: Directory for the output jsonl.
            dataset_name: Dataset label (used in the output filename).
            model_id: Model label recorded in each result and the filename.

        Returns:
            The list of result dictionaries.
        """
        pipeline_name = (
            f"factreasoner-{self.pipeline_version}"
            if self.pipeline == "factreasoner"
            else self.pipeline
        )
        rel_ctx_ctx, remove_dups, ctx_per_atom = _FR_VERSIONS[self.pipeline_version]

        # Load the dataset (one JSON object per line).
        with open(input_file) as f:
            dataset = [json.loads(line) for line in f.read().splitlines() if line]
        print(f"[FactualityRunner] Loaded {len(dataset)} items from {input_file}")

        os.makedirs(output_dir, exist_ok=True)
        out_name = (
            f"eval_{pipeline_name}_{self.service_type}_{dataset_name}_{model_id}.jsonl"
        )
        output_filename = os.path.join(output_dir, out_name)

        # Resume: load any previously computed results and skip their inputs.
        evaluation_data: List[Dict[str, Any]] = []
        if os.path.isfile(output_filename):
            with open(output_filename, "r") as f:
                evaluation_data = [json.loads(line) for line in f if line.strip()]
        done_inputs = {e.get("input") for e in evaluation_data}
        print(f"[FactualityRunner] Found {len(evaluation_data)} existing results")

        for input_data in dataset:
            if input_data.get("input") in done_inputs:
                print("[FactualityRunner] Skipping already-processed input.")
                continue

            context_retriever = self._build_context_retriever()
            pipeline_obj = self._make_pipeline(context_retriever)
            pipeline_obj.from_dict_with_contexts(input_data)

            build_kwargs = dict(has_atoms=True, has_contexts=True, revise_atoms=False)
            if self.pipeline == "factreasoner":
                build_kwargs.update(
                    remove_duplicates=remove_dups,
                    contexts_per_atom_only=ctx_per_atom,
                    rel_atom_context=True,
                    rel_context_context=rel_ctx_ctx,
                    summarize_contexts=self.use_summarizer,
                )
            _run_build(pipeline_obj, **build_kwargs)

            results = self._normalize_results(pipeline_obj.score())
            results["model_name"] = model_id
            evaluation_data.append(results)

            # Write incrementally so a crash keeps completed work.
            with open(output_filename, "w") as f:
                for res in evaluation_data:
                    f.write(f"{json.dumps(res)}\n")

        print(f"[FactualityRunner] Results written to: {output_filename}")
        return evaluation_data
