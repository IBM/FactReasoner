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
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from mellea.backends import Backend

from fact_reasoner.assessor import FactReasoner
from fact_reasoner.baselines.factscore import FactScore
from fact_reasoner.baselines.factverify import FactVerify
from fact_reasoner.baselines.veriscore import VeriScore
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.atomizer_reviser import AtomizerReviser
from fact_reasoner.utils import busy_overlap_s
from fact_reasoner.core.nli import (
    NLI_PROMPTS,
    DecisionNLIExtractor,
    LabelOnlyNLIExtractor,
    NLIAdapterExtractor,
    NLIExtractor,
)
from fact_reasoner.core.nli_config import NLI_MODES, get_pair_config
from fact_reasoner.core.query_builder import QueryBuilder
from fact_reasoner.core.retriever import ContextRetriever, SourceRetriever
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.core.summarizer import ContextSummarizer

# Recognized factuality pipelines.
PIPELINES = ("factreasoner", "factscore", "veriscore", "factverify")

# FactReasoner version -> (rel_atom_context, rel_context_context,
# remove_duplicates, contexts_per_atom_only).
#
# The version selects the *graph shape* only. How many NLI candidate pairs get
# scored within that shape is an orthogonal axis, chosen by `nli_mode` -- so every
# (version, mode) combination is expressible rather than only the handful that
# once had a named `-cheap` row here.
_FR_VERSIONS = {
    "v1": (True, False, False, True),  # context-atom only, allow duplicated contexts
    "v2": (True, False, True, False),  # context-atom only, no duplicated contexts
    "v3": (True, True, True, False),  # context-atom + context-context, no duplicates
}


def _run_build(pipeline_obj, /, *, release_clients_of=(), **kwargs) -> None:
    """Call a pipeline's ``build`` correctly whether it is sync or async.

    ``FactReasoner.build`` is a coroutine; the baseline pipelines' ``build`` is a
    plain method. This drives either uniformly.

    ``release_clients_of`` names backends whose HTTP client for this call's event
    loop is closed before the loop ends (see :func:`_release_loop_clients`).
    """
    result = pipeline_obj.build(**kwargs)
    if inspect.isawaitable(result):
        if release_clients_of:

            async def build_then_release():
                try:
                    return await result
                finally:
                    await _release_loop_clients(release_clients_of)

            asyncio.run(build_then_release())
        else:
            asyncio.run(result)


async def _release_loop_clients(backends) -> None:
    """Close and forget each backend's cached client for the running loop.

    Each item ``assess_file`` runs concurrently gets its own ``asyncio.run`` loop,
    and Mellea caches one client per loop. Left in the cache, a finished item's
    client is closed later by LRU eviction on its loop, which is closed by then:
    nothing fails, but every eviction logs a "Task exception was never
    retrieved ... Event loop is closed" traceback. Closing it here, while its
    loop still runs, is the close the cache would have done, done in time.
    """
    loop = asyncio.get_running_loop()
    for backend in backends:
        cache = getattr(backend, "_client_cache", None)
        entries = getattr(cache, "cache", None)
        lock = getattr(cache, "_lock", None)
        if entries is None or lock is None or not hasattr(cache, "aclose"):
            continue  # not a Mellea client cache
        with lock:
            client = entries.pop(loop, None)
        if client is not None and cache.aclose is not None:
            try:
                await cache.aclose(client)
            except Exception:  # noqa: BLE001  best-effort cleanup at item end
                pass


def _existing_throttle(backend: Any):
    """The backend's shared throttle, if one was created (never creates one)."""
    from fact_reasoner.utils import _backend_throttles, _backend_throttles_lock

    with _backend_throttles_lock:
        try:
            return _backend_throttles.get(backend)
        except TypeError:
            return None


def format_timing_report(report: dict[str, Any]) -> str:
    """Render :meth:`FactualityRunner.timing_report` as a short text table."""
    wall = report["wall_s"]
    lines = [
        "[FactualityRunner] Timing report",
        f"  items {report['items']} at concurrency {report['item_concurrency']}: "
        f"{wall:.0f} s wall ({report['s_per_item']:.2f} s/item)",
    ]
    for role, s in report["servers"].items():
        line = (
            f"  {role:3s} busy {s['busy_s']:7.0f} s ({100 * s['busy_s'] / wall if wall else 0:3.0f}% of wall), "
            f"{s['requests']} requests ({s.get('retries', 0)} retried), "
            f"mean {s['mean_concurrency']:.1f} in flight "
            f"(peak {s['peak_in_flight']}/{s['max_concurrency']}), "
            f"queued here {s['wait_s']:.0f} s"
        )
        if role == "nli":
            line += f", {s['pairs']} pairs = {s['pairs_per_busy_s']:.0f} pairs/busy s"
        lines.append(line)
    if report.get("llm_nli_overlap_s") is not None:
        lines.append(f"  llm and nli both busy: {report['llm_nli_overlap_s']:.0f} s")
    for name, seconds in sorted(report["stage_sums_s"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  stage sum (overlapping items) {name}: {seconds:.0f} s")
    return "\n".join(lines)


def _fit_client_cache(backend: Any, item_concurrency: int) -> None:
    """Let a Mellea backend keep one HTTP client per event loop in use at once.

    Mellea caches one async client per event loop and closes the least recently
    used one when its cache (2 entries) is full. ``assess_file`` with
    ``item_concurrency`` workers drives a backend from that many ``asyncio.run``
    loops plus the shared background loop, so a cache of 2 closes clients that
    other items are still using ("Cannot send a request, as the client has been
    closed"), and their requests fail. A cache with room for every live loop only
    evicts the clients of items that have finished.
    """
    cache = getattr(backend, "_client_cache", None)
    if not isinstance(getattr(cache, "capacity", None), int):
        return  # not a Mellea client cache (another backend type, or a mock)
    # The workers' loops, the background loop, and a finished item's loop that
    # LRU has not yet evicted while its successor starts.
    cache.capacity = max(cache.capacity, 2 * item_concurrency + 2)


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
        service_type: Retrieval service (``google``/``wikipedia``/``chromadb``/``ntrs``).
        cache_dir: Optional retriever cache directory.
        top_k: Top-k contexts retrieved per atom.
        num_workers: Threads fetching each search's result pages in parallel.
            Items (atoms and the query) are always retrieved concurrently, one
            worker each.
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
        nli_prompt: The NLI prompt for the base model — ``"few_shot"``
            (default) has the model reason before its label, ``"label_only"``
            asks for the label alone, which is several times faster. Ignored
            when ``nli_backend`` is given: the adapter has its own prompt.
        nli_decision_model: An Ollama decision model (e.g. ``"kev"``) to
            predict the NLI relations instead of an LLM, through the System One
            API (see :class:`~fact_reasoner.core.nli.DecisionNLIExtractor`).
            Its probabilities come from the model itself, so it cannot be
            combined with ``nli_backend``, ``nli_prompt="label_only"`` or
            ``nli_method="simbauq"``.
        nli_decision_host: The System One server for ``nli_decision_model``
            (Ollama, or ``kev.serve``); None uses ``OLLAMA_HOST`` or
            ``http://localhost:11434``.
        nli_decision_api_key: Bearer token for that server, if it requires one
            (``kev.serve`` started by :class:`~fact_reasoner.serving.KevServer`).
        nli_mode: Which NLI candidate-pair preset to start from — ``"fast"``
            (default) applies the provenance preset (atom-context pairs
            restricted to the atoms that retrieved each context, gated
            context-context pairs, near-duplicate contexts collapsed, and
            one-direction context-pair scoring), while ``"all_pairs"`` scores
            every enumerated pair and reproduces the published numbers. Orthogonal to
            ``pipeline_version``: the version fixes the graph shape, this fixes how
            many pairs are scored within it. Individual ``nli_*`` arguments below
            override whichever preset this selects. Has little effect on ``"v1"``,
            whose atom-context pairs are already restricted to each atom's own
            contexts and which runs no context-context phase.
        atomizer_backend: Optional backend for the one-shot atomizer-reviser LoRA
            adapter (``vllm serve ... --enable-lora --lora-modules <name>=<path>``,
            with the adapter name as ``model_id``). When given, atoms are
            extracted and decontextualized in one call by
            :class:`~fact_reasoner.core.atomizer_reviser.AtomizerReviser` and the
            separate Reviser stage is skipped; every other component still uses
            ``backend``.
        nli_backend: Optional backend for the NLI LoRA adapter (served like the
            atomizer adapter). When given, NLI relations are predicted by
            :class:`~fact_reasoner.core.nli.NLIAdapterExtractor` with the same
            ``nli_method`` and SIMBA-UQ settings; its verdicts have the format
            the pipeline already consumes.
        raise_errors: If True, an exception raised by the backend during atom
            extraction, atom revision, NLI or context summarization propagates
            to the caller unchanged. Default False: the error is printed and the
            stage falls back to its default result (no atoms, an unrevised
            atom, a neutral relation, an empty summary).
        item_concurrency: How many dataset items :meth:`assess_file` works on
            at once. Default 1 (one after another). Raise it with local servers,
            so the LLM and NLI servers are both busy instead of taking turns;
            raise the backends' shared throttles with it (see
            :func:`fact_reasoner.utils.set_throttle`).
    """

    def __init__(
        self,
        backend: Backend,
        *,
        pipeline: str = "factreasoner",
        pipeline_version: str = "v2",
        service_type: str = "google",
        cache_dir: str | None = None,
        top_k: int = 3,
        num_workers: int = 4,
        use_priors: bool = False,
        use_summarizer: bool = False,
        use_query_builder: bool = False,
        merlin_path: str | None = None,
        nli_method: str = "logprobs",
        nli_prompt: str = "few_shot",
        nli_decision_model: str | None = None,
        nli_decision_host: str | None = None,
        nli_decision_api_key: str | None = None,
        nli_similarity_metric: str = "rouge",
        nli_confidence_method: str = "aggregation",
        nli_classifier_path: str | None = None,
        nli_mode: str = "fast",
        nli_pair_policy: str | None = None,
        nli_gate_threshold: float | None = None,
        nli_neighbor_window: int | None = None,
        nli_dedup_near_duplicates: bool | None = None,
        nli_dedup_threshold: float | None = None,
        nli_ctx_ctx_cascade: bool | None = None,
        nli_merge_phases: bool | None = None,
        nli_cache_dir: str | None = None,
        show_progress: bool = False,
        atomizer_backend: Backend | None = None,
        nli_backend: Backend | None = None,
        raise_errors: bool = False,
        item_concurrency: int = 1,
    ) -> None:
        """Initialize the runner and its shared components."""
        if item_concurrency < 1:
            raise ValueError(f"item_concurrency must be at least 1, got {item_concurrency}.")
        if pipeline not in PIPELINES:
            raise ValueError(
                f"Unknown pipeline: {pipeline!r} (expected one of {list(PIPELINES)})."
            )
        if pipeline_version not in _FR_VERSIONS:
            raise ValueError(
                f"Unknown pipeline_version: {pipeline_version!r} "
                f"(expected one of {list(_FR_VERSIONS)})."
            )
        if nli_mode not in NLI_MODES:
            raise ValueError(
                f"Unknown nli_mode: {nli_mode!r} (expected one of {list(NLI_MODES)})."
            )
        if nli_prompt not in NLI_PROMPTS:
            raise ValueError(
                f"Unknown nli_prompt: {nli_prompt!r} "
                f"(expected one of {list(NLI_PROMPTS)})."
            )
        if nli_decision_model is not None:
            conflicts = [
                name
                for name, used in (
                    ("nli_backend", nli_backend is not None),
                    ("nli_prompt='label_only'", nli_prompt != "few_shot"),
                    ("nli_method='simbauq'", nli_method != "logprobs"),
                )
                if used
            ]
            if conflicts:
                raise ValueError(
                    "nli_decision_model replaces the LLM NLI extractor, so it "
                    f"cannot be combined with {', '.join(conflicts)}."
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
        self.nli_prompt = nli_prompt
        self.nli_similarity_metric = nli_similarity_metric
        self.nli_confidence_method = nli_confidence_method
        self.nli_classifier_path = nli_classifier_path
        self.nli_cache_dir = nli_cache_dir
        self.show_progress = show_progress
        self.raise_errors = raise_errors
        self.item_concurrency = item_concurrency
        # Distinct, so a backend shared by two roles is released once.
        self._llm_backends = list(
            {id(b): b for b in (backend, atomizer_backend, nli_backend) if b is not None}.values()
        )
        if item_concurrency > 1:
            for b in self._llm_backends:
                _fit_client_cache(b, item_concurrency)

        # `nli_mode` picks the pair-config preset; any explicit override then wins,
        # so a caller can start from a preset and adjust one knob.
        self.nli_mode = nli_mode
        _overrides = {
            "policy": nli_pair_policy,
            "gate_threshold": nli_gate_threshold,
            "neighbor_window": nli_neighbor_window,
            "dedup_near_duplicates": nli_dedup_near_duplicates,
            "dedup_threshold": nli_dedup_threshold,
            "ctx_ctx_single_direction_cascade": nli_ctx_ctx_cascade,
            "merge_phases": nli_merge_phases,
        }
        _overrides = {k: v for k, v in _overrides.items() if v is not None}
        self.nli_pair_config = replace(get_pair_config(nli_mode), **_overrides)

        # Shared components.
        self.atom_extractor = (
            AtomizerReviser(atomizer_backend, raise_errors=raise_errors)
            if atomizer_backend is not None
            else Atomizer(backend, raise_errors=raise_errors)
        )
        self.atom_reviser = Reviser(backend, raise_errors=raise_errors)
        self.nli_decision_model = nli_decision_model
        if nli_decision_model is not None:
            self.nli_extractor = DecisionNLIExtractor(
                nli_decision_model,
                host=nli_decision_host,
                api_key=nli_decision_api_key,
                show_progress=self.show_progress,
                raise_errors=raise_errors,
            )
        else:
            if nli_backend is not None:
                nli_cls = NLIAdapterExtractor
            elif nli_prompt == "label_only":
                nli_cls = LabelOnlyNLIExtractor
            else:
                nli_cls = NLIExtractor
            self.nli_extractor = nli_cls(
                nli_backend if nli_backend is not None else backend,
                nli_method=self.nli_method,
                simbauq_similarity_metric=self.nli_similarity_metric,
                simbauq_confidence_method=self.nli_confidence_method,
                simbauq_classifier_path=self.nli_classifier_path,
                show_progress=self.show_progress,
                raise_errors=raise_errors,
            )
        self.context_summarizer = ContextSummarizer(
            backend, show_progress=self.show_progress, raise_errors=raise_errors
        )

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
                nli_pair_config=self.nli_pair_config,
                nli_cache_dir=self.nli_cache_dir,
            )
        elif self.pipeline == "factscore":
            return FactScore(
                backend=self.backend,
                atom_extractor=self.atom_extractor,
                atom_reviser=self.atom_reviser,
                context_retriever=context_retriever,
                show_progress=self.show_progress,
            )
        elif self.pipeline == "veriscore":
            return VeriScore(
                backend=self.backend,
                atom_extractor=self.atom_extractor,
                atom_reviser=self.atom_reviser,
                context_retriever=context_retriever,
                show_progress=self.show_progress,
            )
        else:  # factverify
            return FactVerify(
                backend=self.backend,
                atom_extractor=self.atom_extractor,
                atom_reviser=self.atom_reviser,
                context_retriever=context_retriever,
                show_progress=self.show_progress,
            )

    @staticmethod
    def _normalize_results(scored: Any) -> dict[str, Any]:
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
        topic: str | None = None,
        output_file: str | None = None,
    ) -> dict[str, Any]:
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
        results, _pipeline = self.assess_with_pipeline(
            query, response, topic=topic, output_file=output_file
        )
        return results

    def assess_with_pipeline(
        self,
        query: str,
        response: str,
        topic: str | None = None,
        output_file: str | None = None,
    ) -> tuple[dict[str, Any], Any]:
        """Assess a single pair and also return the assessor instance.

        Identical work to :meth:`assess`; the extra return value exposes the
        pipeline object, whose ``atoms``, ``contexts`` and ``fact_graph`` outlive
        the call. A second stage can then reuse the atoms rather than atomizing
        the response again -- see ``fact_reasoner.lcs.priors``, which turns this
        run's posterior marginals into the coherence MRF's atom priors.

        Args:
            query: The input query.
            response: The response to assess.
            topic: Optional topic hint.
            output_file: If set, write the results dict to this path as JSON.

        Returns:
            ``(results, pipeline)``, where ``results`` is what :meth:`assess`
            returns.
        """
        context_retriever = self._build_context_retriever()
        pipeline_obj = self._make_pipeline(context_retriever)

        rel_atom_ctx, rel_ctx_ctx, remove_dups, ctx_per_atom = _FR_VERSIONS[
            self.pipeline_version
        ]
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
            rel_atom_context=rel_atom_ctx,
            rel_context_context=rel_ctx_ctx,
            summarize_contexts=self.use_summarizer,
        )

        results = self._normalize_results(pipeline_obj.score())

        if output_file:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)
            print(f"[FactualityRunner] Results written to: {output_file}")

        return results, pipeline_obj

    def assess_item_with_pipeline(
        self,
        item: dict[str, Any],
        *,
        retrieve_contexts: bool = False,
        release_clients: bool = False,
    ) -> tuple[dict[str, Any], Any]:
        """Assess one dataset item and also return the assessor instance.

        The per-item body of :meth:`assess_file`: the item is expected to already
        carry atoms and contexts (as ``FactReasoner.to_json`` writes them), so
        nothing is atomized or retrieved -- only the NLI relations are scored and
        inference is run. Exposed separately so a caller can score a single item
        without an output file, and so a second stage can reuse the atoms (see
        :meth:`assess_with_pipeline`).

        Args:
            item: The dataset item (with ``atoms`` and ``contexts``).
            retrieve_contexts: Ignore the item's stored contexts and retrieve
                new ones for its atoms (and the query) with this runner's
                retriever. Its atoms and gold labels are kept.
            release_clients: Close this call's HTTP clients on the LLM backends
                before its event loop ends (see :func:`_release_loop_clients`).
                :meth:`assess_file` sets it when items run concurrently.

        Returns:
            ``(results, pipeline)``. ``results["timing"]`` holds the pipeline's
            stage timings and the time ``score()`` took.
        """
        _rel_atom_ctx, rel_ctx_ctx, remove_dups, ctx_per_atom = _FR_VERSIONS[
            self.pipeline_version
        ]

        context_retriever = self._build_context_retriever()
        pipeline_obj = self._make_pipeline(context_retriever)
        if retrieve_contexts:
            # Keep the item's atoms and gold labels; drop its stored contexts
            # so build() retrieves fresh ones with this runner's retriever.
            item = {
                **item,
                "atoms": [{**a, "contexts": []} for a in item["atoms"]],
                "contexts": [],
            }
        pipeline_obj.from_dict_with_contexts(item)

        build_kwargs = {
            "has_atoms": True,
            "has_contexts": not retrieve_contexts,
            "revise_atoms": False,
        }
        if self.pipeline == "factreasoner":
            build_kwargs.update(
                remove_duplicates=remove_dups,
                contexts_per_atom_only=ctx_per_atom,
                rel_atom_context=True,
                rel_context_context=rel_ctx_ctx,
                summarize_contexts=self.use_summarizer,
            )
        release = tuple(self._llm_backends) if release_clients else ()
        _run_build(pipeline_obj, release_clients_of=release, **build_kwargs)

        score_start = time.perf_counter()
        results = self._normalize_results(pipeline_obj.score())
        timing = dict(getattr(pipeline_obj, "timing", None) or {})
        # score() runs Merlin (FactReasoner) or labels atoms (baselines).
        timing["score"] = time.perf_counter() - score_start
        results.setdefault("timing", timing)
        return results, pipeline_obj

    def assess_file(
        self,
        input_file: str,
        output_dir: str,
        *,
        dataset_name: str | None = None,
        model_id: str | None = None,
        retrieve_contexts: bool = False,
    ) -> list[dict[str, Any]]:
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
            retrieve_contexts: Ignore the items' stored contexts and retrieve
                new ones for their atoms (and the query) with this runner's
                ``service_type`` retriever. The atoms and their gold labels
                are kept, so accuracy is still measured. The output filename
                gains a ``-retrieved`` suffix, so it never resumes from, or
                overwrites, a run on the stored contexts.

        Returns:
            The list of result dictionaries.
        """
        pipeline_name = (
            f"factreasoner-{self.pipeline_version}"
            if self.pipeline == "factreasoner"
            else self.pipeline
        )
        # Load the dataset (one JSON object per line).
        with open(input_file) as f:
            dataset = [json.loads(line) for line in f.read().splitlines() if line]
        print(f"[FactualityRunner] Loaded {len(dataset)} items from {input_file}")

        os.makedirs(output_dir, exist_ok=True)
        suffix = "-retrieved" if retrieve_contexts else ""
        out_name = (
            f"eval_{pipeline_name}_{self.service_type}{suffix}_{dataset_name}_{model_id}.jsonl"
        )
        output_filename = os.path.join(output_dir, out_name)

        # Resume: load any previously computed results and skip their inputs.
        evaluation_data: list[dict[str, Any]] = []
        if os.path.isfile(output_filename):
            with open(output_filename, "r") as f:
                evaluation_data = [json.loads(line) for line in f if line.strip()]
        done_inputs = {e.get("input") for e in evaluation_data}
        print(f"[FactualityRunner] Found {len(evaluation_data)} existing results")

        todo = [d for d in dataset if d.get("input") not in done_inputs]
        if len(todo) < len(dataset):
            print(
                f"[FactualityRunner] Skipping {len(dataset) - len(todo)} "
                "already-processed inputs."
            )

        def assess_item(input_data: dict[str, Any]) -> dict[str, Any]:
            results, _pipeline = self.assess_item_with_pipeline(
                input_data,
                retrieve_contexts=retrieve_contexts,
                release_clients=self.item_concurrency > 1,
            )
            results["model_name"] = model_id
            return results

        run_start = time.monotonic()

        def save(results: dict[str, Any]) -> None:
            evaluation_data.append(results)
            # Write incrementally so a crash keeps completed work.
            with open(output_filename, "w") as f:
                f.writelines(f"{json.dumps(res)}\n" for res in evaluation_data)

        if self.item_concurrency == 1:
            for input_data in todo:
                save(assess_item(input_data))
        else:
            # Items in flight at once keep every server busy: while one item's
            # atoms are being revised on the LLM, another's relations are scored
            # by the NLI server and a third waits on retrieval. Each item has its
            # own pipeline object and event loop (asyncio.run in its thread);
            # the backends' shared throttles bound the total load. Results are
            # saved in input order, so a resumed run skips a prefix as before.
            with ThreadPoolExecutor(
                max_workers=self.item_concurrency, thread_name_prefix="item"
            ) as pool:
                for results in pool.map(assess_item, todo):
                    save(results)

        wall_s = time.monotonic() - run_start
        print(f"[FactualityRunner] Results written to: {output_filename}")
        report = self.timing_report(evaluation_data[-len(todo):] if todo else [], wall_s)
        report_file = os.path.join(output_dir, f"timing_{os.path.splitext(out_name)[0]}.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        print(format_timing_report(report))
        print(f"[FactualityRunner] Timing report written to: {report_file}")
        return evaluation_data

    def timing_report(self, results: list[dict[str, Any]], wall_s: float) -> dict[str, Any]:
        """Where a run's time went: per server, on the run's own clock.

        Server rows come from the backends' shared throttles (see
        :meth:`fact_reasoner.utils.BackendThrottle.stats`), which see every
        request, so concurrent items are not double-counted. Stage sums add each
        item's own stage time and do double-count overlapping items; they show
        the per-item mix, not wall time.

        Args:
            results: The results of the items this call assessed.
            wall_s: Wall time of the call, from the first item to the last.
        """
        servers: dict[str, Any] = {}
        throttles: dict[str, Any] = {}
        roles = [("llm", self.backend), ("nli", getattr(self.nli_extractor, "backend", None))]
        seen: set[int] = set()
        for role, backend in roles:
            if backend is None or id(backend) in seen:
                continue
            seen.add(id(backend))
            throttle = _existing_throttle(backend)
            if throttle is None:
                continue
            throttles[role] = throttle
            servers[role] = {"backend": str(backend), **throttle.stats()}
        nli_pairs = sum(
            r.get("nli_stats", {}).get("totals", {}).get("llm_calls", 0) for r in results
        )
        if "nli" in servers:
            servers["nli"]["pairs"] = nli_pairs
            busy = servers["nli"]["busy_s"]
            servers["nli"]["pairs_per_busy_s"] = nli_pairs / busy if busy else 0.0
        overlap = (
            busy_overlap_s(throttles["llm"], throttles["nli"])
            if {"llm", "nli"} <= throttles.keys()
            else None
        )
        stage_sums: dict[str, float] = {}
        for r in results:
            for name, seconds in (r.get("timing") or {}).items():
                stage_sums[name] = stage_sums.get(name, 0.0) + seconds
        return {
            "items": len(results),
            "item_concurrency": self.item_concurrency,
            "wall_s": wall_s,
            "s_per_item": wall_s / len(results) if results else 0.0,
            "nli_pairs": nli_pairs,
            "servers": servers,
            "llm_nli_overlap_s": overlap,
            "item_elapsed_sum_s": sum(r.get("elapsed_time", 0.0) for r in results),
            "stage_sums_s": stage_sums,
        }
