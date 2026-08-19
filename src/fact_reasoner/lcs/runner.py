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

# Unified logical-coherence runner.
#
# A single class that runs the logical-coherence pipeline with any Mellea backend,
# over either a single query+response or a jsonl dataset of pre-annotated
# responses. The factuality counterpart is ``fact_reasoner.runner.FactualityRunner``,
# whose shape this mirrors.
#
# ``CoherencePipeline`` (lcs.pipeline) already orchestrates the three *stages* --
# priors, mining, readout -- but it owns no backend and builds neither the
# ``RelationMiner`` nor the prior provider. This class is the layer above it: it
# owns the backend and the expensive backend-bound components, constructs the
# factuality stage that supplies each atom's prior, and adds the resumable dataset
# sweep the coherence side did not have.
#
# By default the atom priors ARE the FactReasoner posterior marginals
# (``prior_source="factreasoner"``), so the score reflects external support as well
# as internal coherence; ``"none"`` scores coherence alone at a flat 0.5.

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from mellea.backends import Backend

from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.lcs.candidate_pairs import GATE_METHODS, PAIR_POLICIES
from fact_reasoner.lcs.lcs_scorer import LCS_METHODS
from fact_reasoner.lcs.pipeline import (
    COHERENCE_FORMULATIONS,
    CoherencePipeline,
    CoherenceResult,
    build_coherence_model,
)
from fact_reasoner.lcs.priors import AtomPriors, PriorProvider
from fact_reasoner.lcs.relation_miner import RelationMiner
from fact_reasoner.lcs.strength import StrengthCalibrator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fact_reasoner.runner import FactualityRunner

# Where each atom's unary prior comes from. "factreasoner" is the default: the
# priors are the factuality pipeline's posterior marginals (the two-stage model).
COHERENCE_PRIOR_SOURCES = ("factreasoner", "none", "file")

# The backend built by :meth:`CoherenceRunner.from_backend_kind` when the caller
# does not name one. RITS exposes logprobs, which the default relation-strength
# estimator needs.
DEFAULT_BACKEND_KIND = "rits"


def atom_texts_from_item(item: Mapping[str, Any]) -> list[str] | None:
    """Extract the atom texts a dataset item carries, if any.

    Items written by ``FactReasoner.to_json`` (and the ``data/lcs/*.json``
    fixtures) carry an ``atoms`` list of ``{"text": ...}`` objects; bare strings
    are also accepted. Mining those directly avoids atomizing the response again.

    Args:
        item: The dataset item.

    Returns:
        The atom texts, or None when the item carries no usable ``atoms`` list.
    """
    atoms = item.get("atoms")
    if not isinstance(atoms, list) or not atoms:
        return None
    texts = [
        a["text"] if isinstance(a, dict) else str(a)
        for a in atoms
        if not isinstance(a, dict) or a.get("text")
    ]
    return texts or None


class CoherenceRunner:
    """Run the logical-coherence pipeline over a single item or a dataset.

    The runner owns a Mellea backend and the shared components (the atomizer, the
    relation miner, the coherence model), builds the factuality stage that supplies
    each atom's prior, and exposes three entry points:

    * :meth:`assess` — score a single ``query`` / ``response`` pair, atomizing the
      response and (with ``prior_source="factreasoner"``) retrieving contexts for
      the factuality stage.
    * :meth:`assess_item` — score one dataset item that already carries atoms and
      contexts, so nothing is atomized or retrieved.
    * :meth:`assess_file` — sweep a jsonl dataset of such items, writing results
      incrementally and skipping already-processed inputs.

    Example::

        from fact_reasoner.lcs import CoherenceRunner

        runner = CoherenceRunner.from_backend_kind("rits", merlin_path=merlin_path)
        runner.assess("Who was Marie Curie?", response).describe()

    Args:
        backend: The Mellea backend that drives all components.
        merlin_path: Path to the Merlin inference engine. Required — both the
            coherence MRF and the factuality stage are solved with it.
        methods: Which LCS readouts to compute; the first is the headline. Several
            readouts share the base inference runs, so asking for all four costs
            7 Merlin calls rather than 13.
        formulation: Which coherence model to score with — ``"mrf"`` (default) or
            ``"mln"`` (the research branch, which raises when scored).
        reified_prior: Bernoulli prior on the reified coherence node (only used by
            the ``"reified"`` readout).
        ibound: Merlin weighted-mini-bucket i-bound.
        on_low_coverage: What to do when few atoms carry a real prior —
            ``"warn"``, ``"raise"`` or ``"uniform"``.
        prior_source: Where the atom priors come from. ``"factreasoner"`` (default)
            runs the factuality pipeline and uses its posterior marginals, reusing
            that run's atoms so the response is atomized once; ``"none"`` uses a
            flat 0.5, i.e. coherence only; ``"file"`` loads them from a saved
            FactReasoner results JSON (no LLM calls).
        prior_provider: An explicit prior source, which overrides
            ``prior_source``. Accepts anything
            :func:`~fact_reasoner.lcs.priors.coerce_prior_provider` takes — a
            :class:`~fact_reasoner.lcs.priors.PriorProvider`, an
            :class:`~fact_reasoner.lcs.priors.AtomPriors`, a
            ``{atom_id: probability}`` mapping, or a float.
        priors_file: The results JSON for ``prior_source="file"``.
        on_degraded: ``"fallback"`` (default) to score with uniform priors when the
            factuality run yields no marginals, or ``"raise"`` to fail loudly.
        factuality_runner: A pre-built
            :class:`~fact_reasoner.runner.FactualityRunner` to use for the priors
            instead of one constructed from the factuality arguments below.
        pipeline_version: FactReasoner graph shape (``v1``/``v2``/``v3``).
        service_type: Retrieval service for the factuality stage.
        cache_dir: Retriever cache directory.
        top_k: Top-k contexts retrieved per atom.
        num_workers: Parallelism for context retrieval.
        use_summarizer: Summarize retrieved contexts in the factuality stage.
        use_query_builder: Use the QueryBuilder for search queries.
        nli_mode: NLI candidate-pair preset for the factuality stage. Defaults to
            ``"fast"`` rather than ``"all_pairs"``: the factuality run is here to
            supply priors, and the provenance preset is far cheaper for the same
            graph semantics.
        nli_cache_dir: Cross-run NLI verdict cache for the factuality stage.
        nli_method: How relation probabilities are estimated, in *both* stages —
            ``"logprobs"`` needs a logprobs-capable backend (RITS/vLLM/OpenAI), and
            ``"simbauq"`` works on any backend (required for Ollama, and for Claude
            via Anthropic's OpenAI-compatibility endpoint, neither of which returns
            usable logprobs).
        nli_similarity_metric: Similarity metric for the SIMBA-UQ NLI method.
        nli_confidence_method: Confidence method for the SIMBA-UQ NLI method.
        nli_classifier_path: Classifier path for the SIMBA-UQ NLI method.
        strength_method: How the conditional relation strength is estimated
            (``"auto"`` picks ``surrogate_logprobs`` when logprobs are available).
        strength_samples: Samples per edge for ``surrogate_sampled``.
        strength_calibrator: Optional calibrator applied to raw strengths.
        pair_policy: Candidate atom-pair policy. ``"all_pairs"`` is quadratic in
            atoms and over-connects long responses.
        window: Order-window radius for the windowed policy.
        gate: Long-range gate for the gated policy.
        gate_threshold: Similarity threshold for the gate.
        concession_discount: Discount applied to a conflict edge that a concession
            resolves.
        revise_atoms: Decontextualize atoms before mining.
        show_progress: Show progress bars.

    Raises:
        ValueError: If ``merlin_path`` is missing, or any of ``methods``,
            ``formulation``, ``prior_source``, ``pair_policy`` or ``gate`` is
            invalid, or ``prior_source="file"`` is selected without a
            ``priors_file``.
    """

    def __init__(
        self,
        backend: Backend,
        *,
        merlin_path: str,
        # --- coherence scoring ---
        methods: Sequence[str] = ("mean_marginal",),
        formulation: str = "mrf",
        reified_prior: float = 0.5,
        ibound: int = 6,
        on_low_coverage: str = "warn",
        # --- atom priors (the factuality stage) ---
        prior_source: str = "factreasoner",
        prior_provider: PriorProvider
        | AtomPriors
        | Mapping[str, float]
        | float
        | None = None,
        priors_file: str | None = None,
        on_degraded: str = "fallback",
        factuality_runner: FactualityRunner | None = None,
        pipeline_version: str = "v2",
        service_type: str = "google",
        cache_dir: str | None = None,
        top_k: int = 3,
        num_workers: int = 4,
        use_summarizer: bool = False,
        use_query_builder: bool = False,
        nli_mode: str = "fast",
        nli_cache_dir: str | None = None,
        # --- relation mining ---
        nli_method: str = "logprobs",
        nli_similarity_metric: str = "rouge",
        nli_confidence_method: str = "aggregation",
        nli_classifier_path: str | None = None,
        strength_method: str = "auto",
        strength_samples: int = 8,
        strength_calibrator: StrengthCalibrator | None = None,
        pair_policy: str = "windowed",
        window: int = 4,
        gate: str = "embedding",
        gate_threshold: float = 0.3,
        concession_discount: float = 0.45,
        revise_atoms: bool = False,
        show_progress: bool = False,
    ) -> None:
        """Initialize the runner and its shared components."""
        if not merlin_path:
            raise ValueError(
                "CoherenceRunner requires a merlin_path (the coherence MRF is "
                "solved with Merlin)."
            )
        methods = tuple(methods)
        if not methods:
            raise ValueError("methods must name at least one LCS readout.")
        unknown = [m for m in methods if m not in LCS_METHODS]
        if unknown:
            raise ValueError(
                f"Unknown LCS method(s): {unknown!r} "
                f"(expected from {list(LCS_METHODS)})."
            )
        if formulation not in COHERENCE_FORMULATIONS:
            raise ValueError(
                f"Unknown formulation: {formulation!r} "
                f"(expected one of {list(COHERENCE_FORMULATIONS)})."
            )
        if prior_source not in COHERENCE_PRIOR_SOURCES:
            raise ValueError(
                f"Unknown prior_source: {prior_source!r} "
                f"(expected one of {list(COHERENCE_PRIOR_SOURCES)})."
            )
        if prior_source == "file" and not priors_file and prior_provider is None:
            raise ValueError("prior_source='file' requires priors_file=<path>.")
        if pair_policy not in PAIR_POLICIES:
            raise ValueError(
                f"Unknown pair_policy: {pair_policy!r} "
                f"(expected one of {list(PAIR_POLICIES)})."
            )
        if gate not in GATE_METHODS:
            raise ValueError(
                f"Unknown gate: {gate!r} (expected one of {list(GATE_METHODS)})."
            )

        self.backend = backend
        self.merlin_path = merlin_path
        self.methods = methods
        self.formulation = formulation
        self.reified_prior = reified_prior
        self.ibound = ibound
        self.on_low_coverage = on_low_coverage

        self.prior_source = prior_source
        self.prior_provider = prior_provider
        self.priors_file = priors_file
        self.on_degraded = on_degraded
        self.pipeline_version = pipeline_version
        self.service_type = service_type
        self.cache_dir = cache_dir
        self.top_k = top_k
        self.num_workers = num_workers
        self.use_summarizer = use_summarizer
        self.use_query_builder = use_query_builder
        self.nli_mode = nli_mode
        self.nli_cache_dir = nli_cache_dir

        self.nli_method = nli_method
        self.nli_similarity_metric = nli_similarity_metric
        self.nli_confidence_method = nli_confidence_method
        self.nli_classifier_path = nli_classifier_path
        self.revise_atoms = revise_atoms
        self.show_progress = show_progress

        # Shared components. These are backend-bound and cost real work to build,
        # so they are built once here; the pipeline itself is built per call.
        self.atomizer = Atomizer(backend)
        self.reviser = Reviser(backend) if revise_atoms else None
        self.miner = RelationMiner(
            backend,
            nli_method=nli_method,
            atomizer=self.atomizer,
            reviser=self.reviser,
            pair_policy=pair_policy,
            window=window,
            gate=gate,
            gate_threshold=gate_threshold,
            concession_discount=concession_discount,
            strength_method=strength_method,
            strength_samples=strength_samples,
            strength_calibrator=strength_calibrator,
            show_progress=show_progress,
        )
        # Built here rather than left to CoherencePipeline, which constructs its
        # default model without an i-bound -- passing a prebuilt model is the only
        # way `ibound` reaches the scorer.
        self.coherence_model = build_coherence_model(
            formulation, merlin_path=merlin_path, ibound=ibound
        )

        # The factuality stage is built lazily: coherence-only runs never need it,
        # and building it eagerly would construct a retriever nobody uses.
        self._factuality_runner = factuality_runner

    @classmethod
    def from_backend_kind(
        cls,
        kind: str = DEFAULT_BACKEND_KIND,
        *,
        model_id: Any | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model_options: dict[Any, Any] | None = None,
        **kwargs: Any,
    ) -> CoherenceRunner:
        """Build a runner on a freshly constructed backend.

        A convenience constructor for callers that do not already hold a backend.
        Every kind :func:`~fact_reasoner.backends.build_backend` supports works;
        the default is RITS, which exposes the logprobs the default NLI and
        relation-strength estimators read.

        Args:
            kind: Backend to build — ``"rits"`` (default), ``"ollama"``, ``"vllm"``
                or ``"openai"``. With ``"ollama"`` (and Claude via the OpenAI
                compatibility endpoint) pass ``nli_method="simbauq"``, since
                neither returns usable logprobs.
            model_id: Model identifier, resolved per backend by
                :mod:`fact_reasoner.models`. The backend default is used when
                omitted.
            base_url: API endpoint (see :func:`build_backend`).
            api_key: API key; normally left unset so the backend falls back to its
                environment variable.
            model_options: Extra Mellea model options.
            **kwargs: Passed to :class:`CoherenceRunner` (``merlin_path`` is
                required).

        Returns:
            The runner.
        """
        from fact_reasoner.backends import build_backend

        backend = build_backend(
            kind,
            model_id=model_id,
            base_url=base_url,
            api_key=api_key,
            model_options=model_options,
        )
        return cls(backend, **kwargs)

    # -- construction helpers ------------------------------------------------

    def _build_factuality_runner(self) -> FactualityRunner:
        """The factuality stage that supplies the atom priors (built once).

        Imported lazily so ``import fact_reasoner.lcs`` stays cheap for
        coherence-only callers, and memoized so a dataset sweep does not rebuild
        the assessor components per item.
        """
        if self._factuality_runner is None:
            from fact_reasoner.runner import FactualityRunner

            self._factuality_runner = FactualityRunner(
                self.backend,
                pipeline="factreasoner",
                pipeline_version=self.pipeline_version,
                service_type=self.service_type,
                cache_dir=self.cache_dir,
                top_k=self.top_k,
                num_workers=self.num_workers,
                use_priors=True,
                use_summarizer=self.use_summarizer,
                use_query_builder=self.use_query_builder,
                merlin_path=self.merlin_path,
                nli_method=self.nli_method,
                nli_similarity_metric=self.nli_similarity_metric,
                nli_confidence_method=self.nli_confidence_method,
                nli_classifier_path=self.nli_classifier_path,
                nli_mode=self.nli_mode,
                nli_cache_dir=self.nli_cache_dir,
                show_progress=self.show_progress,
            )
        return self._factuality_runner

    def _build_prior_provider(self, item: Mapping[str, Any] | None = None):
        """The prior source for one call.

        Args:
            item: A dataset item that already carries atoms and contexts. When
                given, the factuality stage runs in ``"file_item"`` mode, so it
                scores the NLI relations and infers without retrieving anything.

        Returns:
            A :class:`~fact_reasoner.lcs.priors.PriorProvider`, or whatever the
            caller passed as ``prior_provider``, or None for uniform priors.
        """
        if self.prior_provider is not None:
            # An explicit source wins; CoherencePipeline coerces any accepted form.
            return self.prior_provider
        if self.prior_source == "none":
            return None
        if self.prior_source == "file":
            from fact_reasoner.lcs.priors import PrecomputedPriorProvider

            return PrecomputedPriorProvider(self.priors_file)

        from fact_reasoner.lcs.priors import FactReasonerPriorProvider

        if item is not None:
            return FactReasonerPriorProvider(
                runner=self._build_factuality_runner(),
                mode="file_item",
                item=item,
                on_degraded=self.on_degraded,
            )
        return FactReasonerPriorProvider(
            runner=self._build_factuality_runner(),
            mode="assess",
            on_degraded=self.on_degraded,
        )

    def _make_pipeline(
        self, item: Mapping[str, Any] | None = None
    ) -> CoherencePipeline:
        """Construct the coherence pipeline with the shared components."""
        return CoherencePipeline(
            miner=self.miner,
            coherence_model=self.coherence_model,
            prior_provider=self._build_prior_provider(item),
            methods=self.methods,
            reified_prior=self.reified_prior,
            on_low_coverage=self.on_low_coverage,
        )

    def _write(self, result: CoherenceResult, output_file: str | None) -> None:
        """Write a result to ``output_file`` as JSON, when one was requested."""
        if not output_file:
            return
        with open(output_file, "w") as f:
            json.dump(result.to_json(), f, indent=2)
        print(f"[CoherenceRunner] Result written to: {output_file}")

    # -- single-item entry points --------------------------------------------

    def assess(
        self,
        query: str,
        response: str,
        topic: str | None = None,
        output_file: str | None = None,
        atom_texts: Sequence[str] | None = None,
    ) -> CoherenceResult:
        """Score a single response's logical coherence.

        The response is atomized from scratch unless ``atom_texts`` is given. With
        ``prior_source="factreasoner"`` the factuality stage also retrieves
        contexts and scores each atom, and its posterior marginals become the
        coherence MRF's per-atom priors — so a working retriever is required.

        Args:
            query: The query the response answers.
            response: The response to score.
            topic: Optional topic hint.
            output_file: If set, write the result to this path as JSON.
            atom_texts: Pre-extracted atoms to mine instead of atomizing the
                response (as ``data/lcs/*.json`` carries). The priors still come
                from ``prior_source``, so a factuality stage runs over the response
                as usual.

        Returns:
            The :class:`~fact_reasoner.lcs.pipeline.CoherenceResult`.
        """
        result, _pipeline = self.assess_with_pipeline(
            query,
            response,
            topic=topic,
            output_file=output_file,
            atom_texts=atom_texts,
        )
        return result

    def assess_with_pipeline(
        self,
        query: str,
        response: str,
        topic: str | None = None,
        output_file: str | None = None,
        atom_texts: Sequence[str] | None = None,
    ) -> tuple[CoherenceResult, CoherencePipeline]:
        """Score a single response and also return the pipeline instance.

        Identical work to :meth:`assess`; the extra return value exposes the
        pipeline, whose prior provider (and hence the factuality run behind it)
        outlives the call.

        Args:
            query: The query the response answers.
            response: The response to score.
            topic: Optional topic hint.
            output_file: If set, write the result to this path as JSON.
            atom_texts: Pre-extracted atoms to mine instead of atomizing.

        Returns:
            ``(result, pipeline)``.
        """
        pipeline = self._make_pipeline()
        if atom_texts:
            # Mine the given atoms, but still take the priors from the configured
            # source -- which for "factreasoner" means a full factuality run over
            # the response, with its own atoms aligned onto these by text.
            mining = self.miner.mine_from_atoms(list(atom_texts), response)
            atom_priors = pipeline.prior_provider.priors_for(
                response=response, query=query, topic=topic
            )
            result = pipeline.run_from_mining(mining, priors=atom_priors)
        else:
            result = pipeline.run(response, query=query, topic=topic)
        self._write(result, output_file)
        return result, pipeline

    async def aassess(
        self,
        query: str,
        response: str,
        topic: str | None = None,
        output_file: str | None = None,
        atom_texts: Sequence[str] | None = None,
    ) -> CoherenceResult:
        """Async variant of :meth:`assess` — the per-pair mining calls overlap.

        The two stages still run in sequence, since mining reuses the factuality
        run's atoms; only the mining calls within stage 2 are concurrent.

        Args:
            query: The query the response answers.
            response: The response to score.
            topic: Optional topic hint.
            output_file: If set, write the result to this path as JSON.
            atom_texts: Pre-extracted atoms to mine instead of atomizing.

        Returns:
            The :class:`~fact_reasoner.lcs.pipeline.CoherenceResult`.
        """
        pipeline = self._make_pipeline()
        if atom_texts:
            mining = await self.miner.amine_from_atoms(list(atom_texts), response)
            atom_priors = pipeline.prior_provider.priors_for(
                response=response, query=query, topic=topic
            )
            result = pipeline.run_from_mining(mining, priors=atom_priors)
        else:
            result = await pipeline.arun(response, query=query, topic=topic)
        self._write(result, output_file)
        return result

    def assess_item(
        self, item: Mapping[str, Any], output_file: str | None = None
    ) -> CoherenceResult:
        """Score one pre-annotated dataset item.

        Args:
            item: The dataset item (with ``atoms``, and ``contexts`` for the
                factuality stage).
            output_file: If set, write the result to this path as JSON.

        Returns:
            The :class:`~fact_reasoner.lcs.pipeline.CoherenceResult`.
        """
        result, _pipeline = self.assess_item_with_pipeline(
            item, output_file=output_file
        )
        return result

    def assess_item_with_pipeline(
        self, item: Mapping[str, Any], output_file: str | None = None
    ) -> tuple[CoherenceResult, CoherencePipeline]:
        """Score one dataset item and also return the pipeline instance.

        The per-item body of :meth:`assess_file`. The item is expected to already
        carry atoms and contexts (as ``FactReasoner.to_json`` writes them), so
        nothing is atomized and nothing is retrieved: the factuality stage runs in
        ``"file_item"`` mode, and when the item carries atom texts those are mined
        directly rather than re-atomized.

        Args:
            item: The dataset item.
            output_file: If set, write the result to this path as JSON.

        Returns:
            ``(result, pipeline)``.

        Raises:
            ValueError: If the item carries no response text.
        """
        response = item.get("output") or item.get("response")
        if not response or not str(response).strip():
            raise ValueError("Dataset item has no 'output'/'response' text to score.")
        query = item.get("input")
        topic = item.get("topic")

        pipeline = self._make_pipeline(item)
        atom_texts = atom_texts_from_item(item)
        if atom_texts:
            # The item already carries atoms: mine those, then score them under
            # whatever priors the provider yields for this item.
            mining = self.miner.mine_from_atoms(atom_texts, response)
            atom_priors = pipeline.prior_provider.priors_for(
                response=response, query=query, topic=topic
            )
            result = pipeline.run_from_mining(mining, priors=atom_priors)
        else:
            result = pipeline.run(response, query=query, topic=topic)

        self._write(result, output_file)
        return result, pipeline

    # -- dataset entry point -------------------------------------------------

    def assess_file(
        self,
        input_file: str,
        output_dir: str,
        *,
        dataset_name: str | None = None,
        model_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Score a jsonl dataset of pre-annotated responses.

        Each item is expected to already contain atoms and contexts (as produced
        by ``FactReasoner.to_json``). Results are written incrementally to a jsonl
        file in ``output_dir``, and inputs already present in that file are
        skipped, so the run is resumable.

        The output filename records the formulation and the prior source, so a
        priors ablation over one dataset does not overwrite itself.

        Args:
            input_file: Path to the input jsonl dataset.
            output_dir: Directory for the output jsonl.
            dataset_name: Dataset label (used in the output filename).
            model_id: Model label recorded in each result and the filename.

        Returns:
            The list of result dictionaries.
        """
        with open(input_file) as f:
            dataset = [json.loads(line) for line in f.read().splitlines() if line]
        print(f"[CoherenceRunner] Loaded {len(dataset)} items from {input_file}")

        os.makedirs(output_dir, exist_ok=True)
        prior_tag = (
            "factreasoner" if self.prior_provider is not None else (self.prior_source)
        )
        out_name = f"lcs_{self.formulation}_{prior_tag}_{dataset_name}_{model_id}.jsonl"
        output_filename = os.path.join(output_dir, out_name)

        # Resume: load any previously computed results and skip their inputs.
        evaluation_data: list[dict[str, Any]] = []
        if os.path.isfile(output_filename):
            with open(output_filename, "r") as f:
                evaluation_data = [json.loads(line) for line in f if line.strip()]
        done_inputs = {e.get("input") for e in evaluation_data}
        print(f"[CoherenceRunner] Found {len(evaluation_data)} existing results")

        for input_data in dataset:
            if input_data.get("input") in done_inputs:
                print("[CoherenceRunner] Skipping already-processed input.")
                continue

            result, _pipeline = self.assess_item_with_pipeline(input_data)
            record = result.to_json()
            record["input"] = input_data.get("input")
            record["output"] = input_data.get("output") or input_data.get("response")
            record["topic"] = input_data.get("topic")
            record["model_name"] = model_id
            evaluation_data.append(record)

            # Write incrementally so a crash keeps completed work.
            with open(output_filename, "w") as f:
                f.writelines(f"{json.dumps(res)}\n" for res in evaluation_data)

        print(f"[CoherenceRunner] Results written to: {output_filename}")
        return evaluation_data


__all__ = [
    "COHERENCE_PRIOR_SOURCES",
    "DEFAULT_BACKEND_KIND",
    "CoherenceRunner",
    "atom_texts_from_item",
]
