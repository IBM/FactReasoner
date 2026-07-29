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

from typing import Dict, List, Optional, Tuple, Union

import asyncio
import time
import nltk
from nltk.tokenize import sent_tokenize
import concurrent.futures


# Local imports
from .base import Atom, Context, Relation, PRIOR_PROB_ATOM, PRIOR_PROB_CONTEXT
from .atomizer import Atomizer
from .retriever import ContextRetriever
from .nli import NLIExtractor
from . import nli_pairs as _np
from .nli_cache import extractor_identity
from .nli_config import FAITHFUL, NLIPairConfig
from fact_reasoner.utils import punctuation_only_inside_quotes


def predict_nli_relationships(
    object_pairs: List[Tuple[Union[Atom, Context], Union[Atom, Context]]],
    nli_extractor: NLIExtractor,
    links_type: str = "context_atom",
    use_summary: bool = False,
    cache=None,
    stats: Optional[dict] = None,
) -> List[Relation]:
    """
    Predict the NLI relationship between two objects using an model based NLI extractor.

    Args:
        object_pairs: List
            A list of object pairs e.g., (atom, context) or (context, context)
        nli_extractor: NLIExtractor
            The model based NLI extractor
        links_type: str
            The type of links represented by the object pairs (context_atom, context_context).
        use_summary: bool
            Use the objects' summaries rather than their full text as the NLI
            premise/hypothesis.
        cache: NLIVerdictCache, optional
            A cross-run verdict cache. Cached pairs are served without an LLM call;
            misses are computed and stored. Verdicts are unchanged either way, so
            this only affects cost.
        stats: dict, optional
            Out-param; ``llm_calls`` and ``cache_hits`` are added when provided.

    Returns:
        A list of Relations, positionally aligned with ``object_pairs``.
    """

    # Safety checks
    assert nli_extractor is not None, "NLI extractor cannot be None."
    assert isinstance(nli_extractor, NLIExtractor), (
        "NLI extractor must be NLIExtractor."
    )

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

    # Serve whatever the cache already knows; only misses reach the model. The
    # verdicts are identical either way, so this changes cost and not results.
    results: List[Optional[dict]] = [None] * len(premises)
    cache_hits = 0
    todo = list(range(len(premises)))
    if cache is not None and premises:
        model_id, nli_method = extractor_identity(nli_extractor)
        keys = [
            cache.make_key(model_id, nli_method, premises[i], hypotheses[i])
            for i in range(len(premises))
        ]
        cached = cache.get_many(keys)
        todo = []
        for i, key in enumerate(keys):
            hit = cached.get(key)
            if hit is None:
                todo.append(i)
            else:
                results[i] = hit
                cache_hits += 1
        if cache_hits:
            print(f"[NLI] Cache hits: {cache_hits}/{len(premises)}")

    if todo:
        sub_premises = [premises[i] for i in todo]
        sub_hypotheses = [hypotheses[i] for i in todo]
        try:
            # If an event loop is already running, we cannot call asyncio.run()
            # directly, so run the batch in a separate thread with its own loop.
            asyncio.get_running_loop()

            with concurrent.futures.ThreadPoolExecutor() as pool:
                computed = pool.submit(
                    asyncio.run,
                    nli_extractor.run_batch(sub_premises, sub_hypotheses),
                ).result()
        except RuntimeError:
            # No running loop -- safe to run directly.
            computed = asyncio.run(
                nli_extractor.run_batch(sub_premises, sub_hypotheses)
            )
        for slot, result in zip(todo, computed):
            results[slot] = result
        if cache is not None:
            cache.put_many(
                [
                    (keys[slot], results[slot])
                    for slot in todo
                    if results[slot] is not None
                ]
            )

    if stats is not None:
        stats["llm_calls"] = stats.get("llm_calls", 0) + len(todo)
        stats["cache_hits"] = stats.get("cache_hits", 0) + cache_hits

    relations = []
    for ii, result in enumerate(results):
        # A slot can still be unfilled if the extractor returned a short batch;
        # fall back to the extractor's own neutral verdict rather than crashing.
        if result is None:
            result = {"label": "neutral", "probability": 1.0}
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

    assert response is not None and len(response) > 0, (
        "Please ensure a non empty response."
    )

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
    assert retriever is not None, (
        "Please ensure an existing context retriever instance."
    )

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
            Whether to compare the summaries of the contexts rather than their text.
            Contexts with no summary fall back to their text, so distinct
            unsummarized contexts are not treated as duplicates of one another.

    Returns:
        The updated dicts containing the contexts and atoms.
    """

    # Every atom that retrieved each context, so collapsing a duplicate can
    # transfer its owners to the survivor instead of discarding them.
    owners = _np.context_owners(atoms, contexts)

    seen: Dict[str, str] = {}  # comparison text -> surviving context id
    out = {}
    for k, v in contexts.items():
        # get_summary() returns "" (not None) when a context was never summarized,
        # so fall back on emptiness rather than on None -- otherwise every
        # unsummarized context compares equal to every other.
        text = v.get_summary() if check_summary else ""
        if not text:
            text = v.get_text()

        if text not in seen:
            seen[text] = k
            out[k] = v
            continue

        # Duplicate: repoint each of its owners at the survivor so no atom loses
        # evidence, then drop the duplicate from every atom that referenced it.
        survivor_id = seen[text]
        survivor = out[survivor_id]
        for atom_id in owners.get(k, ()):
            atom = atoms.get(atom_id)
            if atom is None:
                continue
            atom.contexts.pop(k, None)
            atom.contexts.setdefault(survivor_id, survivor)
        for atom in atoms.values():
            atom.contexts.pop(k, None)

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


def _reconcile_ctx_pair(r1: Relation, r2: Relation) -> Relation:
    """Reconcile the two directional NLI relations of a context pair into one.

    ``r1`` = NLI(c_i, c_j) and ``r2`` = NLI(c_j, c_i) for an unordered pair.
    Reconciliation is by *meaning* first, so a confidently-neutral (or failed,
    i.e. neutral@1.0) direction can never hide a real entailment/contradiction in
    the other direction:

    * both ``entailment``     -> mark the stronger one ``equivalence`` (symmetric;
      orientation does not matter for the equivalence factor);
    * exactly one non-neutral -> keep that non-neutral direction;
    * both non-neutral        -> keep the higher-probability one (probabilities
      are only compared here, between two already non-neutral relations);
    * both ``neutral``        -> return the higher-probability neutral (the caller
      filters neutrals out, so this is effectively "drop").

    Note: for entailment/contradiction the returned relation's source/target
    orientation is the kept direction's, which is load-bearing downstream.
    """
    t1, t2 = r1.get_type(), r2.get_type()
    p1, p2 = r1.get_probability(), r2.get_probability()

    if t1 == "entailment" and t2 == "entailment":
        winner = r1 if p1 >= p2 else r2
        winner.type = "equivalence"
        return winner

    r1_neutral = t1 == "neutral"
    r2_neutral = t2 == "neutral"

    # Exactly one non-neutral: always keep the non-neutral relation.
    if r1_neutral and not r2_neutral:
        return r2
    if r2_neutral and not r1_neutral:
        return r1

    # Both non-neutral, or both neutral: keep the higher-probability one. When
    # both are neutral the caller drops it; when both are non-neutral this is a
    # same-status comparison (not neutral-vs-real), so probability is meaningful.
    return r1 if p1 >= p2 else r2


def build_relations(
    atoms: Dict[str, Atom] = {},
    contexts: Dict[str, Context] = {},
    contexts_per_atom_only: bool = False,
    rel_atom_context: bool = True,
    rel_context_context: bool = True,
    nli_extractor: NLIExtractor = None,
    use_summarized_contexts: bool = False,
    *,
    pair_config: Optional[NLIPairConfig] = None,
    stats: Optional[dict] = None,
    cache=None,
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
        pair_config: NLIPairConfig, optional
            Which candidate pairs to score, and how. Defaults to
            :data:`~fact_reasoner.core.nli_config.FAITHFUL`, which enumerates
            every pair exactly as the original implementation did.
        stats: dict, optional
            Out-param populated with per-phase pair counts, LLM call counts and
            timings, plus the ``all_pairs`` counterfactual, so a run reports its
            own saving without needing a separate baseline run.
        cache: NLIVerdictCache, optional
            Cross-run verdict cache. Score-neutral by construction.

    Returns:
        A list of Relations.
    """

    assert len(atoms) > 0, "The atoms must be initialized!"
    assert len(contexts) > 0, "The contexts must be initialized!"
    assert nli_extractor is not None, "The NLI extractor must exist!"

    cfg = pair_config if pair_config is not None else FAITHFUL
    relations = []

    num_atoms, num_contexts = len(atoms), len(contexts)
    phase_stats: Dict[str, dict] = {}

    # Build the similarity gate once and share it across phases: one embedding
    # pass covers both. Under the faithful config no gate is needed, so
    # sentence-transformers is never even imported.
    gate = gate_atom_ids = gate_context_ids = None
    if cfg.needs_gate:
        gate, gate_atom_ids, gate_context_ids = _np.build_gate(
            atoms,
            contexts,
            use_summary=use_summarized_contexts,
            embedding_model=cfg.embedding_model,
        )

    # ---- Phase 1: select the pairs both phases will score.
    ac_pairs: List[Tuple[Context, Atom]] = []
    ac_coverage: dict = {}
    if rel_atom_context:
        print("[NLI] Building atom-context relations...")
        pairs, ac_coverage = _np.select_atom_context_pairs(
            atoms,
            contexts,
            policy=cfg.policy,
            contexts_per_atom_only=contexts_per_atom_only,
            gate_threshold=cfg.gate_threshold,
            neighbor_window=cfg.neighbor_window,
            gate=gate,
            gate_atom_ids=gate_atom_ids,
            gate_context_ids=gate_context_ids,
        )
        print(
            f"[NLI] Atom-context pairs: {ac_coverage['pairs_selected']} selected, "
            f"{ac_coverage['pairs_pruned']} pruned "
            f"(of {ac_coverage['pairs_possible']} possible)."
        )
        # Resolve ids to objects here, so selection deals only in ids.
        ac_pairs = [
            (contexts[context_id], atoms[atom_id]) for context_id, atom_id in pairs
        ]

    cc_pairs1: List[Tuple[Context, Context]] = []
    cc_pairs2: List[Tuple[Context, Context]] = []
    cc_coverage: dict = {}
    if rel_context_context:
        print("[NLI] Building context-context relations...")
        pairs, cc_coverage = _np.select_context_context_pairs(
            contexts,
            policy=cfg.policy,
            gate_threshold=cfg.gate_threshold,
            gate=gate,
            gate_context_ids=gate_context_ids,
        )
        print(
            f"[NLI] Context-context pairs: {cc_coverage['pairs_selected']} selected, "
            f"{cc_coverage['pairs_pruned']} pruned "
            f"(of {cc_coverage['pairs_possible']} possible)."
        )
        cc_pairs1 = [(contexts[ci], contexts[cj]) for ci, cj in pairs]
        cc_pairs2 = [(contexts[cj], contexts[ci]) for ci, cj in pairs]

    # ---- Phase 2: score the atom-context pairs and the first context direction.
    # These do not depend on each other, so with merge_phases they go out as one
    # fan-out rather than draining one batch before starting the next. That is a
    # latency win only -- the call count is identical either way.
    _t = time.perf_counter()
    ac_rels: List[Relation] = []
    relations1: List[Relation] = []
    if cfg.merge_phases and ac_pairs and cc_pairs1:
        merged_stats: dict = {}
        merged = predict_nli_relationships(
            ac_pairs + cc_pairs1,
            nli_extractor=nli_extractor,
            links_type="context_atom",
            use_summary=use_summarized_contexts,
            cache=cache,
            stats=merged_stats,
        )
        ac_rels = merged[: len(ac_pairs)]
        # The merged batch is tagged with one link type, so relabel the
        # context-context slice to its real link before it reaches the graph.
        relations1 = []
        for rel in merged[len(ac_pairs) :]:
            rel.link = "context_context"
            relations1.append(rel)
        # Attribute the shared batch's cost proportionally to each phase.
        _split_merged_stats(
            merged_stats, ac_coverage, cc_coverage, len(ac_pairs), len(cc_pairs1)
        )
        stats1 = {}
    else:
        if ac_pairs:
            ac_rels = predict_nli_relationships(
                ac_pairs,
                nli_extractor=nli_extractor,
                links_type="context_atom",
                use_summary=use_summarized_contexts,
                cache=cache,
                stats=ac_coverage,
            )
        stats1 = {}
        if cc_pairs1:
            relations1 = predict_nli_relationships(
                cc_pairs1,
                nli_extractor=nli_extractor,
                links_type="context_context",
                use_summary=use_summarized_contexts,
                cache=cache,
                stats=stats1,
            )

    # Filter out the neutral atom-context relationships.
    if rel_atom_context:
        kept = 0
        for rel in ac_rels:
            if rel.get_type() != "neutral":
                print(f"[NLI] Found relation: {rel}")
                relations.append(rel)
                kept += 1
        ac_coverage["relations_kept"] = kept
        ac_coverage["neutral_dropped"] = len(ac_rels) - kept
        ac_coverage["seconds"] = round(time.perf_counter() - _t, 3)
        phase_stats["atom_context"] = ac_coverage

    # Phase 3 starts here, so the two phases' timings do not overlap even when
    # phase 2 scored both of their first batches together.
    _t_cc = time.perf_counter()

    # ---- Phase 3: the reverse context direction, which depends on phase 2.
    if rel_context_context:
        coverage = cc_coverage
        context_context_pairs2 = cc_pairs2

        # Get relationships (c_j, c_i). The reverse direction is only needed
        # where it can change the reconciled outcome, so the cascade scores a
        # subset and synthesizes the rest -- see _mirror_needed.
        stats2: dict = {}
        if cfg.ctx_ctx_single_direction_cascade:
            need = _mirror_needed(relations1)
            relations2_partial = (
                predict_nli_relationships(
                    [context_context_pairs2[i] for i in need],
                    nli_extractor=nli_extractor,
                    links_type="context_context",
                    use_summary=use_summarized_contexts,
                    cache=cache,
                    stats=stats2,
                )
                if need
                else []
            )
            relations2 = _synthesize_mirrors(relations1, need, relations2_partial)
            coverage["dir2_skipped"] = len(relations1) - len(need)
        else:
            relations2 = predict_nli_relationships(
                context_context_pairs2,
                nli_extractor=nli_extractor,
                links_type="context_context",
                use_summary=use_summarized_contexts,
                cache=cache,
                stats=stats2,
            )
            coverage["dir2_skipped"] = 0

        # Reconcile each pair's two directions by meaning (so a high-probability
        # neutral direction cannot hide a real entailment/contradiction in the
        # other direction), then keep the non-neutral results.
        relations_tmp = [
            _reconcile_ctx_pair(r1, r2) for r1, r2 in zip(relations1, relations2)
        ]
        assert len(relations_tmp) == len(relations1)  # safety checks

        kept = 0
        for rel in relations_tmp:
            if rel.get_type() != "neutral":
                print(f"[NLI] Found relation: {rel}")
                relations.append(rel)
                kept += 1

        # Under merge_phases the first direction was billed to `coverage` by
        # _split_merged_stats, so take whichever source holds it.
        coverage["llm_calls_dir1"] = stats1.get(
            "llm_calls", coverage.get("llm_calls", 0)
        )
        coverage["llm_calls_dir2"] = stats2.get("llm_calls", 0)
        coverage["llm_calls"] = coverage["llm_calls_dir1"] + coverage["llm_calls_dir2"]
        coverage["cache_hits"] = stats1.get(
            "cache_hits", coverage.get("cache_hits", 0)
        ) + stats2.get("cache_hits", 0)
        coverage["relations_kept"] = kept
        coverage["neutral_dropped"] = len(relations_tmp) - kept
        # Measured from the phase-3 boundary; when the first direction rode along
        # with the atom-context batch, its time is billed to that phase.
        coverage["seconds"] = round(time.perf_counter() - _t_cc, 3)
        phase_stats["context_context"] = coverage

    print(f"[NLI] Relations built: {len(relations)}")

    if stats is not None:
        stats.update(_summarize_stats(cfg, gate, num_atoms, num_contexts, phase_stats))
        _report_savings(stats)

    return relations


def _split_merged_stats(
    merged: dict,
    ac_coverage: dict,
    cc_coverage: dict,
    num_ac: int,
    num_cc: int,
) -> None:
    """Attribute a shared fan-out's call counts back to the two phases.

    The merged batch reports one total, but the per-phase report needs a number for
    each. Cache hits are not tracked per item through the shared call, so both
    counts are apportioned by pair share -- exact in the common cases (all hits or
    all misses) and approximate only when a batch is partially cached. The totals
    remain exact either way.
    """
    total = num_ac + num_cc
    if not total:
        return
    calls = merged.get("llm_calls", 0)
    hits = merged.get("cache_hits", 0)
    ac_calls = round(calls * num_ac / total)
    ac_hits = round(hits * num_ac / total)
    ac_coverage["llm_calls"] = ac_calls
    ac_coverage["cache_hits"] = ac_hits
    cc_coverage["llm_calls"] = calls - ac_calls
    cc_coverage["cache_hits"] = hits - ac_hits


def _mirror_needed(relations1: List[Relation]) -> List[int]:
    """Indices whose reverse direction must actually be scored.

    Scoring only one direction per context pair would be unsound for two
    independent reasons, so the reverse call is skipped only where it provably
    cannot change the reconciled result:

    * ``entailment`` -- MUST be mirrored. ``equivalence`` arises *only* from
      entailment in both directions, and a single call cannot distinguish one-way
      entailment from equivalence. The two produce different factor tables
      downstream, so dropping the mirror would change scores, not just cost.
    * ``neutral`` -- MUST be mirrored. This is the reconciler's second chance.
      Since a backend error is reported as ``neutral`` with probability 1.0,
      skipping here would make relation recall a function of transient network
      failures.
    * ``contradiction`` -- skipped. Contradiction is symmetric, and reconciliation
      can only replace a contradictory direction with a *higher-probability*
      non-neutral one, which changes the retained probability but not the edge
      type.
    """
    return [
        i
        for i, rel in enumerate(relations1)
        if rel.get_type() in ("entailment", "neutral")
    ]


def _synthesize_mirrors(
    relations1: List[Relation],
    need: List[int],
    relations2_partial: List[Relation],
) -> List[Relation]:
    """Build a full reverse-direction list from the subset that was scored.

    Unscored slots get a synthetic ``neutral`` at probability 0.0. That value is
    deliberate: :func:`_reconcile_ctx_pair` compares probabilities only between
    two same-status relations, and a zero-probability neutral takes the
    "exactly one non-neutral" branch, so the real forward relation is returned
    unchanged with its own probability and orientation. This lets the reconciler
    be reused as-is, with no signature change.
    """
    relations2 = [
        Relation(
            source=rel.target,
            target=rel.source,
            type="neutral",
            probability=0.0,
            link="context_context",
        )
        for rel in relations1
    ]
    for slot, rel in zip(need, relations2_partial):
        relations2[slot] = rel
    return relations2


def _summarize_stats(
    cfg: NLIPairConfig,
    gate,
    num_atoms: int,
    num_contexts: int,
    phase_stats: Dict[str, dict],
) -> dict:
    """Assemble the coverage/cost report, including the all_pairs counterfactual."""
    out: Dict[str, object] = {
        "policy": cfg.policy,
        "gate_backend": (gate.backend if gate is not None else None),
        "gate_threshold": (cfg.gate_threshold if cfg.needs_gate else None),
        "cascade": cfg.ctx_ctx_single_direction_cascade,
        "num_atoms": num_atoms,
        "num_contexts": num_contexts,
    }
    out.update(phase_stats)

    llm_calls = sum(p.get("llm_calls", 0) for p in phase_stats.values())
    cache_hits = sum(p.get("cache_hits", 0) for p in phase_stats.values())
    seconds = sum(p.get("seconds", 0.0) for p in phase_stats.values())

    # What the faithful policy would have spent on the same inputs: A*C for the
    # atom-context phase, C*(C-1) for context-context (both directions).
    baseline = 0
    if "atom_context" in phase_stats:
        baseline += phase_stats["atom_context"]["pairs_possible"]
    if "context_context" in phase_stats:
        baseline += 2 * phase_stats["context_context"]["pairs_possible"]

    total = {
        "llm_calls": llm_calls,
        "cache_hits": cache_hits,
        "llm_calls_all_pairs_equivalent": baseline,
        "seconds": round(seconds, 3),
    }
    attempted = llm_calls + cache_hits
    total["reduction_factor"] = (
        round(baseline / attempted, 2) if attempted else None
    )
    out["totals"] = total
    return out


def _report_savings(stats: dict) -> None:
    """Print the headline cost line so a run states its own saving."""
    totals = stats.get("totals", {})
    baseline = totals.get("llm_calls_all_pairs_equivalent", 0)
    calls = totals.get("llm_calls", 0)
    hits = totals.get("cache_hits", 0)
    factor = totals.get("reduction_factor")
    msg = f"[NLI] LLM calls: {calls} (all_pairs equivalent: {baseline}"
    if factor:
        msg += f", {factor}x fewer"
    msg += ")"
    if hits:
        msg += f"; cache hits: {hits}"
    print(msg)
