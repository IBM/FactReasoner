#!/usr/bin/env python
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

"""End-to-end validation of NLI relation-extraction cost control on a live model.

Answers three questions that mocked tests cannot:

1. **Does the cheap path work against a real model?** Run faithful v3 and cheap v3
   on the same input and compare LLM call counts and per-atom factuality scores.
2. **What is the real recall loss?** The interesting number is
   ``P(pruned pair was non-neutral)`` -- how often a prefilter discards a pair the
   model would have called entailment or contradiction. Measured exactly, by
   replaying each policy's selection against the *recorded verdicts* of a full
   faithful run. No extra LLM calls: the faithful run already paid for every pair.
3. **Is the cache genuinely score-neutral?** Re-run with a warm cache and assert
   zero calls and identical scores.

The replay in (2) is the honest way to evaluate a prefilter. Comparing two live
runs conflates pruning with model nondeterminism; replaying against one run's
verdicts isolates the pruning decision itself.

Caveat, learned the hard way: the model is not deterministic. Two runs over these
same inputs produced 13 and 11 non-neutral pairs respectively, which moved
measured recall from 0.923 to 1.000 for an unchanged policy. A single run
therefore cannot establish recall -- run this more than once and take the worst
case per operating point. Each run's verdicts persist in its own cache directory,
so accumulated runs can be replayed together at no cost.

Usage:
    python scripts/e2e_nli_live.py --merlin-path /path/to/merlin
    python scripts/e2e_nli_live.py --merlin-path ... --offline   # no retrieval
"""

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Dict, List, Tuple

from dotenv import load_dotenv

load_dotenv()

from fact_reasoner.assessor import FactReasoner  # noqa: E402
from fact_reasoner.backends import build_backend  # noqa: E402
from fact_reasoner.core.atomizer import Atomizer  # noqa: E402
from fact_reasoner.core.base import Atom, Context  # noqa: E402
from fact_reasoner.core.nli import NLIExtractor  # noqa: E402
from fact_reasoner.core.nli_cache import NLIVerdictCache, extractor_identity  # noqa: E402
from fact_reasoner.core.nli_config import NLIPairConfig  # noqa: E402
from fact_reasoner.core import nli_pairs as npairs  # noqa: E402
from fact_reasoner.core.reviser import Reviser  # noqa: E402
from fact_reasoner.core.retriever import ContextRetriever, SourceRetriever  # noqa: E402
from fact_reasoner.core.summarizer import ContextSummarizer  # noqa: E402

MODEL_ID = "meta-llama/llama-3-3-70b-instruct"
BASE_URL = (
    "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/"
    "llama-3-3-70b-instruct"
)

# A response mixing true and false claims, so the graph has both entailments and
# contradictions rather than degenerating to one edge type.
QUERY = "Tell me about Marie Curie."
RESPONSE = (
    "Marie Curie was a Polish-born physicist and chemist. "
    "She won the Nobel Prize in Physics in 1903, shared with Pierre Curie and "
    "Henri Becquerel. "
    "She later won the Nobel Prize in Chemistry in 1911 for her discovery of "
    "radium and polonium. "
    "She was born in Warsaw in 1867. "
    "Curie was the first woman to win a Nobel Prize. "
    "She died in 1934 of aplastic anaemia caused by radiation exposure."
)
TOPIC = "Marie Curie"

# Offline fixture: hand-built atoms and contexts, used with --offline so the test
# does not depend on live web retrieval. Contexts deliberately include a
# contradiction, a near-duplicate, and an irrelevant passage.
OFFLINE_ATOMS = [
    "Marie Curie was a Polish-born physicist and chemist.",
    "Marie Curie won the Nobel Prize in Physics in 1903.",
    "Marie Curie won the Nobel Prize in Chemistry in 1911.",
    "Marie Curie was born in Warsaw in 1867.",
    "Marie Curie was the first woman to win a Nobel Prize.",
    "Marie Curie died in 1934 of aplastic anaemia.",
]
OFFLINE_CONTEXTS = {
    0: [
        "Maria Sklodowska-Curie was a physicist and chemist born in Poland who "
        "later became a naturalised French citizen.",
        "Curie conducted pioneering research on radioactivity in Paris.",
    ],
    1: [
        "The 1903 Nobel Prize in Physics was awarded jointly to Henri Becquerel, "
        "Pierre Curie and Marie Curie for their work on radiation.",
        # Near-duplicate of the above, to exercise dedup.
        "The Nobel Prize in Physics 1903 was awarded jointly to Henri Becquerel, "
        "Pierre Curie and Marie Curie for their research on radiation phenomena.",
    ],
    2: [
        "In 1911 Marie Curie received the Nobel Prize in Chemistry for the "
        "discovery of the elements radium and polonium.",
    ],
    3: [
        # Contradicts the atom: wrong city.
        "Marie Curie was born in Krakow in 1867 and moved to Paris as a student.",
    ],
    4: [
        "Marie Curie was the first woman to be awarded a Nobel Prize, and remains "
        "the only person to win in two different sciences.",
    ],
    5: [
        "Curie died on 4 July 1934 at a sanatorium in Passy, France, from aplastic "
        "anaemia believed to be caused by prolonged radiation exposure.",
        # Irrelevant passage, which a gate should prune and NLI should call neutral.
        "The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.",
    ],
}


def build_backend_and_components(show_progress: bool):
    backend = build_backend("rits", model_id=MODEL_ID, base_url=BASE_URL)
    return backend, dict(
        atom_extractor=Atomizer(backend),
        atom_reviser=Reviser(backend),
        nli_extractor=NLIExtractor(
            backend, nli_method="logprobs", show_progress=show_progress
        ),
        context_summarizer=ContextSummarizer(backend, show_progress=show_progress),
    )


def load_example(path: str) -> Tuple[str, List[str]]:
    """Load ``(response, atom_texts)`` from a ``data/lcs`` example file."""
    with open(path) as handle:
        data = json.load(handle)
    atoms = [
        a["text"] if isinstance(a, dict) else str(a) for a in data.get("atoms", [])
    ]
    return data.get("response", ""), atoms


def retrieve_contexts(
    atom_texts: List[str], top_k: int, num_workers: int, cache_dir: str
) -> Tuple[Dict[str, Atom], Dict[str, Context]]:
    """Retrieve real web contexts per atom, once, for reuse across configs.

    Retrieval happens exactly once and the resulting objects are deep-copied per
    config. Retrieving separately per config would let search nondeterminism
    change the context set between runs, which would confound the comparison the
    whole test exists to make.
    """
    from fact_reasoner.core.retriever import SourceRetriever

    retriever = SourceRetriever(
        service_type="google",
        top_k=top_k,
        fetch_text=True,
        num_workers=num_workers,
        cache_dir=cache_dir,
    )

    atoms: Dict[str, Atom] = {}
    contexts: Dict[str, Context] = {}
    for i, text in enumerate(atom_texts):
        atom = Atom(id=f"a{i}", text=text)
        atoms[atom.id] = atom
        try:
            hits = retriever.query(text) or []
        except Exception as exc:  # a single failed query must not kill the run
            print(f"  [warn] retrieval failed for a{i}: {exc}")
            hits = []
        for j, hit in enumerate(hits):
            body = str(hit.get("text") or hit.get("snippet") or "").strip()
            if not body:
                continue
            context = Context(
                id=f"c_a{i}_{j}",
                atom=atom,
                text=body,
                title=str(hit.get("title") or ""),
                link=str(hit.get("link") or ""),
                snippet=str(hit.get("snippet") or ""),
                # Use the snippet as the comparison text so the NLI premise stays
                # short; summarizing every page would cost more calls than the
                # experiment being measured.
                synthetic_summary=str(hit.get("snippet") or body)[:600],
            )
            contexts[context.id] = context
            atom.add_context(context)
        print(f"  a{i}: {len(atom.get_contexts())} contexts", flush=True)
    return atoms, contexts


def make_offline_pipeline(components, merlin_path, pair_config, cache_dir):
    """A FactReasoner preloaded with fixture atoms/contexts (no retrieval)."""
    pipeline = FactReasoner(
        merlin_path=merlin_path,
        use_priors=False,
        nli_pair_config=pair_config,
        nli_cache_dir=cache_dir,
        **components,
    )
    pipeline.query = QUERY
    pipeline.response = RESPONSE
    pipeline.topic = TOPIC

    atoms: Dict[str, Atom] = {}
    contexts: Dict[str, Context] = {}
    for i, text in enumerate(OFFLINE_ATOMS):
        atom = Atom(id=f"a{i}", text=text)
        atoms[atom.id] = atom
        for j, ctx_text in enumerate(OFFLINE_CONTEXTS.get(i, [])):
            context = Context(
                id=f"c_a{i}_{j}",
                atom=atom,
                text=ctx_text,
                # Pre-summarized, so the run does not spend calls on summarization
                # and the NLI premise is deterministic.
                synthetic_summary=ctx_text,
            )
            contexts[context.id] = context
            atom.add_context(context)

    pipeline.atoms = atoms
    pipeline.contexts = contexts
    pipeline.summarize_contexts = True
    pipeline.num_retrieved_contexts = len(contexts)
    pipeline.num_summarized_contexts = len(contexts)
    return pipeline


def make_pipeline_from(
    components,
    merlin_path,
    pair_config,
    cache_dir,
    atoms: Dict[str, Atom],
    contexts: Dict[str, Context],
    query: str,
    response: str,
    topic: str,
):
    """A FactReasoner preloaded with a supplied atom/context set."""
    pipeline = FactReasoner(
        merlin_path=merlin_path,
        use_priors=False,
        nli_pair_config=pair_config,
        nli_cache_dir=cache_dir,
        **components,
    )
    pipeline.query = query
    pipeline.response = response
    pipeline.topic = topic
    pipeline.atoms = atoms
    pipeline.contexts = contexts
    pipeline.summarize_contexts = True
    pipeline.num_retrieved_contexts = len(contexts)
    pipeline.num_summarized_contexts = len(contexts)
    return pipeline


def run_offline_case(
    components,
    merlin_path,
    pair_config,
    cache_dir,
    label,
    *,
    rel_context_context: bool = True,
    preloaded=None,
):
    """Build relations + score for one config, returning (results, stats, elapsed).

    Set ``rel_context_context=False`` for v2 (atom-context only). ``preloaded`` is
    a ``(atoms, contexts, query, response, topic)`` tuple; when given it replaces
    the built-in fixture, and the caller is responsible for handing over a fresh
    deep copy per config so dedup cannot mutate a set another config will use.
    """
    from fact_reasoner.core.utils import build_relations

    if preloaded is None:
        pipeline = make_offline_pipeline(
            components, merlin_path, pair_config, cache_dir
        )
    else:
        atoms, contexts, query, response, topic = preloaded
        pipeline = make_pipeline_from(
            components, merlin_path, pair_config, cache_dir,
            atoms, contexts, query, response, topic,
        )

    # Near-duplicate dedup normally happens inside build(); apply it here since the
    # offline path preloads contexts.
    if pipeline.nli_pair_config.dedup_near_duplicates:
        pipeline.contexts, pipeline.atoms, dedup_stats = npairs.dedup_contexts_near(
            pipeline.contexts,
            pipeline.atoms,
            threshold=pipeline.nli_pair_config.dedup_threshold,
            use_summary=True,
            embedding_model=pipeline.nli_pair_config.embedding_model,
        )
        pipeline.nli_stats["dedup"] = dedup_stats

    start = time.perf_counter()
    pipeline.relations = build_relations(
        atoms=pipeline.atoms,
        contexts=pipeline.contexts,
        rel_atom_context=True,
        rel_context_context=rel_context_context,
        contexts_per_atom_only=False,
        nli_extractor=pipeline.nli_extractor,
        use_summarized_contexts=True,
        pair_config=pipeline.nli_pair_config,
        stats=pipeline.nli_stats,
        cache=pipeline.nli_cache,
    )
    pipeline._build_fact_graph()
    pipeline._build_markov_network()
    # score() returns (results, marginals) despite its Dict annotation.
    scored = pipeline.score()
    results = scored[0] if isinstance(scored, tuple) else scored
    elapsed = time.perf_counter() - start

    print(f"[{label}] relations: {len(pipeline.relations)}, elapsed: {elapsed:.1f}s")
    return results, pipeline.nli_stats, elapsed, pipeline


def measure_recall_loss(
    pipeline, cache: NLIVerdictCache, policies: List[Tuple[str, NLIPairConfig]]
) -> List[dict]:
    """Replay each policy's selection against recorded verdicts.

    The faithful run already scored every pair and stored the verdicts, so we can
    ask exactly which pairs each policy *would* have pruned and what the model
    actually said about them. That isolates the pruning decision from model
    nondeterminism, which comparing two live runs cannot do.
    """
    atoms, contexts = pipeline.atoms, pipeline.contexts
    model_id, nli_method = extractor_identity(pipeline.nli_extractor)

    def verdict(premise_obj, hypothesis_obj):
        premise = premise_obj.get_summary() or premise_obj.get_text()
        hypothesis = hypothesis_obj.get_summary() or hypothesis_obj.get_text()
        key = cache.make_key(model_id, nli_method, premise, hypothesis)
        return cache.get_many([key]).get(key)

    # Ground truth: every atom-context pair's recorded verdict.
    truth: Dict[Tuple[str, str], dict] = {}
    for atom_id, atom in atoms.items():
        for context_id, context in contexts.items():
            got = verdict(context, atom)
            if got is not None:
                truth[(context_id, atom_id)] = got

    non_neutral_total = sum(
        1 for v in truth.values() if v.get("label") != "neutral"
    )

    rows = []
    for label, cfg in policies:
        gate, gate_atom_ids, gate_context_ids = npairs.build_gate(
            atoms, contexts, use_summary=True, embedding_model=cfg.embedding_model
        )
        selected, coverage = npairs.select_atom_context_pairs(
            atoms,
            contexts,
            policy=cfg.policy,
            gate_threshold=cfg.gate_threshold,
            neighbor_window=cfg.neighbor_window,
            gate=gate,
            gate_atom_ids=gate_atom_ids,
            gate_context_ids=gate_context_ids,
        )
        selected_set = set(selected)
        pruned = [p for p in truth if p not in selected_set]
        lost = [p for p in pruned if truth[p].get("label") != "neutral"]

        rows.append(
            {
                "policy": label,
                "pairs_total": len(truth),
                "pairs_selected": len(selected_set & set(truth)),
                "pairs_pruned": len(pruned),
                "non_neutral_total": non_neutral_total,
                "non_neutral_lost": len(lost),
                "recall": (
                    (non_neutral_total - len(lost)) / non_neutral_total
                    if non_neutral_total
                    else None
                ),
                "gate_backend": gate.backend,
                "lost_pairs": [
                    {"pair": list(p), "label": truth[p]["label"]} for p in lost
                ],
            }
        )
    return rows


def _per_atom_scores(results: dict) -> Dict[str, float]:
    """Flatten ``factuality_score_per_atom`` to ``{atom_id: score}``.

    The field is a list of single-key dicts, ``[{var: {"score", "support"}}, ...]``.
    """
    out: Dict[str, float] = {}
    for entry in results.get("factuality_score_per_atom") or []:
        for var, payload in entry.items():
            out[var] = (
                payload.get("score") if isinstance(payload, dict) else float(payload)
            )
    return out


def sweep_threshold(
    pipeline, cache: NLIVerdictCache, thresholds=(0.05, 0.10, 0.15, 0.22, 0.30)
) -> List[dict]:
    """Trace the recall/cost curve for each policy across gate thresholds.

    Replayed against recorded verdicts, so the whole sweep costs nothing. This is
    how a threshold should be chosen: the asymmetry between error types means the
    right operating point is the cheapest threshold that still holds recall at
    1.0, not the one with the best headline saving.
    """
    atoms, contexts = pipeline.atoms, pipeline.contexts
    model_id, nli_method = extractor_identity(pipeline.nli_extractor)

    truth: Dict[Tuple[str, str], dict] = {}
    for atom_id, atom in atoms.items():
        for context_id, context in contexts.items():
            premise = context.get_summary() or context.get_text()
            hypothesis = atom.get_summary() or atom.get_text()
            key = cache.make_key(model_id, nli_method, premise, hypothesis)
            got = cache.get_many([key]).get(key)
            if got is not None:
                truth[(context_id, atom_id)] = got
    non_neutral = {p for p, v in truth.items() if v.get("label") != "neutral"}

    rows = []
    for policy in ("gated", "provenance"):
        for threshold in thresholds:
            gate, atom_ids, context_ids = npairs.build_gate(
                atoms, contexts, use_summary=True
            )
            selected, _ = npairs.select_atom_context_pairs(
                atoms,
                contexts,
                policy=policy,
                gate_threshold=threshold,
                gate=gate,
                gate_atom_ids=atom_ids,
                gate_context_ids=context_ids,
            )
            chosen = set(selected)
            lost = len(non_neutral - chosen)
            rows.append(
                {
                    "policy": policy,
                    "gate_threshold": threshold,
                    "pairs_scored": len(chosen),
                    "saving": round(len(truth) / max(len(chosen), 1), 2),
                    "non_neutral_lost": lost,
                    "recall": round(
                        (len(non_neutral) - lost) / len(non_neutral), 3
                    )
                    if non_neutral
                    else None,
                }
            )
    return rows


def compare_scores(a: dict, b: dict) -> dict:
    """Per-atom score deltas between two runs."""
    pa = _per_atom_scores(a)
    pb = _per_atom_scores(b)
    keys = sorted(set(pa) | set(pb))
    deltas = {k: abs(pa.get(k, 0.0) - pb.get(k, 0.0)) for k in keys}
    return {
        "max_abs_delta": max(deltas.values()) if deltas else 0.0,
        "mean_abs_delta": (sum(deltas.values()) / len(deltas)) if deltas else 0.0,
        "overall_delta": abs(
            (a.get("factuality_score") or 0.0) - (b.get("factuality_score") or 0.0)
        ),
        "per_atom": deltas,
    }


def run_big_example(args, components, cache_dir, report) -> int:
    """v2 faithful vs v2-cheap on a real example with live web retrieval.

    v2 is atom-context only, so this isolates the phase the provenance policy
    targets. Contexts are retrieved once and deep-copied per config, so the two
    runs see byte-identical evidence and the only difference is pair selection.
    """
    import copy

    response, atom_texts = load_example(args.example)
    if args.max_atoms:
        atom_texts = atom_texts[: args.max_atoms]
    print(f"example      : {args.example}")
    print(f"atoms        : {len(atom_texts)}")
    print(f"top_k        : {args.top_k}\n")

    print("--- retrieving contexts (live web search, once) ---")
    t0 = time.perf_counter()
    atoms0, contexts0 = retrieve_contexts(
        atom_texts,
        args.top_k,
        args.num_workers,
        os.path.join(args.output_dir, "search_cache"),
    )
    print(
        f"retrieved {len(contexts0)} contexts for {len(atoms0)} atoms "
        f"in {time.perf_counter() - t0:.1f}s\n"
    )
    if not contexts0:
        print("ERROR: retrieval produced no contexts.", file=sys.stderr)
        return 2

    query = f"Tell me about: {response[:80]}"
    faithful = NLIPairConfig()
    cheap = NLIPairConfig(
        policy="provenance",
        dedup_near_duplicates=True,
        ctx_ctx_single_direction_cascade=True,
        merge_phases=True,
    )

    runs = {}
    for label, cfg in (("v2 all_pairs", faithful), ("v2-cheap", cheap)):
        print(f"--- {label} ---")
        preloaded = (
            copy.deepcopy(atoms0),
            copy.deepcopy(contexts0),
            query,
            response,
            "",
        )
        results, stats, elapsed, pipeline = run_offline_case(
            components,
            args.merlin_path,
            cfg,
            cache_dir,
            label,
            rel_context_context=False,  # v2: atom-context only
            preloaded=preloaded,
        )
        runs[label] = {
            "stats": stats,
            "elapsed_s": round(elapsed, 2),
            "results": results,
            "pipeline": pipeline,
        }
        print()

    base, chp = runs["v2 all_pairs"], runs["v2-cheap"]
    bt, ct = base["stats"]["totals"], chp["stats"]["totals"]
    b_att = bt["llm_calls"] + bt["cache_hits"]
    c_att = ct["llm_calls"] + ct["cache_hits"]

    # Recall, replayed against the faithful run's recorded verdicts.
    cache = NLIVerdictCache(cache_dir)
    recall_rows = measure_recall_loss(
        base["pipeline"], cache, [("provenance", cheap)]
    )
    sweep_rows = sweep_threshold(base["pipeline"], cache)

    b_score = base["results"].get("factuality_score")
    c_score = chp["results"].get("factuality_score")
    delta = compare_scores(base["results"], chp["results"])

    print("=" * 72)
    print("SUMMARY (v2: atom-context only)")
    print("=" * 72)
    print(f"atoms={len(atoms0)}  contexts={len(contexts0)}")
    print(
        f"v2 all_pairs : {b_att:>5} pairs scored  "
        f"{base['elapsed_s']:>7.1f}s  score={b_score:.4f}"
    )
    print(
        f"v2-cheap     : {c_att:>5} pairs scored  "
        f"{chp['elapsed_s']:>7.1f}s  score={c_score:.4f}"
    )
    print(
        f"saved        : {b_att - c_att:>5} calls  "
        f"({b_att / max(c_att, 1):.2f}x fewer)"
    )
    print(f"score delta  : {abs(b_score - c_score):.6f} "
          f"(max per-atom {delta['max_abs_delta']:.6f})")
    row = recall_rows[0]
    print(
        f"recall       : {row['recall']:.3f}  "
        f"({row['non_neutral_lost']} of {row['non_neutral_total']} "
        f"non-neutral relations lost)"
    )
    print()
    print("threshold sweep (replayed against recorded verdicts)")
    print(f"  {'policy':12} {'thresh':>7} {'scored':>7} {'saving':>8} "
          f"{'lost':>5} {'recall':>7}")
    for r in sweep_rows:
        print(
            f"  {r['policy']:12} {r['gate_threshold']:>7.2f} "
            f"{r['pairs_scored']:>7} {r['saving']:>7.2f}x "
            f"{r['non_neutral_lost']:>5} {r['recall']:>7.3f}"
        )
    safe = [r for r in sweep_rows if r["recall"] == 1.0]
    if safe:
        best = max(safe, key=lambda r: r["saving"])
        print(
            f"  -> cheapest lossless setting: {best['policy']} @ "
            f"{best['gate_threshold']:.2f} ({best['saving']}x)"
        )

    report.update(
        example=args.example,
        num_atoms=len(atoms0),
        num_contexts=len(contexts0),
        top_k=args.top_k,
        pairs_scored={"faithful": b_att, "cheap": c_att},
        calls_saved=b_att - c_att,
        reduction=round(b_att / max(c_att, 1), 3),
        factuality_score={"faithful": b_score, "cheap": c_score},
        score_delta=delta,
        recall=recall_rows,
        threshold_sweep=sweep_rows,
        elapsed_s={
            "faithful": base["elapsed_s"],
            "cheap": chp["elapsed_s"],
        },
    )
    out = os.path.join(args.output_dir, "report.json")
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"\nReport written to {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-path", required=True)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use the built-in atom/context fixture instead of live retrieval.",
    )
    parser.add_argument("--output-dir", default="results/e2e_nli_live")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument(
        "--example",
        default=None,
        help="Path to a data/lcs example JSON. Runs the v2 faithful vs v2-cheap "
        "comparison with live web retrieval instead of the built-in fixture.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        help="Truncate the example to this many atoms (to bound cost).",
    )
    args = parser.parse_args()

    if not os.getenv("RITS_API_KEY"):
        print("ERROR: RITS_API_KEY is not set (expected in .env).", file=sys.stderr)
        return 2
    if not args.offline and not os.getenv("SERPER_API_KEY"):
        print(
            "ERROR: SERPER_API_KEY is required unless --offline.", file=sys.stderr
        )
        return 2

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = os.path.join(args.output_dir, "nli_cache")
    report: Dict[str, object] = {"model_id": MODEL_ID, "offline": args.offline}

    print(f"=== Live end-to-end NLI cost test: {MODEL_ID} ===\n")
    backend, components = build_backend_and_components(args.progress_bar)

    if args.example:
        return run_big_example(args, components, cache_dir, report)

    if not args.offline:
        print(
            "Live retrieval requires --example; use --offline for the fixture.",
        )
        return 2

    faithful = NLIPairConfig()
    cheap = NLIPairConfig(
        policy="provenance",
        dedup_near_duplicates=True,
        ctx_ctx_single_direction_cascade=True,
        merge_phases=True,
    )
    gated = NLIPairConfig(policy="gated")

    # --- 1. Faithful v3, cold. Populates the cache with every verdict.
    print("--- [1/4] faithful v3 (cold) ---")
    f_results, f_stats, f_elapsed, f_pipeline = run_offline_case(
        components, args.merlin_path, faithful, cache_dir, "faithful"
    )
    report["faithful"] = {
        "stats": f_stats,
        "elapsed_s": round(f_elapsed, 2),
        "factuality_score": f_results.get("factuality_score"),
    }

    # --- 2. Faithful v3 again, warm cache. Must cost nothing and match exactly.
    print("\n--- [2/4] faithful v3 (warm cache) ---")
    w_results, w_stats, w_elapsed, _ = run_offline_case(
        components, args.merlin_path, faithful, cache_dir, "warm"
    )
    warm_calls = w_stats["totals"]["llm_calls"]
    warm_delta = compare_scores(f_results, w_results)
    report["warm"] = {
        "llm_calls": warm_calls,
        "cache_hits": w_stats["totals"]["cache_hits"],
        "elapsed_s": round(w_elapsed, 2),
        "score_delta": warm_delta,
    }

    # --- 3. Cheap v3, live.
    print("\n--- [3/4] cheap v3 (provenance + dedup + cascade) ---")
    c_results, c_stats, c_elapsed, _ = run_offline_case(
        components, args.merlin_path, cheap, cache_dir, "cheap"
    )
    cheap_delta = compare_scores(f_results, c_results)
    report["cheap"] = {
        "stats": c_stats,
        "elapsed_s": round(c_elapsed, 2),
        "factuality_score": c_results.get("factuality_score"),
        "score_delta": cheap_delta,
    }

    # --- 4. Recall loss, replayed offline against the recorded verdicts.
    print("\n--- [4/4] recall loss (replay vs recorded verdicts) ---")
    cache = NLIVerdictCache(cache_dir)
    recall_rows = measure_recall_loss(
        f_pipeline, cache, [("gated", gated), ("provenance", cheap)]
    )
    report["recall"] = recall_rows
    sweep_rows = sweep_threshold(f_pipeline, cache)
    report["threshold_sweep"] = sweep_rows

    # ---- Summary
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    ft, ct = f_stats["totals"], c_stats["totals"]
    f_score = f_results.get("factuality_score")
    c_score = c_results.get("factuality_score")
    print(
        f"faithful v3 : {ft['llm_calls']:>5} calls  "
        f"(all_pairs equivalent {ft['llm_calls_all_pairs_equivalent']})  "
        f"{f_elapsed:.1f}s  score={f_score:.4f}"
    )
    print(
        f"cheap v3    : {ct['llm_calls']:>5} calls  "
        f"{ct['reduction_factor']}x fewer  "
        f"{c_elapsed:.1f}s  score={c_score:.4f}"
    )
    print(f"score delta (faithful vs cheap): {abs(f_score - c_score):.6f}")
    print(
        f"warm cache  : {warm_calls:>5} calls  "
        f"({w_stats['totals']['cache_hits']} hits)  {w_elapsed:.1f}s"
    )
    print()
    print(f"{'policy':12} {'pruned':>8} {'non-neutral lost':>18} {'recall':>8}")
    for row in recall_rows:
        recall = f"{row['recall']:.3f}" if row["recall"] is not None else "n/a"
        print(
            f"{row['policy']:12} {row['pairs_pruned']:>8} "
            f"{row['non_neutral_lost']:>18} {recall:>8}"
        )

    # Compare pairs *attempted* rather than LLM calls: with a warm cache both runs
    # issue zero calls, so call count cannot distinguish the policies. The pair
    # count is what the policy actually controls.
    f_attempted = ft["llm_calls"] + ft["cache_hits"]
    c_attempted = ct["llm_calls"] + ct["cache_hits"]
    print()
    print("threshold sweep (replayed; recall 1.0 is the operating point to pick)")
    print(
        f"  {'policy':12} {'thresh':>7} {'scored':>7} {'saving':>8} "
        f"{'lost':>5} {'recall':>7}"
    )
    for row in sweep_rows:
        print(
            f"  {row['policy']:12} {row['gate_threshold']:>7.2f} "
            f"{row['pairs_scored']:>7} {row['saving']:>7.2f}x "
            f"{row['non_neutral_lost']:>5} {row['recall']:>7.3f}"
        )
    safe = [r for r in sweep_rows if r["recall"] == 1.0]
    if safe:
        best = max(safe, key=lambda r: r["saving"])
        print(
            f"  -> cheapest lossless setting: {best['policy']} @ "
            f"{best['gate_threshold']:.2f} ({best['saving']}x)"
        )

    checks = {
        "warm cache costs nothing": warm_calls == 0,
        "warm cache is score-identical": warm_delta["max_abs_delta"] < 1e-9,
        "cheap path scores fewer pairs": c_attempted < f_attempted,
        "cheap path loses no non-neutral relations": all(
            row["non_neutral_lost"] == 0
            for row in recall_rows
            if row["policy"] == "provenance"
        ),
    }
    report["pairs_attempted"] = {"faithful": f_attempted, "cheap": c_attempted}
    print()
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    report["checks"] = checks
    out = os.path.join(args.output_dir, "report.json")
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"\nReport written to {out}")

    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
