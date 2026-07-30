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

"""FactReasoner v2: all_pairs vs cheap on the Lanny Flaherty biography example.

The example (``data/flaherty_google.json``) ships 26 atoms with gold S/NS labels
and 130 retrieved contexts that carry a title, snippet and link but *no page
text*, so the text has to be fetched from the links and summarized before any NLI
comparison can happen.

Pipeline, in order:

1. **Fetch** page text for each unique link (130 contexts share only 46 URLs, so
   fetching is deduplicated and cached on disk).
2. **Summarize** each context against its own atom with the real
   ``ContextSummarizer`` -- the summary is what the NLI premise actually uses, and
   it is atom-conditioned, so it cannot be shared across atoms.
3. **Score** the atom-context relations twice, under ``all_pairs`` and under the
   cheap policy, on byte-identical prepared contexts.
4. **Report** LLM calls, wall-clock time, factuality scores, accuracy against the
   gold labels, and the recall cost of pruning.

Stages 1 and 2 are cached so the two scoring cells share identical inputs; only
step 3 differs between them, which is the whole point of the comparison.

Usage:
    python scripts/exp_flaherty_v2.py --merlin-path /path/to/merlin
    python scripts/exp_flaherty_v2.py --merlin-path ... --prep-only
"""

import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

from fact_reasoner.assessor import FactReasoner  # noqa: E402
from fact_reasoner.backends import build_backend  # noqa: E402
from fact_reasoner.core.atomizer import Atomizer  # noqa: E402
from fact_reasoner.core.base import Atom, Context  # noqa: E402
from fact_reasoner.core.nli import NLIExtractor  # noqa: E402
from fact_reasoner.core.nli_config import NLIPairConfig  # noqa: E402
from fact_reasoner.core import nli_pairs as npairs  # noqa: E402
from fact_reasoner.core.reviser import Reviser  # noqa: E402
from fact_reasoner.core.summarizer import ContextSummarizer  # noqa: E402
from fact_reasoner.core.utils import build_relations, is_relevant_context  # noqa: E402

MODEL_ID = "meta-llama/llama-3-3-70b-instruct"
BASE_URL = (
    "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/"
    "llama-3-3-70b-instruct"
)
DATA_PATH = "data/flaherty_google.json"

# Cap fetched page text; full pages blow up the summarizer prompt without adding
# signal, and the snippet already localizes the relevant passage.
MAX_PAGE_CHARS = 6000


# ---------------------------------------------------------------------------
# Stage 1: fetch page text for the unique links.
# ---------------------------------------------------------------------------


def fetch_pages(links: List[str], cache_path: str) -> Dict[str, str]:
    """Fetch page text per unique URL, cached to disk as JSON.

    Failures are recorded as empty strings rather than retried forever: a link
    that 403s or times out simply falls back to its snippet downstream, which is
    what the production retriever does too.
    """
    cache: Dict[str, str] = {}
    if os.path.exists(cache_path):
        with open(cache_path) as handle:
            cache = json.load(handle)

    todo = [u for u in dict.fromkeys(links) if u not in cache]
    print(f"[fetch] {len(links)} links, {len(set(links))} unique, {len(todo)} to fetch")
    if not todo:
        return cache

    # The repo's own text extractor, so this matches production behavior.
    from fact_reasoner.core.retriever import SourceRetriever

    retriever = SourceRetriever(service_type="google", top_k=1, fetch_text=True)
    extract = None
    for name in ("_extract_text_from_url", "extract_text_from_url", "_fetch_text"):
        if hasattr(retriever, name):
            extract = getattr(retriever, name)
            break

    for i, url in enumerate(todo, 1):
        text = ""
        try:
            if extract is not None:
                text = str(extract(url) or "")
            else:
                import requests
                from bs4 import BeautifulSoup

                resp = requests.get(
                    url,
                    timeout=20,
                    headers={"User-Agent": "Mozilla/5.0 (FactReasoner experiment)"},
                )
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = " ".join(soup.get_text(" ").split())
        except Exception as exc:
            print(f"  [{i}/{len(todo)}] FAIL {url[:70]}: {type(exc).__name__}")
        else:
            print(f"  [{i}/{len(todo)}] ok   {len(text):>6} chars  {url[:60]}")
        cache[url] = text[:MAX_PAGE_CHARS]
        with open(cache_path, "w") as handle:
            json.dump(cache, handle)
    return cache


# ---------------------------------------------------------------------------
# Stage 2: build atoms/contexts and summarize.
# ---------------------------------------------------------------------------


def load_example() -> Tuple[dict, Dict[str, Atom], Dict[str, Context]]:
    """Build Atom/Context objects from the example file, wiring provenance."""
    with open(DATA_PATH) as handle:
        data = json.load(handle)

    atoms: Dict[str, Atom] = {}
    for record in data["atoms"]:
        atom = Atom(
            id=record["id"], text=record["text"], label=record.get("label")
        )
        atoms[atom.id] = atom

    by_id = {c["id"]: c for c in data["contexts"]}
    contexts: Dict[str, Context] = {}
    for record in data["atoms"]:
        atom = atoms[record["id"]]
        for context_id in record.get("contexts", []):
            raw = by_id.get(context_id)
            if raw is None:
                continue
            context = Context(
                id=context_id,
                atom=atom,  # provenance: retrieved FOR this atom
                text=raw.get("text", ""),
                title=raw.get("title", ""),
                link=raw.get("link", ""),
                snippet=raw.get("snippet", ""),
            )
            contexts[context_id] = context
            atom.add_context(context)
    return data, atoms, contexts


def summarize_contexts(
    atoms: Dict[str, Atom],
    contexts: Dict[str, Context],
    pages: Dict[str, str],
    summarizer: ContextSummarizer,
    cache_path: str,
) -> dict:
    """Summarize each context against its own atom, caching by (atom, context).

    Summaries are atom-conditioned, so they cannot be shared between atoms even
    when two atoms cite the same URL -- which is exactly why this costs one call
    per context rather than one per unique link.
    """
    cache: Dict[str, str] = {}
    if os.path.exists(cache_path):
        with open(cache_path) as handle:
            cache = json.load(handle)

    stats = {"calls": 0, "cached": 0, "irrelevant": 0, "empty_page": 0}
    start = time.perf_counter()

    for atom_id, atom in atoms.items():
        pending: List[Tuple[str, str]] = []  # (context_id, text to summarize)
        for context_id in list(atom.get_contexts()):
            context = contexts.get(context_id)
            if context is None:
                continue
            key = f"{atom_id}|{context_id}"
            if key in cache:
                context.set_synthetic_summary(cache[key])
                stats["cached"] += 1
                continue
            page = pages.get(context.get_link(), "")
            if not page:
                stats["empty_page"] += 1
            # Snippet first: it localizes the passage the search engine matched.
            body = (context.get_snippet() or "") + ("\n\n" + page if page else "")
            pending.append((context_id, body.strip()))

        if not pending:
            continue
        texts = [t for _, t in pending]
        results = summarizer.run(texts, atom_text=atom.get_text())
        stats["calls"] += len(texts)
        for (context_id, _), result in zip(pending, results):
            summary = (result or {}).get("summary", "") or ""
            if summary and not is_relevant_context(summary):
                # Production drops these; keep the context but mark it, so both
                # policies see the same evidence set.
                stats["irrelevant"] += 1
            contexts[context_id].set_synthetic_summary(summary)
            cache[f"{atom_id}|{context_id}"] = summary
        with open(cache_path, "w") as handle:
            json.dump(cache, handle)
        print(
            f"  {atom_id}: summarized {len(pending)} "
            f"(cached {stats['cached']}, calls {stats['calls']})",
            flush=True,
        )

    # Any context that ended up with no summary would become an empty NLI premise;
    # fall back to its snippet so the comparison is not measuring empty strings.
    for context in contexts.values():
        if not context.get_summary():
            context.set_synthetic_summary(
                context.get_snippet() or context.get_title() or "(no content)"
            )

    stats["seconds"] = round(time.perf_counter() - start, 2)
    return stats


# ---------------------------------------------------------------------------
# Stage 3: score under each policy.
# ---------------------------------------------------------------------------


def score_cell(
    label: str,
    cfg: NLIPairConfig,
    atoms: Dict[str, Atom],
    contexts: Dict[str, Context],
    data: dict,
    components: dict,
    merlin_path: str,
    cache_dir: Optional[str],
) -> dict:
    """Run one (version, policy) cell end to end and return its measurements.

    Pass ``cache_dir=None`` to disable the verdict cache, so every selected pair
    reaches the model. That is required for a wall-clock comparison: with the
    cache on, the second cell is served from the first cell's verdicts and
    finishes in milliseconds, which measures the cache rather than the policy.
    """
    pipeline = FactReasoner(
        merlin_path=merlin_path,
        use_priors=False,
        nli_pair_config=cfg,
        nli_cache_dir=cache_dir,
        **components,
    )
    pipeline.query = data["input"]
    pipeline.response = data["output"]
    pipeline.topic = data.get("topic", "")
    pipeline.atoms = atoms
    pipeline.contexts = contexts
    pipeline.summarize_contexts = True
    pipeline.num_retrieved_contexts = len(contexts)
    pipeline.num_summarized_contexts = len(contexts)

    print(f"--- {label} ---", flush=True)

    # Near-duplicate dedup normally runs inside build(); this harness preloads
    # contexts, so apply it here to match.
    if cfg.dedup_near_duplicates:
        pipeline.contexts, pipeline.atoms, dedup = npairs.dedup_contexts_near(
            pipeline.contexts,
            pipeline.atoms,
            threshold=cfg.dedup_threshold,
            use_summary=True,
            embedding_model=cfg.embedding_model,
        )
        pipeline.nli_stats["dedup"] = dedup
        print(
            f"[dedup] {dedup['contexts_before']} -> {dedup['contexts_after']} "
            f"contexts ({dedup['collapsed']} collapsed, "
            f"{dedup['owners_merged']} owners merged)"
        )

    t_nli = time.perf_counter()
    pipeline.relations = build_relations(
        atoms=pipeline.atoms,
        contexts=pipeline.contexts,
        rel_atom_context=True,
        rel_context_context=False,  # v2
        contexts_per_atom_only=False,
        nli_extractor=pipeline.nli_extractor,
        use_summarized_contexts=True,
        pair_config=cfg,
        stats=pipeline.nli_stats,
        cache=pipeline.nli_cache,
    )
    nli_seconds = time.perf_counter() - t_nli

    t_inf = time.perf_counter()
    pipeline._build_fact_graph()
    pipeline._build_markov_network()
    scored = pipeline.score()
    results = scored[0] if isinstance(scored, tuple) else scored
    inference_seconds = time.perf_counter() - t_inf

    totals = pipeline.nli_stats.get("totals", {})
    per_atom = _per_atom_scores(results)
    gold = {a["id"]: a["label"] for a in data["atoms"]}
    acc = _accuracy(results, gold)

    # Record every non-neutral verdict this cell observed, keyed by id pair. With
    # the cache disabled this is the only record of the exhaustive run's verdicts,
    # and it is what the recall replay needs. Neutral relations are dropped by
    # build_relations before returning, so their absence is itself the signal:
    # any pair the baseline scored that is missing here came back neutral.
    verdicts = {
        f"{rel.source.id}|{rel.target.id}": {
            "label": rel.get_type(),
            "probability": rel.get_probability(),
        }
        for rel in pipeline.relations
    }

    return {
        "verdicts": verdicts,
        "label": label,
        "policy": cfg.policy,
        "num_atoms": len(pipeline.atoms),
        "num_contexts": len(pipeline.contexts),
        "llm_calls": totals.get("llm_calls", 0),
        "cache_hits": totals.get("cache_hits", 0),
        "pairs_attempted": totals.get("llm_calls", 0) + totals.get("cache_hits", 0),
        "all_pairs_equivalent": totals.get("llm_calls_all_pairs_equivalent", 0),
        "reduction_factor": totals.get("reduction_factor"),
        "relations": len(pipeline.relations),
        "nli_seconds": round(nli_seconds, 2),
        "inference_seconds": round(inference_seconds, 2),
        "factuality_score": results.get("factuality_score"),
        "per_atom": per_atom,
        "accuracy": acc,
        "stats": pipeline.nli_stats,
        "pipeline": pipeline,
        "results": results,
    }


def _per_atom_scores(results: dict) -> Dict[str, float]:
    """Flatten ``factuality_score_per_atom`` (a list of single-key dicts).

    Delegates to the shared extractor in ``fact_reasoner.lcs.priors``.
    """
    from fact_reasoner.lcs.priors import atom_priors_from_results

    return atom_priors_from_results(results)


def _accuracy(results: dict, gold: Dict[str, str]) -> dict:
    """Accuracy of the S/NS predictions against the gold labels."""
    preds = results.get("predictions") or {}
    common = [a for a in gold if a in preds]
    tp = sum(1 for a in common if gold[a] == "S" and preds[a] == "S")
    tn = sum(1 for a in common if gold[a] == "NS" and preds[a] == "NS")
    correct = tp + tn
    return {
        "n": len(common),
        "correct": correct,
        "accuracy": round(correct / len(common), 4) if common else None,
        "true_S": tp,
        "true_NS": tn,
        "gold_S": sum(1 for a in common if gold[a] == "S"),
        "gold_NS": sum(1 for a in common if gold[a] == "NS"),
        "predictions": {a: preds[a] for a in common},
    }


# ---------------------------------------------------------------------------
# Recall replay and threshold sweep (zero extra LLM cost).
# ---------------------------------------------------------------------------


def _non_neutral_from_verdicts(verdicts: Dict[str, dict]) -> set:
    """The set of (context_id, atom_id) pairs the baseline scored as non-neutral.

    ``build_relations`` returns only non-neutral relations, so every entry here is
    one; pairs absent from the record came back neutral and are safe to prune.
    """
    out = set()
    for key, verdict in (verdicts or {}).items():
        if "|" not in key:
            continue
        source, target = key.split("|", 1)
        if verdict.get("label") != "neutral":
            out.add((source, target))
    return out


def replay_recall(pipeline, verdicts: Dict[str, dict], cfgs) -> List[dict]:
    """Measure P(pruned pair was non-neutral) against the baseline's verdicts.

    Uses the relations the exhaustive cell actually produced rather than a verdict
    cache, so this works with caching disabled.
    """
    atoms, contexts = pipeline.atoms, pipeline.contexts
    # The exhaustive cell scored the full product, so that is the pair universe.
    truth = {(c, a) for a in atoms for c in contexts}
    non_neutral = _non_neutral_from_verdicts(verdicts) & truth

    rows = []
    for label, cfg in cfgs:
        gate, atom_ids, context_ids = npairs.build_gate(
            atoms, contexts, use_summary=True,
            embedding_model=cfg.embedding_model,
        )
        selected, _ = npairs.select_atom_context_pairs(
            atoms, contexts, policy=cfg.policy,
            gate_threshold=cfg.gate_threshold,
            neighbor_window=cfg.neighbor_window,
            gate=gate, gate_atom_ids=atom_ids, gate_context_ids=context_ids,
        )
        chosen = set(selected)
        lost = sorted(non_neutral - chosen)
        rows.append({
            "policy": label,
            "gate_backend": gate.backend,
            "pairs_total": len(truth),
            "pairs_selected": len(chosen & set(truth)),
            "pairs_pruned": len(truth) - len(chosen & set(truth)),
            "non_neutral_total": len(non_neutral),
            "non_neutral_lost": len(lost),
            "recall": round((len(non_neutral) - len(lost)) / len(non_neutral), 4)
            if non_neutral else None,
            "lost_pairs": [
                {
                    "pair": list(p),
                    "label": (verdicts.get(f"{p[0]}|{p[1]}") or {}).get(
                        "label", "non-neutral"
                    ),
                }
                for p in lost
            ],
        })
    return rows


def sweep(pipeline, verdicts: Dict[str, dict],
          thresholds=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)) -> List[dict]:
    """Recall/cost curve per policy across gate thresholds, replayed."""
    atoms, contexts = pipeline.atoms, pipeline.contexts
    truth = {(c, a) for a in atoms for c in contexts}
    non_neutral = _non_neutral_from_verdicts(verdicts) & truth

    gate, atom_ids, context_ids = npairs.build_gate(atoms, contexts, use_summary=True)
    rows = []
    for policy in ("gated", "provenance"):
        for threshold in thresholds:
            selected, _ = npairs.select_atom_context_pairs(
                atoms, contexts, policy=policy, gate_threshold=threshold,
                gate=gate, gate_atom_ids=atom_ids, gate_context_ids=context_ids,
            )
            chosen = set(selected)
            lost = len(non_neutral - chosen)
            rows.append({
                "policy": policy,
                "gate_threshold": threshold,
                "pairs_scored": len(chosen),
                "saving": round(len(truth) / max(len(chosen), 1), 3),
                "non_neutral_lost": lost,
                "recall": round((len(non_neutral) - lost) / len(non_neutral), 4)
                if non_neutral else None,
                "gate_backend": gate.backend,
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merlin-path", required=True)
    parser.add_argument("--output-dir", default="results/exp_flaherty_v2")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the NLI verdict cache, so both cells issue live calls. "
        "Required for a measured (rather than extrapolated) wall-clock "
        "comparison: with the cache on, the second cell is served from the "
        "first cell's verdicts and finishes instantly.",
    )
    parser.add_argument(
        "--prep-only",
        action="store_true",
        help="Fetch and summarize only; skip the scoring cells.",
    )
    parser.add_argument("--progress-bar", action="store_true")
    args = parser.parse_args()

    if not os.getenv("RITS_API_KEY"):
        print("ERROR: RITS_API_KEY is not set (expected in .env).", file=sys.stderr)
        return 2

    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = None if args.no_cache else os.path.join(args.output_dir, "nli_cache")
    report: Dict[str, object] = {
        "model_id": MODEL_ID,
        "data": DATA_PATH,
        "version": "v2 (atom-context only)",
        "nli_cache_enabled": not args.no_cache,
    }

    print(f"=== FactReasoner v2: all_pairs vs cheap ({MODEL_ID}) ===")
    if args.no_cache:
        print("NLI verdict cache DISABLED: both cells issue live calls.\n")
    else:
        print()
    backend = build_backend("rits", model_id=MODEL_ID, base_url=BASE_URL)
    summarizer = ContextSummarizer(backend, show_progress=args.progress_bar)
    components = dict(
        atom_extractor=Atomizer(backend),
        atom_reviser=Reviser(backend),
        nli_extractor=NLIExtractor(
            backend, nli_method="logprobs", show_progress=args.progress_bar
        ),
        context_summarizer=summarizer,
    )

    data, atoms, contexts = load_example()
    print(f"atoms={len(atoms)} contexts={len(contexts)}")

    # Stage 1: fetch.
    print("\n--- stage 1: fetch page text for unique links ---")
    t0 = time.perf_counter()
    pages = fetch_pages(
        [c.get_link() for c in contexts.values() if c.get_link()],
        os.path.join(args.output_dir, "pages.json"),
    )
    fetch_seconds = time.perf_counter() - t0
    nonempty = sum(1 for u, t in pages.items() if t)
    print(
        f"[fetch] {nonempty}/{len(pages)} URLs yielded text in {fetch_seconds:.1f}s"
    )

    # Stage 2: summarize.
    print("\n--- stage 2: summarize each context against its atom ---")
    sum_stats = summarize_contexts(
        atoms, contexts, pages, summarizer,
        os.path.join(args.output_dir, "summaries.json"),
    )
    print(
        f"[summarize] calls={sum_stats['calls']} cached={sum_stats['cached']} "
        f"irrelevant={sum_stats['irrelevant']} empty_page={sum_stats['empty_page']} "
        f"in {sum_stats['seconds']}s"
    )
    report["prep"] = {
        "fetch_seconds": round(fetch_seconds, 2),
        "unique_links": len(pages),
        "links_with_text": nonempty,
        "summarize": sum_stats,
    }

    if args.prep_only:
        with open(os.path.join(args.output_dir, "report.json"), "w") as handle:
            json.dump(report, handle, indent=2, default=str)
        print("\nPrep complete (--prep-only).")
        return 0

    # Stage 3: the two cells, on byte-identical prepared contexts.
    print("\n--- stage 3: scoring cells ---")
    faithful = NLIPairConfig()
    cheap = NLIPairConfig(
        policy="provenance",
        dedup_near_duplicates=True,
        ctx_ctx_single_direction_cascade=True,
        merge_phases=True,
    )

    cells = {}
    for label, cfg in (("v2 all_pairs", faithful), ("v2-cheap", cheap)):
        cells[label] = score_cell(
            label, cfg,
            copy.deepcopy(atoms), copy.deepcopy(contexts),
            data, components, args.merlin_path, cache_dir,
        )
        print()

    base, chp = cells["v2 all_pairs"], cells["v2-cheap"]
    # The exhaustive cell's own relations are the ground truth, so recall can be
    # replayed with or without a verdict cache.
    baseline_verdicts = base["verdicts"]
    recall_rows = replay_recall(
        base["pipeline"], baseline_verdicts, [("provenance", cheap)]
    )
    sweep_rows = sweep(base["pipeline"], baseline_verdicts)

    print("=" * 74)
    print("SUMMARY -- FactReasoner v2 (atom-context only), Lanny Flaherty bio")
    print("=" * 74)
    print(f"{'cell':16} {'calls':>7} {'pairs':>7} {'NLI s':>9} {'score':>7} {'acc':>7}")
    for label in ("v2 all_pairs", "v2-cheap"):
        c = cells[label]
        print(
            f"{label:16} {c['llm_calls']:>7} {c['pairs_attempted']:>7} "
            f"{c['nli_seconds']:>9.1f} {c['factuality_score']:>7.4f} "
            f"{c['accuracy']['accuracy']:>7.3f}"
        )
    saved = base["pairs_attempted"] - chp["pairs_attempted"]
    print()
    print(
        f"pairs saved  : {saved} of {base['pairs_attempted']} "
        f"({base['pairs_attempted'] / max(chp['pairs_attempted'], 1):.2f}x fewer)"
    )
    if base["llm_calls"] > 0 and chp["llm_calls"] > 0:
        print(
            f"time saved   : {base['nli_seconds'] - chp['nli_seconds']:.1f}s of "
            f"{base['nli_seconds']:.1f}s "
            f"({base['nli_seconds'] / max(chp['nli_seconds'], 1e-9):.2f}x faster) "
            f"-- both cells measured live"
        )
    row = recall_rows[0]
    print(
        f"recall       : {row['recall']:.3f} "
        f"({row['non_neutral_lost']}/{row['non_neutral_total']} lost, "
        f"gate={row['gate_backend']})"
    )

    # Keep only the non-neutral count in the report; the full verdict map is
    # large and already served its purpose in the replay above.
    for cell in cells.values():
        cell["num_non_neutral"] = len(cell.get("verdicts") or {})
        for key in ("pipeline", "results", "verdicts"):
            cell.pop(key, None)
    report.update(cells=cells, recall=recall_rows, threshold_sweep=sweep_rows)
    out = os.path.join(args.output_dir, "report.json")
    with open(out, "w") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(f"\nReport written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
