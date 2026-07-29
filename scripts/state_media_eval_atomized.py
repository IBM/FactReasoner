"""
scripts/state_media_eval_atomized.py

Atomized variant of state_media_eval.py: instead of treating each
dataset row's full text as ONE compound claim, this:
  1. Runs Atomizer.run_batch() to split each row's text into individual
     atomic sub-claims (real API: Atomizer(backend).run_batch(texts)).
  2. Runs Reviser.run_batch() to decontextualize each sub-claim (resolve
     pronouns/vague references) using the original text as context
     (real API: Reviser(backend).run_batch(units, response)).
  3. Evaluates EACH sub-claim independently through the same Trust
     Fusion + Serper pipeline as state_media_eval.py.
  4. Aggregates sub-claim verdicts back to ONE post-level verdict using
     a STRICT rule: if ANY sub-claim's Trust verdict is NS, the whole
     post's aggregate verdict is NS. Only if ALL sub-claims verdict S
     does the post get S. (Confirmed choice: one false/unsupported
     sub-claim taints the whole compound statement, mirroring how
     fact-checkers grade compound claims.)

GROUND TRUTH is still applied ONLY at the post level (factual->S,
else->NS) -- we do NOT fabricate per-sub-claim ground truth, since the
dataset provides none. This script tests whether atomizing IMPROVES
the post-level decision, not whether individual sub-claims are
"correct" in isolation.

Known, deliberate limitation NOT fixed by this script: even a cleanly
atomized sub-claim can still be LITERALLY TRUE as a reporting fact
(e.g. "China's CDC director said X at a conference") while the
dataset's factuality_label is judging the CONTENT of X as biased
propaganda, not whether the attribution itself is accurate. Atomizing
fixes compound-claim ambiguity; it does NOT fix this separate label-
semantics mismatch (the model judges literal truth of a sentence, the
dataset's "biased" label often judges something narrower).
"""
import os
import sys
import re
import csv
import json
import math
import time
import asyncio
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mellea_ibm.rits import RITSBackend, RITS
from mellea.backends import ModelOption
from fact_reasoner.core.trust import BayesianTrustFusion
from fact_reasoner.core.nli import NLIExtractor as _NLIExtractor
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.reviser import Reviser
from fact_reasoner.core.base import Context, Atom
from fact_reasoner.core.utils import build_relations
from fact_reasoner.assessor import FactReasoner as FR

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "state_media_dataset.tsv")
MODEL_PATH = "/u/samit/utd_model.pkl"
SEL_PATH = "/u/samit/data/selected_features.json"
STATE_PATH = "/u/samit/dynaTD_state_state_media_atomized.json"  # SEPARATE state from the non-atomized run
MERLIN = "/u/samit/FactReasoner/merlin"
OUT_DIR = "/u/samit/state_media_results"
os.makedirs(OUT_DIR, exist_ok=True)

SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
SERPER_URL = "https://google.serper.dev/search"
N_SERPER_RESULTS = 6
MIN_USABLE_CONTEXTS = 2

LABEL_TO_GT = {
    "factual": "S",
    "false": "NS",
    "biased": "NS",
    "biased/false": "NS",
}

ADVOCACY_GOV_DOMAIN_MARKERS = (
    ".gov", "usembassy", "uhrp.org", "hrw.org", "amnesty.org",
    "freedomhouse.org", "justsecurity.org",
)


def classify_source_type(url: str) -> str:
    u = url.lower()
    if any(m in u for m in ADVOCACY_GOV_DOMAIN_MARKERS):
        return "advocacy_or_government"
    return "general"


class NLIFixed(_NLIExtractor):
    def _get_probability(self, output) -> float:
        try:
            r = output._meta["oai_chat_response"]
            lp = (r.get("choices", [{}])[0].get("logprobs") or r.get("logprobs"))
            if not lp or not lp.get("content"):
                return 1.0
            s, n = 0, 0
            for item in reversed(lp["content"][:-1]):
                if item["token"] == "[":
                    break
                elif item["token"] != "]":
                    s += item["logprob"]
                    n += 1
            return math.exp(s / n) if n > 0 else 0.0
        except Exception:
            return 1.0


def serper_search(query: str, num_results: int = N_SERPER_RESULTS) -> list:
    if not SERPER_API_KEY:
        print("  WARNING: SERPER_API_KEY not set -- skipping web search context.")
        return []
    try:
        resp = requests.post(
            SERPER_URL,
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": num_results},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        organic = data.get("organic", [])[:num_results]
        return [{"title": o.get("title", ""), "link": o.get("link", ""),
                  "snippet": o.get("snippet", "")} for o in organic]
    except Exception as e:
        print(f"  WARNING: Serper search failed ({e}) -- continuing with fewer contexts.")
        return []


def load_dataset(path: str) -> list:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gt = LABEL_TO_GT.get(row["factuality_label"].strip())
            if gt is None:
                continue
            rows.append({
                "platform": row["platform"], "date": row["date"], "text": row["text"],
                "post_url": row["post_url"], "account_name": row["account_name"].strip(),
                "category": row["category_label"], "raw_label": row["factuality_label"].strip(),
                "ground_truth": gt,
            })
    return rows


def make_pipeline(atoms_dict, contexts, relations, gt):
    p = FR.__new__(FR)
    p.atoms = atoms_dict
    p.contexts = contexts
    p.relations = relations
    p.merlin_path = MERLIN
    p.fact_graph = None
    p.markov_network = None
    p.timing = {}
    p.nli_extractor = p.atom_extractor = p.atom_reviser = None
    p.context_retriever = p.context_summarizer = None
    p.revise_atoms = p.summarize_contexts = False
    p.num_retrieved_contexts = len(contexts)
    p.num_summarized_contexts = 0
    p.use_priors = True
    p.start_time = time.perf_counter()
    p.early_exit_evaluation = False
    p.early_exit_evaluator = None
    p.labels_human = {"a0": gt}
    p.query = p.response = p.topic = ""
    p._build_fact_graph()
    p._build_markov_network()
    return p


async def eval_subclaim(subclaim_text, account, post_url, row_idx, sub_idx, nli, trust_scorer):
    """Evaluates ONE atomized sub-claim through the full Trust Fusion +
    Serper pipeline. Mirrors eval_row() from state_media_eval.py but
    operates on a single sub-claim instead of the full compound post text."""
    atom = Atom(id="a0", text=subclaim_text)
    atoms_dict = {"a0": atom}
    contexts = {}

    ctx0 = Context(id="c0", atom=atom, text=subclaim_text, title=account,
                    snippet=subclaim_text[:80], link=post_url)
    fused0 = trust_scorer.score(ctx0)
    ctx0.set_probability(fused0)
    atom.add_contexts([ctx0])
    contexts["c0"] = ctx0

    search_query = re.sub(r"#\w+|@\w+|https?://\S+", "", subclaim_text).strip()
    search_query = re.sub(r"\s+", " ", search_query)[:200]
    search_results = serper_search(search_query)
    for i, res in enumerate(search_results):
        if not res["link"] or not res["snippet"]:
            continue
        cid = f"c{i+1}"
        ctx = Context(id=cid, atom=atom, text=res["snippet"], title=res["title"],
                       snippet=res["snippet"][:80], link=res["link"])
        fused = trust_scorer.score(ctx)
        ctx.set_probability(fused)
        atom.add_contexts([ctx])
        contexts[cid] = ctx

    if len(contexts) < MIN_USABLE_CONTEXTS:
        return {"row_idx": row_idx, "sub_idx": sub_idx, "subclaim": subclaim_text,
                "skipped": True, "reason": "too few contexts from Serper"}

    # Real, transient backend failures (504 Gateway Timeout etc.) can
    # crash the whole run if unhandled -- retry a small, bounded number
    # of times with backoff before giving up on this one sub-claim.
    relations = None
    last_error = None
    for attempt in range(3):
        try:
            relations = build_relations(
                atoms=atoms_dict, contexts=contexts, nli_extractor=nli,
                rel_atom_context=True, rel_context_context=False,
                use_summarized_contexts=False,
            )
            break
        except Exception as e:
            last_error = e
            print(f"    [retry {attempt+1}/3] build_relations failed for sub-claim "
                  f"{sub_idx} ({e.__class__.__name__}: {str(e)[:100]}) -- retrying...")
            await asyncio.sleep(1.0 * (attempt + 1))
    if relations is None:
        return {"row_idx": row_idx, "sub_idx": sub_idx, "subclaim": subclaim_text,
                "skipped": True, "reason": f"NLI backend failed after 3 retries: {last_error}"}

    connected_ids = {r.source.id for r in relations} if relations else set()
    dropped_ids = set(contexts.keys()) - connected_ids
    if dropped_ids:
        for cid in dropped_ids:
            del contexts[cid]
        atom = Atom(id="a0", text=subclaim_text)
        atoms_dict = {"a0": atom}
        atom.add_contexts(list(contexts.values()))
        relations = [r for r in relations if r.source.id in contexts]

    if len(contexts) < MIN_USABLE_CONTEXTS or not relations:
        return {"row_idx": row_idx, "sub_idx": sub_idx, "subclaim": subclaim_text,
                "skipped": True, "reason": "too few usable contexts after NLI filtering"}

    pipeline = make_pipeline(atoms_dict, contexts, relations, "S")  # GT unknown at sub-claim level
    _, marginals = pipeline.score()
    p_true = next((m["probabilities"][1] for m in marginals if m["variable"] == "a0"), 0.5)
    verdict = "S" if p_true > 0.5 else "NS"

    context_details = []
    for cid, ctx in contexts.items():
        rel = next((r for r in relations if r.source.id == cid), None)
        context_details.append({
            "context_id": cid, "title": ctx.title, "link": ctx.link,
            "source_type": classify_source_type(ctx.link) if cid != "c0" else "primary_post",
            "fused_prior": round(ctx.get_probability(), 4),
            "nli_type": rel.type if rel else None,
            "nli_strength": round(rel.probability, 6) if rel else None,
        })

    trust_scorer.update_from_results(contexts, marginals, relations)

    return {
        "row_idx": row_idx, "sub_idx": sub_idx, "subclaim": subclaim_text,
        "skipped": False, "p_trust": round(p_true, 6), "verdict": verdict,
        "num_contexts": len(contexts), "contexts": context_details,
    }


async def main():
    if not SERPER_API_KEY:
        print("ERROR: SERPER_API_KEY environment variable is not set.")
        return

    trust_scorer = BayesianTrustFusion(model_path=MODEL_PATH, state_path=STATE_PATH)
    backend = RITSBackend(RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 1024})
    nli = NLIFixed(backend)
    atomizer = Atomizer(backend)
    reviser = Reviser(backend)

    rows = load_dataset(DATASET_PATH)
    print(f"Loaded {len(rows)} usable rows.")

    print("\nAtomizing all rows (real Atomizer.run_batch call) ...")
    atomized_raw = await atomizer.run_batch([r["text"] for r in rows])

    results = []
    post_correct = post_total = 0

    for i, (row, atoms_dict_raw) in enumerate(zip(rows, atomized_raw)):
        sub_units = list(atoms_dict_raw.values())
        if not sub_units:
            print(f"  [{i}] Atomizer returned no units -- skipping row entirely.")
            continue

        print(f"\n  [{i}] {row['account_name']} ({row['category']}) -- "
              f"{len(sub_units)} sub-claims extracted")

        print(f"    Revising {len(sub_units)} sub-claims (real Reviser.run_batch call) ...")
        revised = await reviser.run_batch(sub_units, row["text"])
        revised_texts = [r["revised_unit"] for r in revised] if revised else sub_units

        sub_results = []
        for j, sub_text in enumerate(revised_texts):
            try:
                sub_result = await eval_subclaim(
                    sub_text, row["account_name"], row["post_url"], i, j, nli, trust_scorer
                )
            except Exception as e:
                print(f"    sub[{j}] ERROR (unhandled, skipping this sub-claim only): "
                      f"{e.__class__.__name__}: {str(e)[:150]}")
                sub_result = {"row_idx": i, "sub_idx": j, "subclaim": sub_text,
                              "skipped": True, "reason": f"unhandled exception: {e}"}
            sub_results.append(sub_result)
            if sub_result.get("skipped"):
                print(f"    sub[{j}] SKIPPED ({sub_result['reason']}): {sub_text!r}")
            else:
                print(f"    sub[{j}] T={sub_result['p_trust']:.4f}\u2192{sub_result['verdict']} "
                      f"(n_ctx={sub_result['num_contexts']}): {sub_text!r}")

        usable_subs = [s for s in sub_results if not s.get("skipped")]
        if not usable_subs:
            print(f"  [{i}] No usable sub-claims -- skipping row's aggregate verdict.")
            continue

        # STRICT aggregation: any sub-claim NS -> whole post NS
        aggregate_verdict = "NS" if any(s["verdict"] == "NS" for s in usable_subs) else "S"
        post_correct_this = aggregate_verdict == row["ground_truth"]
        post_total += 1
        if post_correct_this:
            post_correct += 1

        sym = "\u2713" if post_correct_this else "\u2717"
        print(f"  [{i}] AGGREGATE: GT={row['ground_truth']}  verdict={aggregate_verdict} {sym}  "
              f"({len(usable_subs)}/{len(sub_results)} sub-claims usable)  "
              f"AccPost={post_correct/post_total*100:.1f}%")

        results.append({
            "row_idx": i, "account": row["account_name"], "category": row["category"],
            "raw_label": row["raw_label"], "ground_truth": row["ground_truth"],
            "original_text": row["text"], "aggregate_verdict": aggregate_verdict,
            "correct": post_correct_this, "sub_claims": sub_results,
        })

    print("\n" + "=" * 70)
    print(f"FINAL (ATOMIZED, AGGREGATE): {post_correct}/{post_total} "
          f"({post_correct/max(post_total,1)*100:.1f}%)")

    failed = [r for r in results if not r["correct"]]
    print(f"\nFAILURES: {len(failed)} of {post_total}")
    for r in failed:
        print(f"\n  [{r['row_idx']}] {r['account']:<20} {r['category']:<10} "
              f"raw_label={r['raw_label']!r}  GT={r['ground_truth']}  verdict={r['aggregate_verdict']}")
        print(f"        original: {r['original_text']!r}")
        for s in r["sub_claims"]:
            if s.get("skipped"):
                print(f"        sub[{s['sub_idx']}] SKIPPED: {s['subclaim']!r}")
            else:
                print(f"        sub[{s['sub_idx']}] {s['verdict']} (T={s['p_trust']:.4f}): {s['subclaim']!r}")

    with open(os.path.join(OUT_DIR, "state_media_results_atomized.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/state_media_results_atomized.json")


if __name__ == "__main__":
    asyncio.run(main())
