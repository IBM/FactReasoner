"""
scripts/state_media_eval.py

Tests Trust Fusion on a real, hand-labeled dataset of state-media social
posts (Twitter/Facebook accounts: Global Times, Xinhua, CGTN, ChinaDaily,
etc.), each pre-labeled with a factuality_label (factual/biased/false/
biased-false).

KNOWN, EXPLICITLY-DEFERRED LIMITATION: UTD scores URL STRUCTURE, and
twitter.com/<any-account>/status/<id> URLs are structurally near-
identical regardless of which account posted them -- UTD cannot
currently distinguish "Global Times" from "Reuters" by URL shape alone.
Per explicit decision: DynaTD tracking uses ACCOUNT NAME as the source
key (not the raw URL domain), so the system can still learn per-account
reliability over time even though UTD itself can't yet help bootstrap
that learning for social-media posts. Fixing UTD to handle this is
explicitly deferred to later work.

ATOM CONSTRUCTION, per atom (one dataset row = one atom):
  Context 1 (PRIMARY): the post's own text, scored/tracked under the
    account_name as its DynaTD key (NOT under twitter.com/facebook.com).
  Context 2+ (CORROBORATION): real, live Serper API search results for
    the same claim, scored/tracked under each result's REAL domain
    (these DO get normal UTD+DynaTD treatment, since they're real
    websites with differentiated URLs).

GROUND TRUTH mapping (confirmed):
  factual      -> S
  false         -> NS
  biased        -> NS  (not fully/accurately supported)
  biased/false  -> NS

Requires: SERPER_API_KEY environment variable (never hardcoded).
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
from fact_reasoner.core.base import Context, Atom
from fact_reasoner.core.utils import build_relations
from fact_reasoner.assessor import FactReasoner as FR

DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "state_media_dataset.tsv")
MODEL_PATH = "/u/samit/utd_model.pkl"
SEL_PATH = "/u/samit/data/selected_features.json"
STATE_PATH = "/u/samit/dynaTD_state_state_media.json"  # SEPARATE state -- does not touch the Galati trained state
MERLIN = "/u/samit/FactReasoner/merlin"
OUT_DIR = "/u/samit/state_media_results"
os.makedirs(OUT_DIR, exist_ok=True)

SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
SERPER_URL = "https://google.serper.dev/search"
N_SERPER_RESULTS = 3  # number of corroborating contexts to pull per atom

LABEL_TO_GT = {
    "factual": "S",
    "false": "NS",
    "biased": "NS",
    "biased/false": "NS",
}


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
    """
    Real Serper API call. Returns a list of {title, link, snippet}
    dicts from the organic results. Returns an empty list on any
    failure (rate limit, network error, no API key) -- never raises,
    so a Serper outage degrades to "fewer corroborating contexts"
    rather than crashing the whole atom.
    """
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
                print(f"  Skipping row with unrecognized factuality_label: {row['factuality_label']!r}")
                continue
            rows.append({
                "platform": row["platform"],
                "date": row["date"],
                "text": row["text"],
                "post_url": row["post_url"],
                "account_name": row["account_name"].strip(),
                "category": row["category_label"],
                "raw_label": row["factuality_label"].strip(),
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


async def eval_row(row, trust_scorer, nli, row_idx):
    claim = row["text"][:300]  # cap length for the atom claim text
    gt = row["ground_truth"]
    account = row["account_name"]

    atom = Atom(id="a0", text=claim)
    atoms_dict = {"a0": atom}
    contexts = {}

    # ---- Context 1: the post's own text, tracked under ACCOUNT NAME ----
    # IMPORTANT: rather than assume BayesianTrustFusion has some
    # account-keyed scoring method (unverified -- I have not seen
    # bayesian_fusion.py's real internals for this dataset's use case),
    # this constructs a SYNTHETIC link whose DOMAIN portion is a
    # slugified version of the account name. DynaTD/UTD both key off
    # Context.link's domain via the existing, already-verified
    # score()/update_from_results() machinery -- so this achieves
    # account-level tracking using ONLY real, already-confirmed APIs,
    # with no new/unverified methods required.
    account_slug = re.sub(r"[^a-z0-9]+", "-", account.lower()).strip("-")
    # NOTE: avoid "account" in the synthetic domain -- it collides with
    # UTD's has_login_keyword feature (which matches "account" as a
    # phishing-adjacent term), which would spuriously flag EVERY
    # synthetic account-domain as suspicious, equally and uniformly.
    # Using "social-src" instead avoids this false signal.
    synthetic_link = f"https://{account_slug}.social-src.local/post"
    ctx0 = Context(id="c0", atom=atom, text=row["text"], title=account,
                    snippet=row["text"][:80], link=synthetic_link)
    fused0 = trust_scorer.score(ctx0)
    ctx0.set_probability(fused0)
    atom.add_contexts([ctx0])
    contexts["c0"] = ctx0

    # ---- Context 2+: real Serper results, tracked by their REAL domain ----
    search_results = serper_search(row["text"][:200])
    for i, res in enumerate(search_results):
        if not res["link"] or not res["snippet"]:
            continue
        cid = f"c{i+1}"
        ctx = Context(id=cid, atom=atom, text=res["snippet"], title=res["title"],
                       snippet=res["snippet"][:80], link=res["link"])
        fused = trust_scorer.score(ctx)  # real domain -> normal UTD+DynaTD path
        ctx.set_probability(fused)
        atom.add_contexts([ctx])
        contexts[cid] = ctx

    if len(contexts) < 2:
        print(f"  [{row_idx}] Only {len(contexts)} context(s) (Serper returned nothing usable) -- skipping atom.")
        return None

    relations = build_relations(
        atoms=atoms_dict, contexts=contexts, nli_extractor=nli,
        rel_atom_context=True, rel_context_context=False,
        use_summarized_contexts=False,
    )
    if not relations or len(relations) < len(contexts):
        print(f"  [{row_idx}] build_relations() returned {len(relations) if relations else 0} of "
              f"{len(contexts)} expected -- skipping atom (avoid silently-dropped-source bug).")
        return None

    pipeline = make_pipeline(atoms_dict, contexts, relations, gt)
    _, marginals = pipeline.score()
    p_true = next((m["probabilities"][1] for m in marginals if m["variable"] == "a0"), 0.5)
    verdict = "S" if p_true > 0.5 else "NS"

    # Vanilla comparison
    for ctx in contexts.values():
        ctx.set_probability(0.9)
    pipeline.fact_graph = None
    pipeline.markov_network = None
    pipeline._build_fact_graph()
    pipeline._build_markov_network()
    _, van_marginals = pipeline.score()
    p_van = next((m["probabilities"][1] for m in van_marginals if m["variable"] == "a0"), 0.5)
    van_verdict = "S" if p_van > 0.5 else "NS"

    # Update DynaTD: account-level for c0, real-domain-level for the rest
    trust_scorer.update_from_results(contexts, marginals, relations)

    return {
        "row_idx": row_idx, "account": account, "category": row["category"],
        "raw_label": row["raw_label"], "ground_truth": gt,
        "p_trust": p_true, "verdict": verdict, "correct": verdict == gt,
        "p_van": p_van, "van_verdict": van_verdict, "van_correct": van_verdict == gt,
        "num_contexts": len(contexts),
    }


async def main():
    if not SERPER_API_KEY:
        print("ERROR: SERPER_API_KEY environment variable is not set. "
              "Run: export SERPER_API_KEY=<your key>")
        return

    trust_scorer = BayesianTrustFusion(model_path=MODEL_PATH, state_path=STATE_PATH)
    backend = RITSBackend(RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 1024})
    nli = NLIFixed(backend)

    rows = load_dataset(DATASET_PATH)
    print(f"Loaded {len(rows)} usable rows (factuality_label recognized).")

    results = []
    trust_correct = van_correct = total = 0

    for i, row in enumerate(rows):
        try:
            result = await eval_row(row, trust_scorer, nli, i)
        except Exception as e:
            print(f"  [{i}] ERROR: {e}")
            continue
        if result is None:
            continue
        results.append(result)
        total += 1
        if result["correct"]:
            trust_correct += 1
        if result["van_correct"]:
            van_correct += 1
        t_sym = "\u2713" if result["correct"] else "\u2717"
        v_sym = "\u2713" if result["van_correct"] else "\u2717"
        print(f"  [{i:>2}] {result['account']:<20} {result['category']:<10} "
              f"GT={result['ground_truth']:<3} T={result['p_trust']:.3f}\u2192{result['verdict']} {t_sym}  "
              f"V={result['p_van']:.3f}\u2192{result['van_verdict']} {v_sym}  "
              f"(n_ctx={result['num_contexts']})  "
              f"AccT={trust_correct/total*100:.1f}% AccV={van_correct/total*100:.1f}%")

    print("\n" + "=" * 70)
    print(f"FINAL: Trust {trust_correct}/{total} ({trust_correct/max(total,1)*100:.1f}%)  "
          f"Vanilla {van_correct}/{total} ({van_correct/max(total,1)*100:.1f}%)")

    with open(os.path.join(OUT_DIR, "state_media_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {OUT_DIR}/state_media_results.json")


if __name__ == "__main__":
    asyncio.run(main())
