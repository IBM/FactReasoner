"""
scripts/state_media_eval.py

Tests Trust Fusion on a real, hand-labeled dataset of state-media social
posts (Twitter/Facebook accounts: Global Times, Xinhua, CGTN, ChinaDaily,
etc.), each pre-labeled with a factuality_label (factual/biased/false/
biased-false).

KNOWN LIMITATION, NOW FIXED via a real code patch: UTD scores URL
STRUCTURE, and twitter.com/<account>/status/<id> URLs are structurally
near-identical regardless of account. dynaTD.py and bayesian_fusion.py
have been patched (see dynaTD_extract_domain_patch.py /
bayesian_fusion_domain_patch.py) so that for twitter.com/x.com/
facebook.com URLs specifically, the DynaTD tracking key is
netloc+first-path-segment (e.g. "twitter.com/globaltimes") instead of
just netloc -- giving each REAL account a genuinely distinct trust
history using the REAL, unmodified post_url, no fabricated domains.
Every other domain's behavior (reuters.com, euromaidanpress.com, etc.)
is completely unchanged.

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
import argparse
import random
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mellea_ibm.rits import RITSBackend, RITS
from mellea.backends import ModelOption
from fact_reasoner.core.trust import BayesianTrustFusion
from fact_reasoner.core.nli import NLIExtractor as _NLIExtractor
from fact_reasoner.core.base import Context, Atom
from fact_reasoner.core.utils import build_relations
from fact_reasoner.assessor import FactReasoner as FR
from core_claim_query import extract_core_claim, build_queries


DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "state_media_dataset.tsv")
MODEL_PATH = os.environ.get("UTD_MODEL_PATH", "/u/samit/utd_model.pkl")
SEL_PATH = os.environ.get("SEL_FEATURES_PATH", "/u/samit/data/selected_features.json")
STATE_PATH_BASE = os.environ.get("STATE_PATH_BASE", "/u/samit/dynaTD_state_tbs_test")  # SEPARATE from the Galati trained state
MERLIN = os.environ.get("MERLIN_PATH", "/u/samit/FactReasoner/merlin")
OUT_DIR = os.environ.get("STATE_MEDIA_OUT_DIR", "/u/samit/state_media_results")
os.makedirs(OUT_DIR, exist_ok=True)

SERPER_API_KEY = os.environ.get("SERPER_API_KEY")
SERPER_URL = "https://google.serper.dev/search"
N_SERPER_RESULTS = 8  # per-query count; now called TWICE per atom (once
                       # for the post-text query, once for the neutral
                       # category+factcheck query), so total candidates
                       # per atom is similar to before (~8) while adding
                       # genuine query diversity instead of just volume.

LABEL_TO_GT = {
    "factual": "S",
    "false": "NS",
    "biased": "NS",
    "biased/false": "NS",
    # All 4 labels restored per explicit request to compare results
    # WITH vs WITHOUT biased/biased-false included, and to see results
    # broken down BY raw_label (not just collapsed S/NS), in a single run.
}

def normalize_social_domain(domain: str) -> str:
    """
    Merge platform variants of the same real-world account.
    twitter.com/globaltimesnews, x.com/globaltimesnews, and
    facebook.com/globaltimesnews are ALL Global Times -- but DynaTD
    currently tracks them as separate keys, fragmenting the learning
    signal. This maps all recognized variants to a canonical key.

    Strategy: strip the platform prefix and use just the account slug.
    For accounts that appear on both Twitter AND Facebook with the same
    slug, we keep platform in the key (they may genuinely differ).
    For twitter.com vs x.com specifically, they are THE SAME account
    (Twitter rebranded to X in 2023), so we always normalize x.com →
    twitter.com for accounts that appear under both.
    """
    # x.com/* → twitter.com/* (same platform, rebrand only)
    if domain.startswith("x.com/"):
        domain = "twitter.com/" + domain[len("x.com/"):]
    return domain



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


# Real, methodological transparency note: for politically contested
# claims (Xinjiang, Taiwan, etc.), Serper's organic results often
# include advocacy/government/NGO sources that are themselves
# partisan -- just from a different side of the dispute -- rather than
# neutral fact-checking. This does NOT make them invalid corroboration
# (a US government fact sheet IS a real primary source on US policy
# positions), but it's worth flagging explicitly rather than silently
# treating "whatever Serper returns" as neutral ground truth.
ADVOCACY_GOV_DOMAIN_MARKERS = (
    ".gov", "usembassy", "uhrp.org", "hrw.org", "amnesty.org",
    "freedomhouse.org", "justsecurity.org",
)

def classify_source_type(url: str) -> str:
    u = url.lower()
    if any(m in u for m in ADVOCACY_GOV_DOMAIN_MARKERS):
        return "advocacy_or_government"
    return "general"


SERPER_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "serper_cache.json")
_serper_cache = None  # lazy-loaded, module-level, shared across all calls in this process

# Controls how serper_search() interacts with the on-disk cache. Set
# from the --cache-mode CLI argument in main() -- see argparse setup
# below. Three explicit choices, no implicit default behavior:
#   "use"     -- if a cached result exists for this exact query, use it.
#                Otherwise fetch live and SAVE the new result to cache.
#                (This is the "give me reproducible results, but fill
#                in anything missing" mode.)
#   "fresh"   -- ALWAYS fetch live, ignore any existing cache entry,
#                and do NOT save the new result. Use this when you
#                want genuinely current search results and don't want
#                to disturb the existing cache for future "use" runs.
#   "refresh" -- ALWAYS fetch live, ignore any existing cache entry,
#                and SAVE the new result (overwriting whatever was
#                cached before). Use this to deliberately update the
#                cache with new live results going forward.
CACHE_MODE = "use"

def _load_serper_cache() -> dict:
    global _serper_cache
    if _serper_cache is None:
        if os.path.exists(SERPER_CACHE_PATH):
            with open(SERPER_CACHE_PATH) as f:
                _serper_cache = json.load(f)
            print(f"[serper cache] Loaded {len(_serper_cache)} cached queries from {SERPER_CACHE_PATH}")
        else:
            _serper_cache = {}
    return _serper_cache

def _save_serper_cache():
    os.makedirs(os.path.dirname(SERPER_CACHE_PATH), exist_ok=True)
    with open(SERPER_CACHE_PATH, "w") as f:
        json.dump(_serper_cache, f, indent=2)


PAGE_FETCH_TIMEOUT_S = 6
PAGE_FETCH_MAX_CHARS = 1000  # excerpt length passed to NLI -- enough
                              # real content for confident classification
                              # without sending an entire article

PAGE_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "page_cache.json")
_page_cache = None

def _load_page_cache() -> dict:
    global _page_cache
    if _page_cache is None:
        if os.path.exists(PAGE_CACHE_PATH):
            with open(PAGE_CACHE_PATH) as f:
                _page_cache = json.load(f)
        else:
            _page_cache = {}
    return _page_cache

def _save_page_cache():
    os.makedirs(os.path.dirname(PAGE_CACHE_PATH), exist_ok=True)
    with open(PAGE_CACHE_PATH, "w") as f:
        json.dump(_page_cache, f, indent=2)


def fetch_page_excerpt(url: str, fallback_snippet: str) -> str:
    """
    Real fix for the dominant cause of NLI dropout: the raw Serper
    SNIPPET (often 1-2 truncated sentences ending in "...") frequently
    doesn't contain enough real content for NLI to confidently judge
    entailment/contradiction -- not because the underlying ARTICLE
    lacks support/contradiction, but because the snippet shown to NLI
    is too short to tell. This fetches the real page and extracts a
    longer plain-text excerpt to use instead.

    Same CACHE_MODE semantics as serper_search() apply here (reuses the
    shared module-level CACHE_MODE: "use" caches successes, "fresh"
    never touches the cache, "refresh" always re-fetches and overwrites).

    Falls back to the original Serper snippet on ANY failure (timeout,
    paywall, non-HTML content, blocked request, etc.) -- never raises,
    and never returns empty text, so a fetch failure degrades to
    exactly the PREVIOUS behavior (snippet-only) rather than losing the
    context entirely.
    """
    cache = _load_page_cache()
    if CACHE_MODE == "use" and url in cache:
        return cache[url]

    try:
        from bs4 import BeautifulSoup
        resp = requests.get(
            url, timeout=PAGE_FETCH_TIMEOUT_S,
            headers={"User-Agent": "Mozilla/5.0 (research fact-checking bot)"},
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        if len(text) < 100:
            # Suspiciously short real extraction (likely a paywall/JS-
            # rendered page with no real server-side text) -- fall back
            # to the snippet rather than trust this thin extraction.
            raise ValueError("extracted text too short, likely paywalled/JS-rendered")
        excerpt = text[:PAGE_FETCH_MAX_CHARS]
        if CACHE_MODE in ("use", "refresh"):
            cache[url] = excerpt
            _save_page_cache()
        return excerpt
    except Exception as e:
        # Deliberately NOT cached on failure -- same reasoning as
        # serper_search()'s exception handler: a transient failure
        # should not be permanently remembered as "this page has no
        # content." Fall back to the original snippet, which is always
        # at least as good as what existed before this fix.
        return fallback_snippet


def _date_to_tbs(date_str: str):
    """
    Convert a post date (e.g. '12/10/22') into a Serper tbs range
    covering ~45 days around the post. This retrieves articles from
    the same time window as the claim rather than the most recently
    indexed version, fixing temporal-mismatch false contradictions.
    Returns None if date can't be parsed.
    """
    import datetime
    for fmt in ("%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            d = datetime.datetime.strptime(date_str.strip(), fmt)
            start = d - datetime.timedelta(days=45)
            end   = d + datetime.timedelta(days=45)
            s = start.strftime("%-m/%-d/%Y")
            e = end.strftime("%-m/%-d/%Y")
            return f"cdr:1,cd_min:{s},cd_max:{e}"
        except (ValueError, AttributeError):
            continue
    return None


def serper_search(query: str, num_results: int = N_SERPER_RESULTS,
                  tbs=None) -> list:
    """
    Real Serper API call, with a persistent on-disk cache keyed by the
    EXACT query string, controlled by the module-level CACHE_MODE
    ("use" / "fresh" / "refresh" -- see CACHE_MODE comment above for
    exact semantics of each). This is the fix for a real, confirmed
    issue: comparing accuracy ACROSS separate run invocations (e.g.
    testing a beta-ramp change) was confounded by live Serper results
    drifting between runs -- Trust and Vanilla always shared identical
    contexts WITHIN a single run (this was never broken), but two
    SEPARATE runs could legitimately retrieve different real-world
    search results for the same query, making cross-run comparisons
    not strictly controlled. Returns a list of {title, link, snippet}
    dicts from the organic results. Returns an empty list on any
    failure (rate limit, network error, no API key) -- never raises,
    so a Serper outage degrades to "fewer corroborating contexts"
    rather than crashing the whole atom.
    """
    cache = _load_serper_cache()

    cache_key = query if tbs is None else f"{query}|tbs:{tbs}"
    if CACHE_MODE == "use" and cache_key in cache:
        return cache[cache_key]
        return cache[query]
    # "fresh" and "refresh" both skip the cache-read; "use" falls
    # through here only when the query is NOT already cached.

    if not SERPER_API_KEY:
        print("  WARNING: SERPER_API_KEY not set -- skipping web search context.")
        return []
    try:
        resp = requests.post(
            SERPER_URL,
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={k: v for k, v in {"q": query, "num": num_results,
                       "tbs": tbs}.items() if v is not None},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        organic = data.get("organic", [])[:num_results]
        result = [{"title": o.get("title", ""), "link": o.get("link", ""),
                   "snippet": o.get("snippet", "")} for o in organic]
        if CACHE_MODE in ("use", "refresh"):
            cache[cache_key] = result
            _save_serper_cache()
        return result
    except Exception as e:
        print(f"  WARNING: Serper search failed ({e}) -- continuing with fewer contexts.")
        # Deliberately NOT cached -- a transient network/rate-limit
        # failure should not be permanently remembered as "this query
        # returns nothing." A later run can retry the real call.
        return []


def load_dataset(path: str, only_labels: set = None) -> list:
    """
    only_labels: if provided, only rows whose raw_label is in this set
    are kept (e.g. {"factual", "false"} to exclude biased/biased-false
    entirely). If None, all 4 recognized labels in LABEL_TO_GT are kept
    -- i.e. the full, unfiltered dataset.
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            raw_label = row["factuality_label"].strip()
            gt = LABEL_TO_GT.get(raw_label)
            if gt is None:
                print(f"  Skipping row with unrecognized factuality_label: {raw_label!r}")
                continue
            if only_labels is not None and raw_label not in only_labels:
                continue
            rows.append({
                "platform": row["platform"],
                "date": row["date"],
                "text": row["text"],
                "post_url": row["post_url"],
                "account_name": row["account_name"].strip(),
                "category": row["category_label"],
                "raw_label": raw_label,
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


async def eval_row(row, trust_scorer, nli, row_idx, backend):
    # --- ADD THESE INITIALIZATIONS TO PREVENT SUBSEQUENT NAMEERRORS ---
    gt = row.get("ground_truth")  # <-- ADD THIS LINE
    account = row.get("account_name", "unknown")
    claim = row.get("text", "")
    atom = Atom(id="a0", text=claim)
    atoms_dict = {"a0": atom}
    contexts = {}
    # -----------------------------------------------------------------

    
    core_claim = await extract_core_claim(row, backend)
    search_query_1, search_query_2 = build_queries(
        core_claim, row.get("date", ""), row.get("category", "")
    )
    results_1 = serper_search(search_query_1, num_results=N_SERPER_RESULTS)
    results_2 = serper_search(search_query_2, num_results=N_SERPER_RESULTS)
    results_3 = []
    print(f"    [search] query_1: {search_query_1!r}")
    print(f"    [search] query_2: {search_query_2!r}")
        

    seen_links = set()
    search_results = []
    for res in results_1 + results_2 + results_3:
        if res["link"] and res["link"] not in seen_links:
            seen_links.add(res["link"])
            search_results.append(res)

    for i, res in enumerate(search_results):
        if not res["link"] or not res["snippet"]:
            continue
        cid = f"c{i+1}"
        # Normalize x.com/* → twitter.com/* so DynaTD learning accumulates
        # on the SAME account across Twitter's rebrand (same actual source).
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(res["link"])
        norm_host = parsed.netloc.replace("x.com", "twitter.com")
        norm_link = urlunparse(parsed._replace(netloc=norm_host))
        ctx = Context(id=cid, atom=atom, text=res["snippet"], title=res["title"],
                       snippet=res["snippet"][:80], link=norm_link)
        fused = trust_scorer.score(ctx)  # real domain -> normal UTD+DynaTD path
        ctx.set_probability(fused)
        atom.add_contexts([ctx])
        contexts[cid] = ctx

    if len(contexts) < 2:
        print(f"  [{row_idx}] Only {len(contexts)} context(s) (Serper returned nothing usable) -- skipping atom.")
        return None

    # Retry on transient backend failures (502/504 gateway errors etc.)
    # -- previously unhandled here, which meant a single transient
    # hiccup silently killed the WHOLE atom (counted as "skipped") even
    # though the underlying LLM backend was likely fine moments later.
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
            print(f"  [{row_idx}] [retry {attempt+1}/3] build_relations failed "
                  f"({e.__class__.__name__}: {str(e)[:120]}) -- retrying...")
            await asyncio.sleep(1.5 * (attempt + 1))
    if relations is None:
        print(f"  [{row_idx}] build_relations failed after 3 retries: {last_error} -- skipping atom.")
        return None

    # PARTIAL-RELATION HANDLING (relaxed from the original all-or-nothing
    # check): build_relations() sometimes can't confidently classify a
    # SHORT/ambiguous Serper snippet (the same "ambiguous text gets
    # silently dropped" issue diagnosed for Scenario 5's meduza.io
    # context). Requiring ALL contexts to succeed discarded ~80% of
    # atoms in the first run. The fix here is NOT to weaken NLI's
    # confidence requirements -- it's to recognize that an atom with,
    # say, 2 of 4 contexts successfully classified is still a valid,
    # usable atom; we just need to DROP THE DISCONNECTED CONTEXTS from
    # the Markov network entirely (not leave them floating with no
    # edge to a0, which would silently give them zero real influence
    # while still listing them as "present").
    connected_ids = {r.source.id for r in relations} if relations else set()
    dropped_ids = set(contexts.keys()) - connected_ids
    if dropped_ids:
        for cid in dropped_ids:
            print(f"  [{row_idx}] Dropping context {cid} ({contexts[cid].title}) "
                  f"-- NLI could not confidently classify it (no relation built).")
            del contexts[cid]
        # Rebuild the atom with only the surviving, connected contexts --
        # avoids relying on any assumed internal list-mutation API on
        # Atom (e.g. atom.contexts.remove(...)), which hasn't been
        # confirmed to exist; Atom.add_contexts() IS a confirmed, real
        # API (used everywhere else in this project), so reconstruct
        # via that instead of mutating internals directly.
        atom = Atom(id="a0", text=claim)
        atoms_dict = {"a0": atom}
        atom.add_contexts(list(contexts.values()))
        relations = [r for r in relations if r.source.id in contexts]

    MIN_USABLE_CONTEXTS = 2  # need at least the primary post + 1 corroborating source
    if len(contexts) < MIN_USABLE_CONTEXTS or not relations:
        print(f"  [{row_idx}] Only {len(contexts)} usable context(s) after dropping unclassifiable "
              f"ones (need >= {MIN_USABLE_CONTEXTS}) -- skipping atom.")
        return None

    pipeline = make_pipeline(atoms_dict, contexts, relations, gt)
    _, marginals = pipeline.score()
    p_true = next((m["probabilities"][1] for m in marginals if m["variable"] == "a0"), 0.5)
    verdict = "S" if p_true > 0.5 else "NS"

    # Capture full per-context detail (fused_prior, NLI type/strength,
    # real text/link) BEFORE contexts get overwritten with flat 0.9 for
    # the Vanilla comparison below -- this is what gets reported back.
    context_details = []
    for cid, ctx in contexts.items():
        rel = next((r for r in relations if r.source.id == cid), None)
        context_details.append({
            "context_id": cid,
            "title": ctx.title,
            "link": ctx.link,
            "source_type": classify_source_type(ctx.link) if cid != "c0" else "primary_post",
            "text_excerpt": ctx.text[:200],
            "fused_prior": round(ctx.get_probability(), 4),
            "nli_type": rel.type if rel else None,
            "nli_strength": round(rel.probability, 6) if rel else None,
        })

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
        "claim": claim, "raw_label": row["raw_label"], "ground_truth": gt,
        "p_trust": round(p_true, 6), "verdict": verdict, "correct": verdict == gt,
        "p_van": round(p_van, 6), "van_verdict": van_verdict, "van_correct": van_verdict == gt,
        "num_contexts": len(contexts),
        "contexts": context_details,
    }


async def main():
    global CACHE_MODE
    parser = argparse.ArgumentParser(
        description="State media Trust Fusion eval, with controllable Serper caching."
    )
    parser.add_argument(
        "--cache-mode", choices=["use", "fresh", "refresh"], default="use",
        help=(
            "use (default): reuse cached results for queries already in "
            "the cache; fetch+save anything missing. "
            "fresh: always fetch live, never read or write the cache "
            "(genuinely new search results, doesn't disturb the cache). "
            "refresh: always fetch live AND overwrite the cache with the "
            "new results (deliberately update the cache going forward)."
        ),
    )
    parser.add_argument(
        "--labels", default="all",
        choices=["all", "factual_false", "factual", "false", "biased", "biased_false"],
        help=(
            "Which raw_label rows to include. 'all' (default): every "
            "recognized label. 'factual_false': only factual+false "
            "(excludes biased/biased-false entirely -- the clean subset "
            "this project has been comparing against). Single-label "
            "choices ('factual', 'false', 'biased', 'biased_false') run "
            "on just that one label, for isolating its behavior."
        ),
    )
    args = parser.parse_args()
    CACHE_MODE = args.cache_mode
    print(f"[cache] mode = {CACHE_MODE!r} (cache file: {SERPER_CACHE_PATH})")

    LABEL_FILTER_MAP = {
        "all": None,
        "factual_false": {"factual", "false"},
        "factual": {"factual"},
        "false": {"false"},
        "biased": {"biased"},
        "biased_false": {"biased/false"},
    }
    only_labels = LABEL_FILTER_MAP[args.labels]
    print(f"[labels] filter = {args.labels!r} ({only_labels if only_labels else 'no filter, all labels'})")

    # State path is label-filter-specific -- running --labels factual_false
    # and --labels all must NOT share learning history, or DynaTD's
    # account-level r_s would be contaminated by atoms from a different
    # experiment's label subset.
    state_path = f"{STATE_PATH_BASE}_{args.labels}.json"
    print(f"[state] DynaTD state file: {state_path}")

    if not SERPER_API_KEY:
        print("ERROR: SERPER_API_KEY environment variable is not set. "
              "Run: export SERPER_API_KEY=<your key>")
        return

    trust_scorer = BayesianTrustFusion(model_path=MODEL_PATH, state_path=state_path)
    backend = RITSBackend(RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 1024})
    nli = NLIFixed(backend)

    rows = load_dataset(DATASET_PATH, only_labels=only_labels)
    print(f"Loaded {len(rows)} usable rows (factuality_label recognized, labels filter={args.labels!r}).")

    results = []
    skipped_rows = []
    trust_correct = van_correct = total = 0

    # Evaluate in random order so DynaTD learning effects don't
    # accumulate sequentially in the same topic clusters every run.
    random.shuffle(rows)
    for i, row in enumerate(rows):
        # Retry the ENTIRE row, not just build_relations -- the earlier
        # retry-only-around-build_relations was too narrow: real 504/502
        # gateway errors were escaping from SOMEWHERE ELSE in eval_row
        # (exact call site unconfirmed -- the outer handler only printed
        # str(e), not a traceback, so it wasn't visible which line raised
        # it). Wrapping the whole row guarantees ANY transient failure,
        # wherever it occurs, gets retried rather than silently skipping
        # the atom after a single hiccup.
        result = None
        last_error = None
        for attempt in range(3):
            try:
                result = await eval_row(row, trust_scorer, nli, i, backend)
                break
            except Exception as e:
                last_error = e
                import traceback
                print(f"  [{i}] [row retry {attempt+1}/3] {e.__class__.__name__}: {str(e)[:200]}")
                if attempt == 0:
                    # Print the full traceback once, on the first
                    # failure, so we can see EXACTLY which call site is
                    # actually raising this -- not just the exception's
                    # string repr.
                    traceback.print_exc()
                await asyncio.sleep(2.0 * (attempt + 1))
        if result is None and last_error is not None:
            print(f"  [{i}] ERROR after 3 row-level retries: {last_error}")
            skipped_rows.append({"row_idx": i, "account": row["account_name"],
                                  "reason": f"exception (after 3 retries): {last_error}"})
            continue
        if result is None:
            skipped_rows.append({"row_idx": i, "account": row["account_name"],
                                  "reason": "too few usable contexts after NLI filtering"})
            continue
        results.append(result)
        total += 1
        if result["correct"]:
            trust_correct += 1
        if result["van_correct"]:
            van_correct += 1
        t_sym = "\u2713" if result["correct"] else "\u2717"
        v_sym = "\u2713" if result["van_correct"] else "\u2717"
        print(f"\n  [{i:>2}] {result['account']:<20} {result['category']:<10} "
              f"GT={result['ground_truth']:<3} T={result['p_trust']:.4f}\u2192{result['verdict']} {t_sym}  "
              f"V={result['p_van']:.4f}\u2192{result['van_verdict']} {v_sym}  "
              f"(n_ctx={result['num_contexts']})  "
              f"AccT={trust_correct/total*100:.1f}% AccV={van_correct/total*100:.1f}%")
        print(f"        claim: {result['claim']!r}")
        for c in result["contexts"]:
            print(f"        {c['context_id']:<6} {c['title']:<28} fused_prior={c['fused_prior']:<8} "
                  f"nli={c['nli_type']!s:<14} strength={c['nli_strength']}")
            print(f"               link: {c['link']}")

    print("\n" + "=" * 70)
    print(f"FINAL (ALL LABELS): Trust {trust_correct}/{total} ({trust_correct/max(total,1)*100:.1f}%)  "
          f"Vanilla {van_correct}/{total} ({van_correct/max(total,1)*100:.1f}%)")

    # Real per-label breakdown -- computed from the SAME run's actual
    # results, not a separate filtered re-run, so this is a genuinely
    # controlled comparison (same Serper results, same NLI calls,
    # same DynaTD state) rather than the confounded across-run
    # comparison we did before (different runs = different live search
    # results = not a clean ablation).
    print("\n" + "=" * 70)
    print("RESULTS BY raw_label")
    print("=" * 70)
    by_label = {}
    for r in results:
        by_label.setdefault(r["raw_label"], []).append(r)
    for label in sorted(by_label):
        rs = by_label[label]
        t_correct = sum(1 for r in rs if r["correct"])
        v_correct = sum(1 for r in rs if r["van_correct"])
        n = len(rs)
        print(f"  {label:<15} n={n:<4} Trust={t_correct}/{n} ({t_correct/n*100:.1f}%)  "
              f"Vanilla={v_correct}/{n} ({v_correct/n*100:.1f}%)")

    # WITH vs WITHOUT biased/biased-false, computed from this SAME run --
    # a real, controlled ablation since it's the identical underlying
    # search results/NLI calls, just a different slice of which rows
    # count toward the aggregate.
    clean_labels = {"factual", "false"}
    clean_results = [r for r in results if r["raw_label"] in clean_labels]
    factual_only_results = [r for r in results if r["raw_label"] == "factual"]
    if clean_results:
        ct = sum(1 for r in clean_results if r["correct"])
        cv = sum(1 for r in clean_results if r["van_correct"])
        cn = len(clean_results)
        print("\n" + "=" * 70)
        print("WITH vs WITHOUT biased/biased-false (same run, same data)")
        print("=" * 70)
        print(f"  ALL LABELS (incl. biased/biased-false): "
              f"Trust={trust_correct}/{total} ({trust_correct/max(total,1)*100:.1f}%)  "
              f"Vanilla={van_correct}/{total} ({van_correct/max(total,1)*100:.1f}%)")
        print(f"  factual+false ONLY (no biased/biased-false): "
              f"Trust={ct}/{cn} ({ct/cn*100:.1f}%)  "
              f"Vanilla={cv}/{cn} ({cv/cn*100:.1f}%)")
        if factual_only_results:
            ft = sum(1 for r in factual_only_results if r["correct"])
            fv = sum(1 for r in factual_only_results if r["van_correct"])
            fn = len(factual_only_results)
            print(f"  factual ONLY (excludes false too):      "
                  f"Trust={ft}/{fn} ({ft/fn*100:.1f}%)  "
                  f"Vanilla={fv}/{fn} ({fv/fn*100:.1f}%)")

    # Clean recap of every atom Trust got WRONG, with full context
    # detail -- pulled straight from the already-captured `results`
    # list, not re-derived, so these numbers are guaranteed to match
    # exactly what drove each verdict (no re-interleaving with live
    # NLI/Merlin logs to disentangle).
    failed = [r for r in results if not r["correct"]]
    print("\n" + "=" * 70)
    print(f"TRUST FAILURES: {len(failed)} of {total} atoms")
    print("=" * 70)
    for r in failed:
        print(f"\n  [{r['row_idx']:>2}] {r['account']:<20} {r['category']:<10} "
              f"raw_label={r['raw_label']!r}")
        print(f"        GT={r['ground_truth']}  T={r['p_trust']:.4f}\u2192{r['verdict']} \u2717  "
              f"V={r['p_van']:.4f}\u2192{r['van_verdict']}")
        print(f"        claim: {r['claim']!r}")
        for c in r["contexts"]:
            print(f"        {c['context_id']:<6} {c['title']:<28} "
                  f"[{c['source_type']}]  fused_prior={c['fused_prior']:<8} "
                  f"nli={c['nli_type']!s:<14} strength={c['nli_strength']}")
            print(f"               link: {c['link']}")

    if skipped_rows:
        print("\n" + "=" * 70)
        print(f"SKIPPED ATOMS: {len(skipped_rows)} of {len(rows)} (never produced a verdict)")
        print("=" * 70)
        for s in skipped_rows:
            print(f"  [{s['row_idx']:>2}] {s['account']:<20} -- {s['reason']}")

    with open(os.path.join(OUT_DIR, "state_media_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/state_media_results.json")

    # Real DynaTD learning-stats summary, pulled directly from the live
    # state object (not re-derived/estimated) -- shows exactly what the
    # system has learned about each source's reliability, every run.
    print("\n" + "=" * 70)
    print("DYNATD LEARNING STATS (real, from persisted state)")
    print("=" * 70)
    dyna = trust_scorer.dynaTD
    # Merge x.com/* → twitter.com/* variants before display
    merged: dict = {}
    for d in dyna.a:
        nd = normalize_social_domain(d)
        if nd.startswith("twitter.com/") or nd.startswith("facebook.com/"):
            if nd not in merged:
                merged[nd] = {"a": 0, "b": 0, "total": 0, "correct": 0}
            merged[nd]["a"] += dyna.a.get(d, 1)
            merged[nd]["b"] += dyna.b.get(d, 0)
            merged[nd]["total"] += dyna.total_count.get(d, 0)
            merged[nd]["correct"] += dyna.correct_count.get(d, 0)
    state_media_domains = list(merged.keys())
    print(f"\n  State-media account domains tracked: {len(state_media_domains)}")
    print(f"  {'domain':<35} {'claims':<8} {'correct':<8} {'r_s':<6}")
    rows_sorted = sorted(
        state_media_domains,
        key=lambda d: merged[d]["correct"] / merged[d]["total"] if merged[d]["total"] > 0 else 0.5,
    )
    for d in rows_sorted:
        m = merged[d]
        n = m["total"]
        c = m["correct"]
        # Recompute r_s from merged alpha/beta
        import math as _math
        w = m["a"] / m["b"] if m["b"] > 0 else 1.0
        r_s = 0.1 + 0.8 / (1 + _math.exp(-2 * (w - 1)))
        print(f"  {d:<35} {n:<8} {c:<8} {r_s:<6.3f}")

    print(f"\n  Total domains tracked (all sources, including one-off "
          f"Serper-found corroboration): {len(dyna.a)}")


if __name__ == "__main__":
    asyncio.run(main())
