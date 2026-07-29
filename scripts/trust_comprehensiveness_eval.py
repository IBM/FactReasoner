"""
trust_comprehensiveness_eval.py  (v2)
======================================
Three-way evaluation on the Chinese state media dataset.

KEY DESIGN DECISION (v2):
  Social media posts are 1-3 sentences. They cannot "cover" web articles
  in the comprehensiveness-paper sense. Instead we adapt the framework:

  The NLI direction is REVERSED from the paper:
    Paper:   Does the RESPONSE cover the CONTEXT facts?  (response is long)
    Ours:    Does the CONTEXT support/contradict the POST? (post is the atom)

  This is exactly what FactReasoner already does — but we now use it to
  compute an EVIDENCE DISTRIBUTION SCORE rather than a binary verdict.

SYSTEMS:
  A — Vanilla Evidence Coverage (adapted from Dejl et al. 2025)
      For each retrieved context C_i:
        classify as: entailment / contradiction / neither
      Score_A = |entailment| / (|entailment| + |contradiction|)
      All sources weighted equally regardless of credibility.
      Echo-chamber problem: state media reposts of the same claim
      inflate |entailment|, making false posts look well-supported.

  B — Vanilla FactReasoner (Marinescu et al. 2025)
      P(post is supported) from the Markov Network + Merlin inference.
      Source trust enters via UTD+DynaTD priors on the context nodes.
      Loaded directly from state_media_results.json (already computed).

  C — Trust-Weighted Evidence Coverage (Our System)
      Same as A but each context contributes proportionally to its
      BayesianTrustFusion trust score:
        w_i = trust_score(source_url_i)
        Score_C = Σ_{entail} w_i / (Σ_{entail} w_i + Σ_{contradict} w_i)
      Sources below trust_threshold are excluded entirely.

HYPOTHESIS:
  For FALSE posts:  A > C  (echo chambers inflate A, not C)
  For TRUE posts:   A ≈ C  (trusted sources also agree with real facts)
  System B correlates with C more than A (both are trust-aware)

Usage:
  cd /u/samit/FactReasoner
  python3 scripts/trust_comprehensiveness_eval.py \\
      --dataset   data/state_media_dataset.tsv \\
      --cache     data/serper_cache.json \\
      --fr-results /u/samit/state_media_results/state_media_results.json \\
      --out-dir   /u/samit/comprehensiveness_results \\
      --labels    factual_false \\
      --trust-threshold 0.3
"""

import os, sys, re, json, csv, asyncio, argparse, datetime, math
from collections import defaultdict
from urllib.parse import urlparse

# ── Path setup for IBM server ─────────────────────────────────────────────────
for _p in ["src", ".", "/u/samit/FactReasoner", "/u/samit/FactReasoner/src"]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from fact_reasoner.core.trust import BayesianTrustFusion
    from fact_reasoner.core.nli   import NLIExtractor as _NLIExtractor
    from fact_reasoner.core.base  import Context, Atom
    from fact_reasoner.core.utils import build_relations
    from mellea_ibm.rits          import RITSBackend, RITS
    from mellea.backends          import ModelOption
    HAVE_FR = True
except ImportError as e:
    HAVE_FR = False
    print(f"[warn] FactReasoner not importable ({e}). Run on IBM server.")

MODEL_PATH   = "/u/samit/utd_model.pkl"
DYNATD_STATE = "/u/samit/dynaTD_state_state_media_factual_false.json"

LABEL_MAP = {
    "all":           None,
    "factual_false": {"factual", "false"},
    "factual":       {"factual"},
    "false":         {"false"},
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: str, label_filter: set | None) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            label = row.get("factuality_label", "").strip().lower()
            row["raw_label"] = label
            row["category"]  = row.get("category_label", "")
            if label_filter and label not in label_filter:
                continue
            rows.append(row)
    return rows


def load_serper_cache(path: str) -> dict:
    if not os.path.exists(path): return {}
    with open(path) as f: return json.load(f)


def get_cached_contexts(row: dict, serper_cache: dict) -> list[dict]:
    text = re.sub(r"#\w+|@\w+|https?://\S+", "", row["text"]).strip()
    text = re.sub(r"\s+", " ", text)
    date = row.get("date", "")

    q1 = f"{text[:180]} {date}".strip() if date else text[:180]

    snippet = text[:90]
    ls = snippet.rfind(" ")
    snippet = (snippet[:ls] if ls > 20 else snippet).rstrip(".,;:!?")
    category = row.get("category", "")
    date_suffix = f" {date}" if date else ""
    q2 = ("Xinjiang forced labor fact check independent report"
          if category == "Xinjiang"
          else f"{snippet} fact check{date_suffix}")

    results = []
    seen = set()
    for q in [q1, q2]:
        for h in serper_cache.get(q, []):
            if isinstance(h, dict) and h.get("link") and h["link"] not in seen:
                seen.add(h["link"])
                results.append(h)
    return results


def load_fr_results(path: str) -> dict:
    if not os.path.exists(path): return {}
    with open(path) as f:
        data = json.load(f)
    indexed = {}
    for r in (data if isinstance(data, list) else []):
        for key in [(r.get("claim") or "")[:80],
                    r.get("post_url", ""), r.get("url", "")]:
            if key: indexed[key] = r
    return indexed


# ─────────────────────────────────────────────────────────────────────────────
# TRUST AND NLI
# ─────────────────────────────────────────────────────────────────────────────

def get_trust_scorer():
    if not HAVE_FR: return None
    return BayesianTrustFusion(model_path=MODEL_PATH, state_path=DYNATD_STATE)


def get_nli():
    if not HAVE_FR: return None
    backend = RITSBackend(RITS.LLAMA_3_3_70B_INSTRUCT,
                          model_options={ModelOption.MAX_NEW_TOKENS: 1024})
    class NLIFixed(_NLIExtractor):
        def __init__(self): super().__init__(backend)
    return NLIFixed()


def score_trust(url: str, trust_scorer) -> float:
    if trust_scorer is None: return 0.5
    try:
        atom = Atom(id="a0", text="placeholder")
        ctx  = Context(id="c0", atom=atom, text="placeholder",
                       title="", snippet="", link=url)
        return float(trust_scorer.score(ctx))
    except Exception: return 0.5


def classify_context(claim_text: str, ctx_snippet: str, nli) -> str:
    """
    NLI classify: does ctx_snippet ENTAIL, CONTRADICT, or be NEUTRAL to claim?
    This is the standard FactReasoner direction: context as evidence for claim.
    Returns 'entailment', 'contradiction', or 'neutral'.
    """
    if nli is None: return "neutral"
    try:
        atom = Atom(id="a0", text=claim_text[:300])
        ctx  = Context(id="c0", atom=atom,
                       text=ctx_snippet,
                       title="",
                       snippet=ctx_snippet[:80],
                       link="")
        loop = asyncio.new_event_loop()
        try:
            relations = loop.run_until_complete(
                build_relations([ctx], [atom], nli)
            )
        finally:
            loop.close()
        for rel in relations:
            if rel[2] in ("entailment", "contradiction"):
                return rel[2]
        return "neutral"
    except Exception:
        return "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM A — Vanilla Evidence Coverage
# ─────────────────────────────────────────────────────────────────────────────

def system_a(row: dict, contexts: list[dict], nli) -> dict:
    """
    Fraction of retrieved evidence that entails vs contradicts the post.
    Score = |entail| / (|entail| + |contradict|)
    Neither/neutral contexts excluded from denominator.
    All sources weighted equally.
    """
    claim = row["text"]
    entail = contradict = neutral = 0
    details = []

    for ctx in contexts:
        snippet = (ctx.get("snippet") or ctx.get("title") or "")[:300]
        if len(snippet) < 15: continue
        rel = classify_context(claim, snippet, nli)
        if rel == "entailment":   entail += 1
        elif rel == "contradiction": contradict += 1
        else: neutral += 1
        details.append({"link": ctx["link"], "snippet": snippet[:80],
                         "relation": rel})

    denom = entail + contradict
    score = entail / denom if denom > 0 else None
    return {
        "system": "A_vanilla_evidence_coverage",
        "score":  round(score, 4) if score is not None else None,
        "entail": entail, "contradict": contradict, "neutral": neutral,
        "n_contexts": len(contexts),
        "details": details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM C — Trust-Weighted Evidence Coverage (Our System)
# ─────────────────────────────────────────────────────────────────────────────

def system_c(row: dict, contexts: list[dict], trust_scorer, nli,
             trust_threshold: float = 0.3) -> dict:
    """
    Same as A, but each context weighted by BayesianTrustFusion score.
    Score_C = Σ_{entail} w_i / (Σ_{entail} w_i + Σ_{contradict} w_i)

    Effect:
      - State media repost of same claim (trust ~0.47): low weight
      - DoL / AP / Reuters contradiction (trust ~0.9+): high weight
      - Pure echo sources (trust < threshold): excluded entirely

    For FALSE posts: contradictions come from high-trust sources,
    entailments from low-trust echo → Score_C << Score_A
    For TRUE posts: entailments from both trusted and untrusted sources
    → Score_C ≈ Score_A
    """
    claim = row["text"]
    entail_w = contradict_w = 0.0
    excluded = 0
    details = []

    for ctx in contexts:
        snippet = (ctx.get("snippet") or ctx.get("title") or "")[:300]
        if len(snippet) < 15: continue

        trust = score_trust(ctx["link"], trust_scorer)
        if trust < trust_threshold:
            excluded += 1
            details.append({"link": ctx["link"], "snippet": snippet[:80],
                             "trust": round(trust, 4), "excluded": True})
            continue

        rel = classify_context(claim, snippet, nli)
        if rel == "entailment":      entail_w += trust
        elif rel == "contradiction": contradict_w += trust
        details.append({"link": ctx["link"], "snippet": snippet[:80],
                         "trust": round(trust, 4), "relation": rel,
                         "excluded": False})

    denom = entail_w + contradict_w
    score = entail_w / denom if denom > 0 else None
    return {
        "system":        "C_trust_weighted_evidence_coverage",
        "score":         round(score, 4) if score is not None else None,
        "entail_weight": round(entail_w, 4),
        "contradict_weight": round(contradict_w, 4),
        "n_excluded":    excluded,
        "n_used":        len([d for d in details if not d.get("excluded")]),
        "details":       details,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATION AND REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def stats(scores: list[float]) -> dict:
    if not scores:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    n = len(scores)
    m = sum(scores) / n
    s = (sum((x - m) ** 2 for x in scores) / n) ** 0.5
    return {"n": n, "mean": round(m, 4), "std": round(s, 4),
            "min": round(min(scores), 4), "max": round(max(scores), 4)}


def aggregate(all_results: list[dict], label: str) -> dict:
    def scores_for(key):
        return [r[key]["score"] for r in all_results
                if r[key].get("score") is not None]

    # Also break down by factual vs false
    def scores_by_label(key, lbl):
        return [r[key]["score"] for r in all_results
                if r[key].get("score") is not None
                and r["row"].get("raw_label") == lbl]

    return {
        "label_filter": label,
        "n_total": len(all_results),
        "A": {**stats(scores_for("A")),
              "factual": stats(scores_by_label("A", "factual")),
              "false":   stats(scores_by_label("A", "false"))},
        "B": {**stats(scores_for("B")),
              "factual": stats(scores_by_label("B", "factual")),
              "false":   stats(scores_by_label("B", "false"))},
        "C": {**stats(scores_for("C")),
              "factual": stats(scores_by_label("C", "factual")),
              "false":   stats(scores_by_label("C", "false"))},
    }


def format_report(agg: dict, all_results: list[dict]) -> str:
    L = []
    w = 72
    L += ["=" * w,
          "THREE-WAY EVALUATION — Chinese State Media Factuality Dataset",
          "=" * w,
          f"Generated:    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
          f"Label filter: {agg['label_filter']}",
          f"Total rows:   {agg['n_total']}",
          ""]

    L += ["SYSTEMS", "-" * w,
          "  A: Vanilla Evidence Coverage (adapted from Dejl et al. 2025)",
          "     Score = |entail| / (|entail| + |contradict|)",
          "     All sources weighted equally. Echo-chamber bias expected.",
          "",
          "  B: Vanilla FactReasoner (Marinescu et al. 2025)",
          "     P(claim supported) from Markov Network + Merlin inference.",
          "     Source trust via UTD+DynaTD priors. No comprehensiveness.",
          "",
          "  C: Trust-Weighted Evidence Coverage [OUR SYSTEM]",
          "     Score = Σ_{entail} trust_i / (Σ_{entail}+Σ_{contradict} trust_i)",
          "     State media echo-sources down-weighted. Hypothesis:",
          "       false posts: A > C (echo inflates A, not C)",
          "       factual posts: A ≈ C (trusted sources also agree)",
          ""]

    L += ["AGGREGATE SCORES", "-" * w,
          f"{'System':<45} {'n':>4} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7}",
          "-" * w]
    for key, label in [("A", "A: Vanilla Evidence Coverage    "),
                        ("B", "B: Vanilla FactReasoner         "),
                        ("C", "C: Trust-Weighted Coverage [OURS]")]:
        s = agg[key]
        if s["n"] == 0:
            L.append(f"  {label:<43}  N/A")
        else:
            L.append(f"  {label:<43} {s['n']:>4} {s['mean']:>7.4f} "
                     f"{s['std']:>7.4f} {s['min']:>7.4f} {s['max']:>7.4f}")

    L += ["", "BY LABEL", "-" * w,
          f"{'System':<45} {'factual_n':>9} {'factual_mean':>12} "
          f"{'false_n':>7} {'false_mean':>10}"]
    for key, label in [("A", "A: Vanilla Evidence Coverage    "),
                        ("B", "B: Vanilla FactReasoner         "),
                        ("C", "C: Trust-Weighted Coverage [OURS]")]:
        sf = agg[key]["factual"]
        sl = agg[key]["false"]
        fn = sf["n"] if sf["n"] else 0
        fm = f"{sf['mean']:.4f}" if sf.get("mean") is not None else "N/A"
        ln = sl["n"] if sl["n"] else 0
        lm = f"{sl['mean']:.4f}" if sl.get("mean") is not None else "N/A"
        L.append(f"  {label:<43} {fn:>9} {fm:>12} {ln:>7} {lm:>10}")

    # Echo-chamber delta: A - C for false vs factual
    A_false = agg["A"]["false"].get("mean")
    C_false = agg["C"]["false"].get("mean")
    A_fact  = agg["A"]["factual"].get("mean")
    C_fact  = agg["C"]["factual"].get("mean")
    L += ["", "ECHO-CHAMBER EFFECT (A − C DELTA)", "-" * w]
    if A_false is not None and C_false is not None:
        L.append(f"  false  posts: A={A_false:.4f}  C={C_false:.4f}  "
                 f"Δ={A_false-C_false:+.4f}  "
                 f"({'Echo-chamber confirmed' if A_false > C_false else 'No echo effect'})")
    if A_fact is not None and C_fact is not None:
        L.append(f"  factual posts: A={A_fact:.4f}  C={C_fact:.4f}  "
                 f"Δ={A_fact-C_fact:+.4f}  "
                 f"({'Trusted sources agree — expected' if abs(A_fact-C_fact) < 0.05 else 'Unexpected divergence'})")

    L += ["", "PER-ROW RESULTS", "-" * w,
          f"{'#':<3} {'Account':<22} {'Label':<9} "
          f"{'A':>7} {'B':>7} {'C':>7} {'A-C':>6} "
          f"{'A_e':>4} {'A_c':>4} {'C_excl':>6}"]
    L.append("-" * w)
    for r in all_results:
        row = r["row"]
        a, b, c = r["A"], r["B"], r["C"]
        a_s = f"{a['score']:.3f}" if a.get("score") is not None else "  —  "
        b_s = f"{b['score']:.3f}" if b.get("score") is not None else "  —  "
        c_s = f"{c['score']:.3f}" if c.get("score") is not None else "  —  "
        delta = ""
        if a.get("score") is not None and c.get("score") is not None:
            delta = f"{a['score']-c['score']:+.3f}"
        L.append(f"{r['row_idx']:<3} {row.get('account_name','?')[:21]:<22} "
                 f"{row.get('raw_label','?'):<9} "
                 f"{a_s:>7} {b_s:>7} {c_s:>7} {delta:>6} "
                 f"{a.get('entail',0):>4} {a.get('contradict',0):>4} "
                 f"{c.get('n_excluded',0):>6}")

    L += ["", "=" * w]
    return "\n".join(L)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",      default="data/state_media_dataset.tsv")
    ap.add_argument("--cache",        default="data/serper_cache.json")
    ap.add_argument("--fr-results",
        default="/u/samit/state_media_results/state_media_results.json")
    ap.add_argument("--out-dir",      default="/u/samit/comprehensiveness_results")
    ap.add_argument("--labels",       default="factual_false",
                    choices=list(LABEL_MAP.keys()))
    ap.add_argument("--trust-threshold", type=float, default=0.3)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    label_filter = LABEL_MAP[args.labels]

    rows = load_dataset(args.dataset, label_filter)
    print(f"[data] {len(rows)} rows (filter={args.labels})")

    serper_cache = load_serper_cache(args.cache)
    print(f"[cache] {len(serper_cache)} Serper entries")

    fr_results = load_fr_results(args.fr_results)
    print(f"[fr] {len(fr_results)} FactReasoner results")

    trust_scorer = get_trust_scorer()
    nli          = get_nli()
    print(f"[init] Trust: {'real' if trust_scorer else 'mock'}  "
          f"NLI: {'real' if nli else 'mock'}")

    all_results = []

    for i, row in enumerate(rows):
        print(f"\n[{i+1}/{len(rows)}] {row['account_name']} "
              f"| {row['raw_label']} | {row['text'][:55]}...")

        contexts = get_cached_contexts(row, serper_cache)
        print(f"  {len(contexts)} contexts from cache")

        # System A
        res_a = system_a(row, contexts, nli)
        print(f"  A: score={res_a['score']}  "
              f"entail={res_a['entail']} contradict={res_a['contradict']} "
              f"neutral={res_a['neutral']}")

        # System B — from saved results
        claim_key = row["text"][:80]
        fr_row = (fr_results.get(claim_key) or
                  fr_results.get(row.get("post_url", "")) or {})
        score_b = fr_row.get("p_trust")
        res_b = {"system": "B_vanilla_factreasonerscore",
                 "score": round(float(score_b), 4) if score_b is not None else None}
        print(f"  B: score={res_b['score']}")

        # System C
        res_c = system_c(row, contexts, trust_scorer, nli, args.trust_threshold)
        print(f"  C: score={res_c['score']}  "
              f"entail_w={res_c['entail_weight']} "
              f"contradict_w={res_c['contradict_weight']} "
              f"excluded={res_c['n_excluded']}")

        delta = None
        if res_a.get("score") is not None and res_c.get("score") is not None:
            delta = round(res_a["score"] - res_c["score"], 4)
        print(f"  A−C delta: {delta}  "
              f"({'echo confirmed' if delta and delta > 0.05 else 'no echo' if delta is not None else 'N/A'})")

        all_results.append({
            "row_idx": i,
            "row": row,
            "A": res_a,
            "B": res_b,
            "C": res_c,
            "delta_A_minus_C": delta,
        })

    agg = aggregate(all_results, args.labels)

    # Save JSON
    out_json = os.path.join(args.out_dir, "three_way_eval_results.json")
    with open(out_json, "w") as f:
        clean = []
        for r in all_results:
            clean.append({
                "row_idx": r["row_idx"],
                "account": r["row"]["account_name"],
                "raw_label": r["row"]["raw_label"],
                "category": r["row"].get("category", ""),
                "claim": r["row"]["text"][:100],
                "A": {k: v for k, v in r["A"].items() if k != "details"},
                "B": r["B"],
                "C": {k: v for k, v in r["C"].items() if k != "details"},
                "delta_A_minus_C": r["delta_A_minus_C"],
            })
        json.dump({"aggregate": agg, "rows": clean}, f, indent=2)
    print(f"\n[saved] {out_json}")

    report = format_report(agg, all_results)
    out_txt = os.path.join(args.out_dir, "three_way_eval_report.txt")
    with open(out_txt, "w") as f:
        f.write(report)
    print(f"[saved] {out_txt}")
    print("\n" + report)


if __name__ == "__main__":
    main()
