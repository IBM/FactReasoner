"""
print_all_atoms.py
Usage:
    python3 scripts/print_all_atoms.py --no-fetch
    python3 scripts/print_all_atoms.py --out /u/samit/atoms_full.txt
    python3 scripts/print_all_atoms.py --label false
    python3 scripts/print_all_atoms.py --failures-only
In less: q=quit  Space=forward  b=back  /word=search  n=next
"""
import json, argparse, sys, os, re, textwrap

RESULTS    = "/u/samit/granite_switch_comparison.json"
CACHE_FILE = "/u/samit/page_text_cache.json"
PAGE_CHARS = 4000

STATE_MEDIA   = ["chinadaily","globaltimes","cgtn","cctv","xinhua","english.news.cn","en.people.cn","ecns.cn","cn_humanrights","iprcc.org","china-embassy","china-consulate","mfa.gov.cn"]
FACT_CHECKERS = ["politifact","fullfact","factcheck","snopes","bellingcat"]
REAL_GOV      = ["dol.gov","cdc.gov","who.int","ohchr.org","un.org","ofac.treasury.gov","phila.gov","mass.gov","illinois.gov","joshstein.org","uhrp.org","csis.org","international.gc.ca","parliament.uk","walkfree.org"]
MAINSTREAM    = ["reuters","bbc","nytimes","guardian","npr.org","apnews","voanews","dw.com","pbs.org","nbcnews","abc7","thehill","inquirer.com","whyy","ourworldindata","ama-assn","sinovac.com","aa.com.tr"]

SOURCE_ICONS = {
    "STATE_MEDIA":    "🔴 STATE MEDIA   ",
    "FACT_CHECKER":   "✅ FACT CHECKER  ",
    "REAL_GOV":       "🏛️  REAL GOVT     ",
    "SOCIAL":         "📱 SOCIAL MEDIA  ",
    "REFERENCE":      "📖 REFERENCE     ",
    "MAINSTREAM_NEWS":"📰 MAINSTREAM    ",
    "ACADEMIC/NGO":   "🔬 ACADEMIC/NGO  ",
    "OTHER":          "❓ OTHER         ",
}

KNOWN_ISSUES = {
    "BQ.1":               "TEMPORAL BIAS: claim date mismatches source date window",
    "336 bio":            "ECHO CHAMBER: 5 state-media, 0 fact-checkers — FIX: add site:politifact.com to Serper q2",
    "universal consensus":"POLITICAL OPINION: GT=NS for framing — unfixable by retrieval",
    "Bank of England":    "FRAMING: 'daunting task' is editorial — extractor correctly strips it; NS is about framing",
    "UNGA Resolution 2758":"ECHO CHAMBER: only state-media framing retrieved",
    "Abe":                "FRAMING: funeral cost true, GT=NS for omitted context",
}

def classify(url):
    u = url.lower()
    if any(s in u for s in STATE_MEDIA):   return "STATE_MEDIA"
    if any(s in u for s in FACT_CHECKERS): return "FACT_CHECKER"
    if any(s in u for s in REAL_GOV):      return "REAL_GOV"
    if any(s in u for s in ["facebook.com","twitter.com","youtube.com","dailymotion"]): return "SOCIAL"
    if "wikipedia.org" in u:               return "REFERENCE"
    if any(s in u for s in MAINSTREAM):    return "MAINSTREAM_NEWS"
    if any(s in u for s in ["ncbi","pubmed","springer","bmj","rand.org","amnesty","hrw.org"]): return "ACADEMIC/NGO"
    return "OTHER"

def fetch_page(url, cache):
    if url in cache and cache[url] and not cache[url].startswith("[fetch"):
        return cache[url]
    try:
        import requests
        h = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
             "Accept": "text/html", "Accept-Language": "en-US,en;q=0.9"}
        r = requests.get(url, headers=h, timeout=12, allow_redirects=True)
        t = r.text
        t = re.sub(r'<script[^>]*>.*?</script>', ' ', t, flags=re.DOTALL|re.IGNORECASE)
        t = re.sub(r'<style[^>]*>.*?</style>',  ' ', t, flags=re.DOTALL|re.IGNORECASE)
        t = re.sub(r'<[^>]+>', ' ', t)
        for ent, ch in [('&amp;','&'),('&lt;','<'),('&gt;','>'),('&nbsp;',' '),('&quot;','"'),('&#39;',"'")]:
            t = t.replace(ent, ch)
        t = re.sub(r'\s+', ' ', t).strip()[:PAGE_CHARS]
        cache[url] = t
        return t
    except Exception as ex:
        result = f"[fetch failed: {type(ex).__name__}: {ex}]"
        cache[url] = result
        return result

def pw(text, out, indent=6, width=88):
    if not text: return
    pad = " " * indent
    for sent in re.split(r'(?<=[.!?])\s+', text.strip()):
        sent = sent.strip()
        if not sent: continue
        if len(sent) <= width - indent:
            out.write(pad + sent + "\n")
        else:
            out.write(textwrap.fill(sent, width=width, initial_indent=pad,
                                    subsequent_indent=pad, break_long_words=False,
                                    break_on_hyphens=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label",         default=None)
    ap.add_argument("--failures-only", action="store_true")
    ap.add_argument("--no-fetch",      action="store_true")
    ap.add_argument("--out",           default=None)
    args = ap.parse_args()

    out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout

    with open(RESULTS) as f:
        data = json.load(f)

    cache = {}
    if not args.no_fetch and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cache = json.load(f)
        # Clear stale failed entries so they get re-fetched
        stale = [k for k,v in cache.items() if str(v).startswith("[fetch failed")]
        for k in stale: del cache[k]
        if stale: sys.stderr.write(f"[cache] Cleared {len(stale)} stale entries\n")

    if args.label:        data = [r for r in data if r["raw_label"] == args.label]
    if args.failures_only: data = [r for r in data if not r["fr"]["correct"]]

    out.write("ALL EVALUATED ATOMS — FactReasoner Trust Fusion\n" + "="*90 + "\n")
    out.write(f"Showing {len(data)} atoms | page text: {'NO' if args.no_fetch else f'first {PAGE_CHARS} chars'}\n\n")

    for i, r in enumerate(data):
        gt       = r["ground_truth"]
        label    = r["raw_label"].upper()
        account  = r["account"]
        raw_post = r.get("claim", "")
        claim    = r.get("core_claim", "")
        edges    = r["fr"]["edges"]
        trust_v  = r["fr"]["verdict"]
        van_v    = r["fr"].get("van_verdict", "?")
        gs_raw   = r.get("gs", {}).get("raw", "").replace("\n", " ").strip()
        gs_v     = r.get("gs", {}).get("verdict", "?")
        trust_ok = "✓" if r["fr"]["correct"] else "✗"
        van_ok   = "✓" if r["fr"].get("van_correct", False) else "✗"
        gs_ok    = "✓" if r.get("gs", {}).get("correct", False) else "✗"
        trust_p  = r["fr"].get("probability", None)
        p_str    = f"  P(S)={trust_p:.4f}" if trust_p is not None else ""

        out.write("─"*90 + "\n")
        out.write(f"[{i+1:02d}] {account}  |  {label}  |  GT={gt}\n\n")
        out.write("  ORIGINAL POST:\n");  pw(raw_post, out, 6); out.write("\n")
        out.write("  CORE CLAIM:\n");     pw(claim,    out, 6); out.write("\n")
        out.write("  VERDICTS:\n")
        out.write(f"      Trust Fusion:   {trust_v} {trust_ok}{p_str}\n")
        out.write(f"      Vanilla FR:     {van_v} {van_ok}\n")
        out.write(f"      Granite Guard:  {gs_v} {gs_ok}   raw: {gs_raw[:80]}\n\n")
        out.write(f"  RETRIEVED CONTEXTS ({len(edges)}):\n")

        for j, e in enumerate(edges):
            src  = classify(e["link"])
            icon = SOURCE_ICONS[src]
            d    = "ENTAILS ▲" if e["nli_type"] == "entailment" else "CONTRA  ▼"
            st   = e.get("nli_strength", "")
            ss   = f"  strength={st:.3f}" if isinstance(st, float) else ""
            out.write(f"\n    [{j+1}] {icon}  fp={e['fused_prior']:.4f}  {d}{ss}\n")
            out.write(f"         Title:  {e['title']}\n")
            out.write(f"         URL:    {e['link']}\n")
            if not args.no_fetch:
                sys.stderr.write(f"\r  fetching {i+1}/{len(data)} ctx {j+1}/{len(edges)}...   ")
                text = fetch_page(e["link"], cache)
                if text.startswith("[fetch failed"):
                    out.write(f"         Text:   {text}\n")
                else:
                    out.write(f"         Text ({len(text)} chars):\n")
                    out.write("         " + "·"*70 + "\n")
                    pw(text, out, 9, 88)
                    out.write("         " + "·"*70 + "\n")

        for kw, issue in KNOWN_ISSUES.items():
            if kw.lower() in claim.lower() or kw.lower() in raw_post.lower():
                out.write(f"\n  ⚠️  KNOWN ISSUE: {issue}\n"); break

        entails = [e for e in edges if e["nli_type"] == "entailment"]
        contras = [e for e in edges if e["nli_type"] == "contradiction"]
        sm_e    = [e for e in entails if classify(e["link"]) == "STATE_MEDIA"]
        fc_c    = [e for e in contras if classify(e["link"]) in ("FACT_CHECKER","REAL_GOV","MAINSTREAM_NEWS")]

        if r["raw_label"] in ("false", "biased", "biased/false"):
            if len(sm_e) >= 2 and not fc_c:
                out.write(f"\n  🚨 REPETITION BIAS: {len(sm_e)} state-media entailments, 0 authoritative contradictions\n"
                          f"     FIX → Serper q2: add 'site:politifact.com OR site:reuters.com OR site:apnews.com'\n")
            elif sm_e and not fc_c:
                out.write(f"\n  ⚠️  ECHO CHAMBER: no independent contradiction retrieved\n")
            elif sm_e and fc_c:
                sm_fp = max(e["fused_prior"] for e in sm_e)
                fc_fp = max(e["fused_prior"] for e in fc_c)
                if sm_fp > fc_fp and not r["fr"]["correct"]:
                    out.write(f"\n  ⚠️  TRUST INFLATION: state-media fp={sm_fp:.3f} > fact-checker fp={fc_fp:.3f}\n"
                              f"     FIX → DynaTD warm-up\n")
        out.write("\n")

    if not args.no_fetch:
        sys.stderr.write("\n")
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
        out.write(f"\n[cache] {len(cache)} pages → {CACHE_FILE}\n")

    out.write("="*90 + "\n")
    for k, v in SOURCE_ICONS.items():
        out.write(f"  {v}\n")
    if args.out:
        out.close()
        print(f"Saved → {args.out}")

if __name__ == "__main__":
    main()
