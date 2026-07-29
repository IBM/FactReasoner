"""
test_failure_fixes.py
=====================
Tests four targeted fixes on the 9 failure atoms from the post-warmup run.
Each fix is tested independently on only the atoms where it applies.

Usage:
    cd /u/samit/FactReasoner
    python3 scripts/test_failure_fixes.py --fix echo
    python3 scripts/test_failure_fixes.py --fix temporal
    python3 scripts/test_failure_fixes.py --fix framing
    python3 scripts/test_failure_fixes.py --fix all
"""
import sys, os, asyncio, json, argparse, re
from datetime import datetime, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from fact_reasoner.core.trust.dynaTD import DynaTD
from mellea_ibm.rits import RITSBackend, RITS
from mellea.backends import ModelOption
from mellea.stdlib.components.chat import Message as MMsg
from mellea.stdlib.context import ChatContext

RESULTS_FILE = "/u/samit/granite_switch_comparison.json"
STATE_FILE   = "/u/samit/dynaTD_state_state_media_all.json"

# ── 9 failure atoms ──────────────────────────────────────────────────────────
FAILURES = [
    # ECHO CHAMBER
    {"row_idx":10, "account":"China Xinhua News", "raw_label":"false", "ground_truth":"NS",
     "date":"3/10/22", "category":"Biolabs", "fix":"ECHO",
     "text":"The U.S. needs to release complete details of its biological laboratories. The DoD controls 336 bio-labs in 30 countries.",
     "core_claim":"The U.S. Department of Defense controls 336 biological laboratories in 30 countries around the world."},
    {"row_idx":24, "account":"Global Times", "raw_label":"false", "ground_truth":"NS",
     "date":"3/10/22", "category":"Biolabs", "fix":"ECHO",
     "text":"#GTGraphic: The US needs to release complete details of its 336 bio-labs around the world.",
     "core_claim":"The United States has 336 biological laboratories around the world."},
    {"row_idx":28, "account":"Global Times", "raw_label":"false", "ground_truth":"NS",
     "date":"11/1/22", "category":"Taiwan", "fix":"ECHO",
     "text":"UNGA Resolution 2758 completely resolved the Taiwan question once and for all.",
     "core_claim":"UNGA Resolution 2758 completely resolved the Taiwan question once and for all."},
    # TEMPORAL
    {"row_idx":1, "account":"Global Times", "raw_label":"factual", "ground_truth":"S",
     "date":"11/4/22", "category":"Covid", "fix":"TEMPORAL",
     "text":"BQ.1 and BQ.1.1 Omicron subvariants are becoming the dominant strain in the US.",
     "core_claim":"BQ.1 and BQ.1.1 Omicron subvariants are growing rapidly and becoming dominant in the United States."},
    {"row_idx":16, "account":"CGTN", "raw_label":"factual", "ground_truth":"S",
     "date":"1/6/23", "category":"Covid", "fix":"TEMPORAL",
     "text":"The Omicron subvariant XBB.1.5 makes up an increasingly large proportion of new COVID cases in the US.",
     "core_claim":"The Omicron subvariant XBB.1.5 is making up a rapidly growing percentage of COVID-19 cases in the United States."},
    # FRAMING
    {"row_idx":30, "account":"CGTN", "raw_label":"biased", "ground_truth":"NS",
     "date":"9/10/22", "category":"West", "fix":"FRAMING",
     "text":"The Bank of England faces the daunting task of replacing billions of banknotes and coins that have the late Queen Elizabeth II's image on them with an image of the new monarch.",
     "core_claim":"The Bank of England is replacing billions of banknotes and coins that have the late Queen Elizabeth II's image on them with an image of the new monarch, King Charles III."},
    {"row_idx":25, "account":"CGTN", "raw_label":"false", "ground_truth":"NS",
     "date":"9/27/22", "category":"Japan", "fix":"FRAMING",
     "text":"Shinzo Abe's state funeral cost 1.66 billion yen ($11.5 million), twice the initial estimate, stirring controversy in Japan.",
     "core_claim":"Shinzo Abe's state funeral cost 1.66 billion yen, twice the initial estimate."},
    # POLITICAL
    {"row_idx":33, "account":"Global Times", "raw_label":"false", "ground_truth":"NS",
     "date":"1/9/23", "category":"Taiwan", "fix":"POLITICAL",
     "text":"The one-China principle is a universal consensus, and we believe that relevant country will make a correct choice on the Taiwan question.",
     "core_claim":"The one-China principle is a universal consensus, according to the Chinese Foreign Minister."},
    {"row_idx":18, "account":"CGTN", "raw_label":"biased", "ground_truth":"NS",
     "date":"8/5/22", "category":"Taiwan", "fix":"POLITICAL",
     "text":"The United States has acknowledged that Taiwan is part of China and that there is only one China.",
     "core_claim":"The United States has acknowledged that Taiwan is part of China and that there is only one China."},
]


def load_saved():
    if not os.path.exists(RESULTS_FILE):
        print(f"ERROR: {RESULTS_FILE} not found"); sys.exit(1)
    with open(RESULTS_FILE) as f:
        data = json.load(f)
    return {r["row_idx"]: r for r in data}


# ── Fix 1: Defensive retrieval ────────────────────────────────────────────────
def fix_echo_chamber(row, saved_row):
    """Fire Serper q3 only when saved edges have 0 contradictions."""
    from granite_switch_vs_factreaser_demo import serper_search

    edges   = saved_row["fr"]["edges"]
    contras = [e for e in edges if e["nli_type"] == "contradiction"]
    entails = [e for e in edges if e["nli_type"] == "entailment"]

    print(f"  Saved: {len(entails)} entailments, {len(contras)} contradictions")

    if len(contras) > 0:
        print(f"  Already has contradictions — defensive retrieval not needed")
        return None

    claim = row["core_claim"]
    q3a   = f"{claim} false misleading debunked fact check"
    q3b   = f"{claim} site:politifact.com OR site:apnews.com OR site:reuters.com OR site:factcheck.org"

    print(f"  ► Zero contradictions — firing defensive retrieval")
    print(f"    q3a: {q3a[:70]!r}")
    print(f"    q3b: {q3b[:70]!r}")

    r3a = serper_search(q3a, num_results=5)
    r3b = serper_search(q3b, num_results=5)
    all_r3 = r3a + r3b

    print(f"  Got {len(all_r3)} results:")
    contra_kw = ["false","debunked","misinformation","misleading","no evidence",
                 "fact check","not true","incorrect","myth","claim","fabricated"]
    likely = []
    for r in all_r3:
        combo = (r.get("title","") + " " + r.get("snippet","")).lower()
        n_hits = sum(1 for k in contra_kw if k in combo)
        is_fc  = any(s in r.get("link","").lower() for s in
                     ["politifact","apnews","reuters","factcheck","snopes","fullfact"])
        print(f"    {'✅' if is_fc or n_hits>=2 else '  '} [{n_hits} kw] "
              f"{r.get('title','?')[:55]}")
        print(f"       {r.get('link','?')[:65]}")
        if is_fc or n_hits >= 2:
            likely.append(r)

    print(f"\n  Likely contradictions: {len(likely)}")
    return likely


# ── Fix 2: Temporal date filter ───────────────────────────────────────────────
def fix_temporal(row, saved_row, window_days=21):
    """Filter contexts whose URL date is > window_days from post date."""
    post_str = row["date"]
    edges    = saved_row["fr"]["edges"]
    gt       = row["ground_truth"]

    for fmt in ("%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            post_dt = datetime.strptime(post_str, fmt); break
        except ValueError: pass
    else:
        print(f"  Cannot parse date: {post_str!r}"); return

    print(f"  Post date: {post_dt.strftime('%Y-%m-%d')}  window: ±{window_days} days")
    kept, dropped = [], []
    for e in edges:
        url = e.get("link","")
        m   = re.search(r'/(\d{4})[/\-_](\d{1,2})[/\-_](\d{1,2})/', url)
        if m:
            try:
                src_dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                delta  = abs((src_dt - post_dt).days)
                if delta > window_days:
                    dropped.append((e, delta, src_dt))
                    continue
            except ValueError: pass
        kept.append(e)

    print(f"  Kept: {len(kept)}  Dropped: {len(dropped)}")
    for e, delta, src_dt in dropped:
        print(f"    ✗ +{delta}d  [{e['nli_type']:13}] fp={e['fused_prior']:.3f}  "
              f"{e['title'][:45]}")

    if dropped:
        kept_contras  = [e for e in kept  if e["nli_type"]=="contradiction"]
        kept_entails  = [e for e in kept  if e["nli_type"]=="entailment"]
        drop_contras  = [e for e,d,s in dropped if e["nli_type"]=="contradiction"]
        drop_entails  = [e for e,d,s in dropped if e["nli_type"]=="entailment"]
        print(f"\n  After filter: {len(kept_entails)} entails, {len(kept_contras)} contras")

        if gt == "S":
            # We want more entailments to survive, fewer contradictions
            harmful_dropped = drop_contras
            print(f"  Harmful contradictions removed: {len(harmful_dropped)}")
            if harmful_dropped and len(kept_entails) > 0:
                print(f"  → With contradictions removed, Markov would push toward S ✓")
                return True
        else:
            harmful_dropped = drop_entails
            print(f"  Harmful entailments removed: {len(harmful_dropped)}")

    return False


# ── Fix 3 & 4: Granite Switch factuality ─────────────────────────────────────
def fix_granite_switch(row, backend):
    """
    Call factuality_detection on the original post text.
    The adapter sees the full post including framing/opinion cues.
    """
    from mellea.stdlib.components.intrinsic.guardian import factuality_detection

    post  = row["text"]
    claim = row["core_claim"]
    gt    = row["ground_truth"]

    # Build context: user question + assistant answer = the post content
    # factuality_detection expects: context ends with user Q then assistant A
    ctx = ChatContext()
    ctx = ctx.add(MMsg("user",
        f"Is the following social media post factually accurate and not misleading?\n\n"
        f"Post: {post}\n\nClaim made: {claim}"))
    ctx = ctx.add(MMsg("assistant", post))

    print(f"  Calling factuality_detection...")
    try:
        score = factuality_detection(ctx, backend)
        verdict = "NS" if score > 0.5 else "S"
        ok = "✓" if verdict == gt else "✗"
        print(f"  Score: {score:.4f} → {verdict} {ok}  (GT={gt})")
        return score, verdict
    except Exception as e:
        print(f"  Error: {e}")
        return None, None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", default="all",
                    choices=["echo","temporal","framing","political","all"])
    args = ap.parse_args()

    saved = load_saved()

    # Only init backend if needed for model calls
    need_backend = args.fix in ("framing","political","all")
    backend = None
    if need_backend:
        print("Initialising RITS backend for Granite Switch factuality...")
        backend = RITSBackend(
            RITS.GRANITE_GUARDIAN_3_3_8B,
            model_options={ModelOption.MAX_NEW_TOKENS: 16},
        )
        print(f"  Backend: {RITS.GRANITE_GUARDIAN_3_3_8B.model_name}\n")

    # Also need serper for echo chamber fix
    need_serper = args.fix in ("echo","all")

    results = {}
    base_correct = 24   # from post-warmup run
    base_total   = 33
    gained = 0

    for row in FAILURES:
        fix_type = row["fix"]
        gt       = row["ground_truth"]
        idx      = row["row_idx"]

        if args.fix != "all" and args.fix.upper() not in fix_type:
            continue

        saved_row = saved.get(idx)
        if not saved_row:
            print(f"WARNING: row_idx={idx} not in saved results — skipping")
            continue

        old_trust   = saved_row["fr"]["verdict"]
        old_van     = saved_row["fr"].get("van_verdict","?")
        old_gs      = saved_row.get("gs",{}).get("verdict","?")
        was_correct = old_trust == gt

        print(f"\n{'='*65}")
        print(f"[{fix_type}] {row['account']} | {row['raw_label'].upper()} | GT={gt}")
        print(f"  Post:  {row['text'][:80]}")
        print(f"  Claim: {row['core_claim'][:75]}")
        print(f"  Before: Trust={old_trust}{'✓' if was_correct else '✗'}  "
              f"Van={old_van}  Guard={old_gs}")

        fix_worked = False

        if fix_type == "ECHO" and args.fix in ("echo","all"):
            likely = fix_echo_chamber(row, saved_row)
            if likely is not None and len(likely) > 0:
                fix_worked = True
                print(f"  PREDICTION: defensive retrieval → would flip to NS ✓")
            elif likely is not None:
                print(f"  PREDICTION: no strong debunking found — still uncertain")

        elif fix_type == "TEMPORAL" and args.fix in ("temporal","all"):
            fix_worked = fix_temporal(row, saved_row)
            if fix_worked:
                print(f"  PREDICTION: date filter → would flip to {gt} ✓")

        elif fix_type in ("FRAMING","POLITICAL") and args.fix in ("framing","political","all"):
            score, verdict = fix_granite_switch(row, backend)
            if verdict == gt:
                fix_worked = True
                print(f"  PREDICTION: Granite Switch factuality catches this ✓")
            elif verdict is not None:
                print(f"  PREDICTION: Granite Switch also wrong ({verdict}) ✗")

        results[idx] = {
            "fix_type": fix_type, "gt": gt,
            "was_correct": was_correct, "fix_worked": fix_worked,
        }
        if not was_correct and fix_worked:
            gained += 1

    # Summary
    print(f"\n\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    print(f"\n{'Atom':<30} {'Fix':<10} {'Before':>8} {'After':>8}")
    print("─"*55)

    by_fix = {}
    for idx, r in results.items():
        ft = r["fix_type"]
        by_fix.setdefault(ft, {"gained":0,"total":0,"already_correct":0})
        by_fix[ft]["total"] += 1
        if r["was_correct"]:
            by_fix[ft]["already_correct"] += 1
        elif r["fix_worked"]:
            by_fix[ft]["gained"] += 1
            status = "S✗→✓"
        else:
            status = "S✗→✗" if not r["was_correct"] else "✓"

    for ft, d in by_fix.items():
        print(f"  {ft:<20} gained {d['gained']}/{d['total']-d['already_correct']} failures fixed")

    new_correct = base_correct + gained
    print(f"\n  Before: {base_correct}/{base_total} = {base_correct/base_total*100:.1f}%")
    print(f"  After:  {new_correct}/{base_total} = {new_correct/base_total*100:.1f}%  "
          f"(+{gained} atoms)")
    print(f"\n  Potential ceiling if all fixes implemented: "
          f"{base_correct+len([r for r in results.values() if not r['was_correct']])}"
          f"/{base_total}")


if __name__ == "__main__":
    main()
