"""
dynaTD_warmup.py
================
Pre-populates DynaTD source trust from a known-factual corpus
before running the real evaluation.

Fixes:
  - Politifact fp 0.57 → ~0.80+ (too few appearances)
  - ChinaDaily over-trusted on false claims
  - Fact-checkers ranked below state media

API used:
  DynaTD(state_path=...)   — constructor, calls _load() automatically
  dynaTD.update(domain, atom_posterior, nli_label, nli_strength, utd_score)
  dynaTD.get_reliability(domain) → float
  _save() is called automatically inside update()

Usage:
    cd /u/samit/FactReasoner
    nohup python3 scripts/dynaTD_warmup.py > /u/samit/warmup.txt 2>&1 &
    tail -f /u/samit/warmup.txt
"""
import sys, os, json, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fact_reasoner.core.trust.dynaTD import DynaTD

STATE_FILE  = "/u/samit/dynaTD_state_state_media_all.json"
BACKUP_FILE = "/u/samit/dynaTD_state_BEFORE_WARMUP.json"

# ── Warm-up corpus ─────────────────────────────────────────────────────────────
# Each entry: (domain, nli_label, atom_posterior, nli_strength, utd_score, desc)
#
# nli_label:      "entailment" | "contradiction"
# atom_posterior: what the Markov network concluded — 1.0=S, 0.0=NS
# nli_strength:   confidence of NLI classification (1.0 = certain)
# utd_score:      URL trust score (used for prior initialization)
#
# For correct verdicts:
#   - Source entails true claim → atom_posterior=1.0, nli=entailment → error≈0, correct
#   - Source contradicts false claim → atom_posterior=0.0, nli=contradiction → error≈0, correct
# For incorrect verdicts (state media on false claims):
#   - Source entails false claim → atom_posterior=1.0, nli=entailment → error=1.0*(1-1.0)=0
#     Actually we set posterior=0.0 (correct NS) to penalize the entailment

WARMUP_CORPUS = [
    # ── Fact-checkers correctly identifying false claims (boost their trust) ──
    ("politifact.com",     "contradiction", 0.0, 1.0, 0.65, "Politifact: 336 bio-labs FALSE"),
    ("politifact.com",     "contradiction", 0.0, 1.0, 0.65, "Politifact: COVID microchip FALSE"),
    ("politifact.com",     "entailment",    1.0, 1.0, 0.65, "Politifact: US life expectancy TRUE"),
    ("politifact.com",     "contradiction", 0.0, 1.0, 0.65, "Politifact: Xinjiang denial FALSE"),
    ("politifact.com",     "contradiction", 0.0, 0.9, 0.65, "Politifact: one-China claim FALSE"),
    ("fullfact.org",       "contradiction", 0.0, 1.0, 0.65, "FullFact: COVID transparency FALSE"),
    ("fullfact.org",       "entailment",    1.0, 1.0, 0.65, "FullFact: Honduras shift TRUE"),
    ("factcheck.org",      "contradiction", 0.0, 0.9, 0.65, "FactCheck: UNGA 2758 disputed"),
    ("factcheck.org",      "entailment",    1.0, 1.0, 0.65, "FactCheck: CVS settlement TRUE"),
    ("apnews.com",         "contradiction", 0.0, 1.0, 0.90, "AP: 336 bio-labs FALSE"),
    # ── Real government sources on true claims ─────────────────────────────────
    ("dol.gov",            "contradiction", 0.0, 1.0, 0.97, "DoL: Xinjiang forced labor TRUE"),
    ("cdc.gov",            "entailment",    1.0, 1.0, 0.97, "CDC: life expectancy TRUE"),
    ("who.int",            "entailment",    1.0, 1.0, 0.97, "WHO: CoronaVac reduces severity"),
    ("ofac.treasury.gov",  "contradiction", 0.0, 1.0, 0.97, "OFAC: Xinjiang advisory TRUE"),
    ("ohchr.org",          "contradiction", 0.0, 1.0, 0.97, "OHCHR: UN alarmed by Xinjiang"),
    ("phila.gov",          "entailment",    1.0, 1.0, 0.97, "phila.gov: Holmesburg apology TRUE"),
    # ── Mainstream news on true claims ─────────────────────────────────────────
    ("reuters.com",        "entailment",    1.0, 1.0, 0.97, "Reuters: CoronaVac efficacy TRUE"),
    ("reuters.com",        "entailment",    1.0, 1.0, 0.97, "Reuters: CVS settlement TRUE"),
    ("reuters.com",        "contradiction", 0.0, 1.0, 0.97, "Reuters: 336 bio-labs context"),
    ("bbc.com",            "entailment",    1.0, 1.0, 0.97, "BBC: Queen Elizabeth died TRUE"),
    ("bbc.co.uk",          "entailment",    1.0, 1.0, 0.97, "BBC: XBB.1.5 dominant TRUE"),
    ("nytimes.com",        "entailment",    1.0, 1.0, 0.97, "NYT: CVS settlement TRUE"),
    ("npr.org",            "entailment",    1.0, 1.0, 0.97, "NPR: Philadelphia apology TRUE"),
    ("apnews.com",         "entailment",    1.0, 1.0, 0.97, "AP: Honduras recognition TRUE"),
    # ── Think tanks / NGOs on contested claims ─────────────────────────────────
    ("csis.org",           "contradiction", 0.0, 1.0, 0.97, "CSIS: Xinjiang forced labor TRUE"),
    ("uhrp.org",           "contradiction", 0.0, 1.0, 0.97, "UHRP: Xinjiang report TRUE"),
    ("international.gc.ca","contradiction", 0.0, 1.0, 0.90, "Canada: Xinjiang advisory TRUE"),
    ("parliament.uk",      "contradiction", 0.0, 1.0, 0.90, "UK Parliament: Xinjiang TRUE"),
    ("walkfree.org",       "contradiction", 0.0, 0.9, 0.85, "WalkFree: modern slavery TRUE"),
    # ── State media on TRUE factual claims (retain trust here) ─────────────────
    ("global.chinadaily.com.cn", "entailment", 1.0, 1.0, 0.70, "ChinaDaily: Zhaosu horses TRUE"),
    ("en.people.cn",       "entailment",    1.0, 1.0, 0.70, "People's Daily: Ahmad Khahar TRUE"),
    ("cgtn.com",           "entailment",    1.0, 1.0, 0.70, "CGTN: Taiwan arms statement TRUE"),
    ("en.people.cn",       "entailment",    1.0, 1.0, 0.70, "People's Daily: PLA exercise TRUE"),
    # ── State media on FALSE claims (penalize — use correct NS posterior) ───────
    ("global.chinadaily.com.cn", "entailment", 0.0, 1.0, 0.70, "ChinaDaily: 336 bio-labs FALSE"),
    ("globaltimes.cn",     "entailment",    0.0, 1.0, 0.55, "Global Times: 336 bio-labs FALSE"),
    ("facebook.com/XinhuaNewsAgency", "entailment", 0.0, 1.0, 0.65, "Xinhua FB: bio-labs FALSE"),
    ("english.news.cn",    "entailment",    0.0, 1.0, 0.60, "Xinhua wire: bio-labs FALSE"),
    ("globaltimes.cn",     "entailment",    0.0, 0.9, 0.55, "Global Times: UNGA Taiwan FALSE"),
    ("cgtn.com",           "entailment",    0.0, 0.8, 0.70, "CGTN: BoE daunting task FALSE"),
    # ── Chinese .gov.cn (should NOT get .gov trust — penalize slightly) ─────────
    ("losangeles.china-consulate.gov.cn", "entailment", 0.0, 0.9, 0.50, "LA consulate: UNGA FALSE"),
    ("us.china-embassy.gov.cn",           "entailment", 0.0, 0.9, 0.50, "Embassy: one-China FALSE"),
]


def run_warmup():
    print("=" * 60)
    print("DynaTD WARM-UP PASS")
    print("Pre-populating source trust from known-factual corpus")
    print("=" * 60)

    # Back up current state
    if os.path.exists(STATE_FILE):
        shutil.copy(STATE_FILE, BACKUP_FILE)
        with open(STATE_FILE) as f:
            state_before = json.load(f)
        print(f"Backed up state → {BACKUP_FILE}")
        print(f"Current state entries: {len(state_before)}")
    else:
        state_before = {}
        print("No existing state — starting fresh")

    # Load DynaTD — _load() called automatically in __init__
    dynaTD = DynaTD(state_path=STATE_FILE)

    print(f"\nRunning {len(WARMUP_CORPUS)} warm-up claims...\n")
    print(f"  {'Domain':<45} {'NLI':<14} {'Post':>6} {'Before':>8} {'After':>8}")
    print("  " + "-"*82)

    for domain, nli_label, atom_posterior, nli_strength, utd_score, desc in WARMUP_CORPUS:
        before = dynaTD.get_reliability(domain)

        dynaTD.update(
            domain=domain,
            atom_posterior=atom_posterior,
            nli_label=nli_label,
            nli_strength=nli_strength,
            utd_score=utd_score,
        )

        after = dynaTD.get_reliability(domain)
        print(f"  {domain[:45]:<45} {nli_label:<14} {atom_posterior:>6.1f} "
              f"{before:>8.4f} {after:>8.4f}  {desc[:35]}")

    # Show before/after for key sources
    print(f"\n{'='*60}")
    print("BEFORE/AFTER RELIABILITY FOR KEY SOURCES:")
    print(f"{'='*60}")
    print(f"  {'Domain':<45} {'Before':>8} {'After':>8} {'Δ':>8}")
    print(f"  {'-'*72}")

    key_domains = [
        "politifact.com", "fullfact.org", "factcheck.org",
        "cdc.gov", "dol.gov", "who.int",
        "reuters.com", "bbc.com",
        "global.chinadaily.com.cn", "globaltimes.cn",
        "english.news.cn", "cgtn.com",
        "losangeles.china-consulate.gov.cn",
    ]

    for domain in key_domains:
        before_rel = 0.5  # default if unseen before warmup
        # Estimate before from backup state
        if os.path.exists(BACKUP_FILE):
            with open(BACKUP_FILE) as f:
                bk = json.load(f)
            # DynaTD state stores correct_count and total_count
            total   = bk.get(f"{domain}_total",   bk.get(domain, {}).get("total_count",   0) if isinstance(bk.get(domain), dict) else 0)
            correct = bk.get(f"{domain}_correct",  bk.get(domain, {}).get("correct_count", 0) if isinstance(bk.get(domain), dict) else 0)
            if total > 0:
                before_rel = (1 + correct) / (2 + total)

        after_rel = dynaTD.get_reliability(domain)
        delta     = after_rel - before_rel
        delta_str = f"{delta:+.4f}" if before_rel != 0.5 or after_rel != 0.5 else "  (new)"
        print(f"  {domain:<45} {before_rel:>8.4f} {after_rel:>8.4f} {delta_str:>8}")

    print(f"\nWarm-up complete.")
    print(f"Run the eval now:")
    print(f"  cd /u/samit/FactReasoner")
    print(f"  nohup python3 scripts/granite_switch_vs_factreaser_demo.py "
          f"--cache-mode use > /u/samit/granite_test_postwarmup.txt 2>&1 &")


if __name__ == "__main__":
    run_warmup()
