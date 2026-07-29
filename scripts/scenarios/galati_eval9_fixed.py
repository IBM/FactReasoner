"""
Real-world evaluation: Galati drone incident, May 29 2026.
Four claims (A1-A4), two rounds each = 8 atoms total, same 7 sources.

Pipeline:
  Atom 0 (A1, round 1): UTD priors -> Merlin -> update_from_results (actual Merlin posteriors)
  Every atom after:     fused priors (UTD + DynaTD from all prior atoms) -> Merlin -> update_from_results

DynaTD beta schedule: min(num_claims/20, 0.7)
r_s now uses the Beta-Binomial estimator (Laplace rule of succession):
    r_s = (1 + correct) / (2 + total)
This replaces the old saturating sigmoid and stays sensitive across atom counts.

Design transparency:
  kommersant.ru UTD=0.050 -- short URL /doc/8707171, real source dataset #33.
  Added to create a count imbalance among DENIES sources.
"""
import sys, os, logging, math, time
logging.getLogger('asyncio').setLevel(logging.CRITICAL)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mellea_ibm.rits import RITSBackend, RITS
from mellea.backends import ModelOption
from fact_reasoner.core.trust import BayesianTrustFusion
from fact_reasoner.core.trust.url_trust import UTD, extract_all_candidates
from fact_reasoner.core.nli import NLIExtractor as _NLIExtractor
from fact_reasoner.core.base import Context, Atom
from fact_reasoner.core.utils import build_relations

MODEL_PATH = "/u/samit/utd_model.pkl"
STATE_PATH  = "/u/samit/dynaTD_state_galati8.json"
MERLIN      = "/u/samit/FactReasoner/merlin"
SEL_PATH    = "/u/samit/data/selected_features.json"

class NLIFixed(_NLIExtractor):
    def _get_probability(self, output) -> float:
        try:
            r = output._meta["oai_chat_response"]
            lp = (r.get("choices",[{}])[0].get("logprobs") or r.get("logprobs"))
            if not lp or not lp.get("content"): return 1.0
            logprobs = lp["content"][:-1]
            s, n = 0, 0
            for item in reversed(logprobs):
                if item["token"] == "[": break
                elif item["token"] == "]": continue
                else: s += item["logprob"]; n += 1
            return math.exp(s/n) if n > 0 else 0.0
        except: return 1.0

# ── Claims (4 unique) ──────────────────────────────────────────────────────────
ATOMS = [
    {"id":"a0","text":"The drone that struck Galati, Romania on May 29 2026 was of Russian origin.",
     "ground_truth":"S","label":"A1 [TRUE]",
     "note":"Romanian forensic report confirmed: GERAN-2, engine 1346/11, KOMETA GNSS"},
    {"id":"a0","text":"The origin of the drone that struck Galati, Romania on May 29 2026 cannot be determined from available evidence.",
     "ground_truth":"NS","label":"A2 [FALSE]",
     "note":"FALSE -- forensic evidence definitively identified Russian Geran-2"},
    {"id":"a0","text":"Romania confirmed a Russian-made drone struck the building in Galati.",
     "ground_truth":"S","label":"A3 [TRUE]",
     "note":"Romanian forensic report confirmed: GERAN-2, engine 1346/11, KOMETA GNSS"},
    {"id":"a0","text":"No evidence confirmed the Russian origin of the drone that struck Galati.",
     "ground_truth":"NS","label":"A4 [FALSE]",
     "note":"FALSE -- forensic evidence definitively identified Russian Geran-2"},
]

# Round 2: identical 4 claims, relabeled "Xr2" -- DynaTD state (not atom identity) carries over
ATOMS_ROUND2 = [
    {**a, "label": a["label"].split(" ")[0] + "r2 " + a["label"].split(" ", 1)[1]}
    for a in ATOMS
]
ALL_ATOMS = ATOMS + ATOMS_ROUND2  # 8 atoms total

SOURCES = [
    {"id":"c0","domain":"aljazeera.com","side":"SUPPORTS",
     "url":"https://www.aljazeera.com/news/2026/5/29/nato-states-slam-russia-after-drone-crashes-in-romania",
     "text":"A Russian drone crashed into a residential building in Galati, Romania on May 29 2026, injuring two people. NATO states condemned Russia for the incident. Polish Foreign Minister Sikorski said that whether by purpose or ineptitude, Russia is still dangerous."},
    {"id":"c1","domain":"euromaidanpress.com","side":"SUPPORTS",
     "url":"https://euromaidanpress.com/2026/05/31/romanias-forensic-report-confirms-the-galati-drone-was-a-russian-geran-2/",
     "text":"Romania's forensic report confirms the Galati drone was a Russian Geran-2. Investigators found the Cyrillic inscription GERAN-2 stenciled on a recovered fragment and an engine stamped 1346/11. Russia launched 232 drones at Ukraine that same night."},
    {"id":"c2","domain":"reuters.com","side":"SUPPORTS",
     "url":"https://www.reuters.com/world/europe/apartment-building-hit-by-drone-romanias-galati-close-ukraine-border-radio-says-2026-05-29/",
     "text":"Romania said a Russian drone hit a block of flats in Galati injuring two people. A separate drone was also found in the Maramures region. This marked the 28th recorded breach of Romanian airspace by Russian drones since the war began."},
    {"id":"c3","domain":"themoscowtimes.com","side":"DENIES",
     "url":"https://www.themoscowtimes.com/2026/05/29/no-one-can-say-origin-of-drone-that-crashed-in-romania-putin-says-a92877",
     "text":"The origin of the drone that crashed in Romania was not confirmed as Russian. No forensic examination had been conducted to establish its origin. The drone may have been Ukrainian, as similar incidents occurred in Finland and the Baltic states."},
    {"id":"c4","domain":"nato.news-pravda.com","side":"DENIES",
     "url":"https://nato.news-pravda.com/russia/2026/05/29/105977.html",
     "text_a1":"The Russian Embassy in Bucharest stated the incident was a deliberate provocation by Kiev to drag NATO into war with Russia. The drone was not of Russian origin.",
     "text_a2":"The Russian Embassy in Bucharest stated the incident was a deliberate provocation by Kiev to drag NATO into war with Russia. The origin of the drone cannot be established from available evidence and was not confirmed as Russian."},
    {"id":"c5","domain":"vedomosti.ru","side":"DENIES",
     "url":"https://www.vedomosti.ru/politics/news/2026/05/29/1201351-v-ruminiyu-dron",
     "text":"The drone that entered Romania was not confirmed to be Russian. Its origin was unknown and Putin suggested it may have been Ukrainian. Russia was not established as responsible for the Galati incident."},
    {"id":"c6","domain":"kommersant.ru","side":"DENIES",
     "url":"https://www.kommersant.ru/doc/8707171",
     "text":"Putin doubted the Romanian claim that the drone was Russian and called for a proper forensic examination before any attribution could be made. The drone's Russian origin was not confirmed. Russia offered to cooperate with any objective investigation into the incident."},
]

FNAMES = ["url_length","domain_length","path_length","num_dots","num_hyphens",
          "num_digits_in_domain","domain_digit_ratio","subdomain_count","path_depth",
          "is_ip_address","has_hex_encoding","num_special_chars","is_suspicious_tld",
          "is_trusted_tld","domain_entropy","path_entropy","has_https",
          "has_login_keyword","has_at_symbol","tld_length"]

def msgs(prior, nli_type, strength):
    if nli_type == "contradiction":
        return (1-prior)*0.5+prior*strength, (1-prior)*0.5+prior*(1-strength)
    else:
        return (1-prior)*0.5+prior*(1-strength), (1-prior)*0.5+prior*strength

def r_s(domain, trust_scorer):
    # Beta-Binomial estimator (Laplace rule of succession) -- never saturates
    return trust_scorer.dynaTD.get_reliability(domain)

async def run_atom(atom_def, utd_scores, trust_scorer, nli, text_key, use_trust_priors):
    atom = Atom(id="a0", text=atom_def["text"])
    atoms_dict = {"a0": atom}
    contexts = {}
    for src in SOURCES:
        text = src.get(text_key, src.get("text", ""))
        ctx = Context(id=src["id"], atom=atom, text=text,
                      title=src["domain"], snippet=text[:80], link=src["url"])
        if use_trust_priors:
            ctx.set_probability(trust_scorer.score(ctx))
        else:
            ctx.set_probability(utd_scores[src["id"]])
        atom.add_contexts([ctx])
        contexts[src["id"]] = ctx

    relations = build_relations(
        atoms=atoms_dict, contexts=contexts, nli_extractor=nli,
        rel_atom_context=True, rel_context_context=False,
        use_summarized_contexts=False,
    )

    # Trust run
    if use_trust_priors:
        for ctx in contexts.values():
            ctx.set_probability(trust_scorer.score(ctx))
    else:
        for ctx in contexts.values():
            ctx.set_probability(utd_scores[ctx.id])

    from fact_reasoner.assessor import FactReasoner as FR
    pipeline = FR.__new__(FR)
    pipeline.atoms = atoms_dict; pipeline.contexts = contexts
    pipeline.relations = relations; pipeline.merlin_path = MERLIN
    pipeline.fact_graph = None; pipeline.markov_network = None; pipeline.timing = {}
    pipeline.nli_extractor = pipeline.atom_extractor = pipeline.atom_reviser = None
    pipeline.context_retriever = pipeline.context_summarizer = None
    pipeline.revise_atoms = pipeline.summarize_contexts = False
    pipeline.num_retrieved_contexts = len(contexts)
    pipeline.num_summarized_contexts = 0; pipeline.use_priors = True
    pipeline.start_time = time.perf_counter()
    pipeline.early_exit_evaluation = False; pipeline.early_exit_evaluator = None
    pipeline.labels_human = {"a0": atom_def["ground_truth"]}
    pipeline.query = pipeline.response = pipeline.topic = ""
    pipeline._build_fact_graph(); pipeline._build_markov_network()
    trust_results, trust_marginals = pipeline.score()
    p_trust = next((m["probabilities"][1] for m in trust_marginals if m["variable"]=="a0"), 0.5)
    l_trust = "S" if p_trust > 0.5 else "NS"

    # Vanilla run (same NLI, flat priors)
    for ctx in contexts.values():
        ctx.set_probability(0.9)
    pipeline.fact_graph = None; pipeline.markov_network = None
    pipeline._build_fact_graph(); pipeline._build_markov_network()
    van_results, van_marginals = pipeline.score()
    p_van = next((m["probabilities"][1] for m in van_marginals if m["variable"]=="a0"), 0.5)
    l_van = "S" if p_van > 0.5 else "NS"

    return p_trust, l_trust, p_van, l_van, relations, contexts, trust_marginals

def print_dynatd(trust_scorer, utd_scores, label):
    d = trust_scorer.dynaTD
    utd_map = {src["domain"]: utd_scores[src["id"]] for src in SOURCES}
    print(f"\n  {label}")
    print(f"  {'DOMAIN':<25}  {'UTD':>6}  {'a':>6}  {'b':>8}  {'w':>7}  {'r_s':>6}  "
          f"{'beta':>5}  {'FUSED':>7}  STATUS")
    print(f"  {'-'*25}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*7}  {'-'*6}  "
          f"{'-'*5}  {'-'*7}  {'-'*20}")
    rows = []
    for src in SOURCES:
        dom = src["domain"]
        utd = utd_map.get(dom, 0)
        a = d.a.get(dom, 0); b = d.b.get(dom, 1)
        w = a/b if b > 0 else 0
        rs = trust_scorer.dynaTD.get_reliability(dom)
        beta = min(a/20.0, 0.7)
        fused = max(0.05, min(0.97, (1-beta)*utd + beta*rs))
        rows.append((dom, utd, a, b, w, rs, beta, fused))
    for dom, utd, a, b, w, rs, beta, fused in sorted(rows, key=lambda x: -x[7]):
        status = ("++ high" if w>3 else "+ reliable" if w>1.5 else
                  "~ neutral" if w>0.8 else "- unrel." if w>0.3 else "-- very low")
        note = ""
        if dom == "nato.news-pravda.com": note = " <- UTD anomaly correcting"
        if dom == "kommersant.ru":        note = " <- UTD=0.050 floor"
        print(f"  {dom:<25}  {utd:>6.3f}  {a:>6.3f}  {b:>8.4f}  {w:>7.4f}  "
              f"{rs:>6.4f}  {beta:>5.3f}  {fused:>7.4f}  {status}{note}")

async def main():
    utd_model = UTD(model_path=MODEL_PATH, selection_path=SEL_PATH)
    trust_scorer = BayesianTrustFusion(model_path=MODEL_PATH, state_path=STATE_PATH)
    trust_scorer.dynaTD.reset()
    backend = RITSBackend(RITS.LLAMA_3_3_70B_INSTRUCT,
                          model_options={ModelOption.MAX_NEW_TOKENS: 4096})
    nli = NLIFixed(backend)

    print("="*72)
    print("  GALATI DRONE INCIDENT -- 4 claims x 2 rounds = 8 atoms")
    print("  Beta-Binomial reliability estimator (Laplace rule of succession)")
    print("="*72)

    # UTD scores
    print("\n  STEP 1: UTD SCORING")
    print(f"  {'ID':<4}  {'DOMAIN':<25}  {'SIDE':<10}  {'UTD':>6}  {'url_len':>7}  NOTE")
    print(f"  {'-'*4}  {'-'*25}  {'-'*10}  {'-'*6}  {'-'*7}  {'-'*30}")
    utd_scores = {}
    for src in SOURCES:
        score = utd_model.score(src["url"])
        feats = extract_all_candidates(src["url"])
        fd = dict(zip(FNAMES, feats))
        utd_scores[src["id"]] = score
        note = ""
        if src["domain"] == "nato.news-pravda.com": note = "<- HIGH UTD anomaly"
        if src["domain"] == "kommersant.ru":        note = "<- LOW UTD anomaly"
        print(f"  {src['id']:<4}  {src['domain']:<25}  {src['side']:<10}  "
              f"{score:>6.3f}  {fd['url_length']:>7.0f}  {note}")

    # Initialize DynaTD with UTD scores
    for src in SOURCES:
        trust_scorer.dynaTD.initialize_domain(src["domain"], utd_scores[src["id"]])

    print_dynatd(trust_scorer, utd_scores, "DynaTD INITIAL (a=0, beta=0, fused=UTD)")

    all_results = {}

    for i, atom_def in enumerate(ALL_ATOMS):
        alabel = atom_def["label"]
        is_true_atom = atom_def["ground_truth"] == "S"
        text_key = "text_a1" if is_true_atom else "text_a2"
        # First atom (very first overall) uses raw UTD priors; every atom after uses fused priors
        use_trust = (i > 0)

        print(f"\n{'='*72}")
        print(f"  {alabel}: \"{atom_def['text']}\"")
        print(f"  Ground truth: {atom_def['ground_truth']} -- {atom_def['note']}")
        if use_trust:
            print(f"  Priors: FUSED (UTD + DynaTD from {i} prior atom(s)) -- beta>0 now active")
        else:
            print(f"  Priors: UTD only (a=0, beta=0)")
        print(f"{'='*72}")

        # Show priors being used
        print(f"\n  PRIORS for {alabel}:")
        print(f"  {'ID':<4}  {'DOMAIN':<25}  {'UTD':>6}  {'r_s':>6}  {'beta':>5}  "
              f"{'PRIOR':>7}  CHANGE")
        print(f"  {'-'*4}  {'-'*25}  {'-'*6}  {'-'*6}  {'-'*5}  {'-'*7}  {'-'*10}")
        for src in SOURCES:
            dom = src["domain"]
            utd = utd_scores[src["id"]]
            rs = r_s(dom, trust_scorer)
            d = trust_scorer.dynaTD
            a = d.a.get(dom, 0)
            beta = min(a/20.0, 0.7)
            if use_trust:
                atom_tmp = Atom(id="a0", text=atom_def["text"])
                ctx_tmp = Context(id=src["id"], atom=atom_tmp,
                                  text=src.get(text_key, src.get("text","")), title=dom,
                                  snippet="", link=src["url"])
                prior = trust_scorer.score(ctx_tmp)
            else:
                prior = utd
            delta = prior - utd
            delta_str = f"{delta:+.4f}" if abs(delta) > 0.0001 else "--"
            print(f"  {src['id']:<4}  {dom:<25}  {utd:>6.3f}  {rs:>6.4f}  "
                  f"{beta:>5.3f}  {prior:>7.4f}  {delta_str}")

        p_trust, l_trust, p_van, l_van, relations, contexts, trust_marginals = \
            await run_atom(atom_def, utd_scores, trust_scorer, nli, text_key, use_trust)

        # NLI + calculation
        rel_map = {r.source.id: r for r in relations}
        print(f"\n  NLI + CALCULATION:")
        print(f"  {'ID':<4}  {'DOMAIN':<25}  {'PRIOR':>7}  {'NLI':<14}  {'STR':>6}  "
              f"{'msg_F':>8}  {'msg_T':>8}")
        print(f"  {'-'*4}  {'-'*25}  {'-'*7}  {'-'*14}  {'-'*6}  {'-'*8}  {'-'*8}")
        pf = pt = 0.5
        for src in SOURCES:
            rel = rel_map.get(src["id"])
            if not rel: continue
            if use_trust:
                atom_tmp = Atom(id="a0", text=atom_def["text"])
                ctx_tmp = Context(id=src["id"],atom=atom_tmp,
                                  text=src.get(text_key, src.get("text","")),title=src["domain"],
                                  snippet="",link=src["url"])
                prior = trust_scorer.score(ctx_tmp)
            else:
                prior = utd_scores[src["id"]]
            mf, mt = msgs(prior, rel.type, rel.probability)
            pf *= mf; pt *= mt
            print(f"  {src['id']:<4}  {src['domain']:<25}  {prior:>7.4f}  "
                  f"{rel.type:<14}  {rel.probability:>6.4f}  {mf:>8.4f}  {mt:>8.4f}")

        Z = pf+pt
        ct = "CORRECT" if l_trust==atom_def["ground_truth"] else "WRONG"
        cv = "CORRECT" if l_van==atom_def["ground_truth"] else "WRONG"
        print(f"\n  Trust:   P(TRUE)={pt/Z:.4f}  Merlin={p_trust:.4f}  -> {l_trust}  {ct}")
        print(f"  Vanilla: P(TRUE)=--  Merlin={p_van:.4f}  -> {l_van}  {cv}")

        all_results[alabel] = {"p_trust":p_trust,"l_trust":l_trust,
                               "p_van":p_van,"l_van":l_van,
                               "gt":atom_def["ground_truth"]}

        # DynaTD update using actual Merlin posteriors
        print(f"\n  DynaTD update: using actual Merlin posteriors from trust run")
        trust_scorer.update_from_results(contexts, trust_marginals, relations)
        print_dynatd(trust_scorer, utd_scores, f"DynaTD after {alabel}")

    # Summary
    print("\n"+"="*72)
    print("  SUMMARY")
    print("="*72)
    print(f"\n  {'ATOM':<20}  {'GT':>4}  {'TRUST P':>8}  {'T.V':>5}  "
          f"{'VAN P':>8}  {'V.V':>5}  {'TRUST':>8}  {'VANILLA':>8}")
    print(f"  {'-'*20}  {'-'*4}  {'-'*8}  {'-'*5}  {'-'*8}  {'-'*5}  {'-'*8}  {'-'*8}")
    for alabel, r in all_results.items():
        tc = "OK" if r["l_trust"]==r["gt"] else "WRONG"
        vc = "OK" if r["l_van"]==r["gt"] else "WRONG"
        print(f"  {alabel:<20}  {r['gt']:>4}  {r['p_trust']:>8.4f}  {r['l_trust']:>5}  "
              f"{r['p_van']:>8.4f}  {r['l_van']:>5}  {tc:>8}  {vc:>8}")

    n_correct_trust = sum(1 for r in all_results.values() if r["l_trust"]==r["gt"])
    n_correct_van   = sum(1 for r in all_results.values() if r["l_van"]==r["gt"])
    print(f"\n  TOTAL: Trust {n_correct_trust}/{len(all_results)} correct  |  "
          f"Vanilla {n_correct_van}/{len(all_results)} correct")

    print("\n  KEY: Atom 0 uses UTD priors only (beta=0). All others use fused priors.")
    print("  DynaTD update uses actual Merlin posteriors -- not hardcoded ground truth.")
    print("  r_s = Beta-Binomial estimator (Laplace rule of succession), never saturates.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
