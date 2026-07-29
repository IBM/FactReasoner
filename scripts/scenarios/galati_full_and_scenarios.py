"""
ONE script, run in order, so state flows correctly:

  PART A: Run the real 8-atom Galati demo from scratch (galati_eval9.py
          logic) -- this is what tells us "which websites to trust."
          DynaTD state is reset and rebuilt from these 8 real atoms.

  PART B: Using that exact freshly-trained state (NOT reloaded from disk,
          same in-memory trust_scorer object), run the two scenario atoms:
            S1_NONREPUTABLE_ONLY -- only non-reputable sources present
            S2_REPUTABLE_ONLY    -- only reputable sources present

BUG FIXED from scenario_gen_v3.py: the previous script captured each
context's fused_prior AFTER the vanilla pass had already overwritten it
to flat 0.9 (ctx.set_probability(0.9) mutates the same Context object
later read by the JSON export). This version captures the real fused
prior into a separate dict BEFORE the vanilla overwrite happens.
"""
import sys, os, logging, math, time, json
logging.getLogger('asyncio').setLevel(logging.CRITICAL)
sys.path.insert(0, "/u/samit/FactReasoner/src")

from mellea_ibm.rits import RITSBackend, RITS
from mellea.backends import ModelOption
from fact_reasoner.core.trust import BayesianTrustFusion
from fact_reasoner.core.trust.url_trust import UTD
from fact_reasoner.core.nli import NLIExtractor as _NLIExtractor
from fact_reasoner.core.base import Context, Atom
from fact_reasoner.core.utils import build_relations
from fact_reasoner.fact_graph import FactGraph

MODEL_PATH = "/u/samit/utd_model.pkl"
SEL_PATH   = "/u/samit/data/selected_features.json"
STATE_PATH = "/u/samit/dynaTD_state_galati8_combined.json"
MERLIN     = "/u/samit/FactReasoner/merlin"
OUT_DIR    = "/u/samit/galati_graphs"
os.makedirs(OUT_DIR, exist_ok=True)

class NLIFixed(_NLIExtractor):
    def _get_probability(self, output) -> float:
        try:
            r = output._meta["oai_chat_response"]
            lp = (r.get("choices",[{}])[0].get("logprobs") or r.get("logprobs"))
            if not lp or not lp.get("content"): return 1.0
            s, n = 0, 0
            for item in reversed(lp["content"][:-1]):
                if item["token"] == "[": break
                elif item["token"] != "]":
                    s += item["logprob"]; n += 1
            return math.exp(s/n) if n > 0 else 0.0
        except: return 1.0

ATOMS = [
    {"id":"a0","text":"The drone that struck Galati, Romania on May 29 2026 was of Russian origin.",
     "ground_truth":"S","label":"A1_TRUE"},
    {"id":"a0","text":"The origin of the drone that struck Galati, Romania on May 29 2026 cannot be determined from available evidence.",
     "ground_truth":"NS","label":"A2_FALSE"},
    {"id":"a0","text":"Romania confirmed a Russian-made drone struck the building in Galati.",
     "ground_truth":"S","label":"A3_TRUE"},
    {"id":"a0","text":"No evidence confirmed the Russian origin of the drone that struck Galati.",
     "ground_truth":"NS","label":"A4_FALSE"},
]
ATOMS_ROUND2 = [{**a, "label": a["label"].split("_")[0] + "r2_" + a["label"].split("_",1)[1]} for a in ATOMS]
ALL_ATOMS = ATOMS + ATOMS_ROUND2

SOURCES = [
    {"id":"c0","domain":"aljazeera.com",
     "url":"https://www.aljazeera.com/news/2026/5/29/nato-states-slam-russia-after-drone-crashes-in-romania",
     "text":"A Russian drone crashed into a residential building in Galati, Romania on May 29 2026, injuring two people. NATO states condemned Russia for the incident. Polish Foreign Minister Sikorski said that whether by purpose or ineptitude, Russia is still dangerous."},
    {"id":"c1","domain":"euromaidanpress.com",
     "url":"https://euromaidanpress.com/2026/05/31/romanias-forensic-report-confirms-the-galati-drone-was-a-russian-geran-2/",
     "text":"Romania's forensic report confirms the Galati drone was a Russian Geran-2. Investigators found the Cyrillic inscription GERAN-2 stenciled on a recovered fragment and an engine stamped 1346/11. Russia launched 232 drones at Ukraine that same night."},
    {"id":"c2","domain":"reuters.com",
     "url":"https://www.reuters.com/world/europe/apartment-building-hit-by-drone-romanias-galati-close-ukraine-border-radio-says-2026-05-29/",
     "text":"Romania said a Russian drone hit a block of flats in Galati injuring two people. A separate drone was also found in the Maramures region. This marked the 28th recorded breach of Romanian airspace by Russian drones since the war began."},
    {"id":"c3","domain":"themoscowtimes.com",
     "url":"https://www.themoscowtimes.com/2026/05/29/no-one-can-say-origin-of-drone-that-crashed-in-romania-putin-says-a92877",
     "text_a1":"The origin of the drone that crashed in Romania was not confirmed as Russian. No forensic examination had been conducted to establish its origin. The drone may have been Ukrainian, as similar incidents occurred in Finland and the Baltic states.",
     "text_a2":"The origin of the drone that crashed in Romania was not confirmed as Russian. No forensic examination had been conducted to establish its origin. The drone may have been Ukrainian, as similar incidents occurred in Finland and the Baltic states."},
    {"id":"c4","domain":"nato.news-pravda.com",
     "url":"https://nato.news-pravda.com/russia/2026/05/29/105977.html",
     "text_a1":"The Russian Embassy in Bucharest stated the incident was a deliberate provocation by Kiev to drag NATO into war with Russia. The drone was not of Russian origin.",
     "text_a2":"The Russian Embassy in Bucharest stated the incident was a deliberate provocation by Kiev to drag NATO into war with Russia. The origin of the drone cannot be established from available evidence and was not confirmed as Russian."},
    {"id":"c5","domain":"vedomosti.ru",
     "url":"https://www.vedomosti.ru/politics/news/2026/05/29/1201351-v-ruminiyu-dron",
     "text_a1":"The drone that entered Romania was not confirmed to be Russian. Its origin was unknown and Putin suggested it may have been Ukrainian. Russia was not established as responsible for the Galati incident.",
     "text_a2":"The drone that entered Romania was not confirmed to be Russian. Its origin was unknown and Putin suggested it may have been Ukrainian. Russia was not established as responsible for the Galati incident."},
    {"id":"c6","domain":"kommersant.ru",
     "url":"https://www.kommersant.ru/doc/8707171",
     "text_a1":"Putin doubted the Romanian claim that the drone was Russian and called for a proper forensic examination before any attribution could be made. The drone's Russian origin was not confirmed. Russia offered to cooperate with any objective investigation into the incident.",
     "text_a2":"Putin doubted the Romanian claim that the drone was Russian and called for a proper forensic examination before any attribution could be made. The drone's Russian origin was not confirmed. Russia offered to cooperate with any objective investigation into the incident."},
]

REPUTABLE_IDS    = ["c0", "c1", "c2"]
NONREPUTABLE_IDS = ["c3", "c4", "c5", "c6"]

async def run_full_atom(atom_def, utd_scores, trust_scorer, nli, text_key, use_trust):
    atom = Atom(id="a0", text=atom_def["text"])
    atoms_dict = {"a0": atom}
    contexts = {}
    for src in SOURCES:
        text = src.get(text_key, src.get("text", ""))
        ctx = Context(id=src["id"], atom=atom, text=text,
                      title=src["domain"], snippet=text[:80], link=src["url"])
        prior = trust_scorer.score(ctx) if use_trust else utd_scores[src["id"]]
        ctx.set_probability(prior)
        atom.add_contexts([ctx])
        contexts[src["id"]] = ctx

    relations = build_relations(
        atoms=atoms_dict, contexts=contexts, nli_extractor=nli,
        rel_atom_context=True, rel_context_context=False,
        use_summarized_contexts=False,
    )
    for ctx in contexts.values():
        ctx.set_probability(trust_scorer.score(ctx) if use_trust else utd_scores[ctx.id])

    # Capture REAL fused priors BEFORE any vanilla overwrite happens
    real_fused_priors = {cid: ctx.get_probability() for cid, ctx in contexts.items()}

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

    for ctx in contexts.values():
        ctx.set_probability(0.9)
    pipeline.fact_graph = None; pipeline.markov_network = None
    pipeline._build_fact_graph(); pipeline._build_markov_network()
    van_results, van_marginals = pipeline.score()
    p_van = next((m["probabilities"][1] for m in van_marginals if m["variable"]=="a0"), 0.5)
    l_van = "S" if p_van > 0.5 else "NS"

    return (p_trust, l_trust, p_van, l_van, relations, contexts,
            trust_marginals, real_fused_priors)

async def run_scenario_atom(claim_text, source_ids, text_key, trust_scorer, nli):
    atom = Atom(id="a0", text=claim_text)
    atoms_dict = {"a0": atom}
    contexts = {}
    for src in SOURCES:
        if src["id"] not in source_ids:
            continue
        text = src.get(text_key, src.get("text", ""))
        ctx = Context(id=src["id"], atom=atom, text=text,
                      title=src["domain"], snippet=text[:80], link=src["url"])
        prior = trust_scorer.score(ctx)
        ctx.set_probability(prior)
        atom.add_contexts([ctx])
        contexts[src["id"]] = ctx

    relations = build_relations(
        atoms=atoms_dict, contexts=contexts, nli_extractor=nli,
        rel_atom_context=True, rel_context_context=False,
        use_summarized_contexts=False,
    )
    for ctx in contexts.values():
        ctx.set_probability(trust_scorer.score(ctx))

    real_fused_priors = {cid: ctx.get_probability() for cid, ctx in contexts.items()}

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
    pipeline.labels_human = {"a0": "S"}  # placeholder, set properly by caller
    pipeline.query = pipeline.response = pipeline.topic = ""
    pipeline._build_fact_graph(); pipeline._build_markov_network()
    trust_results, trust_marginals = pipeline.score()
    p_trust = next((m["probabilities"][1] for m in trust_marginals if m["variable"]=="a0"), 0.5)
    l_trust = "S" if p_trust > 0.5 else "NS"

    for ctx in contexts.values():
        ctx.set_probability(0.9)
    pipeline.fact_graph = None; pipeline.markov_network = None
    pipeline._build_fact_graph(); pipeline._build_markov_network()
    van_results, van_marginals = pipeline.score()
    p_van = next((m["probabilities"][1] for m in van_marginals if m["variable"]=="a0"), 0.5)
    l_van = "S" if p_van > 0.5 else "NS"

    return p_trust, l_trust, p_van, l_van, relations, contexts, real_fused_priors

async def main():
    utd_model = UTD(model_path=MODEL_PATH, selection_path=SEL_PATH)
    trust_scorer = BayesianTrustFusion(model_path=MODEL_PATH, state_path=STATE_PATH)
    trust_scorer.dynaTD.reset()   # fresh start -- this IS the source of truth run

    backend = RITSBackend(RITS.LLAMA_3_3_70B_INSTRUCT,
                          model_options={ModelOption.MAX_NEW_TOKENS: 4096})
    nli = NLIFixed(backend)

    print("="*72)
    print("  PART A: Real 8-atom Galati run (fresh DynaTD state)")
    print("="*72)

    utd_scores = {src["id"]: utd_model.score(src["url"]) for src in SOURCES}
    for src in SOURCES:
        trust_scorer.dynaTD.initialize_domain(src["domain"], utd_scores[src["id"]])

    all_results = []
    for i, atom_def in enumerate(ALL_ATOMS):
        label = atom_def["label"]
        is_true = atom_def["ground_truth"] == "S"
        text_key = "text_a1" if is_true else "text_a2"
        use_trust = (i > 0)

        print(f"\n[{i+1}/8] {label}  (use_trust={use_trust})")
        (p_trust, l_trust, p_van, l_van, relations, contexts,
         trust_marginals, real_fused) = await run_full_atom(
            atom_def, utd_scores, trust_scorer, nli, text_key, use_trust)

        trust_scorer.update_from_results(contexts, trust_marginals, relations)

        correct = (l_trust == atom_def["ground_truth"])
        van_correct = (l_van == atom_def["ground_truth"])
        print(f"  Trust:   P={p_trust:.4f} -> {l_trust}  {'OK' if correct else 'WRONG'}")
        print(f"  Vanilla: P={p_van:.4f} -> {l_van}  {'OK' if van_correct else 'WRONG'}")

        all_results.append({
            "label": label, "ground_truth": atom_def["ground_truth"],
            "p_trust": p_trust, "verdict": l_trust, "correct": correct,
            "p_van": p_van, "van_verdict": l_van, "van_correct": van_correct,
            "fused_priors": {SOURCES[int(cid[1:])]["domain"]: v for cid, v in real_fused.items()},
        })

    n_correct_trust = sum(1 for r in all_results if r["correct"])
    n_correct_van = sum(1 for r in all_results if r["van_correct"])
    print(f"\nPART A TOTAL: Trust {n_correct_trust}/8  Vanilla {n_correct_van}/8")

    print(f"\nFinal per-domain DynaTD state (the trust rankings):")
    for src in SOURCES:
        dom = src["domain"]
        a = trust_scorer.dynaTD.a.get(dom, 0)
        b = trust_scorer.dynaTD.b.get(dom, 1)
        w = a/b if b else 0
        rs = trust_scorer.dynaTD.get_reliability(dom)
        print(f"  {dom:<25} a={a:.2f}  b={b:.4f}  w={w:.3f}  r_s={rs:.4f}")

    print("\n" + "="*72)
    print("  PART B: Scenario atoms, using the JUST-TRAINED state (same object)")
    print("="*72)

    scenario_results = []

    print(f"\nRunning S1_NONREPUTABLE_ONLY (sources present: {NONREPUTABLE_IDS}) ...")
    p_t, l_t, p_v, l_v, rels, ctxs, real_fused = await run_scenario_atom(
        "No evidence confirmed the Russian origin of the drone that struck Galati.",
        NONREPUTABLE_IDS, "text_a2", trust_scorer, nli)
    gt = "NS"
    print(f"  Trust:   P={p_t:.4f} -> {l_t}  {'CORRECT' if l_t==gt else 'WRONG'}")
    print(f"  Vanilla: P={p_v:.4f} -> {l_v}  {'CORRECT' if l_v==gt else 'WRONG'}")
    sources_out = []
    for cid, ctx in ctxs.items():
        rel = next((r for r in rels if r.source.id == cid), None)
        dom = SOURCES[int(cid[1:])]["domain"]
        sources_out.append({"domain": dom, "fused_prior": real_fused[cid],
                            "nli_type": rel.type if rel else None,
                            "nli_strength": rel.probability if rel else None})
        print(f"    {dom:<25} fused_prior={real_fused[cid]:.4f}  "
              f"{rel.type if rel else None}  strength={rel.probability if rel else None}")
    scenario_results.append({"label":"S1_NONREPUTABLE_ONLY","ground_truth":gt,
                             "p_trust":p_t,"verdict":l_t,"correct":l_t==gt,
                             "p_van":p_v,"van_verdict":l_v,"van_correct":l_v==gt,
                             "sources":sources_out})

    print(f"\nRunning S2_REPUTABLE_ONLY (sources present: {REPUTABLE_IDS}) ...")
    p_t, l_t, p_v, l_v, rels, ctxs, real_fused = await run_scenario_atom(
        "Romania confirmed a Russian-made drone struck the building in Galati.",
        REPUTABLE_IDS, "text_a1", trust_scorer, nli)
    gt = "S"
    print(f"  Trust:   P={p_t:.4f} -> {l_t}  {'CORRECT' if l_t==gt else 'WRONG'}")
    print(f"  Vanilla: P={p_v:.4f} -> {l_v}  {'CORRECT' if l_v==gt else 'WRONG'}")
    sources_out = []
    for cid, ctx in ctxs.items():
        rel = next((r for r in rels if r.source.id == cid), None)
        dom = SOURCES[int(cid[1:])]["domain"]
        sources_out.append({"domain": dom, "fused_prior": real_fused[cid],
                            "nli_type": rel.type if rel else None,
                            "nli_strength": rel.probability if rel else None})
        print(f"    {dom:<25} fused_prior={real_fused[cid]:.4f}  "
              f"{rel.type if rel else None}  strength={rel.probability if rel else None}")
    scenario_results.append({"label":"S2_REPUTABLE_ONLY","ground_truth":gt,
                             "p_trust":p_t,"verdict":l_t,"correct":l_t==gt,
                             "p_van":p_v,"van_verdict":l_v,"van_correct":l_v==gt,
                             "sources":sources_out})

    with open(f"{OUT_DIR}/full_run_and_scenarios.json", "w") as f:
        json.dump({"part_a_atoms": all_results, "part_b_scenarios": scenario_results}, f, indent=2)
    print(f"\nSaved to {OUT_DIR}/full_run_and_scenarios.json")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
