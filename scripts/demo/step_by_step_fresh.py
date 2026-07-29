"""
Fresh single-atom cycle, using the CURRENT corrected code (Beta-Binomial
r_s, beta=min(a/10,0.7)), starting from the already-trained 8-atom state
(/u/samit/dynaTD_state_galati8_combined.json) so a=8 for both sources --
matching what's already on every other slide in this deck.

Runs ONE more atom (a 9th claim) through the live pipeline for real, to
capture a genuine, consistent before->after cycle:
  score() -> NLI -> Merlin posterior -> update() -> new r_s

This is exactly the live flow the "How One Query Updates Two Sources"
slide describes -- every number is real and consistent with the rest
of the deck (a=8 before, not the stale a=10/a=0 from the pre-fix run).
"""
import sys, os, math, time, json
sys.path.insert(0, "/u/samit/FactReasoner/src")

from mellea_ibm.rits import RITSBackend, RITS
from mellea.backends import ModelOption
from fact_reasoner.core.trust import BayesianTrustFusion
from fact_reasoner.core.nli import NLIExtractor as _NLIExtractor
from fact_reasoner.core.base import Context, Atom
from fact_reasoner.core.utils import build_relations

MODEL_PATH = "/u/samit/utd_model.pkl"
STATE_PATH = "/u/samit/dynaTD_state_galati8_combined.json"
MERLIN     = "/u/samit/FactReasoner/merlin"

TRUSTED_URL   = "https://euromaidanpress.com/2026/05/31/romanias-forensic-report-confirms-the-galati-drone-was-a-russian-geran-2/"
UNTRUSTED_URL = "https://www.kommersant.ru/doc/8707171"
TRUSTED_TEXT   = "Romania's forensic report confirms the Galati drone was a Russian Geran-2. Investigators found the Cyrillic inscription GERAN-2 stenciled on a recovered fragment and an engine stamped 1346/11."
UNTRUSTED_TEXT = "Putin doubted the Romanian claim that the drone was Russian and called for a proper forensic examination before any attribution could be made. The drone's Russian origin was not confirmed."
CLAIM = "Romania confirmed a Russian-made drone struck the building in Galati."

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

def domain_of(url):
    from urllib.parse import urlparse
    netloc = urlparse(url).netloc.lower().split(":")[0]
    return netloc[4:] if netloc.startswith("www.") else netloc

async def main():
    from fact_reasoner.assessor import FactReasoner as FR

    trust_scorer = BayesianTrustFusion(model_path=MODEL_PATH, state_path=STATE_PATH)
    dom_trusted = domain_of(TRUSTED_URL)
    dom_untrusted = domain_of(UNTRUSTED_URL)

    backend = RITSBackend(RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 1024})
    nli = NLIFixed(backend)

    # BEFORE state
    a_t_before = trust_scorer.dynaTD.a[dom_trusted]
    rs_t_before = trust_scorer.dynaTD.get_reliability(dom_trusted)
    a_u_before = trust_scorer.dynaTD.a[dom_untrusted]
    rs_u_before = trust_scorer.dynaTD.get_reliability(dom_untrusted)
    print(f"BEFORE: {dom_trusted} a={a_t_before} r_s={rs_t_before:.4f}")
    print(f"BEFORE: {dom_untrusted} a={a_u_before} r_s={rs_u_before:.4f}")

    atom = Atom(id="a0", text=CLAIM)
    ctx_t = Context(id="c0", atom=atom, text=TRUSTED_TEXT, title=dom_trusted, snippet="", link=TRUSTED_URL)
    ctx_u = Context(id="c1", atom=atom, text=UNTRUSTED_TEXT, title=dom_untrusted, snippet="", link=UNTRUSTED_URL)
    fused_t = trust_scorer.score(ctx_t)
    fused_u = trust_scorer.score(ctx_u)
    ctx_t.set_probability(fused_t); ctx_u.set_probability(fused_u)
    atom.add_contexts([ctx_t, ctx_u])
    print(f"FUSED: {dom_trusted}={fused_t:.4f}  {dom_untrusted}={fused_u:.4f}")

    atoms_dict = {"a0": atom}
    contexts = {"c0": ctx_t, "c1": ctx_u}
    relations = build_relations(atoms=atoms_dict, contexts=contexts, nli_extractor=nli,
                                rel_atom_context=True, rel_context_context=False, use_summarized_contexts=False)
    rel_t = next(r for r in relations if r.source.id == "c0")
    rel_u = next(r for r in relations if r.source.id == "c1")
    print(f"NLI: {dom_trusted}={rel_t.type} str={rel_t.probability:.6f}")
    print(f"NLI: {dom_untrusted}={rel_u.type} str={rel_u.probability:.6f}")

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
    pipeline.labels_human = {"a0": "S"}
    pipeline.query = pipeline.response = pipeline.topic = ""
    pipeline._build_fact_graph(); pipeline._build_markov_network()
    _, marginals = pipeline.score()
    p_true = next(m["probabilities"][1] for m in marginals if m["variable"]=="a0")
    print(f"MERLIN POSTERIOR: P(TRUE)={p_true:.6f}")

    trust_scorer.update_from_results(contexts, marginals, relations)

    a_t_after = trust_scorer.dynaTD.a[dom_trusted]
    rs_t_after = trust_scorer.dynaTD.get_reliability(dom_trusted)
    a_u_after = trust_scorer.dynaTD.a[dom_untrusted]
    rs_u_after = trust_scorer.dynaTD.get_reliability(dom_untrusted)

    err_t = rel_t.probability * (1 - p_true) if rel_t.type == "entailment" else rel_t.probability * p_true
    err_u = rel_u.probability * (1 - p_true) if rel_u.type == "entailment" else rel_u.probability * p_true

    print(f"\nAFTER: {dom_trusted} a={a_t_after} r_s={rs_t_after:.4f}  error={err_t:.6f}")
    print(f"AFTER: {dom_untrusted} a={a_u_after} r_s={rs_u_after:.4f}  error={err_u:.6f}")

    out = {
        "trusted_domain": dom_trusted, "untrusted_domain": dom_untrusted, "claim": CLAIM,
        "utd_trusted": 0.970, "utd_untrusted": 0.050,
        "a_trusted_before": a_t_before, "rs_trusted_before": rs_t_before,
        "a_untrusted_before": a_u_before, "rs_untrusted_before": rs_u_before,
        "beta_trusted": min(a_t_before/10.0, 0.7), "fused_trusted": fused_t,
        "beta_untrusted": min(a_u_before/10.0, 0.7), "fused_untrusted": fused_u,
        "nli_trusted": rel_t.type, "nli_strength_trusted": rel_t.probability,
        "nli_untrusted": rel_u.type, "nli_strength_untrusted": rel_u.probability,
        "merlin_p_true": p_true,
        "error_trusted": err_t, "error_untrusted": err_u,
        "a_trusted_after": a_t_after, "rs_trusted_after": rs_t_after,
        "a_untrusted_after": a_u_after, "rs_untrusted_after": rs_u_after,
    }
    with open("/u/samit/galati_graphs/step_by_step_fresh.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved to /u/samit/galati_graphs/step_by_step_fresh.json")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
