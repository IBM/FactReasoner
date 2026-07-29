"""
ConflictBank evaluation -- comparing TWO fusion formulas head-to-head:

  FORMULA A (current/baseline):
    fused = (1-beta) * UTD + beta * r_s
    UTD is STATIC (computed once from the URL, never changes).
    beta = min(claims_seen / 10, 0.7) grows with experience.
    -> UTD's influence NEVER fully disappears (always >=30% weight once
       beta caps at 0.7).

  FORMULA B (proposed, rolling r_s blend):
    fused = (1-beta) * r_s_before + beta * r_s_after
    r_s_before = this source's r_s BEFORE this atom's update (i.e. the
                 value carried over from the previous atom).
    r_s_after  = this source's r_s AFTER this atom's update (the fresh
                 value computed from the new correct/total counts).
    UTD is used ONLY to seed r_s at the very first atom (atom_global_idx
    == 0), exactly as get_reliability() already falls back to a/b
    seeded by UTD at initialize_domain(). From atom 1 onward, the
    formula no longer references UTD directly at all -- it is purely a
    blend of two successive REAL r_s readings, so UTD's influence
    decays away after round 1 instead of persisting at >=30% forever.

This script runs the corrected (id, subject) grouping (the v3 fix) and
evaluates BOTH formulas on the SAME 300 facts / same source assignments
(same random seed), in the SAME run, so the comparison is apples-to-apples
-- not two separate noisy runs.

Two independent DynaTD/BayesianTrustFusion state objects are kept (one
per formula), updated in parallel as we walk through the atoms, so each
formula's r_s genuinely evolves on its own trajectory.
"""
import sys, os, logging, math, json, asyncio, time, random
logging.getLogger('asyncio').setLevel(logging.CRITICAL)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import load_dataset
from mellea_ibm.rits import RITSBackend, RITS
from mellea.backends import ModelOption
from fact_reasoner.core.trust import BayesianTrustFusion
from fact_reasoner.core.nli import NLIExtractor as _NLIExtractor
from fact_reasoner.core.base import Context, Atom
from fact_reasoner.core.utils import build_relations
import fact_reasoner.assessor as fr_mod

MODEL_PATH  = "/u/samit/utd_model.pkl"
STATE_PATH_A = "/u/samit/dynaTD_state_cb_formulaA.json"
STATE_PATH_B = "/u/samit/dynaTD_state_cb_formulaB.json"
MERLIN      = "/u/samit/FactReasoner/merlin"
RESULTS_OUT = "/u/samit/cb_dynatd_results5_formula_compare.json"

FIXED_UTD    = 0.9
N_SOURCES    = 20
SOURCE_NAMES = [f"source_{chr(97+i)}" for i in range(N_SOURCES)]

TIER_RELIABLE   = SOURCE_NAMES[0:8]
TIER_UNRELIABLE = SOURCE_NAMES[8:16]
TIER_NOISY      = SOURCE_NAMES[16:20]

P_DEFAULT = {}
for n in TIER_RELIABLE:   P_DEFAULT[n] = 0.95
for n in TIER_UNRELIABLE: P_DEFAULT[n] = 0.05
for n in TIER_NOISY:      P_DEFAULT[n] = 0.50

CONFLICT_CATEGORIES = {
    "misinformation_conflict", "temporal_conflict", "semantic_conflict"
}

def get_tier(name: str) -> str:
    if name in TIER_RELIABLE:   return "reliable"
    if name in TIER_UNRELIABLE: return "unreliable"
    return "noisy"

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

# ── Dataset loader: group by (id, subject), the real fact boundary (v3 fix) ──
def load_subjects(n_subjects: int, seed: int = 42) -> list:
    print(f"Loading CB_claim_evidence (streaming, target={n_subjects})...")
    ds = load_dataset("Warrieryes/CB_claim_evidence", split="train",
                      streaming=True)

    by_fact = {}
    n_streamed = 0
    for row in ds:
        fact_key = (row["id"], row["subject"])
        if fact_key not in by_fact:
            by_fact[fact_key] = {"default": [], "conflict": []}
        cat = row["category"]
        if cat == "default":
            by_fact[fact_key]["default"].append(row)
        elif cat in CONFLICT_CATEGORIES:
            by_fact[fact_key]["conflict"].append(row)
        n_streamed += 1
        valid = sum(1 for f in by_fact.values()
                    if f["default"] and f["conflict"])
        if valid >= n_subjects * 3:
            break

    print(f"Streamed {n_streamed} rows, {len(by_fact)} distinct (id, subject) facts")

    valid_facts = {
        k: v for k, v in by_fact.items()
        if v["default"] and v["conflict"]
    }
    print(f"Valid facts: {len(valid_facts)}")

    rng = random.Random(seed)
    selected = rng.sample(list(valid_facts.keys()),
                          min(n_subjects, len(valid_facts)))

    subjects = []
    for fact_key in selected:
        d = valid_facts[fact_key]
        default_row  = rng.choice(d["default"])
        conflict_rows = rng.sample(d["conflict"], min(2, len(d["conflict"])))

        atoms = [{
            "claim":        default_row["claim"],
            "ground_truth": "S",
            "category":     "default",
            "context_text": default_row["evidence"][:1200],
        }]
        for row in conflict_rows:
            atoms.append({
                "claim":        row["claim"],
                "ground_truth": "NS",
                "category":     row["category"],
                "context_text": row["evidence"][:1200],
            })
        subjects.append({"subject": f"{fact_key[1]} ({fact_key[0]})",
                         "atoms": atoms})

    total_atoms = sum(len(s["atoms"]) for s in subjects)
    print(f"Selected {len(subjects)} facts, {total_atoms} atoms "
          f"(~{total_atoms/max(len(subjects),1):.2f} per fact)")
    return subjects

def assign_contexts(atom_def, default_text: str,
                    conflict_texts: list, seed: int) -> dict:
    rng = random.Random(seed)
    out = {}
    for name in SOURCE_NAMES:
        if rng.random() < P_DEFAULT[name]:
            out[name] = default_text
        else:
            out[name] = rng.choice(conflict_texts)
    return out

def make_pipeline(atoms_dict, contexts, relations, gt):
    FR = fr_mod.FactReasoner
    p  = FR.__new__(FR)
    p.atoms = atoms_dict; p.contexts = contexts
    p.relations = relations; p.merlin_path = MERLIN
    p.fact_graph = p.markov_network = None; p.timing = {}
    p.nli_extractor = p.atom_extractor = p.atom_reviser = None
    p.context_retriever = p.context_summarizer = None
    p.revise_atoms = p.summarize_contexts = False
    p.num_retrieved_contexts = len(contexts)
    p.num_summarized_contexts = 0; p.use_priors = True
    p.start_time = time.perf_counter()
    p.early_exit_evaluation = p.early_exit_evaluator = False
    p.labels_human = {"a0": gt}
    p.query = p.response = p.topic = ""
    p._build_fact_graph(); p._build_markov_network()
    return p

def formula_B_fused(domain, dyna_td, utd_score, atom_global_idx, beta):
    """
    FORMULA B, confirmed spec:
      i = 0:        fused_0 = (1-beta)*UTD + beta*r_s_0
      i = 1:        fused_1 = (1-beta)*UTD + beta*r_s_1   (r_s_{-1} undefined -> UTD)
      i >= 2:       fused_i = (1-beta)*r_s_{i-2} + beta*r_s_{i-1}

    where r_s_k denotes this domain's r_s value AFTER atom k's own
    update_from_results() call has run. Note r_s_i itself (this atom's
    own post-update value) is never an input to fused_i -- it doesn't
    exist yet at scoring time. The two real, already-known readings
    used at i>=2 are r_s_{i-2} and r_s_{i-1}.

    Implementation: at the moment we score atom i, get_reliability()
    returns whatever value was last committed by the previous atom
    that touched this domain -- i.e. r_s_{i-1}. dyna_td.prev_r_s holds
    a one-atom-older snapshot, taken just before each update call --
    i.e. r_s_{i-2}. At i=1, prev_r_s is empty, so it defaults to UTD,
    giving fused_1 = (1-beta)*UTD + beta*r_s_1 once the labels are
    aligned (here this call computes the PRIOR for atom 1's scoring,
    using r_s_after=r_s_0 and r_s_before=UTD; the symmetric reading
    above, fused_1=(1-beta)*UTD+beta*r_s_1, refers to the same number
    computed one index later in the global trace -- both are the
    UTD-anchored case, consistent with Formula A at this same atom).
    """
    if atom_global_idx == 0:
        return utd_score
    r_s_after = dyna_td.get_reliability(domain)
    r_s_before = dyna_td.prev_r_s.get(domain, utd_score)  # default = UTD, not r_s_after
    return (1 - beta) * r_s_before + beta * r_s_after

async def eval_atom_dual(atom_def, default_text, conflict_texts,
                         scorer_A, scorer_B, nli, atom_global_idx):
    """Runs Trust-A, Trust-B, and Vanilla on the SAME contexts/relations,
    so all three see identical NLI outcomes -- only the fused-prior
    formula differs."""
    atom = Atom(id="a0", text=atom_def["claim"])
    atoms_dict = {"a0": atom}
    seed = hash(atom_def["claim"]) % 100000

    assignments = assign_contexts(atom_def, default_text, conflict_texts, seed)

    # Build ONE shared context/relation set (NLI is run once, reused for
    # all three scoring passes below -- the only thing that differs is
    # which prior gets set on each context before each pipeline.score()).
    contexts = {}
    for i, name in enumerate(SOURCE_NAMES):
        text = assignments[name]
        url  = f"https://{name}.factcheck-eval.org/claim/{seed}"
        ctx  = Context(id=f"c{i}", atom=atom, text=text,
                       title=name, snippet=text[:60], link=url)
        ctx.set_probability(FIXED_UTD)  # placeholder, NLI doesn't depend on prior
        atom.add_contexts([ctx])
        contexts[f"c{i}"] = ctx

    try:
        relations = build_relations(
            atoms=atoms_dict, contexts=contexts, nli_extractor=nli,
            rel_atom_context=True, rel_context_context=False,
            use_summarized_contexts=False,
        )
    except Exception:
        return None
    if not relations:
        return None

    utd_scores = {name: FIXED_UTD for name in SOURCE_NAMES}  # all flat 0.9, same as before

    # ── Formula A run ──
    for i, name in enumerate(SOURCE_NAMES):
        ctx = contexts[f"c{i}"]
        if atom_global_idx > 0:
            prior = scorer_A.score(ctx)
        else:
            prior = FIXED_UTD
        ctx.set_probability(prior)
    pA = make_pipeline(atoms_dict, contexts, relations, atom_def["ground_truth"])
    _, margA = pA.score()
    p_tA = next((m["probabilities"][1] for m in margA if m["variable"]=="a0"), 0.5)
    l_tA = "S" if p_tA > 0.5 else "NS"
    tA_ok = (l_tA == atom_def["ground_truth"])

    # ── Formula B run (separate state, same contexts/relations) ──
    beta_const = 0.7  # using the capped beta value -- claims_seen-based beta
                       # is computed below per-domain via scorer_B's own 'a'
    for i, name in enumerate(SOURCE_NAMES):
        ctx = contexts[f"c{i}"]
        dom = f"{name}.factcheck-eval.org"
        if atom_global_idx > 0:
            a_seen = scorer_B.dynaTD.a.get(dom, 0.0)
            beta = min(a_seen / 10.0, 0.7)
            prior = formula_B_fused(dom, scorer_B.dynaTD, FIXED_UTD, atom_global_idx, beta)
            prior = float(max(0.05, min(0.97, prior)))
        else:
            scorer_B.dynaTD.initialize_domain(dom, FIXED_UTD)
            prior = FIXED_UTD
        ctx.set_probability(prior)
    pB = make_pipeline(atoms_dict, contexts, relations, atom_def["ground_truth"])
    _, margB = pB.score()
    p_tB = next((m["probabilities"][1] for m in margB if m["variable"]=="a0"), 0.5)
    l_tB = "S" if p_tB > 0.5 else "NS"
    tB_ok = (l_tB == atom_def["ground_truth"])

    # ── Vanilla run (flat 0.9 for everyone, no learning at all) ──
    for ctx in contexts.values():
        ctx.set_probability(FIXED_UTD)
    pV = make_pipeline(atoms_dict, contexts, relations, atom_def["ground_truth"])
    _, margV = pV.score()
    p_v = next((m["probabilities"][1] for m in margV if m["variable"]=="a0"), 0.5)
    l_v = "S" if p_v > 0.5 else "NS"
    v_ok = (l_v == atom_def["ground_truth"])

    # ── Update both DynaTD states using their OWN run's posterior ──
    # Formula A: standard update_from_results (unchanged).
    scorer_A.update_from_results(contexts, margA, relations)

    # Formula B: same update_from_results call (correct_count/total_count
    # logic is identical -- only the FUSION formula differs, not the
    # update rule), but BEFORE updating we snapshot current r_s as the
    # new "prev_r_s" for next time.
    if not hasattr(scorer_B.dynaTD, "prev_r_s"):
        scorer_B.dynaTD.prev_r_s = {}
    for i, name in enumerate(SOURCE_NAMES):
        dom = f"{name}.factcheck-eval.org"
        scorer_B.dynaTD.prev_r_s[dom] = scorer_B.dynaTD.get_reliability(dom)
    scorer_B.update_from_results(contexts, margB, relations)

    source_verdicts = {}
    for rel in relations:
        idx  = int(rel.source.id[1:])
        sname = SOURCE_NAMES[idx]
        lbl  = rel.type
        correct = (
            (atom_def["ground_truth"]=="S"  and lbl=="entailment") or
            (atom_def["ground_truth"]=="NS" and lbl=="contradiction")
        )
        source_verdicts[sname] = {"nli": lbl, "correct": correct,
                                   "tier": get_tier(sname)}

    return {
        "claim":    atom_def["claim"][:80],
        "gt":       atom_def["ground_truth"],
        "category": atom_def["category"],
        "p_trustA": round(p_tA, 4), "l_trustA": l_tA, "trustA_ok": tA_ok,
        "p_trustB": round(p_tB, 4), "l_trustB": l_tB, "trustB_ok": tB_ok,
        "p_van":    round(p_v, 4), "l_van":   l_v, "van_ok":   v_ok,
        "source_verdicts": source_verdicts,
    }

async def main():
    N_SUBJECTS = 300

    scorer_A = BayesianTrustFusion(model_path=MODEL_PATH, state_path=STATE_PATH_A)
    scorer_A.dynaTD.reset()
    scorer_B = BayesianTrustFusion(model_path=MODEL_PATH, state_path=STATE_PATH_B)
    scorer_B.dynaTD.reset()
    scorer_B.dynaTD.prev_r_s = {}

    backend = RITSBackend(RITS.LLAMA_3_3_70B_INSTRUCT,
                          model_options={ModelOption.MAX_NEW_TOKENS: 512})
    nli = NLIFixed(backend)

    subjects = load_subjects(n_subjects=N_SUBJECTS)

    print("\n" + "="*72)
    print("  CB_claim_evidence: Formula A (static UTD anchor) vs Formula B (rolling r_s)")
    print(f"  {N_SUBJECTS} facts (id+subject grouping) · 20 sources · UTD=0.9 flat")
    print("  Formula A: fused = (1-beta)*UTD + beta*r_s")
    print("  Formula B: fused = (1-beta)*r_s[i-2] + beta*r_s[i-1]  (rolling, UTD only at atom 0)")
    print("="*72)

    results = []
    tA_correct = tB_correct = van_correct = total = 0
    atom_global_idx = 0

    cat_stats = {}
    source_stats = {n: {"correct":0,"wrong":0,"total":0} for n in SOURCE_NAMES}

    for si, subject in enumerate(subjects):
        atoms = subject["atoms"]
        default_text   = atoms[0]["context_text"]
        conflict_texts = [a["context_text"] for a in atoms[1:]]

        for ai, atom_def in enumerate(atoms):
            try:
                result = await eval_atom_dual(
                    atom_def, default_text, conflict_texts,
                    scorer_A, scorer_B, nli, atom_global_idx,
                )
            except Exception as e:
                print(f"  ERROR [{si+1}] atom {ai}: {e}")
                continue
            if result is None:
                print(f"  [{si+1}/{N_SUBJECTS}] No NLI -- skip")
                continue

            atom_global_idx += 1
            total += 1
            if result["trustA_ok"]: tA_correct += 1
            if result["trustB_ok"]: tB_correct += 1
            if result["van_ok"]:    van_correct += 1

            cat = result["category"]
            if cat not in cat_stats: cat_stats[cat] = {"A":0,"B":0,"van":0,"total":0}
            cat_stats[cat]["total"] += 1
            if result["trustA_ok"]: cat_stats[cat]["A"] += 1
            if result["trustB_ok"]: cat_stats[cat]["B"] += 1
            if result["van_ok"]:    cat_stats[cat]["van"] += 1

            for sn, v in result.get("source_verdicts",{}).items():
                source_stats[sn]["total"] += 1
                if v["correct"]: source_stats[sn]["correct"] += 1
                else:            source_stats[sn]["wrong"]   += 1

            a_sym = "\u2713" if result["trustA_ok"] else "\u2717"
            b_sym = "\u2713" if result["trustB_ok"] else "\u2717"
            v_sym = "\u2713" if result["van_ok"]   else "\u2717"
            print(f"  [{si+1:>3}/{N_SUBJECTS}] {result['category']:<25} "
                  f"GT={result['gt']:<3} "
                  f"A={result['p_trustA']:.3f}\u2192{result['l_trustA']} {a_sym}  "
                  f"B={result['p_trustB']:.3f}\u2192{result['l_trustB']} {b_sym}  "
                  f"V={result['p_van']:.3f}\u2192{result['l_van']} {v_sym}  "
                  f"AccA={tA_correct/total*100:.1f}% AccB={tB_correct/total*100:.1f}% AccV={van_correct/total*100:.1f}%")
            results.append(result)

    print("\n" + "="*72)
    print("  FINAL RESULTS: Formula A vs Formula B vs Vanilla")
    print("="*72)
    accA = tA_correct/max(total,1); accB = tB_correct/max(total,1); accV = van_correct/max(total,1)
    print(f"\n  {'METRIC':<30}  {'FORMULA A':>10}  {'FORMULA B':>10}  {'VANILLA':>10}")
    print(f"  {'Overall accuracy':<30}  {accA:>10.3f}  {accB:>10.3f}  {accV:>10.3f}")

    print(f"\n  By conflict category:")
    for cat, s in sorted(cat_stats.items()):
        if s["total"]:
            a=s["A"]/s["total"]; b=s["B"]/s["total"]; v=s["van"]/s["total"]
            print(f"    {cat:<28}  A={a:.3f}  B={b:.3f}  V={v:.3f}  (n={s['total']})")

    print(f"\n  Total atoms: {total}")

    print(f"\n  Per-source r_s (Formula A state vs Formula B state):")
    for name in SOURCE_NAMES:
        dom = f"{name}.factcheck-eval.org"
        rsA = scorer_A.dynaTD.get_reliability(dom)
        rsB = scorer_B.dynaTD.get_reliability(dom)
        tier = get_tier(name)
        print(f"  {name:<12}  {tier:<12}  r_s_A={rsA:.4f}  r_s_B={rsB:.4f}")

    with open(RESULTS_OUT, "w") as f:
        json.dump({
            "total": total, "trustA_correct": tA_correct,
            "trustB_correct": tB_correct, "van_correct": van_correct,
            "cat_stats": cat_stats, "source_stats": source_stats,
            "results": results,
        }, f, indent=2)
    print(f"\n  Results saved to {RESULTS_OUT}")

if __name__ == "__main__":
    asyncio.run(main())
