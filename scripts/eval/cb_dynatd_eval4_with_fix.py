"""
ConflictBank DynaTD evaluation — Trust vs Vanilla FactReasoner.
v4: v3's corrected (id, subject) grouping + the proposed UNVERIFIED fix.

This is cb_dynatd_eval3.py with ONE addition: after Trust's pipeline.score()
returns a posterior, the verdict is gated through check_verdict_confidence().
If no source supporting the atom clears max_fused_prior >= threshold, the
verdict is downgraded from S/NS to UNVERIFIED instead of being scored
right/wrong against ground truth.

Scoring convention for UNVERIFIED:
  - "non_abstain_accuracy": accuracy computed ONLY over atoms where Trust
    did NOT abstain (the fairer comparison -- how good is Trust when it
    actually commits to an answer?)
  - "all_atoms_accuracy": UNVERIFIED counted as WRONG against every atom
    (the conservative comparison -- what if abstaining always costs you?)
  - abstain_rate: fraction of atoms where Trust abstained

Vanilla is NOT gated -- it has no fused_prior concept (flat 0.9 for every
source), so the same threshold check would be meaningless for it. Vanilla
stays as the unmodified baseline for a fair before/after comparison.
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
STATE_PATH  = "/u/samit/dynaTD_state_cb_dynatd4.json"
MERLIN      = "/u/samit/FactReasoner/merlin"
RESULTS_OUT = "/u/samit/cb_dynatd_results4.json"

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

# ── THE PROPOSED FIX ─────────────────────────────────────────────────────────
VERDICT_CONFIDENCE_THRESHOLD = 0.5

def check_verdict_confidence(p_true, contexts, threshold=VERDICT_CONFIDENCE_THRESHOLD):
    """
    Downgrades a verdict to UNVERIFIED when no source is individually
    trustworthy (max fused_prior < threshold). Uses only signals the
    pipeline already computes (Context.get_probability() = fused_prior).
    """
    raw_verdict = "S" if p_true > 0.5 else "NS"
    fused_priors = [ctx.get_probability() for ctx in contexts.values()]
    max_fused = max(fused_priors) if fused_priors else 0.0

    if max_fused >= threshold:
        return raw_verdict, max_fused, False  # (verdict, max_fused_prior, abstained)
    return "UNVERIFIED", max_fused, True

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

async def eval_atom(atom_def, default_text, conflict_texts,
                    trust_scorer, nli, atom_global_idx):
    atom = Atom(id="a0", text=atom_def["claim"])
    atoms_dict = {"a0": atom}
    seed = hash(atom_def["claim"]) % 100000

    assignments = assign_contexts(atom_def, default_text, conflict_texts, seed)

    contexts = {}
    for i, name in enumerate(SOURCE_NAMES):
        text = assignments[name]
        url  = f"https://{name}.factcheck-eval.org/claim/{seed}"
        ctx  = Context(id=f"c{i}", atom=atom, text=text,
                       title=name, snippet=text[:60], link=url)
        prior = (trust_scorer.score(ctx)
                 if atom_global_idx > 0 else FIXED_UTD)
        ctx.set_probability(prior)
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

    if atom_global_idx > 0:
        for ctx in contexts.values():
            ctx.set_probability(trust_scorer.score(ctx))
    else:
        for ctx in contexts.values():
            ctx.set_probability(FIXED_UTD)
    tp = make_pipeline(atoms_dict, contexts, relations, atom_def["ground_truth"])
    _, t_marg = tp.score()
    p_t = next((m["probabilities"][1] for m in t_marg if m["variable"]=="a0"), 0.5)

    # ── APPLY THE FIX: gate Trust's verdict through confidence check ──
    l_t, max_fused_t, abstained_t = check_verdict_confidence(p_t, contexts)
    t_ok = (l_t == atom_def["ground_truth"])  # False for both wrong AND abstained

    for ctx in contexts.values():
        ctx.set_probability(FIXED_UTD)
    vp = make_pipeline(atoms_dict, contexts, relations, atom_def["ground_truth"])
    _, v_marg = vp.score()
    p_v = next((m["probabilities"][1] for m in v_marg if m["variable"]=="a0"), 0.5)
    l_v = "S" if p_v > 0.5 else "NS"
    v_ok = (l_v == atom_def["ground_truth"])

    trust_scorer.update_from_results(contexts, t_marg, relations)

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
        "p_trust":  round(p_t, 4), "l_trust": l_t, "trust_ok": t_ok,
        "trust_abstained": abstained_t, "max_fused_prior": round(max_fused_t, 4),
        "p_van":    round(p_v, 4), "l_van":   l_v, "van_ok":   v_ok,
        "source_verdicts": source_verdicts,
    }

async def main():
    N_SUBJECTS = 300

    trust_scorer = BayesianTrustFusion(model_path=MODEL_PATH,
                                        state_path=STATE_PATH)
    trust_scorer.dynaTD.reset()

    backend = RITSBackend(RITS.LLAMA_3_3_70B_INSTRUCT,
                          model_options={ModelOption.MAX_NEW_TOKENS: 512})
    nli = NLIFixed(backend)

    subjects = load_subjects(n_subjects=N_SUBJECTS)

    print("\n" + "="*72)
    print("  CB_claim_evidence DynaTD Evaluation  (v4 -- WITH the UNVERIFIED fix)")
    print(f"  {N_SUBJECTS} facts (grouped by id+subject) · 20 sources · UTD=0.9 flat")
    print(f"  Trust verdicts gated: max(fused_prior) >= {VERDICT_CONFIDENCE_THRESHOLD} required, else UNVERIFIED")
    print("="*72)

    results = []
    trust_correct = trust_abstained_count = van_correct = total = 0
    atom_global_idx = 0

    type_stats = {"S":{"trust":0,"van":0,"total":0,"abstain":0}, "NS":{"trust":0,"van":0,"total":0,"abstain":0}}
    cat_stats = {}
    source_stats = {n: {"correct":0,"wrong":0,"total":0} for n in SOURCE_NAMES}

    for si, subject in enumerate(subjects):
        atoms = subject["atoms"]
        default_text   = atoms[0]["context_text"]
        conflict_texts = [a["context_text"] for a in atoms[1:]]

        for ai, atom_def in enumerate(atoms):
            try:
                result = await eval_atom(
                    atom_def, default_text, conflict_texts,
                    trust_scorer, nli, atom_global_idx,
                )
            except Exception as e:
                print(f"  ERROR [{si+1}] atom {ai}: {e}")
                continue
            if result is None:
                print(f"  [{si+1}/{N_SUBJECTS}] No NLI -- skip")
                continue

            atom_global_idx += 1
            total += 1
            if result["trust_ok"]: trust_correct += 1
            if result["trust_abstained"]: trust_abstained_count += 1
            if result["van_ok"]:   van_correct   += 1

            gt = result["gt"]
            type_stats[gt]["total"] += 1
            if result["trust_ok"]: type_stats[gt]["trust"] += 1
            if result["trust_abstained"]: type_stats[gt]["abstain"] += 1
            if result["van_ok"]:   type_stats[gt]["van"]   += 1

            cat = result["category"]
            if cat not in cat_stats: cat_stats[cat] = {"trust":0,"van":0,"total":0,"abstain":0}
            cat_stats[cat]["total"] += 1
            if result["trust_ok"]: cat_stats[cat]["trust"] += 1
            if result["trust_abstained"]: cat_stats[cat]["abstain"] += 1
            if result["van_ok"]:   cat_stats[cat]["van"]   += 1

            for sn, v in result.get("source_verdicts",{}).items():
                source_stats[sn]["total"] += 1
                if v["correct"]: source_stats[sn]["correct"] += 1
                else:            source_stats[sn]["wrong"]   += 1

            t_sym = "\u2713" if result["trust_ok"] else ("\u2014" if result["trust_abstained"] else "\u2717")
            v_sym = "\u2713" if result["van_ok"]   else "\u2717"
            non_abstained_so_far = total - trust_abstained_count
            non_abstain_acc = (trust_correct/non_abstained_so_far*100) if non_abstained_so_far else 0
            print(f"  [{si+1:>3}/{N_SUBJECTS}] {result['category']:<25} "
                  f"GT={result['gt']:<3} "
                  f"T={result['p_trust']:.3f}\u2192{result['l_trust']:<10} {t_sym}  "
                  f"V={result['p_van']:.3f}\u2192{result['l_van']} {v_sym}  "
                  f"AllAcc T={trust_correct/total*100:.1f}% NonAbstainAcc T={non_abstain_acc:.1f}% V={van_correct/total*100:.1f}%")
            results.append(result)

    print("\n" + "="*72)
    print("  FINAL RESULTS (v4 -- WITH the UNVERIFIED fix)")
    print("="*72)
    t_acc_all = trust_correct / max(total, 1)
    v_acc = van_correct / max(total, 1)
    non_abstained = total - trust_abstained_count
    t_acc_non_abstain = trust_correct / max(non_abstained, 1)
    abstain_rate = trust_abstained_count / max(total, 1)

    print(f"\n  {'METRIC':<38}  {'TRUST':>8}  {'VANILLA':>8}")
    print(f"  {'Accuracy (UNVERIFIED counted wrong)':<38}  {t_acc_all:>8.3f}  {v_acc:>8.3f}")
    print(f"  {'Accuracy (excluding UNVERIFIED atoms)':<38}  {t_acc_non_abstain:>8.3f}  {'n/a':>8}")
    print(f"  {'Abstain rate (UNVERIFIED / total)':<38}  {abstain_rate:>8.3f}")
    print(f"  Total atoms: {total}   Abstained: {trust_abstained_count}   Non-abstained: {non_abstained}")

    for gt, s in [("S","TRUE atoms (default)"), ("NS","FALSE atoms (conflict)")]:
        if s["total"] if False else type_stats[gt]["total"]:
            ts = type_stats[gt]
            ta = ts["trust"]/ts["total"]
            va = ts["van"]/ts["total"]
            ab = ts["abstain"]/ts["total"]
            non_ab = ts["total"]-ts["abstain"]
            ta_non = ts["trust"]/max(non_ab,1)
            print(f"    {gt} -- {('TRUE' if gt=='S' else 'FALSE'):<6}  AllAcc T={ta:.3f}  NonAbstainAcc T={ta_non:.3f}  "
                  f"V={va:.3f}  AbstainRate={ab:.3f}  (n={ts['total']})")

    print(f"\n  By conflict category:")
    for cat, s in sorted(cat_stats.items()):
        if s["total"]:
            ta=s["trust"]/s["total"]; va=s["van"]/s["total"]; ab=s["abstain"]/s["total"]
            non_ab = s["total"]-s["abstain"]
            ta_non = s["trust"]/max(non_ab,1)
            print(f"    {cat:<28}  AllAcc T={ta:.3f}  NonAbstainAcc T={ta_non:.3f}  V={va:.3f}  "
                  f"AbstainRate={ab:.3f}  (n={s['total']})")

    print(f"\n  Per-source correctness + DynaTD r_s:")
    for name in SOURCE_NAMES:
        s = source_stats[name]
        if s["total"] == 0: continue
        acc = s["correct"]/s["total"]*100
        dom = f"{name}.factcheck-eval.org"
        a = trust_scorer.dynaTD.a.get(dom, 0); b = trust_scorer.dynaTD.b.get(dom, 1)
        w = a/b if b > 0 else 0
        rs = trust_scorer.dynaTD.get_reliability(dom)
        tier = get_tier(name)
        print(f"  {name:<12}  {tier:<12}  acc={acc:>5.1f}%  r_s={rs:.4f}  w={w:.2f}")

    rel_w  = [trust_scorer.dynaTD.a.get(f"{n}.factcheck-eval.org",0)/trust_scorer.dynaTD.b.get(f"{n}.factcheck-eval.org",1) for n in TIER_RELIABLE]
    unrel_w = [trust_scorer.dynaTD.a.get(f"{n}.factcheck-eval.org",0)/trust_scorer.dynaTD.b.get(f"{n}.factcheck-eval.org",1) for n in TIER_UNRELIABLE]
    avg_rel = sum(rel_w)/len(rel_w) if rel_w else 0
    avg_unrel = sum(unrel_w)/len(unrel_w) if unrel_w else 0
    print(f"\n  avg w reliable={avg_rel:.2f}  avg w unreliable={avg_unrel:.2f}  separation={avg_rel-avg_unrel:+.2f}")

    with open(RESULTS_OUT, "w") as f:
        json.dump({
            "total": total, "trust_correct": trust_correct,
            "trust_abstained": trust_abstained_count, "van_correct": van_correct,
            "type_stats": type_stats, "cat_stats": cat_stats,
            "source_stats": source_stats, "results": results,
        }, f, indent=2)
    print(f"\n  Results saved to {RESULTS_OUT}")

if __name__ == "__main__":
    asyncio.run(main())
