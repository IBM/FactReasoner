"""
ConflictBank DynaTD evaluation — Trust vs Vanilla FactReasoner.
v2: fixed tier assignment, new tier config (8/8/4), full dataset.

Dataset: Warrieryes/CB_claim_evidence (streaming)
  Categories: default (TRUE), misinformation_conflict/temporal_conflict/
              semantic_conflict (all FALSE)

Design:
  - Per subject: 1 TRUE atom (default) + up to 2 FALSE atoms (conflict variants)
  - 20 fictional sources, ALL UTD=0.9 (flat — no UTD head start)
  - Sources split by context assignment probability:
      Reliable   (source_a..h, 8): 95% → default ctx, 5%  → conflict ctx
      Unreliable (source_i..p, 8): 5%  → default ctx, 95% → conflict ctx
      Noisy      (source_q..t, 4): 50% → default ctx, 50% → conflict ctx
  - Atom 1: Trust = Vanilla (beta=0, no DynaTD learning yet)
  - Atom 2+: DynaTD fused priors diverge based on learned reliability
  - Ground truth: default → S, all conflict categories → NS

Ground truth verification:
  S  : category == "default"
  NS : category in {misinformation_conflict, temporal_conflict, semantic_conflict}
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
STATE_PATH  = "/u/samit/dynaTD_state_cb_dynatd2.json"
MERLIN      = "/u/samit/FactReasoner/merlin"
RESULTS_OUT = "/u/samit/cb_dynatd_results2.json"

FIXED_UTD    = 0.9
N_SOURCES    = 20
SOURCE_NAMES = [f"source_{chr(97+i)}" for i in range(N_SOURCES)]

# ── Tier config ───────────────────────────────────────────────────────────────
# source_a..h  (indices 0-7):  reliable   → 95% default
# source_i..p  (indices 8-15): unreliable → 95% conflict
# source_q..t  (indices 16-19): noisy     → 50/50
TIER_RELIABLE   = SOURCE_NAMES[0:8]    # a..h
TIER_UNRELIABLE = SOURCE_NAMES[8:16]   # i..p
TIER_NOISY      = SOURCE_NAMES[16:20]  # q..t

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

# ── NLI fix ───────────────────────────────────────────────────────────────────
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

def load_subjects(n_subjects: int, seed: int = 42) -> list:
    print(f"Loading CB_claim_evidence (streaming, target={n_subjects})...")
    ds = load_dataset("Warrieryes/CB_claim_evidence", split="train", streaming=True)

    by_fact = {}
    current_fact_id = None
    n_streamed = 0
    
    for row in ds:
        cat = row["category"]
        subj = row["subject"]
        
        # STATE MACHINE: A 'default' row marks the birth of a brand new fact group
        if cat == "default":
            # Create a localized unique identifier using the stream order
            current_fact_id = f"fact_{n_streamed}_{subj}"
            by_fact[current_fact_id] = {
                "subject_name": subj,
                "default": [],
                "conflict": []
            }
            by_fact[current_fact_id]["default"].append(row)
            
        # If it's a conflict row, bind it to the active default fact sequence
        elif cat in CONFLICT_CATEGORIES and current_fact_id is not None:
            by_fact[current_fact_id]["conflict"].append(row)
            
        n_streamed += 1
        
        # Early exit check once we have enough filled groups
        valid_triplets = sum(1 for f in by_fact.values() if f["default"] and f["conflict"])
        if valid_triplets >= n_subjects * 2:
            break

    print(f"Streamed {n_streamed} rows. Captured {len(by_fact)} chronological fact clusters.")

    # Filter out any orphaned facts that lack matching conflict components
    valid_facts = {
        k: v for k, v in by_fact.items() 
        if v["default"] and v["conflict"]
    }
    print(f"Valid sequence groups: {len(valid_facts)}")

    rng = random.Random(seed)
    selected_keys = rng.sample(list(valid_facts.keys()), min(n_subjects, len(valid_facts)))

    subjects = []
    for k in selected_keys:
        d = valid_facts[k]
        default_row = rng.choice(d["default"])
        # Pull up to 2 conflict variations belonging strictly to this exact cluster
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
            
        subjects.append({"subject": d["subject_name"], "atoms": atoms})

    total_atoms = sum(len(s["atoms"]) for s in subjects)
    print(f"Selected {len(subjects)} structural sequences, yielding {total_atoms} total atoms.")
    return subjects

    
# ── Source assignment ─────────────────────────────────────────────────────────
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

# ── Pipeline ──────────────────────────────────────────────────────────────────
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

# ── Single atom ───────────────────────────────────────────────────────────────
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

    # Trust run
    if atom_global_idx > 0:
        for ctx in contexts.values():
            ctx.set_probability(trust_scorer.score(ctx))
    else:
        for ctx in contexts.values():
            ctx.set_probability(FIXED_UTD)
    tp = make_pipeline(atoms_dict, contexts, relations, atom_def["ground_truth"])
    _, t_marg = tp.score()
    p_t = next((m["probabilities"][1] for m in t_marg if m["variable"]=="a0"), 0.5)
    l_t = "S" if p_t > 0.5 else "NS"
    t_ok = (l_t == atom_def["ground_truth"])

    # Vanilla run
    for ctx in contexts.values():
        ctx.set_probability(FIXED_UTD)
    vp = make_pipeline(atoms_dict, contexts, relations, atom_def["ground_truth"])
    _, v_marg = vp.score()
    p_v = next((m["probabilities"][1] for m in v_marg if m["variable"]=="a0"), 0.5)
    l_v = "S" if p_v > 0.5 else "NS"
    v_ok = (l_v == atom_def["ground_truth"])

    # DynaTD update
    trust_scorer.update_from_results(contexts, t_marg, relations)

    # Per-source correctness
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
        "p_van":    round(p_v, 4), "l_van":   l_v, "van_ok":   v_ok,
        "source_verdicts": source_verdicts,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    N_SUBJECTS = 300   # full dataset run

    trust_scorer = BayesianTrustFusion(model_path=MODEL_PATH,
                                        state_path=STATE_PATH)
    trust_scorer.dynaTD.reset()

    backend = RITSBackend(RITS.LLAMA_3_3_70B_INSTRUCT,
                          model_options={ModelOption.MAX_NEW_TOKENS: 512})
    nli = NLIFixed(backend)

    subjects = load_subjects(n_subjects=N_SUBJECTS)

    print("\n" + "="*72)
    print("  CB_claim_evidence DynaTD Evaluation  (v2)")
    print(f"  {N_SUBJECTS} subjects · 20 sources · ALL UTD=0.9 (flat)")
    print("  DynaTD is the ONLY differentiator between Trust and Vanilla")
    print("="*72)
    print(f"\n  Tier config:")
    print(f"    Reliable   (a-h, 8): P(default)=0.95")
    print(f"    Unreliable (i-p, 8): P(default)=0.05")
    print(f"    Noisy      (q-t, 4): P(default)=0.50")

    results = []
    trust_correct = van_correct = total = 0
    atom_global_idx = 0

    type_stats = {
        "S":  {"trust":0,"van":0,"total":0},
        "NS": {"trust":0,"van":0,"total":0},
    }
    cat_stats = {}   # per conflict category
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
                print(f"  [{si+1}/{N_SUBJECTS}] No NLI — skip")
                continue

            atom_global_idx += 1
            total += 1
            if result["trust_ok"]: trust_correct += 1
            if result["van_ok"]:   van_correct   += 1

            gt = result["gt"]
            type_stats[gt]["total"] += 1
            if result["trust_ok"]: type_stats[gt]["trust"] += 1
            if result["van_ok"]:   type_stats[gt]["van"]   += 1

            cat = result["category"]
            if cat not in cat_stats:
                cat_stats[cat] = {"trust":0,"van":0,"total":0}
            cat_stats[cat]["total"] += 1
            if result["trust_ok"]: cat_stats[cat]["trust"] += 1
            if result["van_ok"]:   cat_stats[cat]["van"]   += 1

            for sn, v in result.get("source_verdicts",{}).items():
                source_stats[sn]["total"] += 1
                if v["correct"]: source_stats[sn]["correct"] += 1
                else:            source_stats[sn]["wrong"]   += 1

            t_sym = "✓" if result["trust_ok"] else "✗"
            v_sym = "✓" if result["van_ok"]   else "✗"
            print(f"  [{si+1:>3}/{N_SUBJECTS}] {result['category']:<25} "
                  f"GT={result['gt']:<3} "
                  f"T={result['p_trust']:.3f}→{result['l_trust']} {t_sym}  "
                  f"V={result['p_van']:.3f}→{result['l_van']} {v_sym}  "
                  f"Acc T={trust_correct/total*100:.1f}% V={van_correct/total*100:.1f}%")
            results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("  FINAL RESULTS")
    print("="*72)
    t_acc = trust_correct / max(total, 1)
    v_acc = van_correct   / max(total, 1)

    print(f"\n  {'METRIC':<28}  {'TRUST':>8}  {'VANILLA':>8}  {'Δ':>8}")
    print(f"  {'-'*28}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'Overall accuracy':<28}  {t_acc:>8.3f}  {v_acc:>8.3f}  "
          f"{t_acc-v_acc:>+8.3f}")
    for gt, s in [("S","TRUE atoms (default)"),
                  ("NS","FALSE atoms (conflict)")]:
        if type_stats[gt]["total"]:
            ta = type_stats[gt]["trust"] / type_stats[gt]["total"]
            va = type_stats[gt]["van"]   / type_stats[gt]["total"]
            print(f"  {s:<28}  {ta:>8.3f}  {va:>8.3f}  {ta-va:>+8.3f}  "
                  f"(n={type_stats[gt]['total']})")

    print(f"\n  By conflict category:")
    for cat, s in sorted(cat_stats.items()):
        if s["total"]:
            ta = s["trust"]/s["total"]
            va = s["van"]/s["total"]
            print(f"    {cat:<28}  T={ta:.3f}  V={va:.3f}  Δ={ta-va:+.3f}  "
                  f"(n={s['total']})")

    print(f"\n  Total atoms: {total}")

    # Per-source DynaTD table
    print(f"\n  Per-source correctness + DynaTD r_s:")
    print(f"  {'SOURCE':<12}  {'TIER':<12}  {'P_def':>6}  "
          f"{'ACC%':>6}  {'r_s':>8}  {'w':>7}  STATUS")
    print(f"  {'-'*80}")

    for name in SOURCE_NAMES:
        s = source_stats[name]
        if s["total"] == 0: continue
        acc  = s["correct"] / s["total"] * 100
        dom  = f"{name}.factcheck-eval.org"
        a    = trust_scorer.dynaTD.a.get(dom, 0)
        b    = trust_scorer.dynaTD.b.get(dom, 1)
        w    = a/b if b > 0 else 0
        rs   = 0.1 + 0.8/(1+math.exp(-2*(w-1)))
        tier = get_tier(name)
        p_d  = P_DEFAULT[name]
        status = ("++ reliable"   if w > 3   else
                  "+  reliable"   if w > 1.5 else
                  "~  neutral"    if w > 0.8 else
                  "-  unreliable" if w > 0.3 else
                  "-- very low")
        print(f"  {name:<12}  {tier:<12}  {p_d:>6.2f}  "
              f"{acc:>6.1f}%  {rs:>8.4f}  {w:>7.2f}  {status}")

    print(f"\n  VALIDATION: reliable (a-h) r_s should be > unreliable (i-p) r_s")
    rel_rs  = [trust_scorer.dynaTD.a.get(f"{n}.factcheck-eval.org",0) /
               trust_scorer.dynaTD.b.get(f"{n}.factcheck-eval.org",1)
               for n in TIER_RELIABLE]
    unrel_rs = [trust_scorer.dynaTD.a.get(f"{n}.factcheck-eval.org",0) /
                trust_scorer.dynaTD.b.get(f"{n}.factcheck-eval.org",1)
                for n in TIER_UNRELIABLE]
    avg_rel   = sum(rel_rs)  / len(rel_rs)  if rel_rs  else 0
    avg_unrel = sum(unrel_rs)/ len(unrel_rs) if unrel_rs else 0
    print(f"  avg w reliable={avg_rel:.2f}  avg w unreliable={avg_unrel:.2f}  "
          f"separation={avg_rel-avg_unrel:+.2f}")
    if avg_rel > avg_unrel:
        print("  ✓ DynaTD correctly learned reliable > unreliable")
    else:
        print("  ✗ DynaTD did NOT separate tiers — check NLI signal")

    with open(RESULTS_OUT, "w") as f:
        json.dump({
            "total": total, "trust_correct": trust_correct,
            "van_correct": van_correct, "type_stats": type_stats,
            "cat_stats": cat_stats, "source_stats": source_stats,
            "results": results,
        }, f, indent=2)
    print(f"\n  Results saved to {RESULTS_OUT}")

if __name__ == "__main__":
    asyncio.run(main())
