"""
final_eval_40_refresh.py - 5 systems on the 40-row REFRESHED (July 2026, tbs +/-45d) contexts.
Systems: cc_trust, cc_vanilla, fp_trust, fp_vanilla, guardian. GS41 merged if present.
Trust = clean consensus weights + credibility_scorer_v3 (MBFC prior).
Phase 1: N_WARMUP prequential warmup (fresh state, NO poisoned init, NO cache overwrite).
Phase 2: state frozen -> eval all systems on same rows.
"""
import os as _os, sys as _sys
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
_sys.path.insert(0, _os.path.join(_REPO, 'src'))
_sys.path.insert(0, _os.path.join(_REPO, 'experiments', 'consensus_trust'))
import asyncio, json, sys, os, shutil, time

import fact_reasoner.core.trust.credibility_scorer as _cs
import fact_reasoner.core.trust.credibility_scorer_v3 as _v3
_cs.score_url = _v3.score_url

from mellea_ibm.rits import RITSBackend, RITS
from mellea.backends import ModelOption
import mellea.stdlib.functional as mfuncs
from mellea.stdlib.context import SimpleContext

from state_media_eval import build_relations, make_pipeline
from granite_switch_vs_factreaser_demo import NLIFixed
from fact_reasoner.core.base import Atom, Context
from fact_reasoner.core.trust.credibility_fusion import CredibilityTrustFusion

CC_JSONL  = 'data/trust_eval/eval_dataset_cc.jsonl'   # original cached retrieval — good contexts
FP_JSONL  = 'data/trust_eval/eval_dataset_fp.jsonl'   # original cached retrieval
GS41_JSON = 'data/trust_eval/gs41_eval_refresh2026_results.json'
STATE     = 'data/trust_eval/dynaTD_state_40v2.json'
OUT_DIR   = f"data/trust_eval/final_eval_40v2_{time.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(OUT_DIR, exist_ok=True)
N_WARMUP  = 1

JUDGE_PROMPT = """Context documents:
{{ctx_block}}

Claim: {{claim}}

Based only on the context documents, does the claim contain factual errors or unsupported assertions? Answer with exactly one word: 'yes' (errors/unsupported) or 'no' (consistent)."""

llm      = RITSBackend(RITS.LLAMA_3_3_70B_INSTRUCT, model_options={ModelOption.MAX_NEW_TOKENS: 512})
guardian = RITSBackend(RITS.GRANITE_GUARDIAN_3_3_8B, model_options={ModelOption.MAX_NEW_TOKENS: 16})
nli = NLIFixed(llm)

def load_jsonl(p):
    with open(p) as f: return [json.loads(l) for l in f if l.strip()]

def ctx_block_for(row):
    lut = {c['id']: c for c in row['contexts']}
    ctxs = [lut[cid] for cid in row['atoms'][0]['contexts'][:10] if cid in lut]
    return "\n".join(f"- {c['text'][:300]}" for c in ctxs)

def build(data, scorer=None):
    atoms = {a['id']: Atom(id=a['id'], text=a['text']) for a in data['atoms']}
    lut   = {c['id']: c for c in data['contexts']}
    ctxs  = {}
    for a in data['atoms']:
        atom, mine = atoms[a['id']], []
        for cid in a['contexts'][:10]:
            if cid not in lut: continue
            c = lut[cid]
            ctx = Context(id=cid, atom=atom, text=c['text'],
                          title=c.get('title',''), link=c.get('link',''))
            ctx.set_probability(scorer.score(ctx) if scorer else 0.9)
            ctxs[cid] = ctx; mine.append(ctx)
        atom.add_contexts(mine)
    return atoms, ctxs

async def run_fr(data, scorer=None, label="", update=True):
    if data is None: return None
    gt = data['ground_truth']
    atoms, ctxs = build(data, scorer)
    if len(ctxs) < 2: return None
    rels = None
    import random as _rnd
    for k in range(10):
        try:
            rels = build_relations(atoms=atoms, contexts=ctxs, nli_extractor=nli,
                                   rel_atom_context=True, rel_context_context=False,
                                   use_summarized_contexts=False)
            break
        except Exception as e:
            print(f"  [FAIL:{label}] NLI try {k+1}/10: {type(e).__name__}: {str(e)[:80]}", flush=True)
            await asyncio.sleep(min(45, 5*(k+1)) + _rnd.uniform(0, 3))
    if rels is None:
        print(f"  [GAVE-UP:{label}] after 10 tries", flush=True)
        return None
    if not rels: return None   # rels==[] means NLI ran fine, found nothing informative: legit skip
    live = {r.source.id for r in rels}
    ctxs = {k: v for k, v in ctxs.items() if k in live}
    rels = [r for r in rels if r.source.id in ctxs]
    if not ctxs: return None
    for a in atoms.values():
        a.contexts = {c.id: c for c in a.get_contexts().values() if c.id in ctxs}
    _, marg = make_pipeline(atoms, ctxs, rels, gt).score()
    probs = {m["variable"]: m["probabilities"][1] for m in marg if m["variable"] in atoms}
    if not probs: return None
    prec = sum(1 for p in probs.values() if p > 0.5) / len(probs)
    verdict = "S" if prec > 0.5 else "NS"
    if scorer and update:
        scorer.update_from_results(ctxs, marg, rels)
    return {"verdict": verdict, "precision": round(prec, 4),
            "n_atoms": len(probs), "correct": verdict == gt}

async def run_guardian(row):
    gt = row['ground_truth']
    for k in range(10):
        try:
            out = await mfuncs.ainstruct(
                JUDGE_PROMPT, context=SimpleContext(), backend=guardian,
                user_variables={"ctx_block": ctx_block_for(row),
                                "claim": row['atoms'][0]['text']})
            raw = str(out).strip().lower()
            v = "NS" if "yes" in raw else "S"
            return {"verdict": v, "correct": v == gt, "raw": raw[:25]}
        except Exception as e:
            print(f"  [Guardian] try {k+1}: {str(e)[:45]}", flush=True)
            await asyncio.sleep(5*(k+1))
    return None

async def main():
    if os.path.exists(STATE): os.remove(STATE)
    cc, fp = load_jsonl(CC_JSONL), load_jsonl(FP_JSONL)
    fp_by  = {r['input']: r for r in fp}
    print(f"rows: cc={len(cc)} fp={len(fp)}", flush=True)

    gs41_by = {}
    if os.path.exists(GS41_JSON):
        gs41_by = {r['input'][:80]: r for r in json.load(open(GS41_JSON))}
        print(f"GS41 merged: {len(gs41_by)} rows", flush=True)

    scorer = CredibilityTrustFusion(state_path=STATE)

    print(f"=== WARMUP x{N_WARMUP} ===", flush=True)
    for p in range(N_WARMUP):
        for i, r in enumerate(cc):
            try:
                await run_fr(r, scorer, f"warm_cc{i}", update=True)
                fpr = fp_by.get(r['input'])
                if fpr: await run_fr(fpr, scorer, f"warm_fp{i}", update=True)
            except Exception as e:
                print(f"  warm skip {i}: {str(e)[:50]}", flush=True)
    scorer.dynaTD._save()
    shutil.copy(STATE, f"{OUT_DIR}/state_after_warmup.json")
    print("=== WARMUP done, state frozen ===", flush=True)

    SYS = ["cc_trust", "cc_vanilla", "fp_trust", "fp_vanilla", "guardian"] + (["gs41"] if gs41_by else [])
    ind = {s: {"c": 0, "t": 0} for s in SYS}
    rows_out = []
    for i, r in enumerate(cc):
        gt = r['ground_truth']; fpr = fp_by.get(r['input'])
        o = {}
        o["cc_trust"]   = await run_fr(r,   scorer, "cc_trust",   update=False)
        o["cc_vanilla"] = await run_fr(r,   None,   "cc_vanilla", update=False)
        o["fp_trust"]   = await run_fr(fpr, scorer, "fp_trust",   update=False)
        o["fp_vanilla"] = await run_fr(fpr, None,   "fp_vanilla", update=False)
        o["guardian"]   = await run_guardian(r)
        if gs41_by:
            g = gs41_by.get(r['input'][:80])
            o["gs41"] = ({"verdict": g["gs41"], "correct": g["gs41"] == gt}
                         if g and g.get("gs41") in ("S", "NS") else None)
        line = f"[{i+1:>2}/{len(cc)}] {str(r.get('account',''))[:14]:<14} GT={gt} "
        for s in SYS:
            v = o.get(s)
            if v: ind[s]["t"] += 1; ind[s]["c"] += v["correct"]
            line += f"| {s[:2]}{s.split('_')[-1][:3]}={'--' if not v else v['verdict']}"
        print(line, flush=True)
        rows_out.append({"account": r.get('account'), "gt": gt,
                         "input": r['input'][:80], **{s: o.get(s) for s in SYS}})
        json.dump({"results": rows_out, "individual": ind},
                  open(f"{OUT_DIR}/results.json", "w"), indent=1)

    # ---- SECOND PASS: retry every (row, system) that returned None ----------
    FR = [s for s in SYS if s not in ("guardian", "gs41")]
    print("=== SECOND PASS over failures ===", flush=True)
    for i, r in enumerate(cc):
        ro = rows_out[i]; fpr = fp_by.get(r['input'])
        for s in FR:
            if ro.get(s) is None:
                src_data = r if s.startswith("cc") else fpr
                sc = scorer if s.endswith("trust") else None
                print(f"  retrying row {i} {s}", flush=True)
                ro[s] = await run_fr(src_data, sc, f"retry_{s}{i}", update=False)
        if ro.get("guardian") is None:
            print(f"  retrying row {i} guardian", flush=True)
            ro["guardian"] = await run_guardian(r)
    # ---- rebuild tallies over COMPLETE rows only -----------------------------
    complete = [ro for ro in rows_out if all(ro.get(s) for s in FR)]
    dropped  = [ (i, [s for s in FR if not ro.get(s)]) for i, ro in enumerate(rows_out) if ro not in complete ]
    ind2 = {s: {"c": 0, "t": 0} for s in SYS}
    for ro in complete:
        for s in SYS:
            v = ro.get(s)
            if v: ind2[s]["t"] += 1; ind2[s]["c"] += v["correct"]
    json.dump({"results": rows_out, "individual_all": ind, "individual_complete_rows": ind2,
               "dropped_rows": dropped},
              open(f"{OUT_DIR}/results.json", "w"), indent=1)
    print(f"DROPPED (incomplete after 2 passes): {len(dropped)} rows -> {dropped}", flush=True)
    ind = ind2
    print("=" * 56, flush=True)
    print(f"40-ROW original contexts, v3 prior -- COMPLETE ROWS ONLY (n={len(complete)})", flush=True)
    print("=" * 56, flush=True)
    for s in SYS:
        c, t = ind[s]["c"], ind[s]["t"]
        print(f"  {s:<11}: {c:>2}/{t:<2} = {c/max(t,1):.1%}", flush=True)

asyncio.run(main())
