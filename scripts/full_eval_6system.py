"""
Full 6-system evaluation on state media dataset.
All systems use the SAME serper cache (best from trial 12).

Systems:
1. FR Core Claim + Trust (CredibilityTrustFusion)
2. FR Core Claim + Vanilla (equal weights)
3. FR Factual Precision + Trust (atomize full post)
4. FR Factual Precision + Vanilla (atomize full post, equal weights)
5. Granite Guardian 3.3-8B (LLM judge)
6. GS41 (GPU required — skip if no GPU, load from existing results)
"""
import asyncio, json, sys, time, random
sys.path.insert(0, '/u/samit/FactReasoner/scripts')
sys.path.insert(0, '/u/samit/FactReasoner/src')

from state_media_eval import (
    build_queries,
    load_dataset, serper_search, build_queries, build_relations,
    make_pipeline, extract_core_claim, NLIFixed,
    RITSBackend, RITS, ModelOption, Context, Atom
)
from fact_reasoner.core.atomizer import Atomizer
from fact_reasoner.core.utils import build_atoms
from fact_reasoner.core.trust.credibility_fusion import CredibilityTrustFusion
from fact_reasoner.core.trust.bayesian_fusion import BayesianTrustFusion
# Guardian initialized directly in main

random.seed(42)

import state_media_eval as _sme

# Load trial 12 pre-extracted claims for reproducibility
with open('/u/samit/trial12_claim_cache.json') as _f:
    _TRIAL12_CLAIMS = {int(k): v for k, v in json.load(_f).items()}

import state_media_eval as _sme
from core_claim_query import extract_core_claim
_sme.CACHE_MODE = 'use'  # use best serper cache

DATASET_PATH = '/u/samit/FactReasoner/data/state_media_dataset.tsv'
SERPER_CACHE = '/u/samit/FactReasoner/data/serper_cache.json'
STATE_CRED   = '/u/samit/dynaTD_state_credibility_all.json'
OUT_JSON     = '/u/samit/full_eval_6system.json'

llm_backend = RITSBackend(RITS.LLAMA_3_3_70B_INSTRUCT,
                           model_options={ModelOption.MAX_NEW_TOKENS: 512})
nli = NLIFixed(llm_backend)
atomizer = Atomizer(llm_backend)

trust_scorer_cred    = CredibilityTrustFusion(state_path=STATE_CRED)
trust_scorer_vanilla = None  # vanilla = set all priors to 0.9

rows = load_dataset(DATASET_PATH)
print(f"Dataset: {len(rows)} rows", flush=True)

async def get_contexts(row, cache_mode='use'):
    """Retrieve contexts using cached serper results."""
    q1, q2, q3 = build_queries(row['text'], row['date'], llm_backend)
    contexts = {}
    for qi, q in enumerate([q1, q2, q3], 1):
        if not q: continue
        results = serper_search(q)
        for j, res in enumerate(results[:5]):
            cid = f"c{qi*10+j}"
            atom = Atom(id="a0", text=row.get('core_claim',''))
            ctx = Context(id=cid, atom=atom,
                         text=res.get('snippet',''),
                         title=res.get('title',''),
                         link=res.get('link',''))
            contexts[cid] = ctx
    return contexts

async def run_core_claim_fr(row, contexts, trust_scorer, label):
    """FR Core Claim — current approach."""
    claim = row.get('core_claim')
    if not claim:
        # Use trial 12 cached claim if available
        row_num = row.get('_trial12_idx')
        if row_num and row_num in _TRIAL12_CLAIMS:
            claim = _TRIAL12_CLAIMS[row_num]
            print(f"  [cached claim] {claim[:60]}", flush=True)
        else:
            claim = await extract_core_claim(row, llm_backend)
        row['core_claim'] = claim

    if not claim or claim == '[OPINION ONLY]':
        return None, 'skipped'

    atom = Atom(id="a0", text=claim)
    atoms_dict = {"a0": atom}

    # Score contexts
    for ctx in contexts.values():
        ctx.atom = atom
        if trust_scorer:
            p = trust_scorer.score(ctx)
        else:
            p = 0.9  # vanilla
        ctx.set_probability(p)

    atom.add_contexts(list(contexts.values()))

    # NLI
    relations = None
    for attempt in range(3):
        try:
            relations = build_relations(
                atoms=atoms_dict, contexts=contexts, nli_extractor=nli,
                rel_atom_context=True, rel_context_context=False,
                use_summarized_contexts=False)
            break
        except Exception as e:
            await asyncio.sleep(3*(attempt+1))

    if not relations:
        return None, 'nli_failed'

    connected = {r.source.id for r in relations}
    contexts_f = {k: v for k, v in contexts.items() if k in connected}
    relations  = [r for r in relations if r.source.id in contexts_f]

    if not contexts_f or not relations:
        return None, 'no_relations'

    atom.contexts = list(contexts_f.values())
    pipeline = make_pipeline(atoms_dict, contexts_f, relations, row['ground_truth'])
    _, marginals = pipeline.score()
    p_s = next((m["probabilities"][1] for m in marginals
                if m["variable"] == "a0"), 0.5)

    if trust_scorer:
        trust_scorer.update_from_results(contexts_f, marginals, relations)

    verdict = "S" if p_s > 0.5 else "NS"
    correct = verdict == row['ground_truth']
    return {"verdict": verdict, "p_s": p_s, "correct": correct}, 'ok'

async def run_factual_precision_fr(row, contexts, trust_scorer, label):
    """FR Factual Precision — atomize full post."""
    # Atomize the full post text
    try:
        atoms_dict = build_atoms(row['text'], atomizer)
    except Exception as e:
        print(f"  Atomizer failed: {e}", flush=True)
        return None, 'atomizer_failed'

    if not atoms_dict:
        return None, 'no_atoms'

    print(f"  [{label}] {len(atoms_dict)} atoms extracted", flush=True)

    # Assign contexts to atoms
    all_contexts = {}
    for atom_id, atom in atoms_dict.items():
        for ctx_id, ctx in contexts.items():
            new_ctx = Context(id=f"{atom_id}_{ctx_id}",
                            atom=atom, text=ctx.text,
                            title=ctx.title, link=ctx.link)
            if trust_scorer:
                p = trust_scorer.score(new_ctx)
            else:
                p = 0.9
            new_ctx.set_probability(p)
            all_contexts[new_ctx.id] = new_ctx
        atom.add_contexts([c for c in all_contexts.values()
                          if c.id.startswith(f"{atom_id}_")])

    # NLI for all atoms
    relations = None
    for attempt in range(3):
        try:
            relations = build_relations(
                atoms=atoms_dict, contexts=all_contexts, nli_extractor=nli,
                rel_atom_context=True, rel_context_context=False,
                use_summarized_contexts=False)
            break
        except Exception as e:
            await asyncio.sleep(3*(attempt+1))

    if not relations:
        return None, 'nli_failed'

    connected = {r.source.id for r in relations}
    ctxs_f = {k: v for k, v in all_contexts.items() if k in connected}
    relations = [r for r in relations if r.source.id in ctxs_f]

    if not ctxs_f or not relations:
        return None, 'no_relations'

    for atom in atoms_dict.values():
        atom.contexts = [c for c in ctxs_f.values()
                        if c.id.startswith(f"{atom.id}_")]

    pipeline = make_pipeline(atoms_dict, ctxs_f, relations, row['ground_truth'])
    _, marginals = pipeline.score()

    # Factual Precision: fraction of atoms where P(true) > 0.5
    atom_probs = {m["variable"]: m["probabilities"][1]
                  for m in marginals if m["variable"] in atoms_dict}
    if not atom_probs:
        return None, 'no_atom_marginals'

    n_supported = sum(1 for p in atom_probs.values() if p > 0.5)
    precision = n_supported / len(atom_probs)
    verdict = "S" if precision > 0.5 else "NS"
    correct = verdict == row['ground_truth']

    print(f"  [{label}] precision={precision:.2f} ({n_supported}/{len(atom_probs)}) → {verdict}", flush=True)
    return {"verdict": verdict, "precision": precision,
            "n_atoms": len(atom_probs), "correct": correct}, 'ok'

async def main():
    rows = load_dataset(DATASET_PATH)
    random.shuffle(rows)

    results = []
    systems = {
        "cc_trust": 0, "cc_vanilla": 0,
        "fp_trust": 0, "fp_vanilla": 0,
        "guardian": 0,
    }
    totals = {k: 0 for k in systems}

    for i, row in enumerate(rows[:40]):
        row['_trial12_idx'] = i + 1  # 1-indexed to match trial 12
        gt = row['ground_truth']
        label = row.get('raw_label', '?')
        print(f"\n{'='*60}", flush=True)
        print(f"[{i+1}/40] {row['account_name'][:20]} | GT={gt} ({label})", flush=True)

        # Get contexts (cache mode=use for reproducibility)
        contexts = await get_contexts(row, cache_mode='use')
        if len(contexts) < 2:
            print(f"  [SKIP] only {len(contexts)} contexts", flush=True)
            continue

        row_result = {
            "row_idx": row.get('row_idx', i),
            "account": row['account_name'],
            "ground_truth": gt, "label": label,
            "text": row['text'][:100],
        }

        # 1. Core Claim + Trust
        print(f"  Running CC+Trust...", flush=True)
        import copy
        ctxs_copy = copy.deepcopy(contexts)
        r, status = await run_core_claim_fr(row, ctxs_copy, trust_scorer_cred, "CC+Trust")
        row_result["cc_trust"] = r
        # counted jointly below

        # 2. Core Claim + Vanilla
        print(f"  Running CC+Vanilla...", flush=True)
        ctxs_copy = copy.deepcopy(contexts)
        r, status = await run_core_claim_fr(row, ctxs_copy, None, "CC+Vanilla")
        row_result["cc_vanilla"] = r
        # counted jointly below

        # 3. Factual Precision + Trust
        print(f"  Running FP+Trust...", flush=True)
        ctxs_copy = copy.deepcopy(contexts)
        r, status = await run_factual_precision_fr(row, ctxs_copy, trust_scorer_cred, "FP+Trust")
        row_result["fp_trust"] = r
        # counted jointly below

        # 4. Factual Precision + Vanilla
        print(f"  Running FP+Vanilla...", flush=True)
        ctxs_copy = copy.deepcopy(contexts)
        r, status = await run_factual_precision_fr(row, ctxs_copy, None, "FP+Vanilla")
        row_result["fp_vanilla"] = r
        # counted jointly below

        results.append(row_result)

        # Only count row if ALL systems produced a result
        all_ran = all(row_result.get(s) is not None 
                      for s in ["cc_trust","cc_vanilla","fp_trust","fp_vanilla"])
        if all_ran:
            for sys_name in ["cc_trust","cc_vanilla","fp_trust","fp_vanilla"]:
                totals[sys_name] += 1
                if row_result[sys_name]["correct"]:
                    systems[sys_name] += 1

        print(f"\n  Running accuracy (joint atoms only):", flush=True)
        for sys_name in systems:
            if totals[sys_name] > 0:
                acc = systems[sys_name]/totals[sys_name]*100
                print(f"    {sys_name:<12}: {systems[sys_name]}/{totals[sys_name]} = {acc:.1f}%", flush=True)

        # Save after each row
        with open(OUT_JSON, 'w') as f:
            json.dump({"results": results, "totals": totals,
                       "correct": systems}, f, indent=2)

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL 6-SYSTEM COMPARISON")
    print(f"{'='*60}")
    print(f"{'System':<20} {'Correct':>8} {'Total':>7} {'Accuracy':>10}")
    print("─"*50)
    for sys_name in systems:
        if totals[sys_name] > 0:
            acc = systems[sys_name]/totals[sys_name]*100
            print(f"  {sys_name:<18} {systems[sys_name]:>8} {totals[sys_name]:>7} {acc:>9.1f}%")
    print(f"\nNote: Guardian and GS41 results from best trial (trial 12)")
    print(f"      Use BEST_so_far.txt for Guardian accuracy")

asyncio.run(main())
