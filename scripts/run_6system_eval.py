"""
6-system evaluation using pre-built JSONL datasets.
Uses build_relations + make_pipeline directly (bypasses pipeline.build()).
"""
import asyncio, json, sys, re
sys.path.insert(0, '/u/samit/FactReasoner/scripts')
sys.path.insert(0, '/u/samit/FactReasoner/src')

from mellea_ibm.rits import RITSBackend, RITS
from mellea.backends import ModelOption
from state_media_eval import build_relations, make_pipeline
from granite_switch_vs_factreaser_demo import NLIFixed
from fact_reasoner.core.base import Atom, Context
from fact_reasoner.core.trust.credibility_fusion import CredibilityTrustFusion

CC_JSONL   = '/u/samit/eval_dataset_cc_27.jsonl'
FP_JSONL   = '/u/samit/eval_dataset_fp_27.jsonl'
STATE_CRED = '/u/samit/dynaTD_state_credibility_all.json'
OUT_JSON   = '/u/samit/full_eval_6system_results.json'
TRIAL12    = '/u/samit/overnight_credibility_results/BEST_so_far.txt'

backend = RITSBackend(RITS.LLAMA_3_3_70B_INSTRUCT,
                      model_options={ModelOption.MAX_NEW_TOKENS: 512})
nli = NLIFixed(backend)
trust_scorer = CredibilityTrustFusion(state_path=STATE_CRED)

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def build_atoms_contexts(data, trust_scorer=None):
    """Build Atom/Context objects from JSONL record."""
    atoms_dict = {}
    for a in data['atoms']:
        atom = Atom(id=a['id'], text=a['text'])
        atoms_dict[a['id']] = atom

    ctx_lookup = {c['id']: c for c in data['contexts']}
    MAX_CTX_PER_ATOM = 10
    contexts = {}
    for a in data['atoms']:
        atom = atoms_dict[a['id']]
        atom_ctxs = []
        for cid in a['contexts'][:MAX_CTX_PER_ATOM]:
            if cid not in ctx_lookup:
                continue
            c = ctx_lookup[cid]
            ctx = Context(id=cid, atom=atom,
                         text=c['text'], title=c.get('title',''),
                         link=c.get('link',''))
            p = trust_scorer.score(ctx) if trust_scorer else 0.9
            ctx.set_probability(p)
            contexts[cid] = ctx
            atom_ctxs.append(ctx)
        atom.add_contexts(atom_ctxs)

    return atoms_dict, contexts

async def run_system(data, trust_scorer=None, label=""):
    """Run FR pipeline on pre-built data record."""
    gt = data['ground_truth']
    atoms_dict, contexts = build_atoms_contexts(data, trust_scorer)

    if len(contexts) < 2:
        return None, 'too_few_contexts'

    # NLI
    relations = None
    for attempt in range(5):
        try:
            relations = build_relations(
                atoms=atoms_dict, contexts=contexts, nli_extractor=nli,
                rel_atom_context=True, rel_context_context=False,
                use_summarized_contexts=False,
            )
            break
        except Exception as e:
            print(f"  [{label}] NLI attempt {attempt+1} failed: {str(e)[:60]}", flush=True)
            await asyncio.sleep(5*(attempt+1))

    if not relations:
        return None, 'nli_failed'

    # Filter to connected contexts only
    connected = {r.source.id for r in relations}
    contexts  = {k: v for k, v in contexts.items() if k in connected}
    relations = [r for r in relations if r.source.id in contexts]

    if not contexts or not relations:
        return None, 'no_relations'

    # Update atom contexts
    for atom in atoms_dict.values():
        atom.contexts = {cid: ctx for cid, ctx in
                        {c.id: c for c in atom.get_contexts().values()}.items()
                        if cid in contexts}

    pipeline = make_pipeline(atoms_dict, contexts, relations, gt)
    _, marginals = pipeline.score()

    atom_probs = {m["variable"]: m["probabilities"][1]
                  for m in marginals if m["variable"] in atoms_dict}

    if not atom_probs:
        return None, 'no_marginals'

    n_supported = sum(1 for p in atom_probs.values() if p > 0.5)
    precision   = n_supported / len(atom_probs)
    verdict     = "S" if precision > 0.5 else "NS"
    correct     = verdict == gt

    if trust_scorer:
        trust_scorer.update_from_results(contexts, marginals, relations)

    return {
        "verdict": verdict, "precision": round(precision, 4),
        "n_atoms": len(atom_probs), "correct": correct,
        "n_relations": len(relations),
    }, 'ok'

def parse_guardian_from_trial12():
    with open(TRIAL12) as f:
        content = f.read()
    results = []
    blocks = re.split(r'[═]{40,}', content)
    for block in blocks:
        gt_m    = re.search(r'GT=(\w+)', block)
        guard_m = re.search(r'Granite Guardian:.*?→ (\w+)', block)
        acct_m  = re.search(r'\[\d+\] (.+?)\s+\w+\s+GT=', block)
        label_m = re.search(r'GT=\w+\s+\((\w+)', block)
        if gt_m and guard_m:
            results.append({
                "gt": gt_m.group(1),
                "guardian_verdict": guard_m.group(1),
                "correct": gt_m.group(1) == guard_m.group(1),
                "account": acct_m.group(1).strip() if acct_m else "?",
                "raw_label": label_m.group(1) if label_m else "?",
            })
    return results

async def main():
    cc_data = load_jsonl(CC_JSONL)
    fp_data = load_jsonl(FP_JSONL)
    fp_by_input = {r['input']: r for r in fp_data}

    guardian_results = parse_guardian_from_trial12()
    print(f"CC: {len(cc_data)} | FP: {len(fp_data)} | Guardian: {len(guardian_results)}", flush=True)

    # Individual counters
    ind = {s: {"c":0,"t":0} for s in
           ["cc_trust","cc_vanilla","fp_trust","fp_vanilla"]}
    # Joint counters
    joint = {s: {"c":0,"t":0} for s in
             ["cc_trust","cc_vanilla","fp_trust","fp_vanilla"]}

    # Load existing results to resume
    try:
        with open(OUT_JSON) as f:
            saved = json.load(f)
        results = saved.get('results', [])
        ind = saved.get('individual', ind)
        joint = saved.get('joint', joint)
        done_inputs = {r.get('input','')[:80] for r in results}
        print(f"Resuming from {len(results)} saved results", flush=True)
    except:
        results = []
        done_inputs = set()

    for i, cc in enumerate(cc_data):
        # Skip already processed
        row_key = cc['input'][:80]
        if row_key in done_inputs:
            print(f"[{i+1}] Skipping already processed: {cc['account']}", flush=True)
            continue

        gt = cc['ground_truth']
        print(f"\n{'='*60}", flush=True)
        print(f"[{i+1}/{len(cc_data)}] {cc['account'][:20]} | "
              f"GT={gt} ({cc['raw_label']})", flush=True)

        row = {"account": cc['account'], "gt": gt,
               "raw_label": cc['raw_label'],
               "input": cc['input'][:80],
               "cc_trust": None, "cc_vanilla": None,
               "fp_trust": None, "fp_vanilla": None}

        # 1. CC + Trust
        print("  CC+Trust...", flush=True)
        r, status = await run_system(cc, trust_scorer, "CC+Trust")
        row["cc_trust"] = r
        if r:
            ind["cc_trust"]["t"] += 1
            if r["correct"]: ind["cc_trust"]["c"] += 1
            flag = "✓" if r["correct"] else "✗"
            print(f"  CC+Trust:   prec={r['precision']:.2f} "
                  f"({r['n_relations']} rels) → {r['verdict']} {flag}", flush=True)
        else:
            print(f"  CC+Trust:   {status}", flush=True)

        # 2. CC + Vanilla
        print("  CC+Vanilla...", flush=True)
        r, status = await run_system(cc, None, "CC+Vanilla")
        row["cc_vanilla"] = r
        if r:
            ind["cc_vanilla"]["t"] += 1
            if r["correct"]: ind["cc_vanilla"]["c"] += 1
            flag = "✓" if r["correct"] else "✗"
            print(f"  CC+Vanilla: prec={r['precision']:.2f} "
                  f"({r['n_relations']} rels) → {r['verdict']} {flag}", flush=True)
        else:
            print(f"  CC+Vanilla: {status}", flush=True)

        # 3. FP + Trust
        fp = fp_by_input.get(cc['input'])
        if fp:
            print("  FP+Trust...", flush=True)
            r, status = await run_system(fp, trust_scorer, "FP+Trust")
            row["fp_trust"] = r
            if r:
                ind["fp_trust"]["t"] += 1
                if r["correct"]: ind["fp_trust"]["c"] += 1
                flag = "✓" if r["correct"] else "✗"
                print(f"  FP+Trust:   prec={r['precision']:.2f} "
                      f"({r['n_atoms']} atoms, {r['n_relations']} rels) "
                      f"→ {r['verdict']} {flag}", flush=True)
            else:
                print(f"  FP+Trust:   {status}", flush=True)

            # 4. FP + Vanilla
            print("  FP+Vanilla...", flush=True)
            r, status = await run_system(fp, None, "FP+Vanilla")
            row["fp_vanilla"] = r
            if r:
                ind["fp_vanilla"]["t"] += 1
                if r["correct"]: ind["fp_vanilla"]["c"] += 1
                flag = "✓" if r["correct"] else "✗"
                print(f"  FP+Vanilla: prec={r['precision']:.2f} "
                      f"({r['n_atoms']} atoms, {r['n_relations']} rels) "
                      f"→ {r['verdict']} {flag}", flush=True)
            else:
                print(f"  FP+Vanilla: {status}", flush=True)

        # Joint count — only rows where ALL 4 ran
        if all(row.get(s) for s in
               ["cc_trust","cc_vanilla","fp_trust","fp_vanilla"]):
            for s in ["cc_trust","cc_vanilla","fp_trust","fp_vanilla"]:
                joint[s]["t"] += 1
                if row[s]["correct"]: joint[s]["c"] += 1

        results.append(row)

        # Running summary
        print(f"\n  Individual accuracy so far:", flush=True)
        for s, d in ind.items():
            if d["t"] > 0:
                print(f"    {s:<14}: {d['c']:>3}/{d['t']:>3} = "
                      f"{d['c']/d['t']*100:.1f}%", flush=True)

        # Save after each row
        with open(OUT_JSON, 'w') as f:
            json.dump({"results": results, "individual": ind,
                       "joint": joint}, f, indent=2)

    # Guardian summary
    g_correct = sum(1 for r in guardian_results if r["correct"])
    g_total = len(guardian_results)

    print(f"\n{'='*60}")
    print("FINAL — INDIVIDUAL (each system's own evaluated atoms)")
    print(f"{'='*60}")
    for s, d in ind.items():
        if d["t"] > 0:
            print(f"  {s:<14}: {d['c']:>3}/{d['t']:>3} = {d['c']/d['t']*100:.1f}%")

    print(f"\n{'='*60}")
    print("FINAL — JOINT (rows where all 4 FR systems ran)")
    print(f"{'='*60}")
    for s, d in joint.items():
        if d["t"] > 0:
            print(f"  {s:<14}: {d['c']:>3}/{d['t']:>3} = {d['c']/d['t']*100:.1f}%")

    print(f"\n  Granite Guardian (trial 12): {g_correct}/{g_total} = "
          f"{g_correct/g_total*100:.1f}%")
    print(f"\nSaved → {OUT_JSON}")

asyncio.run(main())
