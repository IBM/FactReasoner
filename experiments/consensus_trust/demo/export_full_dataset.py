"""Full CSV export: every atom, context, relation, and verdict across all 40
eval rows — for the demo UI. Deterministic replay from logged relations,
same method as replay_demo_row.py, generalized to the whole dataset.
"""
import json, glob, re, sys, os, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
import fact_reasoner.core.trust.credibility_scorer as _cs
import fact_reasoner.core.trust.credibility_scorer_v3 as _v3
_cs.score_url = _v3.score_url
from state_media_eval import make_pipeline
from fact_reasoner.core.base import Atom, Context, Relation
from fact_reasoner.core.trust.credibility_fusion import CredibilityTrustFusion

STATE  = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'trust_eval', 'dynaTD_state_40v2.json')
LOG    = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'trust_eval', 'final_40v2.txt')
RESULTS_GLOB = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'trust_eval', 'final_eval_40v2_results', 'results.json')
FP_JSONL = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'trust_eval', 'eval_dataset_fp.jsonl')
OUT = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'trust_eval', 'demo_exports', 'full')
os.makedirs(OUT, exist_ok=True)

d_results = json.load(open(RESULTS_GLOB))
log = open(LOG).read().splitlines()
start = next(i for i, l in enumerate(log) if 'WARMUP done' in l)
marks = [i for i, l in enumerate(log[start:], start) if re.match(r'\[ *\d+/', l)]
REL_RE = re.compile(r'\[NLI\] Found relation: \[(\S+) -> (\S+)\] : (\w+) : ([\d.]+)')

fp_rows_by_index = [json.loads(l) for l in open(FP_JSONL) if l.strip()]
sc = CredibilityTrustFusion(state_path=STATE)

def build(row, scorer):
    A = {a['id']: Atom(id=a['id'], text=a['text']) for a in row['atoms']}
    lut = {c['id']: c for c in row['contexts']}
    C = {}
    for a in row['atoms']:
        at = A[a['id']]; mine = []
        for cid in a['contexts'][:10]:
            if cid not in lut: continue
            c = lut[cid]
            x = Context(id=cid, atom=at, text=c['text'], title=c.get('title',''), link=c.get('link',''))
            x.set_probability(scorer.score(x) if scorer else 0.9)
            C[cid] = x; mine.append(x)
        at.add_contexts(mine)
    return A, C

rows_out, atoms_out, contexts_out, relations_out = [], [], [], []

for idx, r_result in enumerate(d_results['results']):
    input_text = r_result['input']
    row = fp_rows_by_index[idx]
    gt = row['ground_truth']

    seg_start = marks[idx-1] if idx > 0 else start
    seg = log[seg_start:marks[idx]] if idx < len(marks) else log[seg_start:]
    block_starts = [i for i, l in enumerate(seg) if 'Building atom-context relations' in l]
    if len(block_starts) != 4:
        print(f"[{idx+1}/40] WARNING: {len(block_starts)} blocks (expected 4) -- flagging, skipping relation parse")
        rows_out.append({'row_id': idx, 'input': input_text, 'ground_truth': gt,
                          'fp_trust_score': '', 'fp_trust_verdict': '', 'fp_vanilla_score': '', 'fp_vanilla_verdict': '',
                          'note': f'PARSE_ANOMALY_{len(block_starts)}_blocks'})
        continue
    block_starts.append(len(seg))
    def parse_block(b):
        return [REL_RE.search(l).groups() for l in seg[block_starts[b]:block_starts[b+1]] if 'Found relation' in l]
    fp_trust_rels, fp_vanilla_rels = parse_block(2), parse_block(3)

    def replay(rel_tuples, scorer, arm):
        A, C = build(row, scorer)
        rels = []
        for src, tgt, typ, p in rel_tuples:
            if src in C and tgt in A:
                rels.append(Relation(source=C[src], target=A[tgt], type=typ, probability=float(p), link='context_atom'))
        live = {rr.source.id for rr in rels}
        Cf = {k: v for k, v in C.items() if k in live}
        for a in A.values():
            a.contexts = {c.id: c for c in a.get_contexts().values() if c.id in Cf}
        _, marg = make_pipeline(A, Cf, rels, gt).score()
        post = {m['variable']: m['probabilities'][1] for m in marg if m['variable'] in A}
        prec = sum(1 for p in post.values() if p > 0.5) / max(len(post), 1)
        return A, Cf, rels, post, prec

    A_t, C_t, R_t, post_t, prec_t = replay(fp_trust_rels, sc, 'trust')
    A_v, C_v, R_v, post_v, prec_v = replay(fp_vanilla_rels, None, 'vanilla')
    verdict_t = 'S' if prec_t > 0.5 else 'NS'
    verdict_v = 'S' if prec_v > 0.5 else 'NS'

    rows_out.append({'row_id': idx, 'input': input_text, 'ground_truth': gt,
                      'fp_trust_score': round(prec_t,4), 'fp_trust_verdict': verdict_t,
                      'fp_vanilla_score': round(prec_v,4), 'fp_vanilla_verdict': verdict_v, 'note': ''})

    lut = {c['id']: c for c in row['contexts']}
    for a in row['atoms']:
        aid = a['id']
        atoms_out.append({'row_id': idx, 'atom_id': aid, 'atom_text': a['text'],
                           'P_true_trust': round(post_t.get(aid, -1), 4) if aid in post_t else '',
                           'P_true_vanilla': round(post_v.get(aid, -1), 4) if aid in post_v else '',
                           'verdict_trust': 'S' if post_t.get(aid, 0) > 0.5 else ('NS' if aid in post_t else ''),
                           'verdict_vanilla': 'S' if post_v.get(aid, 0) > 0.5 else ('NS' if aid in post_v else ''),
                           'gold_label': a.get('label', '')})
        for cid in a['contexts'][:10]:
            if cid not in lut: continue
            c = lut[cid]
            contexts_out.append({'row_id': idx, 'ctx_id': cid, 'atom_id': aid,
                                  'domain': sc._domain(c.get('link','')),
                                  'text_snippet': c['text'][:200], 'title': c.get('title',''), 'link': c.get('link','')})
    for tag, rl in (('trust', R_t), ('vanilla', R_v)):
        for rr in rl:
            relations_out.append({'row_id': idx, 'ctx_id': rr.source.id, 'atom_id': rr.target.id,
                                   'nli_type': rr.type, 'nli_strength': round(rr.probability, 4), 'arm': tag})
    print(f"[{idx+1}/40] gt={gt} trust={verdict_t}({prec_t:.3f}) vanilla={verdict_v}({prec_v:.3f})")

def write_csv(path, data):
    if not data: return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader(); w.writerows(data)
    print("wrote", path, f"({len(data)} rows)")

write_csv(f"{OUT}/rows.csv", rows_out)
write_csv(f"{OUT}/atoms.csv", atoms_out)
write_csv(f"{OUT}/contexts.csv", contexts_out)
write_csv(f"{OUT}/relations.csv", relations_out)
