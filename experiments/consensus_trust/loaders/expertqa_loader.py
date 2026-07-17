"""ExpertQA -> FactReasoner rows. One row per (question, claim). Binary correct/incorrect.
System fixed to rr_gs_gpt4 by default (retrieve-read, gold-search, GPT-4) for consistency.
"""
import json, re
from urllib.parse import urlparse

CORRECT = {'definitely correct', 'probably correct'}
INCORRECT = {'definitely incorrect', 'likely incorrect'}
# 'unsure' / anything else -> dropped in binary mode

EVID_RE = re.compile(r'^\[(\d+)\]\s*(\S+)\n\n(.*)$', re.S)

def parse_evidence(ev):
    m = EVID_RE.match(ev.strip())
    if m:
        return {'url': m.group(2), 'text': m.group(3).strip()}
    return {'url': '', 'text': ev.strip()}

def load(path='data/trust_eval/FactReasoner/data/expertqa/r2_compiled_anon_fixed.jsonl',
         system='rr_gs_gpt4', binary=True):
    rows = []
    n_nosys = n_droplab = 0
    for i, ex in enumerate(map(json.loads, open(path))):
        ans = ex.get('answers', {}).get(system)
        if not ans:
            n_nosys += 1
            continue
        for j, cl in enumerate(ans.get('claims', [])):
            lab = (cl.get('correctness') or '').strip().lower()
            if binary:
                if lab in CORRECT:
                    gt = 'S'
                elif lab in INCORRECT:
                    gt = 'NS'
                else:
                    n_droplab += 1
                    continue
            else:
                gt = lab
            contexts = []
            for k, ev in enumerate(cl.get('evidence', [])):
                pe = parse_evidence(ev)
                if pe['text']:
                    contexts.append({'id': f'c{k}', 'text': pe['text'], 'link': pe['url']})
            if not contexts:
                continue
            rows.append({
                'id': f'{i}_{j}', 'claim': cl['claim_string'],
                'atoms': [{'id': 'a0', 'text': cl['claim_string']}],
                'contexts': contexts,
                'ground_truth': {'a0': gt},
                'raw_label': lab, 'system': system,
                'field': ex.get('metadata', {}).get('field'),
                'specific_field': ex.get('metadata', {}).get('specific_field'),
            })
    print(f"[expertqa loader] system={system} rows={len(rows)} "
          f"no_system={n_nosys} dropped_label(unsure/etc)={n_droplab}")
    return rows

if __name__ == '__main__':
    import collections
    rows = load()
    print(collections.Counter(r['ground_truth']['a0'] for r in rows))
    print(collections.Counter(r['field'] for r in rows).most_common(10))
