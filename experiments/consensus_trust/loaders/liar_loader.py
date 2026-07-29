"""LIAR2 (chengxuphd/liar2) -> FactReasoner row format.
Binary collapse declared in advance: {pants-fire, false, barely-true} -> NS,
{half-true, mostly-true, true} -> S.  'justification' is NEVER used (it is the
fact-checker's written verdict). Contexts start empty; augment_with_serper fills them.
Env: LIAR_N (default 400).
"""
import os, json
from datasets import load_dataset
NS = {0, 1, 2}   # pants-fire, false, barely-true
S  = {3, 4, 5}   # half-true, mostly-true, true
def load():
    n = int(os.environ.get('LIAR_N', '400'))
    d = load_dataset('chengxuphd/liar2', split='test')
    rows = []
    for r in d:
        lab = r.get('label')
        if lab is None: continue
        gt = 'NS' if lab in NS else 'S'
        stmt = (r.get('statement') or '').strip()
        if not stmt: continue
        rows.append({
            'id': 'liar_%s' % r.get('id'),
            'claim': stmt,
            'atoms': [{'id': 'a0', 'text': stmt, 'contexts': []}],
            'contexts': [],
            'ground_truth': {'a0': gt},
            'label4': str(lab),
            'claim_date': r.get('date') or None,
            'speaker': r.get('speaker') or '',
        })
    print("[liar loader] %d rows available, returning %d" % (len(rows), min(n, len(rows))), flush=True)
    return rows[:n]
