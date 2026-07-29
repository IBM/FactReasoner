"""AVeriTeC dev -> FactReasoner rows. Binary S/R first pass.
v2 fixes: (1) no-URL contexts kept for verdict but trust_eligible=False,
(2) broader archive unwrapping (web.archive.org any-timestamp; archive.ph etc = no inner URL),
(3) per-account trust keys for twitter/x/facebook/youtube/instagram/tiktok.
"""
import json, re
from urllib.parse import urlparse

FACTCHECK_DOMAINS = {
    'snopes.com','politifact.com','factcheck.org','checkyourfact.com','fullfact.org',
    'africacheck.org','leadstories.com','usatoday.com/story/news/factcheck',
    'apnews.com/hub/ap-fact-check','reuters.com/fact-check','boomlive.in','altnews.in',
    'factcrescendo.com','verafiles.org',
}
ARCHIVE_RE = re.compile(r'^https?://web\.archive\.org/web/[^/]+/(https?://.*)$', re.I)
OPAQUE_ARCHIVES = ('archive.ph','archive.today','archive.is','archive.li','archive.vn')
SOCIAL_PLATFORMS = ('twitter.com','x.com','facebook.com','youtube.com','instagram.com','tiktok.com')

def unwrap(url):
    if not url: return None
    m = ARCHIVE_RE.match(url)
    if m: return m.group(1)
    d = urlparse(url).netloc.replace('www.','')
    if any(d == a or d.endswith('.'+a) for a in OPAQUE_ARCHIVES): return None
    if d == 'web.archive.org': return None
    return url

ARCHIVE_HOSTS_FINAL = ('web.archive.org','archive.org','archive.ph','archive.today',
                       'archive.is','archive.li','archive.vn','web-archive-org.translate.goog',
                       'webcache.googleusercontent.com','cachedview.nl')

def domain_key(url):
    if not url: return None
    p = urlparse(url)
    d = p.netloc.lower().replace('www.','').rstrip('.').split(':')[0]
    if not d: return None
    if not p.scheme.startswith('http') or '.' not in d: return None   # malformed (web.archivehttps: etc)
    if any(d == a or d.endswith('.'+a) for a in ARCHIVE_HOSTS_FINAL):
        return None                                                   # final gate: never credit an archiver
    if d.endswith('archives.gov'): return d                           # US National Archives = real source
    for plat in SOCIAL_PLATFORMS:
        if d == plat or d.endswith('.'+plat):
            segs = [s for s in p.path.split('/') if s]
            skip = {'watch','status','statuses','p','reel','video','photo','posts',
                    'home','hashtag','share','shorts','channel','c','user'}
            for s in segs:
                if s.lower() not in skip:
                    return f"{plat}/{s.lower()}"
                if s.lower() in ('channel','c','user') and len(segs) > segs.index(s)+1:
                    return f"{plat}/{segs[segs.index(s)+1].lower()}"
            return plat
    return d

def is_factcheck(url):
    if not url: return False
    d = urlparse(url).netloc.replace('www.','')
    return any(d == f or d.endswith('.'+f) or f in url for f in FACTCHECK_DOMAINS)

def load(path='data/trust_eval/FactReasoner/data/averitec_dev.json',
         binary=True, drop_factcheck_evidence=True):
    rows = []; n_lab = n_fc = n_nourl = n_unans = 0
    for i, ex in enumerate(json.load(open(path))):
        lab = ex['label']
        if binary and lab not in ('Supported','Refuted'):
            n_lab += 1; continue
        contexts = []
        for q in ex.get('questions', []):
            for a in q.get('answers', []):
                if a.get('answer_type') == 'Unanswerable':
                    n_unans += 1; continue
                raw = a.get('source_url') or a.get('cached_source_url') or ''
                url = unwrap(raw)
                if drop_factcheck_evidence and is_factcheck(url):
                    n_fc += 1; continue
                if url is None: n_nourl += 1
                contexts.append({'text': f"{q['question']} {a['answer']}",
                                 'link': url or '',
                                 'trust_key': domain_key(url),
                                 'trust_eligible': url is not None,
                                 'medium': a.get('source_medium')})
        if not contexts: continue
        rows.append({'id': i, 'claim': ex['claim'],
                     'atoms': [{'id':'a0','text': ex['claim']}],
                     'contexts': contexts,
                     'ground_truth': {'a0': 'S' if lab=='Supported' else 'NS'},
                     'label4': lab, 'reannotated': ex.get('required_reannotation', False),
                     'claim_date': ex.get('claim_date'), 'location': ex.get('location_ISO_code'),
                     'speaker': ex.get('speaker'), 'reporting_source': ex.get('reporting_source')})
    print(f"[loader] rows={len(rows)} dropped_labels(N/C)={n_lab} dropped_factcheck={n_fc} "
          f"unanswerable_skipped={n_unans} no_url_contexts(kept,not trust-eligible)={n_nourl}")
    return rows

if __name__ == '__main__':
    import collections
    rows = load()
    print(collections.Counter(r['ground_truth']['a0'] for r in rows))
    keys = collections.Counter(c['trust_key'] for r in rows for c in r['contexts'] if c['trust_key'])
    print('unique trust keys:', len(keys))
    print('top 15:', keys.most_common(15))
    print('sample per-account keys:', [k for k in keys if '/' in k][:10])
    bad = [k for k in keys if any(k==a or k.endswith('.'+a) for a in ARCHIVE_HOSTS_FINAL)]
    print('archive leaks (must be []):', bad)
