"""Credibility prior v3: MBFC-derived domain lookup (idiap corpus, published).
Drop-in for v2: exposes score_url(url) -> float in [0.05, 0.97].
Unrated domains -> 0.5 neutral fallback (counted). Social account keys
(twitter.com/foo) fall back unless the platform itself is rated.
Source: idiap/Factual-Reporting-and-Political-Bias-Web-Interactions data/mbfc.csv
"""
import csv, os
from urllib.parse import urlparse

_CSV = '/u/samit/FactReasoner/data/priors/mbfc_idiap.csv'
_MAP = {  # factual_reporting label -> prior; extend if build reports unmapped labels
    'very high': 0.95, 'very-high': 0.95, 'very_high': 0.95,
    'high': 0.85,
    'mostly factual': 0.70, 'mostly-factual': 0.70, 'mostly_factual': 0.70, 'mostly': 0.70,
    'mixed': 0.50,
    'low': 0.30,
    'very low': 0.15, 'very-low': 0.15, 'very_low': 0.15,
}
_FALLBACK = 0.5
_table, _misses, _inst = None, {'n': 0}, {'n': 0}

def _norm(d):
    return d.lower().strip().replace('www.', '', 1).rstrip('.').split(':')[0]

def _load():
    global _table
    if _table is None:
        _table, unmapped = {}, {}
        with open(_CSV) as f:
            for row in csv.DictReader(f):
                lab = (row.get('factual_reporting') or '').strip().lower()
                if lab in _MAP:
                    _table[_norm(row['source'])] = _MAP[lab]
                elif lab:
                    unmapped[lab] = unmapped.get(lab, 0) + 1
        if unmapped:
            print(f"[cred_v3] UNMAPPED labels (extend _MAP!): {unmapped}")
        print(f"[cred_v3] loaded {_table and len(_table)} rated domains from {_CSV}")
    return _table

def score_url(url: str) -> float:
    t = _load()
    if not url:
        _misses['n'] += 1; return _FALLBACK
    d = _norm(urlparse(url if '://' in url else f'https://{url}').netloc or url.split('/')[0])
    if d in t: return t[d]
    parts = d.split('.')                       # subdomain -> parent walk
    for i in range(1, len(parts) - 1):
        if '.'.join(parts[i:]) in t: return t['.'.join(parts[i:])]
    if d.endswith(('.gov', '.edu', '.int', '.mil')) or d in ('un.org','who.int','europa.eu','worldbank.org','imf.org','oecd.org') \
            or d.endswith(('.un.org','.europa.eu','.nih.gov')):
        _inst['n'] += 1
        return 0.90                              # institutional tier: category rule, not outlet list
    _misses['n'] += 1
    return _FALLBACK

def miss_count(): return _misses['n']

def inst_count(): return _inst['n']
