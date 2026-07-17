"""
Drop-in replacement for UTD using the new credibility scorer v2.
Loads credibility_scorer_v2.pkl and scores URLs by domain credibility.
"""
import pickle
import numpy as np
from urllib.parse import urlparse

_MODEL = None
_MODEL_PATH = '/u/samit/credibility_scorer_v2.pkl'

def _load():
    global _MODEL
    if _MODEL is None:
        with open(_MODEL_PATH, 'rb') as f:
            _MODEL = pickle.load(f)
    return _MODEL

def extract_features(url):
    try:
        parsed = urlparse(url if url.startswith('http') else f'https://{url}')
        domain = parsed.netloc.lower().replace('www.', '')
        tld = domain.split('.')[-1] if '.' in domain else ''
        parts = domain.split('.')
        path = parsed.path.lower()
    except:
        domain, tld, parts, path = url, '', [url], ''
    return [
        float(tld=='gov'), float(tld=='edu'), float(tld=='org'),
        float(tld=='com'), float(tld=='net'), float(tld=='io'),
        float(tld in ('ru','cn','ir','kp','by')),
        float(tld in ('uk','ca','au','de','fr','jp')),
        float(tld=='co'), float(len(parts)), float(len(domain)),
        float(domain.count('-')), float(sum(c.isdigit() for c in domain)),
        float(len(path)), float(path.count('/')),
        float(any(kw in domain for kw in ('reuters','apnews','bbc','nytimes','guardian','bloomberg','npr','wsj','economist'))),
        float(any(kw in domain for kw in ('cdc','nih','fda','census','whitehouse','congress','justice','treasury'))),
        float(any(kw in domain for kw in ('nature','science','pubmed','lancet','nejm','bmj'))),
        float(any(kw in domain for kw in ('snopes','factcheck','politifact'))),
        float(any(kw in domain for kw in ('facebook','twitter','instagram','tiktok','reddit','pinterest','tumblr','snapchat'))),
        float(any(kw in domain for kw in ('infowars','breitbart','naturalnews','zerohedge','globalresearch','sputnik'))),
        float(any(kw in domain for kw in ('xinhua','chinadaily','globaltimes','cgtn'))),
        float('blog' in domain or 'wordpress' in domain or 'blogspot' in domain),
        float('medium.com' in domain or 'substack' in domain),
        float(any(kw in domain for kw in ('wikipedia','wiki'))),
        float('@' in url), float('//' in path),
        float(len(url)>100), float(len(url)>200),
        float(url.startswith('https')),
    ]

def score_url(url: str) -> float:
    """Score a URL for credibility. Returns float in [0.05, 0.97]."""
    m = _load()
    feats = np.array([extract_features(url)], dtype=np.float32)
    g = float(m['gbm'].predict(feats)[0])
    s = float(m['mlp'].predict(m['scaler'].transform(feats))[0])
    return float(np.clip((g + s) / 2, 0.05, 0.97))
