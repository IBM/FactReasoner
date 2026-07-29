"""
CredibilityTrustFusion — drop-in replacement for BayesianTrustFusion
that uses the new credibility scorer (v2) instead of the malicious-URL UTD.
The original bayesian_fusion.py and utd_model_20feat.pkl are untouched.
"""
from fact_reasoner.core.trust.bayesian_fusion import BayesianTrustFusion, DynaTD
from fact_reasoner.core.trust.credibility_scorer import score_url
from urllib.parse import urlparse
import re

class CredibilityTrustFusion(BayesianTrustFusion):
    """
    Same as BayesianTrustFusion but uses new credibility scorer instead of UTD.
    DynaTD still learns from evidence as before.
    """
    def __init__(self, state_path=None, max_beta=0.7):
        # Don't call super().__init__() — skip UTD loading
        self.utd    = None  # not used
        self.dynaTD = DynaTD(state_path=state_path or '/u/samit/dynaTD_state_credibility.json')
        self.max_beta = max_beta

    def _domain(self, url):
        try:
            d = urlparse(url).netloc.lower()
            d = re.sub(r'^www\.', '', d)
            # Normalize x.com → twitter.com for consistency
            if d == 'x.com': d = 'twitter.com'
            return d
        except:
            return ''

    def score_with_explanation(self, context) -> dict:
        """Score with full breakdown for logging/debugging (credibility-prior variant)."""
        url    = getattr(context, "link", "") or ""
        domain = self._domain(url)
        cred_score = score_url(url)
        if domain:
            self.dynaTD.initialize_domain(domain, cred_score)
        dyna_score = self.dynaTD.get_reliability(domain) if domain else 0.5
        num_claims = self.dynaTD.total_count.get(domain, 0)
        beta       = min(num_claims / 20.0, self.max_beta)
        fused      = float(max(0.05, min(0.97,
                            (1.0 - beta) * cred_score + beta * dyna_score)))
        return {
            "url":           url,
            "domain":        domain,
            "utd_score":     round(cred_score, 4),
            "utd_mode":      "credibility_v3",
            "dynaTD_score":  round(dyna_score, 4),
            "dynaTD_claims": int(num_claims),
            "beta":          round(beta, 4),
            "fused_score":   round(fused, 4),
        }

    def score(self, context) -> float:
        url    = getattr(context, 'link', '') or ''
        domain = self._domain(url)
        
        # New credibility scorer instead of UTD
        cred_score = score_url(url) if url else 0.5
        
        if domain:
            self.dynaTD.initialize_domain(domain, cred_score)
        
        dyna_score = self.dynaTD.get_reliability(domain) if domain else 0.5
        num_claims = self.dynaTD.a.get(domain, 0.0)
        beta       = min(num_claims / 2.0, self.max_beta)
        fused      = (1.0 - beta) * cred_score + beta * dyna_score
        return float(max(0.05, min(0.97, fused)))

    def update_from_results(self, contexts, marginals, relations):
        """DynaTD learning — same as parent."""
        return super().update_from_results(contexts, marginals, relations)
