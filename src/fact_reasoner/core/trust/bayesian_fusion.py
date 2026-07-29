# bayesian_fusion.py
#
# Fuses UTD static URL safety score with DynaTD online reliability.
#
# For a new domain (no DynaTD history):
#   final = UTD_score
#
# For a known domain:
#   final = (1 - beta) * UTD_score + beta * DynaTD_score
#   beta  = min(num_claims / 2, max_beta)  # very fast ramp: full trust weight by ~2 atoms
#
# Beta grows as evidence accumulates, capped so UTD always contributes.
from urllib.parse import urlparse
from fact_reasoner.core.trust.url_trust import UTD, DEFAULT_MODEL_PATH
from fact_reasoner.core.trust.dynaTD import DynaTD, DEFAULT_STATE_PATH

# Must stay IDENTICAL to dynaTD.py's SOCIAL_PLATFORMS_KEY_BY_ACCOUNT --
# score() (here) and update_from_factreasoner_results() (dynaTD.py)
# need to compute the SAME key for the same URL, or trust history
# silently fragments across two different keys for the same real
# account.
SOCIAL_PLATFORMS_KEY_BY_ACCOUNT = {"twitter.com", "x.com", "facebook.com"}


class BayesianTrustFusion:
    """
    Plugs into ContextRetriever as trust_scorer.
    Sets context.probability before FactReasoner runs belief propagation,
    replacing the hardcoded PRIOR_PROB_CONTEXT = 0.9.
    """
    def __init__(
        self,
        model_path:        str   = DEFAULT_MODEL_PATH,
        state_path:        str   = DEFAULT_STATE_PATH,
        max_beta:          float = 0.7,
        selection_path:    str   = "/u/samit/data/selected_features.json",
    ):
        self.utd    = UTD(model_path=model_path,
                          selection_path=selection_path)
        self.dynaTD = DynaTD(state_path=state_path)
        self.max_beta = max_beta

    def score(self, context) -> float:
        """
        Fused trust score for a retrieved context.
        Returns float in [0.05, 0.97].
        """
        url    = getattr(context, "link", "") or ""
        domain = self._domain(url)
        utd_score = self.utd.score(url)
        if domain:
            self.dynaTD.initialize_domain(domain, utd_score)
        dyna_score = self.dynaTD.get_reliability(domain) if domain else 0.5
        num_claims = self.dynaTD.total_count.get(domain, 0.0)
        beta       = min(num_claims / 2.0, self.max_beta)
        fused      = (1.0 - beta) * utd_score + beta * dyna_score
        return float(max(0.05, min(0.97, fused)))

    def score_with_explanation(self, context) -> dict:
        """Score with full breakdown for logging and debugging."""
        url    = getattr(context, "link", "") or ""
        domain = self._domain(url)
        utd_result = self.utd.explain(url)
        utd_score  = utd_result["score"]
        if domain:
            self.dynaTD.initialize_domain(domain, utd_score)
        dyna_score = self.dynaTD.get_reliability(domain) if domain else 0.5
        num_claims = self.dynaTD.total_count.get(domain, 0.0)
        beta       = min(num_claims / 20.0, self.max_beta)
        fused      = float(max(0.05, min(0.97,
                            (1.0 - beta) * utd_score + beta * dyna_score)))
        return {
            "url":           url,
            "domain":        domain,
            "utd_score":     round(utd_score, 4),
            "utd_mode":      utd_result.get("mode", "unknown"),
            "dynaTD_score":  round(dyna_score, 4),
            "dynaTD_claims": int(num_claims),
            "beta":          round(beta, 4),
            "fused_score":   round(fused, 4),
        }

    def update_from_results(self, contexts, atom_marginals, nli_relations):
        """Update DynaTD after pipeline.score() returns."""
        self.dynaTD.update_from_factreasoner_results(
            contexts, atom_marginals, nli_relations
        )

    @staticmethod
    def _domain(url: str) -> str:
        if not url:
            return ""
        try:
            parsed = urlparse(url)
            netloc = parsed.netloc.lower().split(":")[0]
            netloc = netloc[4:] if netloc.startswith("www.") else netloc

            if netloc in SOCIAL_PLATFORMS_KEY_BY_ACCOUNT:
                path_parts = [p for p in parsed.path.split("/") if p]
                if path_parts:
                    return f"{netloc}/{path_parts[0].lower()}"

            return netloc
        except Exception:
            return ""
