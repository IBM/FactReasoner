# Unit tests for the trust scoring module.
# Run with: python tests/test_trust.py

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fact_reasoner.core.trust.url_trust import extract_all_candidates, CANDIDATE_FEATURES, UTD
from fact_reasoner.core.trust.dynaTD import DynaTD
from fact_reasoner.core.trust.bayesian_fusion import BayesianTrustFusion

N = len(CANDIDATE_FEATURES)


# -----------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------

class TestFeatureExtraction:

    def test_output_length_is_fixed(self):
        # Feature vector must always be the same length regardless of input
        urls = [
            "https://en.wikipedia.org/wiki/Paris",
            "http://192.168.0.1/login",
            "not_a_url",
            "",
            "x" * 500,
        ]
        for url in urls:
            assert len(extract_all_candidates(url)) == N, \
                f"Wrong length for: {url!r}"

    def test_all_values_are_finite_floats(self):
        # No NaN or Inf should ever appear — these would corrupt training
        import math
        for url in ["https://example.com", "", "http://1.2.3.4/%20"]:
            for val in extract_all_candidates(url):
                assert isinstance(val, float)
                assert math.isfinite(val), \
                    f"Non-finite value {val} for: {url!r}"

    def test_https_and_http_differ(self):
        # HTTPS and HTTP must produce different feature vectors
        f_https = extract_all_candidates("https://example.com/page")
        f_http  = extract_all_candidates("http://example.com/page")
        assert f_https != f_http

    def test_ip_and_domain_differ(self):
        # An IP-based URL and a named domain must produce different vectors
        f_ip     = extract_all_candidates("http://192.168.1.1/admin")
        f_domain = extract_all_candidates("http://example.com/admin")
        assert f_ip != f_domain

    def test_feature_values_are_non_negative(self):
        # All features are counts, flags, ratios or entropies — none can be negative
        for val in extract_all_candidates("https://en.wikipedia.org/wiki/Test"):
            assert val >= 0.0, f"Negative feature value: {val}"


# -----------------------------------------------------------------------
# UTD
# -----------------------------------------------------------------------

class TestUTD:

    def setup_method(self):
        # Use a nonexistent path so model is never loaded — tests the untrained state
        self.utd = UTD(model_path="/tmp/nonexistent_utd_model.pkl")

    def test_untrained_always_returns_half(self):
        # Before training, UTD must return 0.5 (maximum uncertainty) for any URL
        for url in ["https://nih.gov", "http://malicious.xyz", "", "junk"]:
            assert self.utd.score(url) == 0.5, \
                f"Expected 0.5 for untrained model on: {url!r}"

    def test_score_always_in_valid_range(self):
        # Score must stay within [0.05, 0.97] after training
        # Test the range contract even without a trained model
        score = self.utd.score("https://example.com")
        assert 0.0 <= score <= 1.0

    def test_explain_returns_required_keys(self):
        result = self.utd.explain("https://example.com")
        assert "url"   in result
        assert "score" in result
        assert "mode"  in result

    def test_feature_importance_empty_when_untrained(self):
        assert self.utd.feature_importance() == {}


# -----------------------------------------------------------------------
# DynaTD
# -----------------------------------------------------------------------

class TestDynaTD:

    def setup_method(self):
        self.d = DynaTD(state_path="/tmp/test_dynaTD.json")
        self.d.reset()

    def test_unseen_domain_returns_half(self):
        # Any domain with no history must return 0.5 (uninformative prior)
        assert self.d.get_reliability("never-seen-before.com") == 0.5

    def test_reliability_stays_in_valid_range(self):
        # Reliability is a Laplace-smoothed rate: must stay strictly within
        # the open interval (0, 1) no matter how many updates accumulate.
        self.d.initialize_domain("test.com", utd_score=0.5)
        for _ in range(200):
            self.d.update("test.com", 0.9, "entailment", 1.0)
        score = self.d.get_reliability("test.com")
        assert 0.0 < score < 1.0, f"Score {score} out of bounds"

    def test_reliable_source_scores_higher_than_unreliable(self):
        # After equal number of observations, a source whose claims
        # the Markov Network agrees with should score higher than one
        # whose claims it consistently disagrees with.
        self.d.initialize_domain("reliable.com",   utd_score=0.5)
        self.d.initialize_domain("unreliable.com", utd_score=0.5)

        for _ in range(30):
            # Reliable: context entails atom, posterior stays high — low error
            self.d.update("reliable.com",
                          atom_posterior=0.92,
                          nli_label="entailment",
                          nli_strength=0.95)
            # Unreliable: context entails atom, but posterior stays low — high error
            self.d.update("unreliable.com",
                          atom_posterior=0.05,
                          nli_label="entailment",
                          nli_strength=0.95)

        r_reliable   = self.d.get_reliability("reliable.com")
        r_unreliable = self.d.get_reliability("unreliable.com")

        assert r_reliable > r_unreliable, (
            f"Reliable source ({r_reliable:.3f}) should score higher "
            f"than unreliable ({r_unreliable:.3f}) after equal evidence"
        )

    def test_state_persists_across_instances(self):
        path = "/tmp/test_dynaTD_persist.json"
        d1 = DynaTD(state_path=path)
        d1.reset()
        d1.initialize_domain("persistent.com", utd_score=0.5)
        for _ in range(10):
            d1.update("persistent.com", 0.9, "entailment", 0.9)
        d1._save()   # ← explicitly save before reading back
        score_before = d1.get_reliability("persistent.com")

        d2 = DynaTD(state_path=path)
        score_after = d2.get_reliability("persistent.com")

        assert score_before == score_after, \
            f"State not preserved: {score_before:.3f} ≠ {score_after:.3f}"


# -----------------------------------------------------------------------
# BayesianTrustFusion
# -----------------------------------------------------------------------

class TestBayesianTrustFusion:

    def setup_method(self):
        self.fusion = BayesianTrustFusion(
            model_path="/tmp/nonexistent_utd_model.pkl",
            state_path="/tmp/test_fusion.json",
        )
        self.fusion.dynaTD.reset()

    def _ctx(self, url):
        class FakeContext:
            link = url
            def get_probability(self): return 0.5
        return FakeContext()

    def test_score_in_valid_range(self):
        for url in ["https://en.wikipedia.org", "http://1.2.3.4", "", "junk"]:
            score = self.fusion.score(self._ctx(url))
            assert 0.05 <= score <= 0.97, \
                f"Score {score} out of range for: {url!r}"

    def test_untrained_score_is_half(self):
        # With no trained model and no history, fused score must be 0.5
        score = self.fusion.score(self._ctx("https://example.com"))
        assert score == 0.5

    def test_new_domain_has_zero_dynaTD_influence(self):
        # For an unseen domain, beta=0 so DynaTD contributes nothing
        expl = self.fusion.score_with_explanation(self._ctx("https://brand-new-domain.org"))
        assert expl["beta"] == 0.0, \
            f"Expected beta=0 for unseen domain, got {expl['beta']}"

    def test_dynaTD_influence_grows_with_evidence(self):
        # After observing a domain multiple times, DynaTD should contribute more
        domain = "observed-domain.com"
        self.fusion.dynaTD.initialize_domain(domain, 0.5)
        for _ in range(20):
            self.fusion.dynaTD.update(domain, 0.85, "entailment", 0.9)
        expl = self.fusion.score_with_explanation(self._ctx(f"https://{domain}/page"))
        assert expl["beta"] > 0.0, \
            "DynaTD influence should be nonzero after observing domain"
        assert expl["dynaTD_claims"] == 20

    def test_explanation_contains_all_required_fields(self):
        expl = self.fusion.score_with_explanation(self._ctx("https://example.com"))
        for key in ["url", "domain", "utd_score", "dynaTD_score",
                    "beta", "fused_score", "dynaTD_claims"]:
            assert key in expl, f"Missing key: {key}"

    def test_beta_capped_at_max(self):
        # DynaTD influence should never exceed max_beta regardless of evidence volume
        domain = "high-volume-domain.com"
        self.fusion.dynaTD.initialize_domain(domain, 0.5)
        for _ in range(1000):
            self.fusion.dynaTD.update(domain, 0.9, "entailment", 0.9)
        expl = self.fusion.score_with_explanation(self._ctx(f"https://{domain}"))
        assert expl["beta"] <= self.fusion.max_beta, \
            f"Beta {expl['beta']} exceeded max {self.fusion.max_beta}"


# -----------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------

if __name__ == "__main__":
    print("Running tests...\n")

    for cls, label in [
        (TestFeatureExtraction,   "Feature extraction"),
        (TestUTD,                 "UTD"),
        (TestDynaTD,              "DynaTD"),
        (TestBayesianTrustFusion, "BayesianTrustFusion"),
    ]:
        obj = cls()
        methods = [m for m in dir(obj) if m.startswith("test_")]
        for method in methods:
            if hasattr(obj, "setup_method"):
                obj.setup_method()
            getattr(obj, method)()
        print(f"  ✓ {label} ({len(methods)} tests)")

    print("\nAll tests passed.")
